#!/usr/bin/env python3
"""
Proximal Policy Optimization training script for largemouth bass reinforcement learning.

This module implements PPO-based training for a simulated largemouth bass (Micropterus
salmoides) agent operating in a 3D aquatic environment. The training pipeline features:

- Curriculum learning with progressive difficulty stages and automatic promotion/demotion
  based on mastery criteria (food intake, collision rate, predator avoidance).
- Phased environment training decomposing skill acquisition into four courses: precise
  foraging, obstacle navigation, threat-aware feeding, and full mixed-environment
  convergence.
- Multi-environment parallel rollout collection via SubprocVecEnv for throughput
  scaling on multi-core/GPU hardware.
- Entropy coefficient annealing to balance exploration and exploitation over the
  training horizon.
- Comprehensive metric logging (CSV, matplotlib curves, TensorBoard) and periodic
  model checkpointing with best-model tracking by somatic mass gain.
- Optional RecurrentPPO (LSTM policy) support via sb3-contrib.

Typical usage::

    python train.py                          # Default 80M-step training
    python train.py --timesteps 50M          # Custom step budget
    python train.py --staged_env             # Enable phased environment curriculum
    python train.py --recurrent              # Use LSTM policy

Reference implementation accompanies the manuscript submitted to Nature Communications.
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime, timedelta
from collections import deque
from typing import Dict, Any, Optional, List, Callable, Tuple
from dataclasses import dataclass, field

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
import warnings
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy

warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
matplotlib.use('Agg')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 9

try:
    from tqdm import tqdm
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Hint: pip install rich tqdm for enhanced display output")

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed
try:
    from sb3_contrib import RecurrentPPO
    HAS_RECURRENT_PPO = True
except ImportError:
    RecurrentPPO = None
    HAS_RECURRENT_PPO = False


# ═══════════════════════════════════════════════════════════════════════════════
# Hardware Detection
# ═══════════════════════════════════════════════════════════════════════════════

def detect_hardware() -> Dict[str, Any]:
    """Detect available hardware and recommend training configuration.

    Returns:
        Dictionary containing CUDA availability, device name, memory capacity,
        CPU core count, and recommended parallelism/batch-size settings.
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_device': None,
        'cuda_memory_gb': 0,
        'cpu_cores': os.cpu_count() or 8,
        'recommended_n_envs': 8,
        'recommended_batch_size': 2048,
    }
    if info['cuda_available']:
        info['cuda_device'] = torch.cuda.get_device_name(0)
        info['cuda_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9
        if info['cuda_memory_gb'] >= 20:
            info['recommended_n_envs'] = 48
            info['recommended_batch_size'] = 8192
        elif info['cuda_memory_gb'] >= 10:
            info['recommended_n_envs'] = 32
            info['recommended_batch_size'] = 4096
        else:
            info['recommended_n_envs'] = 16
            info['recommended_batch_size'] = 2048
    info['recommended_n_envs'] = min(info['recommended_n_envs'], info['cpu_cores'] * 2)
    return info


HARDWARE = detect_hardware()

# ═══════════════════════════════════════════════════════════════════════════════
# Environment Import
# ═══════════════════════════════════════════════════════════════════════════════

BassEnvironment = None
CONFIG = None
ENV_VERSION = "unknown"

try:
    from environment import BassEnvironment
    from config import CONFIG, CURRICULUM_STAGES, get_curriculum_stage

    ENV_VERSION = "v6.2 (simplified)"
except ImportError:
    try:
        from environment_v5 import BassEnvironment
        from config import CONFIG, CURRICULUM_STAGES, get_curriculum_stage

        ENV_VERSION = "v5.2 (fallback)"
    except ImportError:
        from environment import BassEnvironment
        from config import CONFIG, CURRICULUM_STAGES, get_curriculum_stage

        ENV_VERSION = "standard"

print(f"\n{'=' * 70}")
print(f"Largemouth Bass RL Training System v6.2 - Simplified (PPO defaults)")
print(f"{'=' * 70}")
print(f"Environment version: {ENV_VERSION}")
print(f"CUDA: {HARDWARE['cuda_device'] if HARDWARE['cuda_available'] else 'Not available'}")
if HARDWARE['cuda_available']:
    print(f"VRAM: {HARDWARE['cuda_memory_gb']:.1f} GB")
print(f"CPU cores: {HARDWARE['cpu_cores']}")
print(f"Recommended config: n_envs={HARDWARE['recommended_n_envs']}, batch_size={HARDWARE['recommended_batch_size']}")
print(f"{'=' * 70}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Learning Rate Schedule
# ═══════════════════════════════════════════════════════════════════════════════

def linear_schedule(initial_value: float, final_value: float = 3e-5) -> Callable[[float], float]:
    """Create a linear learning rate schedule.

    The learning rate decays linearly from ``initial_value`` to ``final_value``
    over the course of training, parameterised by the remaining progress fraction
    provided by Stable-Baselines3.

    Args:
        initial_value: Learning rate at the start of training.
        final_value: Learning rate at the end of training. Defaults to 3e-5 to
            avoid numerical instability near zero.

    Returns:
        A callable mapping remaining progress (1.0 -> 0.0) to learning rate.
    """
    def func(progress_remaining: float) -> float:
        return final_value + (initial_value - final_value) * progress_remaining
    return func


# ═══════════════════════════════════════════════════════════════════════════════
# Entropy Coefficient Decay Callback
# ═══════════════════════════════════════════════════════════════════════════════

class EntropyDecayCallback(BaseCallback):
    """Callback for linearly annealing the entropy coefficient during training.

    Stable-Baselines3 does not natively support scheduled entropy coefficients,
    so this callback directly mutates ``model.ent_coef`` at each environment step
    according to a linear interpolation between initial and final values.

    Args:
        initial_ent: Entropy coefficient at the start of training.
        final_ent: Entropy coefficient at the end of training.
        total_timesteps: Total training budget used to compute progress.
        verbose: Verbosity level.
    """

    def __init__(self, initial_ent: float = 0.01, final_ent: float = 0.002,
                 total_timesteps: int = 50000000, verbose: int = 0):
        super().__init__(verbose)
        self.initial_ent = initial_ent
        self.final_ent = final_ent
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        # Compute current training progress
        progress = self.num_timesteps / self.total_timesteps

        # Linear interpolation
        new_ent_coef = self.initial_ent + (self.final_ent - self.initial_ent) * progress

        # Update the model entropy coefficient
        self.model.ent_coef = new_ent_coef

        return True


# ═══════════════════════════════════════════════════════════════════════════════
# Training Metrics Collector
# ═══════════════════════════════════════════════════════════════════════════════

class EnhancedTrainingMetrics:
    """Aggregates and persists training episode statistics.

    Maintains rolling buffers of per-episode metrics (reward, mass change, food
    intake, predation, locomotion, buoyancy, etc.) and periodically flushes
    summary statistics to CSV and in-memory time-series for plotting.

    Args:
        log_dir: Directory for CSV output and plot images.
    """

    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # Time-series data (bounded deques to cap memory usage)
        MAX_HISTORY_SIZE = 2000
        self.timesteps = deque(maxlen=MAX_HISTORY_SIZE)
        self.episode_rewards = deque(maxlen=MAX_HISTORY_SIZE)
        self.episode_lengths = deque(maxlen=MAX_HISTORY_SIZE)
        self.mass_change = deque(maxlen=MAX_HISTORY_SIZE)
        self.growth_events = deque(maxlen=MAX_HISTORY_SIZE)
        self.pellet_eaten = deque(maxlen=MAX_HISTORY_SIZE)
        self.floating_eaten = deque(maxlen=MAX_HISTORY_SIZE)
        self.sinking_eaten = deque(maxlen=MAX_HISTORY_SIZE)
        self.ambient_eaten = deque(maxlen=MAX_HISTORY_SIZE)
        self.surface_env_eaten = deque(maxlen=MAX_HISTORY_SIZE)
        self.benthic_eaten = deque(maxlen=MAX_HISTORY_SIZE)
        self.attached_eaten = deque(maxlen=MAX_HISTORY_SIZE)
        self.fish_eaten = deque(maxlen=MAX_HISTORY_SIZE)
        self.mass_from_pellet = deque(maxlen=MAX_HISTORY_SIZE)
        self.mass_from_fish = deque(maxlen=MAX_HISTORY_SIZE)
        self.total_food_intake = deque(maxlen=MAX_HISTORY_SIZE)
        self.predation_attempts = deque(maxlen=MAX_HISTORY_SIZE)
        self.predation_success_rate = deque(maxlen=MAX_HISTORY_SIZE)
        self.final_energy = deque(maxlen=MAX_HISTORY_SIZE)
        self.final_stomach = deque(maxlen=MAX_HISTORY_SIZE)
        self.energy_efficiency = deque(maxlen=MAX_HISTORY_SIZE)
        self.times_chased = deque(maxlen=MAX_HISTORY_SIZE)
        self.escape_count = deque(maxlen=MAX_HISTORY_SIZE)
        self.escape_success_rate = deque(maxlen=MAX_HISTORY_SIZE)
        self.damage_taken = deque(maxlen=MAX_HISTORY_SIZE)
        self.distance_traveled = deque(maxlen=MAX_HISTORY_SIZE)
        self.collision_count = deque(maxlen=MAX_HISTORY_SIZE)
        self.surface_entries = deque(maxlen=MAX_HISTORY_SIZE)
        self.avg_speed = deque(maxlen=MAX_HISTORY_SIZE)
        self.rest_ratio = deque(maxlen=MAX_HISTORY_SIZE)
        self.state_switches = deque(maxlen=MAX_HISTORY_SIZE)
        self.rest_during_danger = deque(maxlen=MAX_HISTORY_SIZE)
        self.buoyancy_adjustments = deque(maxlen=MAX_HISTORY_SIZE)
        self.buoyancy_energy = deque(maxlen=MAX_HISTORY_SIZE)
        self.avg_relative_density = deque(maxlen=MAX_HISTORY_SIZE)
        self.neutral_buoyancy_ratio = deque(maxlen=MAX_HISTORY_SIZE)
        self.learning_rates = deque(maxlen=MAX_HISTORY_SIZE)
        self.entropies = deque(maxlen=MAX_HISTORY_SIZE)
        self.entropy_losses = deque(maxlen=MAX_HISTORY_SIZE)
        self.value_losses = deque(maxlen=MAX_HISTORY_SIZE)
        self.policy_losses = deque(maxlen=MAX_HISTORY_SIZE)
        self.approx_kl = deque(maxlen=MAX_HISTORY_SIZE)
        self.clip_fractions = deque(maxlen=MAX_HISTORY_SIZE)
        self.explained_variances = deque(maxlen=MAX_HISTORY_SIZE)
        self.curriculum_stages = deque(maxlen=MAX_HISTORY_SIZE)

        # Sliding-window buffers for recent-episode averaging
        self.buffer_size = 100
        self._init_buffers()

        # Death cause statistics
        self.death_reasons = {'energy_depleted': 0, 'air_exposure': 0, 'survived': 0, 'unknown': 0}

        # CSV output
        self.csv_path = os.path.join(log_dir, "training_metrics.csv")
        self._write_csv_header()

        self.start_time = time.time()
        self.total_episodes = 0

    def _init_buffers(self) -> None:
        """Initialize fixed-length sliding-window buffers for all tracked metrics."""
        buffer_names = [
            'reward', 'length', 'mass_change', 'growth',
            'pellet', 'floating', 'sinking', 'ambient', 'surface_env', 'benthic', 'attached', 'fish',
            'mass_from_pellet', 'mass_from_fish',
            'pred_attempts', 'pred_success',
            'energy', 'stomach',
            'chased', 'escape', 'damage',
            'distance', 'collision', 'surface',
            'rest_ratio', 'state_switches', 'rest_danger',
            'buoyancy_adj', 'buoyancy_energy', 'density', 'neutral_steps'
        ]
        for name in buffer_names:
            setattr(self, f'{name}_buffer', deque(maxlen=self.buffer_size))

    def _write_csv_header(self) -> None:
        """Write the CSV column header row."""
        headers = [
            "timestep", "episodes", "reward_mean", "reward_std",
            "length_mean", "mass_change_mean", "growth_events",
            "pellet_eaten", "fish_eaten", "rest_ratio",
            "entropy", "value_loss", "explained_var", "curriculum_stage"
        ]
        with open(self.csv_path, 'w') as f:
            f.write(",".join(headers) + "\n")

    def add_episode(self, info: Dict[str, Any]) -> None:
        """Record metrics from a completed episode.

        Args:
            info: Episode info dictionary returned by the environment at
                termination, containing reward, length, feeding, locomotion,
                and survival statistics.
        """
        self.total_episodes += 1

        if 'episode' in info:
            self.reward_buffer.append(info['episode']['r'])
            self.length_buffer.append(info['episode']['l'])

        self.mass_change_buffer.append(info.get('mass_change', 0))
        self.growth_buffer.append(info.get('growth_event_count', 0))
        self.pellet_buffer.append(info.get('total_food_eaten', 0))
        self.floating_buffer.append(info.get('floating_eaten', 0))
        self.sinking_buffer.append(info.get('sinking_eaten', 0))
        self.ambient_buffer.append(info.get('ambient_eaten', 0))
        self.surface_env_buffer.append(info.get('surface_env_eaten', 0))
        self.benthic_buffer.append(info.get('benthic_eaten', 0))
        self.attached_buffer.append(info.get('attached_eaten', 0))
        self.fish_buffer.append(info.get('total_fish_eaten', 0))
        self.mass_from_pellet_buffer.append(info.get('mass_from_pellets', 0))
        self.mass_from_fish_buffer.append(info.get('mass_from_fish', 0))
        self.pred_attempts_buffer.append(info.get('predation_attempts', info.get('total_fish_eaten', 0)))
        self.pred_success_buffer.append(info.get('total_fish_eaten', 0))
        self.energy_buffer.append(info.get('energy', 0))
        self.stomach_buffer.append(info.get('stomach_fullness', 0))
        self.chased_buffer.append(info.get('times_chased', 0))
        self.escape_buffer.append(info.get('escape_count', 0))
        self.damage_buffer.append(info.get('damage_taken', 0))
        self.distance_buffer.append(info.get('distance_traveled', 0))
        self.collision_buffer.append(info.get('collision_count', 0))
        self.surface_buffer.append(info.get('surface_entries', 0))
        self.rest_ratio_buffer.append(info.get('rest_ratio', 0))
        self.state_switches_buffer.append(info.get('state_switches', 0))
        self.rest_danger_buffer.append(info.get('rest_during_danger', 0))
        self.buoyancy_adj_buffer.append(info.get('buoyancy_adjustments', 0))
        self.buoyancy_energy_buffer.append(info.get('buoyancy_energy_total', 0))
        self.density_buffer.append(info.get('avg_relative_density', 1.0))
        self.neutral_steps_buffer.append(info.get('neutral_buoyancy_steps', 0))

        death_reason = info.get('death_reason', None)
        if death_reason:
            self.death_reasons[death_reason] = self.death_reasons.get(death_reason, 0) + 1
        else:
            self.death_reasons['survived'] += 1

    def log(self, timestep: int, model: Any, curriculum_stage: int = 0) -> None:
        """Flush buffered statistics to time-series storage and CSV.

        Args:
            timestep: Current global environment step count.
            model: The PPO model instance (used to extract optimizer stats).
            curriculum_stage: Index of the active curriculum stage.
        """
        if len(self.reward_buffer) == 0:
            return

        stats = self._compute_stats()
        ppo_stats = self._get_ppo_stats(model, timestep)

        self.timesteps.append(timestep)
        self.episode_rewards.append(stats['reward_mean'])
        self.episode_lengths.append(stats['length_mean'])
        self.mass_change.append(stats['mass_change_mean'])
        self.growth_events.append(stats['growth_mean'])
        self.pellet_eaten.append(stats['pellet_mean'])
        self.floating_eaten.append(stats['floating_mean'])
        self.sinking_eaten.append(stats['sinking_mean'])
        self.fish_eaten.append(stats['fish_mean'])
        self.mass_from_pellet.append(stats['mass_from_pellet_mean'])
        self.mass_from_fish.append(stats['mass_from_fish_mean'])
        self.total_food_intake.append(stats['total_intake'])
        self.predation_attempts.append(stats['pred_attempts_mean'])
        self.predation_success_rate.append(stats['pred_success_rate'])
        self.final_energy.append(stats['energy_mean'])
        self.final_stomach.append(stats['stomach_mean'])
        self.energy_efficiency.append(stats['feeding_efficiency'])
        self.times_chased.append(stats['chased_mean'])
        self.escape_count.append(stats['escape_mean'])
        self.escape_success_rate.append(stats['escape_rate'])
        self.damage_taken.append(stats['damage_mean'])
        self.distance_traveled.append(stats['distance_mean'])
        self.collision_count.append(stats['collision_mean'])
        self.surface_entries.append(stats['surface_mean'])
        self.avg_speed.append(stats['avg_speed'])
        self.rest_ratio.append(stats['rest_ratio_mean'])
        self.state_switches.append(stats['state_switches_mean'])
        self.rest_during_danger.append(stats['rest_danger_mean'])
        self.buoyancy_adjustments.append(stats['buoyancy_adj_mean'])
        self.buoyancy_energy.append(stats['buoyancy_energy_mean'])
        self.avg_relative_density.append(stats['density_mean'])
        self.neutral_buoyancy_ratio.append(stats['neutral_ratio'])
        self.learning_rates.append(ppo_stats['lr'])
        self.entropies.append(ppo_stats['entropy'])
        self.entropy_losses.append(ppo_stats['entropy_loss'])
        self.value_losses.append(ppo_stats['value_loss'])
        self.policy_losses.append(ppo_stats['policy_loss'])
        self.approx_kl.append(ppo_stats['approx_kl'])
        self.clip_fractions.append(ppo_stats['clip_fraction'])
        self.explained_variances.append(ppo_stats['explained_var'])
        self.curriculum_stages.append(curriculum_stage)

        # Append to CSV
        row = [timestep, self.total_episodes, f"{stats['reward_mean']:.2f}", f"{stats['reward_std']:.2f}",
               f"{stats['length_mean']:.0f}", f"{stats['mass_change_mean']:.4f}", f"{stats['growth_mean']:.2f}",
               f"{stats['pellet_mean']:.2f}", f"{stats['fish_mean']:.3f}", f"{stats['rest_ratio_mean']:.4f}",
               f"{ppo_stats['entropy']:.4f}", f"{ppo_stats['value_loss']:.4f}",
               f"{ppo_stats['explained_var']:.4f}", curriculum_stage]
        with open(self.csv_path, 'a') as f:
            f.write(",".join(map(str, row)) + "\n")

    def _compute_stats(self) -> Dict[str, float]:
        """Compute summary statistics from the sliding-window buffers.

        Returns:
            Dictionary of aggregated metric means, ratios, and derived quantities.
        """
        def safe_mean(buf): return np.mean(buf) if len(buf) > 0 else 0

        def safe_std(buf): return np.std(buf) if len(buf) > 0 else 0

        stats = {
            'reward_mean': safe_mean(self.reward_buffer),
            'reward_std': safe_std(self.reward_buffer),
            'length_mean': safe_mean(self.length_buffer),
            'length_std': safe_std(self.length_buffer),
            'mass_change_mean': safe_mean(self.mass_change_buffer),
            'growth_mean': safe_mean(self.growth_buffer),
            'pellet_mean': safe_mean(self.pellet_buffer),
            'floating_mean': safe_mean(self.floating_buffer),
            'sinking_mean': safe_mean(self.sinking_buffer),
            'ambient_mean': safe_mean(self.ambient_buffer),
            'surface_env_mean': safe_mean(self.surface_env_buffer),
            'benthic_mean': safe_mean(self.benthic_buffer),
            'attached_mean': safe_mean(self.attached_buffer),
            'fish_mean': safe_mean(self.fish_buffer),
            'mass_from_pellet_mean': safe_mean(self.mass_from_pellet_buffer),
            'mass_from_fish_mean': safe_mean(self.mass_from_fish_buffer),
            'pred_attempts_mean': safe_mean(self.pred_attempts_buffer),
            'energy_mean': safe_mean(self.energy_buffer),
            'stomach_mean': safe_mean(self.stomach_buffer),
            'chased_mean': safe_mean(self.chased_buffer),
            'escape_mean': safe_mean(self.escape_buffer),
            'damage_mean': safe_mean(self.damage_buffer),
            'distance_mean': safe_mean(self.distance_buffer),
            'collision_mean': safe_mean(self.collision_buffer),
            'surface_mean': safe_mean(self.surface_buffer),
            'rest_ratio_mean': safe_mean(self.rest_ratio_buffer),
            'state_switches_mean': safe_mean(self.state_switches_buffer),
            'rest_danger_mean': safe_mean(self.rest_danger_buffer),
            'buoyancy_adj_mean': safe_mean(self.buoyancy_adj_buffer),
            'buoyancy_energy_mean': safe_mean(self.buoyancy_energy_buffer),
            'density_mean': safe_mean(self.density_buffer),
        }

        total_intake = stats['mass_from_pellet_mean'] + stats['mass_from_fish_mean']
        stats['total_intake'] = total_intake
        stats['feeding_efficiency'] = stats['mass_change_mean'] / total_intake if total_intake > 0 and stats[
            'mass_change_mean'] > 0 else 0
        stats['pred_success_rate'] = stats['fish_mean'] / stats['pred_attempts_mean'] if stats[
                                                                                             'pred_attempts_mean'] > 0 else 0
        stats['escape_rate'] = stats['escape_mean'] / stats['chased_mean'] if stats['chased_mean'] > 0 else 1.0
        stats['avg_speed'] = stats['distance_mean'] / stats['length_mean'] if stats['length_mean'] > 0 else 0
        stats['neutral_ratio'] = safe_mean(self.neutral_steps_buffer) / stats['length_mean'] if stats[
                                                                                                    'length_mean'] > 0 else 0

        return stats

    def _get_ppo_stats(self, model: Any, timestep: int) -> Dict[str, float]:
        """Extract PPO optimizer diagnostics from the model logger.

        Args:
            model: The PPO model instance.
            timestep: Current global step (used to compute LR schedule progress).

        Returns:
            Dictionary of PPO training statistics (learning rate, entropy,
            value loss, policy loss, KL divergence, clip fraction, explained
            variance).
        """
        ppo_stats = {'lr': 0.0, 'entropy': 0.0, 'entropy_loss': 0.0, 'value_loss': 0.0,
                     'policy_loss': 0.0, 'approx_kl': 0.0, 'clip_fraction': 0.0, 'explained_var': 0.0}

        lr = model.learning_rate
        if callable(lr):
            progress = 1.0 - (timestep / model._total_timesteps) if hasattr(model, '_total_timesteps') else 1.0
            lr = lr(progress)
        ppo_stats['lr'] = float(lr) if lr else 0.0

        if hasattr(model, 'logger') and model.logger:
            log_data = model.logger.name_to_value
            ppo_stats['entropy'] = abs(log_data.get('train/entropy_loss', 0))
            ppo_stats['entropy_loss'] = log_data.get('train/entropy_loss', 0)
            ppo_stats['value_loss'] = log_data.get('train/value_loss', 0)
            ppo_stats['policy_loss'] = log_data.get('train/policy_gradient_loss', 0)
            ppo_stats['approx_kl'] = log_data.get('train/approx_kl', 0)
            ppo_stats['clip_fraction'] = log_data.get('train/clip_fraction', 0)
            ppo_stats['explained_var'] = log_data.get('train/explained_variance', 0)

        return ppo_stats

    def get_current_stats(self) -> Dict[str, Any]:
        """Return current sliding-window summary statistics.

        Returns:
            Dictionary of aggregated metrics, or empty dict if no episodes
            have been recorded yet.
        """
        if len(self.reward_buffer) == 0:
            return {}
        return self._compute_stats()

    def get_death_stats(self) -> Dict[str, int]:
        """Return cumulative death-cause counts.

        Returns:
            Dictionary mapping death reason strings to occurrence counts.
        """
        return self.death_reasons.copy()


# ═══════════════════════════════════════════════════════════════════════════════
# Training Callback
# ═══════════════════════════════════════════════════════════════════════════════

class TrainingCallback(BaseCallback):
    """Primary training callback handling logging, plotting, and checkpointing.

    Integrates with ``EnhancedTrainingMetrics`` to periodically print rich
    console tables, generate matplotlib training curves, and save model
    checkpoints (including best-model tracking by mass gain).

    Args:
        total_timesteps: Total training budget in environment steps.
        metrics: Metrics collector instance.
        log_freq: Steps between console log updates.
        plot_freq: Steps between plot generation.
        save_freq: Steps between model checkpoint saves.
        model_dir: Directory for saved model files.
        verbose: Verbosity level.
    """

    def __init__(self, total_timesteps: int, metrics: EnhancedTrainingMetrics,
                 log_freq: int = 10000, plot_freq: int = 50000,
                 save_freq: int = 200000, model_dir: str = "./models", verbose: int = 1):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.metrics = metrics
        self.log_freq = log_freq
        self.plot_freq = plot_freq
        self.save_freq = save_freq
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

        self.start_time = time.time()
        self.curriculum_stage = 0
        self.best_mass_change = -float('inf')
        self.best_reward = -float('inf')

        if RICH_AVAILABLE:
            self.console = Console()
            self.progress = Progress(
                SpinnerColumn(), TextColumn("[bold blue]{task.description}"),
                BarColumn(bar_width=40), TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
                TextColumn("•"), TextColumn("[green]{task.fields[steps]}"),
                TextColumn("•"), TimeElapsedColumn(), TextColumn("•"), TimeRemainingColumn(),
                console=self.console,
            )
            self.task_id = None

    def _on_training_start(self) -> None:
        self._print_header()
        if RICH_AVAILABLE:
            self.progress.start()
            self.task_id = self.progress.add_task("Training", total=self.total_timesteps,
                                                  steps=f"0/{self._format_number(self.total_timesteps)}")

    def _on_step(self) -> bool:
        # Periodic memory monitoring
        if self.num_timesteps % 500000 == 0:
            try:
                import psutil
                import gc
                process = psutil.Process()
                mem_gb = process.memory_info().rss / 1e9
                gc.collect()  # Force garbage collection
                mem_after_gc = process.memory_info().rss / 1e9
                print(f"\n[Memory] {mem_gb:.2f} GB -> post-GC: {mem_after_gc:.2f} GB")
            except:
                pass

        # Collect episode-end metrics from all parallel environments
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                self.metrics.add_episode(info)

        if RICH_AVAILABLE and self.task_id is not None:
            self.progress.update(self.task_id, completed=self.num_timesteps,
                                 steps=f"{self._format_number(self.num_timesteps)}/{self._format_number(self.total_timesteps)}")

        if self.num_timesteps % self.log_freq == 0:
            self.metrics.log(self.num_timesteps, self.model, self.curriculum_stage)
            self._print_metrics()

        if self.num_timesteps % self.plot_freq == 0:
            self._generate_plots()

        if self.num_timesteps % self.save_freq == 0:
            self._save_checkpoint()

        return True

    def _on_training_end(self) -> None:
        if RICH_AVAILABLE:
            self.progress.stop()
        self._generate_plots()
        self._print_summary()

    def _print_header(self) -> None:
        """Print training session header banner."""
        print(f"""
╔═══════════════════════════════════════════════════════════════════════════════════╗
║            Largemouth Bass RL Training - v6.2 Simplified                          ║
╠═══════════════════════════════════════════════════════════════════════════════════╣
║  Environment: {ENV_VERSION:<50}            ║
║  Device: {HARDWARE['cuda_device'] if HARDWARE['cuda_available'] else 'CPU':<55}      ║
║  Total Steps: {self._format_number(self.total_timesteps):>12}                                                       ║
║  Config: PPO defaults (lr=3e-4 linear decay, ent_coef=0.01)                       ║
╚═══════════════════════════════════════════════════════════════════════════════════╝
""")

    def _print_metrics(self) -> None:
        """Print current training metrics to console (rich table or plain text)."""
        stats = self.metrics.get_current_stats()
        if not stats:
            return

        elapsed = time.time() - self.start_time
        fps = self.num_timesteps / elapsed if elapsed > 0 else 0

        mass_change = stats['mass_change_mean']
        reward = stats['reward_mean']
        if mass_change > self.best_mass_change:
            self.best_mass_change = mass_change
        if reward > self.best_reward:
            self.best_reward = reward

        ppo_stats = {}
        if len(self.metrics.entropies) > 0:
            ppo_stats = {
                'entropy': self.metrics.entropies[-1],
                'value_loss': self.metrics.value_losses[-1] if self.metrics.value_losses else 0,
                'explained_var': self.metrics.explained_variances[-1] if self.metrics.explained_variances else 0,
                'lr': self.metrics.learning_rates[-1] if self.metrics.learning_rates else 0,
            }

        if RICH_AVAILABLE:
            table1 = Table(title=f"[bold cyan]Step {self._format_number(self.num_timesteps)} - Core Metrics", box=None)
            table1.add_column("Metric", style="cyan", width=18)
            table1.add_column("Value", style="green", width=15)
            table1.add_column("Metric", style="cyan", width=18)
            table1.add_column("Value", style="green", width=15)
            table1.add_row("Episode Reward:", f"{reward:.2f}±{stats['reward_std']:.1f}",
                           "Survival Steps:", f"{stats['length_mean']:.0f}±{stats['length_std']:.0f}")
            table1.add_row("Mass Change:", f"{mass_change:+.3f}g", "Growth Events:", f"{stats['growth_mean']:.2f}")
            table1.add_row("FPS:", f"{fps:.0f}", "Curriculum:", f"Stage {self.curriculum_stage}")

            table2 = Table(title="[bold yellow]Feeding Details", box=None)
            table2.add_column("Metric", style="yellow", width=18)
            table2.add_column("Value", style="white", width=12)
            table2.add_column("Metric", style="yellow", width=18)
            table2.add_column("Value", style="white", width=12)
            env_total = (stats['ambient_mean'] + stats['surface_env_mean'] +
                         stats['benthic_mean'] + stats['attached_mean'])
            table2.add_row("Env Food Eaten:", f"{env_total:.2f}", "Fish Caught:", f"{stats['fish_mean']:.3f}")
            table2.add_row("  └ Ambient:", f"{stats['ambient_mean']:.2f}", "Predation Success:",
                           f"{stats['pred_success_rate'] * 100:.1f}%")
            table2.add_row("  └ Surface:", f"{stats['surface_env_mean']:.2f}", "Total Intake:",
                           f"{stats['total_intake']:.3f}g")
            table2.add_row("  └ Benthic:", f"{stats['benthic_mean']:.2f}", "Mass from Fish:",
                           f"{stats['mass_from_fish_mean']:.3f}g")
            table2.add_row("  └ Attached:", f"{stats['attached_mean']:.2f}", "Mass from Food:",
                           f"{stats['mass_from_pellet_mean']:.3f}g")

            table3 = Table(title="[bold red]Behavior & Threats", box=None)
            table3.add_column("Metric", style="red", width=18)
            table3.add_column("Value", style="white", width=12)
            table3.add_column("Metric", style="red", width=18)
            table3.add_column("Value", style="white", width=12)
            table3.add_row("Times Chased:", f"{stats['chased_mean']:.1f}", "Escape Count:",
                           f"{stats['escape_mean']:.1f}")
            table3.add_row("Escape Rate:", f"{stats['escape_rate'] * 100:.1f}%", "Damage Taken:",
                           f"{stats['damage_mean']:.2f}")
            table3.add_row("Rest Ratio:", f"{stats['rest_ratio_mean'] * 100:.1f}%", "State Switches:",
                           f"{stats['state_switches_mean']:.1f}")
            table3.add_row("Rest in Danger:", f"{stats['rest_danger_mean']:.1f}", "Collisions:",
                           f"{stats['collision_mean']:.1f}")

            table4 = Table(title="[bold magenta]PPO Training Metrics", box=None)
            table4.add_column("Metric", style="magenta", width=18)
            table4.add_column("Value", style="white", width=12)
            table4.add_column("Metric", style="magenta", width=18)
            table4.add_column("Value", style="white", width=12)
            if ppo_stats:
                table4.add_row("Learning Rate:", f"{ppo_stats.get('lr', 0):.2e}", "Entropy:",
                               f"{ppo_stats.get('entropy', 0):.4f}")
                table4.add_row("Value Loss:", f"{ppo_stats.get('value_loss', 0):.4f}", "Explained Var:",
                               f"{ppo_stats.get('explained_var', 0):.3f}")
                table4.add_row("Episodes:", f"{self.metrics.total_episodes}", "Ent Coef:", f"{self.model.ent_coef:.6f}")

            self.console.print(table1)
            self.console.print(table2)
            self.console.print(table3)
            self.console.print(table4)
            self.console.print("─" * 85)
        else:
            print(f"[{self._format_number(self.num_timesteps)}] R={reward:.1f} Mass={mass_change:+.3f}g "
                  f"Pellet={stats['pellet_mean']:.1f} Fish={stats['fish_mean']:.2f} "
                  f"Entropy={ppo_stats.get('entropy', 0):.3f} FPS={fps:.0f}")

    def _generate_plots(self) -> None:
        """Generate and save matplotlib training curve plots."""
        if len(self.metrics.timesteps) < 2:
            return

        fig, axes = plt.subplots(4, 3, figsize=(16, 18))
        fig.suptitle(f'Training Progress v6.2 Simple - {self._format_number(self.num_timesteps)} steps', fontsize=14,
                     fontweight='bold')

        # Downsampling utility for large time-series
        def downsample(data, max_points=500):
            data = list(data)
            if len(data) <= max_points:
                return np.array(data)
            indices = np.linspace(0, len(data) - 1, max_points, dtype=int)
            return np.array([data[i] for i in indices])

        timesteps = downsample(self.metrics.timesteps) / 1e6

        plot_data = [
            (axes[0, 0], self.metrics.episode_rewards, 'Episode Reward', 'b'),
            (axes[0, 1], self.metrics.mass_change, 'Body Mass Change (CORE)', 'brown'),
            (axes[0, 2], self.metrics.episode_lengths, 'Survival Time', 'g'),
            (axes[1, 0], self.metrics.pellet_eaten, 'Pellet Consumption', 'orange'),
            (axes[1, 1], self.metrics.fish_eaten, 'Fish Predation', 'red'),
            (axes[1, 2], self.metrics.growth_events, 'Growth Events per Episode', 'green'),
            (axes[2, 0], self.metrics.times_chased, 'Chase & Escape Events', 'red'),
            (axes[2, 1], [r * 100 for r in self.metrics.rest_ratio], 'Rest Behavior', 'blue'),
            (axes[2, 2], self.metrics.avg_relative_density, 'Buoyancy Control', 'navy'),
            (axes[3, 0], self.metrics.entropies, 'Policy Entropy', 'purple'),
            (axes[3, 1], self.metrics.value_losses, 'Value Loss', 'red'),
            (axes[3, 2], self.metrics.explained_variances, 'Value Function Quality', 'green'),
        ]

        for ax, data, title, color in plot_data:
            y = downsample(data)
            ax.plot(timesteps[:len(y)], y, color=color, alpha=0.7, linewidth=1.5)
            ax.set_xlabel('Training Steps (M)')
            ax.set_title(title, fontweight='bold')
            ax.grid(True, alpha=0.3)

        # Add reference lines
        axes[0, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[0, 2].axhline(y=CONFIG.environment.max_episode_steps, color='r', linestyle='--', alpha=0.7)
        axes[2, 1].axhspan(20, 40, alpha=0.2, color='green')
        axes[2, 2].axhline(y=1.0, color='green', linestyle='--', alpha=0.7)
        axes[2, 2].axhspan(0.98, 1.02, alpha=0.2, color='green')

        plt.tight_layout()
        plt.savefig(os.path.join(self.metrics.log_dir, f'training_curves_{self.num_timesteps}.png'), dpi=150,
                    bbox_inches='tight')
        plt.close()

    def _save_checkpoint(self) -> None:
        """Save periodic and best-model checkpoints."""
        path = os.path.join(self.model_dir, f"bass_ppo_{self.num_timesteps}.zip")
        self.model.save(path)

        stats = self.metrics.get_current_stats()
        if stats and stats.get('mass_change_mean', -999) > self.best_mass_change:
            best_path = os.path.join(self.model_dir, "bass_ppo_best.zip")
            self.model.save(best_path)
            print(f"\n[Checkpoint] Best model saved: {best_path} (mass_change={stats['mass_change_mean']:+.3f}g)")

    def _print_summary(self) -> None:
        """Print end-of-training summary statistics."""
        stats = self.metrics.get_current_stats()
        elapsed = time.time() - self.start_time
        print(f"""
╔═══════════════════════════════════════════════════════════════════════════════════╗
║                        Training Complete (v6.2 Simplified)                        ║
╠═══════════════════════════════════════════════════════════════════════════════════╣
║  Duration: {str(timedelta(seconds=int(elapsed))):>15}       FPS: {self.num_timesteps / elapsed:>8.0f}       Episodes: {self.metrics.total_episodes:>8}   ║
║  Mean Reward: {stats.get('reward_mean', 0):>10.2f}         Best: {self.best_reward:>10.2f}                             ║
║  Mass Change: {stats.get('mass_change_mean', 0):>+10.4f}g        Best: {self.best_mass_change:>+10.4f}g                            ║
║  Fish/Episode: {stats.get('fish_mean', 0):>9.3f}          Growth Events: {stats.get('growth_mean', 0):>6.2f}                        ║
╚═══════════════════════════════════════════════════════════════════════════════════╝
""")

    def _format_number(self, n: int) -> str:
        """Format large numbers with M/K suffixes for display.

        Args:
            n: Integer to format.

        Returns:
            Human-readable string with appropriate suffix.
        """
        if n >= 1e6:
            return f"{n / 1e6:.1f}M"
        elif n >= 1e3:
            return f"{n / 1e3:.0f}K"
        return str(n)

    def set_curriculum_stage(self, stage: int) -> None:
        """Update the tracked curriculum stage index.

        Args:
            stage: New curriculum stage index.
        """
        self.curriculum_stage = stage


# ═══════════════════════════════════════════════════════════════════════════════
# Curriculum Learning Callback
# ═══════════════════════════════════════════════════════════════════════════════

class CurriculumCallback(BaseCallback):
    """Callback for automatic curriculum stage progression based on performance.

    Monitors recent episode statistics and promotes the agent to the next
    curriculum stage when survival, food intake, and success rate thresholds
    are satisfied. Stage progression can be frozen for ablation studies.

    Args:
        envs: Vectorized environment instance.
        training_callback: Parent training callback (for stage index sync).
        check_freq: Steps between progression checks.
        start_stage: Initial curriculum stage index.
        freeze_progression: If True, disable automatic stage advancement.
        verbose: Verbosity level.
    """

    def __init__(self, envs: Any, training_callback: TrainingCallback, check_freq: int = 50000,
                 start_stage: int = 0, freeze_progression: bool = False,
                 verbose: int = 1):
        super().__init__(verbose)
        self.envs = envs
        self.training_callback = training_callback
        self.check_freq = check_freq
        self.current_stage = int(max(0, start_stage))
        self.freeze_progression = bool(freeze_progression)
        self.episode_stats = {'steps': [], 'food': [], 'mass_change': []}
        self._apply_stage(self.current_stage)

    def _apply_stage(self, stage_idx: int) -> None:
        """Apply the specified curriculum stage to all environments.

        Args:
            stage_idx: Target curriculum stage index.
        """
        if stage_idx >= len(CURRICULUM_STAGES):
            stage_idx = len(CURRICULUM_STAGES) - 1
        stage = CURRICULUM_STAGES[stage_idx]
        self.current_stage = stage_idx
        self.training_callback.set_curriculum_stage(stage_idx)
        config = {'stage': stage.stage, 'name': stage.name,
                  'capture_multiplier': stage.capture_multiplier,
                  'predation_multiplier': stage.predation_multiplier,
                  'energy_cost_multiplier': stage.energy_cost_multiplier,
                  'food_amount_multiplier': stage.food_amount_multiplier}
        try:
            self.envs.set_attr('curriculum_config', config)
        except:
            pass
        if self.verbose:
            print(f"\n[Curriculum] Stage {stage_idx}: {stage.name}")

    def _on_step(self) -> bool:
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                self.episode_stats['steps'].append(info['episode']['l'])
                self.episode_stats['food'].append(info.get('total_food_eaten', 0))
                self.episode_stats['mass_change'].append(info.get('mass_change', 0))
                for key in self.episode_stats:
                    if len(self.episode_stats[key]) > 200:
                        self.episode_stats[key] = self.episode_stats[key][-200:]

        if self.num_timesteps % self.check_freq == 0:
            self._check_progress()
        return True

    def _check_progress(self) -> None:
        """Evaluate whether the agent has met promotion criteria for the current stage."""
        if self.freeze_progression:
            return
        if len(self.episode_stats['steps']) < 50 or self.current_stage >= len(CURRICULUM_STAGES) - 1:
            return
        stage = CURRICULUM_STAGES[self.current_stage]
        threshold = stage.success_threshold
        recent_steps = self.episode_stats['steps'][-100:]
        recent_food = self.episode_stats['food'][-100:]
        avg_steps = np.mean(recent_steps)
        avg_food = np.mean(recent_food)
        success_rate = sum(s >= threshold['avg_steps'] for s in recent_steps) / len(recent_steps)
        if avg_steps >= threshold['avg_steps'] and avg_food >= threshold['avg_food'] and success_rate >= threshold[
            'success_rate']:
            self._apply_stage(self.current_stage + 1)
            self.episode_stats = {'steps': [], 'food': [], 'mass_change': []}


# ═══════════════════════════════════════════════════════════════════════════════
# Phased Environment Callback
# ═══════════════════════════════════════════════════════════════════════════════

class EnvironmentPhaseCallback(BaseCallback):
    """Callback implementing four-course phased environment training.

    Decomposes skill acquisition into progressive courses:
      - course1: Rapid precise foraging (no obstacles, no predators)
      - course2: Obstacle navigation with foraging (maze/obstacles, no predators)
      - course3: Threat-aware foraging and predation (obstacles + large-fish threats)
      - course4: Full mixed-environment convergence

    Progression rules:
      1. Training progress determines the target course ceiling.
      2. Advancement requires mastery of the current course (competence gating).
      3. Automatic demotion occurs upon detecting performance collapse.

    Args:
        envs: Vectorized environment instance.
        total_timesteps: Total training budget for progress ratio computation.
        phase1_ratio: Cumulative progress fraction allocated to course1.
        phase2_ratio: Cumulative progress fraction allocated to course2.
        phase3_ratio: Cumulative progress fraction allocated to course3.
        start_course: Minimum course floor (training never demotes below this).
        check_freq: Steps between phase transition evaluations.
        min_episodes_per_phase: Minimum episodes required before mastery assessment.
        verbose: Verbosity level.
    """

    PHASE_ORDER = ('course1', 'course2', 'course3', 'course4')

    def __init__(self, envs: Any, total_timesteps: int,
                 phase1_ratio: float = 0.20, phase2_ratio: float = 0.45,
                 phase3_ratio: float = 0.75,
                 start_course: str = 'course1',
                 check_freq: int = 50_000,
                 min_episodes_per_phase: int = 40,
                 verbose: int = 1):
        super().__init__(verbose)
        self.envs = envs
        self.total_timesteps = max(1, int(total_timesteps))
        self.phase1_ratio = float(np.clip(phase1_ratio, 0.0, 1.0))
        self.phase2_ratio = float(np.clip(phase2_ratio, self.phase1_ratio, 1.0))
        self.phase3_ratio = float(np.clip(phase3_ratio, self.phase2_ratio, 1.0))
        start_course = str(start_course).lower()
        if start_course not in self.PHASE_ORDER:
            start_course = 'course1'
        self.min_phase_idx = self.PHASE_ORDER.index(start_course)
        self.check_freq = max(1, int(check_freq))
        self.min_episodes_per_phase = max(10, int(min_episodes_per_phase))
        self.current_phase = None
        self.phase_episode_stats = {
            phase: deque(maxlen=240) for phase in self.PHASE_ORDER
        }
        self._last_demote_step = -10**12
        self._last_check_step = 0

    def _phase_desc(self, phase: str) -> str:
        """Return a human-readable description of the given phase.

        Args:
            phase: Phase identifier string.

        Returns:
            English description of the phase objectives.
        """
        desc = {
            'course1': "Rapid precise foraging (no obstacles/predators)",
            'course2': "Obstacle navigation with foraging (maze)",
            'course3': "Threat-aware foraging and predation",
            'course4': "Full mixed-environment convergence",
        }
        return desc.get(phase, phase)

    def _apply_phase(self, phase: str) -> None:
        """Switch all environments to the specified training phase.

        Args:
            phase: Target phase identifier.
        """
        if phase == self.current_phase:
            return

        self.current_phase = phase
        try:
            self.envs.env_method('set_training_phase', phase)
        except Exception as e:
            if self.verbose:
                print(f"\n[Warning] [EnvPhase] set_training_phase failed: {e}")
            return

        if self.verbose:
            print(f"\n[EnvPhase] {phase}: {self._phase_desc(phase)}")

    def _on_training_start(self) -> None:
        self._apply_phase(self.PHASE_ORDER[self.min_phase_idx])

    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [])
        for info in infos:
            if 'episode' not in info:
                continue
            if self.current_phase is None:
                continue

            steps = int(info['episode'].get('l', 0))
            first_intake = int(info.get('first_intake_step', -1))
            if first_intake < 0:
                first_intake = steps + 1

            self.phase_episode_stats[self.current_phase].append({
                'steps': steps,
                'food': float(info.get('total_food_eaten', 0.0)),
                'fish': float(info.get('total_fish_eaten', 0.0)),
                'collisions': float(info.get('collision_count', 0.0)),
                'rest_ratio': float(info.get('rest_ratio', 0.0)),
                'first_intake': float(first_intake),
                'death_reason': str(info.get('death_reason', '') or ''),
            })

        if self.num_timesteps - self._last_check_step < self.check_freq:
            return True
        self._last_check_step = self.num_timesteps

        target_phase = self._desired_phase_by_progress()
        target_idx = self.PHASE_ORDER.index(target_phase)
        target_idx = max(target_idx, self.min_phase_idx)
        current_idx = self.PHASE_ORDER.index(self.current_phase) if self.current_phase in self.PHASE_ORDER else 0

        # Advance one level at a time via competence gating (no level skipping)
        next_idx = current_idx
        while next_idx < target_idx:
            phase_to_master = self.PHASE_ORDER[next_idx]
            if self._is_phase_mastered(phase_to_master):
                next_idx += 1
            else:
                break

        # Collapse protection: auto-demote one level if performance degrades severely
        if next_idx >= 1 and self._should_demote(self.PHASE_ORDER[next_idx]):
            if self.num_timesteps - self._last_demote_step >= 2 * self.check_freq:
                next_idx = max(self.min_phase_idx, next_idx - 1)
                self._last_demote_step = self.num_timesteps
                if self.verbose:
                    print(f"\n[EnvPhase] Performance collapse detected, reverting to {self.PHASE_ORDER[next_idx]}")

        self._apply_phase(self.PHASE_ORDER[next_idx])
        return True

    def _desired_phase_by_progress(self) -> str:
        """Determine the target phase ceiling based on training progress.

        Returns:
            Phase identifier corresponding to the current progress ratio.
        """
        progress = self.num_timesteps / self.total_timesteps
        if progress < self.phase1_ratio:
            return 'course1'
        if progress < self.phase2_ratio:
            return 'course2'
        if progress < self.phase3_ratio:
            return 'course3'
        return 'course4'

    def _recent_stats(self, phase: str, n: int = 80) -> Optional[List[Dict[str, Any]]]:
        """Retrieve recent episode statistics for a given phase.

        Args:
            phase: Phase identifier.
            n: Maximum number of recent episodes to return.

        Returns:
            List of episode stat dictionaries, or None if insufficient data.
        """
        data = list(self.phase_episode_stats.get(phase, []))
        if len(data) < self.min_episodes_per_phase:
            return None
        return data[-n:]

    def _is_phase_mastered(self, phase: str) -> bool:
        """Evaluate whether the agent has mastered the specified phase.

        Mastery criteria vary by phase and assess food intake frequency,
        first-intake latency, collision rate, rest ratio, and predation
        survival as appropriate.

        Args:
            phase: Phase identifier to evaluate.

        Returns:
            True if mastery criteria are satisfied.
        """
        stats = self._recent_stats(phase)
        if not stats:
            return False

        steps = np.array([s['steps'] for s in stats], dtype=np.float32)
        food = np.array([s['food'] for s in stats], dtype=np.float32)
        fish = np.array([s['fish'] for s in stats], dtype=np.float32)
        first = np.array([s['first_intake'] for s in stats], dtype=np.float32)
        collisions = np.array([s['collisions'] for s in stats], dtype=np.float32)
        rest_ratio = np.array([s['rest_ratio'] for s in stats], dtype=np.float32)

        collision_rate = collisions / np.maximum(steps, 1.0)

        if phase == 'course1':
            return (
                float(np.mean(food >= 5.0)) >= 0.65 and
                float(np.mean(first <= 3000.0)) >= 0.55 and
                float(np.mean(collision_rate)) <= 0.030 and
                float(np.mean(rest_ratio)) <= 0.55
            )
        if phase == 'course2':
            return (
                float(np.mean(food >= 3.0)) >= 0.55 and
                float(np.mean(first <= 6500.0)) >= 0.45 and
                float(np.mean(collision_rate)) <= 0.045 and
                float(np.mean(rest_ratio)) <= 0.60
            )
        if phase == 'course3':
            predation_death_rate = float(np.mean([
                1.0 if s['death_reason'] == 'predation' else 0.0 for s in stats
            ]))
            return (
                float(np.mean(food >= 2.0)) >= 0.45 and
                float(np.mean(fish >= 1.0)) >= 0.22 and
                predation_death_rate <= 0.45 and
                float(np.mean(rest_ratio)) <= 0.62
            )
        return True

    def _should_demote(self, phase: str) -> bool:
        """Determine whether performance has collapsed and demotion is warranted.

        Args:
            phase: Current phase to evaluate for collapse.

        Returns:
            True if food intake is critically low and rest ratio is excessively
            high, indicating the agent has ceased productive behaviour.
        """
        if phase == 'course1':
            return False
        stats = self._recent_stats(phase, n=40)
        if not stats:
            return False
        food = np.array([s['food'] for s in stats], dtype=np.float32)
        rest_ratio = np.array([s['rest_ratio'] for s in stats], dtype=np.float32)
        return float(np.mean(food)) < 0.8 and float(np.mean(rest_ratio)) > 0.72


# ═══════════════════════════════════════════════════════════════════════════════
# Environment Factory
# ═══════════════════════════════════════════════════════════════════════════════

def make_env(rank: int, seed: int = 0, training_phase: str = 'course1') -> Callable:
    """Create a factory function for a single BassEnvironment instance.

    Args:
        rank: Index of this environment in the vectorized set (used for
            per-environment seed offset).
        seed: Base random seed.
        training_phase: Initial training phase identifier.

    Returns:
        A zero-argument callable that instantiates and returns a BassEnvironment.
    """
    def _init():
        env = BassEnvironment({'verbose': 0, 'training_phase': training_phase})
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init


def create_vec_env(n_envs: int = 8, seed: int = 42, use_subproc: bool = True,
                   training_phase: str = 'course1') -> VecMonitor:
    """Create a vectorized, monitored environment for parallel rollout collection.

    Attempts SubprocVecEnv for true parallelism; falls back to DummyVecEnv on
    failure or when n_envs <= 4.

    Args:
        n_envs: Number of parallel environments.
        seed: Base random seed.
        use_subproc: Whether to attempt subprocess-based parallelism.
        training_phase: Initial training phase for all environments.

    Returns:
        A VecMonitor-wrapped vectorized environment.
    """
    env_fns = [make_env(i, seed, training_phase) for i in range(n_envs)]
    if use_subproc and n_envs > 4:
        try:
            env = SubprocVecEnv(env_fns)
            print(f"   [OK] SubprocVecEnv ({n_envs} processes)")
        except Exception as e:
            print(f"   [Warning] SubprocVecEnv failed: {e}")
            env = DummyVecEnv(env_fns)
    else:
        env = DummyVecEnv(env_fns)
        print(f"   [OK] DummyVecEnv ({n_envs} environments)")
    return VecMonitor(env)


# ═══════════════════════════════════════════════════════════════════════════════
# Main Training Function
# ═══════════════════════════════════════════════════════════════════════════════

def train(
        total_timesteps: int = 50_000_000,
        n_envs: Optional[int] = None,
        batch_size: Optional[int] = None,
        log_dir: str = "./logs",
        model_dir: str = "./models",
        resume_path: Optional[str] = None,
        seed: int = 42,
        use_gpu: bool = True,
        log_freq: int = 100000,
        plot_freq: int = 500000,
        staged_env: bool = False,
        phase1_ratio: float = 0.20,
        phase2_ratio: float = 0.45,
        phase3_ratio: float = 0.75,
        start_course: str = 'course1',
        start_stage: int = 0,
        freeze_curriculum: bool = False,
        recurrent: bool = False,
        lstm_hidden_size: int = 256,
        lstm_layers: int = 1,
) -> Any:
    """Execute the full PPO training loop for the largemouth bass agent.

    Configures hardware, creates vectorized environments, initialises the PPO
    model (or loads from checkpoint), registers all callbacks, and runs the
    training loop. On completion or interruption, saves the final model.

    Args:
        total_timesteps: Total environment interaction budget.
        n_envs: Number of parallel environments (auto-detected if None).
        batch_size: PPO minibatch size (auto-detected if None).
        log_dir: Root directory for training logs.
        model_dir: Directory for model checkpoints.
        resume_path: Path to a saved model to resume training from.
        seed: Global random seed for reproducibility.
        use_gpu: Whether to use CUDA if available.
        log_freq: Steps between metric logging events.
        plot_freq: Steps between training curve plot generation.
        staged_env: Enable four-course phased environment training.
        phase1_ratio: Cumulative progress fraction for course1.
        phase2_ratio: Cumulative progress fraction for course2.
        phase3_ratio: Cumulative progress fraction for course3.
        start_course: Minimum course floor for phased training.
        start_stage: Initial curriculum difficulty stage.
        freeze_curriculum: Disable automatic curriculum progression.
        recurrent: Use RecurrentPPO with LSTM policy.
        lstm_hidden_size: Hidden size for LSTM layers.
        lstm_layers: Number of LSTM layers.

    Returns:
        The trained PPO model instance.
    """
    if n_envs is None:
        n_envs = HARDWARE['recommended_n_envs']
    if batch_size is None:
        batch_size = HARDWARE['recommended_batch_size']

    if staged_env:
        if phase2_ratio <= phase1_ratio:
            phase2_ratio = min(0.90, phase1_ratio + 0.20)
        if phase3_ratio <= phase2_ratio:
            phase3_ratio = min(0.98, phase2_ratio + 0.20)

    device = "cuda" if use_gpu and HARDWARE['cuda_available'] else "cpu"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(log_dir, f"run_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print(f"\n[Init] Creating training environment...")
    print(f"   Device: {device}")
    print(f"   Environments: {n_envs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Staged Env: {staged_env}")
    if staged_env:
        print(f"   Course ratios: C1<{phase1_ratio:.2f}, C2<{phase2_ratio:.2f}, C3<{phase3_ratio:.2f}, else C4")
        print(f"   Start course floor: {start_course}")
    print(f"   Curriculum start stage: {start_stage}{' (frozen)' if freeze_curriculum else ''}")
    print(f"   Recurrent PPO: {recurrent}")

    env = create_vec_env(n_envs, seed, use_subproc=True, training_phase=start_course)

    # PPO hyperparameters
    LEARNING_RATE = 3e-4  # PPO default
    ENT_COEF = 0.01  # Mild exploration bonus (recommended for continuous action spaces)

    print(f"   Learning Rate: {LEARNING_RATE} (linear decay to 3e-5)")
    print(f"   Entropy Coef: {ENT_COEF} (fixed initial, decayed via callback)")
    print(f"   Other params: PPO defaults from config")

    use_recurrent = bool(recurrent and HAS_RECURRENT_PPO)
    if recurrent and not HAS_RECURRENT_PPO:
        print("   [Warning] sb3-contrib not installed, falling back to standard PPO (MlpPolicy)")

    model_cls = RecurrentPPO if use_recurrent else PPO
    policy_name = "MlpLstmPolicy" if use_recurrent else "MlpPolicy"
    policy_kwargs = {
        'net_arch': dict(pi=list(CONFIG.network.policy_layers), vf=list(CONFIG.network.value_layers)),
    }
    if use_recurrent:
        policy_kwargs.update({
            'lstm_hidden_size': int(lstm_hidden_size),
            'n_lstm_layers': int(lstm_layers),
            'shared_lstm': False,
            'enable_critic_lstm': True,
        })

    # Create or load model
    if resume_path and os.path.exists(resume_path):
        print(f"[Resume] Loading model: {resume_path}")
        model = model_cls.load(resume_path, env=env, device=device)
        model.learning_rate = linear_schedule(3e-4)
    else:
        print("[Init] Creating new model with PPO defaults...")
        model = model_cls(
            policy=policy_name,
            env=env,
            # Core hyperparameters
            learning_rate=linear_schedule(3e-4, 3e-5),  # Decay to 3e-5 (not zero)
            ent_coef=0.01,  # Initial; decayed by EntropyDecayCallback
            # Additional parameters from config
            n_steps=CONFIG.ppo.n_steps,
            batch_size=batch_size,
            n_epochs=CONFIG.ppo.n_epochs,
            gamma=CONFIG.ppo.gamma,
            gae_lambda=CONFIG.ppo.gae_lambda,
            clip_range=CONFIG.ppo.clip_range,
            vf_coef=CONFIG.ppo.vf_coef,
            max_grad_norm=CONFIG.ppo.max_grad_norm,
            # No target_kl constraint for unconstrained exploration
            target_kl=None,
            verbose=0,
            tensorboard_log=os.path.join(log_dir, "tensorboard"),
            seed=seed,
            device=device,
            policy_kwargs=policy_kwargs
        )

    # Register callbacks
    metrics = EnhancedTrainingMetrics(log_dir)
    training_callback = TrainingCallback(total_timesteps, metrics, log_freq, plot_freq, 500000, model_dir)
    curriculum_callback = CurriculumCallback(
        env, training_callback, 50000,
        start_stage=start_stage,
        freeze_progression=freeze_curriculum
    )

    entropy_decay_callback = EntropyDecayCallback(
        initial_ent=0.001,
        final_ent=0.0002,
        total_timesteps=total_timesteps
    )
    callbacks = [training_callback, curriculum_callback, entropy_decay_callback]
    if staged_env:
        callbacks.append(
            EnvironmentPhaseCallback(
                env, total_timesteps,
                phase1_ratio=phase1_ratio,
                phase2_ratio=phase2_ratio,
                phase3_ratio=phase3_ratio,
                start_course=start_course,
                verbose=1
            )
        )

    try:
        model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=False)
    except KeyboardInterrupt:
        print("\n\n[Interrupt] Training interrupted, saving model...")
    finally:
        final_path = os.path.join(model_dir, "bass_ppo_final.zip")
        model.save(final_path)
        print(f"[Done] Final model saved: {final_path}")
        env.close()

    return model


# ═══════════════════════════════════════════════════════════════════════════════
# Command-Line Interface
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the training script.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(description="Largemouth Bass RL Training - v6.2 Simplified")
    parser.add_argument('--timesteps', type=str, default='80M', help='Total training steps (e.g., 50M, 100K)')
    parser.add_argument('--n_envs', type=int, default=None, help='Number of parallel environments')
    parser.add_argument('--batch_size', type=int, default=None, help='PPO minibatch size')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Log output directory')
    parser.add_argument('--model_dir', type=str, default='./models', help='Model checkpoint directory')
    parser.add_argument('--resume', type=str, default=None, help='Path to model for resumed training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--log_freq', type=int, default=10000)
    parser.add_argument('--plot_freq', type=int, default=50000)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--dummy', action='store_true', help='Force DummyVecEnv (no subprocess)')
    parser.add_argument('--staged_env', action='store_true', help='Enable phased environment training')
    parser.add_argument('--phase1_ratio', type=float, default=0.20, help='Cumulative progress ratio for course1')
    parser.add_argument('--phase2_ratio', type=float, default=0.45, help='Cumulative progress ratio for course2')
    parser.add_argument('--phase3_ratio', type=float, default=0.75, help='Cumulative progress ratio for course3')
    parser.add_argument('--start_course', type=str, default='course1',
                        choices=['course1', 'course2', 'course3', 'course4'],
                        help='Minimum course floor for phased training')
    parser.add_argument('--start_stage', type=int, default=0, help='Initial curriculum stage index')
    parser.add_argument('--freeze_curriculum', action='store_true', help='Freeze curriculum (no auto-progression)')
    parser.add_argument('--recurrent', action='store_true', help='Use RecurrentPPO (MlpLstmPolicy)')
    parser.add_argument('--lstm_hidden', type=int, default=256, help='LSTM hidden layer dimension')
    parser.add_argument('--lstm_layers', type=int, default=1, help='Number of LSTM layers')
    return parser.parse_args()


def parse_timesteps(s: str) -> int:
    """Parse a human-readable timestep string (e.g., '50M', '100K') to integer.

    Args:
        s: Timestep string with optional M (million) or K (thousand) suffix.

    Returns:
        Integer number of timesteps.
    """
    s = s.upper().strip()
    if s.endswith('M'):
        return int(float(s[:-1]) * 1_000_000)
    elif s.endswith('K'):
        return int(float(s[:-1]) * 1_000)
    return int(s)


if __name__ == "__main__":
    args = parse_args()
    timesteps = parse_timesteps(args.timesteps)
    use_gpu = not args.cpu and (args.gpu or HARDWARE['cuda_available'])

    train(
        total_timesteps=timesteps,
        n_envs=args.n_envs,
        batch_size=args.batch_size,
        log_dir=args.log_dir,
        model_dir=args.model_dir,
        resume_path=args.resume,
        seed=args.seed,
        use_gpu=use_gpu,
        log_freq=args.log_freq,
        plot_freq=args.plot_freq,
        staged_env=args.staged_env,
        phase1_ratio=args.phase1_ratio,
        phase2_ratio=args.phase2_ratio,
        phase3_ratio=args.phase3_ratio,
        start_course=args.start_course,
        start_stage=args.start_stage,
        freeze_curriculum=args.freeze_curriculum,
        recurrent=args.recurrent,
        lstm_hidden_size=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
    )
