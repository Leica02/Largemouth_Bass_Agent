#!/usr/bin/env python3
"""
Multi-AI Fish Simulation Manager - Academic Style v2.0
======================================================
Multi-agent fish simulation manager for concurrent simulation of multiple
AI-controlled largemouth bass (Micropterus salmoides) sharing a single
BassEnvironment instance. Implements per-fish state isolation, observation
construction, food capture mechanics, and AI-to-AI predation.

Features:
    1. Detailed per-fish AI state panels (speed, action vector, buoyancy)
    2. Real-time data charts (energy, body mass, swimming speed)
    3. Interactive fish list with selection
    4. Coordinate axes with scale markers
    5. Simulation time display

Requirements:
    pip install pygame numpy

Usage:
    python multi_fish_pygame_v2.py --model models/bass_ppo_final.zip
    python multi_fish_pygame_v2.py --model models/bass_ppo_final.zip --num_fish 10
    python multi_fish_pygame_v2.py --num_fish 20  # Random actions (no model)

Controls:
    SPACE  - Pause/Resume
    R      - Reset
    +/-    - Speed up/down
    F      - Toggle fullscreen
    ESC    - Quit
    Mouse  - Click fish list to select
"""

import numpy as np
import sys
import os
import argparse
from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional, Tuple
from collections import deque
from enum import Enum

# ============================================================
# Import PyGame
# ============================================================
try:
    import pygame
    from pygame import gfxdraw

    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False
    print("[ERROR] PyGame not installed. Run: pip install pygame")
    sys.exit(1)

# ============================================================
# Import environment and model
# ============================================================
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from environment import BassEnvironment
    from config import CONFIG
    from systems.interaction import OtherFish, FishBehaviorType, FishSize, InteractionInput
    from systems.feeding import FeedingInput

    HAS_ENV = True
except ImportError as e:
    HAS_ENV = False
    print(f"[ERROR] Cannot import environment: {e}")
    sys.exit(1)

try:
    from stable_baselines3 import PPO

    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    print("[WARN] stable_baselines3 not installed, using random actions")

try:
    from systems.metabolism import ActivityState
except ImportError:
    class ActivityState(Enum):
        ACTIVE = "active"
        RESTING = "resting"


# ============================================================
# Academic Color Scheme
# ============================================================

class AcademicColors:
    """Academic color scheme — professional palette."""
    # Background colors
    BACKGROUND = (18, 22, 28)
    PANEL_BG = (25, 32, 42)
    PANEL_BORDER = (45, 55, 72)
    PANEL_HEADER = (35, 45, 60)

    # Water colors
    WATER_DEEP = (20, 45, 75)
    WATER_SURFACE = (35, 80, 130)
    WATER_DANGER = (80, 35, 40)
    TANK_BORDER = (70, 100, 140)

    # Text colors
    TEXT_PRIMARY = (230, 235, 240)
    TEXT_SECONDARY = (160, 170, 185)
    TEXT_DIM = (100, 110, 125)
    TEXT_ACCENT = (100, 180, 255)
    TEXT_WARNING = (255, 180, 80)
    TEXT_SUCCESS = (80, 200, 120)
    TEXT_DANGER = (255, 100, 100)

    # Food colors
    FOOD_FLOATING = (255, 210, 80)
    FOOD_SINKING = (255, 160, 50)
    FOOD_SETTLING = (200, 120, 40)

    # NPC fish colors
    NPC_SMALL = (100, 180, 220)
    NPC_MEDIUM = (80, 130, 200)
    NPC_LARGE = (220, 100, 60)
    NPC_PREDATOR = (180, 50, 50)
    NPC_CHASING = (255, 60, 60)

    # Chart colors
    CHART_ENERGY = (255, 180, 80)
    CHART_MASS = (80, 200, 120)
    CHART_SPEED = (100, 180, 255)
    CHART_GRID = (40, 50, 65)
    CHART_BG = (20, 28, 38)

    # Selection glow
    SELECTED_GLOW = (100, 200, 255)
    HOVER_GLOW = (80, 150, 200)


# AI fish colors (distinct, vibrant)
AI_FISH_COLORS = [
    (50, 255, 120),  # Bright green
    (255, 120, 120),  # Coral
    (80, 220, 220),  # Cyan
    (255, 220, 80),  # Gold
    (180, 120, 255),  # Purple
    (255, 150, 200),  # Pink
    (120, 255, 200),  # Mint
    (255, 180, 120),  # Peach
    (120, 180, 255),  # Sky blue
    (255, 120, 180),  # Magenta
    (200, 255, 120),  # Lime
    (120, 220, 180),  # Teal
    (255, 200, 150),  # Apricot
    (150, 150, 255),  # Lavender
    (220, 180, 255),  # Violet
]


# ============================================================
# Helper Functions
# ============================================================

def mass_to_length(mass: float) -> float:
    """Convert mass to length"""
    return CONFIG.growth.length_weight_a * (mass ** (1.0 / CONFIG.growth.length_weight_b))


def world_to_screen_top(pos, center, scale):
    """Convert world coordinates to screen (top view: X-Z plane)"""
    x = int(center[0] + pos[0] * scale)
    y = int(center[1] - pos[2] * scale)
    return (x, y)


def world_to_screen_side(pos, left, top, width, height, tank_radius, tank_depth):
    """Convert world coordinates to screen (side view: X-Y plane)"""
    x = int(left + (pos[0] + tank_radius) / (2 * tank_radius) * width)
    y = int(top + (-pos[1]) / tank_depth * height)
    return (x, y)


def format_time(steps: int, time_step: float = 0.3, acceleration: float = 300) -> str:
    """Convert simulation steps to human-readable time string."""
    real_seconds = steps * time_step
    sim_seconds = real_seconds * acceleration

    hours = int(sim_seconds // 3600)
    minutes = int((sim_seconds % 3600) // 60)

    if hours >= 24:
        days = hours // 24
        hours = hours % 24
        return f"{days}d {hours:02d}h {minutes:02d}m"
    else:
        return f"{hours:02d}h {minutes:02d}m"


# ============================================================
# AI Fish State (Extended)
# ============================================================

@dataclass
class AIFishState:
    """Single AI fish state - extended with history tracking"""
    id: int
    position: np.ndarray
    velocity: np.ndarray
    body_mass: float
    total_length: float
    energy: float = 80.0
    stomach_fullness: float = 30.0
    activity_state: ActivityState = ActivityState.ACTIVE

    # Metabolism tracking
    rest_duration_steps: int = 0
    current_metabolism_factor: float = 1.0
    current_growth_bonus: float = 1.0
    fatigue: float = 0.0
    stress_level: float = 0.0
    is_digesting: bool = False

    # Buoyancy state
    relative_density: float = 1.0
    net_buoyancy_force: float = 0.0
    swimbladder_volume: float = 0.0
    # Per-fish independent BuoyancyState object (v3 pattern)
    # Assigned in reset(); reference-swapped into base_env during sync
    _buoyancy_state: Optional[Any] = field(default=None, repr=False)

    # Digestion state
    stomach_content_mass: float = 0.0
    initial_meal_mass: float = 0.0
    digestion_buffer: float = 0.0

    # Stomach nutrient fractions — per-fish fields for detailed metabolism model
    stomach_protein_fraction: float = 0.0
    stomach_lipid_fraction: float = 0.0
    stomach_carbohydrate_fraction: float = 0.0
    stomach_adc_protein: float = 0.0
    stomach_adc_lipid: float = 0.0
    stomach_adc_carbohydrate: float = 0.0
    stomach_include_carbohydrate_energy: bool = False

    energy_from_digestion: float = 0.0

    # Body composition (must be per-fish so metabolism initialises correctly)
    initial_body_mass: float = 0.0   # set after first metabolism call
    lipid_reserve: float = 0.0
    protein_reserve: float = 0.0

    # Growth state
    growth_accumulation: float = 0.0
    total_growth_energy: float = 0.0
    growth_count: int = 0
    initial_mass: float = 20.0

    # State switch control
    state_switch_cooldown: float = 0.0
    last_state_switch_step: int = 0
    total_rest_steps: int = 0

    # Statistics
    food_eaten: int = 0
    fish_eaten: int = 0
    ai_fish_eaten: int = 0
    steps_alive: int = 0

    # Life status
    is_alive: bool = True
    death_reason: str = None
    killed_by: int = None

    # Chronic malnutrition tracking (used by aquaculture batch sim)
    underfed_days: int = 0
    recent_intake_pct: list = field(default_factory=list)  # rolling 3-day intake (%BW/day)
    # Per-prey attack cooldown: dict mapping prey_id → steps_remaining
    attack_cooldowns: dict = field(default_factory=dict)
    # Per-fish starvation threshold (fraction of initial_mass lost before death)
    # Stored here so the shared MetabolismSystem doesn't overwrite it when switching fish
    death_mass_loss_threshold: float = 0.275

    # Visualization
    position_history: deque = field(default_factory=lambda: deque(maxlen=100))
    color: tuple = (0, 255, 0)

    # === NEW: Action and physics tracking ===
    last_action: np.ndarray = field(default_factory=lambda: np.zeros(5))
    last_propulsion_force: float = 0.0
    current_speed: float = 0.0
    buoyancy_control: float = 0.0
    raw_buoyancy_control: float = 0.0
    buoyancy_mode: int = 0  # -1=rise, 0=hold, +1=sink
    last_action_normalized: bool = False  # Track if action was clamped
    speed_clamped: bool = False  # Track if speed was clamped

    # Physics state fields — per-fish (mirrors v3 pattern)
    # Without these, shared physics_state bleeds between fish each step
    using_burst: bool = False
    heading: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=np.float32))
    smoothed_action: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    smoothed_buoyancy_control: float = 0.0
    current_turn_rate: float = 0.0
    inertia_initialized: bool = False
    pitch_angle: float = 0.0
    target_pitch_angle: float = 0.0
    current_pitch_rate: float = 0.0
    is_coasting: bool = False
    coast_steps_remaining: int = 0
    current_reynolds: float = 0.0
    rest_transition_progress: float = 0.0
    active_transition_progress: float = 0.0
    drift_direction: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    drift_update_counter: int = 0
    previous_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    air_exposure_time: float = 0.0
    in_air: bool = False
    is_at_surface: bool = False

    # === NEW: History for charts ===
    energy_history: deque = field(default_factory=lambda: deque(maxlen=200))
    mass_history: deque = field(default_factory=lambda: deque(maxlen=200))
    speed_history: deque = field(default_factory=lambda: deque(maxlen=200))

    def to_other_fish(self) -> OtherFish:
        """Convert to OtherFish for injection"""
        if self.body_mass < 10:
            size_cat = FishSize.SMALL
        elif self.body_mass < 30:
            size_cat = FishSize.MEDIUM
        else:
            size_cat = FishSize.LARGE

        return OtherFish(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            body_mass=self.body_mass,
            total_length=self.total_length,
            energy=self.energy,
            is_alive=self.is_alive,
            behavior_type=FishBehaviorType.PASSIVE,
            original_behavior=FishBehaviorType.PASSIVE,
            size_category=size_cat,
            is_chasing=False,
            is_fleeing=False
        )

    def update_history(self):
        """Update history data for charts"""
        self.energy_history.append(self.energy)
        self.mass_history.append(self.body_mass)
        self.speed_history.append(self.current_speed)


# ============================================================
# Multi-AI Fish Manager (Same as original, with extensions)
# ============================================================

class MultiAIFishManager:
    """Manages multiple AI fish with extended tracking"""

    def __init__(self, base_env: BassEnvironment, model, num_fish: int = 10,
                 initial_mass: float = 20.0, include_npc: bool = True,
                 verbose: bool = True):
        self.base_env = base_env
        self.model = model
        self.num_fish = num_fish
        self.initial_mass = initial_mass
        self.include_npc = include_npc
        self.verbose = verbose

        self.ai_fish: List[AIFishState] = []
        self.colors = AI_FISH_COLORS[:num_fish] if num_fish <= len(AI_FISH_COLORS) else \
            [AI_FISH_COLORS[i % len(AI_FISH_COLORS)] for i in range(num_fish)]

        self.current_step = 0
        self.ai_predation_events = []

        self.feeding_interval = self.base_env.feeding_system.feeding_interval
        self.steps_since_feeding = 0
        self.total_food_spawned = 0
        self.total_food_eaten = 0

        try:
            from systems.metabolism import ActivityState
            self.ActivityState = ActivityState
        except ImportError:
            class ActivityState(Enum):
                ACTIVE = "active"
                RESTING = "resting"

            self.ActivityState = ActivityState

    def _discretize_buoyancy_action_for_fish(self, fish: AIFishState, raw_action: float) -> float:
        """Match environment state: discretize buoyancy (rise/hold/sink) + hysteresis."""
        enter_th = 0.45
        hold_th = 0.30
        a = float(raw_action)

        if abs(a) <= hold_th:
            fish.buoyancy_mode = 0
        elif a <= -enter_th:
            fish.buoyancy_mode = -1
        elif a >= enter_th:
            fish.buoyancy_mode = 1

        return float(fish.buoyancy_mode)

    def reset(self):
        """Reset all AI fish"""
        self.ai_fish = []
        self.current_step = 0
        self.ai_predation_events = []
        self.steps_since_feeding = 0
        self.total_food_spawned = 0
        self.total_food_eaten = 0

        self.base_env.reset()

        tank_radius = CONFIG.environment.tank_radius
        tank_depth = CONFIG.environment.tank_depth

        for i in range(self.num_fish):
            angle = 2 * np.pi * i / self.num_fish + np.random.uniform(-0.3, 0.3)
            radius = np.random.uniform(0.2, tank_radius * 0.7)
            depth = np.random.uniform(-tank_depth + 0.1, -0.15)

            position = np.array([
                radius * np.cos(angle),
                depth,
                radius * np.sin(angle)
            ], dtype=np.float32)

            velocity = np.random.uniform(-0.02, 0.02, 3).astype(np.float32)

            mass = self.initial_mass
            length = mass_to_length(mass)

            fish = AIFishState(
                id=i,
                position=position,
                velocity=velocity,
                body_mass=mass,
                total_length=length,
                energy=np.random.uniform(70, 90),
                stomach_fullness=np.random.uniform(20, 40),
                stomach_content_mass=0.0,
                initial_meal_mass=0.0,
                digestion_buffer=0.0,
                energy_from_digestion=0.0,
                growth_accumulation=0.0,
                total_growth_energy=0.0,
                growth_count=0,
                initial_mass=mass,
                color=self.colors[i],
                state_switch_cooldown=0.0,
                last_state_switch_step=0,
                total_rest_steps=0,
            )
            fish.position_history.append(position.copy())
            self.ai_fish.append(fish)

        # Create an independent BuoyancyState per fish (v3 pattern).
        # Without this, all fish share one buoyancy_state object and
        # gas_amount / neutral_volume / etc. bleed across fish each step.
        try:
            from systems.buoyancy import create_buoyancy_state
            buoyancy_sys = getattr(
                getattr(self.base_env, 'physics_system', None), 'buoyancy_system', None)
            if buoyancy_sys is not None:
                for fish in self.ai_fish:
                    fish._buoyancy_state = create_buoyancy_state()
                    buoyancy_sys.initialize(
                        fish._buoyancy_state,
                        fish.body_mass,
                        fish.total_length,
                    )
                    fish.relative_density   = fish._buoyancy_state.relative_density
                    fish.net_buoyancy_force = fish._buoyancy_state.net_buoyancy_force
                    fish.swimbladder_volume = fish._buoyancy_state.swimbladder_volume
        except Exception:
            pass  # buoyancy system not available; fall back to shared-state path

        # Initial feeding
        total_mass = self.get_total_mass()
        self._initial_feeding(total_mass)

        if self.verbose:
            print(f"[OK] Initialized {self.num_fish} AI fish, total mass: {total_mass:.1f}g")

    def get_total_mass(self) -> float:
        return sum(f.body_mass for f in self.ai_fish if f.is_alive)

    def _initial_feeding(self, total_mass: float):
        """Initial feeding based on total mass"""
        self.base_env.feeding_state.food_items = []

        fc = CONFIG.feeding
        daily_total = total_mass * fc.daily_feeding_rate
        per_feeding = daily_total / fc.feedings_per_day

        base_floating = (fc.floating_pellets_min + fc.floating_pellets_max) // 2
        base_sinking = (fc.sinking_pellets_min + fc.sinking_pellets_max) // 2

        num_floating = int(base_floating * self.num_fish / 5)
        num_sinking = int(base_sinking * self.num_fish / 5)
        total_pellets = num_floating + num_sinking

        pellet_mass = np.clip(per_feeding / total_pellets,
                              fc.pellet_mass_min, fc.pellet_mass_max)

        feed_input = FeedingInput(
            agent_position=np.array([0.0, -0.3, 0.0]),
            agent_mass=total_mass,
            agent_length=0.1,
            stomach_fullness=0.0
        )

        for _ in range(num_floating):
            food = self.base_env.feeding_system._create_floating_pellet(feed_input, pellet_mass)
            food.age = np.random.randint(0, 50)
            self.base_env.feeding_state.food_items.append(food)

        for _ in range(num_sinking):
            food = self.base_env.feeding_system._create_sinking_pellet(feed_input, pellet_mass)
            self.base_env.feeding_state.food_items.append(food)

        self.total_food_spawned = total_pellets

    def _spawn_food(self):
        """Timed feeding"""
        total_mass = self.get_total_mass()
        if total_mass <= 0:
            return 0

        fc = CONFIG.feeding
        daily_total = total_mass * fc.daily_feeding_rate
        per_feeding = daily_total / fc.feedings_per_day

        alive_count = sum(1 for f in self.ai_fish if f.is_alive)
        num_floating = int(np.random.randint(fc.floating_pellets_min, fc.floating_pellets_max + 1) * alive_count / 5)
        num_sinking = int(np.random.randint(fc.sinking_pellets_min, fc.sinking_pellets_max + 1) * alive_count / 5)
        total_pellets = num_floating + num_sinking

        pellet_mass = np.clip(per_feeding / total_pellets,
                              fc.pellet_mass_min, fc.pellet_mass_max)

        feed_input = FeedingInput(
            agent_position=np.array([0.0, -0.3, 0.0]),
            agent_mass=total_mass,
            agent_length=0.1,
            stomach_fullness=0.0
        )

        for _ in range(num_floating):
            food = self.base_env.feeding_system._create_floating_pellet(feed_input, pellet_mass)
            self.base_env.feeding_state.food_items.append(food)

        for _ in range(num_sinking):
            food = self.base_env.feeding_system._create_sinking_pellet(feed_input, pellet_mass)
            self.base_env.feeding_state.food_items.append(food)

        self.total_food_spawned += total_pellets
        return total_pellets

    def _inject_other_ai_fish(self, current_fish_id: int):
        """Inject other AI fish into interaction system"""
        original_fish = self.base_env.interaction_state.other_fish.copy()

        for fish in self.ai_fish:
            if fish.id != current_fish_id and fish.is_alive:
                other_fish_obj = fish.to_other_fish()
                self.base_env.interaction_state.other_fish.append(other_fish_obj)

        return original_fish

    def _restore_original_fish(self, original_fish: List):
        self.base_env.interaction_state.other_fish = original_fish

    def _sync_env_state(self, fish: AIFishState):
        """Sync AI fish state TO environment state"""
        ps = self.base_env.physics_state
        ps.position = fish.position.copy()
        ps.velocity = fish.velocity.copy()

        # Physics fields — per-fish (prevents bleed-through across fish each step)
        if hasattr(ps, 'using_burst'):          ps.using_burst = fish.using_burst
        if hasattr(ps, 'heading'):              ps.heading = fish.heading.copy()
        if hasattr(ps, 'smoothed_action'):      ps.smoothed_action = fish.smoothed_action.copy()
        if hasattr(ps, 'smoothed_buoyancy_control'): ps.smoothed_buoyancy_control = fish.smoothed_buoyancy_control
        if hasattr(ps, 'current_turn_rate'):    ps.current_turn_rate = fish.current_turn_rate
        if hasattr(ps, 'inertia_initialized'):  ps.inertia_initialized = fish.inertia_initialized
        if hasattr(ps, 'pitch_angle'):          ps.pitch_angle = fish.pitch_angle
        if hasattr(ps, 'target_pitch_angle'):   ps.target_pitch_angle = fish.target_pitch_angle
        if hasattr(ps, 'current_pitch_rate'):   ps.current_pitch_rate = fish.current_pitch_rate
        if hasattr(ps, 'is_coasting'):          ps.is_coasting = fish.is_coasting
        if hasattr(ps, 'coast_steps_remaining'):ps.coast_steps_remaining = fish.coast_steps_remaining
        if hasattr(ps, 'current_reynolds'):     ps.current_reynolds = fish.current_reynolds
        if hasattr(ps, 'rest_transition_progress'):   ps.rest_transition_progress = fish.rest_transition_progress
        if hasattr(ps, 'active_transition_progress'): ps.active_transition_progress = fish.active_transition_progress
        if hasattr(ps, 'drift_direction'):      ps.drift_direction = fish.drift_direction.copy()
        if hasattr(ps, 'drift_update_counter'): ps.drift_update_counter = fish.drift_update_counter
        if hasattr(ps, 'previous_velocity'):    ps.previous_velocity = fish.previous_velocity.copy()
        if hasattr(ps, 'activity_state'):       ps.activity_state = fish.activity_state
        if hasattr(ps, 'air_exposure_time'):    ps.air_exposure_time = fish.air_exposure_time
        if hasattr(ps, 'in_air'):               ps.in_air = fish.in_air
        if hasattr(ps, 'is_at_surface'):        ps.is_at_surface = fish.is_at_surface

        ms = self.base_env.metabolism_state
        ms.energy = fish.energy
        ms.stomach_fullness = fish.stomach_fullness
        ms.stomach_content_mass = fish.stomach_content_mass
        ms.initial_meal_mass = fish.initial_meal_mass
        ms.digestion_buffer = fish.digestion_buffer
        ms.energy_from_digestion = fish.energy_from_digestion
        ms.activity_state = fish.activity_state
        ms.rest_duration_steps = fish.rest_duration_steps
        ms.current_metabolism_factor = fish.current_metabolism_factor
        ms.current_growth_bonus = fish.current_growth_bonus
        ms.fatigue = fish.fatigue
        ms.stress_level = fish.stress_level
        ms.is_digesting = fish.is_digesting
        ms.stomach_protein_fraction = fish.stomach_protein_fraction
        ms.stomach_lipid_fraction = fish.stomach_lipid_fraction
        ms.stomach_carbohydrate_fraction = fish.stomach_carbohydrate_fraction
        ms.stomach_adc_protein = fish.stomach_adc_protein
        ms.stomach_adc_lipid = fish.stomach_adc_lipid
        ms.stomach_adc_carbohydrate = fish.stomach_adc_carbohydrate
        ms.stomach_include_carbohydrate_energy = fish.stomach_include_carbohydrate_energy
        ms.state_switch_cooldown = fish.state_switch_cooldown
        ms.last_state_switch_step = fish.last_state_switch_step
        ms.total_rest_steps = fish.total_rest_steps
        ms.initial_body_mass = fish.initial_body_mass
        ms.lipid_reserve = fish.lipid_reserve
        ms.protein_reserve = fish.protein_reserve
        ms.death_mass_loss_threshold = fish.death_mass_loss_threshold

        gs = self.base_env.growth_state
        gs.body_mass = fish.body_mass
        gs.total_length = fish.total_length
        gs.growth_accumulation = fish.growth_accumulation
        gs.total_growth_energy = fish.total_growth_energy
        gs.growth_count = fish.growth_count
        gs.initial_mass = fish.initial_mass

        # Buoyancy: reference-swap so physics system writes into THIS fish's object
        if fish._buoyancy_state is not None:
            ps.buoyancy_state = fish._buoyancy_state
            if hasattr(ps, 'buoyancy_initialized'):
                ps.buoyancy_initialized = True
        elif hasattr(ps, 'buoyancy_state') and ps.buoyancy_state is not None:
            # Fallback: field-copy for fish that predate per-fish buoyancy init
            ps.buoyancy_state.relative_density   = fish.relative_density
            ps.buoyancy_state.net_buoyancy_force  = fish.net_buoyancy_force
            ps.buoyancy_state.swimbladder_volume  = fish.swimbladder_volume

        self.base_env.current_step = self.current_step

    def _get_observation(self, fish: AIFishState) -> np.ndarray:
        self._sync_env_state(fish)
        original_fish = self._inject_other_ai_fish(fish.id)

        perception_input = self.base_env._create_perception_input()
        self.base_env.perception_system.update(
            self.base_env.perception_state, perception_input
        )

        obs = self.base_env._get_observation()
        self._restore_original_fish(original_fish)

        return obs

    def _update_npc_fish(self):
        """Update NPC fish AI"""
        alive_ai = [f for f in self.ai_fish if f.is_alive]
        if not alive_ai:
            return

        ref_fish = alive_ai[0]

        interaction_input = InteractionInput(
            agent_position=ref_fish.position,
            agent_velocity=ref_fish.velocity,
            agent_mass=ref_fish.body_mass,
            agent_length=ref_fish.total_length,
            agent_fatigue=0.0,
            agent_stress=0.0
        )

        self.base_env.interaction_system.update(
            self.base_env.interaction_state,
            interaction_input,
            curriculum_config={'capture_multiplier': 1.0, 'predation_multiplier': 1.0}
        )

    def _update_food_movement(self):
        """Update food movement"""
        feed_input = FeedingInput(
            agent_position=np.array([0.0, -0.5, 0.0]),
            agent_mass=self.get_total_mass(),
            agent_length=0.1,
            stomach_fullness=0.0
        )

        feeding_system = self.base_env.feeding_system
        state = self.base_env.feeding_state

        settle_trigger = feeding_system.settle_trigger_step
        if (self.steps_since_feeding == settle_trigger and
                not state.current_batch_settling):
            feeding_system._trigger_all_food_settling(state)
            state.current_batch_settling = True

        feeding_system._update_food_movement(state, feed_input)

    def _execute_action(self, fish: AIFishState, action: np.ndarray):
        """Execute action with extended tracking"""
        movement_action = action[:3].copy()
        state_action = action[3] if len(action) > 3 else 0.0
        raw_buoyancy_action = action[4] if len(action) > 4 else 0.0
        buoyancy_action = self._discretize_buoyancy_action_for_fish(fish, raw_buoyancy_action)

        # Store original action for visualization
        fish.last_action = action.copy()
        fish.buoyancy_control = buoyancy_action
        fish.raw_buoyancy_control = float(raw_buoyancy_action)

        # === CRITICAL FIX: Normalize movement action ===
        # Action space Box(-1,1) limits each dimension independently
        # but combined magnitude can exceed 1.0 (up to √3 ≈ 1.73)
        # This causes unrealistic "glitchy" high-speed movement
        action_magnitude = np.linalg.norm(movement_action)
        if action_magnitude > 1.0:
            movement_action = movement_action / action_magnitude
            # Store normalized magnitude for debugging
            fish.last_action_normalized = True
        else:
            fish.last_action_normalized = False

        if state_action < -0.3:
            requested_state = ActivityState.RESTING
        else:
            requested_state = ActivityState.ACTIVE

        # Update Physics System
        physics_output = None
        try:
            physics_output = self.base_env._update_physics(
                movement_action, requested_state, buoyancy_action
            )
            # Track propulsion force
            if hasattr(physics_output, 'propulsion_magnitude'):
                fish.last_propulsion_force = physics_output.propulsion_magnitude
            else:
                fish.last_propulsion_force = np.linalg.norm(movement_action) * 0.1
        except Exception as e:
            self._simple_physics_update(fish, movement_action, requested_state)

        # Check NPC predation
        npc_predation_result = self._check_npc_predation(fish)

        # Update Metabolism System
        metabolism_output = None
        try:
            buoyancy_energy = 0.0
            if physics_output and hasattr(physics_output, 'buoyancy_energy_consumed'):
                buoyancy_energy = physics_output.buoyancy_energy_consumed

            from systems import MetabolismInput
            metabolism_input = MetabolismInput(
                body_mass=self.base_env.growth_state.body_mass,
                action_magnitude=np.linalg.norm(movement_action),
                is_burst_swimming=self.base_env.physics_state.using_burst,
                velocity_magnitude=np.linalg.norm(self.base_env.physics_state.velocity),
                requested_activity_state=requested_state,
                current_step=self.current_step,
                buoyancy_energy_cost=buoyancy_energy,
                smr_individual_factor=getattr(fish, 'smr_individual_factor', 1.0)
            )
            metabolism_output = self.base_env.metabolism_system.update(
                self.base_env.metabolism_state,
                metabolism_input,
                curriculum_multiplier=1.0,
                growth_state=self.base_env.growth_state
            )
        except Exception as e:
            speed = np.linalg.norm(self.base_env.physics_state.velocity)
            if requested_state == ActivityState.ACTIVE:
                energy_cost = 0.015 + speed * 0.05
            else:
                energy_cost = 0.003
            self.base_env.metabolism_state.energy -= energy_cost

        # Update Growth System
        if metabolism_output and hasattr(metabolism_output, 'growth_energy') and metabolism_output.growth_energy > 0:
            try:
                self.base_env.growth_system.add_growth_energy(
                    self.base_env.growth_state, metabolism_output.growth_energy
                )
            except:
                pass

        try:
            growth_output = self.base_env.growth_system.process_growth(self.base_env.growth_state)
            if growth_output and growth_output.grew:
                if self.verbose:
                    print(
                        f"[GROWTH] AI#{fish.id} grew! Mass: {fish.body_mass:.1f}g -> {self.base_env.growth_state.body_mass:.1f}g")
        except:
            pass

        # Sync ALL state back to AIFishState
        fish.position = self.base_env.physics_state.position.copy()
        fish.velocity = self.base_env.physics_state.velocity.copy()

        # === CRITICAL FIX: Clamp velocity based on biological constraints ===
        # Bass swimming speed: normal ~1 BL/s, burst max ~3 BL/s
        # Reference: Webb (1975), Domenici & Blake (1997)
        body_length = self.base_env.growth_state.total_length  # in meters
        max_cruise_speed = body_length * 1.0  # 1.0 BL/s for sustained swimming
        max_burst_speed = body_length * 3.0  # 3.0 BL/s absolute maximum

        current_speed = np.linalg.norm(fish.velocity)

        # Determine speed limit based on activity state
        if requested_state == ActivityState.ACTIVE and self.base_env.physics_state.using_burst:
            speed_limit = max_burst_speed
        else:
            speed_limit = max_cruise_speed

        # Clamp velocity if exceeding limit
        if current_speed > speed_limit and current_speed > 0:
            fish.velocity = fish.velocity * (speed_limit / current_speed)
            fish.speed_clamped = True
            # Also update env state to maintain consistency
            self.base_env.physics_state.velocity = fish.velocity.copy()
        else:
            fish.speed_clamped = False

        fish.current_speed = np.linalg.norm(fish.velocity)

        gs = self.base_env.growth_state
        fish.body_mass = gs.body_mass
        fish.total_length = gs.total_length
        fish.growth_accumulation = gs.growth_accumulation
        fish.total_growth_energy = gs.total_growth_energy
        fish.growth_count = gs.growth_count

        ms = self.base_env.metabolism_state
        fish.energy = ms.energy
        fish.stomach_fullness = ms.stomach_fullness
        fish.stomach_content_mass = ms.stomach_content_mass
        fish.initial_meal_mass = ms.initial_meal_mass
        fish.digestion_buffer = ms.digestion_buffer
        fish.energy_from_digestion = ms.energy_from_digestion
        fish.activity_state = ms.activity_state
        fish.rest_duration_steps = ms.rest_duration_steps
        fish.current_metabolism_factor = ms.current_metabolism_factor
        fish.current_growth_bonus = ms.current_growth_bonus
        fish.fatigue = ms.fatigue
        fish.stress_level = ms.stress_level
        fish.is_digesting = ms.is_digesting
        fish.stomach_protein_fraction = ms.stomach_protein_fraction
        fish.stomach_lipid_fraction = ms.stomach_lipid_fraction
        fish.stomach_carbohydrate_fraction = ms.stomach_carbohydrate_fraction
        fish.stomach_adc_protein = ms.stomach_adc_protein
        fish.stomach_adc_lipid = ms.stomach_adc_lipid
        fish.stomach_adc_carbohydrate = ms.stomach_adc_carbohydrate
        fish.stomach_include_carbohydrate_energy = ms.stomach_include_carbohydrate_energy
        fish.state_switch_cooldown = ms.state_switch_cooldown
        fish.last_state_switch_step = ms.last_state_switch_step
        fish.total_rest_steps = ms.total_rest_steps
        fish.initial_body_mass = ms.initial_body_mass
        fish.lipid_reserve = ms.lipid_reserve
        fish.protein_reserve = ms.protein_reserve
        # Preserve per-fish death threshold (don't let shared system overwrite it)
        if ms.death_mass_loss_threshold > 0:
            fish.death_mass_loss_threshold = ms.death_mass_loss_threshold

        if hasattr(self.base_env.physics_state, 'buoyancy_state') and \
                self.base_env.physics_state.buoyancy_state is not None:
            fish.relative_density = self.base_env.physics_state.buoyancy_state.relative_density
            fish.net_buoyancy_force = self.base_env.physics_state.buoyancy_state.net_buoyancy_force
            fish.swimbladder_volume = self.base_env.physics_state.buoyancy_state.swimbladder_volume

        # Sync back physics fields to fish (prevents bleed-through on next fish's step)
        ps = self.base_env.physics_state
        if hasattr(ps, 'using_burst'):          fish.using_burst = ps.using_burst
        if hasattr(ps, 'heading'):              fish.heading = ps.heading.copy()
        if hasattr(ps, 'smoothed_action'):      fish.smoothed_action = ps.smoothed_action.copy()
        if hasattr(ps, 'smoothed_buoyancy_control'): fish.smoothed_buoyancy_control = ps.smoothed_buoyancy_control
        if hasattr(ps, 'current_turn_rate'):    fish.current_turn_rate = ps.current_turn_rate
        if hasattr(ps, 'inertia_initialized'):  fish.inertia_initialized = ps.inertia_initialized
        if hasattr(ps, 'pitch_angle'):          fish.pitch_angle = ps.pitch_angle
        if hasattr(ps, 'target_pitch_angle'):   fish.target_pitch_angle = ps.target_pitch_angle
        if hasattr(ps, 'current_pitch_rate'):   fish.current_pitch_rate = ps.current_pitch_rate
        if hasattr(ps, 'is_coasting'):          fish.is_coasting = ps.is_coasting
        if hasattr(ps, 'coast_steps_remaining'):fish.coast_steps_remaining = ps.coast_steps_remaining
        if hasattr(ps, 'current_reynolds'):     fish.current_reynolds = ps.current_reynolds
        if hasattr(ps, 'rest_transition_progress'):   fish.rest_transition_progress = ps.rest_transition_progress
        if hasattr(ps, 'active_transition_progress'): fish.active_transition_progress = ps.active_transition_progress
        if hasattr(ps, 'drift_direction'):      fish.drift_direction = ps.drift_direction.copy()
        if hasattr(ps, 'drift_update_counter'): fish.drift_update_counter = ps.drift_update_counter
        if hasattr(ps, 'previous_velocity'):    fish.previous_velocity = ps.previous_velocity.copy()
        if hasattr(ps, 'air_exposure_time'):    fish.air_exposure_time = ps.air_exposure_time
        if hasattr(ps, 'in_air'):               fish.in_air = ps.in_air
        if hasattr(ps, 'is_at_surface'):        fish.is_at_surface = ps.is_at_surface

        fish.position_history.append(fish.position.copy())

        # Update history for charts
        fish.update_history()

        if fish.energy <= 0:
            fish.is_alive = False
            fish.death_reason = 'starvation'
            if self.verbose:
                print(f"[DEATH] AI#{fish.id} starved (energy=0)")

    def _calculate_capture_radius(self, body_length: float) -> float:
        """Calculate capture radius based on body length"""
        ic = CONFIG.interaction
        if body_length <= ic.capture_radius_small_threshold:
            coef = ic.capture_radius_small
        elif body_length <= ic.capture_radius_medium_threshold:
            coef = ic.capture_radius_medium
        else:
            coef = ic.capture_radius_large
        return max(body_length * coef, ic.capture_radius_min)

    def _check_food_capture(self, fish: AIFishState):
        """Check and handle food capture"""
        if fish.activity_state != ActivityState.ACTIVE:
            return

        food_items = self.base_env.feeding_state.food_items
        capture_radius = self._calculate_capture_radius(fish.total_length)

        consumed_indices = []

        for i, food in enumerate(food_items):
            distance = np.linalg.norm(food.position - fish.position)
            if distance > capture_radius:
                continue

            # Use the metabolism system's add_food_to_stomach so that
            # initial_meal_mass, stomach nutrient fractions, and is_digesting
            # are all set correctly — these drive digestion → energy → growth.
            food_profile = {
                'protein_fraction':           food.protein_fraction,
                'lipid_fraction':             food.lipid_fraction,
                'carbohydrate_fraction':      food.carbohydrate_fraction,
                'adc_protein':                food.adc_protein,
                'adc_lipid':                  food.adc_lipid,
                'adc_carbohydrate':           food.adc_carbohydrate,
                'include_carbohydrate_energy': food.include_carbohydrate_energy,
            }
            result = self.base_env.metabolism_system.add_food_to_stomach(
                self.base_env.metabolism_state, food.mass, fish.body_mass, food_profile
            )
            if not result['success']:
                break  # stomach full — no point checking more pellets

            consumed_indices.append(i)
            fish.food_eaten += 1
            self.total_food_eaten += 1
            # Keep fish's own fields in sync with metabolism_state
            fish.stomach_content_mass = self.base_env.metabolism_state.stomach_content_mass
            fish.stomach_fullness     = self.base_env.metabolism_state.stomach_fullness
            fish.initial_meal_mass    = self.base_env.metabolism_state.initial_meal_mass

        for i in reversed(consumed_indices):
            self.base_env.feeding_state.food_items.pop(i)

    def _check_npc_predation(self, fish: AIFishState) -> Dict:
        """Check AI fish predation on NPCs"""
        result = {'success': 0, 'mass_gained': 0.0}

        if fish.activity_state != ActivityState.ACTIVE:
            return result

        capture_radius = self._calculate_capture_radius(fish.total_length)
        stomach_capacity = fish.body_mass * CONFIG.feeding.stomach_capacity_ratio
        available_space = stomach_capacity - fish.stomach_content_mass

        if available_space < 0.5:
            return result

        min_ratio = CONFIG.interaction.min_predation_size_ratio

        for npc in self.base_env.interaction_state.other_fish[:]:
            if not npc.is_alive:
                continue

            if npc.behavior_type.value in ['aggressive', 'surface']:
                continue

            size_ratio = fish.body_mass / npc.body_mass
            if size_ratio < min_ratio:
                continue

            distance = np.linalg.norm(npc.position - fish.position)
            if distance > capture_radius * 1.5:
                continue

            digestible_mass = npc.body_mass * 0.7
            if digestible_mass > available_space:
                continue

            if size_ratio >= 3.0:
                success_prob = 0.6
            elif size_ratio >= 2.0:
                success_prob = 0.4
            else:
                success_prob = 0.2

            if np.random.random() < success_prob:
                npc.is_alive = False

                fish.stomach_content_mass += digestible_mass
                fish.stomach_fullness = (fish.stomach_content_mass / stomach_capacity) * 100
                available_space -= digestible_mass

                fish.fish_eaten += 1
                result['success'] += 1
                result['mass_gained'] += digestible_mass

                self.base_env.metabolism_state.stomach_content_mass = fish.stomach_content_mass
                self.base_env.metabolism_state.stomach_fullness = fish.stomach_fullness

        self.base_env.interaction_state.other_fish = [
            f for f in self.base_env.interaction_state.other_fish if f.is_alive
        ]

        return result

    def _simple_physics_update(self, fish: AIFishState, movement: np.ndarray,
                               requested_state):
        """Fallback simple physics"""
        dt = CONFIG.environment.time_step

        if requested_state == ActivityState.ACTIVE:
            max_speed = 0.12
            target_velocity = movement * max_speed
            self.base_env.physics_state.velocity = \
                self.base_env.physics_state.velocity * 0.6 + target_velocity * 0.4
        else:
            self.base_env.physics_state.velocity *= 0.95

        self.base_env.physics_state.position += self.base_env.physics_state.velocity * dt
        self._enforce_boundaries_on_env()

    def _enforce_boundaries_on_env(self):
        """Enforce tank boundaries"""
        tank_radius = CONFIG.environment.tank_radius
        tank_depth = CONFIG.environment.tank_depth
        pos = self.base_env.physics_state.position
        vel = self.base_env.physics_state.velocity

        horizontal_dist = np.sqrt(pos[0] ** 2 + pos[2] ** 2)
        if horizontal_dist > tank_radius * 0.95:
            factor = tank_radius * 0.95 / horizontal_dist
            pos[0] *= factor
            pos[2] *= factor
            vel[0] *= -0.5
            vel[2] *= -0.5

        if pos[1] < -tank_depth + 0.05:
            pos[1] = -tank_depth + 0.05
            vel[1] = abs(vel[1]) * 0.5
        if pos[1] > -0.02:
            pos[1] = -0.02
            vel[1] = -abs(vel[1]) * 0.5

    def _check_ai_predation(self):
        """AI fish predation on each other"""
        min_ratio = CONFIG.interaction.min_predation_size_ratio
        capture_distance = 0.08

        for predator in self.ai_fish:
            if not predator.is_alive:
                continue

            stomach_capacity = predator.body_mass * CONFIG.feeding.stomach_capacity_ratio
            available_space = stomach_capacity - predator.stomach_content_mass

            if available_space < 0.5:
                continue

            for prey in self.ai_fish:
                if prey.id == predator.id or not prey.is_alive:
                    continue

                size_ratio = predator.body_mass / prey.body_mass
                if size_ratio < min_ratio:
                    continue

                distance = np.linalg.norm(predator.position - prey.position)
                if distance > capture_distance:
                    continue

                prey_digestible_mass = prey.body_mass * 0.7
                if prey_digestible_mass > available_space:
                    continue

                if size_ratio >= 3.0:
                    success_prob = 0.7
                elif size_ratio >= 2.0:
                    success_prob = 0.5
                else:
                    success_prob = 0.25

                if np.random.random() < success_prob:
                    prey.is_alive = False
                    prey.death_reason = 'eaten_by_ai'
                    prey.killed_by = predator.id

                    digestible_mass = prey.body_mass * 0.7
                    predator.stomach_content_mass += digestible_mass
                    predator.initial_meal_mass += digestible_mass
                    predator.stomach_fullness = (predator.stomach_content_mass / stomach_capacity) * 100
                    predator.ai_fish_eaten += 1
                    predator.fish_eaten += 1
                    available_space = stomach_capacity - predator.stomach_content_mass

                    self.ai_predation_events.append({
                        'step': self.current_step,
                        'predator_id': predator.id,
                        'prey_id': prey.id,
                        'prey_mass': prey.body_mass
                    })

                    print(f"[PREDATION] AI#{predator.id}({predator.body_mass:.1f}g) "
                          f"ate AI#{prey.id}({prey.body_mass:.1f}g)")

                    if available_space < 0.5:
                        break

    def step(self) -> Dict:
        """Execute one simulation step"""
        self.current_step += 1
        self.steps_since_feeding += 1

        if self.steps_since_feeding >= self.feeding_interval:
            spawned = self._spawn_food()
            self.steps_since_feeding = 0
            self.base_env.feeding_state.current_batch_settling = False
            if spawned > 0 and self.verbose:
                print(f"[FEED] Spawned {spawned} pellets at step {self.current_step}")

        self._update_food_movement()
        self._update_npc_fish()

        for fish in self.ai_fish:
            if not fish.is_alive:
                continue

            fish.steps_alive += 1
            self._sync_env_state(fish)
            original_fish = self._inject_other_ai_fish(fish.id)

            obs = self._get_observation_simple(fish)

            if self.model is not None:
                action, _ = self.model.predict(obs, deterministic=True)
            else:
                action = np.random.uniform(-1, 1, 5).astype(np.float32)

            self._execute_action(fish, action)

            if fish.activity_state == ActivityState.ACTIVE:
                self._check_food_capture(fish)

            self._restore_original_fish(original_fish)

        self._check_ai_predation()

        return {
            'step': self.current_step,
            'alive_count': sum(1 for f in self.ai_fish if f.is_alive),
            'food_count': len(self.base_env.feeding_state.food_items)
        }

    def _get_observation_simple(self, fish: AIFishState) -> np.ndarray:
        """Get observation without re-syncing"""
        perception_input = self.base_env._create_perception_input()
        self.base_env.perception_system.update(
            self.base_env.perception_state, perception_input
        )
        return self.base_env._get_observation()

    def get_statistics(self) -> Dict:
        alive_fish = [f for f in self.ai_fish if f.is_alive]
        dead_fish = [f for f in self.ai_fish if not f.is_alive]

        stats = {
            'total_fish': self.num_fish,
            'alive_count': len(alive_fish),
            'dead_count': len(dead_fish),
            'current_step': self.current_step,
            'food_remaining': len(self.base_env.feeding_state.food_items),
            'total_food_spawned': self.total_food_spawned,
            'total_food_eaten': self.total_food_eaten,
            'ai_predation_events': len(self.ai_predation_events),
        }

        if alive_fish:
            stats['avg_mass'] = np.mean([f.body_mass for f in alive_fish])
            stats['avg_energy'] = np.mean([f.energy for f in alive_fish])
            stats['total_mass'] = sum(f.body_mass for f in alive_fish)

        return stats


# ============================================================
# PyGame Academic Visualizer
# ============================================================

class AcademicVisualizer:
    """Academic-style visualization interface."""

    def __init__(self, model_path: Optional[str] = None, num_fish: int = 10,
                 initial_mass: float = 20.0, include_npc: bool = True,
                 window_size: tuple = (1400, 900)):

        self.window_size = window_size
        self.include_npc = include_npc

        # Load model
        self.model = None
        if model_path and HAS_SB3 and os.path.exists(model_path):
            self.model = PPO.load(model_path)
            print(f"[OK] Model loaded: {model_path}")
        else:
            print("[WARN] No model loaded, using random actions")

        # Create environment
        self.env = BassEnvironment({'verbose': 0})
        self.manager = MultiAIFishManager(
            self.env, self.model, num_fish, initial_mass, include_npc
        )

        # Tank parameters
        self.tank_radius = CONFIG.environment.tank_radius
        self.tank_depth = CONFIG.environment.tank_depth

        # === Layout Configuration ===
        # Left column: Views
        self.left_margin = 20
        self.view_width = 420

        # Top view (circular)
        self.top_view_center = (self.left_margin + self.view_width // 2, 220)
        self.top_view_radius = 180
        self.top_view_scale = self.top_view_radius / self.tank_radius

        # Side view (rectangular)
        self.side_view_left = self.left_margin
        self.side_view_top = 470
        self.side_view_width = self.view_width
        self.side_view_height = 180

        # Right panel
        self.panel_left = 480
        self.panel_width = 900

        # State
        self.paused = False
        self.running = True
        self.speed_multiplier = 1
        self.fullscreen = False

        # Fish selection
        self.selected_fish_id = 0
        self.fish_list_rects = []  # Store clickable areas

        # Initialize pygame
        pygame.init()
        pygame.font.init()

        self.screen = pygame.display.set_mode(window_size, pygame.RESIZABLE)
        pygame.display.set_caption("Bass Fish Simulation - Academic Visualization v2.0")

        self.clock = pygame.time.Clock()

        # Fonts
        self.font_title = pygame.font.SysFont('Arial', 22, bold=True)
        self.font_header = pygame.font.SysFont('Arial', 16, bold=True)
        self.font_normal = pygame.font.SysFont('Consolas', 14)
        self.font_small = pygame.font.SysFont('Consolas', 12)
        self.font_tiny = pygame.font.SysFont('Consolas', 10)

    def handle_events(self):
        """Handle keyboard/mouse events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                    print(f"[{'PAUSED' if self.paused else 'RESUMED'}]")
                elif event.key == pygame.K_r:
                    self.manager.reset()
                    self.selected_fish_id = 0
                    print("[RESET]")
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self.speed_multiplier = min(10, self.speed_multiplier + 1)
                    print(f"[SPEED] x{self.speed_multiplier}")
                elif event.key == pygame.K_MINUS:
                    self.speed_multiplier = max(1, self.speed_multiplier - 1)
                    print(f"[SPEED] x{self.speed_multiplier}")
                elif event.key == pygame.K_f:
                    self.fullscreen = not self.fullscreen
                    if self.fullscreen:
                        self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
                        self.window_size = self.screen.get_size()
                    else:
                        self.window_size = (1400, 900)
                        self.screen = pygame.display.set_mode(self.window_size, pygame.RESIZABLE)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    self._handle_click(event.pos)

            elif event.type == pygame.VIDEORESIZE:
                self.window_size = event.size
                self.screen = pygame.display.set_mode(self.window_size, pygame.RESIZABLE)

    def _handle_click(self, pos):
        """Handle mouse click for fish selection"""
        for fish_id, rect in self.fish_list_rects:
            if rect.collidepoint(pos):
                self.selected_fish_id = fish_id
                return

    def get_selected_fish(self) -> Optional[AIFishState]:
        """Get currently selected fish"""
        for fish in self.manager.ai_fish:
            if fish.id == self.selected_fish_id:
                return fish
        # Fallback to first alive fish
        for fish in self.manager.ai_fish:
            if fish.is_alive:
                self.selected_fish_id = fish.id
                return fish
        return None

    def draw_panel_background(self, x, y, width, height, title=None):
        """Draw a styled panel background"""
        # Main background
        rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.screen, AcademicColors.PANEL_BG, rect)
        pygame.draw.rect(self.screen, AcademicColors.PANEL_BORDER, rect, 1)

        # Title bar
        if title:
            title_rect = pygame.Rect(x, y, width, 28)
            pygame.draw.rect(self.screen, AcademicColors.PANEL_HEADER, title_rect)
            title_text = self.font_header.render(title, True, AcademicColors.TEXT_PRIMARY)
            self.screen.blit(title_text, (x + 10, y + 5))
            return y + 32
        return y + 5

    def draw_tank_top(self):
        """Draw tank top view with scale"""
        cx, cy = self.top_view_center
        radius = int(self.tank_radius * self.top_view_scale)

        # Panel background
        panel_y = self.draw_panel_background(
            self.left_margin, 40, self.view_width, 380, "Top View (X-Z Plane)"
        )

        # Water
        pygame.draw.circle(self.screen, AcademicColors.WATER_DEEP, (cx, cy), radius)
        pygame.draw.circle(self.screen, AcademicColors.TANK_BORDER, (cx, cy), radius, 2)

        # Scale markers (every 0.5m)
        for dist in [0.5, 1.0, 1.5]:
            if dist <= self.tank_radius:
                r = int(dist * self.top_view_scale)
                pygame.draw.circle(self.screen, AcademicColors.CHART_GRID, (cx, cy), r, 1)

                # Label
                label = self.font_tiny.render(f"{dist}m", True, AcademicColors.TEXT_DIM)
                self.screen.blit(label, (cx + r + 3, cy - 6))

        # Axis labels
        pygame.draw.line(self.screen, AcademicColors.TEXT_DIM,
                         (cx - radius - 10, cy), (cx + radius + 10, cy), 1)
        pygame.draw.line(self.screen, AcademicColors.TEXT_DIM,
                         (cx, cy - radius - 10), (cx, cy + radius + 10), 1)

        x_label = self.font_tiny.render("X", True, AcademicColors.TEXT_SECONDARY)
        z_label = self.font_tiny.render("Z", True, AcademicColors.TEXT_SECONDARY)
        self.screen.blit(x_label, (cx + radius + 15, cy - 6))
        self.screen.blit(z_label, (cx - 4, cy - radius - 20))

    def draw_tank_side(self):
        """Draw tank side view with depth scale"""
        left = self.side_view_left
        top = self.side_view_top
        width = self.side_view_width
        height = self.side_view_height

        # Panel background
        panel_y = self.draw_panel_background(
            left, top - 35, width, height + 45, "Side View (X-Y Depth Profile)"
        )

        # Water background
        rect = pygame.Rect(left, top, width, height)
        pygame.draw.rect(self.screen, AcademicColors.WATER_DEEP, rect)

        # Surface danger zone
        danger_ratio = 0.15 / self.tank_depth
        danger_height = int(height * danger_ratio)
        danger_rect = pygame.Rect(left, top, width, danger_height)
        pygame.draw.rect(self.screen, AcademicColors.WATER_DANGER, danger_rect)

        # Border and surface line
        pygame.draw.rect(self.screen, AcademicColors.TANK_BORDER, rect, 2)
        pygame.draw.line(self.screen, AcademicColors.WATER_SURFACE,
                         (left, top), (left + width, top), 3)

        # Depth scale (right side)
        scale_x = left + width + 5
        for depth_cm in [0, 20, 40, 60, 80]:
            depth_m = depth_cm / 100.0
            if depth_m <= self.tank_depth:
                y = int(top + depth_m / self.tank_depth * height)
                pygame.draw.line(self.screen, AcademicColors.TEXT_DIM,
                                 (left + width, y), (scale_x + 5, y), 1)
                label = self.font_tiny.render(f"{depth_cm}cm", True, AcademicColors.TEXT_DIM)
                self.screen.blit(label, (scale_x + 8, y - 6))

        # Horizontal scale (bottom)
        for x_pos in [-1.0, -0.5, 0, 0.5, 1.0]:
            if abs(x_pos) <= self.tank_radius:
                screen_x = int(left + (x_pos + self.tank_radius) / (2 * self.tank_radius) * width)
                pygame.draw.line(self.screen, AcademicColors.TEXT_DIM,
                                 (screen_x, top + height), (screen_x, top + height + 5), 1)
                label = self.font_tiny.render(f"{x_pos:.1f}m", True, AcademicColors.TEXT_DIM)
                self.screen.blit(label, (screen_x - 15, top + height + 8))

    def draw_food(self):
        """Draw food pellets in both views"""
        food_items = self.env.feeding_state.food_items

        for food in food_items:
            pos = food.position

            if food.is_settling:
                color = AcademicColors.FOOD_SETTLING
            elif food.food_type.value == 'floating':
                color = AcademicColors.FOOD_FLOATING
            else:
                color = AcademicColors.FOOD_SINKING

            # Top view
            screen_pos = world_to_screen_top(pos, self.top_view_center, self.top_view_scale)
            pygame.draw.circle(self.screen, color, screen_pos, 4)

            # Side view
            screen_pos = world_to_screen_side(
                pos, self.side_view_left, self.side_view_top,
                self.side_view_width, self.side_view_height,
                self.tank_radius, self.tank_depth
            )
            pygame.draw.circle(self.screen, color, screen_pos, 4)

    def draw_npc_fish(self):
        """Draw NPC fish"""
        if not self.include_npc:
            return

        fish_states = self.env.interaction_system.get_fish_states(self.env.interaction_state)

        for fish in fish_states:
            pos = fish['position']
            behavior = fish.get('behavior_type', 'passive')
            size_cat = fish.get('size_category', 'small')
            is_chasing = fish.get('is_chasing', False)

            if behavior == 'surface':
                color = AcademicColors.NPC_PREDATOR
                size = 12
            elif behavior == 'aggressive':
                color = AcademicColors.NPC_CHASING if is_chasing else AcademicColors.NPC_LARGE
                size = 10
            elif size_cat == 'small':
                color = AcademicColors.NPC_SMALL
                size = 4
            elif size_cat == 'medium':
                color = AcademicColors.NPC_MEDIUM
                size = 6
            else:
                color = AcademicColors.NPC_LARGE
                size = 8

            # Top view
            screen_pos = world_to_screen_top(pos, self.top_view_center, self.top_view_scale)
            pygame.draw.circle(self.screen, color, screen_pos, size)
            if is_chasing:
                pygame.draw.circle(self.screen, (255, 0, 0), screen_pos, size + 3, 2)

            # Side view
            screen_pos = world_to_screen_side(
                pos, self.side_view_left, self.side_view_top,
                self.side_view_width, self.side_view_height,
                self.tank_radius, self.tank_depth
            )
            pygame.draw.circle(self.screen, color, screen_pos, size)

    def draw_ai_fish(self):
        """Draw AI fish with selection highlight"""
        selected_fish = self.get_selected_fish()

        for fish in self.manager.ai_fish:
            if not fish.is_alive:
                continue

            pos = fish.position
            color = fish.color
            size = max(6, int(fish.body_mass / 3))
            is_selected = (fish.id == self.selected_fish_id)

            # Top view
            screen_pos = world_to_screen_top(pos, self.top_view_center, self.top_view_scale)

            # Selection glow
            if is_selected:
                pygame.draw.circle(self.screen, AcademicColors.SELECTED_GLOW, screen_pos, size + 6, 3)

            pygame.draw.circle(self.screen, color, screen_pos, size)
            pygame.draw.circle(self.screen, (0, 0, 0), screen_pos, size, 2)

            # Fish ID
            id_label = self.font_tiny.render(str(fish.id), True, (255, 255, 255))
            self.screen.blit(id_label, (screen_pos[0] - 4, screen_pos[1] - 5))

            # Velocity vector
            if np.linalg.norm(fish.velocity) > 0.01:
                vel_norm = fish.velocity / np.linalg.norm(fish.velocity)
                end_pos = (
                    int(screen_pos[0] + vel_norm[0] * 25),
                    int(screen_pos[1] - vel_norm[2] * 25)
                )
                pygame.draw.line(self.screen, color, screen_pos, end_pos, 2)
                # Arrow head
                pygame.draw.circle(self.screen, color, end_pos, 3)

            # Side view
            screen_pos = world_to_screen_side(
                pos, self.side_view_left, self.side_view_top,
                self.side_view_width, self.side_view_height,
                self.tank_radius, self.tank_depth
            )

            if is_selected:
                pygame.draw.circle(self.screen, AcademicColors.SELECTED_GLOW, screen_pos, size + 6, 3)

            pygame.draw.circle(self.screen, color, screen_pos, size)
            pygame.draw.circle(self.screen, (0, 0, 0), screen_pos, size, 2)
            id_label = self.font_tiny.render(str(fish.id), True, (255, 255, 255))
            self.screen.blit(id_label, (screen_pos[0] - 4, screen_pos[1] - 5))

    def draw_fish_list_panel(self):
        """Draw fish selection list panel"""
        panel_x = self.panel_left
        panel_y = 40
        panel_width = 200
        panel_height = 420

        content_y = self.draw_panel_background(panel_x, panel_y, panel_width, panel_height, "Fish Selection")

        self.fish_list_rects = []

        y = content_y + 5
        for fish in self.manager.ai_fish:
            is_selected = (fish.id == self.selected_fish_id)

            # Background for selection
            item_rect = pygame.Rect(panel_x + 5, y, panel_width - 10, 36)
            if is_selected:
                pygame.draw.rect(self.screen, AcademicColors.PANEL_HEADER, item_rect)
                pygame.draw.rect(self.screen, AcademicColors.SELECTED_GLOW, item_rect, 2)

            self.fish_list_rects.append((fish.id, item_rect))

            # Color indicator
            pygame.draw.circle(self.screen, fish.color, (panel_x + 20, y + 18), 8)

            # Fish info
            if fish.is_alive:
                status_color = AcademicColors.TEXT_PRIMARY
                status_text = f"#{fish.id}: {fish.body_mass:.1f}g"
                energy_text = f"E:{fish.energy:.0f}%"
                energy_color = AcademicColors.TEXT_SUCCESS if fish.energy > 50 else AcademicColors.TEXT_WARNING
            else:
                status_color = AcademicColors.TEXT_DIM
                status_text = f"#{fish.id}: DEAD"
                energy_text = fish.death_reason or ""
                energy_color = AcademicColors.TEXT_DANGER

            text = self.font_small.render(status_text, True, status_color)
            self.screen.blit(text, (panel_x + 35, y + 3))

            energy_label = self.font_tiny.render(energy_text, True, energy_color)
            self.screen.blit(energy_label, (panel_x + 35, y + 20))

            y += 40

    def draw_selected_fish_details(self):
        """Draw detailed information for selected fish"""
        fish = self.get_selected_fish()
        if not fish:
            return

        panel_x = self.panel_left + 220
        panel_y = 40
        panel_width = 260
        panel_height = 420

        content_y = self.draw_panel_background(panel_x, panel_y, panel_width, panel_height,
                                               f"Fish #{fish.id} Details")

        y = content_y + 5
        line_height = 18

        # === Basic Info ===
        section_label = self.font_small.render("─ Basic State ─", True, AcademicColors.TEXT_ACCENT)
        self.screen.blit(section_label, (panel_x + 10, y))
        y += line_height + 5

        basic_info = [
            (f"Mass: {fish.body_mass:.2f} g", AcademicColors.TEXT_PRIMARY),
            (f"Length: {fish.total_length * 100:.1f} cm", AcademicColors.TEXT_PRIMARY),
            (f"Energy: {fish.energy:.1f} %",
             AcademicColors.TEXT_SUCCESS if fish.energy > 50 else AcademicColors.TEXT_WARNING),
            (f"Stomach: {fish.stomach_fullness:.1f} %", AcademicColors.TEXT_PRIMARY),
            (f"Activity: {'🔵 ACTIVE' if fish.activity_state == ActivityState.ACTIVE else '😴 REST'}",
             AcademicColors.TEXT_ACCENT if fish.activity_state == ActivityState.ACTIVE else AcademicColors.TEXT_DIM),
        ]

        for text, color in basic_info:
            label = self.font_small.render(text, True, color)
            self.screen.blit(label, (panel_x + 15, y))
            y += line_height

        y += 10

        # === Motion Info ===
        section_label = self.font_small.render("─ Motion & Force ─", True, AcademicColors.TEXT_ACCENT)
        self.screen.blit(section_label, (panel_x + 10, y))
        y += line_height + 5

        speed_mps = fish.current_speed
        speed_bls = speed_mps / fish.total_length if fish.total_length > 0 else 0

        # Speed with biological reference
        speed_color = AcademicColors.CHART_SPEED
        speed_note = ""
        if speed_bls > 3.0:
            speed_color = AcademicColors.TEXT_DANGER
            speed_note = " ⚠️OVER"
        elif speed_bls > 1.5:
            speed_color = AcademicColors.TEXT_WARNING
            speed_note = " (burst)"

        action_mag = np.linalg.norm(fish.last_action[:3])
        action_color = AcademicColors.TEXT_PRIMARY
        action_note = ""
        if action_mag > 1.0:
            action_color = AcademicColors.TEXT_WARNING
            action_note = " →normalized"

        motion_info = [
            (f"Speed: {speed_mps:.3f} m/s ({speed_bls:.2f} BL/s){speed_note}", speed_color),
            (f"Action: [{fish.last_action[0]:.2f}, {fish.last_action[1]:.2f}, {fish.last_action[2]:.2f}]",
             AcademicColors.TEXT_PRIMARY),
            (f"Action Mag: {action_mag:.3f}{action_note}", action_color),
            (f"Propulsion: {fish.last_propulsion_force:.4f} N", AcademicColors.TEXT_PRIMARY),
            (f"State Ctrl: {fish.last_action[3]:.2f} ({'REST' if fish.last_action[3] < -0.3 else 'ACTIVE'})",
             AcademicColors.TEXT_SECONDARY),
        ]

        for text, color in motion_info:
            label = self.font_small.render(text, True, color)
            self.screen.blit(label, (panel_x + 15, y))
            y += line_height

        # Show warnings if corrections applied
        if hasattr(fish, 'last_action_normalized') and fish.last_action_normalized:
            warn_text = self.font_tiny.render("⚠ Action normalized (mag>1)", True, AcademicColors.TEXT_WARNING)
            self.screen.blit(warn_text, (panel_x + 15, y))
            y += line_height - 4

        if hasattr(fish, 'speed_clamped') and fish.speed_clamped:
            warn_text = self.font_tiny.render("⚠ Speed clamped (bio limit)", True, AcademicColors.TEXT_WARNING)
            self.screen.blit(warn_text, (panel_x + 15, y))
            y += line_height - 4

        y += 10

        # === Buoyancy Info ===
        section_label = self.font_small.render("─ Buoyancy Control ─", True, AcademicColors.TEXT_ACCENT)
        self.screen.blit(section_label, (panel_x + 10, y))
        y += line_height + 5

        buoyancy_ctrl = fish.buoyancy_control
        mode = int(getattr(fish, 'buoyancy_mode', 0))
        ctrl_direction = "↑ UP" if mode < 0 else ("↓ DOWN" if mode > 0 else "● HOLD")

        # Calculate swimbladder percentage
        mass_kg = fish.body_mass / 1000.0
        tissue_density = float(getattr(CONFIG.buoyancy, 'fish_tissue_density', 1055.0))
        tissue_volume = mass_kg / tissue_density
        total_volume = tissue_volume + fish.swimbladder_volume
        swimbladder_pct = (fish.swimbladder_volume / total_volume * 100) if total_volume > 0 else 6.0

        current_depth = max(0, -fish.position[1])

        buoyancy_info = [
            (f"Buoyancy Mode: {mode:+d} {ctrl_direction}",
             AcademicColors.TEXT_WARNING if mode != 0 else AcademicColors.TEXT_PRIMARY),
            (f"Raw Signal: {getattr(fish, 'raw_buoyancy_control', buoyancy_ctrl):+.2f}",
             AcademicColors.TEXT_PRIMARY),
            (f"Rel. Density: {fish.relative_density:.4f}",
             AcademicColors.TEXT_SUCCESS if 0.99 < fish.relative_density < 1.01 else AcademicColors.TEXT_WARNING),
            (f"Net Buoyancy: {fish.net_buoyancy_force:.5f} N", AcademicColors.TEXT_PRIMARY),
            (f"Swimbladder: {swimbladder_pct:.2f} %", AcademicColors.TEXT_PRIMARY),
            (f"Depth: {current_depth * 100:.1f} cm", AcademicColors.TEXT_PRIMARY),
        ]

        for text, color in buoyancy_info:
            label = self.font_small.render(text, True, color)
            self.screen.blit(label, (panel_x + 15, y))
            y += line_height

        y += 10

        # === Statistics ===
        section_label = self.font_small.render("─ Statistics ─", True, AcademicColors.TEXT_ACCENT)
        self.screen.blit(section_label, (panel_x + 10, y))
        y += line_height + 5

        stats_info = [
            (f"Food Eaten: {fish.food_eaten}", AcademicColors.TEXT_PRIMARY),
            (f"Fish Eaten: {fish.fish_eaten}", AcademicColors.TEXT_PRIMARY),
            (f"Steps Alive: {fish.steps_alive}", AcademicColors.TEXT_PRIMARY),
            (f"Growth Count: {fish.growth_count}", AcademicColors.TEXT_SUCCESS),
        ]

        for text, color in stats_info:
            label = self.font_small.render(text, True, color)
            self.screen.blit(label, (panel_x + 15, y))
            y += line_height

    def draw_realtime_charts(self):
        """Draw real-time data charts for selected fish"""
        fish = self.get_selected_fish()
        if not fish:
            return

        panel_x = self.panel_left + 500
        panel_y = 40
        panel_width = 380
        panel_height = 420

        content_y = self.draw_panel_background(panel_x, panel_y, panel_width, panel_height,
                                               "Real-time Data Charts")

        chart_width = panel_width - 60
        chart_height = 90
        chart_left = panel_x + 45

        # Energy Chart
        self._draw_mini_chart(
            chart_left, content_y + 10, chart_width, chart_height,
            list(fish.energy_history), "Energy (%)",
            AcademicColors.CHART_ENERGY, 0, 100
        )

        # Mass Chart
        if fish.mass_history:
            min_mass = max(0, min(fish.mass_history) - 2)
            max_mass = max(fish.mass_history) + 2
        else:
            min_mass, max_mass = 15, 25

        self._draw_mini_chart(
            chart_left, content_y + chart_height + 30, chart_width, chart_height,
            list(fish.mass_history), "Mass (g)",
            AcademicColors.CHART_MASS, min_mass, max_mass
        )

        # Speed Chart - with biological reference lines
        fish_length = fish.total_length
        max_bio_speed = fish_length * 3.5  # Show up to 3.5 BL/s for reference
        self._draw_speed_chart(
            chart_left, content_y + 2 * (chart_height + 30), chart_width, chart_height,
            list(fish.speed_history), "Speed (m/s)",
            AcademicColors.CHART_SPEED, 0, max(0.15, max_bio_speed),
            fish_length  # Pass fish length for reference lines
        )

    def _draw_mini_chart(self, x, y, width, height, data, label, color, y_min, y_max):
        """Draw a mini line chart"""
        # Background
        chart_rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.screen, AcademicColors.CHART_BG, chart_rect)
        pygame.draw.rect(self.screen, AcademicColors.PANEL_BORDER, chart_rect, 1)

        # Label
        label_text = self.font_tiny.render(label, True, color)
        self.screen.blit(label_text, (x - 40, y + height // 2 - 6))

        # Grid lines
        for i in range(5):
            gy = y + int(i * height / 4)
            pygame.draw.line(self.screen, AcademicColors.CHART_GRID,
                             (x, gy), (x + width, gy), 1)

        # Y-axis labels
        y_labels = [y_max, (y_max + y_min) / 2, y_min]
        for i, val in enumerate(y_labels):
            ly = y + int(i * height / 2)
            val_text = self.font_tiny.render(f"{val:.1f}", True, AcademicColors.TEXT_DIM)
            self.screen.blit(val_text, (x + width + 5, ly - 5))

        # Plot data
        if len(data) > 1:
            points = []
            for i, val in enumerate(data):
                px = x + int(i / len(data) * width)
                # Clamp value
                val = max(y_min, min(y_max, val))
                py = y + height - int((val - y_min) / (y_max - y_min) * height)
                points.append((px, py))

            if len(points) >= 2:
                pygame.draw.lines(self.screen, color, False, points, 2)

                # Current value marker
                if points:
                    last_point = points[-1]
                    pygame.draw.circle(self.screen, color, last_point, 4)

    def _draw_speed_chart(self, x, y, width, height, data, label, color, y_min, y_max, fish_length):
        """Draw speed chart with biological reference lines"""
        # Background
        chart_rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.screen, AcademicColors.CHART_BG, chart_rect)
        pygame.draw.rect(self.screen, AcademicColors.PANEL_BORDER, chart_rect, 1)

        # Label
        label_text = self.font_tiny.render(label, True, color)
        self.screen.blit(label_text, (x - 40, y + height // 2 - 6))

        # Grid lines
        for i in range(5):
            gy = y + int(i * height / 4)
            pygame.draw.line(self.screen, AcademicColors.CHART_GRID,
                             (x, gy), (x + width, gy), 1)

        # === Biological reference lines ===
        # Normal cruising speed: 1 BL/s
        cruise_speed = fish_length * 1.0
        if y_min <= cruise_speed <= y_max:
            cruise_y = y + height - int((cruise_speed - y_min) / (y_max - y_min) * height)
            pygame.draw.line(self.screen, AcademicColors.TEXT_SUCCESS,
                             (x, cruise_y), (x + width, cruise_y), 1)
            cruise_label = self.font_tiny.render("1BL/s", True, AcademicColors.TEXT_SUCCESS)
            self.screen.blit(cruise_label, (x + 2, cruise_y - 12))

        # Burst speed limit: 3 BL/s
        burst_speed = fish_length * 3.0
        if y_min <= burst_speed <= y_max:
            burst_y = y + height - int((burst_speed - y_min) / (y_max - y_min) * height)
            pygame.draw.line(self.screen, AcademicColors.TEXT_DANGER,
                             (x, burst_y), (x + width, burst_y), 1)
            burst_label = self.font_tiny.render("3BL/s MAX", True, AcademicColors.TEXT_DANGER)
            self.screen.blit(burst_label, (x + 2, burst_y - 12))

        # Y-axis labels
        y_labels = [y_max, (y_max + y_min) / 2, y_min]
        for i, val in enumerate(y_labels):
            ly = y + int(i * height / 2)
            val_text = self.font_tiny.render(f"{val:.2f}", True, AcademicColors.TEXT_DIM)
            self.screen.blit(val_text, (x + width + 5, ly - 5))

        # Plot data
        if len(data) > 1:
            points = []
            for i, val in enumerate(data):
                px = x + int(i / len(data) * width)
                val_clamped = max(y_min, min(y_max, val))
                py = y + height - int((val_clamped - y_min) / (y_max - y_min) * height)
                points.append((px, py))

            if len(points) >= 2:
                pygame.draw.lines(self.screen, color, False, points, 2)

                # Current value marker
                if points:
                    last_point = points[-1]
                    # Color based on speed
                    marker_color = color
                    if data and len(data) > 0:
                        current_val = data[-1]
                        if current_val > burst_speed:
                            marker_color = AcademicColors.TEXT_DANGER
                        elif current_val > fish_length * 1.5:
                            marker_color = AcademicColors.TEXT_WARNING
                    pygame.draw.circle(self.screen, marker_color, last_point, 5)

    def draw_simulation_info(self):
        """Draw simulation time and global info"""
        panel_x = self.panel_left
        panel_y = 470
        panel_width = 880
        panel_height = 200

        content_y = self.draw_panel_background(panel_x, panel_y, panel_width, panel_height,
                                               "Simulation Status")

        stats = self.manager.get_statistics()

        y = content_y + 10
        col1_x = panel_x + 20
        col2_x = panel_x + 250
        col3_x = panel_x + 480
        col4_x = panel_x + 700

        # Simulation time
        sim_time = format_time(stats['current_step'])
        time_label = self.font_normal.render(f"Simulation Time: {sim_time}", True, AcademicColors.TEXT_ACCENT)
        self.screen.blit(time_label, (col1_x, y))

        step_label = self.font_small.render(f"Step: {stats['current_step']}", True, AcademicColors.TEXT_SECONDARY)
        self.screen.blit(step_label, (col2_x, y))

        speed_label = self.font_small.render(f"Speed: x{self.speed_multiplier}", True, AcademicColors.TEXT_SECONDARY)
        self.screen.blit(speed_label, (col3_x, y))

        status_text = "⏸ PAUSED" if self.paused else "▶ RUNNING"
        status_color = AcademicColors.TEXT_WARNING if self.paused else AcademicColors.TEXT_SUCCESS
        status_label = self.font_normal.render(status_text, True, status_color)
        self.screen.blit(status_label, (col4_x, y))

        y += 35

        # Statistics row 1
        info_items = [
            (f"Alive: {stats['alive_count']}/{stats['total_fish']}", AcademicColors.TEXT_PRIMARY),
            (f"Food: {stats['food_remaining']} pellets", AcademicColors.FOOD_FLOATING),
            (f"Eaten: {stats['total_food_eaten']}", AcademicColors.TEXT_SUCCESS),
            (f"AI Predation: {stats['ai_predation_events']}", AcademicColors.TEXT_DANGER),
        ]

        for i, (text, color) in enumerate(info_items):
            x_pos = [col1_x, col2_x, col3_x, col4_x][i]
            label = self.font_small.render(text, True, color)
            self.screen.blit(label, (x_pos, y))

        y += 25

        # Statistics row 2
        if stats['alive_count'] > 0:
            info_items2 = [
                (f"Avg Mass: {stats.get('avg_mass', 0):.2f} g", AcademicColors.CHART_MASS),
                (f"Avg Energy: {stats.get('avg_energy', 0):.1f} %", AcademicColors.CHART_ENERGY),
                (f"Total Mass: {stats.get('total_mass', 0):.1f} g", AcademicColors.TEXT_PRIMARY),
            ]

            for i, (text, color) in enumerate(info_items2):
                x_pos = [col1_x, col2_x, col3_x][i]
                label = self.font_small.render(text, True, color)
                self.screen.blit(label, (x_pos, y))

        y += 35

        # Controls help
        controls = "Controls: SPACE=Pause  R=Reset  +/-=Speed  Click=Select Fish  ESC=Quit"
        help_label = self.font_tiny.render(controls, True, AcademicColors.TEXT_DIM)
        self.screen.blit(help_label, (col1_x, y))

    def draw_legend(self):
        """Draw legend"""
        panel_x = self.left_margin
        panel_y = 690
        panel_width = self.view_width
        panel_height = 90

        content_y = self.draw_panel_background(panel_x, panel_y, panel_width, panel_height, "Legend")

        y = content_y + 5
        col1_x = panel_x + 15
        col2_x = panel_x + 130
        col3_x = panel_x + 260

        legend_items = [
            [(AcademicColors.NPC_SMALL, "Small NPC"), (AcademicColors.NPC_LARGE, "Large NPC")],
            [(AcademicColors.FOOD_FLOATING, "Floating"), (AcademicColors.FOOD_SINKING, "Sinking")],
            [(AcademicColors.SELECTED_GLOW, "Selected"), (AcademicColors.NPC_PREDATOR, "Predator")],
        ]

        for col_idx, column in enumerate(legend_items):
            x = [col1_x, col2_x, col3_x][col_idx]
            for i, (color, name) in enumerate(column):
                cy = y + i * 22
                pygame.draw.circle(self.screen, color, (x + 8, cy + 6), 6)
                text = self.font_tiny.render(name, True, AcademicColors.TEXT_SECONDARY)
                self.screen.blit(text, (x + 22, cy))

    def draw_title(self):
        """Draw main title"""
        alive = sum(1 for f in self.manager.ai_fish if f.is_alive)
        title = f"Bass Fish Behavior Simulation | Academic Visualization v2.0"
        title_text = self.font_title.render(title, True, AcademicColors.TEXT_PRIMARY)
        self.screen.blit(title_text, (self.left_margin, 10))

    def run(self, target_fps: int = 60):
        """Main loop"""
        print("[START] Academic visualization")
        print("        Click fish list to select | SPACE=Pause | R=Reset | ESC=Quit")

        self.manager.reset()

        while self.running:
            self.handle_events()

            # Update simulation
            if not self.paused:
                for _ in range(self.speed_multiplier):
                    self.manager.step()

            # Draw
            self.screen.fill(AcademicColors.BACKGROUND)

            # Left column: Tank views
            self.draw_tank_top()
            self.draw_tank_side()
            self.draw_food()
            self.draw_npc_fish()
            self.draw_ai_fish()
            self.draw_legend()

            # Right column: Panels
            self.draw_fish_list_panel()
            self.draw_selected_fish_details()
            self.draw_realtime_charts()
            self.draw_simulation_info()

            # Title
            self.draw_title()

            pygame.display.flip()
            self.clock.tick(target_fps)

        pygame.quit()

        # Print final stats
        stats = self.manager.get_statistics()
        print("\n" + "=" * 60)
        print("Final Statistics")
        print("=" * 60)
        for key, value in stats.items():
            print(f"  {key}: {value}")

        return stats


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Multi-AI Fish Academic Visualization v2.0')
    parser.add_argument('--model', type=str, help='Model file path')
    parser.add_argument('--num_fish', type=int, default=10, help='Number of AI fish')
    parser.add_argument('--mass', type=float, default=20.0, help='Initial mass (g)')
    parser.add_argument('--fps', type=int, default=60, help='Target FPS')
    parser.add_argument('--no_npc', action='store_true', help='Hide NPC fish')

    args = parser.parse_args()

    print("=" * 70)
    print("  Bass Fish Behavior Simulation - Academic Visualization v2.0")
    print("=" * 70)
    print(f"  AI Fish Count : {args.num_fish}")
    print(f"  Initial Mass  : {args.mass}g each")
    print(f"  Model         : {args.model or 'Random Actions'}")
    print(f"  Target FPS    : {args.fps}")
    print("=" * 70)

    visualizer = AcademicVisualizer(
        model_path=args.model,
        num_fish=args.num_fish,
        initial_mass=args.mass,
        include_npc=not args.no_npc
    )

    visualizer.run(target_fps=args.fps)


if __name__ == "__main__":
    main()
