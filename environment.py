#!/usr/bin/env python3
"""
 -  v5.2.1 (Fix)
======================================================

v5.2.1 Fix
1. Action vector normalization (prevent resultant force overflow)
2. Biological speed limits (cruise<=1.5BL/s, burst<=3BL/s)

Action Space
- action[0:3] =  (x, y, z)
- action[3] =  (=, =)
- action[4] =  (=/, =/)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque
from typing import Dict, Any, Optional, Tuple
from enum import Enum

from config import CONFIG, CURRICULUM_STAGES, get_curriculum_stage

try:
    from systems.metabolism import ActivityState
except ImportError:
    class ActivityState(Enum):
        ACTIVE = "active"
        RESTING = "resting"

from systems import (
    MetabolismInput,
    create_metabolism_system, create_metabolism_state,
    create_growth_system, create_growth_state,
    PhysicsInput,
    create_physics_system, create_physics_state,
    PerceptionInput,
    create_perception_system, create_perception_state,
    FeedingInput,
    create_feeding_system, create_feeding_state,
    InteractionInput,
    create_interaction_system, create_interaction_state,
)
from systems.tank_geometry import create_random_tank, create_default_tank
from systems.obstacles import generate_obstacles, create_empty_obstacle_field, RockObstacle
from dataclasses import dataclass
from utils.biological_formulas import calculate_fish_energy_density

@dataclass
class EmptyFeedingOutput:
    """Empty feeding output (used during rest state)"""
    food_consumed: int = 0
    energy_potential: float = 0.0
    floating_consumed: int = 0
    sinking_consumed: int = 0
    mass_consumed: float = 0.0

# Singleton instance to avoid repeated creation
_EMPTY_FEEDING_OUTPUT = EmptyFeedingOutput()


class BassEnvironment(gym.Env):
    """
     - v5.2.1 Fix


    1. 5Action Space++
    2. Normalize
    3.
    """

    metadata = {'render_modes': ['human']}

    def __init__(self, config: Optional[Dict] = None):
        super().__init__()

        # ========== 1.  ==========
        self.runtime_config = {
            'verbose': 0,
            'log_frequency': 100,
            'debug_energy': False,
            'debug_curriculum': False,
            'debug_growth': False,
            'debug_predation': False,
            'debug_rest_state': False,
            'debug_buoyancy': False,
            'debug_speed': False,  # new
            'debug_obstacles': False,  # new
            'force_default_tank': False,
            'disable_obstacles': False,
            'obstacle_density_multiplier': 1.0,
            'training_phase': 'course4',
        }
        if config:
            self.runtime_config.update(config)
        # Support passing 'course' as alias for 'training_phase'
        if 'course' in self.runtime_config:
            self.runtime_config['training_phase'] = self.runtime_config.pop('course')
        self.set_training_phase(self.runtime_config.get('training_phase', 'course4'))

        # ========== 2. Action Space5==========
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(5,), dtype=np.float32
        )

        # ========== 3. Observation Space ==========
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(CONFIG.observation.total_dim,),
            dtype=np.float32
        )

        # ========== 4.  ==========
        self._init_systems()

        # ========== 5. Curriculum Learning Configuration ==========
        self.curriculum_config = self._get_initial_curriculum()

        # ========== 6. Episode ==========
        self.current_step = 0
        self._temp_cycle_day_index = -1
        self._temp_cycle_day_bias = 0.0
        self._current_water_temp = self._compute_current_water_temp()
        self.episode_count = 0
        self.episode_rewards = []
        self.survival_time_history = deque(maxlen=100)

        self.all_episode_lengths = deque(maxlen=100)
        self.all_episode_rewards = deque(maxlen=100)
        self.all_episode_food_counts = deque(maxlen=100)

        self.water_current = np.zeros(3, dtype=np.float32)

        # ========== new ==========
        self._action_normalized_count = 0
        self._speed_clamped_count = 0
        self._forced_active_hunger_steps = 0
        self._steps_since_last_intake = 0

        if self.runtime_config.get('verbose', 0) >= 1:
            print(f"✅ v5.2.1 Fix")
            print(f"   Action Space: {self.action_space.shape}")
            print(f"   Observation Space: {self.observation_space.shape}")
            print(f"   Speed Limiting: ≤1.5BL/s, ≤3.0BL/s")

    def _get_initial_curriculum(self) -> Dict[str, Any]:
        stage = CURRICULUM_STAGES[0]
        return {
            'stage': stage.stage,
            'name': stage.name,
            'capture_multiplier': stage.capture_multiplier,
            'predation_multiplier': stage.predation_multiplier,
            'energy_cost_multiplier': stage.energy_cost_multiplier,
            'food_amount_multiplier': stage.food_amount_multiplier
        }

    def _compute_current_water_temp(self) -> float:
        env = CONFIG.environment
        base_temp = float(env.water_temp)
        if not bool(getattr(env, 'enable_temp_daily_cycle', False)):
            return base_temp

        sim_seconds = self.current_step * env.time_step * env.time_acceleration
        day_index = int(sim_seconds // 86400)
        hour_of_day = (sim_seconds % 86400) / 3600.0

        if day_index != self._temp_cycle_day_index:
            self._temp_cycle_day_index = day_index
            noise_sd = max(0.0, float(getattr(env, 'temp_daily_noise_sd', 0.0)))
            self._temp_cycle_day_bias = (
                float(np.random.normal(0.0, noise_sd)) if noise_sd > 0.0 else 0.0
            )

        amplitude = max(0.0, float(getattr(env, 'temp_daily_amplitude', 0.0)))
        peak_hour = float(getattr(env, 'temp_peak_hour', 14.0))
        phase = 2.0 * np.pi * ((hour_of_day - peak_hour) / 24.0)
        temp = base_temp + self._temp_cycle_day_bias + amplitude * np.sin(phase)

        clamp_range = max(0.0, float(getattr(env, 'temp_daily_clamp_range', 0.0)))
        if clamp_range > 0.0:
            temp = np.clip(temp, base_temp - clamp_range, base_temp + clamp_range)

        return float(temp)

    def _init_systems(self):
        self.metabolism_system = create_metabolism_system()
        self.growth_system = create_growth_system()
        self.physics_system = create_physics_system()
        self.perception_system = create_perception_system()
        self.feeding_system = create_feeding_system()
        self.interaction_system = create_interaction_system()

        if self.runtime_config.get('debug_energy', False):
            self.metabolism_system.set_debug(True)
        if self.runtime_config.get('debug_growth', False):
            self.growth_system.set_debug(True)
        if self.runtime_config.get('debug_predation', False):
            self.interaction_system.set_debug(True)
        if self.runtime_config.get('debug_rest_state', False):
            self.perception_system.set_debug(True)
        if self.runtime_config.get('debug_buoyancy', False):
            self.physics_system.set_debug(True)

    def _augment_obstacles_for_phase(self, spawn_position: np.ndarray):
        """/"""
        multiplier = float(self.runtime_config.get('obstacle_density_multiplier', 1.0))
        if multiplier <= 1.0:
            return
        if self.obstacle_field is None:
            return

        oc = CONFIG.obstacles
        base_count = max(1, self.obstacle_field.count)
        target_count = int(np.ceil(base_count * multiplier))
        to_add = max(0, target_count - self.obstacle_field.count)
        if to_add <= 0:
            return

        max_attempts = to_add * 80
        added = 0
        for _ in range(max_attempts):
            if added >= to_add:
                break

            radius = float(np.random.uniform(oc.rock_radius_min, oc.rock_radius_max))
            margin = radius + oc.min_distance_from_wall
            pos = self.tank_geometry.random_interior_point(margin=margin)

            if spawn_position is not None:
                if np.linalg.norm(pos - spawn_position) < oc.spawn_exclusion_radius + radius:
                    continue

            too_close = False
            for existing in self.obstacle_field.obstacles:
                dist = np.linalg.norm(pos - existing.center)
                if dist < existing.radius + radius + oc.min_distance_between:
                    too_close = True
                    break
            if too_close:
                continue

            if not self.tank_geometry.contains_point_xz(pos[0], pos[2]):
                continue
            if pos[1] + radius > -0.05:
                continue
            if pos[1] - radius < -self.tank_geometry.depth + 0.03:
                continue

            self.obstacle_field.add_obstacle(RockObstacle(pos, radius))
            added += 1

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            np.random.seed(seed)

        # ==========  ==========
        if self.runtime_config.get('force_default_tank', False):
            self.tank_geometry = create_default_tank()
        else:
            self.tank_geometry = create_random_tank()
        # spawn_position  None
        self._pending_spawn_position = None
        self.current_step = 0
        self._temp_cycle_day_index = -1
        self._temp_cycle_day_bias = 0.0
        self._current_water_temp = self._compute_current_water_temp()
        self.episode_count += 1
        self.episode_rewards = []
        self._killed_by_predator = False

        # Randomise initial body mass across the full grow-out range so the
        # policy learns buoyancy / locomotion control at every life stage.
        # Largemouth bass commercial grow-out: ~0.5 g fry → ~500 g market size.
        # Log-uniform sampling gives equal representation to each order of
        # magnitude (fry, fingerling, juvenile, sub-adult, adult).
        import math as _math
        _mass_lo, _mass_hi = 0.5, 500.0   # g  (fry → market size)
        initial_mass = _math.exp(
            np.random.uniform(_math.log(_mass_lo), _math.log(_mass_hi))
        )

        # ── Per-episode dynamic scaling ────────────────────────────────────
        # All parameters that should vary with body size are computed here
        # once per episode and patched onto the relevant config/system objects
        # via object.__setattr__ (frozen dataclasses).
        from utils.biological_formulas import mass_to_length as _m2l
        _agent_length = _m2l(initial_mass)   # metres

        # Pellet size: ~1–2 % BW per pellet, clipped to hardware pellet range.
        # No.0 crumble (0.5g fry) → No.4 pellet (500g adult).
        _pellet_lo = float(np.clip(initial_mass * 0.008, 0.001, 0.10))
        _pellet_hi = float(np.clip(initial_mass * 0.020, _pellet_lo * 1.5, 3.0))
        _fc = self.feeding_system.c
        object.__setattr__(_fc, 'pellet_mass_min', _pellet_lo)
        object.__setattr__(_fc, 'pellet_mass_max', _pellet_hi)

        # Perception ranges scale with body length but have a hard floor so
        # even 0.5 g fry retain meaningful sensory capability.
        # Biological basis (largemouth bass):
        #   - Vision: clear-water range ~1-2 m even for fry; adults up to 3 m.
        #     Formula: max(1.0 m floor, body_length * 8), capped at CONFIG max.
        #   - Lateral line: mechanoreceptor, ~1-4 BL; floor 0.3 m (fry can
        #     still detect nearby disturbances).
        #   - Food detection (olfaction): similar to vision range.
        _vision   = float(np.clip(_agent_length * 8.0,  1.0, CONFIG.perception.vision_range))
        _lat_line = float(np.clip(_agent_length * 3.0,  0.3, CONFIG.perception.lateral_line_range))
        _food_det = float(np.clip(_agent_length * 8.0,  1.0, CONFIG.perception.food_detection_range))
        _pc = CONFIG.perception
        object.__setattr__(_pc, 'vision_range',         _vision)
        object.__setattr__(_pc, 'lateral_line_range',   _lat_line)
        object.__setattr__(_pc, 'food_detection_range', _food_det)
        # Keep a reference for reward scaling later in step()
        self._agent_length_m = _agent_length
        # ──────────────────────────────────────────────────────────────────

        # Generate safe spawn point within current tank
        spawn_position = self.tank_geometry.random_interior_point(margin=0.3)
        # Y-axis limited to reasonable depth
        spawn_position[1] = np.clip(spawn_position[1],
                                    -self.tank_geometry.depth + 0.2,
                                    -0.15)

        self.physics_state = create_physics_state(position=spawn_position)
        self._prev_agent_position = self.physics_state.position.copy()

        phase = self.runtime_config.get('training_phase', 'course4')
        if self.runtime_config.get('disable_obstacles', False):
            self.obstacle_field = create_empty_obstacle_field()
            self._course2_scene = None
        elif phase in {'course2', 'phase2'}:
            # Course2 4:3:3
            roll = np.random.random()
            if roll < 0.40:
                scene = 'maze'       # corridor
                layout = 'corridor'
            elif roll < 0.70:
                scene = 'reef'       # reef
                layout = 'reef'
            else:
                scene = 'open'       # random
                layout = 'random'
            self._course2_scene = scene
            self.obstacle_field = generate_obstacles(
                self.tank_geometry, CONFIG, spawn_position=spawn_position,
                layout_hint=layout
            )
            self._augment_obstacles_for_phase(spawn_position)
        else:
            self._course2_scene = None
            self.obstacle_field = generate_obstacles(
                self.tank_geometry, CONFIG, spawn_position=spawn_position
            )
            self._augment_obstacles_for_phase(spawn_position)

        # Initialize inertia system (after creating physics_state)
        self.physics_system.initialize_inertia(self.physics_state)

        self.metabolism_state = create_metabolism_state(
            energy=CONFIG.agent_init.initial_energy,
            stomach_fullness=CONFIG.agent_init.initial_stomach_fullness
        )

        self.growth_state = create_growth_state(initial_mass)
        self.perception_state = create_perception_state()
        self.feeding_state = create_feeding_state()
        self.interaction_state = create_interaction_state()

        phase = self.runtime_config.get('training_phase', 'course4')
        eco_kwargs = {}
        if phase in {'course1', 'phase1'}:
            # Course 1: pure feeding ability (no conspecifics/predators)
            eco_kwargs = {
                'small_fish_count': 0,
                'medium_fish_count': 0,
                'aggressive_count_range': (0, 0),
                'enable_surface_predator': False,
            }
        elif phase in {'course2', 'phase2'}:
            # Course 2: maze foraging (navigation focus, no predators)
            eco_kwargs = {
                'small_fish_count': 0,
                'medium_fish_count': 0,
                'aggressive_count_range': (0, 0),
                'enable_surface_predator': False,
            }
        elif phase in {'course3', 'phase3'}:
            # Course 3: foraging/predation under threat
            eco_kwargs = {
                'small_fish_count': 8,
                'medium_fish_count': 2,
                'aggressive_count_range': (1, 2),
                'enable_surface_predator': True,
            }
        elif phase in {'course4', 'phase4'}:
            # Course 4: full mixed environment with threats
            eco_kwargs = {
                'small_fish_count': 8,
                'medium_fish_count': 3,
                'aggressive_count_range': (1, 2),
                'enable_surface_predator': True,
            }

        self.interaction_system.initialize_ecosystem(
            self.interaction_state,
            tank_geometry=self.tank_geometry,
            obstacle_field=self.obstacle_field,
            agent_mass=initial_mass,
            **eco_kwargs
        )
        feeding_input = self._create_feeding_input()
        # _spawn_env_foodreset
        self.feeding_state.food_items = []

        self.water_current = self.physics_system.initialize_water_current()

        # ========== Tracking variables ==========
        self._last_action = np.zeros(5, dtype=np.float32)
        self._last_action_delta = 0.0
        self._total_food_eaten = 0
        self._total_fish_eaten = 0
        self._energy_from_pellets = 0.0
        self._energy_from_fish = 0.0
        self._mass_from_pellets = 0.0
        self._mass_from_fish = 0.0
        self._death_reason = None
        self._last_food_detect_timer = 400  # feeding reward

        self._floating_eaten = 0
        self._sinking_eaten = 0
        self._ambient_eaten = 0
        self._surface_env_eaten = 0
        self._benthic_eaten = 0
        self._attached_eaten = 0
        self._times_chased = 0
        self._surface_entries = 0
        self._last_in_surface = False
        self._total_damage = 0.0
        self._escape_count = 0

        self._initial_mass = self.growth_state.body_mass
        self._last_step_mass = self.growth_state.body_mass
        self._growth_event_count = 0
        self._last_food_distance = float('inf')
        self._last_threat_distance = float('inf')
        self._last_prey_distance = float('inf')
        self._first_intake_step = None
        self._forced_active_steps = 0
        self._forced_active_hunger_steps = 0
        self._steps_since_last_intake = 0

        self._total_rest_steps = 0
        self._rest_growth_bonus_accumulated = 0.0
        self._state_switches = 0
        self._consecutive_rest_steps = 0
        self._max_consecutive_rest = 0
        self._rest_during_danger = 0

        self._buoyancy_energy_total = 0.0
        self._buoyancy_adjustments = 0
        self._neutral_buoyancy_steps = 0
        self._last_relative_density = 1.0

        # ========== new ==========
        self._action_normalized_count = 0
        self._speed_clamped_count = 0

        self._spinning_steps = 0

        self._recent_positions = []   # 20
        self._circle_penalty_accum = 0.0

        #  12×12×6 25cm/13cm/
        # episode
        self._visited_cells: set = set()
        self._exploration_grid = {
            'nx': 12, 'ny': 6, 'nz': 12,
            'x_min': -CONFIG.environment.tank_radius,
            'x_max':  CONFIG.environment.tank_radius,
            'y_min': -CONFIG.environment.tank_depth,
            'y_max':  0.0,
            'z_min': -CONFIG.environment.tank_radius,
            'z_max':  CONFIG.environment.tank_radius,
        }

        # new
        self._pitch_extreme_count = 0  #
        self._efficient_dive_count = 0  #

        # Obstacle collisionnew
        self._obstacle_collisions = 0
        self._consecutive_collisions = 0

        if self.runtime_config['verbose'] >= 1:
            print(f"\n=== Episode {self.episode_count}  ===")
            # new
            extents = self.tank_geometry.get_extents()
            print(f": {extents}")
            print(f": {self.obstacle_field.count}")
            print(f"initial body mass: {self._initial_mass:.1f}g")
            print(f"initial energy: {self.metabolism_state.energy:.1f}%")

        return self._get_observation(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self.current_step += 1
        self._current_water_temp = self._compute_current_water_temp()
        self._steps_since_last_intake += 1
        self._last_action_delta = float(np.linalg.norm(action[:3] - self._last_action[:3]))
        self._last_action = action.copy()

        # ========== 5 ==========
        movement_action = action[:3].copy()
        state_action = action[3]
        buoyancy_action = action[4] if len(action) > 4 else 0.0

        # action[0]= [-1,1]==0.5BL/s
        # action[1]=pitch [-1,1]action[2]= [-1,1]
        # physics.py

        if state_action < 0:
            requested_state = ActivityState.RESTING
        else:
            requested_state = ActivityState.ACTIVE

        self._prev_agent_position = self.physics_state.position.copy()

        # 1.
        perception_input = self._create_perception_input()
        self.perception_system.update(self.perception_state, perception_input)

        #  detected
        #  detected_timerfeeding reward
        nearest_detected_timer = 400
        if self.perception_state.observed_food:
            observed_positions = [
                f['position'] for f in self.perception_state.observed_food
            ]
            for food_item in self.feeding_state.food_items:
                if food_item.detected_timeout > 0:
                    for obs_pos in observed_positions:
                        if np.linalg.norm(food_item.position - obs_pos) < 0.01:
                            if not food_item.detected:
                                food_item.detected = True
                            nearest_detected_timer = min(
                                nearest_detected_timer, food_item.detected_timer)
                            break
        self._last_food_detect_timer = nearest_detected_timer

        # ""
        delayed_threat = self.perception_system.get_delayed_threat_distance(self.perception_state)
        immediate_threat = self.perception_state.nearest_threat_distance
        threat_distance = min(immediate_threat, delayed_threat)
        proactive_wake_dist = getattr(
            CONFIG.rest_state,
            'proactive_wake_threat_distance',
            getattr(CONFIG.rest_state, 'emergency_wake_threat_distance', 0.22)
        )
        if requested_state == ActivityState.RESTING and threat_distance < proactive_wake_dist:
            requested_state = ActivityState.ACTIVE
            self._forced_active_steps += 1

        # /""
        if requested_state == ActivityState.RESTING:
            hunger_threshold = float(getattr(CONFIG.rest_state, 'forced_active_hunger_threshold', 14.0))
            # 60≈6s""
            no_food_steps = 60
            if self.metabolism_state.stomach_fullness < hunger_threshold and self._steps_since_last_intake >= no_food_steps:
                requested_state = ActivityState.ACTIVE
                self._forced_active_steps += 1
                self._forced_active_hunger_steps += 1

        # 2. Normalize
        physics_output = self._update_physics(movement_action, requested_state, buoyancy_action)
        if physics_output.collision_occurred:
            self._obstacle_collisions += 1

        # ========== new ==========
        self._enforce_biological_speed_limit(requested_state)

        # 3.
        if self.metabolism_state.activity_state == ActivityState.ACTIVE:
            feeding_output = self._update_feeding()
        else:
            feeding_output = self._create_empty_feeding_output()
            feeding_input = self._create_feeding_input()
            self.feeding_system._spawn_env_food(self.feeding_state, feeding_input)

        # 4.
        interaction_output = self._update_interaction()

        # 4.5
        self._track_behavior(interaction_output, physics_output)

        # 5.
        metabolism_output = self._update_metabolism(movement_action, physics_output, requested_state)

        # 6.
        self._process_food_intake(feeding_output, interaction_output)

        # 7.
        damage_reward_penalty = self._process_damage(interaction_output)

        # 8.
        if metabolism_output.growth_energy > 0:
            self.growth_system.add_growth_energy(self.growth_state, metabolism_output.growth_energy)
        growth_output = self.growth_system.process_growth(self.growth_state)

        # 9.
        self._update_rest_tracking(metabolism_output, interaction_output)

        if hasattr(self.physics_state, 'pitch_angle'):
            if abs(self.physics_state.pitch_angle) > 35:
                self._pitch_extreme_count += 1
            elif abs(self.physics_state.pitch_angle) > 5:
                forward_speed = np.linalg.norm(self.physics_state.velocity[:2])
                if forward_speed > 0.01:
                    self._efficient_dive_count += 1

        # 10.
        current_mass = self.growth_state.body_mass
        mass_change_this_step = current_mass - self._last_step_mass

        reward = self._calculate_reward_v52(
            feeding_output, interaction_output, metabolism_output,
            physics_output, growth_output,
            mass_change_this_step=mass_change_this_step,
            action_delta=self._last_action_delta
        )

        reward += damage_reward_penalty

        self._last_step_mass = current_mass
        self.episode_rewards.append(reward)

        # 11.
        terminated = self._check_termination()
        truncated = self.current_step >= CONFIG.environment.max_episode_steps

        # 12.
        if terminated or truncated:
            terminal_reward = self._calculate_terminal_reward_v52()
            reward += terminal_reward

            if self.episode_rewards:
                self.episode_rewards[-1] = reward

            if self.runtime_config['verbose'] >= 1:
                mass_change = self.growth_state.body_mass - self._initial_mass
                rest_ratio = self._total_rest_steps / max(1, self.current_step) * 100
                norm_ratio = self._action_normalized_count / max(1, self.current_step) * 100
                print(f"📊 : {self._initial_mass:.1f}→{self.growth_state.body_mass:.1f}g, "
                      f"{rest_ratio:.1f}%, Normalize{norm_ratio:.1f}%")

            self._on_episode_end(terminated)

        if self.runtime_config['verbose'] >= 2:
            if self.current_step % self.runtime_config['log_frequency'] == 0:
                self._log_status_enhanced(reward, metabolism_output, physics_output)

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _enforce_biological_speed_limit(self, requested_state: ActivityState):
        """Speed Limiting - """
        from utils.biological_formulas import calculate_burst_speed
        body_length = self.growth_state.total_length
        current_speed = np.linalg.norm(self.physics_state.velocity)

        if requested_state == ActivityState.RESTING:
            max_speed = body_length * 0.5  #  0.5 BL/s
        else:
            #  physics.py  burst_speed
            max_speed = calculate_burst_speed(body_length)

        if current_speed > max_speed and current_speed > 0:
            self.physics_state.velocity = self.physics_state.velocity * (max_speed / current_speed)
            self._speed_clamped_count += 1

    def _process_food_intake(self, feeding_output, interaction_output):
        """"""
        intake_happened = False

        if feeding_output.food_consumed > 0:
            intake_happened = True
            self._total_food_eaten += feeding_output.food_consumed
            # _last_food_detect_timer

            if hasattr(feeding_output, 'floating_consumed'):
                self._floating_eaten += feeding_output.floating_consumed
                self._sinking_eaten += feeding_output.sinking_consumed
                self._ambient_eaten += feeding_output.ambient_consumed
                self._surface_env_eaten += feeding_output.surface_env_consumed
                self._benthic_eaten += feeding_output.benthic_consumed
                self._attached_eaten += feeding_output.attached_consumed

            if hasattr(feeding_output, 'mass_consumed') and feeding_output.mass_consumed > 0:
                food_mass = feeding_output.mass_consumed
            else:
                food_mass = feeding_output.food_consumed * CONFIG.feeding.pellet_mass_min

            self._mass_from_pellets += food_mass
            pellet_profile = {
                'protein_fraction': CONFIG.feeding.diet_protein_fraction,
                'lipid_fraction': CONFIG.feeding.diet_lipid_fraction,
                'carbohydrate_fraction': CONFIG.feeding.diet_carbohydrate_fraction,
                'adc_protein': CONFIG.feeding.diet_adc_protein,
                'adc_lipid': CONFIG.feeding.diet_adc_lipid,
                'adc_carbohydrate': CONFIG.feeding.diet_adc_carbohydrate,
                'include_carbohydrate_energy': CONFIG.feeding.diet_include_carbohydrate_energy,
            }

            self.metabolism_system.add_food_to_stomach(
                self.metabolism_state, food_mass, self.growth_state.body_mass,
                food_profile=pellet_profile
            )

            self._energy_from_pellets += getattr(
                feeding_output,
                'energy_potential',
                food_mass * CONFIG.metabolism.pellet_energy_density
            )

        if interaction_output.predation_success > 0:
            intake_happened = True
            fish_mass = interaction_output.mass_gained
            digestible_mass = fish_mass * 0.70
            fish_food_profile = {
                'protein_fraction': CONFIG.metabolism.protein_fraction,
                'lipid_fraction': CONFIG.metabolism.lipid_fraction,
                'carbohydrate_fraction': 0.0,
                'adc_protein': 0.95,
                'adc_lipid': 0.95,
                'adc_carbohydrate': 0.0,
                'include_carbohydrate_energy': False,
            }

            result = self.metabolism_system.add_food_to_stomach(
                self.metabolism_state,
                digestible_mass,
                self.growth_state.body_mass,
                food_profile=fish_food_profile
            )

            if result['success']:
                self._total_fish_eaten += interaction_output.predation_success
                self._mass_from_fish += fish_mass
                prey_energy_density = calculate_fish_energy_density(fish_mass)
                self._energy_from_fish += digestible_mass * prey_energy_density
            else:
                available = self.growth_state.body_mass * CONFIG.feeding.stomach_capacity_ratio
                available -= (self.metabolism_state.stomach_fullness / 100) * available

                if available > 0.01:
                    partial = min(digestible_mass, available * 0.95)
                    self.metabolism_system.add_food_to_stomach(
                        self.metabolism_state, partial, self.growth_state.body_mass,
                        food_profile=fish_food_profile
                    )
                    actual_fish_mass = partial / 0.70
                    self._total_fish_eaten += interaction_output.predation_success
                    self._mass_from_fish += actual_fish_mass
                    prey_energy_density = calculate_fish_energy_density(actual_fish_mass)
                    self._energy_from_fish += partial * prey_energy_density

        if intake_happened:
            if self._first_intake_step is None:
                self._first_intake_step = self.current_step
            self._steps_since_last_intake = 0

    def _create_empty_feeding_output(self):
        return _EMPTY_FEEDING_OUTPUT

    def _update_rest_tracking(self, metabolism_output, interaction_output):
        if metabolism_output.state_switched:
            self._state_switches += 1

        if self.metabolism_state.activity_state == ActivityState.RESTING:
            self._total_rest_steps += 1
            self._consecutive_rest_steps += 1
            self._max_consecutive_rest = max(self._max_consecutive_rest, self._consecutive_rest_steps)
            self._rest_growth_bonus_accumulated += (metabolism_output.growth_bonus - 1.0)

            if self.perception_state.nearest_threat_distance < 0.3:
                self._rest_during_danger += 1
        else:
            self._consecutive_rest_steps = 0

    def _track_behavior(self, interaction_output, physics_output):
        """"""
        if hasattr(interaction_output, 'being_chased') and interaction_output.being_chased:
            self._times_chased += 1

        in_surface_zone = self.physics_state.position[1] > -0.2
        if in_surface_zone and not self._last_in_surface:
            self._surface_entries += 1
        self._last_in_surface = in_surface_zone

        if hasattr(interaction_output, 'escape_success'):
            self._escape_count += interaction_output.escape_success

        if hasattr(physics_output, 'buoyancy_energy_consumed'):
            self._buoyancy_energy_total += physics_output.buoyancy_energy_consumed

        if len(self._last_action) > 4 and abs(self._last_action[4]) > 0.1:
            self._buoyancy_adjustments += 1

        if hasattr(physics_output, 'buoyancy_output') and physics_output.buoyancy_output is not None:
            if physics_output.buoyancy_output.is_neutral:
                self._neutral_buoyancy_steps += 1

        if hasattr(physics_output, 'relative_density'):
            self._last_relative_density = physics_output.relative_density
        elif hasattr(physics_output, 'buoyancy_output') and physics_output.buoyancy_output is not None:
            self._last_relative_density = physics_output.buoyancy_output.relative_density

    def _update_physics(self, action: np.ndarray, requested_state: ActivityState,
                        buoyancy_action: float = 0.0):
        """"""
        physics_input = PhysicsInput(
            action=action,
            body_mass=self.growth_state.body_mass,
            total_length=self.growth_state.total_length,
            net_gravity_in_water=self.growth_state.net_gravity_in_water,
            gravity_in_air=self.growth_state.gravity_in_air,
            water_current=self.water_current,
            nearest_threat_distance=self.perception_state.nearest_threat_distance,
            activity_state=self.metabolism_state.activity_state,
            buoyancy_control=buoyancy_action,
            water_temp=self._current_water_temp,
            use_buoyancy_system=True,
            nearest_food_distance=self.perception_state.nearest_food_distance,
            nearest_prey_distance=self._get_nearest_prey_distance() or float('inf'),
            tank_geometry=self.tank_geometry,
            obstacle_field=self.obstacle_field
        )
        return self.physics_system.update(self.physics_state, physics_input)

    def _update_feeding(self):
        feeding_input = self._create_feeding_input()
        result = self.feeding_system.update(
            self.feeding_state, feeding_input,
            curriculum_multiplier=self.curriculum_config['food_amount_multiplier'],
            capture_radius_multiplier=self.curriculum_config.get('capture_multiplier', 1.0),
            env_food_only=True
        )
        # activity_state
        self.feeding_system._spawn_env_food(self.feeding_state, feeding_input)
        return result

    def _update_interaction(self):
        interaction_input = InteractionInput(
            agent_position=self.physics_state.position,
            agent_velocity=self.physics_state.velocity,
            agent_mass=self.growth_state.body_mass,
            agent_length=self.growth_state.total_length,
            agent_heading=self.physics_state.heading if hasattr(self.physics_state, 'heading') else None,
            agent_is_burst=getattr(self.physics_state, 'using_burst', False),
            agent_fatigue=self.metabolism_state.fatigue,
            agent_stress=self.metabolism_state.stress_level,
            tank_geometry=self.tank_geometry,
            obstacle_field=self.obstacle_field
        )
        return self.interaction_system.update(
            self.interaction_state, interaction_input,
            curriculum_config=self.curriculum_config
        )

    def _update_metabolism(self, action: np.ndarray, physics_output, requested_state: ActivityState):
        """"""
        buoyancy_energy = 0.0
        if hasattr(physics_output, 'buoyancy_energy_consumed'):
            buoyancy_energy = physics_output.buoyancy_energy_consumed

        metabolism_input = MetabolismInput(
            body_mass=self.growth_state.body_mass,
            action_magnitude=np.linalg.norm(action),
            is_burst_swimming=self.physics_state.using_burst,
            water_temp=self._current_water_temp,
            velocity_magnitude=np.linalg.norm(self.physics_state.velocity),
            requested_activity_state=requested_state,
            current_step=self.current_step,
            nearest_threat_distance=self.perception_state.nearest_threat_distance,
            buoyancy_energy_cost=buoyancy_energy,
            turn_angle_deg=getattr(physics_output, 'turn_angle_deg', 0.0)
        )
        return self.metabolism_system.update(
            self.metabolism_state, metabolism_input,
            curriculum_multiplier=self.curriculum_config['energy_cost_multiplier'],
            growth_state=self.growth_state
        )

    def _create_feeding_input(self) -> FeedingInput:
        return FeedingInput(
            agent_position=self.physics_state.position,
            agent_mass=self.growth_state.body_mass,
            agent_length=self.growth_state.total_length,
            stomach_fullness=self.metabolism_state.stomach_fullness,
            agent_prev_position=self._prev_agent_position,
            agent_heading=self.physics_state.heading if hasattr(self.physics_state, 'heading') else None,
            agent_pitch_angle=self.physics_state.pitch_angle if hasattr(self.physics_state, 'pitch_angle') else 0.0,
            tank_geometry=self.tank_geometry,
            obstacle_field=self.obstacle_field
        )

    def _create_perception_input(self) -> PerceptionInput:
        return PerceptionInput(
            agent_position=self.physics_state.position,
            agent_velocity=self.physics_state.velocity,
            agent_heading=self.physics_state.heading if hasattr(self.physics_state, 'heading') else None,
            agent_mass=self.growth_state.body_mass,
            fish_states=self.interaction_system.get_fish_states(self.interaction_state),
            food_positions=self.feeding_system.get_food_positions(self.feeding_state),
            activity_state=self.metabolism_state.activity_state,
            tank_geometry=self.tank_geometry,
            obstacle_field=self.obstacle_field
        )

    def _get_observation(self) -> np.ndarray:
        """"""
        obs = []
        env = CONFIG.environment

        #  (3) -
        pos = self.physics_state.position
        x_scale, z_scale = self.tank_geometry.normalization_scales
        obs.extend([
            pos[0] / x_scale,
            pos[1] / self.tank_geometry.depth,
            pos[2] / z_scale
        ])

        #  (3)
        obs.extend(self.physics_state.velocity.tolist())

        #  (2)
        obs.extend([
            self.growth_state.body_mass / 100.0,
            self.metabolism_state.energy / 100.0
        ])

        #  (3)
        if hasattr(self.physics_state, 'heading'):
            obs.extend(self.physics_state.heading.tolist())
        else:
            obs.extend([1.0, 0.0, 0.0])

        #  (2) - new
        if hasattr(self.physics_state, 'pitch_angle'):
            pitch_rad = np.radians(self.physics_state.pitch_angle)
            obs.extend([
                np.sin(pitch_rad),  # sin
                np.cos(pitch_rad)  # cos
            ])
        else:
            obs.extend([0.0, 1.0])  #

        #  (11×3)
        fish_obs = self.perception_system.get_normalized_fish_observation(
            self.perception_state, self.physics_state.position,
            agent_velocity=self.physics_state.velocity
        )
        obs.extend(fish_obs.tolist())

        #  (8×3)
        food_obs = self.perception_system.get_normalized_food_observation(
            self.perception_state, self.physics_state.position,
            agent_velocity=self.physics_state.velocity
        )
        obs.extend(food_obs.tolist())

        #  (4) -
        if self.tank_geometry.shape_name == 'circular':
            water_force = self.physics_system.calculate_circular_current(
                self.physics_state.position,
                env.water_current_strength
            )
            current_strength = np.linalg.norm(water_force)
        else:
            current_strength = np.linalg.norm(self.water_current)

        obs.extend([
            self.water_current[0],
            self.water_current[2],
            current_strength,
            self.current_step / env.max_episode_steps
        ])

        #  (8)
        obs.extend(self.perception_state.obstacle_distances.tolist())

        #  (3)
        obs.extend(self.perception_state.boundary_vector.tolist())

        #  (2)
        surface_obs = self.perception_system.get_surface_observation(self.perception_state)
        obs.extend(surface_obs.tolist())

        #  (4)
        is_resting = 1.0 if self.metabolism_state.activity_state == ActivityState.RESTING else 0.0
        rest_duration_ratio = min(self.metabolism_state.rest_duration_steps / 100.0, 1.0)
        metabolism_factor = self.metabolism_state.current_metabolism_factor
        growth_bonus = self.metabolism_state.current_growth_bonus / 1.5

        obs.extend([
            is_resting,
            rest_duration_ratio,
            metabolism_factor,
            growth_bonus
        ])

        #  (4)
        if hasattr(self.physics_state, 'buoyancy_state') and self.physics_state.buoyancy_state is not None:
            buoyancy_state = self.physics_state.buoyancy_state

            relative_density_offset = buoyancy_state.relative_density - 1.0
            net_buoyancy_normalized = np.clip(buoyancy_state.net_buoyancy_force * 100, -1, 1)

            mass_kg = self.growth_state.body_mass / 1000.0
            tissue_volume = mass_kg / 1070.0
            total_volume = tissue_volume + buoyancy_state.swimbladder_volume
            swimbladder_ratio = buoyancy_state.swimbladder_volume / total_volume if total_volume > 0 else 0.06
            swimbladder_ratio_offset = swimbladder_ratio - 0.06

            current_depth = max(0, -self.physics_state.position[1])
            depth_normalized = current_depth / self.tank_geometry.depth

            obs.extend([
                relative_density_offset,
                net_buoyancy_normalized,
                swimbladder_ratio_offset,
                depth_normalized
            ])
        else:
            current_depth = max(0, -self.physics_state.position[1])
            depth_normalized = current_depth / self.tank_geometry.depth
            obs.extend([0.0, 0.0, 0.0, depth_normalized])

        return np.array(obs, dtype=np.float32)

    def _get_nearest_prey_distance(self) -> float:
        """"""
        if not hasattr(self, 'interaction_state') or self.interaction_state is None:
            return float('inf')

        agent_pos = self.physics_state.position
        agent_mass = self.growth_state.body_mass
        min_ratio = CONFIG.interaction.min_predation_size_ratio
        fish_states = getattr(self.interaction_state, 'other_fish', None)
        if not fish_states:
            return float('inf')

        min_distance = None
        obstacle_field = getattr(self, 'obstacle_field', None)

        for fish in fish_states:
            if not getattr(fish, 'is_alive', True):
                continue

            fish_mass = getattr(fish, 'body_mass', 5.0)
            if fish_mass <= 0:
                continue
            size_ratio = agent_mass / fish_mass

            if size_ratio >= min_ratio:
                fish_pos = getattr(fish, 'position', None)
                if fish_pos is not None:
                    if obstacle_field is not None:
                        if not obstacle_field.check_line_of_sight(agent_pos, fish_pos):
                            continue
                    distance = np.linalg.norm(agent_pos - fish_pos)
                    if min_distance is None or distance < min_distance:
                        min_distance = distance

        return min_distance if min_distance is not None else float('inf')

    def _get_nearest_predator_distance(self) -> float:
        """"""
        if not hasattr(self, 'interaction_state') or self.interaction_state is None:
            return float('inf')

        agent_pos = self.physics_state.position
        agent_mass = self.growth_state.body_mass
        fish_states = getattr(self.interaction_state, 'other_fish', None)
        if not fish_states:
            return float('inf')

        min_distance = None
        for fish in fish_states:
            if not getattr(fish, 'is_alive', True):
                continue
            fish_mass = getattr(fish, 'body_mass', 5.0)
            if fish_mass <= 0:
                continue

            # 1.5
            if fish_mass > agent_mass * 1.5:
                fish_pos = getattr(fish, 'position', None)
                if fish_pos is not None:
                    distance = np.linalg.norm(agent_pos - fish_pos)
                    if min_distance is None or distance < min_distance:
                        min_distance = distance

        return min_distance if min_distance is not None else float('inf')

    def _calculate_reward_v52(self, feeding_out, interaction_out, metabolism_out,
                              physics_out, growth_out=None,
                              mass_change_this_step: float = 0.0,
                              action_delta: float = 0.0) -> float:
        """
         v5.3
        ============================================================

        -  =  ±0.01~0.05
        - / =  ≈  5~20
        -  =  ≈ 0.01~0.05/
        -      =  2
        -  [-1, +3]/
        ============================================================
        """
        reward = 0.0
        current_mass = self.growth_state.body_mass

        # Body length (used for food-near threshold below)
        length = max(self.growth_state.total_length, 1e-3)
        # Fixed predator/threat distance thresholds (original values)
        _pred_warn_dist   = 0.5
        _pred_danger_dist = 0.10
        _threat_dist      = 0.10

        # ----------------------------------------------------------------
        # 0.
        # ----------------------------------------------------------------
        reward += 0.001

        # ----------------------------------------------------------------
        # 1.  —
        #     ±1e-4g 20g1≈30s
        #    scale=300 → 0.1%+0.060.01~0.10/
        # ----------------------------------------------------------------
        if current_mass > 0 and mass_change_this_step != 0:
            mass_change_ratio = mass_change_this_step / current_mass
            if mass_change_this_step > 0:
                # <50g1.2
                scale = 300.0 * (1.2 if current_mass < 50 else 1.0)
                reward += mass_change_ratio * scale
            else:
                reward += mass_change_ratio * 200.0  #

        # ----------------------------------------------------------------
        # 2.
        # ----------------------------------------------------------------
        if growth_out is not None and growth_out.grew and growth_out.mass_change > 0:
            self._growth_event_count += 1
            growth_ratio = growth_out.mass_change / max(current_mass, 1e-6)
            # 0.1~0.5/
            growth_reward = min(growth_ratio * 150.0, 0.5)
            if self.metabolism_state.activity_state == ActivityState.RESTING:
                growth_reward *= 1.2  # ""
            reward += growth_reward

        # ----------------------------------------------------------------
        # 3.
        #    ""
        #    +2.0/×1.5×0.3
        #    +5.0
        # ----------------------------------------------------------------
        stomach = self.metabolism_state.stomach_fullness

        # 3a. PPO
        if feeding_out.food_consumed > 0:
            pellet_reward = 30.0 * feeding_out.food_consumed
            if stomach < 30:
                pellet_reward *= 3.0
            elif stomach > 80:
                pellet_reward *= 0.3
            # 20"="
            if self._total_food_eaten < 20:
                early_factor = 1.0 + 1.0 * (1.0 - self._total_food_eaten / 20.0)
                pellet_reward *= early_factor

            # feeding reward
            # 1-3s
            # ≈0.1s×300=30s""
            # 0~30≈0~15s×2.5
            # 31~80×1.8
            # 81~150×1.2
            # 150×1.0
            # detected_timer
            best_timer = self._last_food_detect_timer  #
            if best_timer <= 30:
                pellet_reward *= 2.5
            elif best_timer <= 80:
                pellet_reward *= 1.8
            elif best_timer <= 150:
                pellet_reward *= 1.2

            reward += pellet_reward

        # 3b. predation reward+3.0~+8.0/
        if interaction_out.predation_success > 0:
            fish_mass = getattr(interaction_out, 'mass_gained', 0)
            prey_mass = getattr(interaction_out, 'prey_mass', fish_mass / 0.7 if fish_mass > 0 else 1.0)
            prey_value_ratio = min(prey_mass / max(current_mass, 1e-6), 0.5)
            predation_reward = 5.0 + prey_value_ratio * 10.0
            if stomach < 30:
                predation_reward *= 1.3
            if self._total_fish_eaten == 0:  #
                predation_reward += 2.0
            reward += predation_reward

        # ----------------------------------------------------------------
        # 4.
        # ----------------------------------------------------------------
        nearest_prey_dist = self._get_nearest_prey_distance()
        current_food_distance = self.perception_state.nearest_food_distance
        has_reachable_food = bool(getattr(self.perception_state, 'observed_food', []))

        # 4a. +0.01~+0.05/
        if nearest_prey_dist is not None and nearest_prey_dist < float('inf'):
            if hasattr(self, '_last_prey_distance') and self._last_prey_distance < float('inf'):
                prey_approach = self._last_prey_distance - nearest_prey_dist
                if prey_approach > 0:
                    reward += min(prey_approach * 0.3, 0.05)
            self._last_prey_distance = nearest_prey_dist

        # 4b. +0.005~+0.03/
        if has_reachable_food and self.metabolism_state.activity_state == ActivityState.ACTIVE:
            if (current_food_distance < float('inf') and self._last_food_distance < float('inf')
                    and self.perception_state.nearest_food_position is not None):
                food_approach = self._last_food_distance - current_food_distance
                if food_approach > 0:
                    to_food_vec = (self.perception_state.nearest_food_position
                                   - self.physics_state.position)
                    to_food_dist = np.linalg.norm(to_food_vec)
                    if to_food_dist > 1e-4:
                        to_food_unit = to_food_vec / to_food_dist
                        radial_speed = float(np.dot(self.physics_state.velocity, to_food_unit))
                        # ≈0
                        radial_factor = max(0.0, radial_speed) / max(
                            np.linalg.norm(self.physics_state.velocity), 0.01)
                        reward += min(food_approach * 0.15, 0.03) * max(0.3, radial_factor)
                    else:
                        reward += min(food_approach * 0.15, 0.03)

            # +0.01~+0.05
            _food_near = max(length * 3.0, 0.05)
            if current_food_distance < _food_near:
                reward += 0.05 * (1.0 - current_food_distance / _food_near)

            # 3D+0~+0.06/0.3m
            # boost""
            _align_threshold = 0.30  # 3BL
            if (current_food_distance < _align_threshold and
                    current_food_distance > 1e-3 and
                    self.perception_state.nearest_food_position is not None):
                from systems.feeding import FeedingSystem
                pitch = self.physics_state.pitch_angle if hasattr(self.physics_state, 'pitch_angle') else 0.0
                heading = self.physics_state.heading if hasattr(self.physics_state, 'heading') else None
                mouth_dir = FeedingSystem._compute_mouth_direction_3d(heading, pitch)
                if mouth_dir is not None:
                    to_food = self.perception_state.nearest_food_position - self.physics_state.position
                    dist_to_food = np.linalg.norm(to_food)
                    if dist_to_food > 1e-4:
                        alignment = float(np.dot(mouth_dir, to_food / dist_to_food))
                        dist_factor = 1.0 - dist_to_food / _align_threshold
                        # 0.02boost0.06
                        reward += 0.06 * max(0.0, alignment) * dist_factor
                        # -
                        if current_food_distance < _align_threshold * 0.5 and alignment < 0.3:
                            reward -= 0.02 * dist_factor

            # 4c. +0~+0.015/
            # 0.3m1BL/s""
            if (current_food_distance < 0.30 and
                    self.metabolism_state.activity_state != ActivityState.RESTING):
                speed = np.linalg.norm(self.physics_state.velocity)
                speed_bl = speed / length
                # 1.0 BL/s
                # >3BL/s
                speed_error = abs(speed_bl - 1.0)
                speed_match = max(0.0, 1.0 - speed_error / 2.0)
                dist_factor_c = 1.0 - current_food_distance / 0.30
                reward += 0.015 * speed_match * dist_factor_c

        self._last_food_distance = current_food_distance if has_reachable_food else float('inf')

        # ----------------------------------------------------------------
        # 5. /
        # ----------------------------------------------------------------
        current_threat_distance = self.perception_state.nearest_threat_distance
        nearest_predator_dist = self._get_nearest_predator_distance()

        # 5a. +0.01~+0.08/
        if nearest_predator_dist < _pred_warn_dist:
            if hasattr(self, '_last_predator_distance') and self._last_predator_distance < float('inf'):
                escape_delta = nearest_predator_dist - self._last_predator_distance
                if escape_delta > 0:  #
                    proximity_factor = 1.0 + max(0.0, 1.0 - nearest_predator_dist / _pred_warn_dist)
                    reward += min(escape_delta * 0.4 * proximity_factor, 0.08)
            self._last_predator_distance = nearest_predator_dist
            # -0.05~-0.15/
            if nearest_predator_dist < _pred_danger_dist:
                reward -= 0.15 * (_pred_danger_dist - nearest_predator_dist) / _pred_danger_dist
        else:
            self._last_predator_distance = float('inf')

        # 5b. -0.03~-0.10/
        if current_threat_distance < _threat_dist:
            reward -= 0.10 * (_threat_dist - current_threat_distance) / _threat_dist
        self._last_threat_distance = current_threat_distance

        # 5c.
        if interaction_out.escape_success > 0:
            reward += 0.5 * interaction_out.escape_success

        # ----------------------------------------------------------------
        # 6. -0.1~-0.3/_process_damage
        # ----------------------------------------------------------------
        if interaction_out.damage_taken > 0:
            reward -= min(interaction_out.damage_taken * 0.02, 0.15)

        # ----------------------------------------------------------------
        # 7. /
        # ----------------------------------------------------------------
        is_resting = (self.metabolism_state.activity_state == ActivityState.RESTING)

        # 7a.
        if is_resting and self.metabolism_state.is_digesting:
            reward += 0.005

        # 7b. -0.03~-0.10/
        near_food_for_strike = (
            current_food_distance < CONFIG.inertia.strike_trigger_distance * 1.5
            and has_reachable_food
        )
        if is_resting and not near_food_for_strike:
            if stomach < 15:
                hunger_ratio = (15 - stomach) / 15.0
                reward -= 0.05 + 0.05 * hunger_ratio

        # 7c. -0.05~-0.20/
        if is_resting and current_threat_distance < _threat_dist:
            reward -= 0.20 * (_threat_dist - current_threat_distance) / _threat_dist

        # ----------------------------------------------------------------
        # 7d. -0.001~-0.025/
        # /
        # >
        # ----------------------------------------------------------------
        dist_to_bottom = self.physics_state.position[1] - (-self.tank_geometry.depth)
        if dist_to_bottom < 0.08:  # 8cm1BL
            bottom_proximity = (0.08 - dist_to_bottom) / 0.08  # 0~1
            # relative_density > 1
            # relative_density = 1 net_pressure ≈ 0
            _rho = max(self._last_relative_density, 0.95)
            net_pressure = max(0.0, _rho - 1.0) * 40.0 + 0.1   # ≈0.1
            #  ∝ mass^(2/3) ∝ mass^(1/3) mass^0.4
            size_factor = (current_mass / 20.0) ** 0.4
            reward -= 0.015 * bottom_proximity * net_pressure * size_factor

        # ----------------------------------------------------------------
        # 8 & 9. /
        #   A) ≤12cm+  + speed_bl>1.0→
        #   B) ≤1cm+  →
        # ----------------------------------------------------------------
        speed = np.linalg.norm(self.physics_state.velocity)
        speed_bl = speed / length
        boundary_distance = float(getattr(self.perception_state, 'min_boundary_distance', float('inf')))

        # boundary_vector
        vel_dir = self.physics_state.velocity / (speed + 1e-8)
        boundary_vec = self.perception_state.boundary_vector
        b_norm = np.linalg.norm(boundary_vec)
        toward_boundary = 0.0
        if b_norm > 1e-8:
            toward_boundary = max(0.0, -float(np.dot(vel_dir, boundary_vec / b_norm)))

        # A: //
        if not is_resting and boundary_distance < 0.12 and speed_bl > 1.0 and toward_boundary > 0.3:
            near_factor = (0.12 - boundary_distance) / 0.12
            excess_speed = min(speed_bl - 1.0, 2.0)
            reward -= toward_boundary * near_factor * excess_speed * 0.04

        # A: —
        if not is_resting:
            fwd_obs_norm = float(self.perception_state.obstacle_distances[0])
            fwd_obs_dist = fwd_obs_norm * CONFIG.perception.max_obstacle_distance
            if fwd_obs_dist < 0.12 and speed_bl > 1.0:
                near_factor_obs = (0.12 - fwd_obs_dist) / 0.12
                excess_speed = min(speed_bl - 1.0, 2.0)
                reward -= near_factor_obs * excess_speed * 0.04

        # B: ≤1cm
        if physics_out.collision_occurred:
            if toward_boundary > 0.15 or speed_bl > 0.8:
                base = 0.08
                speed_bonus = 0.06 * min(speed_bl, 2.5)
                reward -= base + speed_bonus
        #  _consecutive_collisions

        # ----------------------------------------------------------------
        # 10. cruising speed +
        # ----------------------------------------------------------------
        if not is_resting:
            cur_pos = self.physics_state.position.copy()
            self._recent_positions.append(cur_pos)
            if len(self._recent_positions) > 10:
                self._recent_positions.pop(0)

            # ----------------------------------------------------------------
            # ----------------------------------------------------------------
            g = self._exploration_grid
            gx = int((cur_pos[0] - g['x_min']) / (g['x_max'] - g['x_min']) * g['nx'])
            gy = int((cur_pos[1] - g['y_min']) / (g['y_max'] - g['y_min']) * g['ny'])
            gz = int((cur_pos[2] - g['z_min']) / (g['z_max'] - g['z_min']) * g['nz'])
            gx = int(np.clip(gx, 0, g['nx'] - 1))
            gy = int(np.clip(gy, 0, g['ny'] - 1))
            gz = int(np.clip(gz, 0, g['nz'] - 1))
            cell = (gx, gy, gz)
            if cell not in self._visited_cells:
                self._visited_cells.add(cell)
                reward += 1.0  #

            # ----------------------------------------------------------------
            # cruising speed 0.5~1.5 BL/s cruising speed
            #  0.5~1.5 BL/s
            # +0~+0.02/ +0.01/
            # ----------------------------------------------------------------
            current_speed = np.linalg.norm(self.physics_state.velocity)
            speed_bl = current_speed / max(length, 1e-3)
            #  0.5~1.5 BL/s
            if 0.3 <= speed_bl <= 2.0:
                if speed_bl <= 0.5:
                    cruise_reward = (speed_bl - 0.3) / 0.2 * 0.02
                elif speed_bl <= 1.5:
                    cruise_reward = 0.02
                else:
                    cruise_reward = (2.0 - speed_bl) / 0.5 * 0.02
                if not has_reachable_food:
                    cruise_reward *= 2.0
                reward += cruise_reward

            # ----------------------------------------------------------------
            #  < 0.25
            # >0.3m
            # ----------------------------------------------------------------
            should_check_spinning = (
                (has_reachable_food and current_food_distance > 0.3) or
                (not has_reachable_food)
            )
            if (should_check_spinning and len(self._recent_positions) == 10):
                path_len_xz = sum(
                    np.sqrt(
                        (self._recent_positions[i+1][0] - self._recent_positions[i][0])**2 +
                        (self._recent_positions[i+1][2] - self._recent_positions[i][2])**2
                    )
                    for i in range(9)
                )
                net_disp_xz = np.sqrt(
                    (self._recent_positions[-1][0] - self._recent_positions[0][0])**2 +
                    (self._recent_positions[-1][2] - self._recent_positions[0][2])**2
                )
                if path_len_xz > 0.05:
                    efficiency = net_disp_xz / path_len_xz
                    if efficiency < 0.25:
                        reward -= 0.15 * (0.25 - efficiency) / 0.25

        # ----------------------------------------------------------------
        # 11. =40%
        # ----------------------------------------------------------------
        if stomach <= 0.5:
            reward -= 0.06
        elif stomach < 5:
            reward -= 0.03

        # ----------------------------------------------------------------
        # 12. ""
        #     stomach<5%
        # ----------------------------------------------------------------
        if is_resting and stomach < 5.0 and self._steps_since_last_intake > 300:
            idle_excess = self._steps_since_last_intake - 300
            # 0.06/
            idle_penalty = min(0.02 + 0.01 * (idle_excess / 300.0), 0.06)
            reward -= idle_penalty

        return np.clip(reward, -2.0, 200.0)

    def _calculate_terminal_reward_v52(self) -> float:
        """
         v5.3
        ============================================================
         ≈  20~30%
         PPO ""

         episode5% ~200
           400~600
          → 80~150


          - growth rewardmass_ratio × 505%→2.520%→1030
            800 × mass_ratio^0.65%→149
          - efficiency × 1520
          - fish_eaten × 2.0 + 30
          - death penalty-5 ~ -20
          -
        ============================================================
        """
        reward = 0.0

        initial_mass = self._initial_mass
        final_mass = self.growth_state.body_mass
        mass_change = final_mass - initial_mass
        mass_ratio = mass_change / initial_mass if initial_mass > 0 else 0

        # 1. growth reward 2~30
        if mass_change > 0:
            growth_reward = min(mass_ratio * 50.0, 30.0)
            reward += growth_reward
        else:
            #  -0.5 ~ -5.0
            reward -= min(abs(mass_ratio) * 10.0, 5.0)

        # 2.  ~1.5 for full episode
        survival_bonus = np.log10(self.current_step + 1) * 0.3
        reward += survival_bonus

        # 3. predation reward20
        fish_eaten = getattr(self, '_total_fish_eaten', 0)
        if fish_eaten > 0:
            predation_bonus = min(fish_eaten * 2.0, 10.0)
            reward += predation_bonus
            if fish_eaten >= 1:
                reward += 1.0
            if fish_eaten >= 3:
                reward += 2.0
            if fish_eaten >= 5:
                reward += 3.0
            if fish_eaten >= 10:
                reward += 4.0

        # 4. death penalty -5 ~ -20
        if self._death_reason:
            if self._growth_event_count == 0:
                reward -= 20.0
            elif self.current_step < 500:
                reward -= 10.0
            else:
                survival_factor = 1 - self.current_step / CONFIG.environment.max_episode_steps
                reward -= 5.0 * max(0, survival_factor)

        # 5. 3.0
        if self._growth_event_count > 0:
            growth_count_bonus = min(self._growth_event_count * 0.2, 3.0)
            reward += growth_count_bonus

        # 6. 20
        total_intake = self._mass_from_pellets + self._mass_from_fish
        if total_intake > 0:
            efficiency = max(0, mass_change) / total_intake
            reward += min(efficiency * 15.0, 20.0)

        # 7. 0~2.0
        rest_ratio = self._total_rest_steps / max(1, self.current_step)
        if 0.20 <= rest_ratio <= 0.40:
            reward += 2.0
        elif 0.10 <= rest_ratio < 0.20 or 0.40 < rest_ratio <= 0.50:
            reward += 1.0

        if self._rest_growth_bonus_accumulated > 0:
            reward += min(self._rest_growth_bonus_accumulated * 0.3, 1.5)

        if self._rest_during_danger > 10:
            reward -= min(self._rest_during_danger * 0.03, 1.5)

        return reward

    def _process_damage(self, interaction_output) -> float:
        """
         - v4


        1.  interaction_output  attacker_mass
        2. mass loss3
        3.
        """
        if interaction_output.damage_taken <= 0:
            return 0.0

        # output
        attacker_mass = getattr(interaction_output, 'attacker_mass', 0)
        if attacker_mass <= 0:
            attacker_mass = self.growth_state.body_mass * CONFIG.interaction.threat_size_ratio

        ai_mass = self.growth_state.body_mass
        size_ratio = attacker_mass / ai_mass if ai_mass > 0 else 999

        self._total_damage += interaction_output.damage_taken

        reward_penalty = 0.0
        mass_loss = 0.0

        # ===== =====
        # growth reward
        energy_loss = 0.0
        if size_ratio >= 5:
            energy_loss = self.metabolism_state.energy * 0.30  # 30%
            reward_penalty = -100.0
        elif size_ratio >= 3.5:
            energy_loss = self.metabolism_state.energy * 0.15
            reward_penalty = -50.0
        elif size_ratio >= 2:
            energy_loss = self.metabolism_state.energy * 0.07
            reward_penalty = -25.0
        elif size_ratio >= 1.5:
            energy_loss = self.metabolism_state.energy * 0.03
            reward_penalty = -25.0
        else:
            energy_loss = self.metabolism_state.energy * 0.005
            reward_penalty = -1.0

        if energy_loss > 0:
            old_energy = self.metabolism_state.energy
            self.metabolism_state.energy = max(0.0, old_energy - energy_loss)

            if self.runtime_config.get('verbose', 0) >= 1:
                print(f"⚠️ ! ={attacker_mass:.0f}g, size ratio={size_ratio:.1f}x, "
                      f"={energy_loss:.2f}kJ ({old_energy:.2f}→{self.metabolism_state.energy:.2f}kJ)")

        stress_increase = min(0.3 * (size_ratio / 5), 0.5)
        self.metabolism_state.stress_level = min(1.0, self.metabolism_state.stress_level + stress_increase)

        return reward_penalty

    def _check_termination(self) -> bool:
        """"""
        if hasattr(self, '_killed_by_predator') and self._killed_by_predator:
            self._death_reason = 'predation'
            return True

        # air_exposure death removed: fish that breach the surface are now
        # pulled back by gravity (physics fix).  No death from brief air exposure.

        if self.growth_state.body_mass < 1.0:
            self._death_reason = 'starvation'
            return True

        if self._initial_mass > 0:
            mass_loss_ratio = 1 - (self.growth_state.body_mass / self._initial_mass)

            if self._initial_mass < 20:
                death_threshold = 0.275
            elif self._initial_mass < 100:
                death_threshold = 0.275 + 0.075 * ((self._initial_mass - 20) / 80)
            else:
                death_threshold = 0.35 + 0.05 * np.log10(self._initial_mass / 100)
                death_threshold = min(death_threshold, 0.45)

            if mass_loss_ratio >= death_threshold:
                self._death_reason = 'starvation'
                return True

        return False

    def _get_info(self) -> Dict[str, Any]:
        """"""
        info = {
            'energy': self.metabolism_state.energy,
            'stomach_fullness': self.metabolism_state.stomach_fullness,
            'stress_level': self.metabolism_state.stress_level,
            'water_temp': self._current_water_temp,
            'body_mass': self.growth_state.body_mass,
            'total_length': self.growth_state.total_length,
            'position': self.physics_state.position.copy(),
            'velocity_magnitude': np.linalg.norm(self.physics_state.velocity),

            'total_food_eaten': self._total_food_eaten,
            'floating_eaten': self._floating_eaten,
            'sinking_eaten': self._sinking_eaten,
            'ambient_eaten': self._ambient_eaten,
            'surface_env_eaten': self._surface_env_eaten,
            'benthic_eaten': self._benthic_eaten,
            'attached_eaten': self._attached_eaten,
            'total_fish_eaten': self._total_fish_eaten,
            'energy_from_pellets': self._energy_from_pellets,
            'energy_from_fish': self._energy_from_fish,
            'mass_from_pellets': self._mass_from_pellets,
            'mass_from_fish': self._mass_from_fish,

            'times_chased': self._times_chased,
            'surface_entries': self._surface_entries,
            'damage_taken': self._total_damage,
            'escape_count': self._escape_count,

            'distance_traveled': self.physics_state.total_distance_traveled,
            'collision_count': self.physics_state.collision_count,

            'survival_time': self.current_step,
            'growth_count': self.growth_state.growth_count,
            'initial_mass': self._initial_mass,
            'mass_change': self.growth_state.body_mass - self._initial_mass,
            'growth_event_count': self._growth_event_count,

            'food_available': self.feeding_system.get_food_count(self.feeding_state),
            'fish_remaining': self.interaction_system.get_fish_count(self.interaction_state),
            'death_reason': self._death_reason,

            'activity_state': self.metabolism_state.activity_state.value,
            'rest_duration_steps': self.metabolism_state.rest_duration_steps,
            'total_rest_steps': self._total_rest_steps,
            'rest_ratio': self._total_rest_steps / max(1, self.current_step),
            'state_switches': self._state_switches,
            'max_consecutive_rest': self._max_consecutive_rest,
            'metabolism_factor': self.metabolism_state.current_metabolism_factor,
            'growth_bonus': self.metabolism_state.current_growth_bonus,
            'rest_during_danger': self._rest_during_danger,

            'buoyancy_energy_total': self._buoyancy_energy_total,
            'buoyancy_adjustments': self._buoyancy_adjustments,
            'neutral_buoyancy_steps': self._neutral_buoyancy_steps,
            'relative_density': self._last_relative_density,
            'avg_relative_density': self._last_relative_density,
            'current_depth': max(0, -self.physics_state.position[1]),

            'attacks_received': getattr(self.interaction_state, 'attacks_this_episode', 0),
            'attack_damage_total': self._total_damage,
            'mass_lost_to_attacks': self._initial_mass - self.growth_state.body_mass if hasattr(self, '_mass_lost_to_attacks') else 0,

            # ========== new ==========
            'action_normalized_count': self._action_normalized_count,
            'speed_clamped_count': self._speed_clamped_count,
            'action_normalized_ratio': self._action_normalized_count / max(1, self.current_step),
            'speed_clamped_ratio': self._speed_clamped_count / max(1, self.current_step),
            'last_action_delta': self._last_action_delta,
            'consecutive_collisions': self._consecutive_collisions,

            # new
            'pitch_angle': self.physics_state.pitch_angle if hasattr(self.physics_state, 'pitch_angle') else 0.0,
            'reynolds_number': self.physics_state.current_reynolds if hasattr(self.physics_state,
                                                                              'current_reynolds') else 0.0,
            'is_coasting': self.physics_state.is_coasting if hasattr(self.physics_state, 'is_coasting') else False,
            # ========== new ==========
            'tank_shape': self.tank_geometry.shape_name,
            'tank_extents': self.tank_geometry.get_extents(),
            'obstacle_count': self.obstacle_field.count,
            'obstacle_collisions': self._obstacle_collisions,
            'forced_active_steps': self._forced_active_steps,
            'forced_active_hunger_steps': self._forced_active_hunger_steps,
            'training_phase': self.runtime_config.get('training_phase', 'course4'),
            'first_intake_step': int(self._first_intake_step) if self._first_intake_step is not None else -1,
            'steps_since_last_intake': int(self._steps_since_last_intake),
            'min_obstacle_distance': float(np.min(self.perception_state.obstacle_distances)) * CONFIG.perception.max_obstacle_distance,
            'min_boundary_distance': float(getattr(self.perception_state, 'min_boundary_distance', float('inf'))),
        }

        return info

    def _log_status_enhanced(self, reward: float, metabolism_output, physics_output=None):
        """"""
        mass_change = self.growth_state.body_mass - self._initial_mass
        state_char = "😴" if self.metabolism_state.activity_state == ActivityState.RESTING else "🏃"

        speed = np.linalg.norm(self.physics_state.velocity)
        speed_bls = speed / self.growth_state.total_length if self.growth_state.total_length > 0 else 0

        density_str = ""
        if physics_output and hasattr(physics_output, 'relative_density'):
            density = physics_output.relative_density
            if density < 0.98:
                density_str = f"🎈{density:.2f}"
            elif density > 1.02:
                density_str = f"⚓{density:.2f}"
            else:
                density_str = f"⚖️{density:.2f}"

        pitch_str = ""
        if hasattr(self.physics_state, 'pitch_angle'):
            pitch = self.physics_state.pitch_angle
            if pitch > 10:
                pitch_str = f"↗️{pitch:.0f}°"
            elif pitch < -10:
                pitch_str = f"↘️{abs(pitch):.0f}°"
            else:
                pitch_str = f"→{pitch:.0f}°"

        coast_str = "🏊" if getattr(self.physics_state, 'is_coasting', False) else ""

        re_str = ""
        if physics_output and hasattr(physics_output, 'reynolds_number'):
            re = physics_output.reynolds_number
            if re > 10000:
                re_str = f"Re:{re / 1000:.0f}k"
            else:
                re_str = f"Re:{re:.0f}"

        print(f"[{self.current_step:4d}] {state_char}{coast_str} "
              f":{self.metabolism_state.energy:.1f}% "
              f":{self.growth_state.body_mass:.1f}g({mass_change:+.1f}) "
              f":{speed_bls:.1f}BL/s "
              f"{pitch_str} {re_str} "
              f":{reward:.3f}")

    def _on_episode_end(self, terminated: bool):
        survival_time = self.current_step

        self.all_episode_lengths.append(survival_time)
        self.all_episode_rewards.append(sum(self.episode_rewards))
        self.all_episode_food_counts.append(self._total_food_eaten)
        self.survival_time_history.append(survival_time)

        if self.runtime_config['verbose'] >= 1:
            mass_change = self.growth_state.body_mass - self._initial_mass
            mass_ratio = mass_change / self._initial_mass * 100
            rest_ratio = self._total_rest_steps / max(1, self.current_step) * 100
            neutral_ratio = self._neutral_buoyancy_steps / max(1, self.current_step) * 100
            norm_ratio = self._action_normalized_count / max(1, self.current_step) * 100

            print(f"\n{'=' * 60}")
            print(f"Episode {self.episode_count} ")
            print(f"{'=' * 60}")
            print(f": {survival_time}")
            print(f": {self._total_food_eaten} + {self._total_fish_eaten}")
            print(f": {self._initial_mass:.1f}g → {self.growth_state.body_mass:.1f}g ({mass_ratio:+.1f}%)")
            print(f": {rest_ratio:.1f}%, {self._state_switches}")
            print(f": Normalize{norm_ratio:.1f}%, Speed limit{self._speed_clamped_count}")
            print(f": {self.tank_geometry.get_extents()}, "
                  f": {self.obstacle_field.count}")
            print(f": {sum(self.episode_rewards):.2f}")

            if terminated and self._death_reason:
                print(f": {self._death_reason}")

    def set_curriculum_stage(self, stage: int):
        self.curriculum_config = get_curriculum_stage(stage)

        if self.runtime_config['verbose'] >= 1:
            print(f"📚  {stage}: {self.curriculum_config.get('name', '')}")

    def set_training_phase(self, phase: str):
        """"""
        phase_name = str(phase).lower()

        if phase_name in {'course1', 'phase1', 'foundation', 'easy', 'feed'}:
            # 1
            self.runtime_config['training_phase'] = 'course1'
            self.runtime_config['force_default_tank'] = True
            self.runtime_config['disable_obstacles'] = True
            self.runtime_config['obstacle_density_multiplier'] = 1.0
        elif phase_name in {'course2', 'phase2', 'navigation', 'maze', 'foraging'}:
            # 2/
            self.runtime_config['training_phase'] = 'course2'
            self.runtime_config['force_default_tank'] = False
            self.runtime_config['disable_obstacles'] = False
            self.runtime_config['obstacle_density_multiplier'] = 2.0
        elif phase_name in {'course3', 'phase3', 'threat', 'predation'}:
            # Course 3: foraging/predation under threat
            self.runtime_config['training_phase'] = 'course3'
            self.runtime_config['force_default_tank'] = False
            self.runtime_config['disable_obstacles'] = False
            self.runtime_config['obstacle_density_multiplier'] = 1.4
        else:
            # 4
            self.runtime_config['training_phase'] = 'course4'
            self.runtime_config['force_default_tank'] = False
            self.runtime_config['disable_obstacles'] = False
            self.runtime_config['obstacle_density_multiplier'] = 1.0


# ============================================================
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("v5.2.1 Fix")
    print("=" * 60)

    env = BassEnvironment({'verbose': 1, 'debug_speed': True})

    print(f"\nAction Space: {env.action_space}")
    print(f"Observation Space: {env.observation_space}")
    print(f"\nSpeed Limiting:")
    print(f"  : ≤ 0.5 BL/s")
    print(f"  : ≤ 1.5 BL/s")
    print(f"  : ≤ 3.0 BL/s")

    print("\n...")
    obs, info = env.reset()

    total_reward = 0

    for step in range(200):
        action = np.array([0.9, 0.8, 1.0, 0.5, 0.0], dtype=np.float32)  #

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if step % 50 == 0:
            speed = info['velocity_magnitude']
            length = info['total_length']
            speed_bls = speed / length if length > 0 else 0
            print(f"Step {step}: ={speed:.4f}m/s ({speed_bls:.2f}BL/s), "
                  f"Normalize={info['action_normalized_count']}, Speed limit={info['speed_clamped_count']}")

        if terminated or truncated:
            print(f"\nEpisode: {total_reward:.2f}")
            break

    print("\n✅ ！")
