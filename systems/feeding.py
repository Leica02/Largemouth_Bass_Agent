#!/usr/bin/env python3
"""
Feeding Subsystem — Unified Configuration
==========================================

This module implements the feeding subsystem for a reinforcement learning
environment simulating largemouth bass (*Micropterus salmoides*) foraging
behaviour in recirculating aquaculture systems (RAS). The subsystem manages:

1. **Pellet physics**: floating drift, sinking trajectories, and settling
   dynamics governed by Stokes-law-inspired parameterisation.
2. **Pellet spawning**: timed batch feeding with stochastic meal-size
   variation and body-mass-scaled intake, plus continuous environmental
   food generation (ambient, surface, benthic, obstacle-attached).
3. **Food capture mechanics**: 3D mouth-cone detection with line-of-sight
   constraint and swept-sphere hit testing along the agent trajectory.
4. **Stomach capacity constraints**: species-appropriate gastric fill
   limits scaled to body mass.

All configuration parameters are read from the centralised ``CONFIG.feeding``
frozen dataclass. Settling is triggered at a configurable fraction of the
feeding interval, after which all remaining pellets sink to the tank floor
and are removed upon contact — they are not retained by the bottom boundary.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CONFIG


# ============================================================
# Enumeration types
# ============================================================

class FoodType(Enum):
    """Pellet and environmental food type classification."""
    FLOATING    = "floating"     # floating pellet
    SINKING     = "sinking"      # sinking pellet
    SETTLING    = "settling"     # sinking to bottom
    AMBIENT     = "ambient"      # mid-water drifting (does not sink; removed by age)
    SURFACE_ENV = "surface_env"  # surface drifting (removed by age)
    BENTHIC     = "benthic"      # stationary on tank floor (removed by age)
    ATTACHED    = "attached"     # fixed to obstacle surface (removed by age)

# ============================================================
# Data class definitions
# ============================================================

@dataclass
class FoodItem:
    """Single food particle with position, velocity, and nutrient profile."""
    position: np.ndarray
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    mass: float = 0.1
    age: int = 0
    food_type: FoodType = FoodType.FLOATING
    original_type: FoodType = FoodType.FLOATING

    max_speed: float = 0.02
    direction_change_timer: int = 0
    direction_change_interval: int = 50

    is_settling: bool = False
    settle_speed: float = 0.008

    # Nutrient profile (fraction of pellet mass)
    protein_fraction: float = 0.50
    lipid_fraction: float = 0.09
    carbohydrate_fraction: float = 0.0
    adc_protein: float = 0.90
    adc_lipid: float = 0.90
    adc_carbohydrate: float = 0.30
    include_carbohydrate_energy: bool = False
    max_age: int = 0  # 0 = unlimited lifespan (legacy); >0 = removed after this many steps

    # Detection timer: starts counting after the fish perceives the item
    detected: bool = False          # whether already perceived by the agent
    detected_timer: int = 0         # cumulative steps since detection
    detected_timeout: int = 0       # timeout steps; 0 = disabled

    @property
    def is_floating(self) -> bool:
        return self.food_type == FoodType.FLOATING


@dataclass
class FeedingState:
    """Mutable state container for the feeding subsystem."""
    food_items: List[FoodItem] = field(default_factory=list)
    steps_since_feeding: int = 0
    total_food_spawned: int = 0
    total_food_eaten: int = 0
    total_food_wasted: int = 0
    total_mass_spawned: float = 0.0
    total_mass_eaten: float = 0.0

    floating_food_spawned: int = 0
    sinking_food_spawned: int = 0
    floating_food_eaten: int = 0
    sinking_food_eaten: int = 0

    last_known_fish_mass: float = 20.0
    current_batch_settling: bool = False


@dataclass
class FeedingInput:
    """Per-step input data for the feeding subsystem."""
    agent_position: np.ndarray
    agent_mass: float
    agent_length: float
    stomach_fullness: float
    agent_prev_position: Optional[np.ndarray] = None
    agent_heading: Optional[np.ndarray] = None
    agent_pitch_angle: float = 0.0  # pitch angle (degrees), used for 3D mouth direction
    tank_radius: float = field(default_factory=lambda: CONFIG.environment.tank_radius)
    tank_depth: float = field(default_factory=lambda: CONFIG.environment.tank_depth)
    tank_geometry: Any = None
    obstacle_field: Any = None

@dataclass
class FeedingOutput:
    """Per-step output data from the feeding subsystem."""
    food_consumed: int = 0
    mass_consumed: float = 0.0
    fullness_gained: float = 0.0
    energy_potential: float = 0.0
    food_spawned: int = 0
    mass_spawned: float = 0.0
    floating_spawned: int = 0
    sinking_spawned: int = 0
    floating_consumed: int = 0
    sinking_consumed: int = 0
    ambient_consumed: int = 0
    surface_env_consumed: int = 0
    benthic_consumed: int = 0
    attached_consumed: int = 0


# ============================================================
# Helper functions
# ============================================================

def calculate_capture_radius(body_length: float) -> float:
    """Calculate the gape-limited capture radius for a given body length.

    Args:
        body_length: Total body length of the fish (m).

    Returns:
        Capture radius (m), scaled by body-size-dependent coefficient.
    """
    ic = CONFIG.interaction
    if body_length <= ic.capture_radius_small_threshold:
        coef = ic.capture_radius_small
    elif body_length <= ic.capture_radius_medium_threshold:
        coef = ic.capture_radius_medium
    else:
        coef = ic.capture_radius_large
    return max(body_length * coef, ic.capture_radius_min)


def calculate_stomach_capacity(body_mass: float) -> float:
    """Calculate maximum stomach capacity.

    Args:
        body_mass: Current body mass of the fish (g).

    Returns:
        Maximum stomach content mass (g).
    """
    return body_mass * CONFIG.feeding.stomach_capacity_ratio


# ============================================================
# Feeding subsystem core class
# ============================================================

class FeedingSystem:
    """Feeding subsystem — unified configuration version.

    Manages pellet spawning, physics-based movement, settling transitions,
    environmental food generation, and capture detection for a single agent.
    """

    def __init__(self) -> None:
        self.c = CONFIG.feeding
        self.env = CONFIG.environment
        self.debug = False
        self.mc = CONFIG.metabolism
        self._diet_profile = self._resolve_diet_profile()

        # Dynamic feeding interval
        self.feeding_interval = self._calculate_feeding_interval()
        self.settle_trigger_step = int(self.feeding_interval * self.c.settle_trigger_ratio)

    @staticmethod
    def _to_fraction(value: float) -> float:
        """Allow either fraction (0-1) or percent (0-100) input."""
        value = float(value)
        if value > 1.0:
            value /= 100.0
        return max(0.0, value)

    def _resolve_diet_profile(self) -> Dict[str, Any]:
        """Read default diet profile for pellets (composition x ADC).

        Returns:
            Dictionary containing nutrient fractions and apparent
            digestibility coefficients for the configured diet.
        """
        protein = self._to_fraction(getattr(self.c, 'diet_protein_fraction',
                                            getattr(CONFIG.metabolism, 'feed_protein_fraction', 0.50)))
        lipid = self._to_fraction(getattr(self.c, 'diet_lipid_fraction',
                                          getattr(CONFIG.metabolism, 'feed_lipid_fraction', 0.09)))
        carbohydrate = self._to_fraction(getattr(self.c, 'diet_carbohydrate_fraction',
                                                 getattr(CONFIG.metabolism, 'feed_carbohydrate_fraction', 0.0)))

        total = protein + lipid + carbohydrate
        if total > 1.0 and total > 0:
            protein /= total
            lipid /= total
            carbohydrate /= total

        return {
            'protein_fraction': protein,
            'lipid_fraction': lipid,
            'carbohydrate_fraction': carbohydrate,
            'adc_protein': np.clip(self._to_fraction(getattr(self.c, 'diet_adc_protein',
                                                             getattr(CONFIG.metabolism, 'adc_protein', 0.90))), 0.0, 1.0),
            'adc_lipid': np.clip(self._to_fraction(getattr(self.c, 'diet_adc_lipid',
                                                           getattr(CONFIG.metabolism, 'adc_lipid', 0.90))), 0.0, 1.0),
            'adc_carbohydrate': np.clip(self._to_fraction(getattr(self.c, 'diet_adc_carbohydrate',
                                                                  getattr(CONFIG.metabolism, 'adc_carbohydrate', 0.30))), 0.0, 1.0),
            'include_carbohydrate_energy': bool(getattr(
                self.c,
                'diet_include_carbohydrate_energy',
                getattr(CONFIG.metabolism, 'include_carbohydrate_energy', False)
            )),
        }

    def _calculate_item_digestible_energy_density(self, food: FoodItem) -> float:
        """Compute digestible energy density (kJ/g) from nutrient composition and ADCs.

        Args:
            food: The food item whose nutrient profile is evaluated.

        Returns:
            Digestible energy density in kJ per gram of pellet.
        """
        protein_kj = food.protein_fraction * food.adc_protein * self.mc.protein_energy_density
        lipid_kj = food.lipid_fraction * food.adc_lipid * self.mc.lipid_energy_density

        carb_kj = 0.0
        if food.include_carbohydrate_energy:
            carb_kj = (
                food.carbohydrate_fraction
                * food.adc_carbohydrate
                * self.mc.carbohydrate_energy_density
            )

        return max(0.0, protein_kj + lipid_kj + carb_kj)

    def _calculate_feeding_interval(self) -> int:
        """Dynamically calculate the feeding interval in simulation steps.

        Returns:
            Number of simulation steps between successive feedings.
        """
        time_step = self.env.time_step
        time_acc = self.env.time_acceleration
        steps_per_day = int(24 * 3600 / (time_step * time_acc))
        return max(steps_per_day // self.c.feedings_per_day, 100)

    def _sample_meal_intake_factor(self) -> float:
        """Sample a stochastic multiplicative meal factor around 1.0.

        Returns:
            A factor drawn from a clipped normal distribution representing
            natural variation in individual meal size.
        """
        if not bool(getattr(self.c, 'stochastic_meal_intake_enabled', False)):
            return 1.0

        meal_cv = max(0.0, float(getattr(self.c, 'meal_intake_cv', 0.0)))
        if meal_cv <= 0.0:
            return 1.0

        min_factor = float(getattr(self.c, 'meal_intake_min_factor', 0.6))
        max_factor = float(getattr(self.c, 'meal_intake_max_factor', 1.4))
        if min_factor > max_factor:
            min_factor, max_factor = max_factor, min_factor

        sample = np.random.normal(1.0, meal_cv)
        return float(np.clip(sample, min_factor, max_factor))

    def _mass_intake_scaling_factor(self, fish_mass: float) -> float:
        """Body-mass allometric scaling for effective available intake.

        Args:
            fish_mass: Current fish mass (g).

        Returns:
            Scaling factor applied to the nominal daily ration.
        """
        if not bool(getattr(self.c, 'mass_intake_scaling_enabled', False)):
            return 1.0

        ref_mass = max(1e-6, float(getattr(self.c, 'mass_intake_scaling_ref_mass_g', 200.0)))
        exponent = float(getattr(self.c, 'mass_intake_scaling_exponent', -1.15))
        min_factor = float(getattr(self.c, 'mass_intake_scaling_min', 0.47))
        max_factor = float(getattr(self.c, 'mass_intake_scaling_max', 1.0))
        if min_factor > max_factor:
            min_factor, max_factor = max_factor, min_factor

        factor = (max(float(fish_mass), 1e-6) / ref_mass) ** exponent
        return float(np.clip(factor, min_factor, max_factor))

    def update(self, state: FeedingState, input_data: FeedingInput,
               curriculum_multiplier: float = 1.0,
               capture_radius_multiplier: float = 1.0,
               env_food_only: bool = False) -> FeedingOutput:
        """Execute one simulation step of the feeding subsystem.

        Args:
            state: Mutable feeding state.
            input_data: Agent and environment observations for this step.
            curriculum_multiplier: Curriculum-learning scalar applied to
                spawned food quantity.
            capture_radius_multiplier: Multiplicative adjustment to the
                capture radius (used during curriculum training).
            env_food_only: If True, skip batch feeding logic and only
                process environmental food.

        Returns:
            FeedingOutput summarising spawned and consumed food this step.
        """
        output = FeedingOutput()
        state.last_known_fish_mass = input_data.agent_mass
        state.steps_since_feeding += 1

        if not env_food_only:
            # Trigger settling after configured fraction of the feeding interval.
            if (state.steps_since_feeding == self.settle_trigger_step and
                    not state.current_batch_settling):
                self._trigger_all_food_settling(state)
                state.current_batch_settling = True
                if self.debug:
                    print(f"[feeding] trigger settling at step {state.steps_since_feeding}")

        # Timed batch feeding
            if state.steps_since_feeding >= self.feeding_interval:
                spawned = self._spawn_food(state, input_data, curriculum_multiplier)
                output.food_spawned = spawned['count']
                output.mass_spawned = spawned['mass']
                output.floating_spawned = spawned['floating']
                output.sinking_spawned = spawned['sinking']
                state.steps_since_feeding = 0
                state.current_batch_settling = False

        # Update food movement
        self._update_food_movement(state, input_data)

        # Check food consumption.
        consumed = self._check_consumption(state, input_data, capture_radius_multiplier)
        output.food_consumed = consumed['count']
        output.mass_consumed = consumed['mass']
        output.fullness_gained = consumed['fullness_gained']
        output.energy_potential = consumed['energy_potential']
        output.floating_consumed = consumed['floating_consumed']
        output.sinking_consumed = consumed['sinking_consumed']
        output.ambient_consumed = consumed['ambient_consumed']
        output.surface_env_consumed = consumed['surface_env_consumed']
        output.benthic_consumed = consumed['benthic_consumed']
        output.attached_consumed = consumed['attached_consumed']

        return output

    def _trigger_all_food_settling(self, state: FeedingState) -> None:
        """Trigger settling for all remaining batch-fed pellets.

        Environmental food types (ambient, surface_env, benthic, attached)
        are excluded from the settling mechanism.

        Args:
            state: Current feeding state whose food items will be modified.
        """
        _env_types = {FoodType.AMBIENT, FoodType.SURFACE_ENV,
                      FoodType.BENTHIC, FoodType.ATTACHED}
        for food in state.food_items:
            if food.original_type in _env_types:
                continue  # Environmental food does not participate in settling
            if not food.is_settling:
                food.is_settling = True
                food.food_type = FoodType.SETTLING

                # Set settling speed based on original pellet type
                if food.original_type == FoodType.FLOATING:
                    food.settle_speed = self.c.settle_speed_floating
                else:
                    food.settle_speed = self.c.settle_speed_sinking

                food.velocity = np.array([
                    food.velocity[0] * 0.3,
                    -food.settle_speed,
                    food.velocity[2] * 0.3
                ], dtype=np.float32)

    def _spawn_food(self, state: FeedingState, input_data: FeedingInput,
                    multiplier: float = 1.0) -> Dict[str, Any]:
        """Spawn a batch of pellets at the scheduled feeding time.

        Args:
            state: Mutable feeding state.
            input_data: Current agent and tank information.
            multiplier: Curriculum multiplier for total food mass.

        Returns:
            Dictionary with keys 'count', 'mass', 'floating', 'sinking'.
        """
        fish_mass = input_data.agent_mass
        intake_scaling = self._mass_intake_scaling_factor(fish_mass)
        daily_total = fish_mass * self.c.daily_feeding_rate * multiplier * intake_scaling
        target_mass = daily_total / self.c.feedings_per_day
        target_mass *= self._sample_meal_intake_factor()

        num_floating = np.random.randint(self.c.floating_pellets_min,
                                         self.c.floating_pellets_max + 1)
        num_sinking = np.random.randint(self.c.sinking_pellets_min,
                                        self.c.sinking_pellets_max + 1)
        total = num_floating + num_sinking

        pellet_mass = np.clip(target_mass / total,
                              self.c.pellet_mass_min,
                              self.c.pellet_mass_max)
        actual_mass = pellet_mass * total

        for _ in range(num_floating):
            food = self._create_floating_pellet(input_data, pellet_mass)
            self._place_in_blind_zone(food, input_data, keep_surface=True)
            food.detected_timeout = 400
            state.food_items.append(food)

        for _ in range(num_sinking):
            food = self._create_sinking_pellet(input_data, pellet_mass)
            self._place_in_blind_zone(food, input_data, keep_surface=False)
            food.detected_timeout = 400
            state.food_items.append(food)

        state.total_food_spawned += total
        state.total_mass_spawned += actual_mass
        state.floating_food_spawned += num_floating
        state.sinking_food_spawned += num_sinking

        if self.debug:
            print(f"[feeding] spawned: floating={num_floating} + sinking={num_sinking}, "
                  f"settle trigger at step {self.settle_trigger_step}")

        return {'count': total, 'mass': actual_mass,
                'floating': num_floating, 'sinking': num_sinking}

    def _get_tank_depth(self, input_data: FeedingInput) -> float:
        """Retrieve the effective tank depth from geometry or input data.

        Args:
            input_data: Current feeding input containing tank parameters.

        Returns:
            Tank depth in metres.
        """
        tank_geo = getattr(input_data, 'tank_geometry', None)
        if tank_geo is not None:
            return float(getattr(tank_geo, 'depth', input_data.tank_depth))
        return float(input_data.tank_depth)

    def _place_in_blind_zone(self, food: 'FoodItem', input_data: 'FeedingInput',
                              keep_surface: bool) -> None:
        """Relocate food into the agent's visual blind zone (rear 60-deg cone).

        The agent has a 300-degree field of view, leaving a 60-degree blind
        zone in the rear (+-30 degrees from directly behind). Food is placed
        in this zone so the agent must actively turn to detect new pellets.

        If sampling fails (e.g., tank too small), the original random
        position is retained.

        Args:
            food: The food item whose position may be relocated.
            input_data: Current agent state including heading.
            keep_surface: If True, constrain Y to 0 (surface).
        """
        heading = getattr(input_data, 'agent_heading', None)
        if heading is None or np.linalg.norm(heading[:] if hasattr(heading, '__len__') else heading) < 1e-6:
            return  # No heading information; retain original position

        heading_xz = np.array([heading[0], 0.0, heading[2]], dtype=np.float32)
        heading_xz_norm = np.linalg.norm(heading_xz)
        if heading_xz_norm < 1e-6:
            return
        heading_xz = heading_xz / heading_xz_norm

        agent_pos = input_data.agent_position
        tank_geo = getattr(input_data, 'tank_geometry', None)
        tank_depth = self._get_tank_depth(input_data)
        min_depth = -tank_depth + self.c.bottom_buffer + 0.05
        max_depth = -self.c.surface_buffer - 0.1

        # Blind zone: angle with heading > 150 deg (cos < -0.866), i.e., rear +-30 deg
        blind_cos_threshold = -0.866

        for _ in range(40):
            pos = self._sample_food_position(input_data, keep_surface)
            to_food_xz = np.array([pos[0] - agent_pos[0], 0.0, pos[2] - agent_pos[2]],
                                   dtype=np.float32)
            dist_xz = np.linalg.norm(to_food_xz)
            if dist_xz < 0.05:
                continue  # Too close; skip
            cos_angle = float(np.dot(heading_xz, to_food_xz / dist_xz))
            if cos_angle <= blind_cos_threshold:
                # Within the blind zone; apply this position
                food.position = pos
                return
        # All 40 attempts failed (small tank / agent near boundary); retain original position

    def _sample_food_position(self, input_data: FeedingInput, keep_surface: bool) -> np.ndarray:
        """Sample a valid random food position within the tank volume.

        Args:
            input_data: Tank geometry and obstacle field reference.
            keep_surface: If True, fix Y = 0 (water surface).

        Returns:
            3D position array (float32).
        """
        tank_geo = getattr(input_data, 'tank_geometry', None)
        obs_field = getattr(input_data, 'obstacle_field', None)
        tank_depth = self._get_tank_depth(input_data)

        min_depth = -tank_depth + self.c.bottom_buffer + 0.05
        max_depth = -self.c.surface_buffer - 0.1
        if min_depth > max_depth:
            min_depth, max_depth = max_depth, min_depth

        for _ in range(64):
            if tank_geo is not None:
                extents = tank_geo.get_extents()
                if tank_geo.shape_name == 'circular':
                    max_r = max(0.08, min(
                        extents['radius'] - self.c.boundary_buffer,
                        self.c.spread_radius
                    ))
                    min_r = min(0.08, max_r)
                    angle = np.random.uniform(0, 2 * np.pi)
                    radius = np.random.uniform(min_r, max_r)
                    x = radius * np.cos(angle)
                    z = radius * np.sin(angle)
                else:
                    half_w = max(0.08, extents['width'] * 0.5 - self.c.boundary_buffer)
                    half_l = max(0.08, extents['length'] * 0.5 - self.c.boundary_buffer)
                    span_x = min(half_w, self.c.spread_radius)
                    span_z = min(half_l, self.c.spread_radius)
                    x = np.random.uniform(-span_x, span_x)
                    z = np.random.uniform(-span_z, span_z)
            else:
                angle = np.random.uniform(0, 2 * np.pi)
                max_r = max(0.08, min(
                    input_data.tank_radius - self.c.boundary_buffer,
                    self.c.spread_radius
                ))
                min_r = min(0.08, max_r)
                radius = np.random.uniform(min_r, max_r)
                x = radius * np.cos(angle)
                z = radius * np.sin(angle)

            y = 0.0 if keep_surface else np.random.uniform(min_depth, max_depth)
            pos = np.array([x, y, z], dtype=np.float32)

            if tank_geo is not None and not tank_geo.contains_point_xz(float(pos[0]), float(pos[2])):
                continue

            if obs_field is not None and not obs_field.is_valid_position(pos, min_clearance=0.01):
                continue

            return pos

        if tank_geo is not None:
            fallback = tank_geo.random_interior_point(margin=self.c.boundary_buffer).astype(np.float32)
            fallback[1] = 0.0 if keep_surface else np.clip(fallback[1], min_depth, max_depth)
            return fallback
        return np.array([0.0, 0.0 if keep_surface else (min_depth + max_depth) * 0.5, 0.0], dtype=np.float32)

    def _create_floating_pellet(self, input_data: FeedingInput, mass: float) -> FoodItem:
        """Create a floating pellet with random surface drift velocity.

        Args:
            input_data: Tank geometry for position sampling.
            mass: Pellet mass (g).

        Returns:
            A new FoodItem configured as a floating pellet.
        """
        speed = np.random.uniform(self.c.floating_speed_min,
                                  self.c.floating_speed_max)
        vel_angle = np.random.uniform(0, 2 * np.pi)

        return FoodItem(
            position=self._sample_food_position(input_data, keep_surface=True),
            velocity=np.array([speed * np.cos(vel_angle), 0.0, speed * np.sin(vel_angle)],
                              dtype=np.float32),
            mass=mass,
            food_type=FoodType.FLOATING,
            original_type=FoodType.FLOATING,
            max_speed=self.c.floating_speed_max,
            direction_change_interval=self.c.floating_direction_change,
            protein_fraction=self._diet_profile['protein_fraction'],
            lipid_fraction=self._diet_profile['lipid_fraction'],
            carbohydrate_fraction=self._diet_profile['carbohydrate_fraction'],
            adc_protein=self._diet_profile['adc_protein'],
            adc_lipid=self._diet_profile['adc_lipid'],
            adc_carbohydrate=self._diet_profile['adc_carbohydrate'],
            include_carbohydrate_energy=self._diet_profile['include_carbohydrate_energy']
        )

    def _create_sinking_pellet(self, input_data: FeedingInput, mass: float) -> FoodItem:
        """Create a sinking pellet with randomised 3D velocity.

        Args:
            input_data: Tank geometry for position sampling.
            mass: Pellet mass (g).

        Returns:
            A new FoodItem configured as a sinking pellet.
        """
        speed = np.random.uniform(self.c.sinking_speed_min,
                                  self.c.sinking_speed_max)
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(-0.3, 0.3)

        return FoodItem(
            position=self._sample_food_position(input_data, keep_surface=False),
            velocity=np.array([speed * np.cos(theta),
                               speed * np.sin(phi) * self.c.sinking_vertical_factor,
                               speed * np.sin(theta)], dtype=np.float32),
            mass=mass,
            food_type=FoodType.SINKING,
            original_type=FoodType.SINKING,
            max_speed=self.c.sinking_speed_max,
            direction_change_interval=self.c.sinking_direction_change,
            protein_fraction=self._diet_profile['protein_fraction'],
            lipid_fraction=self._diet_profile['lipid_fraction'],
            carbohydrate_fraction=self._diet_profile['carbohydrate_fraction'],
            adc_protein=self._diet_profile['adc_protein'],
            adc_lipid=self._diet_profile['adc_lipid'],
            adc_carbohydrate=self._diet_profile['adc_carbohydrate'],
            include_carbohydrate_energy=self._diet_profile['include_carbohydrate_energy']
        )

    # ------------------------------------------------------------------ #
    # Environmental food creation functions                                #
    # ------------------------------------------------------------------ #

    def _make_nutrient_kwargs(self) -> dict:
        """Return nutrient parameter dictionary (consistent with batch food).

        Returns:
            Dictionary of nutrient fractions and ADC values for FoodItem.
        """
        return dict(
            protein_fraction=self._diet_profile['protein_fraction'],
            lipid_fraction=self._diet_profile['lipid_fraction'],
            carbohydrate_fraction=self._diet_profile['carbohydrate_fraction'],
            adc_protein=self._diet_profile['adc_protein'],
            adc_lipid=self._diet_profile['adc_lipid'],
            adc_carbohydrate=self._diet_profile['adc_carbohydrate'],
            include_carbohydrate_energy=self._diet_profile['include_carbohydrate_energy'],
        )

    def _create_ambient_food(self, input_data: FeedingInput, mass: float) -> FoodItem:
        """Create mid-water drifting food: spawned in the water column, does not sink.

        Args:
            input_data: Tank geometry for position sampling.
            mass: Pellet mass (g).

        Returns:
            A new FoodItem of type AMBIENT.
        """
        pos = self._sample_food_position(input_data, keep_surface=False)
        # Ensure Y is in the mid-water layer (avoids surface or bottom)
        tank_depth = self._get_tank_depth(input_data)
        pos[1] = np.random.uniform(-tank_depth + 0.15, -0.15)

        speed = np.random.uniform(self.c.floating_speed_min, self.c.floating_speed_max)
        angle = np.random.uniform(0, 2 * np.pi)

        return FoodItem(
            position=pos,
            velocity=np.array([speed * np.cos(angle), 0.0, speed * np.sin(angle)],
                               dtype=np.float32),
            mass=mass,
            food_type=FoodType.AMBIENT,
            original_type=FoodType.AMBIENT,
            max_speed=self.c.floating_speed_max,
            direction_change_interval=self.c.floating_direction_change,
            max_age=self.c.ambient_max_age,
            **self._make_nutrient_kwargs()
        )

    def _create_surface_env_food(self, input_data: FeedingInput, mass: float) -> FoodItem:
        """Create surface-drifting environmental food: fixed at y=0, horizontal drift.

        Args:
            input_data: Tank geometry for position sampling.
            mass: Pellet mass (g).

        Returns:
            A new FoodItem of type SURFACE_ENV.
        """
        pos = self._sample_food_position(input_data, keep_surface=True)
        pos[1] = 0.0

        speed = np.random.uniform(self.c.floating_speed_min, self.c.floating_speed_max)
        angle = np.random.uniform(0, 2 * np.pi)

        return FoodItem(
            position=pos,
            velocity=np.array([speed * np.cos(angle), 0.0, speed * np.sin(angle)],
                               dtype=np.float32),
            mass=mass,
            food_type=FoodType.SURFACE_ENV,
            original_type=FoodType.SURFACE_ENV,
            max_speed=self.c.floating_speed_max,
            direction_change_interval=self.c.floating_direction_change,
            max_age=self.c.surface_env_max_age,
            **self._make_nutrient_kwargs()
        )

    def _create_benthic_food(self, input_data: FeedingInput, mass: float) -> FoodItem:
        """Create stationary benthic food resting on the tank floor.

        Args:
            input_data: Tank geometry for position sampling.
            mass: Pellet mass (g).

        Returns:
            A new FoodItem of type BENTHIC.
        """
        tank_depth = self._get_tank_depth(input_data)
        pos = self._sample_food_position(input_data, keep_surface=False)
        pos[1] = -tank_depth + 0.03

        return FoodItem(
            position=pos,
            velocity=np.zeros(3, dtype=np.float32),
            mass=mass,
            food_type=FoodType.BENTHIC,
            original_type=FoodType.BENTHIC,
            max_age=self.c.benthic_max_age,
            **self._make_nutrient_kwargs()
        )

    def _create_attached_food(self, input_data: FeedingInput, mass: float,
                               obstacle_field) -> Optional['FoodItem']:
        """Create food attached to an obstacle surface (0.03 m offset).

        Samples a random obstacle, then attempts to place a stationary food
        item on its outer surface. Falls back to None after 10 failed attempts.

        Args:
            input_data: Tank geometry for boundary validation.
            mass: Pellet mass (g).
            obstacle_field: The obstacle field providing obstacle references.

        Returns:
            A new FoodItem of type ATTACHED, or None if placement fails.
        """
        import random
        tank_geo = getattr(input_data, 'tank_geometry', None)
        tank_depth = self._get_tank_depth(input_data)

        obstacles = getattr(obstacle_field, 'obstacles', [])
        if not obstacles:
            return None

        obs = random.choice(obstacles)
        obs_type = type(obs).__name__

        for _ in range(10):
            pos = None
            if obs_type == 'RockObstacle':
                center = obs.center
                radius = obs.radius
                u = np.random.randn(3).astype(np.float32)
                norm = np.linalg.norm(u)
                if norm < 1e-8:
                    continue
                u /= norm
                pos = center + u * (radius + 0.03)

            elif obs_type == 'BoxObstacle':
                center = obs.center
                hx, hy, hz = obs.half_size
                face = random.randint(0, 5)
                if face == 0:    # +X
                    pos = center + np.array([hx + 0.03,
                                             np.random.uniform(-hy, hy),
                                             np.random.uniform(-hz, hz)], dtype=np.float32)
                elif face == 1:  # -X
                    pos = center + np.array([-hx - 0.03,
                                             np.random.uniform(-hy, hy),
                                             np.random.uniform(-hz, hz)], dtype=np.float32)
                elif face == 2:  # +Y
                    pos = center + np.array([np.random.uniform(-hx, hx),
                                             hy + 0.03,
                                             np.random.uniform(-hz, hz)], dtype=np.float32)
                elif face == 3:  # -Y
                    pos = center + np.array([np.random.uniform(-hx, hx),
                                             -hy - 0.03,
                                             np.random.uniform(-hz, hz)], dtype=np.float32)
                elif face == 4:  # +Z
                    pos = center + np.array([np.random.uniform(-hx, hx),
                                             np.random.uniform(-hy, hy),
                                             hz + 0.03], dtype=np.float32)
                else:            # -Z
                    pos = center + np.array([np.random.uniform(-hx, hx),
                                             np.random.uniform(-hy, hy),
                                             -hz - 0.03], dtype=np.float32)
            else:
                return None

            if pos is None:
                continue

            # Validate: within water volume and legal Y range
            if not (-tank_depth + 0.03 <= pos[1] <= -0.01):
                continue
            if tank_geo is not None:
                if not tank_geo.contains_point_xz(pos[0], pos[2]):
                    continue
            if not obstacle_field.is_valid_position(pos, min_clearance=0.01):
                continue

            return FoodItem(
                position=pos.astype(np.float32),
                velocity=np.zeros(3, dtype=np.float32),
                mass=mass,
                food_type=FoodType.ATTACHED,
                original_type=FoodType.ATTACHED,
                max_age=self.c.attached_max_age,
                **self._make_nutrient_kwargs()
            )

        return None  # All 10 attempts failed

    # ------------------------------------------------------------------ #
    # Environmental food spawn scheduler                                   #
    # ------------------------------------------------------------------ #

    def _spawn_env_food(self, state: FeedingState, input_data: FeedingInput) -> None:
        """Stochastically spawn environmental food each step (for training).

        This does not interfere with the batch feeding logic.

        Args:
            state: Mutable feeding state.
            input_data: Current agent/tank information.
        """
        c = self.c
        env_types = {FoodType.AMBIENT, FoodType.SURFACE_ENV,
                     FoodType.BENTHIC, FoodType.ATTACHED}
        env_count = sum(1 for f in state.food_items if f.food_type in env_types)
        if env_count >= c.env_food_max_count:
            return

        pellet_mass = np.random.uniform(c.pellet_mass_min, c.pellet_mass_max)

        if np.random.random() < c.ambient_spawn_prob:
            state.food_items.append(self._create_ambient_food(input_data, pellet_mass))

        if np.random.random() < c.surface_env_spawn_prob:
            state.food_items.append(self._create_surface_env_food(input_data, pellet_mass))

        if np.random.random() < c.benthic_spawn_prob:
            state.food_items.append(self._create_benthic_food(input_data, pellet_mass))

        obs_field = getattr(input_data, 'obstacle_field', None)
        if (obs_field is not None and
                len(getattr(obs_field, 'obstacles', [])) > 0 and
                np.random.random() < c.attached_spawn_prob):
            food = self._create_attached_food(input_data, pellet_mass, obs_field)
            if food is not None:
                state.food_items.append(food)

    def _update_food_movement(self, state: FeedingState, input_data: FeedingInput) -> None:
        """Update positions and velocities of all food items.

        Handles age-based removal, detection timeout removal, settling
        dynamics, and type-specific movement patterns.

        Args:
            state: Mutable feeding state.
            input_data: Current tank geometry reference.
        """
        to_remove = []

        for i, food in enumerate(state.food_items):
            food.age += 1
            food.direction_change_timer += 1

            # Lifespan check (only applies to environmental food with finite max_age)
            if food.max_age > 0 and food.age > food.max_age:
                to_remove.append(i)
                continue

            # Detection timeout: disappears if not eaten within timeout after perception
            if food.detected and food.detected_timeout > 0:
                food.detected_timer += 1
                if food.detected_timer >= food.detected_timeout:
                    to_remove.append(i)
                    state.total_food_wasted += 1
                    continue

            if food.is_settling:
                # Settling food is updated with dedicated movement.
                self._update_settling_movement(food, input_data)

                # Remove food after touching bottom.
                tank_depth = self._get_tank_depth(input_data)
                if food.position[1] <= -tank_depth + 0.02:
                    to_remove.append(i)
                    if self.debug:
                        print(f"[feeding] pellet touched bottom and removed")

            elif food.food_type == FoodType.FLOATING:
                self._update_floating_movement(food, input_data)

            elif food.food_type == FoodType.SINKING:
                self._update_sinking_movement(food, input_data)

            elif food.food_type == FoodType.AMBIENT:
                self._update_ambient_movement(food, input_data)

            elif food.food_type == FoodType.SURFACE_ENV:
                self._update_surface_env_movement(food, input_data)

            elif food.food_type in (FoodType.BENTHIC, FoodType.ATTACHED):
                pass  # Stationary; only age counting (handled above)

        # Clean up bottom-touched food.
        for i in reversed(to_remove):
            state.food_items.pop(i)
            state.total_food_wasted += 1

    def _update_settling_movement(self, food: FoodItem, input_data: FeedingInput) -> None:
        """Update movement for a pellet in the settling phase.

        Maintains constant vertical sinking speed with small horizontal
        perturbation. Only horizontal boundaries are enforced (not bottom),
        so the pellet can reach the tank floor for removal.

        Args:
            food: The settling food item.
            input_data: Tank geometry reference.
        """
        # Maintain settling speed
        food.velocity[1] = -food.settle_speed

        # Small horizontal perturbation
        food.velocity[0] += np.random.uniform(-0.001, 0.001)
        food.velocity[2] += np.random.uniform(-0.001, 0.001)

        # Limit horizontal speed
        h_speed = np.sqrt(food.velocity[0] ** 2 + food.velocity[2] ** 2)
        max_h_speed = self.c.settling_horizontal_speed_max
        if h_speed > max_h_speed:
            food.velocity[0] *= max_h_speed / h_speed
            food.velocity[2] *= max_h_speed / h_speed

        # Update position
        food.position += food.velocity

        # Enforce only horizontal boundaries (not bottom!)
        self._enforce_horizontal_boundary_only(food, input_data)

        obs_field = getattr(input_data, 'obstacle_field', None)
        if obs_field is not None:
            col = obs_field.check_collision(food.position, body_radius=0.001)
            if col.collided and col.pushed_position is not None:
                food.position = col.pushed_position.copy()
                food.velocity = obs_field.resolve_collision_velocity(food.velocity, col.normal)

    def _enforce_horizontal_boundary_only(self, food: FoodItem, input_data: FeedingInput) -> None:
        """Enforce only horizontal tank boundaries (allows settling to reach bottom).

        Args:
            food: The food item to constrain horizontally.
            input_data: Tank geometry reference.
        """
        tank_geo = getattr(input_data, 'tank_geometry', None)
        buffer = self.c.boundary_buffer
        if tank_geo is None:
            h_dist = np.sqrt(food.position[0] ** 2 + food.position[2] ** 2)
            max_r = input_data.tank_radius - buffer

            if h_dist > max_r:
                factor = max_r / h_dist * 0.95
                food.position[0] *= factor
                food.position[2] *= factor
                food.velocity[0] *= -0.5
                food.velocity[2] *= -0.5
            return

        extents = tank_geo.get_extents()
        if tank_geo.shape_name == 'circular':
            h_dist = np.sqrt(food.position[0] ** 2 + food.position[2] ** 2)
            max_r = extents['radius'] - buffer
            if h_dist > max_r and h_dist > 1e-8:
                factor = max_r / h_dist * 0.95
                food.position[0] *= factor
                food.position[2] *= factor
                food.velocity[0] *= -0.5
                food.velocity[2] *= -0.5
        else:
            half_w = extents['width'] * 0.5 - buffer
            half_l = extents['length'] * 0.5 - buffer
            if food.position[0] > half_w:
                food.position[0] = half_w
                food.velocity[0] *= -0.5
            elif food.position[0] < -half_w:
                food.position[0] = -half_w
                food.velocity[0] *= -0.5

            if food.position[2] > half_l:
                food.position[2] = half_l
                food.velocity[2] *= -0.5
            elif food.position[2] < -half_l:
                food.position[2] = -half_l
                food.velocity[2] *= -0.5

    def _update_floating_movement(self, food: FoodItem, input_data: FeedingInput) -> None:
        """Update movement for a floating pellet (surface drift).

        Args:
            food: The floating food item.
            input_data: Tank geometry reference.
        """
        if food.direction_change_timer >= food.direction_change_interval:
            food.direction_change_timer = 0
            speed = np.random.uniform(self.c.floating_speed_min,
                                      self.c.floating_speed_max)
            angle = np.random.uniform(0, 2 * np.pi)
            food.velocity[0] = speed * np.cos(angle)
            food.velocity[2] = speed * np.sin(angle)

        food.velocity[0] += np.random.uniform(-0.002, 0.002)
        food.velocity[2] += np.random.uniform(-0.002, 0.002)
        food.velocity[1] = 0

        speed = np.sqrt(food.velocity[0] ** 2 + food.velocity[2] ** 2)
        if speed > food.max_speed:
            food.velocity[0] *= food.max_speed / speed
            food.velocity[2] *= food.max_speed / speed

        food.position += food.velocity
        food.position[1] = 0
        self._enforce_food_boundary(food, input_data, keep_surface=True)

    def _update_sinking_movement(self, food: FoodItem, input_data: FeedingInput) -> None:
        """Update movement for a sinking pellet (3D random walk with drift).

        Args:
            food: The sinking food item.
            input_data: Tank geometry reference.
        """
        if food.direction_change_timer >= food.direction_change_interval:
            food.direction_change_timer = 0
            speed = np.random.uniform(self.c.sinking_speed_min,
                                      self.c.sinking_speed_max)
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(-0.2, 0.2)
            food.velocity[0] = speed * np.cos(theta)
            food.velocity[1] = speed * np.sin(phi) * self.c.sinking_vertical_factor
            food.velocity[2] = speed * np.sin(theta)

        food.velocity += np.random.uniform(-0.003, 0.003, 3).astype(np.float32)
        food.velocity[1] *= 0.8

        speed = np.linalg.norm(food.velocity)
        if speed > food.max_speed:
            food.velocity *= food.max_speed / speed

        food.position += food.velocity
        self._enforce_food_boundary(food, input_data, keep_surface=False)

    def _update_ambient_movement(self, food: FoodItem, input_data: FeedingInput) -> None:
        """Update movement for mid-water ambient food (XZ drift, Y held constant).

        Args:
            food: The ambient food item.
            input_data: Tank geometry reference.
        """
        if food.direction_change_timer >= food.direction_change_interval:
            food.direction_change_timer = 0
            speed = np.random.uniform(self.c.floating_speed_min,
                                      self.c.floating_speed_max)
            angle = np.random.uniform(0, 2 * np.pi)
            food.velocity[0] = speed * np.cos(angle)
            food.velocity[2] = speed * np.sin(angle)

        food.velocity[0] += np.random.uniform(-0.002, 0.002)
        food.velocity[2] += np.random.uniform(-0.002, 0.002)
        # Only minimal Y perturbation; no cumulative sinking
        food.velocity[1] = np.random.uniform(-0.0005, 0.0005)

        h_speed = np.sqrt(food.velocity[0] ** 2 + food.velocity[2] ** 2)
        if h_speed > food.max_speed:
            food.velocity[0] *= food.max_speed / h_speed
            food.velocity[2] *= food.max_speed / h_speed

        food.position += food.velocity

        # Y boundary: prevent drifting to surface or touching bottom
        tank_depth = self._get_tank_depth(input_data)
        food.position[1] = np.clip(food.position[1], -tank_depth + 0.1, -0.1)
        food.velocity[1] = 0.0

        self._enforce_food_boundary(food, input_data, keep_surface=False)

    def _update_surface_env_movement(self, food: FoodItem, input_data: FeedingInput) -> None:
        """Update movement for surface environmental food (XZ drift, Y fixed at 0).

        Args:
            food: The surface environmental food item.
            input_data: Tank geometry reference.
        """
        if food.direction_change_timer >= food.direction_change_interval:
            food.direction_change_timer = 0
            speed = np.random.uniform(self.c.floating_speed_min,
                                      self.c.floating_speed_max)
            angle = np.random.uniform(0, 2 * np.pi)
            food.velocity[0] = speed * np.cos(angle)
            food.velocity[2] = speed * np.sin(angle)

        food.velocity[0] += np.random.uniform(-0.002, 0.002)
        food.velocity[2] += np.random.uniform(-0.002, 0.002)
        food.velocity[1] = 0.0

        h_speed = np.sqrt(food.velocity[0] ** 2 + food.velocity[2] ** 2)
        if h_speed > food.max_speed:
            food.velocity[0] *= food.max_speed / h_speed
            food.velocity[2] *= food.max_speed / h_speed

        food.position += food.velocity
        food.position[1] = 0.0
        self._enforce_food_boundary(food, input_data, keep_surface=True)

    def _enforce_food_boundary(self, food: FoodItem, input_data: FeedingInput,
                               keep_surface: bool = False) -> None:
        """Enforce full tank boundary constraints (used for non-settling food).

        Args:
            food: The food item to constrain.
            input_data: Tank geometry reference.
            keep_surface: If True, clamp Y to 0 (water surface).
        """
        self._enforce_horizontal_boundary_only(food, input_data)

        tank_depth = self._get_tank_depth(input_data)
        if keep_surface:
            food.position[1] = 0
            food.velocity[1] = 0.0
        else:
            # Depth limits (only for non-settling state)
            min_y = -tank_depth + self.c.bottom_buffer
            max_y = -self.c.surface_buffer
            if food.position[1] < min_y:
                food.position[1] = min_y
                food.velocity[1] = abs(food.velocity[1]) * 0.5
            elif food.position[1] > max_y:
                food.position[1] = max_y
                food.velocity[1] = -abs(food.velocity[1]) * 0.5

        obs_field = getattr(input_data, 'obstacle_field', None)
        if obs_field is not None:
            col = obs_field.check_collision(food.position, body_radius=0.001)
            if col.collided and col.pushed_position is not None:
                food.position = col.pushed_position.copy()
                food.velocity = obs_field.resolve_collision_velocity(food.velocity, col.normal)
                if keep_surface:
                    food.position[1] = 0
                    food.velocity[1] = 0.0

    def _check_consumption(self, state: FeedingState,
                           input_data: FeedingInput,
                           capture_radius_multiplier: float = 1.0) -> Dict[str, Any]:
        """Check whether the agent captures any food items this step.

        Uses a combination of direct-hit (within capture radius and mouth
        cone) and sweep-hit (along movement trajectory) detection. Stomach
        capacity is enforced as a hard constraint.

        Args:
            state: Mutable feeding state.
            input_data: Agent position, heading, and stomach fullness.
            capture_radius_multiplier: Multiplier for the capture radius.

        Returns:
            Dictionary with consumption statistics for this step.
        """
        result = {
            'count': 0, 'mass': 0.0, 'fullness_gained': 0.0,
            'energy_potential': 0.0, 'floating_consumed': 0, 'sinking_consumed': 0,
            'ambient_consumed': 0, 'surface_env_consumed': 0,
            'benthic_consumed': 0, 'attached_consumed': 0
        }

        capture_radius = calculate_capture_radius(input_data.agent_length) * capture_radius_multiplier
        stomach_capacity = calculate_stomach_capacity(input_data.agent_mass)
        current = (input_data.stomach_fullness / 100) * stomach_capacity
        agent_pos = input_data.agent_position
        prev_pos = input_data.agent_prev_position
        if prev_pos is None:
            prev_pos = agent_pos

        heading = input_data.agent_heading
        pitch_angle = input_data.agent_pitch_angle
        movement = agent_pos - prev_pos
        movement_norm = np.linalg.norm(movement)
        if heading is None and movement_norm > 1e-6:
            heading = movement / movement_norm

        # Compose horizontal heading + pitch angle into true 3D mouth direction
        mouth_dir_3d = self._compute_mouth_direction_3d(heading, pitch_angle)

        consumed_indices = []

        for i, food in enumerate(state.food_items):
            if current + food.mass > stomach_capacity:
                continue

            distance = np.linalg.norm(food.position - agent_pos)
            in_mouth_cone = self._is_in_mouth_cone(agent_pos, food.position, mouth_dir_3d)
            direct_hit = distance <= capture_radius and in_mouth_cone

            sweep_hit = False
            if movement_norm > 1e-6:
                sweep_dist = self._distance_point_to_segment_3d(food.position, prev_pos, agent_pos)
                sweep_hit = sweep_dist <= capture_radius * 0.85 and in_mouth_cone

            if direct_hit or sweep_hit:
                consumed_indices.append(i)
                current += food.mass

                fullness = (food.mass / stomach_capacity) * 100
                result['fullness_gained'] += fullness
                result['energy_potential'] += (
                    food.mass * self._calculate_item_digestible_energy_density(food)
                )
                result['count'] += 1
                result['mass'] += food.mass

                if food.original_type == FoodType.FLOATING:
                    result['floating_consumed'] += 1
                elif food.original_type == FoodType.SINKING:
                    result['sinking_consumed'] += 1
                elif food.original_type == FoodType.AMBIENT:
                    result['ambient_consumed'] += 1
                elif food.original_type == FoodType.SURFACE_ENV:
                    result['surface_env_consumed'] += 1
                elif food.original_type == FoodType.BENTHIC:
                    result['benthic_consumed'] += 1
                elif food.original_type == FoodType.ATTACHED:
                    result['attached_consumed'] += 1

        for i in reversed(consumed_indices):
            eaten = state.food_items.pop(i)
            state.total_food_eaten += 1
            state.total_mass_eaten += eaten.mass
            if eaten.original_type == FoodType.FLOATING:
                state.floating_food_eaten += 1
            else:
                state.sinking_food_eaten += 1

        return result

    @staticmethod
    def _compute_mouth_direction_3d(heading: Optional[np.ndarray],
                                    pitch_angle_deg: float) -> Optional[np.ndarray]:
        """Compose horizontal heading with pitch angle into a 3D mouth direction.

        The heading is a horizontal unit vector [hx, 0, hz]. The pitch angle
        (degrees; positive = head raised, negative = head lowered) rotates
        this vector in the vertical plane.

        Args:
            heading: Horizontal heading unit vector (may be None).
            pitch_angle_deg: Fish head pitch angle in degrees.

        Returns:
            Normalised 3D mouth direction vector, or None if heading is
            unavailable or degenerate.
        """
        if heading is None:
            return None
        heading_norm = np.linalg.norm(heading)
        if heading_norm < 1e-6:
            return None

        h = heading / heading_norm
        pitch_rad = np.radians(pitch_angle_deg)
        cos_p = np.cos(pitch_rad)
        sin_p = np.sin(pitch_rad)  # positive pitch = head raised -> positive Y component

        # Horizontal component scaled by cos(pitch), vertical component is sin(pitch)
        mouth_dir = np.array([
            h[0] * cos_p,
            sin_p,
            h[2] * cos_p
        ], dtype=np.float32)
        norm = np.linalg.norm(mouth_dir)
        if norm < 1e-6:
            return None
        return mouth_dir / norm

    @staticmethod
    def _distance_point_to_segment_3d(point: np.ndarray, seg_start: np.ndarray,
                                      seg_end: np.ndarray) -> float:
        """Minimum distance between a point and a 3D line segment.

        Args:
            point: Query point (3D).
            seg_start: Segment start point.
            seg_end: Segment end point.

        Returns:
            Euclidean distance from point to the nearest location on the segment.
        """
        segment = seg_end - seg_start
        seg_len_sq = float(np.dot(segment, segment))
        if seg_len_sq < 1e-12:
            return float(np.linalg.norm(point - seg_start))

        t = float(np.dot(point - seg_start, segment) / seg_len_sq)
        t = np.clip(t, 0.0, 1.0)
        projection = seg_start + t * segment
        return float(np.linalg.norm(point - projection))

    @staticmethod
    def _is_in_mouth_cone(agent_pos: np.ndarray, target_pos: np.ndarray,
                          heading: Optional[np.ndarray],
                          cos_half_angle: float = 0.20) -> bool:
        """Test whether a target lies within the frontal capture cone.

        Args:
            agent_pos: Current agent position (3D).
            target_pos: Target food position (3D).
            heading: 3D mouth direction vector (may be None for omnidirectional).
            cos_half_angle: Cosine of the cone half-angle threshold.
                Default 0.20 corresponds to approximately 78.5 degrees
                (wide but directional).

        Returns:
            True if the target is within the capture cone or heading is
            unavailable.
        """
        if heading is None:
            return True

        heading_norm = np.linalg.norm(heading)
        if heading_norm < 1e-6:
            return True

        to_target = target_pos - agent_pos
        dist = np.linalg.norm(to_target)
        if dist < 1e-6:
            return True

        to_target = to_target / dist
        heading_dir = heading / heading_norm
        return float(np.dot(heading_dir, to_target)) >= cos_half_angle

    def initial_feeding(self, state: FeedingState, input_data: FeedingInput,
                        num_pellets: int = 10, multiplier: float = 1.0) -> None:
        """Perform the initial food placement at episode start.

        Spawns a mix of floating and sinking pellets with randomised ages
        for the floating fraction.

        Args:
            state: Mutable feeding state (will be cleared and repopulated).
            input_data: Agent mass and tank geometry.
            num_pellets: Total number of pellets to spawn.
            multiplier: Curriculum multiplier for total food mass.
        """
        state.food_items = []
        fish_mass = input_data.agent_mass
        intake_scaling = self._mass_intake_scaling_factor(fish_mass)
        daily_total = fish_mass * self.c.daily_feeding_rate * multiplier * intake_scaling
        target_mass = daily_total / self.c.feedings_per_day
        pellet_mass = np.clip(target_mass / num_pellets,
                              self.c.pellet_mass_min,
                              self.c.pellet_mass_max)

        num_floating = num_pellets // 3
        num_sinking = num_pellets - num_floating

        for _ in range(num_floating):
            food = self._create_floating_pellet(input_data, pellet_mass)
            food.age = np.random.randint(0, self.settle_trigger_step // 3)
            state.food_items.append(food)

        for _ in range(num_sinking):
            food = self._create_sinking_pellet(input_data, pellet_mass)
            state.food_items.append(food)

        state.total_food_spawned += num_pellets
        state.total_mass_spawned += pellet_mass * num_pellets
        state.floating_food_spawned += num_floating
        state.sinking_food_spawned += num_sinking

    def initial_feeding_random_depth(self, state: FeedingState, input_data: FeedingInput,
                                     num_pellets: int = 10, multiplier: float = 1.0) -> None:
        """Course 1/2 initial feeding: pellets uniformly distributed across full depth.

        Biological rationale: in real aquaculture, pellets entering the water
        from the surface disperse across various depths at different settling
        rates. Using full-depth random placement during training forces the
        agent to learn 3D search behaviour rather than fixating on the surface.

        All pellets are created as sinking type (random depth Y); no
        floating/sinking distinction is made.

        Args:
            state: Mutable feeding state (will be cleared and repopulated).
            input_data: Agent mass and tank geometry.
            num_pellets: Total number of pellets to spawn.
            multiplier: Curriculum multiplier for total food mass.
        """
        state.food_items = []
        fish_mass = input_data.agent_mass
        intake_scaling = self._mass_intake_scaling_factor(fish_mass)
        daily_total = fish_mass * self.c.daily_feeding_rate * multiplier * intake_scaling
        target_mass = daily_total / self.c.feedings_per_day
        pellet_mass = np.clip(target_mass / num_pellets,
                              self.c.pellet_mass_min,
                              self.c.pellet_mass_max)

        for _ in range(num_pellets):
            food = self._create_sinking_pellet(input_data, pellet_mass)
            state.food_items.append(food)

        state.total_food_spawned += num_pellets
        state.total_mass_spawned += pellet_mass * num_pellets
        state.sinking_food_spawned += num_pellets

    def initial_feeding_course2(self, state: FeedingState, input_data: FeedingInput,
                                num_pellets: int = 10, scene: str = 'maze') -> None:
        """Course 2 initial feeding: food placement varies by sub-scene type.

        Args:
            state: Mutable feeding state (will be cleared and repopulated).
            input_data: Agent mass, position, and tank/obstacle geometry.
            num_pellets: Total number of pellets to spawn.
            scene: Sub-scene identifier controlling placement strategy:
                'maze'  -- Corridor maze: food uniformly scattered, some
                    placed on the far side of corridors (requires traversal).
                'reef'  -- Rocky reef: 80% food attached near rock surfaces
                    (stationary), 20% floating pellets drifting.
                'open'  -- Open search: food biased to the hemisphere
                    opposite the spawn point, forcing active exploration.
        """
        state.food_items = []
        fish_mass = input_data.agent_mass
        intake_scaling = self._mass_intake_scaling_factor(fish_mass)
        daily_total = fish_mass * self.c.daily_feeding_rate * intake_scaling
        target_mass = daily_total / self.c.feedings_per_day
        pellet_mass = np.clip(target_mass / max(num_pellets, 1),
                              self.c.pellet_mass_min,
                              self.c.pellet_mass_max)

        obs_field = getattr(input_data, 'obstacle_field', None)
        tank_geo = getattr(input_data, 'tank_geometry', None)
        tank_depth = self._get_tank_depth(input_data)
        agent_pos = input_data.agent_position

        if scene == 'reef' and obs_field is not None and obs_field.count > 0:
            # Reef scene: 80% food near rock surfaces, 20% floating
            n_reef = int(num_pellets * 0.80)
            n_float = num_pellets - n_reef

            obstacles = obs_field.obstacles
            for _ in range(n_reef):
                # Pick a random obstacle and place stationary food near its surface
                obs = obstacles[np.random.randint(len(obstacles))]
                for attempt in range(30):
                    # Sample at 3-8 cm offset from obstacle surface
                    direction = np.random.randn(3).astype(np.float32)
                    direction /= (np.linalg.norm(direction) + 1e-8)
                    offset_dist = obs.radius + np.random.uniform(0.03, 0.08)
                    pos = obs.center + direction * offset_dist
                    # Validate: inside tank and legal Y
                    if tank_geo is not None and not tank_geo.contains_point_xz(
                            float(pos[0]), float(pos[2])):
                        continue
                    pos[1] = float(np.clip(pos[1],
                                           -tank_depth + self.c.bottom_buffer + 0.05,
                                           -self.c.surface_buffer - 0.05))
                    if obs_field.is_valid_position(pos, min_clearance=0.01):
                        break
                else:
                    # Fallback to ordinary random position
                    pos = self._sample_food_position(input_data, keep_surface=False)

                # Stationary food: zero velocity, already settled
                food = FoodItem(
                    position=pos.astype(np.float32),
                    velocity=np.zeros(3, dtype=np.float32),
                    mass=pellet_mass,
                    food_type=FoodType.SETTLING,
                    original_type=FoodType.SINKING,
                    max_speed=0.0,
                    direction_change_interval=9999,
                    protein_fraction=self._diet_profile['protein_fraction'],
                    lipid_fraction=self._diet_profile['lipid_fraction'],
                    carbohydrate_fraction=self._diet_profile['carbohydrate_fraction'],
                    adc_protein=self._diet_profile['adc_protein'],
                    adc_lipid=self._diet_profile['adc_lipid'],
                    adc_carbohydrate=self._diet_profile['adc_carbohydrate'],
                    include_carbohydrate_energy=self._diet_profile['include_carbohydrate_energy']
                )
                food.age = 9999  # Already settled; will not sink further
                food.is_settling = True
                state.food_items.append(food)
                state.sinking_food_spawned += 1

            # Small number of floating pellets
            for _ in range(n_float):
                food = self._create_floating_pellet(input_data, pellet_mass)
                state.food_items.append(food)
                state.floating_food_spawned += 1

        elif scene == 'open' and agent_pos is not None:
            # Open search scene: food placed on the far side from spawn point
            # Invert spawn direction -> food in opposite hemisphere
            away_dir_xz = -agent_pos[[0, 2]]
            away_norm = np.linalg.norm(away_dir_xz)
            if away_norm > 0.01:
                away_dir_xz = away_dir_xz / away_norm
            else:
                away_dir_xz = np.array([1.0, 0.0])

            placed = 0
            for _ in range(num_pellets * 5):
                if placed >= num_pellets:
                    break
                pos = self._sample_food_position(input_data, keep_surface=False)
                # Food must be on the opposite side (dot product > 0 = same side as away)
                to_food_xz = pos[[0, 2]] - agent_pos[[0, 2]]
                if np.dot(to_food_xz, away_dir_xz) > 0:
                    # Also require sufficient distance (>40% of tank radius)
                    extents = tank_geo.get_extents() if tank_geo else {}
                    min_dist = extents.get('radius', 1.0) * 0.40
                    if np.linalg.norm(to_food_xz) >= min_dist:
                        food = self._create_sinking_pellet(input_data, pellet_mass)
                        food.position = pos
                        state.food_items.append(food)
                        state.sinking_food_spawned += 1
                        placed += 1

            # Fill remaining quota with random positions
            while placed < num_pellets:
                food = self._create_sinking_pellet(input_data, pellet_mass)
                state.food_items.append(food)
                state.sinking_food_spawned += 1
                placed += 1

        else:
            # Maze scene (default): uniformly scattered with reduced floating ratio
            # to encourage the agent to dive and explore
            n_float = max(1, num_pellets // 5)
            n_sink = num_pellets - n_float
            for _ in range(n_float):
                food = self._create_floating_pellet(input_data, pellet_mass)
                state.food_items.append(food)
                state.floating_food_spawned += 1
            for _ in range(n_sink):
                food = self._create_sinking_pellet(input_data, pellet_mass)
                state.food_items.append(food)
                state.sinking_food_spawned += 1

        state.total_food_spawned += len(state.food_items)
        state.total_mass_spawned += pellet_mass * len(state.food_items)

    def get_food_count(self, state: FeedingState) -> int:
        """Get the current number of food items in the environment.

        Args:
            state: Current feeding state.

        Returns:
            Number of active food items.
        """
        return len(state.food_items)

    def get_food_positions(self, state: FeedingState) -> List[Dict]:
        """Get position and metadata for all active food items.

        Args:
            state: Current feeding state.

        Returns:
            List of dictionaries with keys: position, velocity, mass,
            food_type, is_settling.
        """
        return [
            {
                'position': f.position.copy(),
                'velocity': f.velocity.copy(),
                'mass': f.mass,
                'food_type': f.original_type.value,
                'is_settling': f.is_settling
            }
            for f in state.food_items
        ]

    def get_statistics(self, state: FeedingState) -> Dict[str, Any]:
        """Get summary statistics for the feeding subsystem.

        Args:
            state: Current feeding state.

        Returns:
            Dictionary with current food counts, totals spawned/eaten/wasted,
            and timing information.
        """
        floating = sum(1 for f in state.food_items
                       if f.original_type == FoodType.FLOATING and not f.is_settling)
        sinking = sum(1 for f in state.food_items
                      if f.original_type == FoodType.SINKING and not f.is_settling)
        settling = sum(1 for f in state.food_items if f.is_settling)

        return {
            'current_food': len(state.food_items),
            'floating': floating,
            'sinking': sinking,
            'settling': settling,
            'total_spawned': state.total_food_spawned,
            'total_eaten': state.total_food_eaten,
            'total_wasted': state.total_food_wasted,
            'steps_since_feeding': state.steps_since_feeding,
            'settle_trigger_at': self.settle_trigger_step,
            'feeding_interval': self.feeding_interval,
        }

    def set_debug(self, enabled: bool) -> None:
        """Enable or disable debug printing.

        Args:
            enabled: If True, verbose debug messages are printed to stdout.
        """
        self.debug = enabled


# ============================================================
# Factory functions
# ============================================================

def create_feeding_system() -> FeedingSystem:
    """Create and return a new FeedingSystem instance.

    Returns:
        Initialised FeedingSystem with parameters from CONFIG.
    """
    return FeedingSystem()


def create_feeding_state() -> FeedingState:
    """Create and return a new empty FeedingState.

    Returns:
        Default-initialised FeedingState.
    """
    return FeedingState()


# ============================================================
# Module self-test
# ============================================================

if __name__ == "__main__":
    # Keep __main__ minimal to avoid encoding-related print issues.
    print("FeedingSystem module loaded.")
