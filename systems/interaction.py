#!/usr/bin/env python3
"""
Conspecific Interaction Subsystem (v4)
======================================

This module implements the inter-individual interaction mechanics for a
reinforcement-learning-based largemouth bass (Micropterus salmoides)
behavioural simulation. It manages four categories of non-player-character
(NPC) fish: aggressive conspecifics, surface predators, passive schooling
fish, and fleeing prey. The subsystem handles ecosystem initialization,
predation event resolution, escape probability calculation, and spatial
boundary enforcement for all NPC entities.

Key features introduced in v4:
    1. Attack cooldown mechanism -- aggressors and surface predators must
       wait a configurable number of time-steps between successive strikes.
    2. Increased body-mass loss scaling upon successful attacks.
    3. Attacker metadata propagation to the reward signal.

Modifications from v3:
    - OtherFish gains an ``attack_cooldown`` field.
    - Successful attacks trigger a cooldown timer.
    - During cooldown the NPC reverts to idle behaviour (no chase/attack).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CONFIG

from utils.biological_formulas import (
    mass_to_length,
    length_to_mass,
    calculate_capture_radius,
    calculate_sustained_speed,
    calculate_strike_range,
    calculate_fish_energy_density
)

# ============================================================
# Enumerations
# ============================================================

class FishBehaviorType(Enum):
    PASSIVE = "passive"
    AGGRESSIVE = "aggressive"
    SURFACE_PREDATOR = "surface"
    FLEEING = "fleeing"


class FishSize(Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    PREDATOR = "predator"


# ============================================================
# Behaviour Configuration
# ============================================================

@dataclass(frozen=True)
class AggressiveBehaviorConfig:
    """Configuration for aggressive conspecific attack behaviour."""
    detection_range: float = 0.25
    chase_speed_multiplier: float = 0.18
    random_speed_multiplier: float = 0.4
    give_up_range: float = 0.35
    direction_noise: float = 0.15
    attack_cooldown_steps: int = 30
    attack_range: float = 0.15


@dataclass(frozen=True)
class SurfacePredatorConfig:
    """Configuration for surface predator behaviour."""
    surface_zone_depth: float = 0.15
    surface_zone_max: float = 0.15
    detection_range: float = 2.50
    chase_speed_multiplier: float = 0.16
    patrol_speed_multiplier: float = 0.4
    attack_damage: float = 2.0
    attack_cooldown_steps: int = 30
    attack_range: float = 0.2
    prey_detection_depth: float = 0.15        # Depth threshold (m) within which the agent is detectable
    give_up_depth: float = 0.25               # Depth (m) beyond which the predator abandons pursuit
    direction_noise: float = 0.1              # Stochastic noise added to chase direction
    vertical_chase_limit: float = 0.05        # Maximum vertical displacement (m) during pursuit


@dataclass(frozen=True)
class FleeingBehaviorConfig:
    """Configuration for prey fleeing behaviour."""
    detection_range: float = 0.16
    flee_speed_multiplier: float = 0.19
    safe_distance: float = 0.50
    flee_duration_max: int = 25
    direction_noise: float = 0.15


AGGRESSIVE_CONFIG = AggressiveBehaviorConfig()
SURFACE_PREDATOR_CONFIG = SurfacePredatorConfig()
FLEEING_CONFIG = FleeingBehaviorConfig()


# ============================================================
# Data Classes
# ============================================================

@dataclass
class OtherFish:
    """NPC fish entity with attack cooldown (v4).

    Attributes:
        position: 3-D world position [x, y, z] in metres.
        velocity: 3-D velocity vector [vx, vy, vz] in m/s.
        body_mass: Wet body mass in grams.
        total_length: Total body length in metres.
        energy: Abstract energy reserve (arbitrary units).
        is_alive: Whether this entity is still in the simulation.
        behavior_type: Current behavioural state.
        original_behavior: Behaviour assigned at initialization.
        size_category: Categorical size class.
        is_chasing: Whether the NPC is actively pursuing the agent.
        chase_target: Last known position of the chase target.
        random_direction_timer: Counter for idle swim direction changes.
        is_fleeing: Whether the NPC is fleeing from the agent.
        flee_direction: Unit vector of current flee trajectory.
        flee_duration: Steps elapsed since flee onset.
        attack_cooldown: Remaining cooldown steps before next attack.
        total_attacks: Cumulative attack count (for statistics).
        successful_attacks: Cumulative successful attack count.
    """
    position: np.ndarray
    velocity: np.ndarray
    body_mass: float
    total_length: float
    energy: float = 100.0
    is_alive: bool = True

    behavior_type: FishBehaviorType = FishBehaviorType.PASSIVE
    original_behavior: FishBehaviorType = FishBehaviorType.PASSIVE
    size_category: FishSize = FishSize.SMALL
    is_chasing: bool = False
    chase_target: Optional[np.ndarray] = None
    random_direction_timer: int = 0

    is_fleeing: bool = False
    flee_direction: Optional[np.ndarray] = None
    flee_duration: int = 0

    # Attack cooldown fields
    attack_cooldown: int = 0  # Remaining cooldown steps
    total_attacks: int = 0  # Total attack count (statistics)
    successful_attacks: int = 0  # Successful attack count

    def __post_init__(self):
        if self.total_length == 0:
            self.total_length = mass_to_length(self.body_mass)
        self.original_behavior = self.behavior_type

    def can_attack(self) -> bool:
        """Check whether the cooldown has expired and attack is permitted.

        Returns:
            True if cooldown is zero or negative (attack allowed).
        """
        return self.attack_cooldown <= 0

    def start_cooldown(self, cooldown_steps: int) -> None:
        """Initiate the post-attack cooldown timer.

        Args:
            cooldown_steps: Number of simulation steps to wait before the
                next attack is permitted.
        """
        self.attack_cooldown = cooldown_steps
        self.total_attacks += 1

    def update_cooldown(self) -> None:
        """Decrement the cooldown timer by one step."""
        if self.attack_cooldown > 0:
            self.attack_cooldown -= 1


@dataclass
class InteractionState:
    """Mutable state container for the interaction subsystem (v4).

    Attributes:
        other_fish: List of all NPC fish entities.
        predation_attempts: Cumulative agent predation attempts.
        predation_successes: Cumulative successful predations by agent.
        escape_attempts: Cumulative escape attempts by agent.
        escape_successes: Cumulative successful escapes by agent.
        damage_taken: Total damage accumulated by the agent.
        aggressive_fish_count: Number of aggressive NPCs initialized.
        surface_predator_present: Whether a surface predator exists.
        times_chased: Number of steps in which agent was chased.
        surface_zone_entries: Steps agent spent in the surface danger zone.
        small_fish_flee_count: Number of flee events triggered by small fish.
        total_attacks_received: Total attacks received over all episodes.
        attacks_this_episode: Attacks received in the current episode.
    """
    other_fish: List[OtherFish] = field(default_factory=list)
    predation_attempts: int = 0
    predation_successes: int = 0
    escape_attempts: int = 0
    escape_successes: int = 0
    damage_taken: float = 0.0

    aggressive_fish_count: int = 0
    surface_predator_present: bool = False
    times_chased: int = 0
    surface_zone_entries: int = 0
    small_fish_flee_count: int = 0

    # Attack statistics
    total_attacks_received: int = 0  # Total attacks received across episodes
    attacks_this_episode: int = 0  # Attacks received in current episode


@dataclass
class InteractionInput:
    """Input data bundle passed to the interaction subsystem each step.

    Attributes:
        agent_position: Agent 3-D position [x, y, z] in metres.
        agent_velocity: Agent 3-D velocity [vx, vy, vz] in m/s.
        agent_mass: Agent body mass in grams.
        agent_length: Agent total length in metres.
        agent_heading: Optional unit heading vector.
        agent_is_burst: Whether the agent is in burst-swim mode.
        agent_fatigue: Current fatigue level (arbitrary units).
        agent_stress: Current stress level (0-1 normalized).
        tank_radius: Radius of the cylindrical tank in metres.
        tank_depth: Depth of the tank in metres.
        tank_geometry: Optional TankGeometry instance for non-cylindrical tanks.
        obstacle_field: Optional ObstacleField instance for structure interactions.
    """
    agent_position: np.ndarray
    agent_velocity: np.ndarray
    agent_mass: float
    agent_length: float
    agent_heading: Optional[np.ndarray] = None
    agent_is_burst: bool = False
    agent_fatigue: float = 0.0
    agent_stress: float = 0.0
    tank_radius: float = field(default_factory=lambda: CONFIG.environment.tank_radius)
    tank_depth: float = field(default_factory=lambda: CONFIG.environment.tank_depth)
    tank_geometry: Any = None      # TankGeometry instance
    obstacle_field: Any = None     # ObstacleField instance

@dataclass
class InteractionOutput:
    """Output data bundle returned by the interaction subsystem each step (v4).

    Attributes:
        predation_success: Number of successful predation events this step.
        energy_gained: Total energy gained from predation (joules).
        mass_gained: Total prey mass consumed (grams).
        escape_success: Number of successful escape events this step.
        damage_taken: Damage received from attacks this step.
        fish_removed: Number of NPC fish removed (consumed) this step.
        being_chased: Whether the agent is currently being chased.
        in_surface_danger_zone: Whether the agent is in the surface zone.
        chase_count: Number of NPCs currently chasing the agent.
        prey_fleeing_count: Number of prey NPCs currently fleeing.
        attacker_mass: Body mass of the most recent attacker (grams).
        attack_count_this_step: Number of attacks received this step.
    """
    predation_success: int = 0
    energy_gained: float = 0.0
    mass_gained: float = 0.0
    escape_success: int = 0
    damage_taken: float = 0.0
    fish_removed: int = 0

    being_chased: bool = False
    in_surface_danger_zone: bool = False
    chase_count: int = 0
    prey_fleeing_count: int = 0

    # Attacker metadata
    attacker_mass: float = 0.0  # Attacker body mass
    attack_count_this_step: int = 0  # Number of attacks this step


# ============================================================
# Interaction System Core Class (v4)
# ============================================================

class InteractionSystem:
    """Core interaction subsystem with attack cooldown mechanics (v4).

    Manages NPC fish AI updates, predation resolution, escape mechanics,
    and spatial boundary enforcement within the simulation environment.
    """

    def __init__(self) -> None:
        self.c = CONFIG.interaction
        self.env = CONFIG.environment
        self.aggressive_cfg = AGGRESSIVE_CONFIG
        self.surface_cfg = SURFACE_PREDATOR_CONFIG
        self.fleeing_cfg = FLEEING_CONFIG
        self.debug = False

        self.aggressive_fish_min = 1
        self.aggressive_fish_max = 1
        self.enable_surface_predator = True

    def initialize_ecosystem(self, state: InteractionState,
                             tank_radius: Optional[float] = None,
                             tank_depth: Optional[float] = None,
                             tank_geometry: Any = None,
                             obstacle_field: Any = None,
                             agent_mass: float = 20.0,
                             small_fish_count: Optional[int] = None,
                             medium_fish_count: Optional[int] = None,
                             aggressive_count_range: Optional[Tuple[int, int]] = None,
                             enable_surface_predator: Optional[bool] = None) -> None:
        """Initialize the NPC ecosystem at the start of an episode.

        Spawns passive prey, medium conspecifics, aggressive conspecifics,
        and an optional surface predator. NPC body masses are scaled
        relative to the agent mass following biological size ratios for
        largemouth bass.

        Args:
            state: Mutable interaction state to populate.
            tank_radius: Tank radius in metres (overrides config).
            tank_depth: Tank depth in metres (overrides config).
            tank_geometry: Optional TankGeometry instance.
            obstacle_field: Optional ObstacleField instance.
            agent_mass: Agent body mass in grams for NPC scaling.
            small_fish_count: Number of small prey fish (overrides config).
            medium_fish_count: Number of medium fish (overrides config).
            aggressive_count_range: (min, max) tuple for aggressive fish count.
            enable_surface_predator: Whether to spawn a surface predator.
        """
        if tank_geometry is not None:
            extents = tank_geometry.get_extents()
            tank_radius = extents.get('radius', extents.get('width', 3.0) / 2)
            tank_depth = extents.get('depth', 0.8)
        else:
            if tank_radius is None:
                tank_radius = self.env.tank_radius
            if tank_depth is None:
                tank_depth = self.env.tank_depth

        state.other_fish = []
        state.attacks_this_episode = 0

        small_count = self.c.small_fish_count if small_fish_count is None else max(0, int(small_fish_count))
        medium_count = self.c.medium_fish_count if medium_fish_count is None else max(0, int(medium_fish_count))

        if aggressive_count_range is None:
            aggr_min = self.aggressive_fish_min
            aggr_max = self.aggressive_fish_max
        else:
            aggr_min = max(0, int(aggressive_count_range[0]))
            aggr_max = max(aggr_min, int(aggressive_count_range[1]))

        surface_enabled = self.enable_surface_predator if enable_surface_predator is None else bool(enable_surface_predator)

        # ── NPC size ranges scaled to agent body mass ─────────────────────
        # Biological rationale (largemouth bass):
        #   small prey  : 5-20 % agent mass  (easily caught)
        #   medium fish : 30-70 % agent mass (neutral / competition)
        #   aggressive  : 150-400 % agent mass (credible threat)
        #   surface pred: 400-1000 % agent mass (apex predator)
        # All ranges are clamped so values stay physically plausible.
        _sm_lo  = float(np.clip(agent_mass * 0.05,   0.1,  50.0))
        _sm_hi  = float(np.clip(agent_mass * 0.20,   _sm_lo * 1.5, 100.0))
        _md_lo  = float(np.clip(agent_mass * 0.30,   0.5,  150.0))
        _md_hi  = float(np.clip(agent_mass * 0.70,   _md_lo * 1.5, 300.0))
        _ag_lo  = float(np.clip(agent_mass * 1.50,   2.0,  500.0))
        _ag_hi  = float(np.clip(agent_mass * 4.00,   _ag_lo * 1.5, 800.0))
        _pr_lo  = float(np.clip(agent_mass * 4.00,   5.0,  800.0))
        _pr_hi  = float(np.clip(agent_mass * 10.0,   _pr_lo * 1.5, 1500.0))
        # ──────────────────────────────────────────────────────────────────

        # 1. Small prey fish
        for _ in range(small_count):
            fish = self._create_passive_fish(
                mass_range=(_sm_lo, _sm_hi),
                size_category=FishSize.SMALL,
                tank_radius=tank_radius,
                tank_depth=tank_depth
            )
            state.other_fish.append(fish)

        # 2. Medium conspecifics
        for _ in range(medium_count):
            fish = self._create_passive_fish(
                mass_range=(_md_lo, _md_hi),
                size_category=FishSize.MEDIUM,
                tank_radius=tank_radius,
                tank_depth=tank_depth
            )
            state.other_fish.append(fish)

        # 3. Aggressive conspecifics
        num_aggressive = np.random.randint(
            aggr_min,
            aggr_max + 1
        )
        state.aggressive_fish_count = num_aggressive

        for _ in range(num_aggressive):
            fish = self._create_aggressive_fish(
                mass_range=(_ag_lo, _ag_hi),
                tank_radius=tank_radius,
                tank_depth=tank_depth
            )
            state.other_fish.append(fish)

        # 4. Surface predator
        if surface_enabled:
            predator = self._create_surface_predator(
                tank_radius=tank_radius,
                mass_range=(_pr_lo, _pr_hi),
            )
            state.other_fish.append(predator)
            state.surface_predator_present = True

        if self.debug:
            print(f"[InteractionSystem] Ecosystem initialized (v4 cooldown):")
            print(f"   Aggressive attack cooldown: {self.aggressive_cfg.attack_cooldown_steps} steps")
            print(f"   Surface predator cooldown: {self.surface_cfg.attack_cooldown_steps} steps")

    def _create_passive_fish(self, mass_range: Tuple[float, float],
                             size_category: FishSize,
                             tank_radius: float,
                             tank_depth: float) -> OtherFish:
        """Create a passive NPC fish with random position and velocity.

        Args:
            mass_range: (min, max) body mass in grams.
            size_category: Categorical size class for this NPC.
            tank_radius: Tank radius for spawn positioning.
            tank_depth: Tank depth for spawn positioning.

        Returns:
            Initialized OtherFish instance with passive behaviour.
        """
        mass = np.random.uniform(*mass_range)
        length = mass_to_length(mass)

        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(0.2, tank_radius * 0.8)
        depth = np.random.uniform(-tank_depth + 0.1, -0.1)

        position = np.array([
            radius * np.cos(angle), depth, radius * np.sin(angle)
        ], dtype=np.float32)

        velocity = np.random.uniform(-0.05, 0.05, 3).astype(np.float32)
        velocity[1] *= 0.3

        return OtherFish(
            position=position, velocity=velocity,
            body_mass=mass, total_length=length,
            behavior_type=FishBehaviorType.PASSIVE,
            original_behavior=FishBehaviorType.PASSIVE,
            size_category=size_category
        )

    def _create_aggressive_fish(self, mass_range: Tuple[float, float],
                                tank_radius: float,
                                tank_depth: float) -> OtherFish:
        """Create an aggressive NPC fish.

        Args:
            mass_range: (min, max) body mass in grams.
            tank_radius: Tank radius for spawn positioning.
            tank_depth: Tank depth for spawn positioning.

        Returns:
            Initialized OtherFish instance with aggressive behaviour.
        """
        mass = np.random.uniform(*mass_range)
        length = mass_to_length(mass)

        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(0.3, tank_radius * 0.9)
        depth = np.random.uniform(-tank_depth + 0.15, -0.15)

        position = np.array([
            radius * np.cos(angle), depth, radius * np.sin(angle)
        ], dtype=np.float32)

        velocity = np.random.uniform(-0.03, 0.03, 3).astype(np.float32)
        velocity[1] *= 0.2

        return OtherFish(
            position=position, velocity=velocity,
            body_mass=mass, total_length=length,
            behavior_type=FishBehaviorType.AGGRESSIVE,
            original_behavior=FishBehaviorType.AGGRESSIVE,
            size_category=FishSize.LARGE,
            is_chasing=False
        )

    def _create_surface_predator(self, tank_radius: float,
                                 mass_range: Tuple[float, float] = (400.0, 500.0)) -> OtherFish:
        """Create a surface predator NPC.

        Args:
            tank_radius: Tank radius for spawn positioning.
            mass_range: (min, max) body mass in grams.

        Returns:
            Initialized OtherFish instance with surface predator behaviour.
        """
        mass = np.random.uniform(*mass_range)
        length = mass_to_length(mass)

        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(0.3, tank_radius * 0.85)
        depth = np.random.uniform(-self.surface_cfg.surface_zone_depth, -0.02)

        position = np.array([
            radius * np.cos(angle), depth, radius * np.sin(angle)
        ], dtype=np.float32)

        velocity = np.random.uniform(-0.02, 0.02, 3).astype(np.float32)
        velocity[1] = 0

        return OtherFish(
            position=position, velocity=velocity,
            body_mass=mass, total_length=length,
            behavior_type=FishBehaviorType.SURFACE_PREDATOR,
            original_behavior=FishBehaviorType.SURFACE_PREDATOR,
            size_category=FishSize.PREDATOR,
            is_chasing=False
        )

    def update(self, state: InteractionState, input_data: InteractionInput,
               curriculum_config: Optional[Dict[str, float]] = None) -> InteractionOutput:
        """Execute one simulation step of all interaction mechanics.

        Sequentially updates NPC AI, resolves agent predation opportunities,
        and evaluates attack/escape events with cooldown enforcement.

        Args:
            state: Mutable interaction state.
            input_data: Current-step agent and environment data.
            curriculum_config: Optional difficulty multipliers with keys
                'capture_multiplier' and 'predation_multiplier'.

        Returns:
            InteractionOutput containing all events that occurred this step.
        """
        output = InteractionOutput()

        if curriculum_config is None:
            curriculum_config = {
                'capture_multiplier': 1.0,
                'predation_multiplier': 1.0
            }

        # Update cooldown timers for all NPC fish
        for fish in state.other_fish:
            fish.update_cooldown()

        in_surface_zone = input_data.agent_position[1] > -self.surface_cfg.surface_zone_depth
        output.in_surface_danger_zone = in_surface_zone
        if in_surface_zone:
            state.surface_zone_entries += 1

        # 1. Update NPC AI behaviours
        chase_count, flee_count = self._update_fish_ai_v4(state, input_data)
        output.being_chased = chase_count > 0
        output.chase_count = chase_count
        output.prey_fleeing_count = flee_count

        # 2. Agent predation check
        predation_result = self._check_predation(state, input_data, curriculum_config)
        output.predation_success = predation_result['success_count']
        output.energy_gained = predation_result['energy_gained']
        output.mass_gained = predation_result['mass_gained']
        output.fish_removed = predation_result['fish_removed']

        # 3. Attack/escape detection (with cooldown)
        escape_result = self._check_escape_with_cooldown(state, input_data, curriculum_config)
        output.escape_success = escape_result['escape_count']
        output.damage_taken = escape_result['damage']
        output.attacker_mass = escape_result['attacker_mass']
        output.attack_count_this_step = escape_result['attack_count']

        return output

    def _update_fish_ai_v4(self, state: InteractionState,
                           input_data: InteractionInput) -> Tuple[int, int]:
        """Update AI for all NPC fish (v4 with cooldown awareness).

        Args:
            state: Mutable interaction state.
            input_data: Current-step agent and environment data.

        Returns:
            Tuple of (chase_count, flee_count) indicating the number of
            NPCs currently chasing and fleeing respectively.
        """
        chase_count = 0
        flee_count = 0

        for fish in state.other_fish:
            if not fish.is_alive:
                continue

            if fish.behavior_type == FishBehaviorType.SURFACE_PREDATOR:
                is_chasing = self._update_surface_predator_v4(fish, input_data)
                if is_chasing:
                    chase_count += 1

            elif fish.behavior_type == FishBehaviorType.AGGRESSIVE:
                is_chasing = self._update_aggressive_fish_v4(fish, input_data)
                if is_chasing:
                    chase_count += 1

            elif fish.behavior_type == FishBehaviorType.PASSIVE or fish.is_fleeing:
                is_fleeing = self._update_passive_fish_with_fleeing(fish, input_data)
                if is_fleeing:
                    flee_count += 1

        if chase_count > 0:
            state.times_chased += 1

        return chase_count, flee_count

    def _update_aggressive_fish_v4(self, fish: OtherFish,
                                   input_data: InteractionInput) -> bool:
        """Update aggressive conspecific AI with cooldown (v4).

        Args:
            fish: The aggressive NPC entity to update.
            input_data: Current-step agent and environment data.

        Returns:
            True if the NPC is actively chasing the agent.
        """
        cfg = self.aggressive_cfg
        distance_to_agent = np.linalg.norm(fish.position - input_data.agent_position)
        base_speed = calculate_sustained_speed(fish.total_length)
        size_ratio = fish.body_mass / input_data.agent_mass
        is_prey = size_ratio >= self.c.threat_size_ratio

        # During cooldown: idle swimming, no chase
        if not fish.can_attack():
            fish.is_chasing = False
            self._random_swim(fish, base_speed, cfg.random_speed_multiplier)
            fish.position += fish.velocity * CONFIG.environment.time_step
            self._enforce_fish_boundaries(fish, input_data, is_chasing=False)
            return False

        # Line-of-sight check (obstacles + tank walls)
        obs_field = getattr(input_data, 'obstacle_field', None)
        tank_geo = getattr(input_data, 'tank_geometry', None)
        can_see_agent = True
        if obs_field is not None:
            can_see_agent = obs_field.check_line_of_sight(
                fish.position, input_data.agent_position, tank_geo
            )

        if distance_to_agent <= cfg.detection_range and is_prey and can_see_agent:
            fish.is_chasing = True
            fish.chase_target = input_data.agent_position.copy()

            direction = input_data.agent_position - fish.position
            dist = np.linalg.norm(direction)
            if dist > 0.01:
                direction = direction / dist

            direction += np.random.uniform(-cfg.direction_noise, cfg.direction_noise, 3)
            direction = direction / (np.linalg.norm(direction) + 1e-6)

            chase_speed = base_speed * cfg.chase_speed_multiplier
            fish.velocity = direction * chase_speed

        elif distance_to_agent > cfg.give_up_range or not is_prey or not can_see_agent:
            fish.is_chasing = False
            self._random_swim(fish, base_speed, cfg.random_speed_multiplier)

        max_speed = base_speed * cfg.chase_speed_multiplier
        speed = np.linalg.norm(fish.velocity)
        if speed > max_speed:
            fish.velocity = fish.velocity / speed * max_speed

        fish.position += fish.velocity * CONFIG.environment.time_step
        self._enforce_fish_boundaries(fish, input_data)

        return fish.is_chasing

    def _update_surface_predator_v4(self, fish: OtherFish,
                                    input_data: InteractionInput) -> bool:
        """Update surface predator AI (v4.1 with coordinate and depth fixes).

        Fixes applied over v3:
            1. Coordinate calculation uses X-Z horizontal plane distance
               (previously erroneously used X-Y).
            2. Detection depth uses an independent ``prey_detection_depth``
               parameter.
            3. Chase logic incorporates a give-up depth threshold to avoid
               futile deep pursuit.
            4. Vertical displacement during pursuit is clamped.

        Args:
            fish: The surface predator NPC entity to update.
            input_data: Current-step agent and environment data.

        Returns:
            True if the predator is actively chasing the agent.
        """
        cfg = self.surface_cfg
        dt = CONFIG.environment.time_step

        # Base parameters
        base_speed = calculate_sustained_speed(fish.total_length)
        size_ratio = fish.body_mass / input_data.agent_mass
        is_prey = size_ratio >= self.c.threat_size_ratio

        # Fix 1: Independent detection depth parameter
        # Agent Y > -prey_detection_depth means within detection zone
        agent_in_detection_zone = input_data.agent_position[1] > -cfg.prey_detection_depth

        # Agent has dived beyond the give-up depth
        agent_too_deep = input_data.agent_position[1] < -cfg.give_up_depth

        # During cooldown: patrol only, no chase
        if not fish.can_attack():
            fish.is_chasing = False
            self._patrol_surface(fish, base_speed, cfg.patrol_speed_multiplier)
            fish.position += fish.velocity * dt
            self._enforce_surface_predator_boundaries(fish, input_data, is_chasing=False)
            return False

        # Fix 2: Use X-Z horizontal plane distance
        horizontal_distance = np.sqrt(
            (fish.position[0] - input_data.agent_position[0]) ** 2 +
            (fish.position[2] - input_data.agent_position[2]) ** 2
        )

        # Line-of-sight check (obstacles + tank walls)
        obs_field = getattr(input_data, 'obstacle_field', None)
        tank_geo = getattr(input_data, 'tank_geometry', None)
        can_see_agent = True
        if obs_field is not None:
            can_see_agent = obs_field.check_line_of_sight(
                fish.position, input_data.agent_position, tank_geo
            )

        should_chase = (
                agent_in_detection_zone and
                not agent_too_deep and
                is_prey and
                horizontal_distance <= cfg.detection_range and
                can_see_agent
        )

        if should_chase:
            fish.is_chasing = True

            # Compute chase direction
            direction = input_data.agent_position - fish.position

            # Fix 3: Clamp vertical pursuit amplitude
            max_vertical = cfg.vertical_chase_limit
            direction[1] = np.clip(direction[1], -max_vertical, max_vertical)

            # Normalize direction
            dist = np.linalg.norm(direction)
            if dist > 0.01:
                direction = direction / dist

            # Add stochastic noise for naturalistic pursuit paths
            if cfg.direction_noise > 0:
                noise = np.random.uniform(-cfg.direction_noise, cfg.direction_noise, 3)
                noise[1] *= 0.3  # Reduced vertical noise
                direction = direction + noise
                direction = direction / (np.linalg.norm(direction) + 1e-6)

            # Set chase velocity
            chase_speed = base_speed * cfg.chase_speed_multiplier
            fish.velocity = direction * chase_speed

        else:
            # Chase conditions not met; revert to patrol
            fish.is_chasing = False
            self._patrol_surface(fish, base_speed, cfg.patrol_speed_multiplier)

        # Speed clamping
        max_speed = base_speed * cfg.chase_speed_multiplier
        speed = np.linalg.norm(fish.velocity)
        if speed > max_speed:
            fish.velocity = fish.velocity / speed * max_speed

        # Position integration
        fish.position += fish.velocity * dt
        self._enforce_surface_predator_boundaries(fish, input_data, is_chasing=fish.is_chasing)

        return fish.is_chasing

    def _random_swim(self, fish: OtherFish, base_speed: float, speed_mult: float) -> None:
        """Apply idle random swimming behaviour.

        Periodically selects a new random heading and applies minor
        velocity perturbations between direction changes.

        Args:
            fish: NPC fish entity to update.
            base_speed: Maximum sustained speed for this fish (m/s).
            speed_mult: Fraction of base speed used during idle swimming.
        """
        fish.random_direction_timer += 1
        if fish.random_direction_timer > 50:
            fish.random_direction_timer = 0
            angle = np.random.uniform(0, 2 * np.pi)
            fish.velocity = np.array([
                np.cos(angle),
                np.random.uniform(-0.1, 0.1),
                np.sin(angle)
            ], dtype=np.float32)
            fish.velocity *= base_speed * speed_mult

        fish.velocity += np.random.uniform(-0.005, 0.005, 3).astype(np.float32)

    def _patrol_surface(self, fish: OtherFish, base_speed: float, speed_mult: float) -> None:
        """Apply surface patrol swimming behaviour for surface predators.

        Similar to idle swimming but confined near the surface with
        periodic heading changes and minor vertical oscillation.

        Args:
            fish: Surface predator NPC to update.
            base_speed: Maximum sustained speed for this fish (m/s).
            speed_mult: Fraction of base speed used during patrol.
        """
        fish.random_direction_timer += 1
        if fish.random_direction_timer > 60:
            fish.random_direction_timer = 0
            angle = np.random.uniform(0, 2 * np.pi)
            # Allow minor vertical movement during patrol
            fish.velocity = np.array([
                np.cos(angle),
                np.random.uniform(-0.1, 0.1),
                np.sin(angle)
            ], dtype=np.float32)
            fish.velocity *= base_speed * speed_mult

    def _update_passive_fish_with_fleeing(self, fish: OtherFish,
                                          input_data: InteractionInput) -> bool:
        """Update passive prey fish with threat-triggered fleeing behaviour.

        When the agent is large enough relative to the prey and within
        detection range, the prey initiates a flee response. The flee
        direction decays over time and the prey eventually resumes idle
        swimming once safe.

        Args:
            fish: Passive NPC fish entity to update.
            input_data: Current-step agent and environment data.

        Returns:
            True if the fish is currently in a fleeing state.
        """
        cfg = self.fleeing_cfg

        size_ratio = input_data.agent_mass / fish.body_mass
        is_threat = size_ratio >= self.c.min_predation_size_ratio
        distance_to_agent = np.linalg.norm(fish.position - input_data.agent_position)
        base_speed = calculate_sustained_speed(fish.total_length)

        if is_threat and distance_to_agent <= cfg.detection_range:
            if not fish.is_fleeing:
                fish.is_fleeing = True
                fish.flee_duration = 0

            fish.flee_duration += 1

            flee_direction = fish.position - input_data.agent_position
            dist = np.linalg.norm(flee_direction)
            if dist > 0.01:
                flee_direction = flee_direction / dist
            else:
                flee_direction = np.random.uniform(-1, 1, 3)
                flee_direction = flee_direction / (np.linalg.norm(flee_direction) + 1e-6)

            flee_direction += np.random.uniform(-cfg.direction_noise, cfg.direction_noise, 3)
            flee_direction = flee_direction / (np.linalg.norm(flee_direction) + 1e-6)

            fish.flee_direction = flee_direction
            flee_speed = base_speed * cfg.flee_speed_multiplier
            fish.velocity = flee_direction * flee_speed

        elif fish.is_fleeing:
            should_stop = (
                    distance_to_agent > cfg.safe_distance or
                    fish.flee_duration > cfg.flee_duration_max or
                    not is_threat
            )

            if should_stop:
                fish.is_fleeing = False
                fish.flee_duration = 0
                fish.flee_direction = None
            else:
                fish.flee_duration += 1
                if fish.flee_direction is not None:
                    speed_factor = max(0.8, 1.0 - fish.flee_duration * 0.005)
                    flee_speed = base_speed * cfg.flee_speed_multiplier * speed_factor
                    fish.velocity = fish.flee_direction * flee_speed

        if not fish.is_fleeing:
            fish.velocity += np.random.uniform(-0.01, 0.01, 3).astype(np.float32)
            fish.velocity[1] *= 0.5
            max_speed = base_speed
            speed = np.linalg.norm(fish.velocity)
            if speed > max_speed:
                fish.velocity = fish.velocity / speed * max_speed

        fish.position += fish.velocity * CONFIG.environment.time_step
        self._enforce_fish_boundaries(fish, input_data, is_chasing=False)

        return fish.is_fleeing

    def _check_escape_with_cooldown(self, state: InteractionState,
                                    input_data: InteractionInput,
                                    curriculum_config: Dict[str, float]) -> Dict[str, Any]:
        """Evaluate attack/escape events with cooldown enforcement (v4).

        Iterates over all threatening NPCs that are off cooldown and within
        attack range. For each valid attack, computes an escape probability
        and stochastically resolves the outcome.

        Args:
            state: Mutable interaction state.
            input_data: Current-step agent and environment data.
            curriculum_config: Difficulty multipliers.

        Returns:
            Dictionary with keys:
                - 'escape_count': Number of successful escapes.
                - 'damage': Total damage inflicted on the agent.
                - 'attacker_mass': Mass of the last attacker (grams).
                - 'attack_count': Total attacks resolved this step.
        """
        result = {
            'escape_count': 0,
            'damage': 0.0,
            'attacker_mass': 0.0,
            'attack_count': 0
        }

        for fish in state.other_fish:
            if not fish.is_alive:
                continue

            # Skip NPCs still in cooldown
            if not fish.can_attack():
                continue

            is_threat = False
            damage_multiplier = 1.0
            cooldown_steps = 0
            attack_range = 0.0

            if fish.behavior_type == FishBehaviorType.SURFACE_PREDATOR:
                if fish.is_chasing:
                    is_threat = True
                    damage_multiplier = 1.5
                    cooldown_steps = self.surface_cfg.attack_cooldown_steps
                    attack_range = self.surface_cfg.attack_range

            elif fish.behavior_type == FishBehaviorType.AGGRESSIVE:
                if fish.is_chasing:
                    is_threat = True
                    damage_multiplier = 1.2
                    cooldown_steps = self.aggressive_cfg.attack_cooldown_steps
                    attack_range = self.aggressive_cfg.attack_range

            else:
                size_ratio = fish.body_mass / input_data.agent_mass
                if size_ratio >= self.c.threat_size_ratio:
                    is_threat = True

            if not is_threat:
                continue

            distance = np.linalg.norm(fish.position - input_data.agent_position)

            # Use configured attack range; fall back to biological calculation
            if attack_range <= 0:
                attack_range = calculate_strike_range(fish.total_length)

            if distance > attack_range:
                continue

            # Pre-attack line-of-sight check: blocked by obstacles or walls
            obs_field = getattr(input_data, 'obstacle_field', None)
            tank_geo = getattr(input_data, 'tank_geometry', None)
            if obs_field is not None:
                if not obs_field.check_line_of_sight(
                        fish.position, input_data.agent_position, tank_geo
                ):
                    continue  # Line of sight blocked; attack fails

            # Within attack range -- resolve attack
            state.escape_attempts += 1
            result['attack_count'] += 1

            size_ratio = fish.body_mass / input_data.agent_mass
            escape_prob = self._calculate_escape_success(
                size_ratio, distance, input_data, fish
            )

            if np.random.random() < escape_prob:
                # Escape successful
                state.escape_successes += 1
                result['escape_count'] += 1
                if self.debug:
                    print(f"[Escape] Success! [{fish.behavior_type.value}]")
            else:
                # Escape failed -- agent takes damage
                fish.successful_attacks += 1
                state.attacks_this_episode += 1

                if fish.behavior_type == FishBehaviorType.SURFACE_PREDATOR:
                    damage = self.surface_cfg.attack_damage * damage_multiplier
                else:
                    damage = self.c.attack_damage_base * (size_ratio ** 0.5) * damage_multiplier

                result['damage'] += damage
                result['attacker_mass'] = fish.body_mass  # Record attacker mass
                state.damage_taken += damage

                # Post-attack cooldown
                fish.start_cooldown(cooldown_steps)

                if self.debug:
                    print(f"[Attack] Hit! [{fish.behavior_type.value}] "
                          f"damage={damage:.1f}, cooldown={cooldown_steps} steps")

        return result

    def _check_predation(self, state: InteractionState, input_data: InteractionInput,
                         curriculum_config: Dict[str, float]) -> Dict[str, Any]:
        """Evaluate predation opportunities for the agent.

        Checks each prey-sized NPC within strike range and stochastically
        resolves capture success based on size ratio, distance, speed
        advantage, and agent condition.

        Args:
            state: Mutable interaction state.
            input_data: Current-step agent and environment data.
            curriculum_config: Difficulty multipliers.

        Returns:
            Dictionary with keys:
                - 'success_count': Number of prey captured.
                - 'energy_gained': Total energy gained (joules).
                - 'mass_gained': Total prey mass consumed (grams).
                - 'fish_removed': Number of NPCs removed from simulation.
        """
        result = {
            'success_count': 0,
            'energy_gained': 0.0,
            'mass_gained': 0.0,
            'fish_removed': 0
        }

        capture_radius = calculate_capture_radius(input_data.agent_length)
        capture_radius *= curriculum_config.get('capture_multiplier', 1.0)
        strike_range = calculate_strike_range(input_data.agent_length)

        for fish in state.other_fish[:]:
            if not fish.is_alive:
                continue

            if fish.behavior_type == FishBehaviorType.SURFACE_PREDATOR:
                continue

            size_ratio = input_data.agent_mass / fish.body_mass
            if size_ratio < self.c.min_predation_size_ratio:
                continue

            distance = np.linalg.norm(fish.position - input_data.agent_position)
            if distance > strike_range:
                continue

            state.predation_attempts += 1

            success_prob = self._calculate_predation_success(
                size_ratio, distance, input_data, fish
            )

            if fish.is_fleeing:
                success_prob *= 0.9

            success_prob *= curriculum_config.get('predation_multiplier', 1.0)

            if np.random.random() < success_prob:
                state.predation_successes += 1
                fish.is_alive = False

                prey_energy_density = calculate_fish_energy_density(fish.body_mass)
                energy_gain = fish.body_mass * prey_energy_density
                energy_gain *= self.c.predation_energy_efficiency
                result['energy_gained'] += energy_gain
                result['mass_gained'] += fish.body_mass
                result['success_count'] += 1

                if self.debug:
                    print(f"[Predation] Success! Gained {fish.body_mass:.1f}g")

        state.other_fish = [f for f in state.other_fish if f.is_alive]
        result['fish_removed'] = result['success_count']

        return result

    def _calculate_predation_success(self, size_ratio: float, distance: float,
                                     input_data: InteractionInput,
                                     target: OtherFish) -> float:
        """Compute the probability of a successful predation attempt.

        Factors include size advantage, proximity to target, relative
        speed, agent fatigue, and stress level.

        Args:
            size_ratio: Agent mass / target mass.
            distance: Euclidean distance to target (metres).
            input_data: Current-step agent data.
            target: The prey NPC being attacked.

        Returns:
            Probability of capture success, clipped to [0.0, 0.8].
        """
        if size_ratio >= 3.0:
            base_prob = 0.8
        elif size_ratio >= 2.0:
            base_prob = 0.6
        elif size_ratio >= 1.5:
            base_prob = 0.3
        else:
            base_prob = 0.1

        strike_range = calculate_strike_range(input_data.agent_length)
        distance_factor = 1.0 - (distance / strike_range) * 0.5

        agent_speed = np.linalg.norm(input_data.agent_velocity)
        target_speed = np.linalg.norm(target.velocity)
        if target_speed > 0.01:
            speed_factor = min(agent_speed / target_speed, 1.5)
        else:
            speed_factor = 1.5

        fatigue_factor = 1.0 - input_data.agent_fatigue / 200.0
        stress_factor = 1.0 - input_data.agent_stress * 0.3

        final_prob = (base_prob * distance_factor * speed_factor *
                      fatigue_factor * stress_factor)

        return np.clip(final_prob, 0.0, 0.8)

    def _calculate_escape_success(self, threat_ratio: float, distance: float,
                                  input_data: InteractionInput,
                                  threat: OtherFish) -> float:
        """Compute the probability of successfully escaping an attack.

        Factors include threat size ratio, distance to attacker, whether
        the attacker is actively chasing, relative speed, and agent
        fatigue.

        Args:
            threat_ratio: Attacker mass / agent mass.
            distance: Euclidean distance to the attacker (metres).
            input_data: Current-step agent data.
            threat: The attacking NPC entity.

        Returns:
            Probability of escape success, clipped to [0.05, 0.95].
        """
        if threat_ratio >= 4.0:
            base_prob = 0.15
        elif threat_ratio >= 3.0:
            base_prob = 0.22
        elif threat_ratio >= 2.0:
            base_prob = 0.30
        else:
            base_prob = 0.40

        if threat.is_chasing:
            base_prob *= 0.8

        threat_range = calculate_strike_range(threat.total_length)
        distance_factor = 0.5 + 0.5 * (distance / threat_range)

        agent_speed = np.linalg.norm(input_data.agent_velocity)
        threat_speed = np.linalg.norm(threat.velocity)
        if threat_speed > 0.01:
            # Speed ratio has amplified influence: fast escape confers
            # a significant survival advantage
            speed_ratio = agent_speed / threat_speed
            speed_factor = np.clip(speed_ratio ** 1.5, 0.3, 2.0)
        else:
            speed_factor = 2.0

        fatigue_factor = 1.0 - input_data.agent_fatigue / 150.0

        final_prob = base_prob * distance_factor * speed_factor * fatigue_factor
        return np.clip(final_prob, 0.05, 0.95)

    def _enforce_fish_boundaries(self, fish: OtherFish,
                                 input_data: InteractionInput,
                                 is_chasing: bool = False) -> None:
        """Enforce spatial boundaries for NPC fish.

        Supports both TankGeometry-based and fallback cylindrical
        boundary logic. Also resolves collisions with obstacle fields.

        Args:
            fish: NPC fish entity whose position/velocity may be corrected.
            input_data: Provides tank geometry and obstacle field references.
            is_chasing: If True, allows tighter boundary margins.
        """
        tank_geo = getattr(input_data, 'tank_geometry', None)
        obs_field = getattr(input_data, 'obstacle_field', None)

        if tank_geo is not None:
            fish.position, fish.velocity, _ = tank_geo.enforce_boundary(
                fish.position, fish.velocity
            )
        else:
            # Fallback: original cylindrical tank logic
            boundary_margin = 0.03 if is_chasing else 0.05
            max_radius = input_data.tank_radius - boundary_margin
            horizontal_dist = np.sqrt(fish.position[0] ** 2 + fish.position[2] ** 2)

            if horizontal_dist > max_radius:
                factor = max_radius / horizontal_dist
                fish.position[0] *= factor
                fish.position[2] *= factor
                fish.velocity[0] *= -0.5
                fish.velocity[2] *= -0.5

            depth = getattr(tank_geo, 'depth', input_data.tank_depth) if tank_geo else input_data.tank_depth

            if fish.position[1] < -input_data.tank_depth + 0.05:
                fish.position[1] = -input_data.tank_depth + 0.05
                fish.velocity[1] = abs(fish.velocity[1]) * 0.5
            if fish.position[1] > -0.05:
                fish.position[1] = -0.05
                fish.velocity[1] = -abs(fish.velocity[1]) * 0.5

        # Obstacle collision resolution
        if obs_field is not None:
            col = obs_field.check_collision(fish.position)
            if col.collided:
                fish.position = col.pushed_position.copy()
                # NPC hitting obstacle: velocity reflected with damping
                fish.velocity = obs_field.resolve_collision_velocity(
                    fish.velocity, col.normal
                )

    def _enforce_surface_predator_boundaries(self, fish: OtherFish,
                                             input_data: InteractionInput,
                                             is_chasing: bool = False) -> None:
        """Enforce spatial boundaries for surface predator NPCs.

        Similar to general boundary enforcement but additionally constrains
        the predator to remain within the surface zone depth band.

        Args:
            fish: Surface predator entity whose position may be corrected.
            input_data: Provides tank geometry and obstacle field references.
            is_chasing: If True, allows tighter boundary margins.
        """
        cfg = self.surface_cfg
        tank_geo = getattr(input_data, 'tank_geometry', None)
        obs_field = getattr(input_data, 'obstacle_field', None)

        # Horizontal boundary
        if tank_geo is not None:
            fish.position, fish.velocity, _ = tank_geo.enforce_boundary(
                fish.position, fish.velocity
            )
        else:
            boundary_margin = 0.03 if is_chasing else 0.05
            max_radius = input_data.tank_radius - boundary_margin
            horizontal_dist = np.sqrt(fish.position[0] ** 2 + fish.position[2] ** 2)

            if horizontal_dist > max_radius:
                factor = max_radius / horizontal_dist
                fish.position[0] *= factor
                fish.position[2] *= factor
                fish.velocity[0] *= -0.5
                fish.velocity[2] *= -0.5

        # Depth constraint (predator confined to surface zone)
        if fish.position[1] < -cfg.surface_zone_depth:
            fish.position[1] = -cfg.surface_zone_depth
            fish.velocity[1] = abs(fish.velocity[1]) * 0.3
        if fish.position[1] > cfg.surface_zone_max:
            fish.position[1] = cfg.surface_zone_max
            fish.velocity[1] = -abs(fish.velocity[1]) * 0.3

        # Obstacle collision resolution
        if obs_field is not None:
            col = obs_field.check_collision(fish.position)
            if col.collided:
                fish.position = col.pushed_position.copy()
                fish.velocity = obs_field.resolve_collision_velocity(
                    fish.velocity, col.normal
                )

    def get_fish_count(self, state: InteractionState) -> int:
        """Return the number of living NPC fish.

        Args:
            state: Current interaction state.

        Returns:
            Count of alive NPC fish entities.
        """
        return sum(1 for f in state.other_fish if f.is_alive)

    def get_fish_states(self, state: InteractionState) -> List[Dict[str, Any]]:
        """Return a snapshot of all living NPC fish states.

        Args:
            state: Current interaction state.

        Returns:
            List of dictionaries, each containing position, velocity,
            mass, behaviour type, chase/flee status, and cooldown info.
        """
        return [
            {
                'position': f.position.copy(),
                'velocity': f.velocity.copy(),
                'body_mass': f.body_mass,
                'behavior_type': f.behavior_type.value,
                'is_chasing': f.is_chasing,
                'is_fleeing': f.is_fleeing,
                'size_category': f.size_category.value,
                'attack_cooldown': f.attack_cooldown,
                'can_attack': f.can_attack()
            }
            for f in state.other_fish if f.is_alive
        ]

    def get_statistics(self, state: InteractionState) -> Dict[str, Any]:
        """Return aggregate interaction statistics for the current episode.

        Args:
            state: Current interaction state.

        Returns:
            Dictionary of statistical counters including fish counts,
            chase events, predation attempts/successes, escape events,
            and cumulative damage.
        """
        passive_count = sum(1 for f in state.other_fish
                            if f.is_alive and f.original_behavior == FishBehaviorType.PASSIVE)
        aggressive_count = sum(1 for f in state.other_fish
                               if f.is_alive and f.behavior_type == FishBehaviorType.AGGRESSIVE)
        predator_count = sum(1 for f in state.other_fish
                             if f.is_alive and f.behavior_type == FishBehaviorType.SURFACE_PREDATOR)
        chasing_count = sum(1 for f in state.other_fish
                            if f.is_alive and f.is_chasing)
        cooling_count = sum(1 for f in state.other_fish
                            if f.is_alive and not f.can_attack())

        return {
            'fish_alive': self.get_fish_count(state),
            'passive_fish': passive_count,
            'aggressive_fish': aggressive_count,
            'surface_predator': predator_count,
            'currently_chasing': chasing_count,
            'in_cooldown': cooling_count,
            'attacks_this_episode': state.attacks_this_episode,
            'times_chased': state.times_chased,
            'predation_attempts': state.predation_attempts,
            'predation_successes': state.predation_successes,
            'escape_attempts': state.escape_attempts,
            'escape_successes': state.escape_successes,
            'damage_taken': state.damage_taken
        }

    def set_debug(self, enabled: bool) -> None:
        """Enable or disable debug logging.

        Args:
            enabled: If True, print diagnostic messages during updates.
        """
        self.debug = enabled


# ============================================================
# Factory Functions
# ============================================================

def create_interaction_system() -> InteractionSystem:
    """Create and return a new InteractionSystem instance.

    Returns:
        Fully initialized InteractionSystem ready for use.
    """
    return InteractionSystem()


def create_interaction_state() -> InteractionState:
    """Create and return a new InteractionState instance.

    Returns:
        Empty InteractionState ready for ecosystem initialization.
    """
    return InteractionState()
