#!/usr/bin/env python3
"""
Sensory Perception Subsystem
=============================

This module implements the sensory perception subsystem that constructs the
observation vector for the reinforcement learning agent. It integrates multiple
sensory modalities consistent with largemouth bass (Micropterus salmoides)
neurobiology:

- Visual detection: identification of conspecifics and food items within a
  binocular/monocular field of view, subject to line-of-sight occlusion by
  obstacles and tank boundaries.
- Lateral line mechanoreception: omnidirectional short-range detection of
  nearby objects via hydrodynamic pressure gradients, independent of visual
  field constraints.
- Proprioception: internal state signals including energy reserves, stomach
  fullness, and current activity state (active vs. resting).
- Spatial awareness: boundary proximity sensing (tank walls, bottom, surface)
  and obstacle detection via simulated ray-casting.

The resting-state extension reduces effective perception ranges and introduces
reaction delays during quiescent periods, modeling the reduced vigilance
documented in teleost rest behavior (Reebs, 2002).
"""

import numpy as np
import math
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum
from collections import deque
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CONFIG

# Import activity state enumeration
try:
    from systems.metabolism import ActivityState
except ImportError:
    class ActivityState(Enum):
        ACTIVE = "active"
        RESTING = "resting"


@dataclass
class PerceptionState:
    """Internal state of the perception subsystem (resting-state enhanced).

    Attributes:
        nearest_fish_position: World-space position of the nearest perceived
            conspecific, or None if no fish is detected.
        nearest_fish_distance: Euclidean distance to the nearest conspecific.
        nearest_fish_mass_ratio: Body mass ratio (target / agent).
        nearest_food_position: World-space position of the nearest food item.
        nearest_food_distance: Euclidean distance to the nearest food item.
        nearest_threat_distance: Euclidean distance to the nearest threat.
        obstacle_distances: Normalized ray-cast distances in 8 directions.
        boundary_vector: Unit vector pointing away from the nearest boundary.
        min_boundary_distance: Distance to the closest tank boundary (m).
        surface_distance: Normalized distance to the water surface.
        is_near_surface: Whether the agent is within 0.1 m of the surface.
        observed_fish: List of perceived conspecific dictionaries.
        observed_food: List of perceived food item dictionaries.
        threat_count: Number of detected threats in the current frame.
        activity_state: Current behavioral activity state.
        reaction_delay_timer: Accumulated time since threat first detected (s).
        delayed_threat_distance: Threat distance after reaction delay applied.
        threat_detection_queue: Circular buffer of recent threat detections.
        effective_vision_range: State-adjusted visual detection range (m).
        effective_food_range: State-adjusted food detection range (m).
    """
    # Core perception fields
    nearest_fish_position: Optional[np.ndarray] = None
    nearest_fish_distance: float = float('inf')
    nearest_fish_mass_ratio: float = 1.0
    nearest_food_position: Optional[np.ndarray] = None
    nearest_food_distance: float = float('inf')
    nearest_threat_distance: float = float('inf')
    obstacle_distances: np.ndarray = field(default_factory=lambda: np.ones(8, dtype=np.float32))
    boundary_vector: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    min_boundary_distance: float = float('inf')
    surface_distance: float = 1.0
    is_near_surface: bool = False

    observed_fish: List[Dict] = field(default_factory=list)
    observed_food: List[Dict] = field(default_factory=list)
    threat_count: int = 0

    # ==================== Activity state fields ====================
    activity_state: ActivityState = ActivityState.ACTIVE

    # Reaction delay buffer
    reaction_delay_timer: float = 0.0  # Reaction delay timer (s)
    delayed_threat_distance: float = float('inf')  # Threat distance after delay
    threat_detection_queue: deque = field(default_factory=lambda: deque(maxlen=30))  # Threat detection queue

    # Effective perception ranges (adjusted by activity state)
    effective_vision_range: float = field(default_factory=lambda: CONFIG.perception.vision_range)
    effective_food_range: float = field(default_factory=lambda: CONFIG.perception.food_detection_range)


@dataclass
class PerceptionInput:
    """Input data structure for the perception subsystem.

    Attributes:
        agent_position: World-space position of the focal agent (m).
        agent_velocity: Velocity vector of the focal agent (m/s).
        agent_mass: Body mass of the focal agent (g).
        fish_states: List of conspecific state dictionaries.
        food_positions: List of food item dictionaries with positions.
        agent_heading: Optional heading direction unit vector.
        tank_radius: Radius of the cylindrical tank (m).
        tank_depth: Depth of the tank (m).
        activity_state: Current activity state of the agent.
        time_step: Simulation time step duration (s).
        tank_geometry: TankGeometry instance for non-circular enclosures.
        obstacle_field: ObstacleField instance for line-of-sight checks.
    """
    agent_position: np.ndarray
    agent_velocity: np.ndarray
    agent_mass: float
    fish_states: List[Dict]
    food_positions: List[Dict]
    agent_heading: Optional[np.ndarray] = None
    tank_radius: float = field(default_factory=lambda: CONFIG.environment.tank_radius)
    tank_depth: float = field(default_factory=lambda: CONFIG.environment.tank_depth)

    # ==================== Activity state fields ====================
    activity_state: ActivityState = ActivityState.ACTIVE
    time_step: float = field(default_factory=lambda: CONFIG.environment.time_step)
    tank_geometry: Any = None      # TankGeometry instance
    obstacle_field: Any = None     # ObstacleField instance


class PerceptionSystem:
    """Sensory perception system with resting-state modulation.

    This class implements the full perception pipeline including visual field
    filtering, lateral line mechanoreception, ecological priority sorting of
    detected entities, and reaction delay modeling during rest states.
    """

    def __init__(self) -> None:
        self.c = CONFIG.perception
        self.env = CONFIG.environment
        self.rc = CONFIG.rest_state
        self.debug = False
        self.max_fish = self.c.max_fish_observed
        self.max_food = self.c.max_food_observed

    def update(self, state: PerceptionState, input_data: PerceptionInput) -> PerceptionState:
        """Execute a full perception update for the current time step.

        Args:
            state: Mutable perception state to be updated in place.
            input_data: Snapshot of the environment visible to the agent.

        Returns:
            The updated PerceptionState (same object, mutated in place).
        """
        # Update activity state
        state.activity_state = input_data.activity_state

        # Update effective perception ranges based on activity state
        self._update_effective_ranges(state)

        # Execute perception subroutines
        self._perceive_fish(state, input_data)
        self._perceive_food(state, input_data)
        self._perceive_threats(state, input_data)
        self._detect_obstacles(state, input_data)
        self._perceive_boundaries(state, input_data)
        self._perceive_surface(state, input_data)

        # Process reaction delay for threat response
        self._process_reaction_delay(state, input_data)

        return state

    def _get_vision_heading(self, input_data: PerceptionInput) -> Optional[np.ndarray]:
        """Compute the horizontal heading unit vector for visual field filtering.

        Extracts the XZ-plane projection of the agent heading or, as fallback,
        the agent velocity. Returns None if no valid heading can be determined,
        in which case the caller should degrade to omnidirectional perception.

        Args:
            input_data: Current perception input containing heading/velocity.

        Returns:
            Normalized XZ heading vector, or None if indeterminate.
        """
        heading = getattr(input_data, 'agent_heading', None)
        if heading is not None and np.linalg.norm(heading) > 0.01:
            h = np.array([heading[0], 0.0, heading[2]], dtype=np.float32)
        else:
            v = input_data.agent_velocity
            h = np.array([v[0], 0.0, v[2]], dtype=np.float32)
        norm = np.linalg.norm(h)
        if norm < 0.01:
            return None  # No valid heading; caller should use omnidirectional perception
        return h / norm

    def _in_visual_field(self, forward_xz: np.ndarray,
                         to_target: np.ndarray,
                         distance: float) -> bool:
        """Determine whether a target is perceptible via dual-layer sensing.

        Two sensing layers are evaluated:
        - Visual layer: target within vision_range AND horizontal angle within
          half field-of-view (approximately +/-80 degrees).
        - Lateral line layer: target within lateral_line_range (omnidirectional,
          no angular constraint, simulating mechanoreceptive pressure sensing).

        Args:
            forward_xz: Normalized horizontal heading vector of the agent.
            to_target: Unit vector from agent to target in world space.
            distance: Euclidean distance to the target (m).

        Returns:
            True if the target is perceptible by either sensing modality.
        """
        # Lateral line fallback: omnidirectional short-range perception
        if distance <= self.c.lateral_line_range:
            return True

        # Visual angle check (XZ horizontal plane projection only)
        half_fov_cos = math.cos(math.radians(self.c.vision_angle / 2.0))  # cos(80 deg) ~ -0.174

        to_xz = np.array([to_target[0], 0.0, to_target[2]], dtype=np.float32)
        norm = np.linalg.norm(to_xz)
        if norm < 0.01:
            # Target directly above/below; no horizontal angle constraint
            return True
        cos_a = float(np.dot(forward_xz, to_xz / norm))
        return cos_a >= half_fov_cos

    def _update_effective_ranges(self, state: PerceptionState) -> None:
        """Adjust effective perception ranges based on activity state.

        During resting, visual and food detection ranges are reduced by
        configured attenuation factors to model decreased vigilance.

        Args:
            state: Perception state whose effective ranges will be updated.
        """
        rc = self.rc
        base_vision = self.c.vision_range
        base_food = self.c.food_detection_range

        if state.activity_state == ActivityState.RESTING:
            state.effective_vision_range = base_vision * rc.rest_vision_reduction
            state.effective_food_range = base_food * rc.rest_food_detection_reduction
        else:
            state.effective_vision_range = base_vision
            state.effective_food_range = base_food

    def _perceive_fish(self, state: PerceptionState, input_data: PerceptionInput) -> None:
        """Detect conspecifics using ecological priority sorting.

        Fish are ranked by ecological relevance rather than raw distance:
        actively chasing threats > stationary threats > prey > neutral.
        Selection fills dedicated threat and prey observation slots.

        Args:
            state: Perception state to populate with observed fish.
            input_data: Environment snapshot with conspecific positions.
        """
        state.nearest_fish_position = None
        state.nearest_fish_distance = float('inf')
        state.nearest_fish_mass_ratio = 1.0
        state.observed_fish = []
        state.threat_count = 0

        if not input_data.fish_states:
            return

        all_fish = []
        effective_range = state.effective_vision_range

        # Pre-compute heading for visual field filtering
        forward_xz = self._get_vision_heading(input_data)

        # ===== Step 1: Collect all perceptible conspecifics =====
        for fish in input_data.fish_states:
            fish_pos = fish['position']
            fish_mass = fish['body_mass']
            diff = fish_pos - input_data.agent_position
            distance = np.linalg.norm(diff)

            if distance > effective_range:
                continue

            # ===== Visual field angle filtering (vision + lateral line) =====
            if forward_xz is not None:
                to_target = diff / (distance + 1e-8)
                if not self._in_visual_field(forward_xz, to_target, distance):
                    continue

            # ===== Line-of-sight check (obstacles + tank walls) =====
            obstacle_field = getattr(input_data, 'obstacle_field', None)
            tank_geometry = getattr(input_data, 'tank_geometry', None)
            if obstacle_field is not None:
                if not obstacle_field.check_line_of_sight(
                        input_data.agent_position, fish_pos, tank_geometry
                ):
                    continue  # Occluded by obstacle or tank wall

            if distance > 0.01:
                rel_pos = (fish_pos - input_data.agent_position) / distance
            else:
                rel_pos = np.zeros(3)

            mass_ratio = (fish_mass - input_data.agent_mass) / input_data.agent_mass
            mass_ratio = np.clip(mass_ratio, -2, 5)

            behavior = fish.get('behavior_type', 'passive')
            size = fish.get('size_category', 'small')

            # Threat classification
            is_threat = (
                    behavior == 'surface' or
                    behavior == 'aggressive' or
                    size == 'large' or
                    fish_mass > input_data.agent_mass * CONFIG.interaction.threat_size_ratio
            )

            # Prey classification (consumable target)
            is_prey = fish_mass < input_data.agent_mass / CONFIG.interaction.min_predation_size_ratio

            if is_threat:
                state.threat_count += 1

            all_fish.append({
                'position': fish_pos.copy(),
                'distance': distance,
                'relative_position': rel_pos.copy(),
                'mass_ratio': mass_ratio,
                'body_mass': fish_mass,
                'velocity': fish.get('velocity', np.zeros(3)),
                'is_threat': is_threat,
                'is_prey': is_prey,
                'is_chasing': fish.get('is_chasing', False),
                'behavior_type': behavior,
                'size_category': size
            })

        # ===== Step 2: Ecological priority sorting =====
        def ecological_priority(fish_data):
            distance = fish_data['distance']
            if fish_data['is_threat'] and fish_data['is_chasing']:
                return distance
            elif fish_data['is_threat']:
                return 100 + distance
            elif fish_data['is_prey']:
                return 200 + distance
            else:
                return 300 + distance

        all_fish.sort(key=ecological_priority)

        # ===== Step 3: Slot-based selection (threat slots + prey slots) =====
        threat_slots = CONFIG.perception.max_threat_slots
        prey_slots   = CONFIG.perception.max_prey_slots

        threats = [f for f in all_fish if f['is_threat']][:threat_slots]
        preys   = [f for f in all_fish if f['is_prey']][:prey_slots]

        selected = threats + preys
        seen_ids = set()
        deduped = []
        for f in selected:
            fid = id(f)
            if fid not in seen_ids:
                seen_ids.add(fid)
                deduped.append(f)
        state.observed_fish = deduped[:self.max_fish]

        if state.observed_fish:
            nearest = state.observed_fish[0]
            state.nearest_fish_position = nearest['position']
            state.nearest_fish_distance = nearest['distance']
            state.nearest_fish_mass_ratio = nearest['body_mass'] / input_data.agent_mass

    def _perceive_food(self, state: PerceptionState, input_data: PerceptionInput) -> None:
        """Detect food items within effective range using vectorized distance filtering.

        Uses partial sorting (argpartition) for O(N) candidate selection,
        followed by visual field and line-of-sight filtering on the shortlist.

        Args:
            state: Perception state to populate with observed food items.
            input_data: Environment snapshot with food item positions.
        """
        state.nearest_food_position = None
        state.nearest_food_distance = float('inf')
        state.observed_food = []

        if not input_data.food_positions:
            return

        # Pre-compute heading for visual field filtering
        forward_xz = self._get_vision_heading(input_data)

        # 1. Vectorized position extraction (loop -> matrix operation)
        food_positions = np.array([f['position'] for f in input_data.food_positions])

        # 2. Vectorized squared-distance computation (avoids sqrt for speed)
        diff = food_positions - input_data.agent_position
        dist_sq = np.sum(diff ** 2, axis=1)

        # 3. Fast top-N selection via argpartition (O(N) complexity)
        n_nearest = min(len(dist_sq), self.max_food)
        # argpartition places the smallest n elements at the front without full sort
        nearest_idx = np.argpartition(dist_sq, n_nearest - 1)[:n_nearest]

        # 4. Process only the shortlisted candidates (reduces dict creation overhead)
        for idx in nearest_idx:
            d_sq = dist_sq[idx]
            if d_sq > state.effective_food_range ** 2:
                continue
            dist = math.sqrt(d_sq)
            food_pos = food_positions[idx]

            # ===== Visual field angle filtering (vision + lateral line) =====
            if forward_xz is not None:
                to_food = (food_pos - input_data.agent_position) / (dist + 1e-8)
                if not self._in_visual_field(forward_xz, to_food, dist):
                    continue

            # ===== Line-of-sight check (obstacles + tank walls) =====
            obstacle_field = getattr(input_data, 'obstacle_field', None)
            tank_geometry = getattr(input_data, 'tank_geometry', None)
            if obstacle_field is not None:
                if not obstacle_field.check_line_of_sight(
                        input_data.agent_position, food_pos, tank_geometry
                ):
                    continue  # Occluded by obstacle or tank wall
            rel_pos = (food_pos - input_data.agent_position) / (dist + 1e-8)

            state.observed_food.append({
                'position': food_pos,
                'distance': dist,
                'relative_position': rel_pos,
                'velocity': input_data.food_positions[idx].get('velocity', np.zeros(3))
            })

        # 5. Final sort on shortlisted items only (negligible cost)
        state.observed_food.sort(key=lambda x: x['distance'])
        if state.observed_food:
            state.nearest_food_distance = state.observed_food[0]['distance']
            state.nearest_food_position = state.observed_food[0]['position']

    def _perceive_threats(self, state: PerceptionState, input_data: PerceptionInput) -> None:
        """Compute instantaneous threat distance with reaction delay buffering.

        Identifies the closest threat among observed conspecifics and appends
        the detection to the temporal queue for delayed response processing.

        Args:
            state: Perception state containing observed fish and threat queue.
            input_data: Current perception input (unused directly but kept for
                interface consistency).
        """
        # Instantaneous threat distance (actual detection)
        instant_threat_distance = float('inf')

        for fish in state.observed_fish:
            if fish['is_threat'] and fish['distance'] < instant_threat_distance:
                instant_threat_distance = fish['distance']

        # Append instantaneous threat to the temporal queue
        state.threat_detection_queue.append({
            'distance': instant_threat_distance,
            'timestamp': 0  # Processed in _process_reaction_delay
        })

        # Update instantaneous threat distance (for internal computation)
        state.nearest_threat_distance = instant_threat_distance

    def _process_reaction_delay(self, state: PerceptionState, input_data: PerceptionInput) -> None:
        """Apply reaction delay to threat perception.

        During resting states, a temporal delay is imposed between threat
        detection and the agent's ability to respond, modeling the reduced
        alertness characteristic of quiescent fish (Reebs, 2002).

        Args:
            state: Perception state with threat queue and delay timer.
            input_data: Input containing time_step for delay computation.
        """
        rc = self.rc
        dt = input_data.time_step

        # Determine reaction delay duration based on activity state
        if state.activity_state == ActivityState.RESTING:
            reaction_delay = rc.rest_reaction_delay
        else:
            reaction_delay = rc.active_reaction_delay

        # Update reaction delay timer
        if state.nearest_threat_distance < 1.0:
            state.reaction_delay_timer += dt
        else:
            state.reaction_delay_timer = 0  # Reset when no threat present

        # Compute the number of steps to look back in the queue
        delay_steps = int(reaction_delay / dt)

        # Retrieve the delayed threat distance from the temporal queue
        if len(state.threat_detection_queue) > delay_steps:
            delayed_entry = state.threat_detection_queue[-(delay_steps + 1)]
            state.delayed_threat_distance = delayed_entry['distance']
        else:
            # Insufficient history in the queue
            state.delayed_threat_distance = float('inf')

        if self.debug and state.nearest_threat_distance < 0.5:
            print(f"[THREAT] instant={state.nearest_threat_distance:.2f}m, "
                  f"delayed={state.delayed_threat_distance:.2f}m, "
                  f"delay={reaction_delay:.2f}s")

    def _detect_obstacles(self, state: PerceptionState, input_data: PerceptionInput) -> None:
        """Perform ray-cast obstacle detection with dual-layer sensing.

        Casts 8 rays in cardinal and inter-cardinal directions. Rays falling
        outside the visual field of view are truncated to the lateral line
        range, preserving near-field mechanoreceptive awareness while limiting
        far-field detection to the forward visual cone.

        Args:
            state: Perception state whose obstacle_distances will be updated.
            input_data: Environment snapshot with geometry and obstacle data.
        """
        agent_pos = input_data.agent_position
        agent_vel = input_data.agent_velocity
        heading = getattr(input_data, 'agent_heading', None)
        if heading is not None and np.linalg.norm(heading) > 0.01:
            forward = np.array([heading[0], 0.0, heading[2]], dtype=np.float32)
            forward = forward / (np.linalg.norm(forward) + 1e-8)
        elif np.linalg.norm(agent_vel) > 0.01:
            forward = agent_vel / np.linalg.norm(agent_vel)
        else:
            forward = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        up = np.array([0.0, 1.0, 0.0])
        right = np.cross(up, forward)
        if np.linalg.norm(right) < 0.01:
            right = np.array([0.0, 0.0, 1.0])
        else:
            right = right / np.linalg.norm(right)
        up = np.cross(forward, right)
        up = up / (np.linalg.norm(up) + 1e-8)

        directions = [
            forward, -forward, -right, right,
            (forward + up) / math.sqrt(2), (forward - up) / math.sqrt(2),
            (-right + up) / math.sqrt(2), (right - up) / math.sqrt(2),
        ]

        # Visual field angle threshold (horizontal plane): rays outside FOV
        # are truncated to lateral line range
        half_fov_cos = math.cos(math.radians(self.c.vision_angle / 2.0))
        lateral_dist = self.c.lateral_line_range
        full_dist = self.c.max_obstacle_distance

        # Horizontal component of forward vector (for FOV membership test)
        forward_xz = np.array([forward[0], 0.0, forward[2]], dtype=np.float32)
        forward_xz_norm = np.linalg.norm(forward_xz)
        if forward_xz_norm > 0.01:
            forward_xz = forward_xz / forward_xz_norm
        else:
            forward_xz = None  # Fish oriented vertically; degrade to full-range omnidirectional

        for i, direction in enumerate(directions):
            if forward_xz is not None:
                dir_xz = np.array([direction[0], 0.0, direction[2]], dtype=np.float32)
                xz_norm = np.linalg.norm(dir_xz)
                if xz_norm > 0.01:
                    cos_a = float(np.dot(forward_xz, dir_xz / xz_norm))
                    # Outside FOV: truncate to lateral line distance (retain near-field sensing)
                    max_dist = full_dist if cos_a >= half_fov_cos else lateral_dist
                else:
                    # Pure vertical ray (up/down); not constrained by horizontal FOV
                    max_dist = full_dist
            else:
                max_dist = full_dist

            state.obstacle_distances[i] = self._cast_ray(agent_pos, direction, input_data,
                                                         max_dist=max_dist)

    def _cast_ray(self, origin: np.ndarray, direction: np.ndarray,
                  input_data: PerceptionInput,
                  max_dist: Optional[float] = None) -> float:
        """Cast a single perception ray and return normalized hit distance.

        Checks intersections in order: tank walls -> floor/surface -> obstacles.
        Returns the nearest collision distance normalized to [0, 1].

        Args:
            origin: Ray origin in world coordinates (m).
            direction: Ray direction vector (will be normalized internally).
            input_data: Environment data for geometry intersection tests.
            max_dist: Override for maximum ray distance (used for lateral line
                truncation outside the visual field).

        Returns:
            Normalized distance to nearest obstacle in [0, 1], where 1.0 means
            no collision detected within max_dist.
        """
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        max_distance = max_dist if max_dist is not None else self.c.max_obstacle_distance
        distances = []

        # ===== 1. Tank walls =====
        tank_geo = getattr(input_data, 'tank_geometry', None)
        if tank_geo is not None:
            # Horizontal wall intersection
            wall_dist = tank_geo.ray_to_wall_xz(origin, direction)
            if 0.01 < wall_dist < float('inf'):
                distances.append(wall_dist)
            # Vertical surfaces (floor / water surface)
            vert_dist = tank_geo.ray_to_vertical_surface(origin[1], direction[1])
            if 0.01 < vert_dist < float('inf'):
                distances.append(vert_dist)
        else:
            # Fallback: original circular tank ray-cylinder intersection
            ox, oz, dx, dz = origin[0], origin[2], direction[0], direction[2]
            a = dx ** 2 + dz ** 2
            if abs(a) > 1e-8:
                b = 2 * (ox * dx + oz * dz)
                c = ox ** 2 + oz ** 2 - input_data.tank_radius ** 2
                discriminant = b ** 2 - 4 * a * c
                if discriminant >= 0:
                    t1 = (-b - math.sqrt(discriminant)) / (2 * a)
                    t2 = (-b + math.sqrt(discriminant)) / (2 * a)
                    for t in [t1, t2]:
                        if t > 0.01:
                            distances.append(t)

            dy = direction[1]
            if abs(dy) > 1e-8:
                t_bottom = (-input_data.tank_depth - origin[1]) / dy
                t_surface = (0.0 - origin[1]) / dy
                for t in [t_bottom, t_surface]:
                    if t > 0.01:
                        distances.append(t)

        # ===== 2. Obstacles =====
        obs_field = getattr(input_data, 'obstacle_field', None)
        if obs_field is not None:
            obs_dist = obs_field.cast_ray(origin, direction, max_distance)
            if obs_dist < max_distance:
                distances.append(obs_dist)

        # Return nearest normalized distance
        if distances:
            return min(min(distances) / max_distance, 1.0)
        return 1.0

    def _perceive_boundaries(self, state: PerceptionState,
                             input_data: PerceptionInput) -> None:
        """Compute boundary proximity and repulsion direction.

        Supports both custom TankGeometry instances and the default circular
        tank fallback. Updates the boundary vector to point away from the
        nearest enclosure surface.

        Args:
            state: Perception state to update with boundary information.
            input_data: Environment data with tank geometry.
        """
        agent_pos = input_data.agent_position
        tank_geo = getattr(input_data, 'tank_geometry', None)

        if tank_geo is not None:
            dist_to_wall, wall_normal = tank_geo.nearest_wall_info(
                agent_pos[0], agent_pos[2]
            )
            depth = tank_geo.depth
        else:
            # Fallback: original circular tank
            horizontal_dist = math.sqrt(agent_pos[0] ** 2 + agent_pos[2] ** 2)
            dist_to_wall = input_data.tank_radius - horizontal_dist
            depth = input_data.tank_depth

            if horizontal_dist > 1e-6:
                wall_normal = np.array([
                    -agent_pos[0] / horizontal_dist,
                    0.0,
                    -agent_pos[2] / horizontal_dist
                ])
            else:
                wall_normal = np.array([0.0, 0.0, 0.0])

        dist_to_bottom = agent_pos[1] + depth
        dist_to_surface = abs(agent_pos[1])

        min_dist = min(dist_to_wall, dist_to_bottom, dist_to_surface)
        state.min_boundary_distance = min_dist

        if min_dist == dist_to_wall:
            state.boundary_vector = wall_normal.astype(np.float32)
        elif min_dist == dist_to_bottom:
            state.boundary_vector = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        else:
            state.boundary_vector = np.array([0.0, -1.0, 0.0], dtype=np.float32)

    def _perceive_surface(self, state: PerceptionState, input_data: PerceptionInput) -> None:
        """Compute normalized distance to the water surface.

        Args:
            state: Perception state to update with surface proximity.
            input_data: Environment data containing agent position.
        """
        distance = abs(input_data.agent_position[1])
        state.surface_distance = min(distance / 0.5, 1.0)
        state.is_near_surface = distance < 0.1

    # ==================== Observation vector generation ====================

    def get_normalized_fish_observation(self, state: PerceptionState,
                                        agent_position: np.ndarray,
                                        agent_velocity: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate the normalized conspecific observation sub-vector.

        Produces an 11-dimensional feature vector per observed fish slot:
        direction (3) + velocity (3) + distance (1) + mass ratio (1) +
        is_threat (1) + is_prey (1) + closing_speed (1).

        The closing_speed component encodes the rate of approach between the
        agent and the target fish (positive = closing, negative = separating),
        enabling the policy to assess pursuit effectiveness.

        Args:
            state: Current perception state with observed fish list.
            agent_position: World-space position of the focal agent (m).
            agent_velocity: Velocity of the focal agent (m/s). Defaults to
                zero vector if not provided.

        Returns:
            Flat numpy array of shape (max_fish * 11,) with float32 dtype.
        """
        obs = []

        max_distance = CONFIG.perception.vision_range
        pc = CONFIG.physics
        _agent_vel = agent_velocity if agent_velocity is not None else np.zeros(3)

        for i in range(self.max_fish):  # = 3
            if i < len(state.observed_fish):
                fish = state.observed_fish[i]

                # Direction (3 dimensions)
                obs.extend([
                    fish['relative_position'][0],
                    fish['relative_position'][1],
                    fish['relative_position'][2]
                ])

                # Normalized velocity (3 dimensions)
                fish_mass = fish.get('body_mass', 20.0)
                fish_length = CONFIG.growth.length_weight_a * (fish_mass ** (1.0 / CONFIG.growth.length_weight_b))
                length_cm = fish_length * 100
                if length_cm <= pc.swim_speed_small_threshold:
                    bl_per_s = pc.swim_speed_small_bl
                elif length_cm <= pc.swim_speed_medium_threshold:
                    bl_per_s = pc.swim_speed_medium_bl
                else:
                    bl_per_s = pc.swim_speed_large_bl
                max_fish_speed = max(fish_length * bl_per_s * pc.burst_speed_multiplier, 0.02)

                velocity = fish.get('velocity', np.zeros(3))
                obs.extend([
                    np.clip(velocity[0] / max_fish_speed, -1, 1),
                    np.clip(velocity[1] / max_fish_speed, -1, 1),
                    np.clip(velocity[2] / max_fish_speed, -1, 1)
                ])

                # Distance (1 dimension)
                obs.append(np.clip(fish['distance'] / max_distance, 0, 1))

                # Mass ratio (1 dimension)
                obs.append(fish['mass_ratio'] / 5.0)

                # Ecological role flags (2 dimensions)
                obs.append(1.0 if fish['is_threat'] else 0.0)
                obs.append(1.0 if fish['is_prey']   else 0.0)

                # Closing speed (1 dimension): positive = approaching, negative = separating
                # closing_speed = dot(agent_vel - fish_vel, to_fish_dir)
                to_fish_dir = np.array(fish['relative_position'], dtype=np.float32)
                dir_norm = np.linalg.norm(to_fish_dir)
                if dir_norm > 1e-6:
                    to_fish_dir = to_fish_dir / dir_norm
                    rel_vel = _agent_vel - np.array(velocity, dtype=np.float32)
                    closing = float(np.dot(rel_vel, to_fish_dir))
                    obs.append(np.clip(closing / max(max_fish_speed, 0.1), -1, 1))
                else:
                    obs.append(0.0)
            else:
                obs.extend([0.0] * 11)

        return np.array(obs, dtype=np.float32)

    def get_normalized_food_observation(self, state: PerceptionState,
                                        agent_position: np.ndarray,
                                        agent_velocity: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate the normalized food observation sub-vector.

        Produces an 8-dimensional feature vector per observed food slot:
        direction (3) + velocity (3) + distance (1) + closing_speed (1).

        The closing_speed component allows the policy to detect overshoot
        scenarios during food pursuit maneuvers.

        Args:
            state: Current perception state with observed food list.
            agent_position: World-space position of the focal agent (m).
            agent_velocity: Velocity of the focal agent (m/s). Defaults to
                zero vector if not provided.

        Returns:
            Flat numpy array of shape (max_food * 8,) with float32 dtype.
        """
        obs = []
        _agent_vel = agent_velocity if agent_velocity is not None else np.zeros(3)

        # Retrieve normalization parameters from config
        fc = CONFIG.feeding
        max_food_speed = max(fc.sinking_speed_max, fc.floating_speed_max)  # ~0.02 m/s
        max_distance = CONFIG.perception.food_detection_range  # 3.0 m
        # Closing speed normalized by approximate max fish speed (~0.3 m/s for 20g bass)
        max_closing_speed = 0.3

        for i in range(self.max_food):  # = 3
            if i < len(state.observed_food):
                food = state.observed_food[i]

                # Direction (3 dimensions)
                obs.extend([
                    food['relative_position'][0],
                    food['relative_position'][1],
                    food['relative_position'][2]
                ])

                # Velocity (3 dimensions)
                velocity = food.get('velocity', np.zeros(3))
                obs.extend([
                    np.clip(velocity[0] / max_food_speed, -1, 1),
                    np.clip(velocity[1] / max_food_speed, -1, 1),
                    np.clip(velocity[2] / max_food_speed, -1, 1)
                ])

                # Distance (1 dimension)
                obs.append(np.clip(food['distance'] / max_distance, 0, 1))

                # Closing speed (1 dimension): positive = approaching food, negative = separating
                to_food_dir = np.array(food['relative_position'], dtype=np.float32)
                dir_norm = np.linalg.norm(to_food_dir)
                if dir_norm > 1e-6:
                    to_food_dir = to_food_dir / dir_norm
                    rel_vel = _agent_vel - np.array(velocity, dtype=np.float32)
                    closing = float(np.dot(rel_vel, to_food_dir))
                    obs.append(np.clip(closing / max_closing_speed, -1, 1))
                else:
                    obs.append(0.0)
            else:
                obs.extend([0.0] * 8)

        return np.array(obs, dtype=np.float32)

    def get_surface_observation(self, state: PerceptionState) -> np.ndarray:
        """Generate the water surface proximity observation.

        Args:
            state: Current perception state with surface distance.

        Returns:
            Array of shape (2,): [normalized_surface_distance, is_near_surface].
        """
        return np.array([state.surface_distance, 1.0 if state.is_near_surface else 0.0], dtype=np.float32)

    # ==================== Delayed threat query interface ====================

    def get_delayed_threat_distance(self, state: PerceptionState) -> float:
        """Retrieve the reaction-delay-adjusted threat distance.

        The RL agent should use this value for decision-making rather than the
        instantaneous threat distance, enabling the policy to learn that
        resting states incur slower threat responses.

        Args:
            state: Current perception state with delayed threat computation.

        Returns:
            Effective threat distance after reaction delay (m).
        """
        return state.delayed_threat_distance

    def get_effective_ranges(self, state: PerceptionState) -> Dict[str, float]:
        """Query current effective perception ranges.

        Args:
            state: Current perception state with range information.

        Returns:
            Dictionary with keys 'vision_range', 'food_range', and 'is_resting'.
        """
        return {
            'vision_range': state.effective_vision_range,
            'food_range': state.effective_food_range,
            'is_resting': state.activity_state == ActivityState.RESTING
        }

    def set_debug(self, enabled: bool) -> None:
        """Enable or disable debug output for threat detection logging.

        Args:
            enabled: If True, threat proximity events are printed to stdout.
        """
        self.debug = enabled


def create_perception_system() -> PerceptionSystem:
    """Factory function for PerceptionSystem instantiation.

    Returns:
        A new PerceptionSystem instance with default configuration.
    """
    return PerceptionSystem()


def create_perception_state() -> PerceptionState:
    """Factory function for PerceptionState instantiation.

    Returns:
        A new PerceptionState instance with default field values.
    """
    return PerceptionState()


# ============================================================
# Self-test
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Perception System Test (Resting-State Enhanced)")
    print("=" * 60)

    perception = create_perception_system()
    perception.set_debug(True)
    state = create_perception_state()

    # Simulate a threatening conspecific
    fish_states = [
        {'position': np.array([0.3, -0.3, 0.1]), 'body_mass': 80.0,
         'behavior_type': 'aggressive', 'size_category': 'large', 'is_chasing': True},
    ]

    food_positions = [
        {'position': np.array([0.15, 0.0, 0.1])},
    ]

    print("\nTest 1: Active state perception")
    print("-" * 40)

    input_data = PerceptionInput(
        agent_position=np.array([0.0, -0.3, 0.0]),
        agent_velocity=np.array([0.1, 0.0, 0.0]),
        agent_mass=25.0,
        fish_states=fish_states,
        food_positions=food_positions,
        activity_state=ActivityState.ACTIVE
    )

    perception.update(state, input_data)
    ranges = perception.get_effective_ranges(state)
    print(f"Effective vision range: {ranges['vision_range']:.2f} m")
    print(f"Effective food range: {ranges['food_range']:.2f} m")
    print(f"Observed fish: {len(state.observed_fish)}")
    print(f"Observed food: {len(state.observed_food)}")

    print("\nTest 2: Resting state perception")
    print("-" * 40)

    state = create_perception_state()
    input_data.activity_state = ActivityState.RESTING

    perception.update(state, input_data)
    ranges = perception.get_effective_ranges(state)
    print(f"Effective vision range: {ranges['vision_range']:.2f} m (reduced)")
    print(f"Effective food range: {ranges['food_range']:.2f} m (reduced)")
    print(f"Observed fish: {len(state.observed_fish)}")
    print(f"Observed food: {len(state.observed_food)}")

    print("\nTest 3: Reaction delay simulation")
    print("-" * 40)

    # Simulate a threat approaching gradually
    state = create_perception_state()

    for step in range(20):
        # Threat approaches incrementally
        threat_distance = 0.5 - step * 0.02
        fish_states[0]['position'] = np.array([threat_distance, -0.3, 0.0])

        input_data = PerceptionInput(
            agent_position=np.array([0.0, -0.3, 0.0]),
            agent_velocity=np.array([0.0, 0.0, 0.0]),
            agent_mass=25.0,
            fish_states=fish_states,
            food_positions=food_positions,
            activity_state=ActivityState.RESTING,
            time_step=0.3
        )

        perception.update(state, input_data)
        delayed_threat = perception.get_delayed_threat_distance(state)

        if step % 5 == 0:
            print(f"Step {step}: actual_threat={state.nearest_threat_distance:.2f} m, "
                  f"delayed_perception={delayed_threat:.2f} m")

    print("\nAll tests completed.")
