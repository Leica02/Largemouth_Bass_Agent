#!/usr/bin/env python3
"""
Obstacle System Module
======================

Implements rock and box obstacle generation, collision detection,
line-of-sight checks, and layout generators (corridor, reef, random, maze,
raceway, stream channel, U-shape channel, and natural pond). All obstacles
support unified interfaces for collision resolution, ray casting, and
position validation.

Usage:
    from systems.obstacles import generate_obstacles, ObstacleField

    obstacle_field = generate_obstacles(tank_geometry, config, spawn_pos)

    # Collision detection
    result = obstacle_field.check_collision(position)

    # Line-of-sight check
    can_see = obstacle_field.check_line_of_sight(point_a, point_b)

    # Ray casting (used by the perception system's 8 directional rays)
    dist = obstacle_field.cast_ray(origin, direction, max_dist)
"""

import numpy as np
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any, Union


# ============================================================
# Collision Result
# ============================================================

@dataclass
class CollisionResult:
    """Result of a collision detection query.

    Attributes:
        collided: Whether a collision was detected.
        pushed_position: Safe position after push-out, or None.
        normal: Collision surface normal (pointing outward), or None.
        obstacle_index: Index of the collided obstacle, or -1.
    """
    collided: bool = False
    pushed_position: Optional[np.ndarray] = None
    normal: Optional[np.ndarray] = None
    obstacle_index: int = -1


# ============================================================
# Rock Obstacle
# ============================================================

class RockObstacle:
    """Spherical rock obstacle.

    All rock obstacles are modeled as spheres:
    - Mathematically simple (ray-sphere intersection = quadratic equation)
    - Unambiguous collision handling (push to sphere surface)
    - Ecologically meaningful (stones, boulders)

    Attributes:
        center: Sphere center position [x, y, z].
        radius: Sphere radius (m).
    """

    def __init__(self, center: np.ndarray, radius: float):
        self.center = np.array(center, dtype=np.float32)
        self.radius = float(radius)

    def contains_point(self, point: np.ndarray) -> bool:
        """Checks whether a point lies inside the obstacle.

        Args:
            point: 3D position to test.

        Returns:
            True if the point is inside the sphere.
        """
        dist = np.linalg.norm(point - self.center)
        return dist < self.radius

    def distance_to_surface(self, point: np.ndarray) -> float:
        """Computes the signed distance from a point to the sphere surface.

        Args:
            point: 3D position.

        Returns:
            Distance to surface (negative if inside).
        """
        return np.linalg.norm(point - self.center) - self.radius

    def nearest_surface_point(self, point: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Computes the nearest point on the sphere surface and the outward
        normal.

        Args:
            point: 3D query position.

        Returns:
            Tuple of (surface_point, outward_normal).
        """
        direction = point - self.center
        dist = np.linalg.norm(direction)

        if dist < 1e-6:
            # Exactly at center; choose an arbitrary push direction
            direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            dist = 1.0

        normal = (direction / dist).astype(np.float32)
        surface_point = (self.center + normal * self.radius).astype(np.float32)
        return surface_point, normal

    def ray_intersect(self, origin: np.ndarray, direction: np.ndarray) -> Optional[float]:
        """Performs ray-sphere intersection detection.

        Mathematical derivation:
            Ray P(t) = O + t*D
            Sphere |P - C|^2 = r^2

            Let L = O - C
            Expanding: t^2 + 2(L.D)t + (L.L - r^2) = 0

            Discriminant delta = (L.D)^2 - (L.L - r^2)
            delta < 0 implies no intersection
            t = -(L.D) +/- sqrt(delta)
            Take the smallest positive t.

        Args:
            origin: Ray origin point.
            direction: Ray direction (does not need to be normalized).

        Returns:
            Intersection distance, or None if no intersection.
        """
        dir_norm = np.linalg.norm(direction)
        if dir_norm < 1e-8:
            return None
        d = direction / dir_norm

        L = origin - self.center

        # Quadratic equation coefficients
        # a = 1 (d is normalized)
        b = float(np.dot(L, d))
        c = float(np.dot(L, L)) - self.radius ** 2

        discriminant = b ** 2 - c

        if discriminant < 0:
            return None

        sqrt_disc = math.sqrt(discriminant)
        t1 = -b - sqrt_disc
        t2 = -b + sqrt_disc

        # Take smallest positive value
        if t1 > 0.001:
            return t1
        elif t2 > 0.001:
            return t2
        else:
            return None  # Ray origin inside sphere or sphere behind ray


class BoxObstacle:
    """Axis-aligned bounding box (AABB) obstacle.

    Attributes:
        center: Box center position [x, y, z].
        size: Box dimensions [width, height, depth].
        half_size: Half of the box dimensions.
    """

    def __init__(self, center: np.ndarray, size: np.ndarray):
        self.center = np.array(center, dtype=np.float32)
        self.size = np.array(size, dtype=np.float32)  # [width, height, depth]
        self.half_size = self.size / 2.0

    def contains_point(self, point: np.ndarray) -> bool:
        """Checks whether a point lies inside the box.

        Args:
            point: 3D position to test.

        Returns:
            True if the point is inside the AABB.
        """
        diff = np.abs(point - self.center)
        return np.all(diff <= self.half_size)

    def distance_to_surface(self, point: np.ndarray) -> float:
        """Computes the signed distance from a point to the box surface.

        Args:
            point: 3D position.

        Returns:
            Positive if outside, negative if inside.
        """
        diff = np.abs(point - self.center) - self.half_size
        outside_dist = np.linalg.norm(np.maximum(diff, 0))
        inside_dist = np.min(diff)
        return outside_dist if np.any(diff > 0) else inside_dist

    def nearest_surface_point(self, point: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Computes the nearest point on the box surface and the outward
        normal.

        Args:
            point: 3D query position.

        Returns:
            Tuple of (surface_point, outward_normal).
        """
        clamped = np.clip(point, self.center - self.half_size, self.center + self.half_size)
        diff = point - clamped
        dist = np.linalg.norm(diff)
        if dist < 1e-6:
            normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        else:
            normal = (diff / dist).astype(np.float32)
        return clamped.astype(np.float32), normal

    def ray_intersect(self, origin: np.ndarray, direction: np.ndarray) -> Optional[float]:
        """Performs ray-AABB intersection detection using the slab method.

        Args:
            origin: Ray origin point.
            direction: Ray direction (does not need to be normalized).

        Returns:
            Intersection distance, or None if no intersection.
        """
        dir_norm = np.linalg.norm(direction)
        if dir_norm < 1e-8:
            return None
        d = direction / dir_norm

        box_min = self.center - self.half_size
        box_max = self.center + self.half_size

        t_min = -np.inf
        t_max = np.inf

        for i in range(3):
            if abs(d[i]) < 1e-8:
                if origin[i] < box_min[i] or origin[i] > box_max[i]:
                    return None
            else:
                t1 = (box_min[i] - origin[i]) / d[i]
                t2 = (box_max[i] - origin[i]) / d[i]
                t_min = max(t_min, min(t1, t2))
                t_max = min(t_max, max(t1, t2))

        if t_min > t_max or t_max < 0.001:
            return None
        return t_min if t_min > 0.001 else t_max


# ============================================================
# Obstacle Field
# ============================================================

class ObstacleField:
    """Collection of obstacles providing a unified interface for collision,
    line-of-sight, and ray-cast queries.

    This is the single entry point for all subsystems to access obstacle
    information.
    """

    def __init__(self, obstacles: List[Union[RockObstacle, BoxObstacle]] = None):
        self.obstacles: List[Union[RockObstacle, BoxObstacle]] = obstacles or []

    def add_obstacle(self, obstacle: Union[RockObstacle, BoxObstacle]) -> None:
        """Adds an obstacle to the field.

        Args:
            obstacle: A RockObstacle or BoxObstacle instance.
        """
        self.obstacles.append(obstacle)

    @property
    def count(self) -> int:
        return len(self.obstacles)

    # ==================== Collision Detection ====================

    def check_collision(self, position: np.ndarray,
                        body_radius: float = 0.01) -> CollisionResult:
        """Checks whether a position collides with any obstacle.

        Collision behavior is consistent with tank wall collisions:
        - Pushes to a safe position (outside obstacle surface)
        - Returns the collision surface normal

        Args:
            position: Position to test [x, y, z].
            body_radius: Fish body radius (default 1cm, spherical approximation).

        Returns:
            CollisionResult with collision details.
        """
        for i, obs in enumerate(self.obstacles):
            if hasattr(obs, 'radius'):
                # RockObstacle
                dist = np.linalg.norm(position - obs.center)
                collision_dist = obs.radius + body_radius
                if dist < collision_dist:
                    _, normal = obs.nearest_surface_point(position)
                    pushed = obs.center + normal * (collision_dist + 0.005)
                    return CollisionResult(
                        collided=True,
                        pushed_position=pushed.astype(np.float32),
                        normal=normal.astype(np.float32),
                        obstacle_index=i
                    )
            else:
                # BoxObstacle
                if obs.contains_point(position):
                    _, normal = obs.nearest_surface_point(position)
                    pushed = position + normal * (body_radius + 0.005)
                    return CollisionResult(
                        collided=True,
                        pushed_position=pushed.astype(np.float32),
                        normal=normal.astype(np.float32),
                        obstacle_index=i
                    )

        return CollisionResult(collided=False)

    def resolve_collision_velocity(self, velocity: np.ndarray,
                                   normal: np.ndarray) -> np.ndarray:
        """Resolves post-collision velocity, consistent with tank wall behavior.

        - Normal velocity is zeroed (no bounce)
        - Tangential velocity is damped (friction)
        - Heading is not modified

        Args:
            velocity: Pre-collision velocity vector.
            normal: Collision surface outward normal.

        Returns:
            Post-collision velocity vector.
        """
        v_dot_n = np.dot(velocity, normal)

        if v_dot_n < 0:
            # Moving into obstacle: zero normal component
            velocity = velocity - v_dot_n * normal
            # Tangential velocity friction damping
            velocity *= 0.5

        return velocity.astype(np.float32)

    # ==================== Line-of-Sight Detection ====================

    def check_line_of_sight(self, point_a: np.ndarray,
                            point_b: np.ndarray,
                            tank_geometry=None) -> bool:
        """Checks whether there is a clear line of sight between two points
        (not occluded by obstacles or tank walls).

        Args:
            point_a: Observer position.
            point_b: Target position.
            tank_geometry: Tank geometry instance (optional); if provided, also
                checks for wall occlusion.

        Returns:
            True if visible, False if occluded.
        """
        direction = point_b - point_a
        max_dist = np.linalg.norm(direction)

        if max_dist < 1e-6:
            return True

        # 1. Obstacle occlusion check
        for obs in self.obstacles:
            t = obs.ray_intersect(point_a, direction)
            if t is not None and t < max_dist:
                return False

        # 2. Tank wall occlusion check (if tank geometry is provided)
        if tank_geometry is not None:
            wall_t = tank_geometry.ray_to_wall_xz(point_a, direction)
            if wall_t < max_dist:
                return False
            vert_t = tank_geometry.ray_to_vertical_surface(point_a[1], direction[1])
            if vert_t < max_dist:
                return False

        return True

    # ==================== Ray Casting ====================

    def cast_ray(self, origin: np.ndarray, direction: np.ndarray,
                 max_dist: float = 10.0) -> float:
        """Casts a ray and returns the distance to the nearest obstacle hit.

        Used by the perception system's 8-directional ray detection.

        Args:
            origin: Ray origin point.
            direction: Ray direction (does not need to be normalized).
            max_dist: Maximum detection distance.

        Returns:
            Hit distance, or max_dist if no intersection.
        """
        min_dist = max_dist

        for obs in self.obstacles:
            t = obs.ray_intersect(origin, direction)
            if t is not None and t < min_dist:
                min_dist = t

        return min_dist

    # ==================== Position Validation ====================

    def is_valid_position(self, position: np.ndarray,
                          min_clearance: float = 0.05) -> bool:
        """Checks whether a position is safe (not inside or too close to any
        obstacle).

        Used for food placement, NPC fish spawning, and agent spawn point
        validation.

        Args:
            position: 3D position to validate.
            min_clearance: Minimum required distance from obstacle surfaces.

        Returns:
            True if the position is safe.
        """
        for obs in self.obstacles:
            # Compatible with both BoxObstacle and RockObstacle
            if hasattr(obs, 'radius'):
                # RockObstacle
                if np.linalg.norm(position - obs.center) < obs.radius + min_clearance:
                    return False
            elif hasattr(obs, 'contains_point'):
                # BoxObstacle
                if obs.contains_point(position):
                    return False
                # Check proximity to box boundary
                dist = obs.distance_to_surface(position)
                if dist < min_clearance:
                    return False
        return True

    def get_obstacle_info(self) -> List[dict]:
        """Returns information about all obstacles (for debugging and info
        output).

        Returns:
            List of dictionaries with obstacle center, radius, and type.
        """
        return [
            {
                'center': obs.center.tolist(),
                'radius': obs.radius,
                'type': 'rock'
            }
            for obs in self.obstacles
        ]


# ============================================================
# Obstacle Generators
# ============================================================

def _place_rock(obstacle_field: ObstacleField, pos: np.ndarray,
                radius: float, tank_geometry, spawn_position: np.ndarray,
                oc) -> bool:
    """Validates placement constraints and adds a rock to the field if valid.

    Shared by random layout and corridor layout to avoid code duplication.

    Args:
        obstacle_field: Target obstacle field.
        pos: Proposed rock center position.
        radius: Rock radius.
        tank_geometry: Tank geometry for boundary checks.
        spawn_position: Agent spawn position (exclusion zone center).
        oc: Obstacle configuration object.

    Returns:
        True if placement succeeded, False otherwise.
    """
    if spawn_position is not None:
        if np.linalg.norm(pos - spawn_position) < oc.spawn_exclusion_radius + radius:
            return False

    for existing in obstacle_field.obstacles:
        dist = np.linalg.norm(pos - existing.center)
        # Compatible with BoxObstacle (use max dimension / 2 as effective radius)
        existing_radius = getattr(existing, 'radius', np.max(existing.size) / 2.0 if hasattr(existing, 'size') else 0.2)
        if dist < existing_radius + radius + oc.min_distance_between:
            return False

    if not tank_geometry.contains_point_xz(pos[0], pos[2]):
        return False
    if pos[1] + radius > -0.05:
        return False
    if pos[1] - radius < -tank_geometry.depth + 0.03:
        return False

    obstacle_field.add_obstacle(RockObstacle(pos, radius))
    return True


def _generate_random_layout(tank_geometry, oc, spawn_position: np.ndarray,
                            obstacle_field: ObstacleField) -> None:
    """Generates a random scatter layout (baseline obstacle avoidance).

    Args:
        tank_geometry: Tank geometry instance.
        oc: Obstacle configuration object.
        spawn_position: Agent spawn position for exclusion zone.
        obstacle_field: Target obstacle field to populate.
    """
    num_obstacles = np.random.randint(oc.min_obstacles, oc.max_obstacles + 1)
    for _ in range(num_obstacles):
        for _ in range(50):
            radius = np.random.uniform(oc.rock_radius_min, oc.rock_radius_max)
            margin = radius + oc.min_distance_from_wall
            pos = tank_geometry.random_interior_point(margin=margin)
            if _place_rock(obstacle_field, pos, radius, tank_geometry,
                           spawn_position, oc):
                break


def _generate_corridor_layout(tank_geometry, oc, spawn_position: np.ndarray,
                               obstacle_field: ObstacleField) -> None:
    """Generates a corridor layout: spherical rocks arranged in two rows along
    an axis, forming a passage.

    Biological relevance: Simulates partition boards, pipes, or natural stone
    walls in aquaculture tanks. The agent must locate the passage entrance and
    navigate through rather than circumventing isolated obstacles.

    Layout logic:
    1. Randomly selects the corridor axis direction (X or Z)
    2. Uniformly places two rows of rocks (left wall + right wall) along the axis
    3. Randomly leaves a gap in the middle section as the passage entrance
    4. Passage width guarantees fish can pass (at least 2 x rock_radius_max)

    Args:
        tank_geometry: Tank geometry instance.
        oc: Obstacle configuration object.
        spawn_position: Agent spawn position for exclusion zone.
        obstacle_field: Target obstacle field to populate.
    """
    extents = tank_geometry.get_extents()
    depth = tank_geometry.depth

    # Determine available range based on tank shape
    if extents.get('shape') == 'circular':
        half_x = extents['radius'] * 0.75
        half_z = extents['radius'] * 0.75
    else:
        half_x = extents.get('width', 3.0) / 2.0 * 0.80
        half_z = extents.get('length', 3.0) / 2.0 * 0.80

    # Corridor axis: along Z means walls on both sides in X direction
    along_z = np.random.random() < 0.5

    rock_r = float(np.random.uniform(oc.rock_radius_min, oc.rock_radius_max))
    gap_between_rocks = oc.min_distance_between  # Gap between adjacent rocks

    # Corridor centerline offset (not always centered, for diversity)
    if along_z:
        corridor_center_x = np.random.uniform(-half_x * 0.4, half_x * 0.4)
        axis_half = half_z
    else:
        corridor_center_z = np.random.uniform(-half_z * 0.4, half_z * 0.4)
        axis_half = half_x

    # Passage width: wide enough for fish to pass, but not too wide
    passage_width = float(np.random.uniform(
        rock_r * 3.0,   # Minimum: 3x rock radius
        rock_r * 5.0    # Maximum: 5x rock radius
    ))
    wall_offset = passage_width / 2.0 + rock_r  # Wall centerline distance from corridor axis

    # Number of rocks per side, determined by axis length and rock diameter
    step = 2 * rock_r + gap_between_rocks
    num_rocks_per_side = max(2, int(axis_half * 1.6 / step))

    # Gap position (randomly leave a section empty as entrance)
    # Ensure gap is not at axis endpoints (to prevent bypassing from the end)
    gap_start_idx = np.random.randint(1, max(2, num_rocks_per_side - 2))
    gap_length = np.random.randint(1, 3)  # Gap spans 1-2 rock positions

    depth_center = -depth / 2.0  # Vertical placement at mid-depth

    for side in [-1, 1]:  # Left wall / right wall
        offset = side * wall_offset
        for i in range(num_rocks_per_side):
            # Skip gap positions to form passage entrance
            if gap_start_idx <= i < gap_start_idx + gap_length:
                continue

            axis_pos = -axis_half + step * (i + 0.5)

            # Slight random perturbation for natural appearance
            axis_pos += np.random.uniform(-rock_r * 0.3, rock_r * 0.3)
            depth_pos = depth_center + np.random.uniform(-0.05, 0.05)
            depth_pos = float(np.clip(depth_pos, -depth + rock_r + 0.05, -rock_r - 0.05))

            if along_z:
                pos = np.array([corridor_center_x + offset, depth_pos, axis_pos],
                                dtype=np.float32)
            else:
                pos = np.array([axis_pos, depth_pos, corridor_center_z + offset],
                                dtype=np.float32)

            _place_rock(obstacle_field, pos, rock_r, tank_geometry,
                        spawn_position, oc)


def _generate_cluster_layout(tank_geometry, oc, spawn_position: np.ndarray,
                              obstacle_field: ObstacleField) -> None:
    """Generates a cluster layout: 2-3 rock clusters with tightly packed stones.

    Biological relevance: Simulates underwater reef groups where the fish must
    navigate around entire reef clusters rather than individual stones.
    Generalization difficulty is between random layout and corridor layout.

    Args:
        tank_geometry: Tank geometry instance.
        oc: Obstacle configuration object.
        spawn_position: Agent spawn position for exclusion zone.
        obstacle_field: Target obstacle field to populate.
    """
    extents = tank_geometry.get_extents()
    depth = tank_geometry.depth

    num_clusters = np.random.randint(2, 4)
    rocks_per_cluster = np.random.randint(2, 5)

    for _ in range(num_clusters):
        # Cluster center
        margin = oc.rock_radius_max * 2 + oc.min_distance_from_wall
        for _ in range(30):
            cluster_center = tank_geometry.random_interior_point(margin=margin)
            if spawn_position is None or np.linalg.norm(
                    cluster_center - spawn_position) > oc.spawn_exclusion_radius * 1.5:
                break

        rock_r = float(np.random.uniform(oc.rock_radius_min, oc.rock_radius_max))
        spread = rock_r * 2.5  # Intra-cluster rock distribution radius

        for _ in range(rocks_per_cluster):
            offset = np.random.uniform(-spread, spread, 3).astype(np.float32)
            offset[1] *= 0.3  # Reduced vertical perturbation
            pos = cluster_center + offset
            _place_rock(obstacle_field, pos, rock_r, tank_geometry,
                        spawn_position, oc)


def _generate_reef_layout(tank_geometry, oc, spawn_position: np.ndarray,
                          obstacle_field: ObstacleField) -> None:
    """Generates a wild reef layout: 2-4 large irregular reef groups with
    varied-size spherical rocks tightly stacked, visually resembling coral
    reefs or submerged rock formations.

    Differences from cluster layout:
    - Larger rocks (radius 1.5-2.5x cluster)
    - More rocks per group (5-9), more tightly packed
    - Group centers intentionally biased toward tank edges, leaving central
      swimming space

    Args:
        tank_geometry: Tank geometry instance.
        oc: Obstacle configuration object.
        spawn_position: Agent spawn position for exclusion zone.
        obstacle_field: Target obstacle field to populate.
    """
    extents = tank_geometry.get_extents()
    depth = tank_geometry.depth

    # Reef rock radius is 1.5-2x larger than normal clusters
    rock_r_min = min(oc.rock_radius_max * 1.2, 0.25)
    rock_r_max = min(oc.rock_radius_max * 2.2, 0.40)

    num_reefs = np.random.randint(2, 5)
    rocks_per_reef = np.random.randint(4, 9)

    # Reef group centers near tank walls (15-40% radius from wall)
    if extents.get('shape') == 'circular':
        r_tank = extents['radius']
        reef_r_range = (r_tank * 0.40, r_tank * 0.72)
    else:
        hw = extents.get('width', 3.0) / 2.0
        hl = extents.get('length', 3.0) / 2.0
        reef_r_range = (min(hw, hl) * 0.40, min(hw, hl) * 0.80)

    for _ in range(num_reefs):
        # Sample reef group center (biased toward edge)
        for attempt in range(40):
            angle = np.random.uniform(0, 2 * np.pi)
            if extents.get('shape') == 'circular':
                reef_dist = np.random.uniform(*reef_r_range)
                cx = reef_dist * np.cos(angle)
                cz = reef_dist * np.sin(angle)
            else:
                cx = np.random.uniform(-hw * 0.75, hw * 0.75)
                cz = np.random.uniform(-hl * 0.75, hl * 0.75)
            cy = np.random.uniform(-depth * 0.75, -depth * 0.25)
            center = np.array([cx, cy, cz], dtype=np.float32)

            if not tank_geometry.contains_point_xz(cx, cz):
                continue
            if spawn_position is not None and np.linalg.norm(
                    center - spawn_position) < oc.spawn_exclusion_radius * 2.0:
                continue
            break

        # Stack varied-size rocks around the group center
        spread = float(np.random.uniform(rock_r_max * 1.5, rock_r_max * 3.0))
        for _ in range(rocks_per_reef):
            rock_r = float(np.random.uniform(rock_r_min, rock_r_max))
            offset = np.random.uniform(-spread, spread, 3).astype(np.float32)
            offset[1] *= 0.4
            pos = center + offset
            _place_rock(obstacle_field, pos, rock_r, tank_geometry,
                        spawn_position, oc)


def _generate_box_maze_layout(tank_geometry, oc, spawn_position: np.ndarray,
                               obstacle_field: ObstacleField) -> None:
    """Generates a multi-room maze layout using BoxObstacle partition walls
    with multiple passages.

    Layout logic:
    - Divides the tank into 4 quadrants with a wall in each
    - Each wall has a randomly placed gap (passage) at a different position
    - All 4 walls together form a maze-like room structure with multiple
      entrances/exits
    - Additional short walls in some quadrants increase complexity

    Args:
        tank_geometry: Tank geometry instance.
        oc: Obstacle configuration object.
        spawn_position: Agent spawn position for exclusion zone.
        obstacle_field: Target obstacle field to populate.
    """
    extents = tank_geometry.get_extents()
    depth = tank_geometry.depth

    if extents.get('shape') == 'circular':
        half_x = extents['radius'] * 0.65
        half_z = extents['radius'] * 0.65
    else:
        half_x = extents.get('width', 3.0) / 2.0 * 0.72
        half_z = extents.get('length', 3.0) / 2.0 * 0.72

    y_center = -depth / 2.0
    wall_height = depth * 0.65
    wt = 0.07  # Wall thickness

    # Passage width: 0.25-0.40m (sufficient for fish passage)
    gap = float(np.random.uniform(0.25, 0.40))

    # -- 4 main partition walls, each with a random gap --
    # Wall 1: Horizontal wall, left segment (X: -half_x to -gap/2), along Z~0
    z_offset = float(np.random.uniform(-half_z * 0.25, half_z * 0.25))
    seg_len = half_x - gap / 2
    if seg_len > 0.15:
        obstacle_field.add_obstacle(BoxObstacle(
            center=np.array([-(half_x / 2 + gap / 4), y_center, z_offset]),
            size=np.array([seg_len, wall_height, wt])
        ))
    # Wall 1 right segment (X: gap/2 to half_x)
    if seg_len > 0.15:
        obstacle_field.add_obstacle(BoxObstacle(
            center=np.array([(half_x / 2 + gap / 4), y_center, z_offset]),
            size=np.array([seg_len, wall_height, wt])
        ))

    # Wall 2: Vertical wall, upper segment (Z: -half_z to -gap/2), along X~0
    x_offset = float(np.random.uniform(-half_x * 0.25, half_x * 0.25))
    seg_len2 = half_z - gap / 2
    if seg_len2 > 0.15:
        obstacle_field.add_obstacle(BoxObstacle(
            center=np.array([x_offset, y_center, -(half_z / 2 + gap / 4)]),
            size=np.array([wt, wall_height, seg_len2])
        ))
    # Wall 2 lower segment (Z: gap/2 to half_z)
    if seg_len2 > 0.15:
        obstacle_field.add_obstacle(BoxObstacle(
            center=np.array([x_offset, y_center, (half_z / 2 + gap / 4)]),
            size=np.array([wt, wall_height, seg_len2])
        ))

    # -- Additional short walls in 1-2 quadrants for maze complexity --
    num_extra = np.random.randint(1, 3)
    for _ in range(num_extra):
        qx = np.random.choice([-1, 1])
        qz = np.random.choice([-1, 1])
        cx = qx * np.random.uniform(half_x * 0.25, half_x * 0.55)
        cz = qz * np.random.uniform(half_z * 0.25, half_z * 0.55)
        # Short wall length: 40-70% of quadrant width
        short_len = float(np.random.uniform(half_x * 0.35, half_x * 0.60))
        vertical = np.random.random() < 0.5
        if vertical:
            sz = np.array([wt, wall_height * 0.8, short_len])
        else:
            sz = np.array([short_len, wall_height * 0.8, wt])
        obstacle_field.add_obstacle(BoxObstacle(
            center=np.array([cx, y_center, cz]),
            size=sz
        ))


def _generate_real_maze_layout(tank_geometry, oc, spawn_position: np.ndarray,
                                obstacle_field: ObstacleField) -> None:
    """Generates a grid-based maze layout with multiple dead ends and one
    through-path.

    Design:
    - Divides the tank into a 3x3 grid
    - Places wall segments at grid cell boundaries with random gaps
    - Ensures no walls near the spawn point

    Args:
        tank_geometry: Tank geometry instance.
        oc: Obstacle configuration object.
        spawn_position: Agent spawn position for exclusion zone.
        obstacle_field: Target obstacle field to populate.
    """
    extents = tank_geometry.get_extents()
    depth = tank_geometry.depth

    if extents.get('shape') == 'circular':
        half_x = extents['radius'] * 0.70
        half_z = extents['radius'] * 0.70
    else:
        half_x = extents.get('width', 3.0) / 2.0 * 0.78
        half_z = extents.get('length', 3.0) / 2.0 * 0.78

    y_center = -depth / 2.0
    wall_height = depth * 0.65
    wt = 0.07
    gap = float(np.random.uniform(0.22, 0.38))

    cell_w = half_x * 2 / 3
    cell_d = half_z * 2 / 3

    # Horizontal inner walls (Z = +/-1/3 relative to tank center)
    for z_row in [-1, 1]:
        z_wall = z_row * half_z / 3
        # Wall split into 3 segments, each randomly decides whether to have a gap
        for col in range(3):
            cx = -half_x + cell_w * (col + 0.5)
            has_gap = np.random.random() < 0.55  # 55% probability of gap
            if has_gap:
                gap_pos = float(np.random.uniform(cx - cell_w * 0.3, cx + cell_w * 0.3))
                # Left segment
                left_len = gap_pos - gap / 2 - (cx - cell_w / 2)
                if left_len > 0.10:
                    obstacle_field.add_obstacle(BoxObstacle(
                        center=np.array([cx - cell_w / 2 + left_len / 2, y_center, z_wall]),
                        size=np.array([left_len, wall_height, wt])
                    ))
                # Right segment
                right_start = gap_pos + gap / 2
                right_len = (cx + cell_w / 2) - right_start
                if right_len > 0.10:
                    obstacle_field.add_obstacle(BoxObstacle(
                        center=np.array([right_start + right_len / 2, y_center, z_wall]),
                        size=np.array([right_len, wall_height, wt])
                    ))
            else:
                obstacle_field.add_obstacle(BoxObstacle(
                    center=np.array([cx, y_center, z_wall]),
                    size=np.array([cell_w * 0.92, wall_height, wt])
                ))

    # Vertical inner walls (X = +/-half_x/3)
    for x_col in [-1, 1]:
        x_wall = x_col * half_x / 3
        for row in range(3):
            cz = -half_z + cell_d * (row + 0.5)
            has_gap = np.random.random() < 0.55
            if has_gap:
                gap_pos = float(np.random.uniform(cz - cell_d * 0.3, cz + cell_d * 0.3))
                bottom_len = gap_pos - gap / 2 - (cz - cell_d / 2)
                if bottom_len > 0.10:
                    obstacle_field.add_obstacle(BoxObstacle(
                        center=np.array([x_wall, y_center, cz - cell_d / 2 + bottom_len / 2]),
                        size=np.array([wt, wall_height, bottom_len])
                    ))
                top_start = gap_pos + gap / 2
                top_len = (cz + cell_d / 2) - top_start
                if top_len > 0.10:
                    obstacle_field.add_obstacle(BoxObstacle(
                        center=np.array([x_wall, y_center, top_start + top_len / 2]),
                        size=np.array([wt, wall_height, top_len])
                    ))
            else:
                obstacle_field.add_obstacle(BoxObstacle(
                    center=np.array([x_wall, y_center, cz]),
                    size=np.array([wt, wall_height, cell_d * 0.92])
                ))


def _generate_raceway_layout(tank_geometry, oc, spawn_position: np.ndarray,
                              obstacle_field: ObstacleField) -> None:
    """Generates a raceway layout simulating aquaculture raceway partition
    structures.

    Features:
    - 2-3 parallel divider walls along the long axis, each with an off-center
      passage
    - Dividers alternate left/right gap placement, forcing S-shaped navigation
    - Simulates high-density flow-through aquaculture environments

    Args:
        tank_geometry: Tank geometry instance.
        oc: Obstacle configuration object.
        spawn_position: Agent spawn position for exclusion zone.
        obstacle_field: Target obstacle field to populate.
    """
    extents = tank_geometry.get_extents()
    depth = tank_geometry.depth

    if extents.get('shape') == 'circular':
        half_x = extents['radius'] * 0.72
        half_z = extents['radius'] * 0.72
    else:
        half_x = extents.get('width', 3.0) / 2.0 * 0.80
        half_z = extents.get('length', 3.0) / 2.0 * 0.80

    y_center = -depth / 2.0
    wall_height = depth * 0.70
    wt = 0.07
    gap = float(np.random.uniform(0.28, 0.45))

    # Place dividers along the longer axis
    use_z_axis = half_z >= half_x  # Along Z means dividers perpendicular to Z
    num_dividers = np.random.randint(2, 4)

    if use_z_axis:
        step = half_z * 2 / (num_dividers + 1)
        for i in range(num_dividers):
            z_pos = -half_z + step * (i + 1)
            # Alternate left/right gap placement
            if i % 2 == 0:
                gap_center_x = -half_x * 0.5 + np.random.uniform(-0.1, 0.1)
            else:
                gap_center_x = half_x * 0.5 + np.random.uniform(-0.1, 0.1)
            # Left segment
            left_len = gap_center_x - gap / 2 - (-half_x)
            if left_len > 0.12:
                obstacle_field.add_obstacle(BoxObstacle(
                    center=np.array([-half_x + left_len / 2, y_center, z_pos]),
                    size=np.array([left_len, wall_height, wt])
                ))
            # Right segment
            right_start = gap_center_x + gap / 2
            right_len = half_x - right_start
            if right_len > 0.12:
                obstacle_field.add_obstacle(BoxObstacle(
                    center=np.array([right_start + right_len / 2, y_center, z_pos]),
                    size=np.array([right_len, wall_height, wt])
                ))
    else:
        step = half_x * 2 / (num_dividers + 1)
        for i in range(num_dividers):
            x_pos = -half_x + step * (i + 1)
            if i % 2 == 0:
                gap_center_z = -half_z * 0.5 + np.random.uniform(-0.1, 0.1)
            else:
                gap_center_z = half_z * 0.5 + np.random.uniform(-0.1, 0.1)
            bottom_len = gap_center_z - gap / 2 - (-half_z)
            if bottom_len > 0.12:
                obstacle_field.add_obstacle(BoxObstacle(
                    center=np.array([x_pos, y_center, -half_z + bottom_len / 2]),
                    size=np.array([wt, wall_height, bottom_len])
                ))
            top_start = gap_center_z + gap / 2
            top_len = half_z - top_start
            if top_len > 0.12:
                obstacle_field.add_obstacle(BoxObstacle(
                    center=np.array([x_pos, y_center, top_start + top_len / 2]),
                    size=np.array([wt, wall_height, top_len])
                ))


def _generate_pond_natural_layout(tank_geometry, oc, spawn_position: np.ndarray,
                                   obstacle_field: ObstacleField) -> None:
    """Generates a natural pond layout simulating wild water body environments
    with edge reef clusters and central scattered rocks.

    Features:
    - 2-3 large edge rock piles (near tank walls, simulating shoreline debris)
    - 3-5 medium scattered rocks in the center (simulating exposed bedrock)
    - Large size variation (small rocks 0.05m to large rocks 0.35m)

    Args:
        tank_geometry: Tank geometry instance.
        oc: Obstacle configuration object.
        spawn_position: Agent spawn position for exclusion zone.
        obstacle_field: Target obstacle field to populate.
    """
    extents = tank_geometry.get_extents()
    depth = tank_geometry.depth

    if extents.get('shape') == 'circular':
        r_tank = extents['radius']
    else:
        hw = extents.get('width', 3.0) / 2.0
        hl = extents.get('length', 3.0) / 2.0
        r_tank = min(hw, hl)

    y_mid = -depth / 2.0

    # -- Edge rock piles --
    num_edge_clusters = np.random.randint(2, 4)
    for _ in range(num_edge_clusters):
        for attempt in range(40):
            angle = np.random.uniform(0, 2 * np.pi)
            dist = float(np.random.uniform(r_tank * 0.55, r_tank * 0.82))
            cx = dist * np.cos(angle)
            cz = dist * np.sin(angle)
            cy = float(np.random.uniform(-depth * 0.8, -depth * 0.35))
            center = np.array([cx, cy, cz], dtype=np.float32)
            if not tank_geometry.contains_point_xz(cx, cz):
                continue
            if spawn_position is not None and np.linalg.norm(
                    center - spawn_position) < oc.spawn_exclusion_radius * 1.8:
                continue
            break
        num_rocks = np.random.randint(3, 7)
        big_r = float(np.random.uniform(0.18, 0.35))
        spread = big_r * 2.5
        for j in range(num_rocks):
            r = float(np.random.uniform(0.06, big_r))
            off = np.random.uniform(-spread, spread, 3).astype(np.float32)
            off[1] *= 0.3
            _place_rock(obstacle_field, center + off, r,
                        tank_geometry, spawn_position, oc)

    # -- Central scattered rocks --
    num_center = np.random.randint(3, 6)
    for _ in range(num_center):
        for _ in range(30):
            pos = tank_geometry.random_interior_point(margin=0.15)
            pos[1] = float(np.random.uniform(-depth * 0.85, -depth * 0.2))
            r = float(np.random.uniform(oc.rock_radius_min, oc.rock_radius_max * 1.4))
            if _place_rock(obstacle_field, pos, r,
                           tank_geometry, spawn_position, oc):
                break


def _generate_stream_channel_layout(tank_geometry, oc, spawn_position: np.ndarray,
                                     obstacle_field: ObstacleField) -> None:
    """Generates a stream channel layout simulating a fast-flowing natural
    stream environment.

    Features:
    - Dense large rocks on both banks (simulating rocky channel walls)
    - Medium pebbles scattered in the channel center (simulating exposed
      riverbed rocks)
    - Curved channel: the passage follows an S-shaped or Z-shaped offset

    Args:
        tank_geometry: Tank geometry instance.
        oc: Obstacle configuration object.
        spawn_position: Agent spawn position for exclusion zone.
        obstacle_field: Target obstacle field to populate.
    """
    extents = tank_geometry.get_extents()
    depth = tank_geometry.depth

    if extents.get('shape') == 'circular':
        half_x = extents['radius'] * 0.80
        half_z = extents['radius'] * 0.80
    else:
        half_x = extents.get('width', 3.0) / 2.0 * 0.85
        half_z = extents.get('length', 3.0) / 2.0 * 0.85

    # Channel centerline: S-shape (along Z axis, S-curve in X direction)
    num_sections = 4
    section_len = half_z * 2 / num_sections

    # S-shaped axis sample points
    x_shifts = [0.0]
    for i in range(num_sections):
        shift = float(np.random.uniform(-half_x * 0.25, half_x * 0.25))
        x_shifts.append(x_shifts[-1] + shift)
    # Center the shifts
    mean_shift = sum(x_shifts) / len(x_shifts)
    x_shifts = [x - mean_shift for x in x_shifts]

    channel_width = float(np.random.uniform(half_x * 0.30, half_x * 0.50))
    bank_rock_r_min = 0.12
    bank_rock_r_max = min(0.30, oc.rock_radius_max * 1.8)

    for sec in range(num_sections):
        z_start = -half_z + section_len * sec
        z_end = z_start + section_len
        z_mid = (z_start + z_end) / 2
        x_center = (x_shifts[sec] + x_shifts[sec + 1]) / 2

        # Place 2-4 rocks on each bank
        for side in [-1, 1]:
            num_bank_rocks = np.random.randint(2, 5)
            for _ in range(num_bank_rocks):
                r = float(np.random.uniform(bank_rock_r_min, bank_rock_r_max))
                bank_offset = channel_width / 2 + r + float(
                    np.random.uniform(0.03, 0.15))
                x_pos = x_center + side * bank_offset + float(
                    np.random.uniform(-0.08, 0.08))
                z_pos = float(np.random.uniform(z_start + 0.05, z_end - 0.05))
                y_pos = float(np.random.uniform(-depth * 0.85, -depth * 0.25))
                pos = np.array([x_pos, y_pos, z_pos], dtype=np.float32)
                _place_rock(obstacle_field, pos, r,
                            tank_geometry, spawn_position, oc)

    # Channel center pebbles
    num_mid_rocks = np.random.randint(3, 7)
    for _ in range(num_mid_rocks):
        for attempt in range(30):
            z_pos = float(np.random.uniform(-half_z * 0.8, half_z * 0.8))
            sec_idx = int((z_pos + half_z) / (half_z * 2 / num_sections))
            sec_idx = max(0, min(num_sections - 1, sec_idx))
            x_center_s = x_shifts[sec_idx]
            x_pos = x_center_s + float(np.random.uniform(
                -channel_width * 0.35, channel_width * 0.35))
            r = float(np.random.uniform(oc.rock_radius_min * 0.8,
                                         oc.rock_radius_max * 0.9))
            y_pos = float(np.random.uniform(-depth * 0.9, -depth * 0.3))
            pos = np.array([x_pos, y_pos, z_pos], dtype=np.float32)
            if _place_rock(obstacle_field, pos, r,
                           tank_geometry, spawn_position, oc):
                break


def _generate_ushape_channel_layout(tank_geometry, oc, spawn_position: np.ndarray,
                                     obstacle_field: ObstacleField) -> None:
    """Generates a U-shaped aquaculture trough layout common in industrial
    fish farming.

    Features:
    - Two parallel long walls, one end closed, one end open (U-shape opening)
    - Each wall has 1-2 through-holes (inspection/feeding ports)
    - A few scattered rocks in the central corridor (simulating sediment or
      feeding points)

    Args:
        tank_geometry: Tank geometry instance.
        oc: Obstacle configuration object.
        spawn_position: Agent spawn position for exclusion zone.
        obstacle_field: Target obstacle field to populate.
    """
    extents = tank_geometry.get_extents()
    depth = tank_geometry.depth

    if extents.get('shape') == 'circular':
        half_x = extents['radius'] * 0.60
        half_z = extents['radius'] * 0.65
    else:
        half_x = extents.get('width', 3.0) / 2.0 * 0.68
        half_z = extents.get('length', 3.0) / 2.0 * 0.72

    y_center = -depth / 2.0
    wall_height = depth * 0.68
    wt = 0.07
    gap = float(np.random.uniform(0.25, 0.40))

    # U-shape opening direction (randomly selected)
    open_end = np.random.choice(['north', 'south', 'east', 'west'])

    if open_end in ('north', 'south'):
        # Two vertical walls (parallel to Z axis)
        wall_x = half_x * 0.50
        wall_z_len = half_z * 1.80
        closed_z = -half_z * 0.90 if open_end == 'north' else half_z * 0.90

        for side_x in [-wall_x, wall_x]:
            # Main wall body (with random gap)
            gap_z = float(np.random.uniform(-half_z * 0.3, half_z * 0.3))
            # Lower segment
            bot_len = gap_z - gap / 2 - (-half_z * 0.90)
            if bot_len > 0.12:
                obstacle_field.add_obstacle(BoxObstacle(
                    center=np.array([side_x, y_center, -half_z * 0.90 + bot_len / 2]),
                    size=np.array([wt, wall_height, bot_len])
                ))
            # Upper segment
            top_start = gap_z + gap / 2
            top_len = half_z * 0.90 - top_start
            if top_len > 0.12:
                obstacle_field.add_obstacle(BoxObstacle(
                    center=np.array([side_x, y_center, top_start + top_len / 2]),
                    size=np.array([wt, wall_height, top_len])
                ))

        # U-shape closed-end cross wall (between the two main walls)
        cross_len = wall_x * 2 - wt
        obstacle_field.add_obstacle(BoxObstacle(
            center=np.array([0.0, y_center, closed_z]),
            size=np.array([cross_len, wall_height, wt])
        ))
    else:
        # Two horizontal walls (parallel to X axis)
        wall_z = half_z * 0.50
        closed_x = -half_x * 0.90 if open_end == 'east' else half_x * 0.90

        for side_z in [-wall_z, wall_z]:
            gap_x = float(np.random.uniform(-half_x * 0.3, half_x * 0.3))
            left_len = gap_x - gap / 2 - (-half_x * 0.90)
            if left_len > 0.12:
                obstacle_field.add_obstacle(BoxObstacle(
                    center=np.array([-half_x * 0.90 + left_len / 2, y_center, side_z]),
                    size=np.array([left_len, wall_height, wt])
                ))
            right_start = gap_x + gap / 2
            right_len = half_x * 0.90 - right_start
            if right_len > 0.12:
                obstacle_field.add_obstacle(BoxObstacle(
                    center=np.array([right_start + right_len / 2, y_center, side_z]),
                    size=np.array([right_len, wall_height, wt])
                ))

        cross_len = wall_z * 2 - wt
        obstacle_field.add_obstacle(BoxObstacle(
            center=np.array([closed_x, y_center, 0.0]),
            size=np.array([wt, wall_height, cross_len])
        ))


def generate_obstacles(tank_geometry,
                       config=None,
                       spawn_position: np.ndarray = None,
                       layout_hint: str = None) -> ObstacleField:
    """Generates obstacles for an episode, selecting among 10 layouts by
    probability.

    Aquaculture environments (55%):
    - 12% random scatter     -- baseline obstacle avoidance
    - 12% corridor           -- single-channel navigation
    - 12% box maze           -- multi-room maze
    - 10% grid maze          -- 3x3 room maze
    -  9% raceway dividers   -- S-shaped obstacles

    Natural/wild environments (45%):
    - 10% cluster reef       -- medium-density reef area
    - 10% wild reef          -- large reef formations
    - 10% natural pond       -- shoreline rock piles
    -  8% stream channel     -- S-shaped rocky banks
    -  7% U-shape trough     -- enclosed channel

    The layout_hint parameter can force a specific layout type.

    Args:
        tank_geometry: Tank geometry instance.
        config: Configuration object. If None, attempts to import CONFIG.
        spawn_position: Agent spawn position for exclusion zone.
        layout_hint: Optional layout type string to force selection.

    Returns:
        Populated ObstacleField instance.
    """
    if config is None:
        try:
            from config import CONFIG
            config = CONFIG
        except ImportError:
            return ObstacleField()

    oc = getattr(config, 'obstacles', None)
    if oc is None:
        return ObstacleField()

    obstacle_field = ObstacleField()

    _LAYOUT_MAP = {
        'random':         _generate_random_layout,
        'corridor':       _generate_corridor_layout,
        'box_maze':       _generate_box_maze_layout,
        'real_maze':      _generate_real_maze_layout,
        'raceway':        _generate_raceway_layout,
        'cluster':        _generate_cluster_layout,
        'reef':           _generate_reef_layout,
        'pond_natural':   _generate_pond_natural_layout,
        'stream_channel': _generate_stream_channel_layout,
        'ushape_channel': _generate_ushape_channel_layout,
    }

    if layout_hint in _LAYOUT_MAP:
        _LAYOUT_MAP[layout_hint](tank_geometry, oc, spawn_position, obstacle_field)
    else:
        # Cumulative probability distribution (order matches documentation above)
        thresholds = [
            (0.12, _generate_random_layout),
            (0.24, _generate_corridor_layout),
            (0.36, _generate_box_maze_layout),
            (0.46, _generate_real_maze_layout),
            (0.55, _generate_raceway_layout),
            (0.65, _generate_cluster_layout),
            (0.75, _generate_reef_layout),
            (0.85, _generate_pond_natural_layout),
            (0.93, _generate_stream_channel_layout),
            (1.00, _generate_ushape_channel_layout),
        ]
        rand = np.random.random()
        for threshold, func in thresholds:
            if rand < threshold:
                func(tank_geometry, oc, spawn_position, obstacle_field)
                break

    return obstacle_field


# ============================================================
# Empty Obstacle Field (for when obstacles are disabled)
# ============================================================

_EMPTY_FIELD = ObstacleField()


def create_empty_obstacle_field() -> ObstacleField:
    """Creates an empty obstacle field.

    Returns:
        An ObstacleField with no obstacles.
    """
    return ObstacleField()


# ============================================================
# Self-Test
# ============================================================

if __name__ == "__main__":
    from tank_geometry import CircularTank, RectangularTank

    print("=" * 60)
    print("Obstacle System Test")
    print("=" * 60)

    # Create a circular tank
    tank = CircularTank(1.5, 0.8)

    # Manually create a few obstacles
    field = ObstacleField()
    field.add_obstacle(RockObstacle(np.array([0.5, -0.4, 0.3]), 0.15))
    field.add_obstacle(RockObstacle(np.array([-0.3, -0.5, -0.4]), 0.10))
    field.add_obstacle(RockObstacle(np.array([0.0, -0.3, 0.8]), 0.20))

    print(f"\nObstacle count: {field.count}")
    for i, info in enumerate(field.get_obstacle_info()):
        print(f"  [{i}] center={info['center']}, radius={info['radius']:.2f}m")

    # --- Collision detection ---
    print("\n--- Collision Detection ---")
    test_points = [
        np.array([0.5, -0.4, 0.3]),  # At first obstacle center
        np.array([0.65, -0.4, 0.3]),  # At first obstacle edge
        np.array([1.0, -0.4, 0.3]),  # Far from obstacles
        np.array([0.0, -0.3, 0.0]),  # Tank center
    ]

    for pt in test_points:
        result = field.check_collision(pt)
        if result.collided:
            print(f"  {pt} -> Collision! Pushed to {result.pushed_position}, "
                  f"normal={result.normal}")
        else:
            print(f"  {pt} -> Safe")

    # --- Line-of-sight detection ---
    print("\n--- Line-of-Sight (LOS) ---")
    observer = np.array([0.0, -0.4, 0.0])
    targets = [
        np.array([1.0, -0.4, 0.0]),  # No obstacle
        np.array([1.0, -0.4, 0.6]),  # Possibly blocked by first obstacle
        np.array([0.0, -0.3, 1.2]),  # Possibly blocked by third obstacle
    ]

    for target in targets:
        can_see = field.check_line_of_sight(observer, target)
        status = "Visible" if can_see else "Occluded"
        print(f"  {observer} -> {target}: {status}")

    # --- Ray casting ---
    print("\n--- Ray Casting ---")
    origin = np.array([0.0, -0.4, 0.0])
    directions = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.5, 0.0, 0.3]),
        np.array([0.0, 0.0, 1.0]),
        np.array([-1.0, 0.0, 0.0]),
    ]

    for d in directions:
        dist = field.cast_ray(origin, d, 3.0)
        print(f"  Direction {d} -> distance={dist:.3f}m")

    # --- Position validation ---
    print("\n--- Position Validation ---")
    for _ in range(5):
        pos = tank.random_interior_point()
        valid = field.is_valid_position(pos)
        print(f"  {pos} -> {'Safe' if valid else 'Inside obstacle'}")

    # --- Auto-generation test ---
    print("\n--- Auto-Generation ---")


    # Mock configuration
    class MockObstacleConfig:
        min_obstacles = 3
        max_obstacles = 6
        rock_radius_min = 0.05
        rock_radius_max = 0.20
        min_distance_between = 0.15
        min_distance_from_wall = 0.10
        spawn_exclusion_radius = 0.30


    class MockConfig:
        obstacles = MockObstacleConfig()


    spawn = np.array([0.0, -0.3, 0.0])
    auto_field = generate_obstacles(tank, MockConfig(), spawn)
    print(f"  Generated {auto_field.count} obstacles:")
    for info in auto_field.get_obstacle_info():
        print(f"    position={[f'{v:.2f}' for v in info['center']]}, "
              f"radius={info['radius']:.2f}m")

    # Rectangular tank test
    print("\n--- Rectangular Tank Obstacle Generation ---")
    tank_r = RectangularTank(3.0, 2.0, 0.8)
    auto_field_r = generate_obstacles(tank_r, MockConfig(), spawn)
    print(f"  Generated {auto_field_r.count} obstacles:")
    for info in auto_field_r.get_obstacle_info():
        print(f"    position={[f'{v:.2f}' for v in info['center']]}, "
              f"radius={info['radius']:.2f}m")

    print("\nTest complete.")
