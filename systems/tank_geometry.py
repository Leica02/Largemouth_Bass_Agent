#!/usr/bin/env python3
"""
Tank Geometry Module
====================

Implements circular, rectangular, and irregular polygon tank geometries with
boundary enforcement and random interior point generation. Each episode
randomly selects a tank shape and dimensions, forcing the agent to rely on
ray-based perception rather than memorized geometry.

Usage:
    from systems.tank_geometry import create_random_tank, CircularTank, RectangularTank

    tank = create_random_tank()           # Random generation
    tank = CircularTank(1.5, 0.8)         # Specific circular tank
    tank = RectangularTank(3.0, 2.0, 0.8) # Specific rectangular tank
"""

import numpy as np
import math
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any


# ============================================================
# Abstract Base Class
# ============================================================

class TankGeometry(ABC):
    """Abstract base class for tank geometries."""

    def __init__(self, depth: float):
        self.depth = depth

    @property
    @abstractmethod
    def shape_name(self) -> str:
        """Returns the shape identifier string."""
        pass

    @property
    @abstractmethod
    def normalization_scales(self) -> Tuple[float, float]:
        """Returns (x_scale, z_scale) for observation normalization.

        For circular tanks both directions are equal (= radius).
        For rectangular tanks they equal half-width and half-length respectively.
        """
        pass

    @abstractmethod
    def contains_point_xz(self, x: float, z: float) -> bool:
        """Checks whether a point in the horizontal plane lies inside the tank
        (including a safety margin)."""
        pass

    @abstractmethod
    def enforce_boundary(self, position: np.ndarray, velocity: np.ndarray
                         ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Enforces tank boundary constraints.

        Collision behavior is consistent with the original _enforce_boundaries:
        - Normal velocity is zeroed (no bounce)
        - Tangential velocity is damped (friction)
        - Heading is not modified

        Args:
            position: 3D position array [x, y, z].
            velocity: 3D velocity array [vx, vy, vz].

        Returns:
            Tuple of (new_position, new_velocity, collision_occurred).
        """
        pass

    @abstractmethod
    def ray_to_wall_xz(self, origin: np.ndarray, direction: np.ndarray) -> float:
        """Computes the distance from a horizontal-plane ray to the tank wall.

        The direction does not need to be normalized (handled internally).

        Args:
            origin: Ray origin position.
            direction: Ray direction vector.

        Returns:
            Distance to wall intersection, or float('inf') if no intersection.
        """
        pass

    @abstractmethod
    def nearest_wall_info(self, x: float, z: float) -> Tuple[float, np.ndarray]:
        """Computes the nearest wall distance and inward-pointing normal.

        Args:
            x: Horizontal X coordinate.
            z: Horizontal Z coordinate.

        Returns:
            Tuple of (distance, inward_normal_3d).
        """
        pass

    @abstractmethod
    def random_interior_point(self, margin: float = 0.1) -> np.ndarray:
        """Generates a random 3D point inside the tank volume.

        Args:
            margin: Minimum distance from boundaries.

        Returns:
            3D position array [x, y, z].
        """
        pass

    @abstractmethod
    def get_extents(self) -> Dict[str, Any]:
        """Returns tank parameters for info output and serialization."""
        pass

    # ========== Common Methods ==========

    def enforce_vertical_bounds(self, position: np.ndarray, velocity: np.ndarray
                                ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Enforces vertical constraints (tank bottom + water surface/air).

        Common to all tank shapes.

        Args:
            position: 3D position array [x, y, z].
            velocity: 3D velocity array [vx, vy, vz].

        Returns:
            Tuple of (new_position, new_velocity, collision_occurred).
        """
        collision = False

        # --- Tank bottom ---
        bottom_y = -self.depth + 0.01
        if position[1] < bottom_y:
            position[1] = bottom_y
            if velocity[1] < 0:
                velocity[1] = 0.0
            velocity[0] *= 0.5
            velocity[2] *= 0.5
            collision = True

        # --- Above water surface ---
        max_jump = 0.5   # Physical upper limit (half meter)
        if position[1] > max_jump:
            position[1] = max_jump
            velocity[1] = min(velocity[1], 0.0)
        # When y > 0, no artificial damping is applied: the in_air branch of
        # _calculate_forces already applies gravity, so the fish returns to
        # water naturally without additional intervention.
        elif position[1] > -0.02:
            # Water re-entry moment
            if velocity[1] < -0.1:
                velocity[1] *= 0.6
                velocity[0] *= 0.85
                velocity[2] *= 0.85
                position[1] = -0.01
                collision = True

        return position, velocity, collision

    def ray_to_vertical_surface(self, origin_y: float, direction_y: float) -> float:
        """Computes the ray distance to the tank bottom or water surface.

        Args:
            origin_y: Vertical component of the ray origin.
            direction_y: Vertical component of the ray direction.

        Returns:
            Distance to the nearest vertical boundary, or float('inf').
        """
        min_t = float('inf')
        if abs(direction_y) > 1e-8:
            t_bottom = (-self.depth - origin_y) / direction_y
            t_surface = (0.0 - origin_y) / direction_y
            for t in [t_bottom, t_surface]:
                if t > 0.01:
                    min_t = min(min_t, t)
        return min_t


# ============================================================
# Circular Tank
# ============================================================

class CircularTank(TankGeometry):
    """Circular tank with boundary behavior consistent with the original
    _enforce_boundaries implementation."""

    def __init__(self, radius: float, depth: float):
        super().__init__(depth)
        self.radius = radius

    @property
    def shape_name(self) -> str:
        return "circular"

    @property
    def normalization_scales(self) -> Tuple[float, float]:
        return (self.radius, self.radius)

    def contains_point_xz(self, x: float, z: float) -> bool:
        margin = 0.05
        return (x ** 2 + z ** 2) <= (self.radius - margin) ** 2

    def enforce_boundary(self, position: np.ndarray, velocity: np.ndarray
                         ) -> Tuple[np.ndarray, np.ndarray, bool]:
        collision = False
        max_radius = self.radius - 0.05
        horizontal_dist = math.sqrt(position[0] ** 2 + position[2] ** 2)

        if horizontal_dist > max_radius:
            # Pull back inside the tank
            factor = max_radius / horizontal_dist
            position[0] *= factor
            position[2] *= factor

            # Zero normal velocity; also zero tangential velocity:
            # Retaining tangential velocity produces a component perpendicular
            # to heading, which after nonholonomic constraint projection becomes
            # zero, causing the fish to get stuck. Clearing all horizontal
            # velocity lets the fish restart from rest more naturally.
            normal = np.array([position[0], 0.0, position[2]], dtype=np.float32)
            normal = normal / (np.linalg.norm(normal) + 1e-6)
            v_dot_n = float(np.dot(np.array([velocity[0], 0.0, velocity[2]]), normal))
            if v_dot_n > 0:
                velocity[0] = 0.0
                velocity[2] = 0.0
                velocity[1] = min(velocity[1], 0.0)  # Clear upward component to prevent surface breach

            collision = True

        # Vertical constraints
        position, velocity, v_col = self.enforce_vertical_bounds(position, velocity)
        collision = collision or v_col
        return position, velocity, collision

    def ray_to_wall_xz(self, origin: np.ndarray, direction: np.ndarray) -> float:
        ox, oz = origin[0], origin[2]
        dx, dz = direction[0], direction[2]

        a = dx ** 2 + dz ** 2
        if abs(a) < 1e-8:
            return float('inf')

        b = 2.0 * (ox * dx + oz * dz)
        c = ox ** 2 + oz ** 2 - self.radius ** 2
        discriminant = b ** 2 - 4.0 * a * c

        if discriminant < 0:
            return float('inf')

        sqrt_disc = math.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2.0 * a)
        t2 = (-b + sqrt_disc) / (2.0 * a)

        min_t = float('inf')
        for t in [t1, t2]:
            if t > 0.01:
                min_t = min(min_t, t)
        return min_t

    def nearest_wall_info(self, x: float, z: float) -> Tuple[float, np.ndarray]:
        dist_from_center = math.sqrt(x ** 2 + z ** 2)
        wall_dist = self.radius - dist_from_center

        if dist_from_center > 1e-6:
            # Inward normal (toward center)
            normal = np.array([-x / dist_from_center, 0.0, -z / dist_from_center],
                              dtype=np.float32)
        else:
            normal = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        return wall_dist, normal

    def random_interior_point(self, margin: float = 0.1) -> np.ndarray:
        r = self.radius - margin
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(0.1, r)
        depth = np.random.uniform(-self.depth + margin, -margin)
        return np.array([radius * np.cos(angle), depth, radius * np.sin(angle)],
                        dtype=np.float32)

    def get_extents(self) -> Dict[str, Any]:
        return {'shape': 'circular', 'radius': self.radius, 'depth': self.depth}


# ============================================================
# Rectangular Tank
# ============================================================

class RectangularTank(TankGeometry):
    """Rectangular tank geometry.

    Coordinate convention:
        - X axis = width direction, range [-width/2, width/2]
        - Z axis = length direction, range [-length/2, length/2]
        - Y axis = depth direction, range [-depth, 0]
    """

    def __init__(self, width: float, length: float, depth: float):
        super().__init__(depth)
        self.width = width
        self.length = length
        self.half_w = width / 2.0
        self.half_l = length / 2.0

    @property
    def shape_name(self) -> str:
        return "rectangular"

    @property
    def normalization_scales(self) -> Tuple[float, float]:
        return (self.half_w, self.half_l)

    def contains_point_xz(self, x: float, z: float) -> bool:
        margin = 0.05
        return (abs(x) <= self.half_w - margin) and (abs(z) <= self.half_l - margin)

    def enforce_boundary(self, position: np.ndarray, velocity: np.ndarray
                         ) -> Tuple[np.ndarray, np.ndarray, bool]:
        collision = False
        margin = 0.05
        max_x = self.half_w - margin
        max_z = self.half_l - margin

        # --- X-direction walls ---
        if position[0] > max_x:
            position[0] = max_x
            if velocity[0] > 0:
                velocity[0] = 0.0
                velocity[2] = 0.0   # Clear all to avoid residual tangential velocity perpendicular to heading causing stuck state
                velocity[1] = min(velocity[1], 0.0)  # Clear upward component to prevent surface breach
            collision = True
        elif position[0] < -max_x:
            position[0] = -max_x
            if velocity[0] < 0:
                velocity[0] = 0.0
                velocity[2] = 0.0
                velocity[1] = min(velocity[1], 0.0)
            collision = True

        # --- Z-direction walls ---
        if position[2] > max_z:
            position[2] = max_z
            if velocity[2] > 0:
                velocity[2] = 0.0
                velocity[0] = 0.0
                velocity[1] = min(velocity[1], 0.0)
            collision = True
        elif position[2] < -max_z:
            position[2] = -max_z
            if velocity[2] < 0:
                velocity[2] = 0.0
                velocity[0] = 0.0
                velocity[1] = min(velocity[1], 0.0)
            collision = True

        # --- Vertical constraints ---
        position, velocity, v_col = self.enforce_vertical_bounds(position, velocity)
        collision = collision or v_col
        return position, velocity, collision

    def ray_to_wall_xz(self, origin: np.ndarray, direction: np.ndarray) -> float:
        """Ray-AABB intersection in the horizontal plane."""
        ox, oz = origin[0], origin[2]
        dx, dz = direction[0], direction[2]

        min_t = float('inf')

        # Two walls in the X direction
        if abs(dx) > 1e-8:
            for wall_x in [self.half_w, -self.half_w]:
                t = (wall_x - ox) / dx
                if t > 0.01:
                    hit_z = oz + t * dz
                    if abs(hit_z) <= self.half_l:
                        min_t = min(min_t, t)

        # Two walls in the Z direction
        if abs(dz) > 1e-8:
            for wall_z in [self.half_l, -self.half_l]:
                t = (wall_z - oz) / dz
                if t > 0.01:
                    hit_x = ox + t * dx
                    if abs(hit_x) <= self.half_w:
                        min_t = min(min_t, t)

        return min_t

    def nearest_wall_info(self, x: float, z: float) -> Tuple[float, np.ndarray]:
        # Distance to each of the four walls
        d_right = self.half_w - x      # +X wall
        d_left = self.half_w + x       # -X wall
        d_front = self.half_l - z      # +Z wall
        d_back = self.half_l + z       # -Z wall

        min_d = min(d_right, d_left, d_front, d_back)

        if min_d == d_right:
            normal = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        elif min_d == d_left:
            normal = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        elif min_d == d_front:
            normal = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        else:
            normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        return min_d, normal

    def random_interior_point(self, margin: float = 0.1) -> np.ndarray:
        x = np.random.uniform(-self.half_w + margin, self.half_w - margin)
        z = np.random.uniform(-self.half_l + margin, self.half_l - margin)
        y = np.random.uniform(-self.depth + margin, -margin)
        return np.array([x, y, z], dtype=np.float32)

    def get_extents(self) -> Dict[str, Any]:
        return {
            'shape': 'rectangular',
            'width': self.width,
            'length': self.length,
            'depth': self.depth
        }


# ============================================================
# Irregular Polygon Tank (Pond / River / Wetland)
# ============================================================

def _make_bottom_heightmap(rng, bbox_x: float, bbox_z: float, depth: float):
    """Generates a bottom elevation height field function bottom_y(x, z) -> float.

    Uses 3-5 superimposed cosine waves of varying frequency with amplitudes up
    to +/-25% of depth. The bottom is guaranteed to remain within
    [-depth+0.05, -depth*0.5]. Pure numpy implementation with no external
    dependencies.

    Args:
        rng: numpy random generator instance.
        bbox_x: Half-extent of the bounding box in X.
        bbox_z: Half-extent of the bounding box in Z.
        depth: Water depth (m).

    Returns:
        A callable bottom_y(x, z) -> float.
    """
    n_waves = int(rng.integers(3, 6))
    amplitudes = rng.uniform(0.04, 0.12, n_waves) * depth
    fx = rng.uniform(0.4, 2.5, n_waves) / max(bbox_x, 0.5)
    fz = rng.uniform(0.4, 2.5, n_waves) / max(bbox_z, 0.5)
    phases = rng.uniform(0, 2 * np.pi, n_waves)
    base_y = -depth + 0.06  # Mean bottom elevation

    def bottom_y(x: float, z: float) -> float:
        val = base_y
        for i in range(n_waves):
            val += amplitudes[i] * math.cos(fx[i] * x + fz[i] * z + phases[i])
        # Clamp to a reasonable range
        return float(np.clip(val, -depth + 0.04, -depth * 0.40))

    return bottom_y


class IrregularPolygonTank(TankGeometry):
    """Irregular polygon tank simulating natural water bodies (ponds, rivers,
    wetlands).

    Supports three shape types:
        'pond'    -- Ellipse with radial perturbation, simulating aquaculture ponds
        'river'   -- Meandering channel (elongated with S-shaped curvature)
        'wetland' -- Irregular polygon with concave bays, simulating wetlands

    Features:
        - Polygon vertices define XZ-plane boundaries
        - Uneven bottom height field (superimposed cosine waves)
        - Complete implementation of all TankGeometry abstract methods
        - Backward-compatible with feeding.py (get_extents returns width/length)

    Args:
        shape_type: 'pond' / 'river' / 'wetland' / None (random selection).
        size_scale: Overall size scaling (base radius/half-width, in meters).
        depth: Water depth (m).
        seed: Random seed for reproducibility.
    """

    def __init__(self, shape_type: str = None,
                 size_scale: float = None,
                 depth: float = None,
                 seed: int = None):
        # -- Random parameters --
        rng_seed = seed if seed is not None else int(np.random.randint(0, 100000))
        rng = np.random.default_rng(rng_seed)

        if depth is None:
            depth = float(rng.uniform(0.5, 1.2))
        super().__init__(depth)

        if shape_type is None:
            shape_type = rng.choice(['pond', 'river', 'wetland'])
        self.shape_type = shape_type

        if size_scale is None:
            size_scale = float(rng.uniform(1.2, 2.2))

        # -- Generate vertices --
        if shape_type == 'pond':
            self.vertices = self._gen_pond(rng, size_scale)
        elif shape_type == 'river':
            self.vertices = self._gen_river(rng, size_scale)
        else:  # wetland
            self.vertices = self._gen_wetland(rng, size_scale)

        # -- AABB --
        xs = self.vertices[:, 0]
        zs = self.vertices[:, 1]
        self.bbox_half_x = float(np.max(np.abs(xs)))
        self.bbox_half_z = float(np.max(np.abs(zs)))

        # -- Bottom height field --
        self._bottom_y = _make_bottom_heightmap(
            rng, self.bbox_half_x, self.bbox_half_z, depth)

    # -- Vertex Generation --

    @staticmethod
    def _gen_pond(rng, scale: float) -> np.ndarray:
        """Generates an ellipse with radial perturbation, 12-16 vertices.

        Args:
            rng: numpy random generator instance.
            scale: Overall size scale.

        Returns:
            Vertex array of shape (N, 2) as float32.
        """
        N = int(rng.integers(12, 17))
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        ax = scale * rng.uniform(0.85, 1.15)
        az = scale * rng.uniform(0.75, 1.10)
        # Low-frequency perturbation (3rd-5th order sinusoids)
        noise = np.zeros(N)
        for k in range(2, 5):
            amp = rng.uniform(0.06, 0.18)
            ph = rng.uniform(0, 2 * np.pi)
            noise += amp * np.sin(k * angles + ph)
        r = 1.0 + noise
        xs = ax * r * np.cos(angles)
        zs = az * r * np.sin(angles)
        # Randomly compress 2-3 vertices (simulating inlet/outlet indentations)
        n_dents = int(rng.integers(2, 4))
        dent_idx = rng.choice(N, n_dents, replace=False)
        for idx in dent_idx:
            r_dent = rng.uniform(0.60, 0.80)
            xs[idx] *= r_dent
            zs[idx] *= r_dent
        return np.column_stack([xs, zs]).astype(np.float32)

    @staticmethod
    def _gen_river(rng, scale: float) -> np.ndarray:
        """Generates a meandering river channel along the Z axis, 6-8 points
        per side.

        Args:
            rng: numpy random generator instance.
            scale: Overall size scale.

        Returns:
            Vertex array of shape (N, 2) as float32.
        """
        n_side = int(rng.integers(6, 9))
        half_z = scale * rng.uniform(1.0, 1.5)
        base_width = scale * rng.uniform(0.35, 0.60)

        z_pts = np.linspace(-half_z, half_z, n_side)
        # Channel centerline S-shaped curvature
        freq = rng.uniform(0.5, 1.2)
        amp  = scale * rng.uniform(0.10, 0.28)
        center_x = amp * np.sin(freq * np.pi * z_pts / half_z + rng.uniform(0, np.pi))
        # Channel width variation (narrow sections 60%, wide sections 140%)
        width_var = base_width * rng.uniform(0.6, 1.4, n_side)

        # Right bank (+X), bottom to top
        right_x = center_x + width_var
        # Left bank (-X), top to bottom
        left_x  = center_x - width_var

        xs = np.concatenate([right_x, left_x[::-1]])
        zs = np.concatenate([z_pts, z_pts[::-1]])
        return np.column_stack([xs, zs]).astype(np.float32)

    @staticmethod
    def _gen_wetland(rng, scale: float) -> np.ndarray:
        """Generates an irregular polygon with concave bays, 16-22 vertices.

        Args:
            rng: numpy random generator instance.
            scale: Overall size scale.

        Returns:
            Vertex array of shape (N, 2) as float32.
        """
        N = int(rng.integers(16, 23))
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        # Base ellipse
        ax = scale * rng.uniform(0.8, 1.1)
        az = scale * rng.uniform(0.9, 1.2)
        # Medium-high frequency perturbation
        noise = np.zeros(N)
        for k in [2, 3, 5, 7]:
            amp = rng.uniform(0.04, 0.14)
            ph = rng.uniform(0, 2 * np.pi)
            noise += amp * np.sin(k * angles + ph)
        r = 1.0 + noise
        xs = ax * r * np.cos(angles)
        zs = az * r * np.sin(angles)
        # 2-4 deep concave bays (simulating reed beds / shallow areas)
        n_bays = int(rng.integers(2, 5))
        bay_idx = rng.choice(N, n_bays, replace=False)
        for idx in bay_idx:
            depth_frac = rng.uniform(0.45, 0.70)
            xs[idx] *= depth_frac
            zs[idx] *= depth_frac
            # Neighboring vertices also slightly indented
            for nbr in [(idx - 1) % N, (idx + 1) % N]:
                xs[nbr] *= rng.uniform(0.78, 0.92)
                zs[nbr] *= rng.uniform(0.78, 0.92)
        return np.column_stack([xs, zs]).astype(np.float32)

    # -- Bottom Elevation --

    def bottom_depth(self, x: float, z: float) -> float:
        """Returns the bottom Y coordinate at the given XZ position (negative
        values; more negative means deeper).

        Args:
            x: Horizontal X coordinate.
            z: Horizontal Z coordinate.

        Returns:
            Bottom elevation as a negative float.
        """
        return self._bottom_y(x, z)

    # -- TankGeometry Abstract Method Implementations --

    @property
    def shape_name(self) -> str:
        return 'irregular_polygon'

    @property
    def normalization_scales(self) -> Tuple[float, float]:
        return (self.bbox_half_x, self.bbox_half_z)

    def contains_point_xz(self, x: float, z: float) -> bool:
        """Point-in-polygon test using the ray casting algorithm, O(N)."""
        inside = False
        verts = self.vertices
        n = len(verts)
        j = n - 1
        for i in range(n):
            xi, zi = float(verts[i, 0]), float(verts[i, 1])
            xj, zj = float(verts[j, 0]), float(verts[j, 1])
            if ((zi > z) != (zj > z)):
                denom = zj - zi
                if abs(denom) > 1e-10:
                    x_cross = (xj - xi) * (z - zi) / denom + xi
                    if x < x_cross:
                        inside = not inside
            j = i
        return inside

    def enforce_boundary(self, position: np.ndarray,
                         velocity: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
        collision = False
        x, z = float(position[0]), float(position[2])
        margin = 0.05

        if not self.contains_point_xz(x, z):
            # Find the nearest edge and push inside
            best_t = float('inf')
            best_proj = None
            best_normal = None
            verts = self.vertices
            n = len(verts)

            for i in range(n):
                ax_v, az_v = float(verts[i, 0]), float(verts[i, 1])
                bx_v, bz_v = float(verts[(i + 1) % n, 0]), float(verts[(i + 1) % n, 1])
                # Closest point on line segment
                dx_e, dz_e = bx_v - ax_v, bz_v - az_v
                len_sq = dx_e * dx_e + dz_e * dz_e
                if len_sq < 1e-10:
                    continue
                t = ((x - ax_v) * dx_e + (z - az_v) * dz_e) / len_sq
                t = max(0.0, min(1.0, t))
                px, pz = ax_v + t * dx_e, az_v + t * dz_e
                dist = math.sqrt((x - px) ** 2 + (z - pz) ** 2)
                if dist < best_t:
                    best_t = dist
                    best_proj = (px, pz)
                    # Normal pointing inward (from nearest edge point toward polygon interior)
                    nx = x - px
                    nz = z - pz
                    nlen = math.sqrt(nx * nx + nz * nz)
                    if nlen > 1e-8:
                        # Current normal points outward (from projection to current point); negate for inward
                        nx, nz = -nx / nlen, -nz / nlen
                    else:
                        nx, nz = 0.0, 0.0
                    best_normal = (nx, nz)

            if best_proj is not None:
                nx, nz = best_normal  # Inward normal (pointing into polygon)
                # Push to inner side of edge: projection point + inward normal * margin
                position[0] = best_proj[0] + nx * margin
                position[2] = best_proj[1] + nz * margin

                # Velocity: clear horizontal components (retaining tangential velocity
                # produces a component perpendicular to heading, which after
                # nonholonomic constraint projection zeroes out, causing stuck state)
                vx, vz = float(velocity[0]), float(velocity[2])
                # Outward normal = -inward normal; v dot outward_normal > 0 means
                # velocity is directed outward and should be cleared
                v_dot_n_out = vx * (-nx) + vz * (-nz)
                if v_dot_n_out > 0:
                    velocity[0] = 0.0
                    velocity[2] = 0.0
                    velocity[1] = min(velocity[1], 0.0)  # Clear upward component to prevent surface breach
                collision = True

        # Vertical constraints (using uneven bottom)
        x_now, z_now = float(position[0]), float(position[2])
        local_bottom = self.bottom_depth(x_now, z_now)
        v_collision = False

        if position[1] < local_bottom:
            position[1] = local_bottom
            if velocity[1] < 0:
                velocity[1] = 0.0
            velocity[0] *= 0.5
            velocity[2] *= 0.5
            v_collision = True

        # Water surface
        max_jump = 0.5
        if position[1] > max_jump:
            position[1] = max_jump
            velocity[1] = min(velocity[1], 0.0)
        elif position[1] > -0.02:
            if velocity[1] < -0.1:
                velocity[1] *= 0.6
                velocity[0] *= 0.85
                velocity[2] *= 0.85
                position[1] = -0.01
                v_collision = True

        collision = collision or v_collision
        return position, velocity, collision

    def ray_to_wall_xz(self, origin: np.ndarray,
                       direction: np.ndarray) -> float:
        """Ray-polygon edge intersection; returns the nearest positive hit
        distance, O(N)."""
        ox, oz = float(origin[0]), float(origin[2])
        dx, dz = float(direction[0]), float(direction[2])
        dir_len = math.sqrt(dx * dx + dz * dz)
        if dir_len < 1e-8:
            return float('inf')
        dx /= dir_len
        dz /= dir_len

        min_t = float('inf')
        verts = self.vertices
        n = len(verts)

        for i in range(n):
            ax_v, az_v = float(verts[i, 0]), float(verts[i, 1])
            bx_v, bz_v = float(verts[(i + 1) % n, 0]), float(verts[(i + 1) % n, 1])
            ex, ez = bx_v - ax_v, bz_v - az_v
            # Solve: origin + t*dir = v[i] + s*edge
            denom = dx * ez - dz * ex
            if abs(denom) < 1e-10:
                continue
            wx, wz = ax_v - ox, az_v - oz
            t = (wx * ez - wz * ex) / denom
            s = (wx * dz - wz * dx) / denom
            if t > 0.01 and 0.0 <= s <= 1.0:
                min_t = min(min_t, t)

        return min_t

    def nearest_wall_info(self, x: float, z: float) -> Tuple[float, np.ndarray]:
        """Finds the distance to the nearest polygon edge and the inward
        normal."""
        best_dist = float('inf')
        best_normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        verts = self.vertices
        n = len(verts)

        for i in range(n):
            ax_v, az_v = float(verts[i, 0]), float(verts[i, 1])
            bx_v, bz_v = float(verts[(i + 1) % n, 0]), float(verts[(i + 1) % n, 1])
            dx_e, dz_e = bx_v - ax_v, bz_v - az_v
            len_sq = dx_e * dx_e + dz_e * dz_e
            if len_sq < 1e-10:
                continue
            t = ((x - ax_v) * dx_e + (z - az_v) * dz_e) / len_sq
            t = max(0.0, min(1.0, t))
            px, pz = ax_v + t * dx_e, az_v + t * dz_e
            dist = math.sqrt((x - px) ** 2 + (z - pz) ** 2)
            if dist < best_dist:
                best_dist = dist
                # Inward normal = from nearest edge point toward current point (inward)
                nx = x - px
                nz = z - pz
                nlen = math.sqrt(nx * nx + nz * nz)
                if nlen > 1e-8:
                    best_normal = np.array([-nx / nlen, 0.0, -nz / nlen],
                                           dtype=np.float32)

        return best_dist, best_normal

    def random_interior_point(self, margin: float = 0.1) -> np.ndarray:
        """Rejection sampling: samples within the AABB and filters out points
        outside the polygon."""
        for _ in range(200):
            x = np.random.uniform(-self.bbox_half_x + margin,
                                   self.bbox_half_x - margin)
            z = np.random.uniform(-self.bbox_half_z + margin,
                                   self.bbox_half_z - margin)
            # Check if the point is inside the polygon
            if not self.contains_point_xz(x, z):
                continue
            # Simple distance check: require at least margin/2 from nearest edge
            dist_w, _ = self.nearest_wall_info(x, z)
            if dist_w < margin * 0.5:
                continue
            local_bottom = self.bottom_depth(x, z)
            y = np.random.uniform(local_bottom + margin, -margin)
            return np.array([x, y, z], dtype=np.float32)
        # Fallback: return center point
        return np.array([0.0, -self.depth * 0.5, 0.0], dtype=np.float32)

    def get_extents(self) -> dict:
        return {
            'shape': 'irregular_polygon',
            'shape_type': self.shape_type,
            'depth': self.depth,
            # The following fields maintain compatibility with rectangular tanks
            # (used by feeding.py / multi_fish)
            'width':  self.bbox_half_x * 2,
            'length': self.bbox_half_z * 2,
            'radius': max(self.bbox_half_x, self.bbox_half_z),  # Circular tank compatibility
            'num_vertices': len(self.vertices),
        }


# ============================================================
# Factory Functions
# ============================================================

def create_random_tank(config=None) -> TankGeometry:
    """Creates a randomly generated tank based on configuration.

    Probability distribution (adjustable via config.env_randomization):
    - 35% circular tank
    - 35% rectangular tank
    - 30% irregular polygon tank (pond / river / wetland)

    Args:
        config: Configuration object. If None, attempts to import CONFIG.

    Returns:
        A TankGeometry instance.
    """
    if config is None:
        try:
            from config import CONFIG
            config = CONFIG
        except ImportError:
            return CircularTank(1.5, 0.8)

    rc = getattr(config, 'env_randomization', None)
    if rc is None or not rc.enable_random_shape:
        env = config.environment
        return CircularTank(env.tank_radius, env.tank_depth)

    depth = float(np.random.uniform(*rc.depth_range))
    roll = np.random.random()

    irregular_prob = getattr(rc, 'irregular_probability', 0.30)
    circ_prob = rc.circular_probability * (1.0 - irregular_prob)
    rect_prob = (1.0 - rc.circular_probability) * (1.0 - irregular_prob)

    if roll < irregular_prob:
        size_scale = float(np.random.uniform(
            min(rc.radius_range), max(rc.radius_range)))
        return IrregularPolygonTank(
            size_scale=size_scale, depth=depth,
            seed=int(np.random.randint(0, 100000)))
    elif roll < irregular_prob + circ_prob:
        radius = float(np.random.uniform(*rc.radius_range))
        return CircularTank(radius, depth)
    else:
        width  = float(np.random.uniform(*rc.rect_width_range))
        length = float(np.random.uniform(*rc.rect_length_range))
        return RectangularTank(width, length, depth)


def create_default_tank(config=None) -> TankGeometry:
    """Creates a default circular tank (for evaluation/testing).

    Args:
        config: Configuration object. If None, attempts to import CONFIG.

    Returns:
        A CircularTank instance with default parameters.
    """
    if config is None:
        try:
            from config import CONFIG
            config = CONFIG
        except ImportError:
            return CircularTank(1.5, 0.8)

    env = config.environment
    return CircularTank(env.tank_radius, env.tank_depth)


# ============================================================
# Self-Test
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Tank Geometry Module Test")
    print("=" * 60)

    # --- Circular tank ---
    tank_c = CircularTank(1.5, 0.8)
    print(f"\nCircular tank: radius={tank_c.radius}m, depth={tank_c.depth}m")
    print(f"  (0, 0) inside: {tank_c.contains_point_xz(0, 0)}")
    print(f"  (1.4, 0) inside: {tank_c.contains_point_xz(1.4, 0)}")
    print(f"  (1.6, 0) inside: {tank_c.contains_point_xz(1.6, 0)}")

    dist, normal = tank_c.nearest_wall_info(1.0, 0.0)
    print(f"  Nearest wall from (1,0): {dist:.2f}m, normal={normal}")

    pt = tank_c.random_interior_point()
    print(f"  Random point: {pt}")

    # Ray test
    origin = np.array([0.0, -0.3, 0.0])
    direction = np.array([1.0, 0.0, 0.0])
    ray_dist = tank_c.ray_to_wall_xz(origin, direction)
    print(f"  Ray from origin toward +X to wall distance: {ray_dist:.2f}m")

    # --- Rectangular tank ---
    tank_r = RectangularTank(3.0, 2.0, 0.8)
    print(f"\nRectangular tank: {tank_r.width}m x {tank_r.length}m, depth={tank_r.depth}m")
    print(f"  (0, 0) inside: {tank_r.contains_point_xz(0, 0)}")
    print(f"  (1.4, 0) inside: {tank_r.contains_point_xz(1.4, 0)}")
    print(f"  (1.6, 0) inside: {tank_r.contains_point_xz(1.6, 0)}")

    dist, normal = tank_r.nearest_wall_info(0.0, 0.5)
    print(f"  Nearest wall from (0, 0.5): {dist:.2f}m, normal={normal}")

    ray_dist = tank_r.ray_to_wall_xz(origin, direction)
    print(f"  Ray from origin toward +X to wall distance: {ray_dist:.2f}m")

    # --- Boundary enforcement ---
    print("\n--- Collision Test ---")
    pos = np.array([1.6, -0.3, 0.0], dtype=np.float32)
    vel = np.array([0.5, 0.0, 0.1], dtype=np.float32)
    new_pos, new_vel, col = tank_c.enforce_boundary(pos.copy(), vel.copy())
    print(f"  Circular tank: {pos} -> {new_pos}, collision={col}")

    pos = np.array([1.6, -0.3, 0.0], dtype=np.float32)
    new_pos, new_vel, col = tank_r.enforce_boundary(pos.copy(), vel.copy())
    print(f"  Rectangular tank: {pos} -> {new_pos}, collision={col}")

    print("\nTest complete.")
