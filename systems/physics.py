#!/usr/bin/env python3
"""
 - v3
=====================================


1. Drag Model -
2.  - ""
3. Burst-and-Coast -
4.  -


-  Re = ρVL/μ
- Re < 1000:
- Re > 10000:
- +""


     systems/physics.py
"""

import numpy as np
import math
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CONFIG = None
try:
    from config import CONFIG
except ImportError:
    pass

try:
    from systems.buoyancy import (
        BuoyancySystem, BuoyancyState, BuoyancyInput, BuoyancyOutput,
        create_buoyancy_system, create_buoyancy_state
    )

    BUOYANCY_AVAILABLE = True
except ImportError:
    BUOYANCY_AVAILABLE = False

try:
    from utils.biological_formulas import calculate_sustained_speed, calculate_burst_speed
except ImportError:
    def calculate_sustained_speed(length):
        return length * 1.5


    def calculate_burst_speed(length):
        return length * 3.0

try:
    from systems.metabolism import ActivityState
except ImportError:
    class ActivityState(Enum):
        ACTIVE = "active"
        RESTING = "resting"


# ============================================================
# Data Class Definitions
# ============================================================

@dataclass
class PhysicsState:
    """ - v3 """
    position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    is_at_surface: bool = False
    in_air: bool = False
    air_exposure_time: float = 0.0
    using_burst: bool = False
    total_distance_traveled: float = 0.0
    collision_count: int = 0

    activity_state: ActivityState = ActivityState.ACTIVE
    drift_direction: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    drift_update_counter: int = 0

    buoyancy_state: Any = None
    buoyancy_initialized: bool = False

    # === ===
    heading: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=np.float32))

    # === new===
    #  =  =
    pitch_angle: float = 0.0

    # === new===
    target_pitch_angle: float = 0.0

    smoothed_action: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    smoothed_buoyancy_control: float = 0.0

    #  (deg/s)turn rate
    current_turn_rate: float = 0.0

    # BCF 0.0 ~ 1.01.0
    # Jayne & Lauder (1995)  1.2-7.5 Hz
    bcf_phase: float = 0.0

    caudal_fatigue: float = 0.0

    # === new/s===
    current_pitch_rate: float = 0.0

    # ===  ===
    # Beamish (1978), Webb (1984)2-20s
    burst_fatigue: float = 0.0          #  0.0() ~ 1.0()
    burst_recovery_counter: float = 0.0  # s

    # ===  ===
    pectoral_effort: float = 0.0    #  [0,1]
    # Type I/IIa
    #  180s 75s
    # Drucker & Lauder 1999: pectoral fins predominantly slow-oxidative fibers
    pectoral_fatigue: float = 0.0   #  0.0() ~ 1.0()

    previous_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))

    # Inertia System
    inertia_initialized: bool = False

    # === newBurst-and-Coast  ===
    is_coasting: bool = False  #
    coast_steps_remaining: int = 0  #

    # === new===
    current_reynolds: float = 0.0

    # === new ===
    rest_transition_progress: float = 0.0  # 0.0=, 1.0=
    active_transition_progress: float = 0.0  # 0.0=, 1.0=


@dataclass
class PhysicsInput:
    """"""
    action: np.ndarray  # [0:2], [2]
    body_mass: float
    total_length: float
    net_gravity_in_water: float
    gravity_in_air: float
    water_current: np.ndarray = field(default_factory=lambda: np.zeros(3))
    time_step: float = field(default_factory=lambda: CONFIG.environment.time_step if CONFIG else 0.3)
    time_acceleration: float = field(default_factory=lambda: CONFIG.environment.time_acceleration if CONFIG else 300)
    nearest_threat_distance: float = float('inf')
    activity_state: ActivityState = ActivityState.ACTIVE
    buoyancy_control: float = 0.0
    water_temp: float = field(default_factory=lambda: CONFIG.environment.water_temp if CONFIG else 25.0)
    use_buoyancy_system: bool = True

    # 【new】
    energy_debuff: dict = field(default_factory=lambda: {
        'speed_factor': 1.0,
        'reaction_factor': 1.0,
        'propulsion_factor': 1.0,
        'burst_available': True
    })

    # 【new】
    nearest_food_distance: float = float('inf')
    nearest_prey_distance: float = float('inf')
    # 【new】
    tank_geometry: Any = None  # TankGeometry
    obstacle_field: Any = None  # ObstacleField


@dataclass
class PhysicsOutput:
    """"""
    new_position: np.ndarray = None
    new_velocity: np.ndarray = None
    distance_traveled: float = 0.0
    collision_occurred: bool = False
    is_at_surface: bool = False
    in_air: bool = False
    effective_propulsion_factor: float = 1.0

    buoyancy_output: Any = None
    net_buoyancy_force: float = 0.0
    relative_density: float = 1.0
    buoyancy_energy_consumed: float = 0.0
    current_depth: float = 0.0

    # Inertia System
    heading: np.ndarray = None
    turn_rate_deg_s: float = 0.0
    acceleration_magnitude: float = 0.0
    smoothed_action_magnitude: float = 0.0

    pitch_angle: float = 0.0
    reynolds_number: float = 0.0
    is_coasting: bool = False
    drag_coefficient: float = 0.0

    # 【new】
    energy_speed_factor: float = 1.0
    burst_blocked_by_energy: bool = False

    turn_angle_deg: float = 0.0  #


# ============================================================
# ============================================================

class PhysicsSystem:
    """ - v3 """

    def __init__(self):
        self._init_config()
        self.debug = False

        time_step = getattr(self.env, 'time_step', 0.3)
        time_acc = getattr(self.env, 'time_acceleration', 300)
        self.drift_update_interval = max(3, int(300 / (time_step * time_acc)))

        if BUOYANCY_AVAILABLE:
            self.buoyancy_system = create_buoyancy_system()
        else:
            self.buoyancy_system = None

    def _init_config(self):
        """"""
        if CONFIG is not None and hasattr(CONFIG, 'physics'):
            self.c = CONFIG.physics
        else:
            raise ImportError(" CONFIG.physics")

        if CONFIG is not None and hasattr(CONFIG, 'environment'):
            self.env = CONFIG.environment
        else:
            raise ImportError(" CONFIG.environment")

        if CONFIG is not None and hasattr(CONFIG, 'rest_state'):
            self.rc = CONFIG.rest_state
        else:
            raise ImportError(" CONFIG.rest_state")

        if CONFIG is not None and hasattr(CONFIG, 'inertia'):
            self.ic = CONFIG.inertia
        else:
            raise ImportError(" CONFIG.inertia")

    # ============================================================
    # new
    # ============================================================

    def _calculate_reynolds_number(self, velocity_magnitude: float,
                                   body_length: float) -> float:
        """


        Re = ρ × V × L / μ = V × L / ν

         ν = μ/ρ

        Args:
            velocity_magnitude:  (m/s)
            body_length:  (m)

        Returns:

        """
        kinematic_viscosity = getattr(self.c, 'water_kinematic_viscosity', 0.89e-6)

        if velocity_magnitude < 1e-6:
            return 0.0

        reynolds = velocity_magnitude * body_length / kinematic_viscosity
        return reynolds

    def _get_drag_coefficient_by_reynolds(self, reynolds: float) -> float:
        """
        drag coefficient



        - Re < 1000drag coefficient
        - 1000 < Re < 10000
        - Re > 10000drag coefficient

        Args:
            reynolds:

        Returns:
            drag coefficient
        """
        laminar_threshold = getattr(self.c, 'reynolds_laminar_threshold', 1000.0)
        turbulent_threshold = getattr(self.c, 'reynolds_turbulent_threshold', 10000.0)

        # drag coefficient
        cd_laminar = getattr(self.c, 'drag_coeff_laminar', 0.8)
        cd_transition = getattr(self.c, 'drag_coeff_transition', 0.4)
        cd_turbulent = getattr(self.c, 'drag_coeff_turbulent', 0.15)

        if reynolds < laminar_threshold:
            return cd_laminar
        elif reynolds < turbulent_threshold:
            log_re = np.log10(reynolds)
            log_low = np.log10(laminar_threshold)
            log_high = np.log10(turbulent_threshold)
            t = (log_re - log_low) / (log_high - log_low)
            return cd_laminar + t * (cd_turbulent - cd_laminar)
        else:
            return cd_turbulent

    def _get_coast_decay_by_reynolds(self, reynolds: float) -> float:
        """




        Args:
            reynolds:

        Returns:
            0.85-0.98
        """
        laminar_threshold = getattr(self.c, 'reynolds_laminar_threshold', 1000.0)
        turbulent_threshold = getattr(self.c, 'reynolds_turbulent_threshold', 10000.0)

        coast_small = getattr(self.ic, 'coast_decay_small_fish', 0.85)
        coast_large = getattr(self.ic, 'coast_decay_large_fish', 0.98)

        if reynolds < laminar_threshold:
            return coast_small
        elif reynolds < turbulent_threshold:
            log_re = np.log10(max(reynolds, 1))
            log_low = np.log10(laminar_threshold)
            log_high = np.log10(turbulent_threshold)
            t = (log_re - log_low) / (log_high - log_low)
            return coast_small + t * (coast_large - coast_small)
        else:
            return coast_large

    # ============================================================
    # new
    # ============================================================

    def _update_pitch_angle(self, state: PhysicsState, pitch_action: float,
                            dt: float, is_resting: bool) -> float:
        """
         —



         15~40°/s
        """
        max_pitch = getattr(self.c, 'max_pitch_angle', 45.0)
        pitch_rate_max = getattr(self.c, 'pitch_rate_max', 60.0)
        restoration_factor = getattr(self.c, 'pitch_restoration_factor', 0.1)

        if is_resting:
            rest_target = getattr(self.rc, 'rest_target_pitch', 0.0)
            target = pitch_action * max_pitch * 0.5 + rest_target * 0.5
        else:
            target = pitch_action * max_pitch

        # 0.3s
        pitch_tau = 0.3
        target_pitch_rate = np.clip((target - state.pitch_angle) / pitch_tau,
                                    -pitch_rate_max, pitch_rate_max)
        alpha_p = dt / max(pitch_tau, dt)
        state.current_pitch_rate += (target_pitch_rate - state.current_pitch_rate) * alpha_p
        state.current_pitch_rate = np.clip(state.current_pitch_rate, -pitch_rate_max, pitch_rate_max)

        state.pitch_angle += state.current_pitch_rate * dt

        input_magnitude = abs(pitch_action)
        if input_magnitude > 0.05:
            effective_restoration = restoration_factor * 0.1
        else:
            effective_restoration = restoration_factor
        state.pitch_angle += -state.pitch_angle * effective_restoration * dt

        state.pitch_angle = float(np.clip(state.pitch_angle, -max_pitch, max_pitch))

        return state.current_pitch_rate

    def _calculate_vertical_component_from_pitch(self, state: PhysicsState,
                                                 forward_speed: float,
                                                 body_length: float) -> float:
        """


        ""！


        - θ = V × sin(θ)
        -

        Args:
            state:
            forward_speed:  (m/s)
            body_length:  (m)

        Returns:
             (m/s)=
        """
        lift_coeff = getattr(self.c, 'lift_coefficient', 0.8)
        min_velocity_bl = getattr(self.c, 'min_lift_velocity_bl', 0.3)

        min_velocity = min_velocity_bl * body_length

        if forward_speed < min_velocity:
            effectiveness = (forward_speed / min_velocity) ** 2
        else:
            effectiveness = 1.0

        pitch_rad = np.radians(state.pitch_angle)
        vertical_component = (
                forward_speed *           # V × sin(θ) — correct kinematic formula
                np.sin(pitch_rad) *
                lift_coeff *
                effectiveness
        )

        return vertical_component

    # ============================================================
    # Burst-and-Coast new
    # ============================================================

    def _update_coast_state(self, state: PhysicsState, action_magnitude: float):
        """
         Burst-and-Coast


        - Burst
        - Coast

        Args:
            state:
            action_magnitude:
        """
        coast_threshold = getattr(self.ic, 'coast_action_threshold', 0.15)
        coast_min_duration = getattr(self.ic, 'coast_min_duration', 3)

        if action_magnitude < coast_threshold:
            if not state.is_coasting:
                state.is_coasting = True
                state.coast_steps_remaining = coast_min_duration
            elif state.coast_steps_remaining > 0:
                state.coast_steps_remaining -= 1
        else:
            state.is_coasting = False
            state.coast_steps_remaining = 0

    # ============================================================
    # ============================================================

    def initialize_buoyancy(self, state: PhysicsState, body_mass: float, total_length: float):
        """"""
        if self.buoyancy_system is not None and not state.buoyancy_initialized:
            state.buoyancy_state = create_buoyancy_state()
            self.buoyancy_system.initialize(state.buoyancy_state, body_mass, total_length)
            state.buoyancy_initialized = True

    def initialize_inertia(self, state: PhysicsState, initial_heading: np.ndarray = None):
        """Inertia System"""
        if not state.inertia_initialized:
            if initial_heading is not None:
                state.heading = self._normalize(initial_heading)
            else:
                angle = np.random.uniform(0, 2 * np.pi)
                state.heading = np.array([np.cos(angle), 0.0, np.sin(angle)], dtype=np.float32)

            state.smoothed_action = np.zeros(3, dtype=np.float32)
            state.smoothed_buoyancy_control = 0.0
            state.current_turn_rate = 0.0
            state.previous_velocity = state.velocity.copy()
            state.pitch_angle = 0.0  #
            state.target_pitch_angle = 0.0
            state.is_coasting = False
            state.coast_steps_remaining = 0
            state.inertia_initialized = True

            if self.debug:
                print(f"✅ Inertia Systemv3: {state.heading}")

    # ============================================================
    # ============================================================

    def _normalize(self, v: np.ndarray) -> np.ndarray:
        """Normalize"""
        norm = np.linalg.norm(v)
        if norm < 1e-6:
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        return (v / norm).astype(np.float32)

    def _get_motion_param(self, name: str, default: float) -> float:
        """


         inertia physics
        """
        if hasattr(self.ic, name):
            return float(getattr(self.ic, name))
        if hasattr(self.c, name):
            return float(getattr(self.c, name))
        return float(default)

    @staticmethod
    def _wrap_pi(angle: float) -> float:
        """ [-pi, pi]"""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def _get_inertia_mode(self, nearest_food_distance: float,
                          nearest_threat_distance: float,
                          is_resting: bool) -> tuple:
        """
         (smoothing, deceleration, mode_name)

        :  >  >  >  >

        Returns:
            (smoothing_factor, deceleration_factor, mode_name)
        """
        ic = self.ic

        if is_resting:
            return (
                ic.rest_action_smoothing_factor,  # 0.08
                getattr(ic, 'rest_velocity_decay', 0.92),
                'rest'
            )

        escape_dist = getattr(ic, 'escape_trigger_distance', 0.30)
        if nearest_threat_distance < escape_dist:
            return (
                getattr(ic, 'escape_smoothing', 0.85),
                getattr(ic, 'escape_deceleration', 0.96),
                'escape'
            )

        strike_dist = getattr(ic, 'strike_trigger_distance', 0.06)
        if nearest_food_distance < strike_dist:
            return (
                getattr(ic, 'strike_smoothing', 0.85),
                getattr(ic, 'strike_deceleration', 0.65),
                'strike'
            )

        approach_dist = getattr(ic, 'approach_trigger_distance', 0.25)
        if nearest_food_distance < approach_dist:
            return (
                getattr(ic, 'approach_smoothing', 0.55),
                getattr(ic, 'approach_deceleration', 0.80),
                'approach'
            )

        return (
            getattr(ic, 'cruise_smoothing', 0.30),
            getattr(ic, 'cruise_deceleration', 0.94),
            'cruise'
        )

    def _smooth_action(self, state: PhysicsState, new_action: np.ndarray,
                       is_resting: bool,
                       nearest_food_distance: float = float('inf'),
                       nearest_threat_distance: float = float('inf')
                       ) -> np.ndarray:
        """
         -

        /
        -
        -
        - /
        """
        smoothing, deceleration, mode = self._get_inertia_mode(
            nearest_food_distance, nearest_threat_distance, is_resting
        )
        state._current_inertia_mode = mode
        state._current_deceleration = deceleration

        state.smoothed_action = (smoothing * new_action +
                                 (1 - smoothing) * state.smoothed_action).astype(np.float32)
        return state.smoothed_action

    def _smooth_buoyancy_control(self, state: PhysicsState, new_control: float) -> float:
        """"""
        alpha = self.ic.buoyancy_control_smoothing
        state.smoothed_buoyancy_control = (alpha * new_control +
                                           (1 - alpha) * state.smoothed_buoyancy_control)
        return state.smoothed_buoyancy_control

    def _update_heading(self, state: PhysicsState, desired_direction: np.ndarray,
                        dt: float, is_burst: bool, is_resting: bool,
                        body_length: float = 0.12,
                        raw_turn_input: float = 0.0) -> float:
        """


        turn rate
            (<0.3) : ~30–80°/s   (Lauder & Jayne 1996)
          (0.3-0.6): +BCF~80–250°/s
            (>0.6) : BCF~250–600°/s  (Domenici & Blake 1997)


        """
        if dt <= 1e-6:
            return 0.0

        vel_mag = float(np.linalg.norm(state.velocity))
        speed_bl_s = vel_mag / max(body_length, 0.01)

        lateral = float(desired_direction[2])
        lateral_mag = abs(lateral)
        turn_sign = float(np.sign(lateral)) if lateral_mag > 0.01 else 0.0

        #  ∝ mass·L²
        ref_length_m = 0.093
        mass_turn_factor = float(np.clip(
            (ref_length_m / max(body_length, 0.04)) ** 0.75,
            0.5, 1.6
        ))

        if lateral_mag < 0.05 or is_resting:
            # τ≈0.12s
            state.current_turn_rate *= max(0.0, 1.0 - dt / 0.12)
        else:
            # →
            x = lateral_mag
            if x <= 0.3:
                target_rate = (x / 0.3) * 80.0          # 0→80°/s
            elif x <= 0.6:
                t = (x - 0.3) / 0.3
                target_rate = 80.0 + t * 170.0           # 80→250°/s
            else:
                t = (x - 0.6) / 0.4
                target_rate = 250.0 + t * 350.0          # 250→600°/s

            target_rate *= mass_turn_factor

            # 1-2BL
            #         BCF
            # 5g(5.8cm):  r_min=0.3BL   500g(26.4cm): r_min=0.7BL
            r_min_bl = float(np.clip(
                0.3 + 0.4 * (body_length - 0.04) / max(0.27 - 0.04, 0.01),
                0.3, 0.7
            ))
            r_min_m = r_min_bl * max(body_length, 0.04)
            #  omega_max
            # 0.05 BL/s ≈
            min_effective_speed = 0.05 * max(body_length, 0.04)
            effective_speed = max(vel_mag, min_effective_speed)
            omega_max_from_radius = math.degrees(effective_speed / r_min_m)
            target_rate = min(target_rate, omega_max_from_radius)

            if speed_bl_s > 2.0:
                target_rate *= max(0.75, 1.0 - 0.08 * (speed_bl_s - 2.0))

            # BCF
            #  ∝ m·L²
            # 5g(5.8cm):  tau_scale≈1.6  →  tau≈0.048s
            # 20g(9.3cm): tau_scale≈1.0  →  tau≈0.030s
            # 500g(26.4cm): tau_scale≈0.35 →  tau≈0.085s
            tau_base = max(0.20 - lateral_mag * 0.17, 0.030)   # 0.20s()→0.030s()
            tau = tau_base / mass_turn_factor                   #  tau
            alpha = min(dt / tau, 1.0)
            state.current_turn_rate += (turn_sign * target_rate - state.current_turn_rate) * alpha

        max_yaw = 700.0 * mass_turn_factor
        state.current_turn_rate = float(np.clip(
            state.current_turn_rate, -max_yaw, max_yaw))

        actual_turn_deg = state.current_turn_rate * dt
        turn_abs_deg = abs(actual_turn_deg)

        if turn_abs_deg > 1e-3:
            current_yaw = float(np.arctan2(state.heading[2], state.heading[0]))
            new_yaw = current_yaw + np.radians(actual_turn_deg)
            state.heading = np.array(
                [np.cos(new_yaw), 0.0, np.sin(new_yaw)], dtype=np.float32)
            state.heading = self._normalize(state.heading)

            #  update() Speed Limiting

        state.caudal_fatigue = 0.0  #

        return turn_abs_deg

    def _limit_acceleration(self, state: PhysicsState, new_velocity: np.ndarray,
                            body_length: float, dt: float, is_burst: bool,
                            body_mass: float = 50.0) -> np.ndarray:
        """
         - v3.1

        (>100g)
        """
        ic = self.ic

        if dt < 1e-6:
            return new_velocity

        velocity_change = new_velocity - state.previous_velocity
        acceleration = velocity_change / dt
        accel_magnitude = np.linalg.norm(acceleration)

        if accel_magnitude < 1e-6:
            return new_velocity

        # heading
        prev_speed = np.linalg.norm(state.previous_velocity)
        new_speed = np.linalg.norm(new_velocity)
        speed_change = new_speed - prev_speed

        # max_accel_bl
        max_accel_bl = ic.max_acceleration_bl_s2

        if is_burst:
            max_accel_bl *= ic.burst_acceleration_multiplier

        # =====  =====
        #  F∝m^0.50 a=F/m ∝ m^(-0.50)
        #  20gfactor=1.0
        # 5g:  (5/20)^(-0.50) = 2.0  →
        # 100g: (100/20)^(-0.50) = 0.45
        # 500g: (500/20)^(-0.50) = 0.20
        mass_kg = body_mass / 1000.0
        reference_mass_kg = 0.020  # 20g
        mass_factor = (mass_kg / reference_mass_kg) ** (-0.50)
        mass_factor = float(np.clip(mass_factor, 0.15, 2.5))

        max_accel = max_accel_bl * body_length * mass_factor  # m/s²
        max_speed_change = max_accel * dt  #  (m/s)

        #  heading  Y
        # Fixprevious_velocity  heading  heading
        #  prev_fwd  heading  0
        hdg = state.heading  # XZ _update_heading
        prev_fwd = float(
            state.previous_velocity[0] * hdg[0] + state.previous_velocity[2] * hdg[2])
        new_fwd = float(new_velocity[0] * hdg[0] + new_velocity[2] * hdg[2])
        fwd_change = new_fwd - prev_fwd

        if fwd_change > max_speed_change:
            #  Y
            allowed_fwd = prev_fwd + max_speed_change
            new_velocity = new_velocity.copy()
            new_velocity[0] = hdg[0] * allowed_fwd
            new_velocity[2] = hdg[2] * allowed_fwd
        elif fwd_change < 0:
            max_dec_change = ic.max_deceleration_bl_s2 * body_length * mass_factor * dt
            if fwd_change < -max_dec_change:
                allowed_fwd = prev_fwd - max_dec_change
                new_velocity = new_velocity.copy()
                new_velocity[0] = hdg[0] * allowed_fwd
                new_velocity[2] = hdg[2] * allowed_fwd

        return new_velocity.astype(np.float32)

    # ============================================================
    # ============================================================

    def update(self, state: PhysicsState, input_data: PhysicsInput) -> PhysicsOutput:
        """Physics update - """
        dt = input_data.time_step

        if self.buoyancy_system is not None and not state.buoyancy_initialized:
            self.initialize_buoyancy(state, input_data.body_mass, input_data.total_length)

        if not state.inertia_initialized:
            self.initialize_inertia(state)

        state.previous_velocity = state.velocity.copy()

        state.activity_state = input_data.activity_state
        is_resting = (state.activity_state == ActivityState.RESTING)

        # 【new】
        energy_debuff = input_data.energy_debuff
        speed_factor = energy_debuff.get('speed_factor', 1.0)
        propulsion_factor_energy = energy_debuff.get('propulsion_factor', 1.0)
        burst_available = energy_debuff.get('burst_available', True)

        # 1.
        surface_info = self._check_surface_state(state, dt)
        state.is_at_surface = surface_info['is_at_surface']
        state.in_air = surface_info['in_air']
        state.air_exposure_time = surface_info['air_exposure_time']
        control_factor = surface_info['control_factor']

        # 2.
        current_depth = max(0, -state.position[1])

        # 3.
        smoothed_action = self._smooth_action(
            state, input_data.action, is_resting,
            nearest_food_distance=input_data.nearest_food_distance,
            nearest_threat_distance=input_data.nearest_threat_distance
        )

        # 4.
        smoothed_buoyancy = self._smooth_buoyancy_control(state, input_data.buoyancy_control)

        # 5.
        buoyancy_output = None
        buoyancy_force = 0.0
        buoyancy_energy = 0.0
        relative_density = 1.0

        if (self.buoyancy_system is not None and
                state.buoyancy_state is not None and
                input_data.use_buoyancy_system and
                not state.in_air):
            buoyancy_input = BuoyancyInput(
                buoyancy_control=smoothed_buoyancy,
                body_mass=input_data.body_mass,
                total_length=input_data.total_length,
                depth=current_depth,
                water_temp=input_data.water_temp,
                time_step=input_data.time_step,
                time_acceleration=input_data.time_acceleration,
                is_resting=is_resting
            )
            buoyancy_output = self.buoyancy_system.update(state.buoyancy_state, buoyancy_input)
            buoyancy_force = buoyancy_output.net_buoyancy_force
            buoyancy_energy = buoyancy_output.energy_consumed
            relative_density = buoyancy_output.relative_density

        # ====================================================================
        # action[0] ==
        # 0.5BL/s25%
        raw_action0 = float(input_data.action[0])
        is_reversing = (raw_action0 < 0.0)
        raw_horizontal_mag = abs(raw_action0)

        _max_boost = getattr(self.ic, 'continuous_thrust_max_boost', 2.0)
        if is_reversing:
            thrust_boost = 0.25  # 25%
        elif raw_horizontal_mag <= 0.5:
            thrust_boost = 1.0 + raw_horizontal_mag * 0.4
        else:
            t = (raw_horizontal_mag - 0.5) / 0.5
            thrust_boost = 1.2 + t * (_max_boost - 1.2)
        thrust_boost = float(np.clip(thrust_boost, 0.25, _max_boost))

        # energy penalty
        # /
        state.burst_fatigue = 0.0
        state.burst_recovery_counter = 0.0

        # is_burstC-start
        lateral_input = abs(float(input_data.action[2])) if len(input_data.action) > 2 else 0.0
        is_burst = (
            (lateral_input > 0.7 or raw_horizontal_mag > 0.7) and
            state.activity_state == ActivityState.ACTIVE
        )
        state.using_burst = is_burst

        # 7.
        horizontal_action = np.array([smoothed_action[0], 0.0, smoothed_action[2]], dtype=np.float32)
        pitch_action = smoothed_action[1]

        # 8.  raw action[2]  C-start
        raw_turn = float(input_data.action[2]) if len(input_data.action) > 2 else 0.0
        turn_angle = self._update_heading(state, horizontal_action, dt, is_burst, is_resting,
                                          body_length=input_data.total_length,
                                          raw_turn_input=raw_turn)

        # 9.
        self._update_pitch_angle(state, pitch_action, dt, is_resting)

        # 10.
        current_speed = np.linalg.norm(state.velocity)
        state.current_reynolds = self._calculate_reynolds_number(current_speed, input_data.total_length)

        # 11.  Burst-and-Coast
        action_magnitude = np.linalg.norm(horizontal_action)
        self._update_coast_state(state, action_magnitude)

        # velocityheading
        # /
        # ""heading
        _horiz_vel = np.array([state.velocity[0], 0.0, state.velocity[2]], dtype=np.float32)
        _proj = float(np.dot(_horiz_vel, state.heading))  # ==
        state.velocity[0] = state.heading[0] * _proj
        state.velocity[2] = state.heading[2] * _proj

        # 12.  thrust_boost  is_burst
        forces = self._calculate_forces_with_pitch(
            state, input_data, horizontal_action, control_factor,
            buoyancy_force, is_burst, propulsion_factor_energy, pitch_action,
            turn_angle_deg=turn_angle, dt=dt,
            thrust_boost=thrust_boost
        )

        #  burst_speedcruising speed
        max_velocity = self._calculate_max_velocity(
            input_data.total_length,
            0.0,
            state.activity_state
        )
        if state.activity_state != ActivityState.RESTING:
            # burst speed
            max_velocity = calculate_burst_speed(input_data.total_length)

        max_velocity *= speed_factor  #

        # 14-19.
        mass_kg = input_data.body_mass / 1000.0
        acceleration = forces['net_force'] / mass_kg
        new_velocity = state.velocity + acceleration * dt

        new_velocity = self._limit_acceleration(
            state, new_velocity, input_data.total_length, dt, is_burst,
            body_mass=input_data.body_mass
        )

        if state.is_coasting or action_magnitude < 0.1:
            mode = getattr(state, '_current_inertia_mode', 'cruise')

            if mode in ('approach', 'strike'):
                # /
                decel = getattr(state, '_current_deceleration', 0.80)
                new_velocity *= decel
            else:
                # /
                coast_decay = self._get_coast_decay_by_reynolds(state.current_reynolds)
                new_velocity *= coast_decay

        # ===  ===
        if is_resting:
            # 30
            transition_speed = 0.033  # 3.3%30100%
            state.rest_transition_progress = min(1.0, state.rest_transition_progress + transition_speed)
            state.active_transition_progress = max(0.0, state.active_transition_progress - transition_speed * 2)

            base_damping = self.rc.rest_velocity_damping  # 0.85
            current_damping = 1.0 + (base_damping - 1.0) * state.rest_transition_progress
            new_velocity *= current_damping
        else:
            #  0.05/20→  0.15/72s
            #  AI  ACTIVE/RESTING  50~75%
            state.active_transition_progress = min(1.0, state.active_transition_progress + 0.15)
            state.rest_transition_progress = max(0.0, state.rest_transition_progress - 0.10)

        vel_mag = np.linalg.norm(new_velocity)
        if vel_mag > max_velocity:
            new_velocity = new_velocity / vel_mag * max_velocity

        state.velocity = new_velocity

        # heading
        # /
        _hv = np.array([state.velocity[0], 0.0, state.velocity[2]], dtype=np.float32)
        _proj = float(np.dot(_hv, state.heading))  #
        state.velocity[0] = state.heading[0] * _proj
        state.velocity[2] = state.heading[2] * _proj

        # 0.3 BL/s
        if _proj < 0.0:
            max_rev_speed = 0.30 * input_data.total_length  # m/s
            back_speed = abs(_proj)
            if back_speed > max_rev_speed:
                _s = max_rev_speed / back_speed
                state.velocity[0] *= _s
                state.velocity[2] *= _s

        # Speed Limiting _limit_acceleration
        # "new_velocity = previous_velocity + limited_change"
        # 20%15°/→-5%33°/→-11%90°→-29%
        if turn_angle > 3.0:
            _turn_rad_lat = np.radians(turn_angle)
            _lat_drag = max(1.0 - 0.20 * _turn_rad_lat, 0.70)
            state.velocity[0] *= _lat_drag
            state.velocity[2] *= _lat_drag

        old_pos = state.position.copy()
        state.position = state.position + state.velocity * dt

        if not state.in_air and np.any(input_data.water_current != 0):
            _cm = 1.5 if is_resting else 1.0
            _current_vel = input_data.water_current * 0.5 * _cm  # m/sF/m  mass_kg
            state.position = state.position + _current_vel * dt

        distance = np.linalg.norm(state.position - old_pos)
        state.total_distance_traveled += distance

        collision = self._enforce_boundaries(
            state,
            tank_geometry=input_data.tank_geometry,
            obstacle_field=input_data.obstacle_field
        )
        if collision:
            state.collision_count += 1

        accel_mag = np.linalg.norm(state.velocity - state.previous_velocity) / dt if dt > 0 else 0

        return PhysicsOutput(
            new_position=state.position.copy(),
            new_velocity=state.velocity.copy(),
            distance_traveled=distance,
            collision_occurred=collision,
            is_at_surface=state.is_at_surface,
            in_air=state.in_air,
            effective_propulsion_factor=forces['propulsion_factor'],
            buoyancy_output=buoyancy_output,
            net_buoyancy_force=buoyancy_force,
            relative_density=relative_density,
            buoyancy_energy_consumed=buoyancy_energy,
            current_depth=current_depth,
            heading=state.heading.copy(),
            turn_rate_deg_s=state.current_turn_rate,
            acceleration_magnitude=accel_mag,
            smoothed_action_magnitude=np.linalg.norm(smoothed_action),
            pitch_angle=state.pitch_angle,
            reynolds_number=state.current_reynolds,
            is_coasting=state.is_coasting,
            drag_coefficient=forces.get('drag_coefficient', 0.25),
            # 【new】
            energy_speed_factor=speed_factor,
            burst_blocked_by_energy=False,
            turn_angle_deg=turn_angle
        )

    def _calculate_forces_with_pitch(self, state: PhysicsState, input_data: PhysicsInput,
                                     horizontal_action: np.ndarray, control_factor: float,
                                     buoyancy_force: float, is_burst: bool,
                                     energy_propulsion_factor: float = 1.0,
                                     pitch_action: float = 0.0,
                                     turn_angle_deg: float = 0.0,
                                     dt: float = 0.1,
                                     thrust_boost: float = 1.0
                                     ) -> Dict[str, Any]:
        """ - """
        c = self.c
        rc = self.rc
        mass_kg = input_data.body_mass / 1000.0
        body_length = input_data.total_length

        # propulsion coefficient
        # 0.051.0
        # active_transition_progress
        # ""
        if state.activity_state == ActivityState.RESTING:
            propulsion_factor = rc.rest_propulsion_factor  # 0.05
        else:
            propulsion_factor = 1.0

        # 【new】
        propulsion_factor *= energy_propulsion_factor

        # ===== Propulsion Calculation =====
        # BCFaction[0]
        # action[2]
        # ——
        #  action[0] action[2]
        # action[0] thrust_boost0.25
        raw_action0_force = float(horizontal_action[0])  #
        is_reversing = (raw_action0_force < 0.0)
        forward_action_mag = abs(raw_action0_force)
        action_magnitude = forward_action_mag

        # base_thrust_per_length20g
        # old: 1.2 × L × m^0.67  → new: 2.2 × L × m^0.50
        # 20g: 2.2 × 0.093 × 0.020^0.50 ≈ 0.00289 N6BL/s²
        base_thrust_per_length = 2.2  # N/m

        if action_magnitude > 0.01 and not state.is_coasting:
            pitch_rad = np.radians(state.pitch_angle)

            if is_reversing:
                # heading/
                # 0.5BL/s thrust_boost=0.25
                swim_direction = np.array([
                    -state.heading[0] * np.cos(pitch_rad * 0.3),
                    -np.sin(pitch_rad * 0.3),
                    -state.heading[2] * np.cos(pitch_rad * 0.3)
                ], dtype=np.float32)
            else:
                # heading projected onto pitch plane.
                # F_horizontal = F_thrust × cos(θ),  F_vertical = F_thrust × sin(θ)
                swim_direction = np.array([
                    state.heading[0] * np.cos(pitch_rad),
                    np.sin(pitch_rad),
                    state.heading[2] * np.cos(pitch_rad)
                ], dtype=np.float32)
            swim_direction = self._normalize(swim_direction)

            # Thrust magnitude (N).
            #  F ∝ muscle_cross_section ∝ mass^(2/3) a=F/m ∝ mass^(-1/3)
            # 5g~20BL/s²500g~3BL/s²7
            #  mass^0.50 a = base×L×m^0.50/m = base×L/m^0.50
            # 5g vs 500g = (500/5)^0.50 = 106-8
            #  base 20g
            #   old: 1.2 × 0.093 × 0.020^0.67 = 2.16 mN
            #   new: 2.2 × 0.093 × 0.020^0.50 = 2.16 mN  ✓ 20g
            thrust_magnitude = (
                    action_magnitude *
                    control_factor *
                    propulsion_factor *
                    base_thrust_per_length *
                    input_data.total_length *
                    (mass_kg ** 0.50)
            )

            # /
            # ~1.5 BL/s
            # Beamish (1978) 1.0-1.5 BL/s
            #       Ucrit3-4 BL/sreduced to~50%
            # 60-70%cruising speed100%
            speed_bl_s = float(np.linalg.norm(state.velocity)) / max(body_length, 0.01)
            aerobic_threshold_bl = 1.5   # BL/s
            if speed_bl_s > aerobic_threshold_bl:
                excess = speed_bl_s - aerobic_threshold_bl
                # 1 BL/s15%40%Ucrit5 BL/s
                aerobic_efficiency = max(1.0 - 0.15 * excess, 0.40)
                thrust_magnitude *= aerobic_efficiency

            #  thrust_boost
            #  thrust_boost=0.25/
            thrust_magnitude *= thrust_boost

            # -
            # Webb 1984reduced to40-60%C-start
            # 10°90°reduced to50%
            if not is_reversing and turn_angle_deg > 10.0:
                turn_thrust_fraction = 1.0 - 0.50 * math.sin(math.radians(
                    min(turn_angle_deg, 90.0)))
                thrust_magnitude *= max(turn_thrust_fraction, 0.45)

            propulsion_force = swim_direction * thrust_magnitude
        else:
            propulsion_force = np.zeros(3, dtype=np.float32)

        # =====  =====
        # action<0.15action
        # max propulsion force30-50%Higham 2005 J. Exp. Biol.
        braking_force = np.zeros(3, dtype=np.float32)
        vel_mag_for_brake = np.linalg.norm(state.velocity)
        if vel_mag_for_brake > 0.05 and not is_burst and not state.is_coasting:
            # 1:
            # 2: action>90°
            is_braking = False
            brake_strength = 0.0
            if action_magnitude < 0.15:
                is_braking = True
                brake_strength = (0.15 - action_magnitude) / 0.15  # 0→1
            else:
                vel_dir = state.velocity / vel_mag_for_brake
                # ×action
                desired_dir = np.array([
                    state.heading[0] * np.cos(np.radians(state.pitch_angle)),
                    np.sin(np.radians(state.pitch_angle)),
                    state.heading[2] * np.cos(np.radians(state.pitch_angle))
                ], dtype=np.float32)
                alignment = float(np.dot(vel_dir, desired_dir /
                                         max(np.linalg.norm(desired_dir), 1e-6)))
                if alignment < -0.3:
                    is_braking = True
                    brake_strength = min((-alignment - 0.3) / 0.7, 1.0)

            if is_braking:
                # 10%
                # Higham (2005) 2-5 BL/s²
                # 0.80→0.10→burst-and-coast
                braking_coeff = 0.10
                pec_fatigue_penalty = 1.0 - state.pectoral_fatigue * 0.70
                brake_mag = (braking_coeff * base_thrust_per_length *
                             body_length * (mass_kg ** 0.50) * brake_strength * pec_fatigue_penalty)
                braking_force = -(state.velocity / vel_mag_for_brake) * brake_mag
                state.pectoral_effort = brake_strength
                # 180sType I/IIa
                # mass^0.33
                _pec_mass_kg = mass_kg
                _pec_mass_scale = (_pec_mass_kg / 0.020) ** 0.33
                _pec_t_exhaust = 180.0 * _pec_mass_scale
                state.pectoral_fatigue = min(
                    1.0, state.pectoral_fatigue + (brake_strength / _pec_t_exhaust) * dt
                )
            else:
                state.pectoral_effort = 0.0
                # 75sPCr
                _pec_mass_kg = mass_kg
                _pec_mass_scale = (_pec_mass_kg / 0.020) ** 0.33
                _pec_t_recover = 75.0 * _pec_mass_scale
                state.pectoral_fatigue = max(
                    0.0, state.pectoral_fatigue - (1.0 / _pec_t_recover) * dt
                )

        drift_force = np.zeros(3, dtype=np.float32)
        if state.activity_state == ActivityState.RESTING:
            state.drift_update_counter += 1
            if state.drift_update_counter > self.drift_update_interval:
                state.drift_update_counter = 0
                state.drift_direction = np.random.uniform(-1, 1, 3).astype(np.float32)
                state.drift_direction[1] *= 0.1
                norm = np.linalg.norm(state.drift_direction)
                if norm > 0:
                    state.drift_direction = state.drift_direction / norm
            drift_force = state.drift_direction * mass_kg * 0.003

        vel_mag = np.linalg.norm(state.velocity)
        if vel_mag < 0.001:
            drag_force = np.zeros(3)
            drag_coeff_used = 0.0
        else:
            if state.in_air:
                drag_coeff_used = c.drag_coefficient_air
                density = c.air_density
            elif state.is_at_surface:
                drag_coeff_used = c.drag_coefficient_surface
                density = 500
            else:
                drag_coeff_used = self._get_drag_coefficient_by_reynolds(state.current_reynolds)
                density = c.water_density

            cross_area = body_length * (body_length * 0.18)  # ×
            drag_magnitude = 0.5 * density * drag_coeff_used * cross_area * vel_mag ** 2
            drag_force = -drag_magnitude * (state.velocity / vel_mag)

        # /
        if state.in_air:
            mass_kg = input_data.body_mass / 1000.0
            real_gravity = mass_kg * self.c.gravity
            gravity_force = np.array([0, -real_gravity, 0])
        elif input_data.use_buoyancy_system and buoyancy_force != 0:
            gravity_force = np.array([0, buoyancy_force, 0])
        elif state.is_at_surface:
            gravity_force = np.array([0, -input_data.net_gravity_in_water * 0.5, 0])
        else:
            gravity_force = np.array([0, -input_data.net_gravity_in_water, 0])

        if not state.in_air:
            current_multiplier = 1.5 if state.activity_state == ActivityState.RESTING else 1.0
            current_force = input_data.water_current * mass_kg * 0.5 * current_multiplier
        else:
            current_force = np.zeros(3)

        #  net_force  update()
        net_force = propulsion_force + drag_force + gravity_force + drift_force + braking_force

        return {
            'propulsion': propulsion_force,
            'drag': drag_force,
            'gravity': gravity_force,
            'buoyancy': buoyancy_force,
            'current': current_force,
            'drift': drift_force,
            'net_force': net_force,
            'propulsion_factor': propulsion_factor,
            'drag_coefficient': drag_coeff_used
        }

    def _check_surface_state(self, state: PhysicsState, dt: float) -> Dict[str, Any]:
        """"""
        c = self.c
        y_pos = state.position[1]

        if y_pos > 0:
            # Fish has breached the surface — it is in air.
            # Control authority is reduced but NOT zeroed: the fish can still
            # flap to re-enter the water.  No air_exposure death timer; instead
            # gravity pulls it back down naturally (handled in _calculate_forces).
            return {
                'in_air': True,
                'is_at_surface': False,
                'air_exposure_time': 0.0,   # no death-by-air-exposure
                'control_factor': 0.3       # reduced but functional
            }
        elif y_pos > -c.surface_tension_zone:
            # Surface tension zone: fish is just below the surface.
            # Full control — this is normal foraging territory for surface food.
            return {
                'in_air': False,
                'is_at_surface': True,
                'air_exposure_time': 0.0,
                'control_factor': 1.0
            }
        else:
            return {
                'in_air': False,
                'is_at_surface': False,
                'air_exposure_time': 0.0,
                'control_factor': 1.0
            }

    def _calculate_max_velocity(self, total_length: float, threat_distance: float,
                                activity_state: ActivityState) -> float:
        """"""
        sustained = calculate_sustained_speed(total_length)

        if activity_state == ActivityState.RESTING:
            return sustained * 0.15

        return sustained

    def _enforce_boundaries(self, state: PhysicsState,
                            tank_geometry=None,
                            obstacle_field=None) -> bool:
        """
        Boundary Constraints - v5


        1. TankGeometry/
        2. ObstacleField
        3.  tank_geometry
        """
        collision = False

        # ===== 1.  =====
        if tank_geometry is not None:
            state.position, state.velocity, wall_collision = \
                tank_geometry.enforce_boundary(state.position, state.velocity)
            collision = collision or wall_collision
        else:
            collision = self._enforce_boundaries_legacy(state)

        # ===== 2. Obstacle collision =====
        if obstacle_field is not None:
            col_result = obstacle_field.check_collision(state.position)
            if col_result.collided:
                state.position = col_result.pushed_position.copy()
                state.velocity = obstacle_field.resolve_collision_velocity(
                    state.velocity, col_result.normal
                )
                collision = True

        return collision

    def _enforce_boundaries_legacy(self, state: PhysicsState) -> bool:
        """Boundary Constraints"""
        collision = False
        pos = state.position
        vel = state.velocity

        tank_depth = getattr(self.env, 'tank_depth', 0.8)
        tank_radius = getattr(self.env, 'tank_radius', 1.5)

        bottom_y = -tank_depth + 0.01
        if pos[1] < bottom_y:
            pos[1] = bottom_y
            if vel[1] < 0:
                vel[1] = 0.0
            vel[0] *= 0.5
            vel[2] *= 0.5
            collision = True

        #  —
        max_jump_height = 0.5   #
        if pos[1] > max_jump_height:
            pos[1] = max_jump_height
            vel[1] = min(vel[1], 0.0)
        # y > 0 in_air _calculate_forces
        elif pos[1] > -0.02:
            if vel[1] < -0.1:
                vel[1] *= 0.6
                vel[0] *= 0.85
                vel[2] *= 0.85
                pos[1] = -0.01
                collision = True

        horizontal_dist = np.sqrt(pos[0] ** 2 + pos[2] ** 2)
        max_radius = tank_radius - 0.05
        if horizontal_dist > max_radius:
            factor = max_radius / horizontal_dist
            pos[0] *= factor
            pos[2] *= factor

            normal = np.array([pos[0], 0, pos[2]], dtype=np.float32)
            normal = normal / (np.linalg.norm(normal) + 1e-6)

            vel_horizontal = np.array([vel[0], 0, vel[2]])
            v_dot_n = np.dot(vel_horizontal, normal)

            if v_dot_n > 0:
                vel_tangent = vel_horizontal - v_dot_n * normal
                vel_tangent *= 0.8
                vel[0] = vel_tangent[0]
                vel[2] = vel_tangent[2]
                vel[1] = min(vel[1], 0.0)  #

            collision = True

        return collision

    # ============================================================
    # ============================================================

    def generate_random_position(self) -> np.ndarray:
        """"""
        tank_radius = getattr(self.env, 'tank_radius', 1.5)
        tank_depth = getattr(self.env, 'tank_depth', 0.8)

        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(0.2, tank_radius * 0.8)
        depth = np.random.uniform(-tank_depth + 0.2, -0.2)
        return np.array([radius * np.cos(angle), depth, radius * np.sin(angle)], dtype=np.float32)

    def initialize_water_current(self) -> np.ndarray:
        """"""
        return np.array([np.random.uniform(-0.02, 0.02), 0.0, np.random.uniform(-0.02, 0.02)], dtype=np.float32)

    def calculate_circular_current(self, position: np.ndarray, base_strength: float = None) -> np.ndarray:
        """"""
        if base_strength is None:
            base_strength = getattr(self.env, 'water_current_strength', 0.3)

        x, z = position[0], position[2]
        r = np.sqrt(x ** 2 + z ** 2)
        if r < 0.01:
            return np.zeros(3)

        tank_radius = getattr(self.env, 'tank_radius', 1.5)
        tangent = np.array([-z / r, 0, x / r])
        strength = base_strength * (r / tank_radius)
        return tangent * strength

    def set_debug(self, enabled: bool):
        self.debug = enabled
        if self.buoyancy_system is not None:
            self.buoyancy_system.set_debug(enabled)


# ============================================================
# Factory Functions
# ============================================================

def create_physics_system() -> PhysicsSystem:
    return PhysicsSystem()


def create_physics_state(position: Optional[np.ndarray] = None) -> PhysicsState:
    if position is None:
        position = np.array([0.0, -0.3, 0.0], dtype=np.float32)
    return PhysicsState(position=position.copy(), velocity=np.zeros(3, dtype=np.float32))


# ============================================================
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("v3 ")
    print("=" * 70)

    physics = create_physics_system()
    physics.set_debug(True)

    state = create_physics_state(np.array([0.0, -0.3, 0.0], dtype=np.float32))
    physics.initialize_inertia(state)

    print("\n1")
    print("-" * 50)

    # +
    for step in range(30):
        # action[0]=, action[1]=(), action[2]=
        action = np.array([0.8, 0.6, 0.0], dtype=np.float32)  # +

        input_data = PhysicsInput(
            action=action,
            body_mass=50.0,  # 50g
            total_length=0.12,  # 12cm
            net_gravity_in_water=0.001,
            gravity_in_air=0.01,
            use_buoyancy_system=False
        )
        output = physics.update(state, input_data)

        if step % 5 == 0:
            print(f"{step:2d}: Y={state.position[1]:.3f}m, "
                  f"={output.pitch_angle:.1f}°, "
                  f"Re={output.reynolds_number:.0f}")

    print(f"\n: {state.position[1] - (-0.3):.3f}m")

    print("\n2")
    print("-" * 50)

    state_small = create_physics_state(np.array([0.0, -0.3, 0.0]))
    physics.initialize_inertia(state_small)

    for _ in range(10):
        physics.update(state_small, PhysicsInput(
            action=np.array([1.0, 0.0, 0.0]),
            body_mass=5.0,  # 5g
            total_length=0.03,  # 3cm
            net_gravity_in_water=0.001, gravity_in_air=0.01,
            use_buoyancy_system=False
        ))

    speed_before_small = np.linalg.norm(state_small.velocity)

    for _ in range(10):
        physics.update(state_small, PhysicsInput(
            action=np.array([0.0, 0.0, 0.0]),  #
            body_mass=5.0, total_length=0.03,
            net_gravity_in_water=0.001, gravity_in_air=0.01,
            use_buoyancy_system=False
        ))

    speed_after_small = np.linalg.norm(state_small.velocity)

    print(f"(5g,3cm): {speed_before_small:.4f} → {speed_after_small:.4f} m/s "
          f"({speed_after_small / speed_before_small * 100:.1f}%)")

    state_large = create_physics_state(np.array([0.0, -0.3, 0.0]))
    physics.initialize_inertia(state_large)

    for _ in range(10):
        physics.update(state_large, PhysicsInput(
            action=np.array([1.0, 0.0, 0.0]),
            body_mass=200.0,  # 200g
            total_length=0.25,  # 25cm
            net_gravity_in_water=0.001, gravity_in_air=0.01,
            use_buoyancy_system=False
        ))

    speed_before_large = np.linalg.norm(state_large.velocity)

    for _ in range(10):
        physics.update(state_large, PhysicsInput(
            action=np.array([0.0, 0.0, 0.0]),
            body_mass=200.0, total_length=0.25,
            net_gravity_in_water=0.001, gravity_in_air=0.01,
            use_buoyancy_system=False
        ))

    speed_after_large = np.linalg.norm(state_large.velocity)

    print(f"(200g,25cm): {speed_before_large:.4f} → {speed_after_large:.4f} m/s "
          f"({speed_after_large / speed_before_large * 100:.1f}%)")

    print("\n✅ ！")
    print("\n:")
    print("  1.  - +")
    print("  2.  - ")
    print("  3.  - ")
    print("  4. Burst-and-Coast - ")
