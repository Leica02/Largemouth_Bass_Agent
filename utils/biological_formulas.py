#!/usr/bin/env python3
"""
Biological formula utilities -- piecewise length-weight model (v2).

Implements piecewise allometric length-weight relationships for largemouth bass
(Micropterus salmoides) spanning the full ontogenetic range from yolk-sac larvae
to adults, with sigmoid-smoothed stage transitions and user-calibrated scaling.

Piecewise allometric equations (L in mm):
    - Stage 1 (yolk-sac larvae, 3-6.4 mm):   log W = 1.343 * log L - 3.798
    - Stage 2 (ascending phase, 6.45-11.95 mm): log W = 3.896 * log L - 5.801
    - Stage 3 (juvenile, 12-80 mm):           log W = 2.962 * log L - 4.798
    - Adult (>80 mm):                         log W = 3.08 * log L - 5.06

Calibration: Based on empirical measurement 0.002 g -> 5 mm, length_scale = 0.80.

References:
    FAO length-weight relationship database for Micropterus salmoides.
"""

import math
import numpy as np
from typing import Tuple
from dataclasses import dataclass
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CONFIG


# ============================================================
# Configuration
# ============================================================

@dataclass
class LengthWeightParams:
    """Parameters for the piecewise length-weight allometric relationship.

    Attributes:
        stage1_a: Log-intercept for yolk-sac larval stage.
        stage1_b: Log-slope for yolk-sac larval stage.
        stage1_L_max: Upper length boundary (mm) for stage 1.
        stage2_a: Log-intercept for ascending growth phase.
        stage2_b: Log-slope for ascending growth phase.
        stage2_L_max: Upper length boundary (mm) for stage 2.
        stage3_a: Log-intercept for juvenile stage.
        stage3_b: Log-slope for juvenile stage.
        stage3_L_max: Upper length boundary (mm) for stage 3.
        adult_a: Log-intercept for adult stage.
        adult_b: Log-slope for adult stage.
        transition_width: Sigmoid transition width for smooth boundary blending.
        length_scale: Calibration scaling factor based on empirical data
            (0.002 g -> 5 mm; standard formula yields 6.25 mm;
            scale = 5 / 6.25 = 0.80).
    """
    # Stage 1: yolk-sac larvae
    stage1_a: float = -3.798
    stage1_b: float = 1.343
    stage1_L_max: float = 6.4

    # Stage 2: ascending growth phase
    stage2_a: float = -5.801
    stage2_b: float = 3.896
    stage2_L_max: float = 11.95

    # Stage 3: juvenile
    stage3_a: float = -4.798
    stage3_b: float = 2.962
    stage3_L_max: float = 80.0

    # Adult stage
    adult_a: float = -5.06
    adult_b: float = 3.08

    # Smooth transition width
    transition_width: float = 0.15

    # Calibration scaling factor (from empirical measurement)
    # 0.002 g -> 5 mm, standard formula gives 6.25 mm
    # Scaling factor = 5 / 6.25 = 0.80
    length_scale: float = 0.80


# Default parameters (user-calibrated version)
_params = LengthWeightParams()

# Precomputed boundary masses between stages
_W1 = 10 ** (_params.stage1_a + _params.stage1_b * np.log10(_params.stage1_L_max))  # ~0.002 g
_W2 = 10 ** (_params.stage2_a + _params.stage2_b * np.log10(_params.stage2_L_max))  # ~0.025 g
_W3 = 10 ** (_params.stage3_a + _params.stage3_b * np.log10(_params.stage3_L_max))  # ~7 g


# ============================================================
# Core length-weight conversion functions
# ============================================================

def _sigmoid(x: float, center: float, width: float) -> float:
    """Compute a sigmoid smoothing function for stage transitions.

    Args:
        x: Input value.
        center: Sigmoid midpoint.
        width: Transition width controlling steepness.

    Returns:
        Sigmoid output in [0, 1].
    """
    return 1.0 / (1.0 + np.exp(-(x - center) / width))


def mass_to_length(mass: float) -> float:
    """Convert body mass to total length using piecewise allometry.

    Applies sigmoid-smoothed transitions between ontogenetic stages.
    Parameters are read from CONFIG.length_weight.

    Args:
        mass: Body mass in grams.

    Returns:
        Total length in meters.
    """
    if mass <= 0:
        return 0.003 * CONFIG.length_weight.length_scale

    lw = CONFIG.length_weight
    log_w = np.log10(mass)
    width = lw.transition_width

    # Compute boundary masses between stages
    W1 = 10 ** (lw.stage1_a + lw.stage1_b * np.log10(lw.stage1_L_max))
    W2 = 10 ** (lw.stage2_a + lw.stage2_b * np.log10(lw.stage2_L_max))
    W3 = 10 ** (lw.stage3_a + lw.stage3_b * np.log10(lw.stage3_L_max))

    def calc_L(a: float, b: float) -> float:
        log_l = (log_w - a) / b
        return max(10 ** log_l, 0.1)

    L1 = calc_L(lw.stage1_a, lw.stage1_b)
    L2 = calc_L(lw.stage2_a, lw.stage2_b)
    L3 = calc_L(lw.stage3_a, lw.stage3_b)
    L4 = calc_L(lw.adult_a, lw.adult_b)

    def sigmoid(x: float, center: float, w: float) -> float:
        return 1.0 / (1.0 + np.exp(-(x - center) / w))

    log_W1, log_W2, log_W3 = np.log10(W1), np.log10(W2), np.log10(W3)
    w12 = sigmoid(log_w, log_W1, width)
    w23 = sigmoid(log_w, log_W2, width)
    w34 = sigmoid(log_w, log_W3, width)

    log_L = (
        (1 - w12) * np.log10(L1) +
        w12 * (1 - w23) * np.log10(L2) +
        w12 * w23 * (1 - w34) * np.log10(L3) +
        w12 * w23 * w34 * np.log10(L4)
    )

    length_mm = (10 ** log_L) * lw.length_scale
    length_m = length_mm / 1000.0

    return np.clip(length_m, 0.003, 1.0)


def length_to_mass(length_m: float) -> float:
    """Convert total length to body mass via numerical inversion.

    Uses geometric bisection to invert the piecewise mass_to_length function.

    Args:
        length_m: Total length in meters.

    Returns:
        Body mass in grams.
    """
    if length_m <= 0:
        return 0.0001

    # Geometric bisection search
    mass_low = 0.0001
    mass_high = 10000.0

    for _ in range(50):
        mass_mid = np.sqrt(mass_low * mass_high)
        length_calc = mass_to_length(mass_mid)

        if abs(length_calc - length_m) / length_m < 0.001:
            return mass_mid

        if length_calc < length_m:
            mass_low = mass_mid
        else:
            mass_high = mass_mid

    return mass_mid


def get_growth_stage(mass: float) -> str:
    """Determine the ontogenetic growth stage from body mass.

    Args:
        mass: Body mass in grams.

    Returns:
        Growth stage label string.
    """
    if mass <= _W1:
        return "Stage 1 - Yolk-sac larvae"
    elif mass <= _W2:
        return "Stage 2 - Ascending phase"
    elif mass <= _W3:
        return "Stage 3 - Juvenile"
    else:
        return "Adult"


# ============================================================
# Buoyancy-related calculations
# ============================================================

def calculate_fish_volume(length_m: float) -> float:
    """Calculate fish body volume approximated as a prolate ellipsoid.

    Uses width and height ratios from CONFIG.physics.

    Args:
        length_m: Total length in meters.

    Returns:
        Body volume in cubic meters.
    """
    width_ratio = CONFIG.physics.fish_width_ratio
    height_ratio = CONFIG.physics.fish_height_ratio

    width = length_m * width_ratio
    height = length_m * height_ratio

    return (4 / 3) * math.pi * (length_m / 2) * (width / 2) * (height / 2)


def calculate_buoyancy_forces(length_m: float,
                              water_density: float = None,
                              fish_density: float = None) -> Tuple[float, float]:
    """Calculate net gravitational and buoyancy forces on the fish body.

    Args:
        length_m: Total length in meters.
        water_density: Water density in kg/m^3. Defaults to CONFIG value.
        fish_density: Fish tissue density in kg/m^3. Defaults to CONFIG value.

    Returns:
        Tuple of (net_gravity_in_water, gravity_in_air) in Newtons.
            net_gravity_in_water: Apparent weight submerged (clipped to [0.0001, 10.0]).
            gravity_in_air: Weight in air (minimum 0.001 N).
    """
    if water_density is None:
        water_density = CONFIG.physics.water_density
    if fish_density is None:
        fish_density = CONFIG.physics.fish_density

    gravity = CONFIG.physics.gravity
    volume = calculate_fish_volume(length_m)

    buoyancy_force = water_density * volume * gravity
    fish_gravity = fish_density * volume * gravity

    net_gravity_in_water = fish_gravity - buoyancy_force
    gravity_in_air = fish_gravity

    net_gravity_in_water = np.clip(net_gravity_in_water, 0.0001, 10.0)
    gravity_in_air = max(gravity_in_air, 0.001)

    return net_gravity_in_water, gravity_in_air


# ============================================================
# Swimming speed calculations
# ============================================================

def calculate_sustained_speed(length_m: float) -> float:
    """Calculate sustained swimming speed based on body length.

    Estimates sustained cruising speed using size-dependent body-lengths
    per second (BL/s) scaling, following general teleost swimming
    performance relationships.

    Args:
        length_m: Total length in meters.

    Returns:
        Sustained swimming speed in m/s.
    """
    length_cm = length_m * 100

    if length_cm <= 5:
        bl_per_s = 3.0  # Small fish: higher relative speed
    elif length_cm <= 15:
        bl_per_s = 2.5
    elif length_cm <= 30:
        bl_per_s = 2.0
    else:
        bl_per_s = 1.5  # Large fish: lower relative speed

    return length_m * bl_per_s


def calculate_burst_speed(length_m: float, burst_multiplier: float = 3.0) -> float:
    """Calculate burst swimming speed.

    Args:
        length_m: Total length in meters.
        burst_multiplier: Multiplicative factor over sustained speed.

    Returns:
        Burst swimming speed in m/s.
    """
    return calculate_sustained_speed(length_m) * burst_multiplier


# ============================================================
# Predation and foraging
# ============================================================

def calculate_capture_radius(length_m: float) -> float:
    """Calculate food capture (strike) radius from CONFIG parameters.

    Uses size-dependent capture radius ratios defined in
    CONFIG.interaction.

    Args:
        length_m: Total length in meters.

    Returns:
        Capture radius in meters.
    """
    import sys
    import os
    # Ensure root config is importable
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import CONFIG

    ic = CONFIG.interaction

    if length_m <= ic.capture_radius_small_threshold:
        ratio = ic.capture_radius_small
    elif length_m <= ic.capture_radius_medium_threshold:
        ratio = ic.capture_radius_medium
    else:
        ratio = ic.capture_radius_large

    return max(length_m * ratio, ic.capture_radius_min)


def calculate_detection_range(length_m: float) -> float:
    """Calculate sensory detection range.

    Args:
        length_m: Total length in meters.

    Returns:
        Detection range in meters (minimum 0.5 m).
    """
    return max(length_m * 10.0, 0.5)


def calculate_strike_range(length_m: float) -> float:
    """Calculate predatory strike distance.

    Args:
        length_m: Total length in meters.

    Returns:
        Strike range in meters (minimum 0.02 m).
    """
    return max(length_m * 0.8, 0.02)


# ============================================================
# Physiological calculations
# ============================================================

def calculate_stomach_capacity(body_mass_g: float, ratio: float = 0.05) -> float:
    """Calculate maximum stomach capacity.

    Args:
        body_mass_g: Body mass in grams.
        ratio: Stomach capacity as a fraction of body mass.

    Returns:
        Stomach capacity in grams.
    """
    return body_mass_g * ratio


def calculate_fish_energy_density(body_mass_g: float) -> float:
    """Calculate mass-dependent fish body energy density (wet basis).

    Uses the allometric relationship:
        E_fish(W) = alpha * W^beta

    where W is body mass in grams and parameters are from CONFIG.metabolism.

    Args:
        body_mass_g: Body mass in grams.

    Returns:
        Energy density in kJ/g (wet weight).
    """
    m = max(float(body_mass_g), 1e-6)
    mc = CONFIG.metabolism
    density = mc.fish_energy_density_alpha * (m ** mc.fish_energy_density_beta)
    return float(np.clip(density, mc.fish_energy_density_min, mc.fish_energy_density_max))


def calculate_growth_increment(current_mass_g: float,
                               growth_rate: float = 0.01) -> Tuple[float, float]:
    """Calculate a single growth increment in mass and length.

    Args:
        current_mass_g: Current body mass in grams.
        growth_rate: Proportional mass growth rate per step.

    Returns:
        Tuple of (new_mass_g, new_length_m).
    """
    new_mass = current_mass_g * (1 + growth_rate)
    new_length = mass_to_length(new_mass)
    return new_mass, new_length


def calculate_q10_factor(water_temp: float,
                         optimal_temp: float = 25.0,
                         q10: float = 2.3) -> float:
    """Calculate Q10 temperature coefficient factor.

    Computes the metabolic rate multiplier relative to the optimal
    temperature using Van't Hoff's Q10 rule.

    Args:
        water_temp: Ambient water temperature in degrees Celsius.
        optimal_temp: Reference optimal temperature in degrees Celsius.
        q10: Q10 coefficient (default 2.3 for largemouth bass).

    Returns:
        Dimensionless temperature scaling factor.
    """
    return q10 ** ((water_temp - optimal_temp) / 10.0)


# ============================================================
# Configuration functions
# ============================================================

def set_length_scale(scale: float) -> None:
    """Set the length calibration scaling factor.

    Args:
        scale: Scaling factor (>1 increases length, <1 decreases length).
    """
    global _params
    _params.length_scale = scale


def get_current_config() -> dict:
    """Retrieve the current length-weight configuration state.

    Returns:
        Dictionary containing length_scale and boundary masses W1, W2, W3.
    """
    return {
        'length_scale': _params.length_scale,
        'W1': _W1,
        'W2': _W2,
        'W3': _W3,
    }


# ============================================================
# Self-test
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Biological Formula Test (v2 Piecewise Length-Weight Model)")
    print("=" * 70)

    print(f"\nCurrent config: length_scale = {_params.length_scale}")
    print(f"Stage boundaries: W1={_W1:.4f} g, W2={_W2:.4f} g, W3={_W3:.2f} g")

    # Validate against empirical data
    print("\n[Empirical Data Validation]")
    print(f"Measured: 0.002 g -> 5 mm")
    calc_L = mass_to_length(0.002) * 1000
    print(f"Computed: 0.002 g -> {calc_L:.2f} mm (error: {abs(calc_L - 5) / 5 * 100:.1f}%)")

    # Mass-to-length conversion table
    print("\n[Mass-Length Conversion Table]")
    print("-" * 60)
    print(f"{'Mass (g)':<12} {'Length (cm)':<12} {'Length (mm)':<12} {'Stage'}")
    print("-" * 60)

    for mass in [0.002, 0.01, 0.1, 1, 5, 10, 20, 40, 80, 160, 320]:
        L_m = mass_to_length(mass)
        L_cm = L_m * 100
        L_mm = L_m * 1000
        stage = get_growth_stage(mass)
        print(f"{mass:<12.3f} {L_cm:<12.2f} {L_mm:<12.1f} {stage}")

    print("-" * 60)
    print("\nTest complete.")
