#!/usr/bin/env python3
"""
Unified Configuration Module
=============================
Central configuration for the largemouth bass (Micropterus salmoides)
reinforcement learning simulation. All bioenergetic, physical, and
environmental parameters are defined here as frozen dataclasses to
ensure immutability and reproducibility.

Usage:
    from config import CONFIG

    smr_coef = CONFIG.metabolism.smr_coefficient
    tank_radius = CONFIG.environment.tank_radius
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple


# ============================================================
# 1. Biological Constants (Metabolism Subsystem)
# ============================================================

@dataclass(frozen=True)
class MetabolismConfig:
    """Metabolic parameters — single source of truth."""
    # Standard metabolic rate parameters (Rice et al., 1983 — Table 3 calibrated values)
    smr_coefficient: float = 0.3393  # a2: mg O2·g⁻¹·h⁻¹
    smr_exponent: float = -0.3486  # b2: mass-dependent exponent (negative)
    smr_temp_coeff: float = 0.028  # m: temperature-dependence coefficient
    smr_swim_coeff: float = 0.0196  # g: swimming speed coefficient
    smr_swim_speed: float = 5.0  # S: cruising speed (cm/s)
    smr_time_conv: float = 0.024  # r: unit conversion (J·g⁻¹·h⁻¹ → kJ·g⁻¹·d⁻¹)

    # Energy conversion
    oxycalorific_coeff: float = 0.0136  # kJ/mg O2

    # Swimming efficiency (Beamish, 1978)
    swimming_efficiency: float = 0.20

    # Active metabolism multipliers
    active_metabolism_cruise: float = 3.0  # SMR multiplier during cruising
    active_metabolism_burst: float = 10.0  # SMR multiplier during burst
    action_threshold: float = 0.01  # minimum effective action magnitude

    # Digestion parameters (legacy; refined model now uses nutrient-specific ADCs)
    digestion_efficiency: float = 0.85  # backward compatibility only
    sda_coefficient: float = 0.200  # refined model calibrated kSDA
    excretion_coefficient: float = 0.150  # refined model calibrated u

    # Energy density
    energy_density_fish: float = 7.0  # legacy fallback (kJ/g, wet basis)
    pellet_energy_density: float = 19.4  # gross feed energy density (kJ/g)
    # Literature mass-dependent fish energy density: E_fish(W) = alpha * W^beta
    fish_energy_density_alpha: float = 5.01
    fish_energy_density_beta: float = 0.046
    fish_energy_density_min: float = 3.5
    fish_energy_density_max: float = 7.5

    # ===== Refined feed architecture: composition × ADCs =====
    # Default commercial diet profile (fraction of feed mass, dry matter basis)
    feed_protein_fraction: float = 0.53
    feed_lipid_fraction: float = 0.08
    feed_carbohydrate_fraction: float = 0.0

    # Nutrient-specific apparent digestibility coefficients (ADCs)
    adc_protein: float = 0.877
    adc_lipid: float = 0.880
    adc_carbohydrate: float = 0.30
    include_carbohydrate_energy: bool = False

    # Q10 temperature coefficient
    q10: float = 1.32
    optimal_temp: float = 27.5  # optimal temperature (°C)

    # Energy allocation ratios
    growth_allocation: float = 0.31  # fraction allocated to growth
    maintenance_allocation: float = 0.69  # fraction allocated to maintenance

    # ===== Dynamic growth allocation parameters (literature-fitted) =====
    growth_allocation_coefficient: float = 1.22
    growth_allocation_exponent: float = -0.22
    growth_allocation_min: float = 0.24
    growth_allocation_max: float = 0.62
    juvenile_growth_boost_threshold: float = 120.0
    juvenile_growth_boost_max: float = 1.18

    # Starvation body-mass loss parameters
    lipid_fraction: float = 0.08  # initial lipid fraction (8%)
    lipid_scaling_exponent: float = 0.25  # lipid scaling exponent
    protein_fraction: float = 0.16  # protein fraction (16%)
    lipid_energy_density: float = 39.5  # lipid energy density (kJ/g)
    protein_energy_density: float = 23.6  # protein energy density (kJ/g)
    carbohydrate_energy_density: float = 17.2  # carbohydrate energy density (kJ/g)
    starvation_protein_efficiency: float = 0.7  # protein catabolism efficiency


# ============================================================
# 1.5 Rest State Configuration
# ============================================================

@dataclass(frozen=True)
class RestStateConfig:
    """Rest state parameters — based on biological literature."""

    # ===== Metabolic regulation =====
    # Resting BMR reduced to 70% of active state (Chabot & Claireaux, 2008)
    base_metabolism_reduction: float = 0.70
    # Deep rest minimum: 60%
    deep_rest_metabolism_min: float = 0.60
    # Metabolic decay rate (exponential decay rate parameter)
    metabolism_decay_rate: float = 0.05
    # Steps to reach deep rest (reference)
    deep_rest_threshold_steps: int = 300

    # ===== Growth promotion =====
    # Rest-enhanced growth efficiency (Takahashi et al., 2006 - )
    rest_growth_bonus_base: float = 1.15  # initial 15% boost
    rest_growth_bonus_max: float = 1.50  # deep rest max 50% boost
    # Growth bonus increase rate
    growth_bonus_rate: float = 0.05

    # ===== Physical movement constraints =====
    # Propulsion force greatly reduced during rest
    rest_propulsion_factor: float = 0.25  # 25% propulsion only (posture maintenance)
    # Additional velocity damping
    rest_velocity_damping: float = 0.85  # velocity decays to 85% per step

    # ===== Sensory capability reduction =====
    # Vision range reduced during rest (decreased alertness)
    rest_vision_reduction: float = 0.50  # reduced to 50%
    # Food detection range reduced
    rest_food_detection_reduction: float = 0.40  # reduced to 40%

    # ===== Reaction delay =====
    # Based on Domenici & Blake (1997) reaction time data
    active_reaction_delay: float = 0.3  # Reaction delay (s)
    rest_reaction_delay: float = 1.2  # Reaction delay (s)

    # ===== State transition =====
    # Switch cooldown time (s)
    rest_to_active_cooldown: float = 2.5  # rest-to-active cooldown
    active_to_rest_cooldown: float = 1.2  # active-to-rest cooldown (shortened to reduce lock-in)
    # minimum rest duration (prevent rapid toggling)
    min_rest_duration_steps: int = 6

    # ===== Hunger wake-up=====
    hunger_wake_stomach_threshold: float = 12.0
    forced_active_hunger_threshold: float = 14.0
    forced_active_no_food_steps: int = 180

    # ===== Digestion regulation =====
    # Slower digestion during rest but improved absorption
    rest_digestion_rate: float = 0.70  # digestion rate reduced to 70%
    rest_absorption_bonus: float = 1.10  # absorption efficiency +10%
    rest_sda_reduction: float = 0.85  # SDA reduced by 15%

    # === Rest state pitch control ===
    rest_target_pitch: float = 0.0
    rest_pitch_restoration: float = 0.3
    emergency_wake_threat_distance: float = 0.22
    emergency_wake_cooldown: float = 1.0
    proactive_wake_threat_distance: float = 0.38


# ============================================================
# 1.6 Digestion System Configuration
# ============================================================

@dataclass(frozen=True)
class DigestionConfig:
    """Digestion system configuration — power-law model from literature.

    Reference: Seasonal aspects of daily ration and diet of largemouth bass

    Gastric evacuation rate formula:
    - Small fish (30-54g):   ln α = 1.670 ln T - 7.455
    - Large fish (100-300g): ln α = 1.707 ln T - 7.103

    Large fish
    """

    # ===== Literature power-law model parameters=====
    # Small fish (30-54g): ln α = a_small * ln(T) + b_small
    alpha_a_small: float = 1.670
    alpha_b_small: float = -7.455

    # Large fish (100-300g): ln α = a_large * ln(T) + b_large
    alpha_a_large: float = 1.707
    alpha_b_large: float = -7.103

    # Small fish/Large fish
    mass_transition_center: float = 75.0  # transition center mass (g)
    mass_transition_width: float = 25.0  # transition width (g)

    # ===== Effective temperature range =====
    min_temp_for_digestion: float = 10.0  # below this temp digestion is negligible
    max_temp_for_digestion: float = 35.0  # above this temp digestion is inhibited

    # ===== 1 Fix=====
    # 1.14%
    #  FeedingConfig.stomach_capacity_ratio

    # =====  =====
    base_absorption_efficiency: float = 0.85
    max_absorption_efficiency: float = 0.92
    min_absorption_efficiency: float = 0.70

    #  = base + meal_size_effect *
    meal_size_absorption_effect: float = -0.005  # 1%0.5%


# ============================================================
# 2.
# ============================================================

@dataclass(frozen=True)
class GrowthConfig:
    """Growth-related constants. - single source of truth"""
    growth_threshold: float = 0.66  #  (%)
    growth_rate: float = 0.0033  #  (%)
    max_growth_per_step: int = 7  #

    # - (Carlander, 1977)
    length_weight_a: float = 0.0148
    length_weight_b: float = 3.02

    # =====  =====
    # SGR(%/d) = sgr_coefficient × W^sgr_exponent
    sgr_coefficient: float = 100.0  # a
    sgr_exponent: float = -0.891  # b

    # ===== FCR =====
    fcr_small_fish: float = 1.13  # Small fishFCR (43-166g)
    fcr_large_fish: float = 2.0  # Large fishFCR (285-475g)
    fcr_transition_mass: float = 200.0  # FCR

    # ===== FCR =====
    target_fcr_pond: float = 1.13           # FCR
    target_fcr_raceway: float = 1.30        # FCR
    target_ere: float = 0.38                #
    target_nre: float = 0.35                #


# ============================================================
# 3.
# ============================================================

@dataclass(frozen=True)
class PhysicsConfig:
    """ - single source of truth"""
    water_density: float = 1000.0  # kg/m³
    air_density: float = 1.2  # kg/m³
    fish_density: float = 1055.0  # kg/m³
    gravity: float = 9.8  # m/s²
    propulsion_efficiency_air: float = 0.03 #

    # drag coefficient
    drag_coefficient_surface: float = 0.35
    drag_coefficient_air: float = 0.1

    surface_tension_zone: float = 0.05  #  (m)

    collision_velocity_reduction: float = 0.5
    collision_damage: float = 1.0  #

    force_amplification: float = 1.0

    fish_width_ratio: float = 0.25  # /
    fish_height_ratio: float = 0.18  # /

    net_gravity_min: float = 0.001
    net_gravity_max: float = 0.002
    air_gravity_multiplier: float = 3.0
    air_gravity_min: float = 0.01

    #  (BL/s)
    swim_speed_small_bl: float = 1.0  # ≤4cm
    swim_speed_medium_bl: float = 0.8  # ≤8cm
    swim_speed_large_bl: float = 0.6  # >8cm
    swim_speed_small_threshold: float = 8.0  # cm
    swim_speed_medium_threshold: float = 16.0  # cm
    burst_speed_multiplier: float = 4.0  #

    # === new ===
    water_kinematic_viscosity: float = 0.89e-6  # ← new (m²/s)

    reynolds_laminar_threshold: float = 3000.0   # →20g3000-8000
    reynolds_turbulent_threshold: float = 30000.0  # →

    drag_coeff_laminar: float = 0.55   # drag coefficientSmall fish
    drag_coeff_transition: float = 0.35  # 20g
    drag_coeff_turbulent: float = 0.20   # Large fish

    # === new ===
    max_pitch_angle: float = 65.0  # 60-80°Webb & Skadsen 1980, Higham et al. 2005
    pitch_rate_max: float = 50.0  # /s30-50°/sWebb 1984, Drucker & Lauder 1999
    pitch_restoration_factor: float = 2.0   # max_pitch_angle

    # === new ===
    lift_coefficient: float = 0.70  # 0.25
    min_lift_velocity_bl: float = 0.15  # 0.5Small fish

    # === new ===
    # 1BL/s1BL/s
    high_speed_turn_damping: float = 0.50
    turn_speed_loss_coeff: float = 0.18
    # 70%
    turn_speed_loss_min: float = 0.55
    # heading
    turn_lateral_slip: float = 0.04
    # <0.5 BL/s
    low_speed_turn_boost: float = 1.2


# ============================================================
# 4.5
# ============================================================

@dataclass(frozen=True)
class InertiaConfig:
    """"""

    # ===  ===
    #  AI
    #  0.68 action  > 0.82tanh  ~0.8 BL/s
    #  0.55 > 0.74PPO tanh
    burst_threshold: float = 0.55

    action_smoothing_factor: float = 0.12
    rest_action_smoothing_factor: float = 0.08

    #  (/s)
    max_turn_rate_cruise: float = 35.0
    max_turn_rate_burst: float = 150.0
    max_turn_rate_rest: float = 30.0
    turn_rate_smoothing: float = 0.3

    # Speed Limiting (/s²)
    # 5g: 8×2.0=16BL/s²，20g: 8×1.0=8BL/s²，500g: 8×0.15=1.2BL/s²
    # 8.0(1.5BL/s)~2(~4BL/s)7
    max_acceleration_bl_s2: float = 8.0
    max_deceleration_bl_s2: float = 15.0
    burst_acceleration_multiplier: float = 1.5

    buoyancy_control_smoothing: float = 0.3

    # ===  ===
    natural_deceleration_factor: float = 0.85
    rest_velocity_decay: float = 0.92

    # === new ===
    # Small fish(Re<1000)Large fish(Re>10000)
    #  →  →
    coast_decay_small_fish: float = 0.94   # Small fish94%0.88
    coast_decay_large_fish: float = 0.992  # Large fish99.2%0.985

    # === new===
    thrust_mass_exponent: float = 0.67  #
    small_fish_mass_kg: float = 0.005  # Small fish5g
    large_fish_mass_kg: float = 0.1  # Large fish100g

    # === newBurst-and-Coast  ===
    coast_action_threshold: float = 0.20  # ← new
    coast_min_duration: int = 5  # ← new

# ============================================
    # ============================================
    #
    #   (cruise) → (approach) → (strike)
    #
    #   Webb (1984) - Body form, locomotion and foraging
    #   Domenici & Blake (1997) - The kinematics of fast-starts

    # ---  (/) ---
    # cruise_smoothing  →  →
    cruise_smoothing: float = 0.30           # 8
    cruise_deceleration: float = 0.88        #

    # ---  () ---
    approach_smoothing: float = 0.65         #
    approach_deceleration: float = 0.60      #
    approach_trigger_distance: float = 0.40  #  (m)

    # ---  (S-start/) ---
    # 2~15cm~30cm
    strike_smoothing: float = 0.85           # ~20msC-start
    strike_deceleration: float = 0.52        #
    strike_trigger_distance: float = 0.35    # 2

    # ---  (C-start) ---
    escape_smoothing: float = 0.70           #  (~20ms)
    escape_deceleration: float = 0.90        #
    escape_trigger_distance: float = 0.30    #  (m)

# ============================================================
# 4.
# ============================================================

@dataclass(frozen=True)
class PerceptionConfig:
    """ - single source of truth"""
    # ~180°300-330°30-60°
    #  300°±150°60°
    vision_range: float = 3.0  # vision range (m)
    vision_angle: float = 300.0  #

    #  1-2 9.3cm0.1-0.2m0.5m
    lateral_line_range: float = 0.5  # lateral line range (m)1.0m0.5m

    food_detection_range: float = 3.0  # food detection range (m)

    num_rays: int = 8  #
    max_obstacle_distance: float = 1.5  #  (m)

    enable_boundary_vector: bool = True
    enable_surface_detection: bool = True

    max_fish_observed: int = 3   # 23
    max_threat_slots: int = 2    #
    max_prey_slots: int = 1      #
    max_food_observed: int = 3


# ============================================================
# 5. /
# ============================================================

@dataclass(frozen=True)
class FeedingConfig:
    """ - """

    # =====  =====
    daily_feeding_rate: float = 0.024  #  2.2%
    feedings_per_day: int = 2  #

    # =====  =====
    floating_pellets_min: int = 0  #
    floating_pellets_max: int = 3  #
    sinking_pellets_min: int = 7  #
    sinking_pellets_max: int = 9  #

    pellet_mass_min: float = 0.02  #  (g)
    pellet_mass_max: float = 2.5  #  (g)

    # =====  =====
    settle_trigger_ratio: float = 0.4  # feeding interval
    settle_speed_floating: float = 0.002  #  (m/)
    settle_speed_sinking: float = 0.003  #  (m/)

    # =====  =====
    floating_speed_min: float = 0.003  #  (m/)
    floating_speed_max: float = 0.005  #  (m/)
    floating_direction_change: int = 300  #  ()

    sinking_speed_min: float = 0.003  #  (m/)
    sinking_speed_max: float = 0.005  #  (m/)
    sinking_direction_change: int = 300  #  ()

    sinking_vertical_factor: float = 0.3       #
    settling_horizontal_speed_max: float = 0.005  #  (m/)

    # =====  =====
    boundary_buffer: float = 0.15  # boundary buffer (m)
    surface_buffer: float = 0.05  # surface buffer (m)
    bottom_buffer: float = 0.10  # bottom buffer (m)
    spread_radius: float = 1.2  #  (m)

    # =====  =====
    stomach_capacity_ratio: float = 0.015  # 1.50%
    stochastic_meal_intake_enabled: bool = True
    meal_intake_cv: float = 0.15
    meal_intake_min_factor: float = 0.6
    meal_intake_max_factor: float = 1.4
    mass_intake_scaling_enabled: bool = True
    mass_intake_scaling_ref_mass_g: float = 200.0
    mass_intake_scaling_exponent: float = -1.15
    mass_intake_scaling_min: float = 0.47
    mass_intake_scaling_max: float = 1.00

    # ===== Diet profile for refined metabolism (composition × ADCs) =====
    diet_protein_fraction: float = 0.53
    diet_lipid_fraction: float = 0.08
    diet_carbohydrate_fraction: float = 0.0

    diet_adc_protein: float = 0.90
    diet_adc_lipid: float = 0.90
    diet_adc_carbohydrate: float = 0.30
    diet_include_carbohydrate_energy: bool = False

    # ===== =====
    env_food_max_count:      int   = 14     #
    ambient_spawn_prob:      float = 0.004  #
    surface_env_spawn_prob:  float = 0.003  #
    benthic_spawn_prob:      float = 0.003  #
    attached_spawn_prob:     float = 0.008  #
    ambient_max_age:         int   = 300    #
    surface_env_max_age:     int   = 300    #
    benthic_max_age:         int   = 300    #
    attached_max_age:        int   = 300    #

    # ===== =====
    # float_duration: int = 600           #  settle_trigger_ratio
    # sink_rate: float = 0.01             #  settle_speed_*
    # feeding_interval: int = 320         #
    # pellet_mass: float = 0.1            #


# ============================================================
# new
# ============================================================

@dataclass(frozen=True)
class EnergySystemConfig:
    """
     -


    -  = f(, ) -
    -  -
    """

    # =====  =====
    # initial energy
    baseline_energy: float = 80.0

    #  = baseline + gain_max × (1 - 1/(1 + mass_gain × gain_sensitivity))
    energy_gain_max: float = 20.0  # baseline + gain_max = 100
    energy_gain_sensitivity: float = 5.0  #

    # =====  =====
    energy_healthy_threshold: float = 55.0  #
    energy_mild_fatigue_threshold: float = 40.0  #
    energy_moderate_fatigue_threshold: float = 15.0  #
    #  moderate_fatigue_threshold

    # =====  >= 60%=====
    healthy_speed_factor: float = 1.0
    healthy_reaction_factor: float = 1.0
    healthy_propulsion_factor: float = 1.0

    # ===== 40% <=  < 60%=====
    mild_speed_min: float = 0.85  # reduced to85%
    mild_reaction_max: float = 1.2  # 20%
    mild_propulsion_min: float = 0.90  # 90%

    # ===== 20% <=  < 40%=====
    moderate_speed_min: float = 0.60  # reduced to60%
    moderate_reaction_max: float = 1.6  # 60%
    moderate_propulsion_min: float = 0.70  # 70%
    burst_disable_threshold: float = 15.0  #

    # =====  < 20%=====
    severe_speed_min: float = 0.35  # reduced to35%
    severe_reaction_max: float = 2.2  # 120%
    severe_propulsion_min: float = 0.45  # 45%

    # ===== =====
    rest_energy_recovery_rate: float = 0.0  # 0

    # =====  =====
    debug_energy_changes: bool = False


# ============================================================
# 6.
# ============================================================

@dataclass(frozen=True)
class InteractionConfig:
    """ - single source of truth"""
    min_predation_size_ratio: float = 1.5  # min predation size ratio
    predation_energy_efficiency: float = 0.7  #

    threat_size_ratio: float = 1.5  # threat size ratio

    predation_attempt_fatigue: float = 2.0
    predation_success_fatigue: float = 5.0
    escape_fatigue: float = 3.0

    attack_damage_base: float = 5.0

    small_fish_count: int = 6
    medium_fish_count: int = 3
    large_fish_count: int = 1

    capture_radius_small: float = 0.8  # ≤5cm
    capture_radius_medium: float = 0.65  # ≤10cm~6cm
    capture_radius_large: float = 0.25  # >10cm
    capture_radius_small_threshold: float = 0.08  # 8cm ()
    capture_radius_medium_threshold: float = 0.25  # 23cm ()
    capture_radius_min: float = 0.03  #

    strike_range_multiplier: float = 0.8  #  =  ×
    strike_range_min: float = 0.05  #


# ============================================================
# 7.
# ============================================================

@dataclass(frozen=True)
class BuoyancyConfig:
    '''



    - Harden Jones (1951), Alexander (1959) - Boyle
    - Denton et al. (1972) -
    - Harden Jones & Scholes (1985) -
    '''

    # =====  =====
    # 5-7%
    neutral_swimbladder_ratio: float = 0.05

    max_volume_change_ratio: float = 0.35

    min_volume_ratio: float = 0.02

    # =====  =====
    fish_tissue_density: float = 1055.0  # kg/m³
    water_density: float = 1000.0  # kg/m³
    gas_density: float = 1.2  # kg/m³

    # =====  =====
    # physoclistous0.0003-0.001/sAlexander 1966
    # RL0.005/s~70s
    # ""
    secretion_rate_max: float = 0.005  # s0.5%
    absorption_rate_max: float = 0.010  # s1%
    rest_adjustment_factor: float = 0.75  #

    # =====  =====
    leakage_rate: float = 0.000002  # s0.0002%
    depth_leakage_factor: float = 0.2  #

    # =====  =====
    secretion_energy_base: float = 2.5  # kJ/mol
    absorption_energy_base: float = 0.25  # kJ/mol
    maintenance_energy_rate: float = 0.0001  # kJ/s
    depth_energy_factor: float = 0.1  #

    # ===== Boyle=====
    atmospheric_pressure: float = 1.0  # atm
    pressure_per_meter: float = 0.1  # atm/m

    # =====  =====
    swimbladder_aspect_ratio: float = 3.0
    wall_permeability: float = 0.5

# ============================================================
# 8.
# ============================================================

@dataclass(frozen=True)
class EnvironmentConfig:
    """ - single source of truth"""
    tank_radius: float = 1.5  #  (m)
    tank_depth: float = 0.8  #  (m)
    water_surface_y: float = 0.0  #

    water_current_strength: float = 0.3

    # water temperature
    water_temp: float = 25.0  # °C
    enable_temp_daily_cycle: bool = True
    temp_daily_amplitude: float = 0.8
    temp_daily_noise_sd: float = 0.2
    temp_peak_hour: float = 14.0
    temp_daily_clamp_range: float = 3.0

    time_step: float = 0.1  # time step (s)
    time_acceleration: float = 300  #
    max_episode_steps: int = 86400  #


# ============================================================
# 7.5 Obstacle Configuration（new）
# ============================================================

@dataclass(frozen=True)
class ObstacleConfig:
    """Obstacle Configuration - """
    # episode
    min_obstacles: int = 2
    max_obstacles: int = 6

    rock_radius_min: float = 0.05   # 5cm
    rock_radius_max: float = 0.20   # 20cm

    # Placement constraints
    min_distance_between: float = 0.15      # min distance between obstacles (m)
    min_distance_from_wall: float = 0.10    # min distance from wall (m)
    spawn_exclusion_radius: float = 0.30    # no obstacles near agent spawn (m)


# ============================================================
# 7.6 Environment Randomization Configuration（new）
# ============================================================

@dataclass(frozen=True)
class EnvironmentRandomizationConfig:
    """Environment Randomization Configuration"""
    # enable random shapes（False = fixed circular for evaluation）
    enable_random_shape: bool = True

    # Shape probabilities（should sum to ≈ 1.0）
    circular_probability: float = 0.5    # circular (post-irregular actual: = 0.5*(1-irr)）
    # rectangular probability = 1 - circular_probability = 0.5*(1-irr)

    # irregular polygon probability（pond/river/wetland），0 = disabled
    irregular_probability: float = 0.30

    radius_range: tuple = (1.0, 2.5)

    rect_width_range: tuple = (1.5, 3.5)    # X
    rect_length_range: tuple = (1.5, 3.5)   # Z

    depth_range: tuple = (0.5, 1.2)


# ============================================================
# new
# ============================================================

@dataclass(frozen=True)
class LiteratureValidationConfig:
    ''' - '''

    # =====  =====
    experiment_duration_days: int = 72
    initial_weight_g: float = 43.0
    water_temp_range: Tuple[float, float] = (20.0, 34.1)

    # =====  =====
    feed_moisture_pct: float = 8.0
    feed_protein_pct: float = 50.0
    feed_lipid_pct: float = 9.0
    feed_energy_kj_g: float = 19.4
    feed_carbon_pct: float = 47.0

    # ===== F3=====
    target_final_weight_g: float = 166.1
    target_weight_gain_g: float = 122.9
    target_fcr: float = 1.13
    target_ere_pct: float = 38.04
    target_nre_pct: float = 34.61
    target_daily_feed_rate_pct: float = 2.20

    # ===== =====
    fish_initial_moisture_pct: float = 70.8
    fish_initial_protein_pct: float = 16.3
    fish_initial_lipid_pct: float = 8.3
    fish_initial_energy_kj_g: float = 6.81

    # ===== =====
    fish_final_moisture_pct: float = 68.2
    fish_final_protein_pct: float = 18.0
    fish_final_lipid_pct: float = 8.4
    fish_final_energy_kj_g: float = 7.65


# ============================================================
# 9. Agent
# ============================================================

@dataclass(frozen=True)
class AgentInitConfig:
    """Agent initial state configuration."""
    initial_mass: float = 20.0  # initial body mass (g)
    initial_energy: float = 80.0  # initial energy (%)
    initial_stomach_fullness: float = 40.0  #  (%)  [0→401]

    spawn_x_range: Tuple[float, float] = (-0.3, 0.3)
    spawn_y_range: Tuple[float, float] = (-0.5, -0.2)
    spawn_z_range: Tuple[float, float] = (-0.3, 0.3)


# ============================================================
# 9.5 Mass_to_length
# ============================================================

@dataclass(frozen=True)
class LengthWeightConfig:
    """- - FAO + """

    # 1:  (3-6.4mm)
    stage1_a: float = -3.798
    stage1_b: float = 1.343
    stage1_L_max: float = 6.4

    # 2:  (6.45-11.95mm)
    stage2_a: float = -5.801
    stage2_b: float = 3.896
    stage2_L_max: float = 11.95

    # 3:  (12-80mm)
    stage3_a: float = -4.798
    stage3_b: float = 2.962
    stage3_L_max: float = 80.0

    #  (>80mm)
    adult_a: float = -5.06
    adult_b: float = 3.08

    # transition width
    transition_width: float = 0.15

    #  0.002g → 5mm
    length_scale: float = 0.80


# ============================================================
# 9.5
# ============================================================

@dataclass(frozen=True)
class AggressiveBehaviorConfig:
    """Large fish"""
    detection_range: float = 0.25
    chase_speed_multiplier: float = 0.18
    random_speed_multiplier: float = 0.4
    give_up_range: float = 0.35
    direction_noise: float = 0.15
    attack_cooldown_steps: int = 30
    attack_range: float = 0.15


@dataclass(frozen=True)
class SurfacePredatorConfig:
    """Surface predator behavior configuration."""
    surface_zone_depth: float = 0.15
    surface_zone_max: float = 0.15
    detection_range: float = 2.50
    chase_speed_multiplier: float = 0.16
    patrol_speed_multiplier: float = 0.4
    attack_damage: float = 2.0
    attack_cooldown_steps: int = 30
    attack_range: float = 0.2
    prey_detection_depth: float = 0.15        #
    give_up_depth: float = 0.25               #
    direction_noise: float = 0.1              #
    vertical_chase_limit: float = 0.05        #


@dataclass(frozen=True)
class FleeingBehaviorConfig:
    """"""
    detection_range: float = 0.16
    flee_speed_multiplier: float = 0.19
    safe_distance: float = 0.50
    flee_duration_max: int = 25
    direction_noise: float = 0.15

# ============================================================
# 10. Curriculum Learning Configuration
# ============================================================

@dataclass
class CurriculumStage:
    """"""
    name: str
    stage: int
    capture_multiplier: float
    predation_multiplier: float
    energy_cost_multiplier: float
    food_amount_multiplier: float
    success_threshold: Dict[str, float]


CURRICULUM_STAGES: List[CurriculumStage] = [
    CurriculumStage(
        name='', stage=0,
        capture_multiplier=3.5, predation_multiplier=2.5,
        energy_cost_multiplier=1.0, food_amount_multiplier=2.0,
        success_threshold={'avg_steps': 1200, 'avg_food': 6, 'success_rate': 0.5}
    ),
    CurriculumStage(
        name='', stage=1,
        capture_multiplier=2.8, predation_multiplier=2.0,
        energy_cost_multiplier=1.0, food_amount_multiplier=1.9,
        success_threshold={'avg_steps': 1800, 'avg_food': 10, 'success_rate': 0.4}
    ),
    CurriculumStage(
        name='A', stage=2,
        capture_multiplier=2.4, predation_multiplier=1.7,
        energy_cost_multiplier=1.0, food_amount_multiplier=1.8,
        success_threshold={'avg_steps': 2000, 'avg_food': 18, 'success_rate': 0.35}
    ),
    CurriculumStage(
        name='B', stage=3,
        capture_multiplier=2.2, predation_multiplier=1.5,
        energy_cost_multiplier=1.0, food_amount_multiplier=1.7,
        success_threshold={'avg_steps': 2100, 'avg_food': 25, 'success_rate': 0.3}
    ),
    CurriculumStage(
        name='', stage=4,
        capture_multiplier=2.0, predation_multiplier=1.3,
        energy_cost_multiplier=1.0, food_amount_multiplier=1.6,
        success_threshold={'avg_steps': 2200, 'avg_food': 45, 'success_rate': 0.3}
    ),
    CurriculumStage(
        name='A', stage=5,
        capture_multiplier=1.8, predation_multiplier=1.3,
        energy_cost_multiplier=1.0, food_amount_multiplier=1.5,
        success_threshold={'avg_steps': 2500, 'avg_food': 60, 'success_rate': 0.3}
    ),
    CurriculumStage(
        name='B', stage=6,
        capture_multiplier=1.6, predation_multiplier=1.25,
        energy_cost_multiplier=1.0, food_amount_multiplier=1.4,
        success_threshold={'avg_steps': 2500, 'avg_food': 75, 'success_rate': 0.3}
    ),
    CurriculumStage(
        name='C', stage=7,
        capture_multiplier=1.4, predation_multiplier=1.2,
        energy_cost_multiplier=1.0, food_amount_multiplier=1.3,
        success_threshold={'avg_steps': 2500, 'avg_food': 85, 'success_rate': 0.3}
    ),
    CurriculumStage(
        name='', stage=8,
        capture_multiplier=1.2, predation_multiplier=1.15,
        energy_cost_multiplier=1.0, food_amount_multiplier=1.2,
        success_threshold={'avg_steps': 2800, 'avg_food': 100, 'success_rate': 0.3}
    ),
    CurriculumStage(
        name='', stage=9,
        capture_multiplier=1.1, predation_multiplier=1.1,
        energy_cost_multiplier=1.0, food_amount_multiplier=1.1,
        success_threshold={'avg_steps': 2500, 'avg_food': 105, 'success_rate': 0.3}
    ),
    CurriculumStage(
        name='', stage=10,
        capture_multiplier=1.0, predation_multiplier=1.0,
        energy_cost_multiplier=1.0, food_amount_multiplier=1.0,
        success_threshold={'avg_steps': 3700, 'avg_food': 115, 'success_rate': 0.3}
    ),
]


# ============================================================
# 11. Observation Space Configuration 【！】
# ============================================================

@dataclass(frozen=True)
class ObservationConfig:
    '''Observation Space - v2 '''
    position_dim: int = 3
    velocity_dim: int = 3
    body_state_dim: int = 2
    heading_dim: int = 3
    pitch_dim: int = 2
    nearest_fish_dim: int = 33   # 3 × 11 (3+3+1+1+is_threat1+is_prey1+closing_speed1)
    nearest_food_dim: int = 24   # 3 × 8 (3+3+1+closing_speed1)
    environment_dim: int = 4
    obstacle_dim: int = 8
    boundary_dim: int = 3
    surface_dim: int = 2
    rest_state_dim: int = 4
    buoyancy_dim: int = 4

    @property
    def total_dim(self) -> int:
        """: 3+3+2+3+2+33+24+4+8+3+2+4+4 = 95"""
        return (self.position_dim + self.velocity_dim + self.body_state_dim +
                self.heading_dim + self.pitch_dim +
                self.nearest_fish_dim + self.nearest_food_dim + self.environment_dim +
                self.obstacle_dim + self.boundary_dim + self.surface_dim +
                self.rest_state_dim + self.buoyancy_dim)


# ============================================================
# 12.  【！ + 】
# ============================================================

@dataclass(frozen=True)
class RewardConfig:
    """ v5.2.1"""

    # ===== =====
    weight_gain_per_step_scale: float = 50.0  # 1%0.5
    weight_loss_per_step_scale: float = 30.0

    # =====  =====
    weight_gain_scale: float = 100.0   # ⬆️ 80
    weight_gain_exponent: float = 0.6  # ⬇️ 0.7
    weight_loss_penalty: float = 50.0  # ⬆️ 10
    bonus_threshold: float = 0.15
    bonus_multiplier: float = 150.0    # ⬆️ 50

    # ===== survival reward =====
    survival_reward: float = 0.001

    # =====  =====
    growth_event_reward: float = 1.5   # ⬆️ 1.0
    growth_mass_bonus: float = 2.0

    # ===== feeding reward =====
    food_reward: float = 0.5

    # =====  =====
    energy_maintain_threshold: float = 50.0
    energy_maintain_reward: float = 0.002

    # =====  =====
    collision_penalty: float = 0.02    # ⬇️ 0.05
    injury_penalty_scale: float = 0.02 # ⬇️ 0.05

    # ===== death penalty =====
    early_death_threshold: int = 1500
    early_death_penalty: float = 8.0   # ⬆️ 3
    no_growth_death_penalty: float = 15.0  # ⬆️ 8
    normal_death_penalty: float = 3.0  # ⬆️ 1

    # =====  =====
    rest_while_digesting_reward: float = 0.02  # ⬆️ 0.02
    deep_rest_bonus: float = 0.01
    rest_growth_multiplier: float = 1.2  # ⬆️ 1.2
    rest_during_danger_penalty: float = 0.10  # ⬆️ 0.10
    rest_while_hungry_penalty: float = 0.02

    # =====  =====
    optimal_rest_ratio_min: float = 0.20
    optimal_rest_ratio_max: float = 0.40
    optimal_rest_bonus: float = 2.0    # ⬆️ 2.0
    rest_growth_contribution_bonus: float = 1.0
    rest_during_danger_count_penalty: float = 0.05

    # =====  【】=====
    reward_clip_min: float = -2.0      # ⬆️ -1.0
    reward_clip_max: float = 10.0      # ⬆️ 5.0

    # ===== new=====
    predation_base_reward: float = 3.0
    predation_first_kill_bonus: float = 5.0
    predation_milestone_3: float = 5.0
    predation_milestone_5: float = 10.0
    predation_terminal_per_fish: float = 5.0

    # ===== disabled =====
    predation_reward: float = 0.0
    food_base_reward: float = 0.0
    base_survival_reward: float = 0.001
    time_bonus_scale: float = 0.0
    time_bonus_max: float = 0.0
    time_bonus_offset: int = 100
    energy_critical_threshold: float = 20.0
    energy_low_threshold: float = 60.0
    energy_low_multiplier: float = 1.0
    energy_low_base: float = 1.0
    low_energy_penalty_scale: float = 100.0
    food_high_fullness_threshold: float = 90.0
    food_medium_fullness_threshold: float = 75.0
    food_high_fullness_multiplier: float = 0.0
    food_medium_fullness_multiplier: float = 0.0
    food_waste_penalty: float = 0.0
    death_penalty_no_food: float = 8.0
    death_penalty_early: float = 3.0
    death_penalty_normal: float = 1.0
    energy_efficiency_threshold: float = 70.0
    energy_efficiency_reward: float = 0.001


# ============================================================
# 13.
# ============================================================

@dataclass(frozen=True)
class TrainingConfig:
    """"""
    total_timesteps: int = 50000000
    n_envs: int = 16
    eval_freq: int = 25000
    save_freq: int = 50000
    model_dir: str = './models/'
    log_dir: str = './logs/'


# ============================================================
# 14. PPO
# ============================================================

@dataclass(frozen=True)
class PPOConfig:
    """PPO algorithm hyperparameters."""
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 4096
    n_epochs: int = 10
    gamma: float = 0.997
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.005
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = None  # KLNone


# ============================================================
# 15.
# ============================================================

@dataclass(frozen=True)
class NetworkArchitectureConfig:
    """"""
    policy_layers: Tuple[int, ...] = (256, 256, 128)
    value_layers: Tuple[int, ...] = (256, 256, 128)
    activation_fn: str = 'Tanh'

    log_std_init: float = -1.0
    ortho_init: bool = True

    optimizer_eps: float = 1e-5
    optimizer_weight_decay: float = 0.0


# ============================================================
# 16.
# ============================================================

@dataclass(frozen=True)
class EntropyDecayConfig:
    """"""
    initial_coef: float = 0.005
    final_coef: float = 0.001
    total_steps: int = 20000000  #


# ============================================================
# 17.
# ============================================================

@dataclass(frozen=True)
class CallbacksConfig:
    """"""
    optimizer_reset_frequency: int = 500000

    dead_neuron_check_freq: int = 100000

    performance_check_freq: int = 50000

    obstacle_check_freq: int = 20000


# ============================================================
# 18. Curriculum Learning Configuration
# ============================================================

@dataclass(frozen=True)
class CurriculumConfig:
    """Curriculum Learning Configuration"""
    check_frequency: int = 50000  #
    min_episodes_for_check: int = 50  # episode
    max_history: int = 200  #


# ============================================================
# 19.
# ============================================================

@dataclass(frozen=True)
class LearningRateScheduleConfig:
    """"""
    boost_duration: int = 100000  # boost
    adaptation_threshold: float = 0.1  #
    decay_rate: float = 0.5  #
    decay_frequency: int = 10000000  #
    min_lr_ratio: float = 0.1  #

    decay_start_step: int = 30000000


# ============================================================
# 20.
# ============================================================

@dataclass
class Config:
    """ - """
    metabolism: MetabolismConfig = field(default_factory=MetabolismConfig)
    rest_state: RestStateConfig = field(default_factory=RestStateConfig)
    digestion: DigestionConfig = field(default_factory=DigestionConfig)
    growth: GrowthConfig = field(default_factory=GrowthConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    length_weight: LengthWeightConfig = field(default_factory=LengthWeightConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    feeding: FeedingConfig = field(default_factory=FeedingConfig)
    interaction: InteractionConfig = field(default_factory=InteractionConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    agent_init: AgentInitConfig = field(default_factory=AgentInitConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    network: NetworkArchitectureConfig = field(default_factory=NetworkArchitectureConfig)
    entropy_decay: EntropyDecayConfig = field(default_factory=EntropyDecayConfig)
    callbacks: CallbacksConfig = field(default_factory=CallbacksConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    lr_schedule: LearningRateScheduleConfig = field(default_factory=LearningRateScheduleConfig)
    buoyancy: BuoyancyConfig = field(default_factory=BuoyancyConfig)
    inertia: InertiaConfig = field(default_factory=InertiaConfig)
    aggressive_behavior: AggressiveBehaviorConfig = field(default_factory=AggressiveBehaviorConfig)
    surface_predator: SurfacePredatorConfig = field(default_factory=SurfacePredatorConfig)
    fleeing_behavior: FleeingBehaviorConfig = field(default_factory=FleeingBehaviorConfig)
    energy: EnergySystemConfig = field(default_factory=EnergySystemConfig)
    obstacles: ObstacleConfig = field(default_factory=ObstacleConfig)
    env_randomization: EnvironmentRandomizationConfig = field(default_factory=EnvironmentRandomizationConfig)

CONFIG = Config()


# ============================================================
# Helper Functions
# ============================================================

def get_initial_curriculum_config() -> Dict[str, Any]:
    """Curriculum configuration"""
    stage = CURRICULUM_STAGES[0]
    return {
        'stage': stage.stage,
        'name': stage.name,
        'capture_multiplier': stage.capture_multiplier,
        'predation_multiplier': stage.predation_multiplier,
        'energy_cost_multiplier': stage.energy_cost_multiplier,
        'food_amount_multiplier': stage.food_amount_multiplier
    }


def get_curriculum_stage(stage_index: int) -> Dict[str, Any]:
    """Curriculum configuration"""
    if 0 <= stage_index < len(CURRICULUM_STAGES):
        stage = CURRICULUM_STAGES[stage_index]
    else:
        stage = CURRICULUM_STAGES[-1]

    return {
        'stage': stage.stage,
        'name': stage.name,
        'capture_multiplier': stage.capture_multiplier,
        'predation_multiplier': stage.predation_multiplier,
        'energy_cost_multiplier': stage.energy_cost_multiplier,
        'food_amount_multiplier': stage.food_amount_multiplier
    }


# ============================================================
#  CONFIG
# ============================================================
#
#  CONFIG
#   - CONFIG.metabolism.smr_coefficient  BIOLOGICAL_CONSTANTS['smr_coefficient']
#   - CONFIG.environment.tank_radius  DEFAULT_ENV_CONFIG['tank_radius']
#   - CONFIG.observation.total_dim  TOTAL_OBSERVATION_DIM
#
# ============================================================

import warnings


def _deprecated_dict_access(name):
    warnings.warn(
        f"{name}  CONFIG ",
        DeprecationWarning,
        stacklevel=3
    )


TOTAL_OBSERVATION_DIM = CONFIG.observation.total_dim

# ============================================================
# ============================================================

TRAINING_CONFIG = {
    'total_timesteps': CONFIG.training.total_timesteps,
    'n_envs': CONFIG.training.n_envs,
    'eval_freq': CONFIG.training.eval_freq,
    'save_freq': CONFIG.training.save_freq,
    'model_dir': CONFIG.training.model_dir,
    'log_dir': CONFIG.training.log_dir,
}

# PPO
PPO_CONFIG = {
    'learning_rate': CONFIG.ppo.learning_rate,
    'n_steps': CONFIG.ppo.n_steps,
    'batch_size': CONFIG.ppo.batch_size,
    'n_epochs': CONFIG.ppo.n_epochs,
    'gamma': CONFIG.ppo.gamma,
    'gae_lambda': CONFIG.ppo.gae_lambda,
    'clip_range': CONFIG.ppo.clip_range,
    'ent_coef': CONFIG.ppo.ent_coef,
    'vf_coef': CONFIG.ppo.vf_coef,
    'max_grad_norm': CONFIG.ppo.max_grad_norm,
    'target_kl': CONFIG.ppo.target_kl,
}

NETWORK_ARCHITECTURE = {
    'net_arch': dict(
        pi=list(CONFIG.network.policy_layers),
        vf=list(CONFIG.network.value_layers)
    ),
    'activation_fn': CONFIG.network.activation_fn,
    'log_std_init': CONFIG.network.log_std_init,
    'ortho_init': CONFIG.network.ortho_init,
    'optimizer_kwargs': {
        'eps': CONFIG.network.optimizer_eps,
        'weight_decay': CONFIG.network.optimizer_weight_decay
    }
}

ENTROPY_DECAY_CONFIG = {
    'initial_coef': CONFIG.entropy_decay.initial_coef,
    'final_coef': CONFIG.entropy_decay.final_coef,
    'total_steps': CONFIG.entropy_decay.total_steps,
}

CALLBACKS_CONFIG = {
    'optimizer_reset_frequency': CONFIG.callbacks.optimizer_reset_frequency,
    'dead_neuron_check_freq': CONFIG.callbacks.dead_neuron_check_freq,
    'performance_check_freq': CONFIG.callbacks.performance_check_freq,
    'obstacle_check_freq': CONFIG.callbacks.obstacle_check_freq,
}

# Curriculum Learning Configuration
CURRICULUM_CONFIG = {
    'check_frequency': CONFIG.curriculum.check_frequency,
    'min_episodes_for_check': CONFIG.curriculum.min_episodes_for_check,
    'max_history': CONFIG.curriculum.max_history,
}


# CURRICULUM_STAGESdataclass
def _convert_curriculum_stages_to_dict() -> List[Dict[str, Any]]:
    """dataclass"""
    result = []
    for stage in CURRICULUM_STAGES:
        result.append({
            'name': stage.name,
            'stage': stage.stage,
            'capture_multiplier': stage.capture_multiplier,
            'predation_multiplier': stage.predation_multiplier,
            'energy_cost_multiplier': stage.energy_cost_multiplier,
            'food_amount_multiplier': stage.food_amount_multiplier,
            'success_threshold': stage.success_threshold.copy()
        })
    return result


CURRICULUM_STAGES_DICT = _convert_curriculum_stages_to_dict()

LEARNING_RATE_SCHEDULE_CONFIG = {
    'boost_duration': CONFIG.lr_schedule.boost_duration,
    'adaptation_threshold': CONFIG.lr_schedule.adaptation_threshold,
    'decay_rate': CONFIG.lr_schedule.decay_rate,
    'decay_frequency': CONFIG.lr_schedule.decay_frequency,
    'min_lr_ratio': CONFIG.lr_schedule.min_lr_ratio,
}

# ============================================================
# Rest State Configuration
# ============================================================

REST_STATE_CONFIG = {
    'base_metabolism_reduction': CONFIG.rest_state.base_metabolism_reduction,
    'deep_rest_metabolism_min': CONFIG.rest_state.deep_rest_metabolism_min,
    'metabolism_decay_rate': CONFIG.rest_state.metabolism_decay_rate,
    'deep_rest_threshold_steps': CONFIG.rest_state.deep_rest_threshold_steps,
    'rest_growth_bonus_base': CONFIG.rest_state.rest_growth_bonus_base,
    'rest_growth_bonus_max': CONFIG.rest_state.rest_growth_bonus_max,
    'growth_bonus_rate': CONFIG.rest_state.growth_bonus_rate,
    'rest_propulsion_factor': CONFIG.rest_state.rest_propulsion_factor,
    'rest_velocity_damping': CONFIG.rest_state.rest_velocity_damping,
    'rest_vision_reduction': CONFIG.rest_state.rest_vision_reduction,
    'rest_food_detection_reduction': CONFIG.rest_state.rest_food_detection_reduction,
    'active_reaction_delay': CONFIG.rest_state.active_reaction_delay,
    'rest_reaction_delay': CONFIG.rest_state.rest_reaction_delay,
    'rest_to_active_cooldown': CONFIG.rest_state.rest_to_active_cooldown,
    'active_to_rest_cooldown': CONFIG.rest_state.active_to_rest_cooldown,
    'min_rest_duration_steps': CONFIG.rest_state.min_rest_duration_steps,
    'hunger_wake_stomach_threshold': CONFIG.rest_state.hunger_wake_stomach_threshold,
    'forced_active_hunger_threshold': CONFIG.rest_state.forced_active_hunger_threshold,
    'forced_active_no_food_steps': CONFIG.rest_state.forced_active_no_food_steps,
    'rest_digestion_rate': CONFIG.rest_state.rest_digestion_rate,
    'rest_absorption_bonus': CONFIG.rest_state.rest_absorption_bonus,
    'rest_sda_reduction': CONFIG.rest_state.rest_sda_reduction,
    'proactive_wake_threat_distance': CONFIG.rest_state.proactive_wake_threat_distance,
}


def get_env_config_with_overrides(**overrides) -> Dict[str, Any]:
    """"""
    base_config = {
        'tank_radius': CONFIG.environment.tank_radius,
        'tank_depth': CONFIG.environment.tank_depth,
        'water_temp': CONFIG.environment.water_temp,
        'time_step': CONFIG.environment.time_step,
        'time_acceleration': CONFIG.environment.time_acceleration,
        'max_episode_steps': CONFIG.environment.max_episode_steps,
        'vision_range': CONFIG.perception.vision_range,
        'food_detection_range': CONFIG.perception.food_detection_range,
        'fish_density': CONFIG.physics.fish_density,
        'water_current_strength': CONFIG.environment.water_current_strength,
        'collision_damage': CONFIG.physics.collision_damage,
        'obstacle_sensing': {
            'num_rays': CONFIG.perception.num_rays,
            'max_distance': CONFIG.perception.max_obstacle_distance,
            'enable_boundary_vector': CONFIG.perception.enable_boundary_vector,
            'enable_surface_detection': CONFIG.perception.enable_surface_detection,
        },
        'feeding_system': {
            'daily_feeding_rate': CONFIG.feeding.daily_feeding_rate,
            'feedings_per_day': CONFIG.feeding.feedings_per_day,
            'settle_trigger_ratio': CONFIG.feeding.settle_trigger_ratio,
            'spread_radius': CONFIG.feeding.spread_radius,
        },
        'verbose': 0,
        'log_frequency': 100,
        'debug_energy': False,
        'debug_curriculum': False,
        'debug_growth': False,
        'debug_predation': False,
        'debug_rest_state': False,
    }
    base_config.update(overrides)
    return base_config


def get_training_config_with_overrides(**overrides) -> Dict[str, Any]:
    """"""
    base_config = TRAINING_CONFIG.copy()
    base_config.update(overrides)
    return base_config


# ============================================================
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("")
    print("=" * 60)

    print(f"\n【new】Rest State Configuration:")
    print(f"  : {CONFIG.rest_state.base_metabolism_reduction}")
    print(f"  : {CONFIG.rest_state.deep_rest_metabolism_min}")
    print(f"  : {CONFIG.rest_state.metabolism_decay_rate}")
    print(f"  : {CONFIG.rest_state.rest_growth_bonus_base} ~ {CONFIG.rest_state.rest_growth_bonus_max}")
    print(f"  : {CONFIG.rest_state.growth_bonus_rate}")
    print(f"  propulsion coefficient: {CONFIG.rest_state.rest_propulsion_factor}")
    print(f"  : {CONFIG.rest_state.rest_velocity_damping}")
    print(
        f"  Reaction delay(/): {CONFIG.rest_state.active_reaction_delay}s / {CONFIG.rest_state.rest_reaction_delay}s")

    print(f"\n【】:")
    print(f"  : {CONFIG.observation.nearest_fish_dim} (3×4)")
    print(f"  : {CONFIG.observation.nearest_food_dim} (3×3)")
    print(f"  : {CONFIG.observation.rest_state_dim} (new)")
    print(f"  total dimensions: {CONFIG.observation.total_dim} (46→50)")

    print(f"\n【】:")
    print(f"  : {CONFIG.reward.weight_gain_per_step_scale}")
    print(f"  : {CONFIG.reward.weight_loss_per_step_scale}")
    print(f"  : {CONFIG.reward.rest_while_digesting_reward}")
    print(f"  : {CONFIG.reward.deep_rest_bonus}")
    print(f"  : {CONFIG.reward.rest_during_danger_penalty}")
    print(f"  : {CONFIG.reward.optimal_rest_ratio_min}~{CONFIG.reward.optimal_rest_ratio_max}")

    print(f"\n:")
    print(f"   - SMR: {CONFIG.metabolism.smr_coefficient}")
    print(f"   - : {CONFIG.environment.tank_radius}m")
    print(f"   - vision range: {CONFIG.perception.vision_range}m")

    print(f"\n: {len(CURRICULUM_STAGES)}")

    print("\n✅ ！")
