#!/usr/bin/env python3
"""
Metabolism Subsystem for Largemouth Bass RL Simulation
======================================================

Bioenergetics model for *Micropterus salmoides* based on the Rice et al. (1983)
framework, extended with activity-state management and mass-based digestion
tracking for reinforcement learning applications.

Key features:
    1. Standard metabolic rate (SMR) computed from the Rice et al. (1983)
       oxycalorific equation with allometric mass scaling and Arrhenius
       temperature dependence.
    2. Activity state management (active/resting) with exponential transition
       dynamics and cooldown enforcement.
    3. Mass-based digestion pipeline using nutrient-specific apparent
       digestibility coefficients (ADCs) for protein, lipid, and carbohydrate.
    4. Starvation mechanics with prioritized lipid reserve depletion followed
       by protein catabolism, and body-mass-dependent mortality thresholds.
    5. Energy-state debuff system that modulates locomotor capacity (speed,
       propulsion efficiency, burst availability) as reserves decline.

Starvation logic:
    - stomach_content_mass > 0: digestion in progress, no reserve depletion.
    - stomach_content_mass = 0: fasting state, lipid/protein reserves consumed
      to meet basal metabolic demand.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CONFIG


# ═══════════ Activity State Enumeration ═══════════

class ActivityState(Enum):
    """Enumeration of fish activity states."""
    ACTIVE = "active"    # Locomotor / foraging state
    RESTING = "resting"  # Quiescent / sheltering state


# ═══════════ Data Classes ═══════════

@dataclass
class MetabolismState:
    """Mutable state container for the metabolism subsystem.

    Tracks energy reserves, digestion progress, activity state, and
    physiological condition across simulation steps.
    """
    energy: float = field(default_factory=lambda: CONFIG.agent_init.initial_energy)
    stomach_fullness: float = field(default_factory=lambda: CONFIG.agent_init.initial_stomach_fullness)
    stress_level: float = 0.0
    fatigue: float = 0.0
    is_digesting: bool = False
    digestion_buffer: float = 0.0
    energy_from_digestion: float = 0.0
    energy_lost_to_metabolism: float = 0.0
    growth_accumulation: float = 0.0

    # Body composition tracking
    initial_body_mass: float = 0.0
    lipid_reserve: float = 0.0
    protein_reserve: float = 0.0
    initial_lipid_reserve: float = 0.0
    initial_protein_reserve: float = 0.0

    # Activity state fields
    activity_state: ActivityState = ActivityState.ACTIVE  # Current activity state
    rest_duration_steps: int = 0       # Current resting duration (steps)
    total_rest_steps: int = 0          # Cumulative rest steps (for statistics)
    state_switch_cooldown: float = 0.0 # State switch cooldown timer (seconds)
    last_state_switch_step: int = 0    # Step number of last state transition

    # Rest effect coefficients (computed each step)
    current_metabolism_factor: float = 1.0  # Current metabolic rate multiplier
    current_growth_bonus: float = 1.0       # Current growth efficiency bonus

    # Reaction delay buffer (used by perception system)
    reaction_delay_buffer: float = 0.0      # Accumulated reaction delay
    pending_threat_response: bool = False    # Whether a pending threat response exists

    # Buoyancy energy tracking
    buoyancy_energy_consumed: float = 0.0   # Current-step buoyancy cost (kJ)
    total_buoyancy_energy: float = 0.0      # Cumulative buoyancy cost (kJ)

    # Mass-based digestion tracking
    stomach_content_mass: float = 0.0       # Stomach content mass (g)
    initial_meal_mass: float = 0.0          # Initial meal mass (g)
    stomach_protein_fraction: float = 0.0
    stomach_lipid_fraction: float = 0.0
    stomach_carbohydrate_fraction: float = 0.0
    stomach_adc_protein: float = 0.0
    stomach_adc_lipid: float = 0.0
    stomach_adc_carbohydrate: float = 0.0
    stomach_include_carbohydrate_energy: bool = False

    energy_debuff_factor: float = 1.0       # Energy-induced capacity debuff factor
    death_mass_loss_threshold: float = 0.275  # Per-fish starvation threshold (fraction of initial mass)


@dataclass
class MetabolismInput:
    """Input data for a single metabolism update step.

    Aggregates environmental conditions, locomotor activity, and
    requested state transitions from the agent controller.
    """
    body_mass: float
    action_magnitude: float = 0.0
    is_burst_swimming: bool = False
    water_temp: float = field(default_factory=lambda: CONFIG.environment.water_temp)
    time_step: float = field(default_factory=lambda: CONFIG.environment.time_step)
    time_acceleration: float = field(default_factory=lambda: CONFIG.environment.time_acceleration)
    velocity_magnitude: float = 0.0

    # Requested activity state transition
    requested_activity_state: Optional[ActivityState] = None
    current_step: int = 0                       # Current environment step number
    nearest_threat_distance: float = float('inf')

    # Buoyancy energy input
    buoyancy_energy_cost: float = 0.0  # Buoyancy cost from physics system (kJ)
    turn_angle_deg: float = 0.0        # Actual turning angle this frame (degrees)
    smr_individual_factor: float = 1.0


@dataclass
class MetabolismOutput:
    """Output of a single metabolism update step.

    All energy values are in absolute units (kJ).
    """
    total_metabolic_demand: float  # kJ
    base_metabolic_cost: float     # kJ
    active_metabolic_cost: float   # kJ
    stress_cost: float             # kJ
    energy_from_digestion: float = 0.0  # kJ
    growth_energy: float = 0.0          # kJ
    new_energy: float = 0.0             # % (display value)
    new_stomach_fullness: float = 0.0   # %
    new_stress_level: float = 0.0
    new_fatigue: float = 0.0
    mass_loss: float = 0.0              # g
    is_starving: bool = False

    # Activity state outputs
    activity_state: ActivityState = ActivityState.ACTIVE
    rest_duration_steps: int = 0
    metabolism_factor: float = 1.0
    growth_bonus: float = 1.0
    state_switched: bool = False  # Whether a state transition occurred this step

    # Buoyancy metabolic output
    buoyancy_metabolic_cost: float = 0.0  # Buoyancy metabolic cost this step (kJ)


# ═══════════ Metabolism System Core ═══════════

class MetabolismSystem:
    """Core metabolism system implementing the Rice et al. (1983) bioenergetics
    model with activity-state-dependent modulation.

    Manages standard metabolic rate computation, active metabolism scaling,
    digestion with nutrient-specific ADCs, and starvation reserve depletion.
    """

    def __init__(self) -> None:
        self.c = CONFIG.metabolism
        self.rc = CONFIG.rest_state       # Rest state configuration
        self.dc = CONFIG.digestion
        self.ec = CONFIG.energy
        self.debug = False
        self.debug_frequency = 100

        # Body composition parameters
        self.base_lipid_fraction = self.c.lipid_fraction
        self.lipid_scaling_exponent = self.c.lipid_scaling_exponent
        self.protein_fraction = self.c.protein_fraction

        # Energy densities (kJ/g)
        self.lipid_energy_density = self.c.lipid_energy_density
        self.protein_energy_density = self.c.protein_energy_density

        # Death threshold (deferred initialization)
        self.death_mass_loss_threshold = None

    def initialize_body_composition(self, state: MetabolismState, body_mass: float) -> None:
        """Initialize body composition with allometric lipid scaling.

        Sets initial lipid and protein reserves based on body mass, computes
        the starvation death threshold, and resets activity state.

        Args:
            state: Mutable metabolism state to initialize.
            body_mass: Current body mass in grams.
        """
        lipid_fraction = self.base_lipid_fraction * (body_mass / 10.0) ** self.lipid_scaling_exponent
        lipid_fraction = np.clip(lipid_fraction, 0.05, 0.30)

        state.initial_body_mass = body_mass
        state.lipid_reserve = body_mass * lipid_fraction
        state.protein_reserve = body_mass * self.protein_fraction
        state.initial_lipid_reserve = state.lipid_reserve
        state.initial_protein_reserve = state.protein_reserve

        self.death_mass_loss_threshold = self._calculate_death_threshold(body_mass)
        state.death_mass_loss_threshold = self.death_mass_loss_threshold  # store per-fish
        state.energy = 100.0

        # Initialize activity state
        state.activity_state = ActivityState.ACTIVE
        state.rest_duration_steps = 0
        state.current_metabolism_factor = 1.0
        state.current_growth_bonus = 1.0

        # Initialize buoyancy energy tracking
        state.buoyancy_energy_consumed = 0.0
        state.total_buoyancy_energy = 0.0

        # Initialize digestion state
        state.stomach_content_mass = 0.0
        state.initial_meal_mass = 0.0
        state.stomach_protein_fraction = 0.0
        state.stomach_lipid_fraction = 0.0
        state.stomach_carbohydrate_fraction = 0.0
        state.stomach_adc_protein = 0.0
        state.stomach_adc_lipid = 0.0
        state.stomach_adc_carbohydrate = 0.0
        state.stomach_include_carbohydrate_energy = False

        if self.debug:
            print(f"Init: mass={body_mass:.1f}g, "
                  f"lipid={state.lipid_reserve:.2f}g ({lipid_fraction * 100:.1f}%), "
                  f"death_threshold={self.death_mass_loss_threshold * 100:.1f}%")

    def _calculate_death_threshold(self, body_mass: float) -> float:
        """Compute body-mass-dependent starvation mortality threshold.

        The threshold increases with body mass following a log-linear
        relationship, with individual variability (CV ~2%).

        Threshold distribution:
            - 10g  -> 27.5%
            - 20g  -> 27.5%
            - 50g  -> 29.8%
            - 77g  -> 30.9%
            - 100g -> 31.5%
            - 200g -> 35.0%
            - 300g -> 37.0% (upper limit ~42%)

        Args:
            body_mass: Fish body mass in grams.

        Returns:
            Fractional mass loss threshold at which starvation death occurs.
        """
        if body_mass < 20:
            threshold = 0.275
        elif body_mass < 100:
            # 20-100g: slow logarithmic increase
            threshold = 0.275 + 0.025 * np.log(body_mass / 20)
        else:
            # >100g: continued increase with upper bound
            threshold = 0.315 + 0.05 * np.log(body_mass / 100)
            threshold = min(threshold, 0.42)

        # Add individual variability (CV ~2%)
        std = 0.02
        threshold = np.random.normal(threshold, std)

        # Clamp to valid range
        threshold = np.clip(threshold, 0.22, 0.45)

        return threshold

    # ═══════════ Activity State Management ═══════════

    def _update_activity_state(self, state: MetabolismState,
                               input_data: MetabolismInput) -> bool:
        """Update fish activity state based on requests and constraints.

        Handles emergency wake-up from threats, cooldown enforcement,
        minimum rest duration requirements, and state transition logic.

        Args:
            state: Current metabolism state (modified in-place).
            input_data: Input data containing requested state and threat info.

        Returns:
            True if a state transition occurred this step.
        """
        rc = self.rc
        dt = input_data.time_step
        state_switched = False

        # Update cooldown timer
        if state.state_switch_cooldown > 0:
            state.state_switch_cooldown -= dt
            state.state_switch_cooldown = max(0, state.state_switch_cooldown)

        emergency_threat_dist = getattr(rc, 'emergency_wake_threat_distance', 0.22)
        if (state.activity_state == ActivityState.RESTING and
                input_data.nearest_threat_distance < emergency_threat_dist):
            state.activity_state = ActivityState.ACTIVE
            state.state_switch_cooldown = getattr(rc, 'emergency_wake_cooldown', rc.rest_to_active_cooldown)
            state.rest_duration_steps = 0
            state.last_state_switch_step = input_data.current_step
            state_switched = True

            if self.debug:
                print(f"[rest] emergency wake-up at threat distance {input_data.nearest_threat_distance:.3f}m")

        # Check if a state switch was requested
        if input_data.requested_activity_state is not None:
            requested = input_data.requested_activity_state
            current = state.activity_state

            # Check whether switching is permitted
            can_switch = state.state_switch_cooldown <= 0

            threat_forced_wake = (
                current == ActivityState.RESTING and
                requested == ActivityState.ACTIVE and
                input_data.nearest_threat_distance < getattr(
                    rc, 'proactive_wake_threat_distance',
                    getattr(rc, 'emergency_wake_threat_distance', 0.22)
                )
            )
            hunger_forced_wake = (
                current == ActivityState.RESTING and
                requested == ActivityState.ACTIVE and
                state.stomach_fullness < getattr(rc, 'hunger_wake_stomach_threshold', 12.0)
            )
            if threat_forced_wake or hunger_forced_wake:
                can_switch = True

            # Additional check: resting state requires minimum duration
            if (current == ActivityState.RESTING and
                    requested == ActivityState.ACTIVE and
                    not threat_forced_wake and
                    not hunger_forced_wake):
                if state.rest_duration_steps < rc.min_rest_duration_steps:
                    can_switch = False

            if can_switch and requested != current:
                # Execute state transition
                old_state = state.activity_state
                state.activity_state = requested
                state.last_state_switch_step = input_data.current_step
                state_switched = True

                # Set cooldown duration
                if requested == ActivityState.ACTIVE:
                    # Transition from resting to active
                    state.state_switch_cooldown = rc.rest_to_active_cooldown
                    state.rest_duration_steps = 0  # Reset rest counter
                else:
                    # Transition from active to resting
                    state.state_switch_cooldown = rc.active_to_rest_cooldown

                if self.debug:
                    print(f"State switch: {old_state.value} -> {requested.value}")

        # Update rest duration counter
        if state.activity_state == ActivityState.RESTING:
            state.rest_duration_steps += 1
            state.total_rest_steps += 1

        # Compute current metabolism factor and growth bonus
        self._calculate_rest_effects(state)

        return state_switched

    def _calculate_rest_effects(self, state: MetabolismState) -> None:
        """Compute rest-state effect coefficients using exponential models.

        During resting, metabolic rate decreases exponentially toward a
        minimum, while growth efficiency increases toward a maximum.
        Both transitions are governed by time constants from configuration.

        Args:
            state: Current metabolism state (modified in-place).
        """
        rc = self.rc

        if state.activity_state == ActivityState.ACTIVE:
            # Active state: restore normal values
            state.current_metabolism_factor = 1.0
            state.current_growth_bonus = 1.0
        else:
            # Resting state: compute progressive effects
            steps = state.rest_duration_steps

            # Metabolism factor: exponential decay
            # factor = base + (min - base) * (1 - exp(-rate * steps))
            # Metabolic rate decreases with increasing rest duration
            base = rc.base_metabolism_reduction
            min_val = rc.deep_rest_metabolism_min
            decay = 1 - np.exp(-rc.metabolism_decay_rate * steps)
            state.current_metabolism_factor = base + (min_val - base) * decay

            # Growth bonus: exponential growth
            # bonus = base + (max - base) * (1 - exp(-rate * steps))
            # Growth efficiency increases with increasing rest duration
            base_bonus = rc.rest_growth_bonus_base
            max_bonus = rc.rest_growth_bonus_max
            growth_increase = 1 - np.exp(-rc.growth_bonus_rate * steps)
            state.current_growth_bonus = base_bonus + (max_bonus - base_bonus) * growth_increase

            if self.debug and steps % 20 == 0:
                print(
                    f"[rest] step={steps}, metabolism_factor={state.current_metabolism_factor:.3f}, "
                    f"growth_bonus={state.current_growth_bonus:.2f}"
                )

    def update(self,
               state: MetabolismState,
               input_data: MetabolismInput,
               curriculum_multiplier: float = 1.0,
               growth_state=None) -> MetabolismOutput:
        """Execute one metabolism update step.

        Computes SMR, active metabolism, stress costs, digestion/starvation,
        and updates all physiological state variables.

        Args:
            state: Mutable metabolism state.
            input_data: Environmental and locomotor inputs for this step.
            curriculum_multiplier: Scaling factor for curriculum learning.
            growth_state: Growth system state (provides body_mass for energy calc).

        Returns:
            MetabolismOutput with all computed costs and updated values.
        """

        if state.initial_body_mass == 0 and growth_state is not None:
            self.initialize_body_composition(state, growth_state.body_mass)

        dt = input_data.time_step
        time_acc = input_data.time_acceleration

        # Step 0: Update activity state
        state_switched = self._update_activity_state(state, input_data)

        # Step 1: Compute standard metabolic rate (SMR)
        base_metabolic_cost = self._calculate_smr(
            body_mass=input_data.body_mass,
            time_step=dt,
            time_acceleration=time_acc,
            water_temp=input_data.water_temp
        )

        base_metabolic_cost *= input_data.smr_individual_factor  # Individual metabolic variation

        # Apply rest-state metabolism factor
        base_metabolic_cost *= state.current_metabolism_factor

        # Step 2: Compute active metabolism
        # During rest, active metabolism is near zero
        if state.activity_state == ActivityState.RESTING:
            # Resting: minimal active metabolism
            active_metabolic_cost = base_metabolic_cost * 0.05
        else:
            active_metabolic_cost = self._calculate_active_metabolism(
                base_cost=base_metabolic_cost,
                action_magnitude=input_data.action_magnitude,
                is_burst=input_data.is_burst_swimming,
                turn_angle_deg=input_data.turn_angle_deg
            )

        # Step 3: Compute stress metabolism (kJ)
        # Stress cost is reduced during rest
        stress_multiplier = 0.3 if state.activity_state == ActivityState.RESTING else 1.0
        stress_cost = state.stress_level * base_metabolic_cost * 0.5 * stress_multiplier

        # Step 4: Temperature effect
        temp_factor = self._calculate_temperature_factor(input_data.water_temp)

        # Step 5: Total metabolic demand (kJ)
        buoyancy_cost = input_data.buoyancy_energy_cost

        total_metabolic_demand = (
                base_metabolic_cost * curriculum_multiplier
                + (active_metabolic_cost + stress_cost)
                * temp_factor * curriculum_multiplier
        ) + buoyancy_cost

        # Update buoyancy energy tracking
        state.buoyancy_energy_consumed = buoyancy_cost
        state.total_buoyancy_energy += buoyancy_cost

        # Step 6: Digestion / starvation processing
        # Activity state is passed implicitly via state object
        digestion_result = self._process_digestion_v4(
            state=state,
            input_data=input_data,
            total_metabolic_demand=total_metabolic_demand,
            temp_factor=temp_factor,
            growth_state=growth_state
        )

        # Step 7: Update physiological state (stress, fatigue)
        self._update_physiological_state(state, input_data, total_metabolic_demand)

        # Step 8: Clamp energy to valid range
        state.energy = np.clip(state.energy, 0.0, 100.0)

        return MetabolismOutput(
            total_metabolic_demand=total_metabolic_demand,
            base_metabolic_cost=base_metabolic_cost,
            active_metabolic_cost=active_metabolic_cost,
            stress_cost=stress_cost,
            energy_from_digestion=digestion_result['available_energy'],
            growth_energy=digestion_result['growth_energy'],
            new_energy=state.energy,
            new_stomach_fullness=state.stomach_fullness,
            new_stress_level=state.stress_level,
            new_fatigue=state.fatigue,
            mass_loss=digestion_result['mass_loss'],
            is_starving=digestion_result['is_starving'],
            # Activity state outputs
            activity_state=state.activity_state,
            rest_duration_steps=state.rest_duration_steps,
            metabolism_factor=state.current_metabolism_factor,
            growth_bonus=state.current_growth_bonus,
            state_switched=state_switched,
            buoyancy_metabolic_cost=buoyancy_cost
        )

    # ═══════════ Metabolic Rate Calculations ═══════════

    def _calculate_smr(self, body_mass: float, time_step: float,
                       time_acceleration: float, water_temp: float = 25.0) -> float:
        """Compute standard metabolic rate using the Rice et al. (1983) equation.

        Formula:
            RA (kJ/g/d) = d_O2 * r * a2 * W^b2 * exp(m*T + g*S)

        Temperature is embedded in the exponential term; no external Q10
        correction is applied.

        Args:
            body_mass: Fish body mass in grams.
            time_step: Simulation time step in seconds.
            time_acceleration: Time acceleration factor.
            water_temp: Water temperature in degrees Celsius.

        Returns:
            SMR energy cost for this step in kJ.
        """
        import math
        c = self.c
        # mg O2 per gram per hour (includes temperature and swim speed correction)
        smr_per_g = (c.smr_coefficient
                     * (body_mass ** c.smr_exponent)
                     * math.exp(c.smr_temp_coeff * water_temp
                                + c.smr_swim_coeff * c.smr_swim_speed))
        # Whole-fish O2 consumption (mg O2/h) -> kJ/h -> kJ/step
        smr_mg_o2_per_hour = smr_per_g * body_mass
        smr_kj_per_hour = smr_mg_o2_per_hour * c.oxycalorific_coeff
        smr_per_step = smr_kj_per_hour * (time_step / 3600) * time_acceleration
        return smr_per_step

    def _calculate_active_metabolism(self, base_cost: float, action_magnitude: float,
                                     is_burst: bool, turn_angle_deg: float = 0.0) -> float:
        """Compute active metabolic cost above SMR.

        Includes swimming propulsion cost (power-law scaling with speed)
        and turning cost (proportional to turn angle in radians).

        Biological basis: fish swimming oxygen consumption scales approximately
        exponentially with speed (Brett, 1964). Turning requires trunk muscle
        contraction; large-angle C-starts approach burst swimming costs.

        Args:
            base_cost: Basal metabolic cost this step (kJ).
            action_magnitude: Normalized locomotor action intensity [0, 1].
            is_burst: Whether burst swimming mode is active.
            turn_angle_deg: Actual turning angle this frame (degrees).

        Returns:
            Additional active metabolic cost in kJ.
        """
        import math
        c = self.c
        # Propulsion metabolism (swimming)
        swim_cost = 0.0
        if action_magnitude >= c.action_threshold:
            efficiency = c.swimming_efficiency
            multiplier = c.active_metabolism_burst if is_burst else c.active_metabolism_cruise
            # Power-law exponent 1.5: high speeds incur disproportionate cost,
            # incentivizing the model to learn energy-efficient cruising speeds.
            # action=0.5 -> cost ~0.18*max; action=1.0 -> cost=1.0*max
            swim_cost = base_cost * multiplier * (action_magnitude ** 1.5) / efficiency
        # Turning metabolism (trunk muscle work)
        # turn_cost = base_cost * turn_multiplier * turn_angle_rad
        # turn_multiplier ~2.0: each radian of turning costs ~2x basal rate
        turn_cost = 0.0
        if turn_angle_deg > 1.0:
            turn_angle_rad = math.radians(turn_angle_deg)
            turn_multiplier = getattr(c, 'turn_metabolism_multiplier', 2.0)
            turn_cost = base_cost * turn_multiplier * turn_angle_rad
        return swim_cost + turn_cost

    def _calculate_temperature_factor(self, water_temp: float) -> float:
        """Compute Q10 temperature correction factor.

        Args:
            water_temp: Current water temperature in degrees Celsius.

        Returns:
            Multiplicative temperature factor relative to optimal temperature.
        """
        c = self.c
        return c.q10 ** ((water_temp - c.optimal_temp) / 10.0)

    # ═══════════ Digestion Pipeline ═══════════

    def _calculate_evacuation_rate(self,
                                   body_mass: float,
                                   stomach_content_mass: float,
                                   water_temp: float,
                                   is_resting: bool = False) -> float:
        """Compute gastric evacuation rate using a power-law temperature model.

        Based on literature models for fish gastric evacuation with allometric
        body mass dependence and sigmoid transition between juvenile and adult
        parameterizations.

        Args:
            body_mass: Fish body mass in grams.
            stomach_content_mass: Current stomach content in grams.
            water_temp: Water temperature in degrees Celsius.
            is_resting: Whether the fish is in resting state.

        Returns:
            Evacuation rate in g/hour.
        """
        if stomach_content_mass <= 0:
            return 0.0

        dc = self.dc

        # Temperature boundary check
        temp_clamped = np.clip(
            water_temp,
            dc.min_temp_for_digestion,
            dc.max_temp_for_digestion
        )

        # Power-law temperature model
        ln_T = np.log(temp_clamped)
        alpha_small = np.exp(dc.alpha_a_small * ln_T + dc.alpha_b_small)
        alpha_large = np.exp(dc.alpha_a_large * ln_T + dc.alpha_b_large)

        # Body-mass-dependent sigmoid transition
        weight = 1.0 / (1.0 + np.exp(
            -(body_mass - dc.mass_transition_center) / dc.mass_transition_width
        ))
        alpha = alpha_small * (1 - weight) + alpha_large * weight

        # Rest-state adjustment
        if is_resting:
            alpha *= self.rc.rest_digestion_rate

        return alpha * stomach_content_mass

    def _calculate_absorption_efficiency(self,
                                         stomach_content_mass: float,
                                         body_mass: float,
                                         is_resting: bool = False) -> float:
        """Compute nutrient absorption efficiency.

        Efficiency decreases with relative meal size and increases during rest.

        Args:
            stomach_content_mass: Current stomach content in grams.
            body_mass: Fish body mass in grams.
            is_resting: Whether the fish is in resting state.

        Returns:
            Absorption efficiency coefficient [0, 1].
        """
        dc = self.dc

        relative_meal_pct = (stomach_content_mass / body_mass) * 100 if body_mass > 0 else 0
        efficiency = dc.base_absorption_efficiency + dc.meal_size_absorption_effect * relative_meal_pct

        if is_resting:
            efficiency *= self.rc.rest_absorption_bonus

        return np.clip(efficiency, dc.min_absorption_efficiency, dc.max_absorption_efficiency)

    @staticmethod
    def _to_fraction(value: float) -> float:
        """Allow both fraction [0,1] and percent [0,100] style inputs."""
        value = float(value)
        if value > 1.0:
            value /= 100.0
        return max(0.0, value)

    def _build_default_diet_profile(self) -> Dict[str, float]:
        """Default pellet profile from refined configuration."""
        fc = CONFIG.feeding
        return {
            'protein_fraction': getattr(fc, 'diet_protein_fraction',
                                        getattr(self.c, 'feed_protein_fraction', 0.50)),
            'lipid_fraction': getattr(fc, 'diet_lipid_fraction',
                                      getattr(self.c, 'feed_lipid_fraction', 0.09)),
            'carbohydrate_fraction': getattr(fc, 'diet_carbohydrate_fraction',
                                             getattr(self.c, 'feed_carbohydrate_fraction', 0.0)),
            'adc_protein': getattr(fc, 'diet_adc_protein',
                                   getattr(self.c, 'adc_protein', 0.90)),
            'adc_lipid': getattr(fc, 'diet_adc_lipid',
                                 getattr(self.c, 'adc_lipid', 0.90)),
            'adc_carbohydrate': getattr(fc, 'diet_adc_carbohydrate',
                                        getattr(self.c, 'adc_carbohydrate', 0.30)),
            'include_carbohydrate_energy': getattr(
                fc,
                'diet_include_carbohydrate_energy',
                getattr(self.c, 'include_carbohydrate_energy', False)
            ),
        }

    def _normalize_diet_profile(self, profile: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """Normalize and clamp composition/ADC values.

        Args:
            profile: Optional dict with nutrient fractions and ADC values.

        Returns:
            Normalized diet profile with fractions summing to <= 1.0.
        """
        raw = self._build_default_diet_profile()
        if profile:
            raw.update(profile)

        protein = self._to_fraction(raw.get('protein_fraction', 0.0))
        lipid = self._to_fraction(raw.get('lipid_fraction', 0.0))
        carbohydrate = self._to_fraction(raw.get('carbohydrate_fraction', 0.0))

        total = protein + lipid + carbohydrate
        if total > 1.0 and total > 0:
            protein /= total
            lipid /= total
            carbohydrate /= total

        return {
            'protein_fraction': protein,
            'lipid_fraction': lipid,
            'carbohydrate_fraction': carbohydrate,
            'adc_protein': float(np.clip(self._to_fraction(raw.get('adc_protein', 0.90)), 0.0, 1.0)),
            'adc_lipid': float(np.clip(self._to_fraction(raw.get('adc_lipid', 0.90)), 0.0, 1.0)),
            'adc_carbohydrate': float(np.clip(self._to_fraction(raw.get('adc_carbohydrate', 0.30)), 0.0, 1.0)),
            'include_carbohydrate_energy': bool(raw.get('include_carbohydrate_energy', False)),
        }

    def _get_active_diet_profile(self, state: MetabolismState) -> Dict[str, float]:
        """Use mixed stomach profile when available, otherwise default diet.

        Args:
            state: Current metabolism state with stomach composition.

        Returns:
            Normalized diet profile dictionary.
        """
        if state.stomach_content_mass > 0 and (
            state.stomach_protein_fraction > 0
            or state.stomach_lipid_fraction > 0
            or state.stomach_carbohydrate_fraction > 0
        ):
            return self._normalize_diet_profile({
                'protein_fraction': state.stomach_protein_fraction,
                'lipid_fraction': state.stomach_lipid_fraction,
                'carbohydrate_fraction': state.stomach_carbohydrate_fraction,
                'adc_protein': state.stomach_adc_protein,
                'adc_lipid': state.stomach_adc_lipid,
                'adc_carbohydrate': state.stomach_adc_carbohydrate,
                'include_carbohydrate_energy': state.stomach_include_carbohydrate_energy,
            })
        return self._normalize_diet_profile(None)

    def _mix_stomach_profile(self, state: MetabolismState, food_mass: float,
                             profile: Dict[str, Any]) -> None:
        """Blend incoming food profile into the current stomach mixture.

        Uses mass-weighted averaging to combine existing stomach contents
        with newly ingested food.

        Args:
            state: Current metabolism state (modified in-place).
            food_mass: Mass of newly ingested food in grams.
            profile: Nutrient profile of the incoming food.
        """
        normalized = self._normalize_diet_profile(profile)
        existing_mass = max(0.0, state.stomach_content_mass - food_mass)
        total_mass = existing_mass + food_mass
        if total_mass <= 0:
            return

        def _blend(old_value: float, new_value: float) -> float:
            return ((old_value * existing_mass) + (new_value * food_mass)) / total_mass

        state.stomach_protein_fraction = _blend(state.stomach_protein_fraction, normalized['protein_fraction'])
        state.stomach_lipid_fraction = _blend(state.stomach_lipid_fraction, normalized['lipid_fraction'])
        state.stomach_carbohydrate_fraction = _blend(
            state.stomach_carbohydrate_fraction, normalized['carbohydrate_fraction']
        )
        state.stomach_adc_protein = _blend(state.stomach_adc_protein, normalized['adc_protein'])
        state.stomach_adc_lipid = _blend(state.stomach_adc_lipid, normalized['adc_lipid'])
        state.stomach_adc_carbohydrate = _blend(
            state.stomach_adc_carbohydrate, normalized['adc_carbohydrate']
        )
        state.stomach_include_carbohydrate_energy = (
            state.stomach_include_carbohydrate_energy or normalized['include_carbohydrate_energy']
        )

    def _calculate_digestible_energy_density(self, profile: Dict[str, float]) -> float:
        """Compute digestible energy density from composition and ADC values.

        Args:
            profile: Normalized diet profile with fractions and ADC values.

        Returns:
            Digestible energy density in kJ/g.
        """
        protein_kj = profile['protein_fraction'] * profile['adc_protein'] * self.c.protein_energy_density
        lipid_kj = profile['lipid_fraction'] * profile['adc_lipid'] * self.c.lipid_energy_density

        carb_kj = 0.0
        if profile['include_carbohydrate_energy']:
            carb_kj = (
                profile['carbohydrate_fraction']
                * profile['adc_carbohydrate']
                * self.c.carbohydrate_energy_density
            )

        return max(0.0, protein_kj + lipid_kj + carb_kj)

    def _get_dynamic_growth_allocation(self, body_mass: float) -> float:
        """Compute body-mass-dependent growth allocation with juvenile boost.

        Allocation follows an allometric power law with enhanced allocation
        for juvenile fish below a configurable mass threshold.

        Args:
            body_mass: Fish body mass in grams.

        Returns:
            Growth allocation fraction [growth_allocation_min, growth_allocation_max].
        """
        c = self.c
        safe_mass = max(float(body_mass), 1.0)

        allocation = c.growth_allocation_coefficient * (safe_mass ** c.growth_allocation_exponent)
        allocation = float(np.clip(allocation, c.growth_allocation_min, c.growth_allocation_max))

        threshold = max(1.0, float(getattr(c, 'juvenile_growth_boost_threshold', 0.0)))
        boost_max = max(1.0, float(getattr(c, 'juvenile_growth_boost_max', 1.0)))
        if safe_mass < threshold and boost_max > 1.0:
            juvenile_ratio = 1.0 - (safe_mass / threshold)
            boost = 1.0 + (boost_max - 1.0) * juvenile_ratio
            allocation *= boost

        return float(np.clip(allocation, c.growth_allocation_min, c.growth_allocation_max))

    # ═══════════ Energy State Calculation ═══════════

    def _calculate_energy_from_body_state(self, state: MetabolismState,
                                          growth_state) -> float:
        """Compute energy percentage from body mass change relative to initial.

        Maps mass gain to energy via saturating function, and mass loss
        linearly to zero at the starvation death threshold.

        Args:
            state: Current metabolism state.
            growth_state: Growth system state providing current body mass.

        Returns:
            Energy level as percentage [0, 100].
        """
        ec = self.ec

        if state.initial_body_mass <= 0 or growth_state is None:
            return state.energy

        current_mass = growth_state.body_mass
        initial_mass = state.initial_body_mass

        # Compute mass change fraction
        mass_change_fraction = (current_mass - initial_mass) / initial_mass

        if mass_change_fraction >= 0:
            # Mass gain or stable: healthy energy state
            # energy = baseline + gain_max * (1 - 1/(1 + mass_gain * sensitivity))
            gain = ec.energy_gain_max * (1 - 1 / (1 + mass_change_fraction * ec.energy_gain_sensitivity))
            energy = ec.baseline_energy + gain
        else:
            # Mass loss: linear mapping to death threshold
            mass_loss_fraction = -mass_change_fraction

            if self.death_mass_loss_threshold > 0:
                # From baseline down to zero at death threshold
                # mass_loss = 0 -> energy = baseline
                # mass_loss = death_threshold -> energy = 0
                # Use per-fish threshold from state (fixes shared-system bug in multi-fish sim)
                threshold = state.death_mass_loss_threshold if state.death_mass_loss_threshold > 0 else self.death_mass_loss_threshold
                energy = ec.baseline_energy * (1 - mass_loss_fraction / threshold)
            else:
                energy = ec.baseline_energy

        return np.clip(energy, 0.0, 100.0)

    def _calculate_energy_debuff(self, energy: float) -> dict:
        """Compute locomotor capacity debuff coefficients from energy state.

        Returns modifiers for the physics and perception systems:
            - speed_factor: Maximum speed multiplier.
            - reaction_factor: Reaction time multiplier (>1 means slower).
            - propulsion_factor: Propulsion efficiency multiplier.
            - burst_available: Whether burst swimming is permitted.

        Args:
            energy: Current energy level as percentage [0, 100].

        Returns:
            Dictionary with debuff coefficients.
        """
        ec = self.ec

        if energy >= ec.energy_healthy_threshold:
            # Healthy state
            return {
                'speed_factor': ec.healthy_speed_factor,
                'reaction_factor': ec.healthy_reaction_factor,
                'propulsion_factor': ec.healthy_propulsion_factor,
                'burst_available': True
            }

        elif energy >= ec.energy_mild_fatigue_threshold:
            # Mild fatigue
            t = (ec.energy_healthy_threshold - energy) / (
                        ec.energy_healthy_threshold - ec.energy_mild_fatigue_threshold)
            return {
                'speed_factor': ec.healthy_speed_factor - (ec.healthy_speed_factor - ec.mild_speed_min) * t,
                'reaction_factor': ec.healthy_reaction_factor + (ec.mild_reaction_max - ec.healthy_reaction_factor) * t,
                'propulsion_factor': ec.healthy_propulsion_factor - (
                            ec.healthy_propulsion_factor - ec.mild_propulsion_min) * t,
                'burst_available': True
            }

        elif energy >= ec.energy_moderate_fatigue_threshold:
            # Moderate fatigue
            t = (ec.energy_mild_fatigue_threshold - energy) / (
                        ec.energy_mild_fatigue_threshold - ec.energy_moderate_fatigue_threshold)
            return {
                'speed_factor': ec.mild_speed_min - (ec.mild_speed_min - ec.moderate_speed_min) * t,
                'reaction_factor': ec.mild_reaction_max + (ec.moderate_reaction_max - ec.mild_reaction_max) * t,
                'propulsion_factor': ec.mild_propulsion_min - (ec.mild_propulsion_min - ec.moderate_propulsion_min) * t,
                'burst_available': energy > ec.burst_disable_threshold
            }

        else:
            # Severe fatigue
            t = energy / ec.energy_moderate_fatigue_threshold  # 1 -> 0
            return {
                'speed_factor': ec.severe_speed_min + (ec.moderate_speed_min - ec.severe_speed_min) * t,
                'reaction_factor': ec.severe_reaction_max - (ec.severe_reaction_max - ec.moderate_reaction_max) * t,
                'propulsion_factor': ec.severe_propulsion_min + (
                            ec.moderate_propulsion_min - ec.severe_propulsion_min) * t,
                'burst_available': False
            }

    def _process_digestion_v4(self, state: MetabolismState, input_data: MetabolismInput,
                              total_metabolic_demand: float, temp_factor: float,
                              growth_state=None) -> Dict[str, float]:
        """Process digestion or starvation for one step.

        Two operating modes:
            1. Digestion (stomach_content_mass > 0): Evacuate stomach, compute
               digestible energy via ADC, allocate to growth vs. maintenance.
               Surplus energy replenishes lipid reserves; short-term deficits
               are ignored (handled by the energy percentage system).
            2. Starvation (stomach empty): Deplete lipid reserves first, then
               protein reserves. Update body mass accordingly.

        Args:
            state: Mutable metabolism state.
            input_data: Input data for this step.
            total_metabolic_demand: Total metabolic demand this step (kJ).
            temp_factor: Temperature correction factor.
            growth_state: Growth system state (for mass tracking).

        Returns:
            Dictionary with keys: available_energy, growth_energy,
            digested_mass, mass_loss, is_starving.
        """
        c = self.c
        ec = self.ec

        result = {
            'available_energy': 0.0,
            'growth_energy': 0.0,
            'digested_mass': 0.0,
            'mass_loss': 0.0,
            'is_starving': False
        }

        is_resting = state.activity_state == ActivityState.RESTING

        # ===== Case 1: Stomach contains food (digestion active) =====
        if state.stomach_content_mass > 0.001:
            state.is_digesting = True

            # 1. Compute gastric evacuation rate
            evacuation_rate = self._calculate_evacuation_rate(
                body_mass=input_data.body_mass,
                stomach_content_mass=state.stomach_content_mass,
                water_temp=input_data.water_temp,
                is_resting=is_resting
            )

            # 2. Compute mass digested this step
            actual_time_hours = (input_data.time_step * input_data.time_acceleration) / 3600.0
            digested_mass = evacuation_rate * actual_time_hours
            digested_mass = min(digested_mass, state.stomach_content_mass)
            diet_profile = self._get_active_diet_profile(state)

            # 3. Update stomach content mass
            state.stomach_content_mass -= digested_mass
            result['digested_mass'] = digested_mass

            # 4. Synchronize stomach fullness percentage
            stomach_capacity = input_data.body_mass * CONFIG.feeding.stomach_capacity_ratio
            state.stomach_fullness = (
                                             state.stomach_content_mass / stomach_capacity) * 100 if stomach_capacity > 0 else 0
            if state.stomach_content_mass <= 0.001:
                state.stomach_protein_fraction = 0.0
                state.stomach_lipid_fraction = 0.0
                state.stomach_carbohydrate_fraction = 0.0
                state.stomach_adc_protein = 0.0
                state.stomach_adc_lipid = 0.0
                state.stomach_adc_carbohydrate = 0.0
                state.stomach_include_carbohydrate_energy = False

            # 5. Compute digestible energy
            digestible_energy_density = self._calculate_digestible_energy_density(diet_profile)
            digestible_energy_kj = digested_mass * digestible_energy_density

            # 6. SDA + non-faecal excretion losses
            sda_factor = self.rc.rest_sda_reduction if is_resting else 1.0
            sda_cost_kj = digestible_energy_kj * c.sda_coefficient * sda_factor
            excretion_cost_kj = digestible_energy_kj * c.excretion_coefficient
            net_energy_kj = max(0.0, digestible_energy_kj - sda_cost_kj - excretion_cost_kj)
            state.energy_lost_to_metabolism += (sda_cost_kj + excretion_cost_kj)

            # 7. Energy allocation (growth vs. maintenance)
            base_allocation = self._get_dynamic_growth_allocation(input_data.body_mass)
            growth_allocation = base_allocation * state.current_growth_bonus
            growth_allocation = min(growth_allocation, 0.95)

            growth_energy_kj = net_energy_kj * growth_allocation
            available_energy_kj = net_energy_kj * (1 - growth_allocation)

            result['available_energy'] = available_energy_kj
            result['growth_energy'] = growth_energy_kj

            # Energy balance during digestion:
            # Do NOT deplete reserves on short-term step deficits.
            # Surplus replenishes lipid; deficits are transient and handled
            # by the energy-percentage tracking system.
            if available_energy_kj >= total_metabolic_demand:
                # Surplus: add to lipid reserves
                surplus_kj = available_energy_kj - total_metabolic_demand
                lipid_gain = surplus_kj / self.lipid_energy_density
                state.lipid_reserve += lipid_gain

                if ec.debug_energy_changes and self.debug:
                    print(f"  Digestion surplus: +{surplus_kj:.2f}kJ, lipid+{lipid_gain:.4f}g")
            else:
                # Deficit during digestion: ignore (transient timing issue)
                # Long-term energy state is tracked by the body-mass energy system
                if ec.debug_energy_changes and self.debug:
                    deficit = total_metabolic_demand - available_energy_kj
                    print(f"  Digestion deficit (ignored): -{deficit:.2f}kJ")
                # Do not modify result['mass_loss']; keep at 0
                pass

            state.energy_from_digestion += available_energy_kj
            state.growth_accumulation += growth_energy_kj

            # Update energy based on body mass change (unaffected by transient deficits)
            if growth_state is not None:
                state.energy = self._calculate_energy_from_body_state(state, growth_state)

            return result

        # ===== Case 2: Stomach empty (starvation/fasting state) =====
        # Reserve depletion logic applies here
        state.is_digesting = False
        state.stomach_fullness = 0.0
        state.stomach_protein_fraction = 0.0
        state.stomach_lipid_fraction = 0.0
        state.stomach_carbohydrate_fraction = 0.0
        state.stomach_adc_protein = 0.0
        state.stomach_adc_lipid = 0.0
        state.stomach_adc_carbohydrate = 0.0
        state.stomach_include_carbohydrate_energy = False
        result['is_starving'] = True

        if growth_state is not None:
            # Starvation metabolic reduction
            base_reduction = 0.70
            if is_resting:
                starvation_metabolic_reduction = base_reduction * state.current_metabolism_factor
            else:
                starvation_metabolic_reduction = base_reduction

            metabolic_demand_kj = total_metabolic_demand * starvation_metabolic_reduction

            # Deplete reserves: lipid first, then protein
            lipid_used = 0.0
            protein_used = 0.0
            available_lipid_energy = state.lipid_reserve * self.lipid_energy_density

            if available_lipid_energy >= metabolic_demand_kj:
                lipid_used = metabolic_demand_kj / self.lipid_energy_density
                state.lipid_reserve -= lipid_used
            else:
                lipid_used = state.lipid_reserve
                state.lipid_reserve = 0.0
                remaining_demand = metabolic_demand_kj - available_lipid_energy
                protein_used = remaining_demand / (
                    self.protein_energy_density * c.starvation_protein_efficiency
                )
                state.protein_reserve = max(0, state.protein_reserve - protein_used)

            # Update body mass (tissue loss + associated water loss)
            tissue_loss = lipid_used + protein_used
            water_loss = tissue_loss * 0.5
            total_mass_loss = tissue_loss + water_loss

            growth_state.body_mass = max(1.0, growth_state.body_mass - total_mass_loss)
            result['mass_loss'] = total_mass_loss

            # Update energy from body state
            state.energy = self._calculate_energy_from_body_state(state, growth_state)

        return result

    # ═══════════ Physiological State Updates ═══════════

    def get_energy_debuff(self, state: MetabolismState) -> dict:
        """Get current energy-state locomotor debuff coefficients.

        Provides modifiers for the physics and perception systems based
        on the fish's current energy level.

        Args:
            state: Current metabolism state.

        Returns:
            Dictionary with speed_factor, reaction_factor, propulsion_factor,
            and burst_available.
        """
        return self._calculate_energy_debuff(state.energy)

    def _update_physiological_state(self, state: MetabolismState, input_data: MetabolismInput,
                                    total_metabolic_demand: float) -> None:
        """Update stress and fatigue based on energy state and locomotion.

        Stress increases at low energy levels and decreases during rest.
        Fatigue accumulates during burst swimming and recovers during rest
        at an accelerated rate (up to 3x normal).

        Args:
            state: Mutable metabolism state.
            input_data: Input data for this step.
            total_metabolic_demand: Total metabolic demand (unused, reserved).
        """
        # Stress update based on energy level
        if state.energy < 20:
            state.stress_level = min(1.0, state.stress_level + 0.01)
        elif state.energy < 40:
            state.stress_level = min(1.0, state.stress_level + 0.002)
        else:
            state.stress_level = max(0, state.stress_level - 0.005)

        # Resting accelerates stress recovery
        if state.activity_state == ActivityState.RESTING:
            state.stress_level = max(0, state.stress_level - 0.003)

        # Fatigue update
        if input_data.velocity_magnitude < 0.05:
            state.fatigue = max(0, state.fatigue - 0.5)
        elif input_data.is_burst_swimming:
            state.fatigue = min(100, state.fatigue + 2.0)

        # Resting accelerates fatigue recovery (up to 3x normal rate)
        if state.activity_state == ActivityState.RESTING:
            fatigue_recovery = 1.5 * (1 + 0.01 * state.rest_duration_steps)
            fatigue_recovery = min(fatigue_recovery, 3.0)  # Upper limit
            state.fatigue = max(0, state.fatigue - fatigue_recovery)

    # ═══════════ Public Interface Methods ═══════════

    def add_food_to_stomach(self, state: MetabolismState, food_mass: float, body_mass: float,
                            food_profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Add ingested food to the stomach with capacity check.

        Blends the incoming food's nutrient profile into the existing
        stomach mixture using mass-weighted averaging.

        Args:
            state: Mutable metabolism state.
            food_mass: Mass of food item in grams.
            body_mass: Current fish body mass in grams.
            food_profile: Optional nutrient composition dictionary.

        Returns:
            Dictionary with 'success' bool and details (food_mass,
            total_content_mass, fullness_pct, diet_profile) or 'reason'
            on failure.
        """
        stomach_capacity = body_mass * CONFIG.feeding.stomach_capacity_ratio

        # Check capacity
        if state.stomach_content_mass + food_mass > stomach_capacity:
            return {'success': False, 'reason': 'stomach_full'}

        # Add mass
        state.stomach_content_mass += food_mass
        self._mix_stomach_profile(state, food_mass, food_profile or {})

        # Record initial meal mass
        if state.initial_meal_mass == 0 or state.stomach_content_mass == food_mass:
            state.initial_meal_mass = food_mass
        else:
            state.initial_meal_mass += food_mass

        # Synchronize fullness percentage (for compatibility)
        state.stomach_fullness = (state.stomach_content_mass / stomach_capacity) * 100

        return {
            'success': True,
            'food_mass': food_mass,
            'total_content_mass': state.stomach_content_mass,
            'fullness_pct': state.stomach_fullness,
            'diet_profile': {
                'protein_fraction': state.stomach_protein_fraction,
                'lipid_fraction': state.stomach_lipid_fraction,
                'carbohydrate_fraction': state.stomach_carbohydrate_fraction,
                'adc_protein': state.stomach_adc_protein,
                'adc_lipid': state.stomach_adc_lipid,
                'adc_carbohydrate': state.stomach_adc_carbohydrate,
                'include_carbohydrate_energy': state.stomach_include_carbohydrate_energy,
            }
        }

    def get_state_summary(self, state: MetabolismState) -> Dict[str, Any]:
        """Get a comprehensive state summary including energy debuff.

        Args:
            state: Current metabolism state.

        Returns:
            Dictionary with all tracked metabolic quantities, activity state,
            buoyancy costs, digestion state, and energy debuff coefficients.
        """
        mass_loss_pct = 0
        if state.initial_body_mass > 0:
            current_mass = state.lipid_reserve + state.protein_reserve + state.initial_body_mass * 0.72
            mass_loss_pct = (1 - current_mass / state.initial_body_mass) * 100

        # Compute energy debuff coefficients
        debuff = self._calculate_energy_debuff(state.energy)

        return {
            'energy': round(state.energy, 2),
            'stomach_fullness': round(state.stomach_fullness, 2),
            'stress_level': round(state.stress_level, 3),
            'fatigue': round(state.fatigue, 2),
            'is_digesting': state.is_digesting,
            'lipid_reserve': round(state.lipid_reserve, 3),
            'protein_reserve': round(state.protein_reserve, 3),
            'mass_loss_pct': round(mass_loss_pct, 2),
            # Activity state
            'activity_state': state.activity_state.value,
            'rest_duration_steps': state.rest_duration_steps,
            'total_rest_steps': state.total_rest_steps,
            'metabolism_factor': round(state.current_metabolism_factor, 3),
            'growth_bonus': round(state.current_growth_bonus, 3),
            # Buoyancy
            'buoyancy_energy_consumed': round(state.buoyancy_energy_consumed, 4),
            'total_buoyancy_energy': round(state.total_buoyancy_energy, 4),
            # Digestion
            'stomach_content_mass': round(state.stomach_content_mass, 4),
            'initial_meal_mass': round(state.initial_meal_mass, 4),
            'stomach_diet_profile': {
                'protein_fraction': round(state.stomach_protein_fraction, 4),
                'lipid_fraction': round(state.stomach_lipid_fraction, 4),
                'carbohydrate_fraction': round(state.stomach_carbohydrate_fraction, 4),
                'adc_protein': round(state.stomach_adc_protein, 4),
                'adc_lipid': round(state.stomach_adc_lipid, 4),
                'adc_carbohydrate': round(state.stomach_adc_carbohydrate, 4),
                'include_carbohydrate_energy': state.stomach_include_carbohydrate_energy
            },
            # Energy debuff coefficients
            'energy_debuff': {
                'speed_factor': round(debuff['speed_factor'], 3),
                'reaction_factor': round(debuff['reaction_factor'], 3),
                'propulsion_factor': round(debuff['propulsion_factor'], 3),
                'burst_available': debuff['burst_available']
            }
        }

    # ═══════════ Activity State Query Methods ═══════════

    def is_resting(self, state: MetabolismState) -> bool:
        """Check whether the fish is currently in resting state.

        Args:
            state: Current metabolism state.

        Returns:
            True if the fish is resting.
        """
        return state.activity_state == ActivityState.RESTING

    def get_reaction_delay(self, state: MetabolismState) -> float:
        """Get current reaction delay based on activity state.

        Args:
            state: Current metabolism state.

        Returns:
            Reaction delay in seconds.
        """
        rc = self.rc
        if state.activity_state == ActivityState.RESTING:
            return rc.rest_reaction_delay
        return rc.active_reaction_delay

    def can_switch_state(self, state: MetabolismState) -> bool:
        """Check whether an activity state transition is currently permitted.

        Args:
            state: Current metabolism state.

        Returns:
            True if a state switch is allowed (cooldown expired and minimum
            rest duration met).
        """
        if state.state_switch_cooldown > 0:
            return False
        if (state.activity_state == ActivityState.RESTING and
                state.rest_duration_steps < self.rc.min_rest_duration_steps):
            return False
        return True

    def set_debug(self, enabled: bool, frequency: int = 100) -> None:
        """Enable or disable debug output.

        Args:
            enabled: Whether to enable debug printing.
            frequency: Step interval for periodic debug output.
        """
        self.debug = enabled
        self.debug_frequency = frequency


# ═══════════ Factory Functions ═══════════

def create_metabolism_system() -> MetabolismSystem:
    """Create and return a new MetabolismSystem instance.

    Returns:
        Initialized MetabolismSystem.
    """
    return MetabolismSystem()


def create_metabolism_state(
        energy: float = None,
        stomach_fullness: float = None
) -> MetabolismState:
    """Create a MetabolismState with optional initial overrides.

    Args:
        energy: Initial energy percentage (default from CONFIG).
        stomach_fullness: Initial stomach fullness percentage (default from CONFIG).

    Returns:
        Initialized MetabolismState.
    """
    state = MetabolismState(
        energy=energy if energy is not None else CONFIG.agent_init.initial_energy,
        stomach_fullness=stomach_fullness if stomach_fullness is not None else CONFIG.agent_init.initial_stomach_fullness
    )
    return state


# ═══════════ Module Entry Point ═══════════

if __name__ == "__main__":
    # Keep __main__ minimal to avoid encoding-related test-print issues.
    print("MetabolismSystem module loaded.")
