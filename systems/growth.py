#!/usr/bin/env python3
"""
Somatic Growth Subsystem for Largemouth Bass (Micropterus salmoides)
====================================================================

Implements mass-to-length conversion and growth energy accumulation for
an individual-based bioenergetics model of juvenile largemouth bass.

Key features (v3 - length synchronisation revision):
1. Bidirectional body length updates when mass changes (gain or loss).
2. Stage-specific length-weight relationships (larval / juvenile / adult)
   derived from literature allometric parameters.
3. Threshold mechanism for length reduction to avoid spurious
   high-frequency updates during minor mass fluctuations.

Design principles:
- Total length and body mass remain mutually consistent at all times.
- Segmented allometric formulae follow published relationships.
- Mass gain triggers immediate length increase.
- Mass loss triggers length decrease only when a relative threshold
  is exceeded, reflecting skeletal rigidity constraints.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CONFIG

from utils.biological_formulas import (
    mass_to_length,
    length_to_mass,
    get_growth_stage,
    calculate_fish_volume,
    calculate_buoyancy_forces,
    calculate_fish_energy_density
)

# ============================================================
# Length-weight conversion utilities (smooth transition)
# ============================================================

def _sigmoid(x: float, center: float, width: float) -> float:
    """Smooth sigmoid transition function.

    Args:
        x: Input value.
        center: Inflection point of the sigmoid.
        width: Controls the steepness of the transition.

    Returns:
        Sigmoid output in (0, 1).
    """
    return 1.0 / (1.0 + np.exp(-(x - center) / width))

# ============================================================
# Activity state enumeration
# ============================================================

class ActivityState(Enum):
    ACTIVE = "active"
    RESTING = "resting"


# ============================================================
# Data class definitions
# ============================================================

@dataclass
class GrowthState:
    """Somatic growth state container (v3, length-synchronised).

    Attributes:
        body_mass: Current wet body mass (g).
        total_length: Current total length (m).
        initial_mass: Mass at initialisation (g).
        growth_accumulation: Accumulated growth energy not yet converted
            to somatic tissue (J).
        growth_count: Total number of discrete growth events.
        total_growth_energy: Cumulative energy allocated to growth (J).
        volume: Estimated body volume (m^3).
        net_gravity_in_water: Net gravitational force in water (N).
        gravity_in_air: Weight in air (N).
        growth_during_rest: Mass gained during resting periods (g).
        growth_count_during_rest: Number of growth events during rest.
        energy_added_during_rest: Energy allocated to growth while
            resting (J).
        current_activity_state: Current behavioural activity state.
        last_length_update_mass: Body mass at last length update (g).
        length_shrink_count: Number of length-reduction events
            (diagnostic).
    """
    body_mass: float = field(default_factory=lambda: CONFIG.agent_init.initial_mass)
    total_length: float = 0.0
    initial_mass: float = field(default_factory=lambda: CONFIG.agent_init.initial_mass)
    growth_accumulation: float = 0.0
    growth_count: int = 0
    total_growth_energy: float = 0.0
    volume: float = 0.0
    net_gravity_in_water: float = 0.001
    gravity_in_air: float = 0.01

    # Resting-state growth statistics
    growth_during_rest: float = 0.0
    growth_count_during_rest: int = 0
    energy_added_during_rest: float = 0.0
    current_activity_state: ActivityState = ActivityState.ACTIVE

    # Length synchronisation fields
    last_length_update_mass: float = 0.0  # Mass at last length update
    length_shrink_count: int = 0  # Cumulative shrink events (diagnostic)

    def __post_init__(self):
        if self.total_length == 0.0:
            self.total_length = mass_to_length(self.body_mass)
        if self.last_length_update_mass == 0.0:
            self.last_length_update_mass = self.body_mass


@dataclass
class GrowthOutput:
    """Output container for a single growth processing step.

    Attributes:
        grew: Whether a growth event occurred.
        growth_times: Number of growth events this step.
        mass_change: Change in body mass (g).
        length_change: Change in total length (m).
        new_mass: Updated body mass (g).
        new_length: Updated total length (m).
        grew_during_rest: Whether growth occurred during a rest period.
        length_shrunk: Whether total length decreased.
    """
    grew: bool = False
    growth_times: int = 0
    mass_change: float = 0.0
    length_change: float = 0.0
    new_mass: float = 0.0
    new_length: float = 0.0
    grew_during_rest: bool = False

    # Length reduction flag
    length_shrunk: bool = False


# ============================================================
# Growth system core class
# ============================================================

class GrowthSystem:
    """Somatic growth system (v3, length-synchronisation revision).

    Converts accumulated growth energy into mass gain and maintains
    consistency between body mass and total length using stage-specific
    allometric relationships. Optionally applies logistic growth
    suppression at high body masses.
    """

    def __init__(self) -> None:
        self.c = CONFIG.growth
        self.debug: bool = False
        self.energy_density: float = calculate_fish_energy_density(CONFIG.agent_init.initial_mass)

        # Logistic growth parameters
        self.max_body_mass: float = 557.0
        # Refined metabolism already embeds mass-dependent growth dynamics.
        # Keep logistic damping disabled by default to avoid double suppression.
        self.logistic_strength: float = 0.0

        # Growth mode
        self.growth_mode: str = 'logistic'
        self.sgr_a: float = self.c.sgr_coefficient
        self.sgr_b: float = self.c.sgr_exponent

        # Length synchronisation parameters
        # Length shrink threshold: mass must decrease by this fraction
        # before length is updated (avoids spurious micro-adjustments).
        self.length_shrink_threshold: float = 0.02  # 2%
        # Length shrink factor: actual shrinkage = computed value * factor
        # (models incomplete skeletal contraction).
        self.length_shrink_factor: float = 0.7

    def initialize_state(self, body_mass: float) -> GrowthState:
        """Create and return an initialised growth state.

        Args:
            body_mass: Initial wet body mass (g).

        Returns:
            A fully initialised GrowthState with consistent length,
            volume, and buoyancy parameters.
        """
        length = mass_to_length(body_mass)
        net_gravity, air_gravity = calculate_buoyancy_forces(length)
        volume = calculate_fish_volume(length)

        return GrowthState(
            body_mass=body_mass,
            total_length=length,
            initial_mass=body_mass,
            volume=volume,
            net_gravity_in_water=net_gravity,
            gravity_in_air=air_gravity,
            last_length_update_mass=body_mass
        )

    def add_growth_energy(self, state: GrowthState, energy: float,
                          is_resting: bool = False) -> None:
        """Add energy to the growth accumulator.

        Args:
            state: Current growth state to update.
            energy: Energy to allocate to growth (J).
            is_resting: Whether the fish is currently in a resting state.
        """
        state.growth_accumulation += energy
        state.total_growth_energy += energy

        if is_resting:
            state.energy_added_during_rest += energy

    def update_activity_state(self, state: GrowthState, activity_state: ActivityState) -> None:
        """Update the current behavioural activity state.

        Args:
            state: Current growth state to update.
            activity_state: New activity state (ACTIVE or RESTING).
        """
        state.current_activity_state = activity_state

    def _calculate_dynamic_threshold(self, body_mass: float) -> float:
        """Calculate the dynamic growth energy threshold.

        The threshold scales with body mass and energy density so that
        larger fish require proportionally more energy per growth event.

        Args:
            body_mass: Current body mass (g).

        Returns:
            Energy threshold for triggering a growth event (J).
        """
        threshold_pct = self.c.growth_threshold / 100
        energy_density = calculate_fish_energy_density(body_mass)
        self.energy_density = energy_density
        return body_mass * energy_density * threshold_pct

    def _calculate_logistic_factor(self, body_mass: float) -> float:
        """Calculate the logistic growth suppression factor.

        Implements density-independent growth inhibition as mass
        approaches the asymptotic maximum, preventing unrealistic
        growth at large body sizes.

        Args:
            body_mass: Current body mass (g).

        Returns:
            Suppression factor in [0.05, 1.0].
        """
        if self.max_body_mass <= 0:
            return 1.0

        mass_ratio = body_mass / self.max_body_mass
        power = 1.5
        factor = 1.0 - self.logistic_strength * (mass_ratio ** power)

        return max(0.05, min(1.0, factor))

    def _calculate_growth_factor(self, body_mass: float) -> float:
        """Calculate the effective growth rate factor.

        Selects between power-law SGR scaling or logistic suppression
        depending on the configured growth mode.

        Args:
            body_mass: Current body mass (g).

        Returns:
            Dimensionless growth rate scaling factor.
        """
        if self.growth_mode == 'power_law':
            sgr = self.predict_sgr(body_mass)
            base_sgr = self.predict_sgr(100)
            return self.c.growth_rate * (sgr / base_sgr)
        else:
            return self.c.growth_rate * self._calculate_logistic_factor(body_mass)

    def predict_sgr(self, mass: float) -> float:
        """Predict specific growth rate from body mass.

        Uses the allometric relationship SGR = a * W^b, where
        coefficients are derived from literature values for juvenile
        largemouth bass.

        Args:
            mass: Body mass (g).

        Returns:
            Predicted specific growth rate (% day^-1).
        """
        return self.sgr_a * (mass ** self.sgr_b)

    def sync_length_from_mass(self, state: GrowthState, force: bool = False) -> bool:
        """Synchronise total length from current body mass.

        When mass increases, length is updated immediately using the
        allometric formula. When mass decreases, length is reduced only
        if the relative mass loss exceeds a threshold, and a shrinkage
        coefficient is applied to model skeletal rigidity.

        Args:
            state: Current growth state to update.
            force: If True, bypass the shrinkage threshold and force
                an update regardless of mass-loss magnitude.

        Returns:
            True if total length was modified, False otherwise.
        """
        old_length = state.total_length
        expected_length = mass_to_length(state.body_mass)

        # Mass increase -> length always grows
        if state.body_mass > state.last_length_update_mass:
            state.total_length = expected_length
            state.last_length_update_mass = state.body_mass
            self._update_buoyancy_physics(state)

            if self.debug:
                print(f"[Length increase] {old_length * 100:.2f} cm -> "
                      f"{state.total_length * 100:.2f} cm "
                      f"(mass: {state.body_mass:.2f} g)")
            return True

        # Mass decrease -> check whether threshold is exceeded
        if state.body_mass < state.last_length_update_mass:
            mass_loss_ratio = (state.last_length_update_mass - state.body_mass) / state.last_length_update_mass

            if force or mass_loss_ratio >= self.length_shrink_threshold:
                # Compute expected length change
                length_diff = old_length - expected_length

                # Apply shrinkage coefficient (incomplete skeletal contraction)
                actual_shrink = length_diff * self.length_shrink_factor
                state.total_length = old_length - actual_shrink

                # Ensure length does not fall below a minimum plausible value
                min_length = expected_length * 0.9
                state.total_length = max(state.total_length, min_length)

                state.last_length_update_mass = state.body_mass
                state.length_shrink_count += 1
                self._update_buoyancy_physics(state)

                if self.debug:
                    print(f"[Length decrease] {old_length * 100:.2f} cm -> "
                          f"{state.total_length * 100:.2f} cm "
                          f"(mass loss {mass_loss_ratio * 100:.1f}%)")
                return True

        return False

    def process_growth(self, state: GrowthState) -> GrowthOutput:
        """Process accumulated growth energy and convert to somatic mass.

        Converts the energy accumulated in state.growth_accumulation
        into mass gain via the current body energy density, optionally
        applying logistic suppression. Updates length accordingly.

        Args:
            state: Current growth state (modified in place).

        Returns:
            GrowthOutput summarising any mass and length changes.
        """
        output = GrowthOutput()

        growth_count = 0
        old_mass = state.body_mass
        old_length = state.total_length

        is_resting = state.current_activity_state == ActivityState.RESTING
        if state.growth_accumulation > 0:

            mass_before_growth = state.body_mass

            # Execute growth conversion
            body_energy_density = calculate_fish_energy_density(state.body_mass)
            if body_energy_density <= 0:
                mass_gain = 0.0
            else:
                logistic_factor = (
                    self._calculate_logistic_factor(state.body_mass)
                    if self.logistic_strength > 0
                    else 1.0
                )
                mass_gain = (state.growth_accumulation / body_energy_density) * logistic_factor

            if mass_gain > 0:
                state.body_mass += mass_gain
                state.growth_accumulation = 0.0
                state.growth_count += 1
                growth_count = 1

            # Resting-state growth statistics
            if is_resting and mass_gain > 0:
                mass_gained = state.body_mass - mass_before_growth
                state.growth_during_rest += mass_gained
                state.growth_count_during_rest += 1
                output.grew_during_rest = True

            if self.debug:
                rest_indicator = "[REST]" if is_resting else "[ACTIVE]"
                print(f"[Growth] {rest_indicator} event #{state.growth_count}: "
                      f"{old_mass:.2f} g -> {state.body_mass:.2f} g "
                      f"(gain={mass_gain:.4f} g)")

        if growth_count > 0:
            # Mass increased; synchronise length
            self.sync_length_from_mass(state)

            output.grew = True
            output.growth_times = growth_count
            output.mass_change = state.body_mass - old_mass
            output.length_change = state.total_length - old_length

        output.new_mass = state.body_mass
        output.new_length = state.total_length

        return output

    def process_mass_loss(self, state: GrowthState, mass_loss: float) -> GrowthOutput:
        """Process body mass reduction (starvation, injury, etc.).

        Reduces body mass by the specified amount and conditionally
        updates total length if the relative loss exceeds the
        configured threshold.

        Args:
            state: Current growth state (modified in place).
            mass_loss: Amount of mass to remove (g, positive value).

        Returns:
            GrowthOutput containing mass and length change information.
        """
        output = GrowthOutput()
        old_mass = state.body_mass
        old_length = state.total_length

        # Apply mass reduction (floor at 1 g to prevent zero-mass states)
        state.body_mass = max(1.0, state.body_mass - mass_loss)

        # Synchronise length
        length_changed = self.sync_length_from_mass(state)

        output.mass_change = state.body_mass - old_mass  # Negative
        output.length_change = state.total_length - old_length  # May be negative
        output.new_mass = state.body_mass
        output.new_length = state.total_length
        output.length_shrunk = length_changed and state.total_length < old_length

        return output

    def _update_buoyancy_physics(self, state: GrowthState) -> None:
        """Recalculate buoyancy-related physical parameters.

        Updates body volume and gravitational forces after a length
        change to maintain consistency with the hydrodynamic model.

        Args:
            state: Growth state to update in place.
        """
        state.volume = calculate_fish_volume(state.total_length)
        net_gravity, air_gravity = calculate_buoyancy_forces(state.total_length)
        state.net_gravity_in_water = net_gravity
        state.gravity_in_air = air_gravity

    def get_growth_progress(self, state: GrowthState) -> Dict[str, Any]:
        """Retrieve a summary of current growth progress.

        Args:
            state: Current growth state to query.

        Returns:
            Dictionary containing current mass, length, theoretical
            length, deviation percentage, accumulation progress,
            logistic factor, and resting growth statistics.
        """
        dynamic_threshold = self._calculate_dynamic_threshold(state.body_mass)
        logistic_factor = self._calculate_logistic_factor(state.body_mass)

        # Theoretical length corresponding to current mass
        theoretical_length = mass_to_length(state.body_mass)
        length_deviation = (state.total_length - theoretical_length) / theoretical_length * 100

        return {
            'current_mass': state.body_mass,
            'current_length': state.total_length * 100,  # cm
            'theoretical_length': theoretical_length * 100,  # cm
            'length_deviation_pct': length_deviation,
            'mass_gain': state.body_mass - state.initial_mass,
            'growth_count': state.growth_count,
            'accumulation': state.growth_accumulation,
            'threshold': dynamic_threshold,
            'threshold_pct': self.c.growth_threshold,
            'progress_pct': (state.growth_accumulation / dynamic_threshold) * 100 if dynamic_threshold > 0 else 0,
            'logistic_factor': logistic_factor,
            'max_body_mass': self.max_body_mass,
            'growth_during_rest': state.growth_during_rest,
            'growth_count_during_rest': state.growth_count_during_rest,
            'energy_added_during_rest': state.energy_added_during_rest,
            'rest_growth_ratio': (state.growth_during_rest / (state.body_mass - state.initial_mass)
                                  if state.body_mass > state.initial_mass else 0.0),
            'length_shrink_count': state.length_shrink_count,
        }

    def get_rest_growth_stats(self, state: GrowthState) -> Dict[str, float]:
        """Retrieve growth statistics accumulated during rest periods.

        Args:
            state: Current growth state to query.

        Returns:
            Dictionary with rest-period mass gain, event count, energy
            allocated, and the ratio of rest growth to total growth.
        """
        total_growth = state.body_mass - state.initial_mass
        return {
            'growth_during_rest': state.growth_during_rest,
            'growth_count_during_rest': state.growth_count_during_rest,
            'energy_added_during_rest': state.energy_added_during_rest,
            'rest_growth_ratio': state.growth_during_rest / total_growth if total_growth > 0 else 0.0,
        }

    def reset_rest_growth_stats(self, state: GrowthState) -> None:
        """Reset rest-period growth statistics to zero.

        Args:
            state: Growth state whose rest counters will be cleared.
        """
        state.growth_during_rest = 0.0
        state.growth_count_during_rest = 0
        state.energy_added_during_rest = 0.0

    def set_logistic_params(self, max_mass: Optional[float] = None,
                            strength: Optional[float] = None) -> None:
        """Configure logistic growth suppression parameters.

        Args:
            max_mass: Asymptotic maximum body mass (g). If None,
                the current value is retained.
            strength: Logistic suppression strength in [0, 1]. If None,
                the current value is retained.
        """
        if max_mass is not None:
            self.max_body_mass = max_mass
        if strength is not None:
            self.logistic_strength = strength

    def set_growth_mode(self, mode: str) -> None:
        """Set the growth scaling mode.

        Args:
            mode: Either 'logistic' (default) or 'power_law'.
        """
        if mode in ['logistic', 'power_law']:
            self.growth_mode = mode

    def set_debug(self, enabled: bool) -> None:
        """Enable or disable debug output.

        Args:
            enabled: If True, print diagnostic messages during growth
                processing.
        """
        self.debug = enabled


# ============================================================
# Factory functions
# ============================================================

def create_growth_system() -> GrowthSystem:
    """Create and return a default GrowthSystem instance.

    Returns:
        A new GrowthSystem with default configuration.
    """
    return GrowthSystem()


def create_growth_state(body_mass: Optional[float] = None) -> GrowthState:
    """Create an initialised GrowthState via the GrowthSystem.

    Args:
        body_mass: Initial body mass (g). If None, uses the default
            from CONFIG.agent_init.initial_mass.

    Returns:
        A fully initialised GrowthState.
    """
    mass = body_mass if body_mass is not None else CONFIG.agent_init.initial_mass
    return GrowthSystem().initialize_state(mass)


# ============================================================
# Self-test
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Growth System Test (v3 - length synchronisation revision)")
    print("=" * 70)

    # Test length-weight conversion
    print("\n[Mass-Length Conversion Test]")
    print("-" * 50)

    test_weights = [1, 5, 10, 20, 40, 80, 160, 320]
    for w in test_weights:
        length = mass_to_length(w)
        back_mass = length_to_mass(length)
        print(f"Mass: {w:>6.1f} g -> Length: {length * 100:>6.2f} cm -> Back-calc: {back_mass:>6.1f} g")

    # Test growth system
    print("\n[Growth System Test]")
    print("-" * 50)

    growth = create_growth_system()
    growth.set_debug(True)

    state = create_growth_state(20.0)
    print(f"Initial: mass={state.body_mass:.1f} g, length={state.total_length * 100:.2f} cm")

    # Simulate growth via energy addition
    print("\nSimulating growth (adding energy):")
    for step in range(5):
        growth.add_growth_energy(state, 5.0)
        output = growth.process_growth(state)
        if output.grew:
            print(f"  After growth: mass={state.body_mass:.2f} g, "
                  f"length={state.total_length * 100:.2f} cm")

    # Simulate mass loss (starvation)
    print("\nSimulating mass loss (starvation):")
    initial_mass = state.body_mass
    for step in range(5):
        mass_loss = state.body_mass * 0.03  # 3% loss per step
        output = growth.process_mass_loss(state, mass_loss)
        print(f"  Step {step + 1}: mass={state.body_mass:.2f} g, "
              f"length={state.total_length * 100:.2f} cm, "
              f"shrunk={output.length_shrunk}")

    # Growth progress report
    print("\n[Growth Progress Report]")
    progress = growth.get_growth_progress(state)
    for k, v in progress.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")

    print("\nTest complete.")
