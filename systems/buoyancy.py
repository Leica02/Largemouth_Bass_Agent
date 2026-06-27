#!/usr/bin/env python3
"""
Swim Bladder Buoyancy Subsystem
================================

Physostomous gas bladder physics model implementing depth-dependent neutral
buoyancy regulation based on Boyle's law pressure-volume relationships.

This module simulates the active and passive processes by which teleost fish
regulate swim bladder volume to achieve neutral buoyancy at varying depths.
The implementation couples ideal gas law thermodynamics with bioenergetic
constraints on gas secretion and resorption.

Literature basis:
    - Harden Jones (1951), Alexander (1959): Boyle's law application to
      swim bladder volume-depth relationships.
    - Denton et al. (1972): Gas diffusion across swim bladder walls.
    - Harden Jones & Scholes (1985): Gas secretion rate equations.
    - Pelster & Scheid (1992): Energetics of gas secretion via the
      rete mirabile countercurrent system.

Core principles:
    1. Swim bladder volume modulates buoyancy (Archimedes' principle).
    2. Gas secretion requires metabolic energy (isothermal compression
       work scaled by biological efficiency).
    3. Passive gas leakage occurs via diffusion to surrounding tissues.
    4. Depth changes alter swim bladder volume according to Boyle's law
       (P1 * V1 = P2 * V2 at constant temperature and gas amount).

Simplified implementation:
    - Complex gas secretion/resorption physiology is abstracted as a
      continuous density adjustment signal.
    - The AI agent outputs a control signal in [-1, 1]:
        * Negative values: decrease density (secrete gas, increase buoyancy).
        * Positive values: increase density (resorb gas, decrease buoyancy).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple
from enum import Enum
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Attempt to import configuration; fall back to defaults on failure
try:
    from config import CONFIG, BuoyancyConfig
except ImportError:
    CONFIG = None

# ============================================================
# Data class definitions
# ============================================================

@dataclass
class BuoyancyState:
    """Swim bladder buoyancy state variables."""

    # Current swim bladder volume (m^3)
    swimbladder_volume: float = 0.0

    # Target volume for neutral buoyancy (m^3)
    neutral_volume: float = 0.0

    # Current relative density (relative to water; 1.0 = neutral buoyancy)
    relative_density: float = 1.0

    # Current net buoyancy force (N); positive = upward, negative = downward
    net_buoyancy_force: float = 0.0

    # Gas amount in swim bladder (mol)
    gas_amount: float = 0.0

    # Cumulative energy consumed (kJ)
    total_energy_consumed: float = 0.0

    # Previous step adjustment action
    last_adjustment: float = 0.0

    # Whether the fish is in a neutrally buoyant state
    is_neutral: bool = False

    # Depth (m, positive values indicate depth below surface)
    current_depth: float = 0.0

    # Used for passive gas adjustment on body mass change
    last_body_mass: float = 0.0  # Previous step body mass (g)

    # Adjustment history (for smoothing)
    adjustment_history: np.ndarray = field(
        default_factory=lambda: np.zeros(5, dtype=np.float32)
    )


@dataclass
class BuoyancyInput:
    """Input parameters for the buoyancy system per time step."""

    # AI buoyancy control signal [-1, 1]
    # -1 = maximum density decrease (secrete gas, increase buoyancy)
    # +1 = maximum density increase (resorb gas, decrease buoyancy)
    #  0 = maintain current state
    buoyancy_control: float = 0.0
    body_mass: float = 20.0
    total_length: float = 0.07
    depth: float = 0.3
    water_temp: float = field(default_factory=lambda: CONFIG.environment.water_temp if CONFIG else 25.0)
    time_step: float = field(default_factory=lambda: CONFIG.environment.time_step if CONFIG else 0.3)
    time_acceleration: float = field(default_factory=lambda: CONFIG.environment.time_acceleration if CONFIG else 300.0)
    is_resting: bool = False


@dataclass
class BuoyancyOutput:
    """Output results from a single buoyancy system update."""

    # Net buoyancy force (N)
    net_buoyancy_force: float = 0.0

    # Relative density (dimensionless)
    relative_density: float = 1.0

    # Equivalent vertical acceleration (m/s^2)
    vertical_acceleration: float = 0.0

    # Energy consumed this step (kJ)
    energy_consumed: float = 0.0

    # Swim bladder volume change (m^3)
    volume_change: float = 0.0

    # Gas leaked via diffusion (mol)
    gas_leaked: float = 0.0

    # Whether neutral buoyancy has been achieved
    is_neutral: bool = False

    # Current swim bladder volume fraction
    swimbladder_ratio: float = 0.0


# ============================================================
# Buoyancy system core class
# ============================================================

class BuoyancySystem:
    """Swim bladder buoyancy control system.

    Simulates the process by which fish regulate swim bladder volume to
    control buoyancy. The model couples Archimedes' principle with ideal
    gas law thermodynamics under Boyle's law compression.

    Physical principles:
        1. Archimedes' principle: buoyant force equals weight of displaced
           water.
        2. Effective fish density = (tissue mass + gas mass) / total volume.
        3. Neutral buoyancy is achieved when effective density equals water
           density.

    Simplifying assumptions:
        - Gas mass is negligible compared to tissue mass.
        - Tissue volume remains constant over a single simulation step.
        - Temperature effects on gas volume are incorporated via the ideal
          gas law.
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize the buoyancy system.

        Args:
            config: Buoyancy configuration object. If None, the global
                CONFIG.buoyancy is used.

        Raises:
            ImportError: If CONFIG.buoyancy cannot be resolved.
        """
        if config is not None:
            self.c = config
        elif CONFIG is not None and hasattr(CONFIG, 'buoyancy'):
            self.c = CONFIG.buoyancy
        else:
            raise ImportError("Cannot import CONFIG.buoyancy; check config.py")

        self.debug = False
        self.gravity = CONFIG.physics.gravity if CONFIG else 9.8
        self.gas_constant = 8.314  # J/(mol*K)

    def initialize(self, state: BuoyancyState, body_mass: float, total_length: float) -> None:
        """Initialize the swim bladder state for a given fish.

        Computes the neutral buoyancy volume from body mass and tissue
        density, then sets the initial gas amount using the ideal gas law
        at the birth depth pressure.

        Args:
            state: Buoyancy state object to initialize.
            body_mass: Fish body mass (g).
            total_length: Fish total length (m).
        """
        c = self.c

        # Compute fish tissue volume (assuming density slightly above water)
        mass_kg = body_mass / 1000.0
        tissue_volume = mass_kg / c.fish_tissue_density  # m^3

        # Compute swim bladder volume required for neutral buoyancy
        # Neutral buoyancy condition: total_mass = displaced water mass
        # mass = (tissue_volume + swimbladder_volume) * water_density
        # Solving: swimbladder_volume = mass / water_density - tissue_volume
        neutral_volume = mass_kg / c.water_density - tissue_volume
        neutral_volume = max(neutral_volume, tissue_volume * c.min_volume_ratio)

        state.neutral_volume = neutral_volume
        state.swimbladder_volume = neutral_volume  # Initialize at neutral buoyancy

        # Compute initial gas amount using actual pressure at birth depth
        # (not surface pressure). Fish typically hatch at 0.2-0.4 m depth,
        # where pressure is approximately 1.02-1.04 atm.
        init_depth = state.current_depth if state.current_depth > 0 else 0.3
        pressure_atm = c.atmospheric_pressure + init_depth * c.pressure_per_meter
        temp_kelvin = 25.0 + 273.15

        # PV = nRT -> n = PV / (RT)
        # Unit conversion: atm -> Pa (*101325), m^3, K
        pressure_pa = pressure_atm * 101325
        state.gas_amount = (pressure_pa * neutral_volume) / (self.gas_constant * temp_kelvin)

        state.relative_density = 1.0
        state.net_buoyancy_force = 0.0
        state.is_neutral = True
        state.current_depth = 0.3
        state.last_adjustment = 0.0
        state.total_energy_consumed = 0.0
        state.last_body_mass = body_mass

        if self.debug:
            print(f"  Buoyancy system initialized:")
            print(f"   Body mass: {body_mass:.1f} g")
            print(f"   Tissue volume: {tissue_volume * 1e6:.2f} cm^3")
            print(f"   Neutral swim bladder volume: {neutral_volume * 1e6:.2f} cm^3")
            print(f"   Swim bladder fraction: {neutral_volume / (tissue_volume + neutral_volume) * 100:.1f}%")

    def update(self, state: BuoyancyState, input_data: BuoyancyInput) -> BuoyancyOutput:
        """Execute one buoyancy update step.

        Time convention:
            - Swim bladder adjustment (secretion/resorption): uses real time
              dt for real-time feedback control.
            - Gas leakage: uses accelerated time effective_dt to model the
              biological diffusion process.
            - Energy consumption: uses accelerated time effective_dt to model
              the biological metabolic process.

        Args:
            state: Current buoyancy state (modified in place).
            input_data: Input parameters for this time step.

        Returns:
            BuoyancyOutput containing net force, density, acceleration, and
            energy metrics for this step.
        """
        c = self.c
        dt = input_data.time_step  # Real time: 0.3 s
        time_acc = input_data.time_acceleration
        effective_dt = dt * time_acc  # Accelerated time: for biological processes

        # Update depth
        state.current_depth = input_data.depth

        # ==================== Dynamic adaptation to body mass changes ====================
        mass_kg = input_data.body_mass / 1000.0
        tissue_volume = mass_kg / c.fish_tissue_density

        new_neutral_volume = mass_kg / c.water_density - tissue_volume
        new_neutral_volume = max(new_neutral_volume, tissue_volume * c.min_volume_ratio)
        state.neutral_volume = new_neutral_volume

        # ===== Passive gas amount adjustment (maintain relative density on mass change) =====
        if state.last_body_mass > 0 and abs(input_data.body_mass - state.last_body_mass) > 0.001:
            # Body mass has changed; adjust gas amount accordingly
            last_mass_kg = state.last_body_mass / 1000.0
            last_tissue_volume = last_mass_kg / c.fish_tissue_density

            # Previous total volume
            old_total_volume = last_tissue_volume + state.swimbladder_volume

            # Scale target total volume by mass ratio
            mass_ratio = mass_kg / last_mass_kg
            new_total_volume = old_total_volume * mass_ratio

            # New target swim bladder volume
            target_swimbladder_volume = new_total_volume - tissue_volume
            target_swimbladder_volume = max(target_swimbladder_volume, tissue_volume * c.min_volume_ratio)

            # Adjust gas amount via the ideal gas law
            pressure_atm = c.atmospheric_pressure + input_data.depth * c.pressure_per_meter
            temp_kelvin = input_data.water_temp + 273.15
            pressure_pa = pressure_atm * 101325

            state.gas_amount = (pressure_pa * target_swimbladder_volume) / (self.gas_constant * temp_kelvin)

            if self.debug:
                print(f"  Passive gas adjustment: mass {state.last_body_mass:.2f} -> {input_data.body_mass:.2f} g")

        # Update recorded body mass
        state.last_body_mass = input_data.body_mass

        # 1. Compute current pressure (Boyle's law)
        pressure_atm = c.atmospheric_pressure + input_data.depth * c.pressure_per_meter

        # 2. Apply depth-induced swim bladder volume change
        temp_kelvin = input_data.water_temp + 273.15
        pressure_pa = pressure_atm * 101325

        if state.gas_amount > 0:
            volume_from_gas = (state.gas_amount * self.gas_constant * temp_kelvin) / pressure_pa
        else:
            volume_from_gas = 0

        # 3. Passive gas leakage [uses accelerated time] (passive biological process)
        gas_leaked = self._calculate_leakage(state, input_data, effective_dt)
        state.gas_amount = max(0, state.gas_amount - gas_leaked)

        # 4. AI-controlled buoyancy adjustment [uses real time] (active control)
        adjustment = input_data.buoyancy_control

        if input_data.is_resting:
            adjustment *= c.rest_adjustment_factor

        # Gas secretion/resorption [key: uses real time dt for adjustment]
        volume_change, energy_consumed = self._process_adjustment_realtime(
            state, input_data, adjustment, pressure_atm, temp_kelvin,
            dt,  # Real time for adjustment rate
            effective_dt  # Accelerated time for energy consumption
        )

        # 5. Update swim bladder volume
        mass_kg = input_data.body_mass / 1000.0
        tissue_volume = mass_kg / c.fish_tissue_density

        new_volume_from_gas = (state.gas_amount * self.gas_constant * temp_kelvin) / pressure_pa

        max_volume = state.neutral_volume * (1 + c.max_volume_change_ratio)
        min_volume = tissue_volume * c.min_volume_ratio
        state.swimbladder_volume = np.clip(new_volume_from_gas, min_volume, max_volume)

        # 6. Compute buoyancy force and density
        total_volume = tissue_volume + state.swimbladder_volume
        effective_mass = mass_kg
        effective_density = effective_mass / total_volume

        state.relative_density = effective_density / c.water_density
        displaced_water_mass = total_volume * c.water_density
        state.net_buoyancy_force = (displaced_water_mass - mass_kg) * self.gravity

        # 7. Determine whether neutral buoyancy is achieved
        state.is_neutral = abs(state.relative_density - 1.0) < 0.02

        # 8. Compute vertical acceleration
        vertical_acceleration = state.net_buoyancy_force / mass_kg

        # 9. Update energy statistics
        state.total_energy_consumed += energy_consumed
        state.last_adjustment = adjustment

        swimbladder_ratio = state.swimbladder_volume / total_volume

        if self.debug and np.random.random() < 0.01:
            print(f"  Buoyancy: depth={input_data.depth:.2f} m, "
                  f"density_ratio={state.relative_density:.3f}, "
                  f"net_force={state.net_buoyancy_force:.4f} N, "
                  f"bladder_fraction={swimbladder_ratio * 100:.1f}%")

        return BuoyancyOutput(
            net_buoyancy_force=state.net_buoyancy_force,
            relative_density=state.relative_density,
            vertical_acceleration=vertical_acceleration,
            energy_consumed=energy_consumed,
            volume_change=volume_change,
            gas_leaked=gas_leaked,
            is_neutral=state.is_neutral,
            swimbladder_ratio=swimbladder_ratio
        )

    def _process_adjustment_realtime(self, state: BuoyancyState, input_data: BuoyancyInput,
                                     adjustment: float, pressure_atm: float, temp_kelvin: float,
                                     dt_realtime: float,
                                     dt_accelerated: float
                                     ) -> Tuple[float, float]:
        """Process buoyancy adjustment with separated real-time response and energy cost.

        The adjustment rate (volume change per second) uses real time to
        provide immediate control feedback, while energy consumption is
        scaled by accelerated time to reflect the biological metabolic cost
        over the simulated duration.

        Args:
            state: Current buoyancy state (modified in place).
            input_data: Input parameters for this step.
            adjustment: Control signal in [-1, 1].
            pressure_atm: Current hydrostatic pressure (atm).
            temp_kelvin: Water temperature (K).
            dt_realtime: Real time step (s); used for adjustment rate.
            dt_accelerated: Accelerated time step (s); used for energy cost.

        Returns:
            Tuple of (volume_change in m^3, energy_consumed in kJ).
        """
        c = self.c

        if abs(adjustment) < 0.01:
            return 0.0, 0.0

        mass_kg = input_data.body_mass / 1000.0
        tissue_volume = mass_kg / c.fish_tissue_density

        if adjustment < 0:
            # Secrete gas (decrease density, increase buoyancy)
            # [Key] Adjustment rate uses real time for real-time feedback
            rate = c.secretion_rate_max * abs(adjustment)
            max_volume_increase = state.neutral_volume * rate * dt_realtime  # <- real time

            max_volume = state.neutral_volume * (1 + c.max_volume_change_ratio)
            allowed_increase = max(0, max_volume - state.swimbladder_volume)
            volume_change = min(max_volume_increase, allowed_increase)

            pressure_pa = pressure_atm * 101325
            gas_needed = (pressure_pa * volume_change) / (self.gas_constant * temp_kelvin)
            state.gas_amount += gas_needed

            # [Key] Energy consumption uses accelerated time (biological process)
            depth_factor = 1.0 + c.depth_energy_factor * input_data.depth
            energy_consumed = gas_needed * c.secretion_energy_base * depth_factor
            # Scale energy by the acceleration ratio
            energy_consumed *= (dt_accelerated / dt_realtime) if dt_realtime > 0 else 1.0

        else:
            # Resorb gas (increase density, decrease buoyancy)
            rate = c.absorption_rate_max * abs(adjustment)
            max_volume_decrease = state.neutral_volume * rate * dt_realtime  # <- real time

            min_volume = tissue_volume * c.min_volume_ratio
            allowed_decrease = max(0, state.swimbladder_volume - min_volume)
            volume_change = -min(max_volume_decrease, allowed_decrease)

            pressure_pa = pressure_atm * 101325
            gas_released = (pressure_pa * abs(volume_change)) / (self.gas_constant * temp_kelvin)
            state.gas_amount = max(0, state.gas_amount - gas_released)

            # Scale energy by the acceleration ratio
            energy_consumed = gas_released * c.absorption_energy_base
            energy_consumed *= (dt_accelerated / dt_realtime) if dt_realtime > 0 else 1.0

        return volume_change, energy_consumed

    def _calculate_leakage(self, state: BuoyancyState, input_data: BuoyancyInput,
                           effective_dt: float) -> float:
        """Calculate passive gas leakage through the swim bladder wall.

        Based on the literature diffusion model:
            Leakage = G * Ss * (P - P0) / (R * T)

        Simplified implementation:
            leakage_rate = base_rate * (1 + depth_factor * depth) * dt

        Note:
            Uses real time step to avoid excessive gas loss under time
            acceleration, which would cause chronic negative buoyancy.

        Args:
            state: Current buoyancy state.
            input_data: Input parameters for this step.
            effective_dt: Effective (accelerated) time step (s). Not used
                directly; real time step is extracted from input_data.

        Returns:
            Amount of gas leaked (mol).
        """
        c = self.c
        real_dt = input_data.time_step

        # Base leakage rate (proportional to swim bladder volume)
        base_leakage_rate = c.leakage_rate * c.wall_permeability

        # Depth-dependent enhancement
        depth_factor = 1.0 + c.depth_leakage_factor * input_data.depth

        # Compute leaked gas amount (mol)
        # Uses real time step to prevent excessively rapid leakage under
        # time acceleration that would cause long-term sinking.
        leakage_fraction = base_leakage_rate * depth_factor * real_dt
        gas_leaked = state.gas_amount * leakage_fraction

        return gas_leaked

    def _process_adjustment(self, state: BuoyancyState, input_data: BuoyancyInput,
                            adjustment: float, pressure_atm: float, temp_kelvin: float,
                            effective_dt: float) -> Tuple[float, float]:
        """Process buoyancy adjustment via gas secretion or resorption.

        This is the legacy adjustment method that uses a single effective_dt
        for both rate and energy. Retained for backward compatibility.

        Args:
            state: Current buoyancy state (modified in place).
            input_data: Input parameters for this step.
            adjustment: Control signal in [-1, 1].
            pressure_atm: Current hydrostatic pressure (atm).
            temp_kelvin: Water temperature (K).
            effective_dt: Effective time step (s).

        Returns:
            Tuple of (volume_change in m^3, energy_consumed in kJ).
        """
        c = self.c

        if abs(adjustment) < 0.01:
            return 0.0, 0.0

        mass_kg = input_data.body_mass / 1000.0
        tissue_volume = mass_kg / c.fish_tissue_density

        if adjustment < 0:
            # Secrete gas (decrease density, increase buoyancy)
            rate = c.secretion_rate_max * abs(adjustment)

            # Compute maximum secretion volume
            max_volume_increase = state.neutral_volume * rate * effective_dt
            max_volume = state.neutral_volume * (1 + c.max_volume_change_ratio)
            allowed_increase = max(0, max_volume - state.swimbladder_volume)
            volume_change = min(max_volume_increase, allowed_increase)

            # Compute gas amount to secrete
            pressure_pa = pressure_atm * 101325
            gas_needed = (pressure_pa * volume_change) / (self.gas_constant * temp_kelvin)
            state.gas_amount += gas_needed

            # Energy consumption (secretion)
            # Based on isothermal compression work: W = nRT * ln(P2/P1)
            # Simplified: energy = gas_amount * base_cost * (1 + depth_factor * depth)
            depth_factor = 1.0 + c.depth_energy_factor * input_data.depth
            energy_consumed = gas_needed * c.secretion_energy_base * depth_factor

        else:
            # Resorb gas (increase density, decrease buoyancy)
            rate = c.absorption_rate_max * abs(adjustment)

            # Compute maximum resorption volume
            max_volume_decrease = state.neutral_volume * rate * effective_dt
            min_volume = tissue_volume * c.min_volume_ratio
            allowed_decrease = max(0, state.swimbladder_volume - min_volume)
            volume_change = -min(max_volume_decrease, allowed_decrease)

            # Compute resorbed gas amount
            pressure_pa = pressure_atm * 101325
            gas_released = (pressure_pa * abs(volume_change)) / (self.gas_constant * temp_kelvin)
            state.gas_amount = max(0, state.gas_amount - gas_released)

            # Energy consumption (resorption, lower cost)
            energy_consumed = gas_released * c.absorption_energy_base

        return volume_change, energy_consumed

    def get_maintenance_energy(self, state: BuoyancyState, time_step: float,
                               time_acceleration: float) -> float:
        """Compute baseline energy cost of maintaining swim bladder integrity.

        Args:
            state: Current buoyancy state.
            time_step: Simulation time step (s).
            time_acceleration: Time acceleration factor.

        Returns:
            Maintenance energy consumption (kJ).
        """
        c = self.c
        effective_dt = time_step * time_acceleration
        return c.maintenance_energy_rate * effective_dt

    def calculate_optimal_buoyancy(self, target_depth: float, current_depth: float) -> float:
        """Compute the buoyancy control signal needed to reach a target depth.

        This heuristic can be used for rule-based strategies or as a
        reference signal for reinforcement learning reward shaping.

        Args:
            target_depth: Desired depth (m, positive downward).
            current_depth: Current depth (m, positive downward).

        Returns:
            Recommended control signal in [-1, 1]. Positive values command
            sinking; negative values command rising.
        """
        depth_diff = target_depth - current_depth

        if abs(depth_diff) < 0.1:
            return 0.0  # Near target; maintain current state

        # Desire to sink (increase depth) -> increase density -> positive
        # Desire to rise (decrease depth) -> decrease density -> negative
        control = np.tanh(depth_diff * 2.0)

        return np.clip(control, -1.0, 1.0)

    def set_debug(self, enabled: bool) -> None:
        """Enable or disable debug output.

        Args:
            enabled: If True, diagnostic messages are printed during
                simulation steps.
        """
        self.debug = enabled


# ============================================================
# Factory functions
# ============================================================

def create_buoyancy_system(config: Optional['BuoyancyConfig'] = None) -> BuoyancySystem:
    """Create and return a BuoyancySystem instance.

    Args:
        config: Optional buoyancy configuration. If None, the global
            CONFIG.buoyancy is used.

    Returns:
        Initialized BuoyancySystem instance.
    """
    return BuoyancySystem(config)


def create_buoyancy_state() -> BuoyancyState:
    """Create and return a default BuoyancyState instance.

    Returns:
        BuoyancyState with default initial values.
    """
    return BuoyancyState()


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Swim Bladder Buoyancy Control System Test")
    print("=" * 60)

    # Create system
    buoyancy = create_buoyancy_system()
    buoyancy.set_debug(True)

    state = create_buoyancy_state()

    # Initialize (20 g fish, total length 7 cm)
    buoyancy.initialize(state, body_mass=20.0, total_length=0.07)

    print("\nScenario 1: Neutral buoyancy state")
    print("-" * 40)

    for step in range(5):
        input_data = BuoyancyInput(
            buoyancy_control=0.0,
            body_mass=20.0,
            total_length=0.07,
            depth=0.3
        )
        output = buoyancy.update(state, input_data)
        print(f"Step {step}: density_ratio={output.relative_density:.3f}, "
              f"net_force={output.net_buoyancy_force:.4f} N, "
              f"neutral={output.is_neutral}")

    print("\nScenario 2: Increase buoyancy (gas secretion)")
    print("-" * 40)

    for step in range(10):
        input_data = BuoyancyInput(
            buoyancy_control=-0.5,  # Secrete gas, increase buoyancy
            body_mass=20.0,
            total_length=0.07,
            depth=0.3
        )
        output = buoyancy.update(state, input_data)
        print(f"Step {step}: density_ratio={output.relative_density:.3f}, "
              f"net_force={output.net_buoyancy_force:.4f} N, "
              f"energy={output.energy_consumed:.4f} kJ")

    print("\nScenario 3: Decrease buoyancy (gas resorption)")
    print("-" * 40)

    for step in range(10):
        input_data = BuoyancyInput(
            buoyancy_control=0.8,  # Resorb gas, decrease buoyancy
            body_mass=20.0,
            total_length=0.07,
            depth=0.3
        )
        output = buoyancy.update(state, input_data)
        print(f"Step {step}: density_ratio={output.relative_density:.3f}, "
              f"net_force={output.net_buoyancy_force:.4f} N")

    print("\nScenario 4: Depth-induced volume change (Boyle's law)")
    print("-" * 40)

    # Re-initialize
    state = create_buoyancy_state()
    buoyancy.initialize(state, body_mass=20.0, total_length=0.07)

    depths = [0.1, 0.3, 0.5, 0.7, 0.5, 0.3, 0.1]
    for depth in depths:
        input_data = BuoyancyInput(
            buoyancy_control=0.0,  # No active adjustment
            body_mass=20.0,
            total_length=0.07,
            depth=depth
        )
        output = buoyancy.update(state, input_data)
        print(f"Depth={depth:.1f} m: density_ratio={output.relative_density:.3f}, "
              f"net_force={output.net_buoyancy_force:.4f} N, "
              f"bladder_fraction={output.swimbladder_ratio * 100:.1f}%")

    print("\nScenario 5: Adjustment during resting state")
    print("-" * 40)

    state = create_buoyancy_state()
    buoyancy.initialize(state, body_mass=20.0, total_length=0.07)

    print("Active state adjustment:")
    for step in range(3):
        input_data = BuoyancyInput(
            buoyancy_control=-0.8,
            body_mass=20.0,
            depth=0.3,
            is_resting=False
        )
        output = buoyancy.update(state, input_data)
        print(f"  Step {step}: faster density change, density_ratio={output.relative_density:.3f}")

    print("\nResting state adjustment:")
    for step in range(3):
        input_data = BuoyancyInput(
            buoyancy_control=-0.8,
            body_mass=20.0,
            depth=0.3,
            is_resting=True
        )
        output = buoyancy.update(state, input_data)
        print(f"  Step {step}: slower density change, density_ratio={output.relative_density:.3f}")

    print(f"\nTotal energy consumed: {state.total_energy_consumed:.4f} kJ")
    print("\nTest complete.")
