#!/usr/bin/env python3
"""
Utility module for largemouth bass bioenergetics simulation.

Provides biological formula utilities including allometric length-weight
relationships, swimming performance models, buoyancy physics, and
physiological calculations used throughout the reinforcement learning
environment.
"""

from .biological_formulas import (
    mass_to_length,
    length_to_mass,
    calculate_sustained_speed,
    calculate_burst_speed,
    calculate_fish_volume,
    calculate_buoyancy_forces,
    calculate_capture_radius,
    calculate_detection_range,
    calculate_strike_range,
    calculate_stomach_capacity,
    calculate_growth_increment,
    calculate_q10_factor,
)

__all__ = [
    'mass_to_length',
    'length_to_mass',
    'calculate_sustained_speed',
    'calculate_burst_speed',
    'calculate_fish_volume',
    'calculate_buoyancy_forces',
    'calculate_capture_radius',
    'calculate_detection_range',
    'calculate_strike_range',
    'calculate_stomach_capacity',
    'calculate_growth_increment',
    'calculate_q10_factor',
]
