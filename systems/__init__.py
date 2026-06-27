#!/usr/bin/env python3
"""
Subsystems Package - Unified Configuration
===========================================

All subsystems obtain constants from config.py; no subsystem defines its own.

Includes the following subsystems:
- metabolism: Metabolic system
- growth: Growth system
- physics: Physics system
- perception: Perception system
- feeding: Feeding system
- interaction: Interaction system
- buoyancy: Buoyancy system
"""

# Metabolism system
from .metabolism import (
    MetabolismSystem,
    MetabolismState,
    MetabolismInput,
    MetabolismOutput,
    create_metabolism_system,
    create_metabolism_state,
)

# Growth system
from .growth import (
    GrowthSystem,
    GrowthState,
    GrowthOutput,
    create_growth_system,
    create_growth_state,
)

# Physics system
from .physics import (
    PhysicsSystem,
    PhysicsState,
    PhysicsInput,
    PhysicsOutput,
    create_physics_system,
    create_physics_state,
)

# Perception system
from .perception import (
    PerceptionSystem,
    PerceptionState,
    PerceptionInput,
    create_perception_system,
    create_perception_state,
)

# Feeding system
from .feeding import (
    FeedingSystem,
    FeedingState,
    FeedingInput,
    FeedingOutput,
    FoodItem,
    create_feeding_system,
    create_feeding_state,
)

# Interaction system
from .interaction import (
    InteractionSystem,
    InteractionState,
    InteractionInput,
    InteractionOutput,
    OtherFish,
    create_interaction_system,
    create_interaction_state,
)

# Additional exports
from .tank_geometry import (
    TankGeometry, CircularTank, RectangularTank,
    create_random_tank, create_default_tank
)
from .obstacles import (
    ObstacleField, RockObstacle, CollisionResult,
    generate_obstacles, create_empty_obstacle_field
)

# Buoyancy system
try:
    from systems.buoyancy import (
        BuoyancySystem,
        BuoyancyState,
        BuoyancyInput,
        BuoyancyOutput,
        BuoyancyConfig,
        create_buoyancy_system,
        create_buoyancy_state
    )

    BUOYANCY_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import buoyancy system: {e}")
    BUOYANCY_AVAILABLE = False

    # Provide empty placeholders
    BuoyancySystem = None
    BuoyancyState = None
    BuoyancyInput = None
    BuoyancyOutput = None
    BuoyancyConfig = None
    create_buoyancy_system = None
    create_buoyancy_state = None

__all__ = [
    # Metabolism
    'MetabolismSystem', 'MetabolismState', 'MetabolismInput', 'MetabolismOutput',
    'create_metabolism_system', 'create_metabolism_state',
    # Growth
    'GrowthSystem', 'GrowthState', 'GrowthOutput',
    'create_growth_system', 'create_growth_state',
    # Physics
    'PhysicsSystem', 'PhysicsState', 'PhysicsInput', 'PhysicsOutput',
    'create_physics_system', 'create_physics_state',
    # Perception
    'PerceptionSystem', 'PerceptionState', 'PerceptionInput',
    'create_perception_system', 'create_perception_state',
    # Feeding
    'FeedingSystem', 'FeedingState', 'FeedingInput', 'FeedingOutput', 'FoodItem',
    'create_feeding_system', 'create_feeding_state',
    # Interaction
    'InteractionSystem', 'InteractionState', 'InteractionInput', 'InteractionOutput', 'OtherFish',
    'create_interaction_system', 'create_interaction_state',
    # Buoyancy
    'BuoyancySystem', 'BuoyancyState', 'BuoyancyInput', 'BuoyancyOutput',
    'BuoyancyConfig', 'create_buoyancy_system', 'create_buoyancy_state',
    'BUOYANCY_AVAILABLE',
]
