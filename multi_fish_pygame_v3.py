#!/usr/bin/env python3
"""
Multi-AI Fish Visualization - Academic Style v3.1 (Training-Aligned Edition)
=============================================================================

v3.1 Fix
    1. action[3] < 0  < -0.3
    2. Speed Limiting0.15 BL/s
    3. NPC
    4. Curriculum configuration Stage 7
    5. Inertia System physics_system.initialize_inertia()
    6. Aligned with training env
    7.
    8.

Requirements:
    pip install pygame numpy stable-baselines3

Controls:
    SPACE  - Pause/Resume
    R      - Reset (Settings dialog)
    T      - Toggle time mode (real-time / accelerated)
    +/-    - Speed up/down
    F      - Toggle fullscreen
    ESC    - Quit
    Mouse  - Click fish list to select
    Scroll - Scroll fish list
"""

import numpy as np
import sys
import os
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from collections import deque
from enum import Enum
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ============================================================
# Import PyGame
# ============================================================
try:
    import pygame
    from pygame import gfxdraw

    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False
    print("[ERROR] PyGame not installed. Run: pip install pygame")
    sys.exit(1)

# ============================================================
# Import environment and model
# ============================================================
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from environment import BassEnvironment
    from config import CONFIG, CURRICULUM_STAGES
    from systems.interaction import (
        OtherFish, FishBehaviorType, FishSize, InteractionInput,
        calculate_sustained_speed, calculate_capture_radius, calculate_strike_range,
        AGGRESSIVE_CONFIG, SURFACE_PREDATOR_CONFIG, FLEEING_CONFIG
    )
    from systems.feeding import FeedingInput
    from systems.tank_geometry import create_random_tank, create_default_tank
    from systems.obstacles import generate_obstacles, create_empty_obstacle_field
    from systems.buoyancy import BuoyancyState, create_buoyancy_state, create_buoyancy_system

    HAS_ENV = True
except ImportError as e:
    HAS_ENV = False
    print(f"[ERROR] Cannot import environment: {e}")
    sys.exit(1)

try:
    from stable_baselines3 import PPO

    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    print("[WARN] stable_baselines3 not installed, using random actions")

try:
    from systems.metabolism import ActivityState
except ImportError:
    class ActivityState(Enum):
        ACTIVE = "active"
        RESTING = "resting"


# ============================================================
# Settings Dialog (Tkinter) - Same as original
# ============================================================

class SettingsDialog:
    """Settings dialog for startup/reset."""

    def __init__(self, parent=None, title="Bass Simulator Settings",
                 default_num_fish=10, default_mass=20.0, default_model_path=""):
        self.result = None
        self.model_path = default_model_path

        self.root = tk.Tk() if parent is None else tk.Toplevel(parent)
        self.root.title(title)
        self.root.geometry("520x420")
        self.root.resizable(False, False)

        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'+{x}+{y}')

        style = ttk.Style()
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'))
        style.configure('Header.TLabel', font=('Arial', 11, 'bold'))
        style.configure('Info.TLabel', font=('Arial', 9), foreground='gray')

        self._create_widgets(default_num_fish, default_mass)
        self.root.protocol("WM_DELETE_WINDOW", self._on_cancel)

    def _create_widgets(self, default_num_fish, default_mass):
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        title_label = ttk.Label(main_frame, text="🐟 Bass Fish Simulator v3.1", style='Title.TLabel')
        title_label.pack(pady=(0, 20))

        # Model selection
        model_frame = ttk.LabelFrame(main_frame, text="AI Model", padding="10")
        model_frame.pack(fill=tk.X, pady=(0, 15))

        model_inner = ttk.Frame(model_frame)
        model_inner.pack(fill=tk.X)

        self.model_var = tk.StringVar(value=self.model_path if self.model_path else "No model (Random Actions)")
        self.model_entry = ttk.Entry(model_inner, textvariable=self.model_var, width=40, state='readonly')
        self.model_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        browse_btn = ttk.Button(model_inner, text="Browse...", command=self._browse_model)
        browse_btn.pack(side=tk.RIGHT)

        clear_btn = ttk.Button(model_inner, text="Clear", command=self._clear_model)
        clear_btn.pack(side=tk.RIGHT, padx=(0, 5))

        model_info = ttk.Label(model_frame, text="Select a trained PPO model (.zip) or leave empty for random actions",
                               style='Info.TLabel')
        model_info.pack(anchor=tk.W, pady=(5, 0))

        # Parameter settings
        fish_frame = ttk.LabelFrame(main_frame, text="Simulation Parameters", padding="10")
        fish_frame.pack(fill=tk.X, pady=(0, 15))

        num_frame = ttk.Frame(fish_frame)
        num_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(num_frame, text="Number of AI Fish:", width=20).pack(side=tk.LEFT)

        self.num_fish_var = tk.IntVar(value=default_num_fish)
        self.num_fish_scale = ttk.Scale(num_frame, from_=1, to=30, variable=self.num_fish_var,
                                        orient=tk.HORIZONTAL, length=200,
                                        command=lambda v: self.num_fish_label.config(text=str(int(float(v)))))
        self.num_fish_scale.pack(side=tk.LEFT, padx=(10, 10))

        self.num_fish_label = ttk.Label(num_frame, text=str(default_num_fish), width=5)
        self.num_fish_label.pack(side=tk.LEFT)

        mass_frame = ttk.Frame(fish_frame)
        mass_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(mass_frame, text="Initial Mass (g):", width=20).pack(side=tk.LEFT)

        self.mass_var = tk.DoubleVar(value=default_mass)
        self.mass_scale = ttk.Scale(mass_frame, from_=5, to=50, variable=self.mass_var,
                                    orient=tk.HORIZONTAL, length=200,
                                    command=lambda v: self.mass_label.config(text=f"{float(v):.1f}"))
        self.mass_scale.pack(side=tk.LEFT, padx=(10, 10))

        self.mass_label = ttk.Label(mass_frame, text=f"{default_mass:.1f}", width=5)
        self.mass_label.pack(side=tk.LEFT)

        self.include_npc_var = tk.BooleanVar(value=True)
        npc_check = ttk.Checkbutton(fish_frame, text="Include NPC fish (environment fish)",
                                    variable=self.include_npc_var)
        npc_check.pack(anchor=tk.W)

        # Course
        course_frame = ttk.Frame(fish_frame)
        course_frame.pack(fill=tk.X, pady=(10, 5))
        ttk.Label(course_frame, text="Course:", width=20).pack(side=tk.LEFT)
        self.course_var = tk.StringVar(value='course2')
        course_combo = ttk.Combobox(course_frame, textvariable=self.course_var,
                                    values=['course1', 'course2', 'course3', 'course4'],
                                    state='readonly', width=15)
        course_combo.pack(side=tk.LEFT, padx=(10, 0))

        # Layout
        layout_frame = ttk.Frame(fish_frame)
        layout_frame.pack(fill=tk.X, pady=(5, 0))
        ttk.Label(layout_frame, text="Layout (course2):", width=20).pack(side=tk.LEFT)
        self.layout_var = tk.StringVar(value='vertical_maze')
        layout_combo = ttk.Combobox(layout_frame, textvariable=self.layout_var,
                                    values=['vertical_maze', 'corridor', 'reef', 'random'],
                                    state='readonly', width=15)
        layout_combo.pack(side=tk.LEFT, padx=(10, 0))

        # Buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=(20, 0))

        start_btn = ttk.Button(btn_frame, text="Start Simulation", command=self._on_start)
        start_btn.pack(side=tk.RIGHT, padx=(10, 0))

        cancel_btn = ttk.Button(btn_frame, text="Cancel", command=self._on_cancel)
        cancel_btn.pack(side=tk.RIGHT)

        self.root.bind('<Return>', lambda e: self._on_start())
        self.root.bind('<Escape>', lambda e: self._on_cancel())

    def _browse_model(self):
        filetypes = [("PPO Model files", "*.zip"), ("All files", "*.*")]
        initial_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        if not os.path.exists(initial_dir):
            initial_dir = os.path.dirname(os.path.abspath(__file__))

        filename = filedialog.askopenfilename(title="Select AI Model", filetypes=filetypes, initialdir=initial_dir)
        if filename:
            self.model_path = filename
            self.model_var.set(filename)

    def _clear_model(self):
        self.model_path = ""
        self.model_var.set("No model (Random Actions)")

    def _on_start(self):
        self.result = {
            'model_path': self.model_path if self.model_path else None,
            'num_fish': int(self.num_fish_var.get()),
            'initial_mass': float(self.mass_var.get()),
            'include_npc': self.include_npc_var.get(),
            'course': self.course_var.get(),
            'layout': self.layout_var.get()
        }
        self.root.destroy()

    def _on_cancel(self):
        self.result = None
        self.root.destroy()

    def show(self):
        self.root.grab_set()
        self.root.wait_window()
        return self.result


def show_settings_dialog(default_num_fish=10, default_mass=20.0, default_model_path=""):
    dialog = SettingsDialog(
        default_num_fish=default_num_fish,
        default_mass=default_mass,
        default_model_path=default_model_path
    )
    return dialog.show()


# ============================================================
# Academic Color Scheme - Same as original
# ============================================================

class AcademicColors:
    BACKGROUND = (18, 22, 28)
    PANEL_BG = (25, 32, 42)
    PANEL_BORDER = (45, 55, 72)
    PANEL_HEADER = (35, 45, 60)
    WATER_DEEP = (20, 45, 75)
    WATER_SURFACE = (35, 80, 130)
    WATER_DANGER = (80, 35, 40)
    TANK_BORDER = (70, 100, 140)
    TEXT_PRIMARY = (230, 235, 240)
    TEXT_SECONDARY = (160, 170, 185)
    TEXT_DIM = (100, 110, 125)
    TEXT_ACCENT = (100, 180, 255)
    TEXT_WARNING = (255, 180, 80)
    TEXT_SUCCESS = (80, 200, 120)
    TEXT_DANGER = (255, 100, 100)
    FOOD_FLOATING = (255, 210, 80)
    FOOD_SINKING = (255, 160, 50)
    FOOD_SETTLING = (200, 120, 40)
    NPC_SMALL = (100, 180, 220)
    NPC_MEDIUM = (80, 130, 200)
    NPC_LARGE = (220, 100, 60)
    NPC_PREDATOR = (180, 50, 50)
    NPC_CHASING = (255, 60, 60)
    CHART_ENERGY = (255, 180, 80)
    CHART_MASS = (80, 200, 120)
    CHART_SPEED = (100, 180, 255)
    CHART_GRID = (40, 50, 65)
    CHART_BG = (20, 28, 38)
    SELECTED_GLOW = (100, 200, 255)
    HOVER_GLOW = (80, 150, 200)
    SCROLLBAR_BG = (35, 45, 60)
    SCROLLBAR_THUMB = (70, 85, 105)
    HEADING_ARROW = (255, 255, 100)


AI_FISH_COLORS = [
    (50, 255, 120), (255, 120, 120), (80, 220, 220), (255, 220, 80),
    (180, 120, 255), (255, 150, 200), (120, 255, 200), (255, 180, 120),
    (120, 180, 255), (255, 120, 180), (200, 255, 120), (120, 220, 180),
    (255, 200, 150), (150, 150, 255), (220, 180, 255), (255, 255, 120),
    (120, 255, 255), (255, 120, 255), (180, 255, 180), (255, 180, 180),
    (180, 180, 255), (255, 220, 180), (180, 255, 220), (220, 180, 180),
    (180, 220, 255), (255, 200, 220), (200, 255, 200), (220, 220, 180),
    (180, 200, 220), (220, 200, 255),
]


# ============================================================
# Helper Functions
# ============================================================

def mass_to_length(mass: float) -> float:
    """Mass to length conversion -  CONFIG """
    if mass <= 0:
        return 0.003 * CONFIG.length_weight.length_scale

    lw = CONFIG.length_weight
    log_w = np.log10(mass)
    width = lw.transition_width

    W1 = 10 ** (lw.stage1_a + lw.stage1_b * np.log10(lw.stage1_L_max))
    W2 = 10 ** (lw.stage2_a + lw.stage2_b * np.log10(lw.stage2_L_max))
    W3 = 10 ** (lw.stage3_a + lw.stage3_b * np.log10(lw.stage3_L_max))

    def calc_L(a, b):
        log_l = (log_w - a) / b
        return max(10 ** log_l, 0.1)

    L1 = calc_L(lw.stage1_a, lw.stage1_b)
    L2 = calc_L(lw.stage2_a, lw.stage2_b)
    L3 = calc_L(lw.stage3_a, lw.stage3_b)
    L4 = calc_L(lw.adult_a, lw.adult_b)

    def sigmoid(x, center, w):
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


def world_to_screen_top(pos, center, scale):
    x = int(center[0] + pos[0] * scale)
    y = int(center[1] - pos[2] * scale)
    return (x, y)


def world_to_screen_side(pos, left, top, width, height, tank_half_x, tank_depth):
    """tank_half_x: X=radius=width/2"""
    x = int(left + (pos[0] + tank_half_x) / (2 * tank_half_x) * width)
    y = int(top + (-pos[1]) / tank_depth * height)
    return (x, y)


def format_time(steps: int, time_step: float = 0.1, acceleration: float = 300, realtime: bool = False) -> str:
    if realtime:
        sim_seconds = steps * time_step
    else:
        sim_seconds = steps * time_step * acceleration

    if sim_seconds < 60:
        return f"{sim_seconds:.1f}s"
    elif sim_seconds < 3600:
        minutes = int(sim_seconds // 60)
        seconds = int(sim_seconds % 60)
        return f"{minutes}m {seconds}s"
    else:
        hours = int(sim_seconds // 3600)
        minutes = int((sim_seconds % 3600) // 60)
        if hours >= 24:
            days = hours // 24
            hours = hours % 24
            return f"{days}d {hours:02d}h {minutes:02d}m"
        return f"{hours:02d}h {minutes:02d}m"


def truncate_text(text: str, font, max_width: int) -> str:
    if font.size(text)[0] <= max_width:
        return text
    while len(text) > 3 and font.size(text + "...")[0] > max_width:
        text = text[:-1]
    return text + "..."


# ============================================================
# NPC Target Lock System (new)
# ============================================================

@dataclass
class NPCTargetLock:
    """NPC target lock state."""
    target_id: Optional[int] = None  # Locked AI fish ID
    chase_duration: int = 0  # Chase steps elapsed
    lock_cooldown: int = 0  # Target switch cooldown
    last_target_distance: float = float('inf')  # Last target distance

    max_chase_duration: int = 150  # Max chase duration
    target_switch_cooldown: int = 30  # Target switch cooldown
    give_up_distance: float = 0.5  # Give-up distance

    def should_switch_target(self, current_distance: float, target_alive: bool) -> bool:
        """Determine if target should be switched"""
        # Target dead
        if not target_alive:
            return True
        # Chase timeout
        if self.chase_duration >= self.max_chase_duration:
            return True
        # Target fled too far
        if current_distance > self.give_up_distance:
            return True
        # Cannot switch during cooldown
        if self.lock_cooldown > 0:
            return False
        return False

    def lock_target(self, target_id: int):
        """Lock new target"""
        self.target_id = target_id
        self.chase_duration = 0
        self.lock_cooldown = self.target_switch_cooldown

    def update(self):
        """Update per step"""
        if self.target_id is not None:
            self.chase_duration += 1
        if self.lock_cooldown > 0:
            self.lock_cooldown -= 1

    def release(self):
        """Release target"""
        self.target_id = None
        self.chase_duration = 0


# ============================================================
# AI Fish State (Extended)
# ============================================================

@dataclass
class AIFishState:
    """Single AI fish state - """
    id: int
    position: np.ndarray
    velocity: np.ndarray
    body_mass: float
    total_length: float
    energy: float = 80.0
    stomach_fullness: float = 30.0
    activity_state: ActivityState = ActivityState.ACTIVE

    # Metabolism tracking
    rest_duration_steps: int = 0
    current_metabolism_factor: float = 1.0
    current_growth_bonus: float = 1.0
    fatigue: float = 0.0
    stress_level: float = 0.0
    is_digesting: bool = False

    # Buoyancy state
    _buoyancy_state: Optional[BuoyancyState] = None
    air_exposure_time: float = 0.0
    in_air: bool = False
    is_at_surface: bool = False

    # Digestion state
    stomach_content_mass: float = 0.0
    initial_meal_mass: float = 0.0
    digestion_buffer: float = 0.0
    energy_from_digestion: float = 0.0

    # Growth state
    growth_accumulation: float = 0.0
    total_growth_energy: float = 0.0
    growth_count: int = 0
    initial_mass: float = 20.0

    # State switch control
    state_switch_cooldown: float = 0.0
    last_state_switch_step: int = 0
    total_rest_steps: int = 0

    # Statistics
    food_eaten: int = 0
    fish_eaten: int = 0
    ai_fish_eaten: int = 0
    steps_alive: int = 0

    # Life status
    is_alive: bool = True
    death_reason: str = None
    killed_by: int = None

    # Visualization
    position_history: deque = field(default_factory=lambda: deque(maxlen=100))
    color: tuple = (0, 255, 0)

    # Action and physics tracking
    last_action: np.ndarray = field(default_factory=lambda: np.zeros(5))
    last_propulsion_force: float = 0.0
    current_speed: float = 0.0
    buoyancy_control: float = 0.0
    raw_buoyancy_control: float = 0.0
    buoyancy_mode: int = 0  # -1, 0, +1
    last_action_normalized: bool = False
    speed_clamped: bool = False

    # Inertia System
    using_burst: bool = False
    heading: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=np.float32))
    smoothed_action: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    smoothed_buoyancy_control: float = 0.0
    current_turn_rate: float = 0.0
    inertia_initialized: bool = False
    smoothed_action_magnitude: float = 0.0
    turn_rate: float = 0.0

    # History for charts
    energy_history: deque = field(default_factory=lambda: deque(maxlen=200))
    mass_history: deque = field(default_factory=lambda: deque(maxlen=200))
    speed_history: deque = field(default_factory=lambda: deque(maxlen=200))

    # v3 new
    pitch_angle: float = 0.0
    target_pitch_angle: float = 0.0
    current_pitch_rate: float = 0.0
    is_coasting: bool = False
    coast_steps_remaining: int = 0
    current_reynolds: float = 0.0
    rest_transition_progress: float = 0.0
    active_transition_progress: float = 0.0
    drift_direction: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    drift_update_counter: int = 0
    previous_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))

    def to_other_fish(self) -> OtherFish:
        """Convert to OtherFish for injection"""
        if self.body_mass < 10:
            size_cat = FishSize.SMALL
        elif self.body_mass < 30:
            size_cat = FishSize.MEDIUM
        else:
            size_cat = FishSize.LARGE

        return OtherFish(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            body_mass=self.body_mass,
            total_length=self.total_length,
            energy=self.energy,
            is_alive=self.is_alive,
            behavior_type=FishBehaviorType.PASSIVE,
            original_behavior=FishBehaviorType.PASSIVE,
            size_category=size_cat,
            is_chasing=False,
            is_fleeing=False
        )

    def update_history(self):
        self.energy_history.append(self.energy)
        self.mass_history.append(self.body_mass)
        self.speed_history.append(self.current_speed)


# ============================================================
# Multi-AI Fish Manager - Training Aligned v3.1
# ============================================================

class MultiAIFishManager:
    """Multi-AI fish manager - training-aligned version. v3.1"""

    def __init__(self, base_env: BassEnvironment, model, num_fish: int = 10,
                 initial_mass: float = 20.0, include_npc: bool = True):
        self.base_env = base_env
        self.model = model
        self.num_fish = num_fish
        self.initial_mass = initial_mass
        self.include_npc = include_npc

        self.ai_fish: List[AIFishState] = []
        self.colors = AI_FISH_COLORS[:num_fish] if num_fish <= len(AI_FISH_COLORS) else \
            [AI_FISH_COLORS[i % len(AI_FISH_COLORS)] for i in range(num_fish)]

        self.current_step = 0
        self.ai_predation_events = []

        self.feeding_interval = self.base_env.feeding_system.feeding_interval
        self.steps_since_feeding = 0
        self.total_food_spawned = 0
        self.total_food_eaten = 0

        self.realtime_mode = False
        self.realtime_frame_counter = 0

        # ===== Fix1: Stage 7 =====
        self.curriculum_config = self._get_stage7_config()

        # ===== new: NPC target lock state. =====
        self.npc_target_locks: Dict[int, NPCTargetLock] = {}

        try:
            from systems.metabolism import ActivityState
            self.ActivityState = ActivityState
        except ImportError:
            class ActivityState(Enum):
                ACTIVE = "active"
                RESTING = "resting"

            self.ActivityState = ActivityState

    def _get_stage7_config(self) -> Dict[str, Any]:
        """Get Stage 7 real difficulty configuration."""
        return {
            'stage': 10,
            'name': '',
            'capture_multiplier': 1.0,
            'predation_multiplier': 1.0,
            'energy_cost_multiplier': 1.0,
            'food_amount_multiplier': 1.0
        }

    def _discretize_buoyancy_action_for_fish(self, fish: AIFishState, raw_action: float) -> float:
        """//+ """
        enter_th = 0.45
        hold_th = 0.30
        a = float(raw_action)

        if abs(a) <= hold_th:
            fish.buoyancy_mode = 0
        elif a <= -enter_th:
            fish.buoyancy_mode = -1
        elif a >= enter_th:
            fish.buoyancy_mode = 1

        return float(fish.buoyancy_mode)

    def reset(self):
        """Reset all AI fish"""
        self.ai_fish = []
        self.current_step = 0
        self.ai_predation_events = []
        self.steps_since_feeding = 0
        self.total_food_spawned = 0
        self.total_food_eaten = 0
        self.npc_target_locks = {}

        # Reset with random seed to avoid deterministic behavior
        reset_seed = int(np.random.randint(0, 2_147_483_647))
        self.base_env.reset(seed=reset_seed)

        # Get tank geometry and obstacles from environment
        self.tank_geometry = self.base_env.tank_geometry
        self.obstacle_field = self.base_env.obstacle_field
        tank_depth = self.tank_geometry.depth

        for i in range(self.num_fish):
            #  tank_geometry
            position = self.tank_geometry.random_interior_point(margin=0.3)
            position[1] = np.clip(position[1], -tank_depth + 0.1, -0.15)
            angle = 2 * np.pi * i / self.num_fish + np.random.uniform(-0.3, 0.3)
            velocity = np.random.uniform(-0.02, 0.02, 3).astype(np.float32)

            mass = self.initial_mass
            length = mass_to_length(mass)

            fish = AIFishState(
                id=i,
                position=position,
                velocity=velocity,
                body_mass=mass,
                total_length=length,
                energy=np.random.uniform(70, 90),
                stomach_fullness=np.random.uniform(20, 40),
                stomach_content_mass=0.0,
                initial_meal_mass=0.0,
                digestion_buffer=0.0,
                energy_from_digestion=0.0,
                growth_accumulation=0.0,
                total_growth_energy=0.0,
                growth_count=0,
                initial_mass=mass,
                color=self.colors[i % len(self.colors)],
                state_switch_cooldown=0.0,
                last_state_switch_step=0,
                total_rest_steps=0,
                using_burst=False,
                heading=np.array([np.cos(angle), 0.0, np.sin(angle)], dtype=np.float32),
                smoothed_action=np.zeros(3, dtype=np.float32),
                smoothed_buoyancy_control=0.0,
                current_turn_rate=0.0,
                inertia_initialized=False,
            )
            fish.position_history.append(position.copy())
            self.ai_fish.append(fish)

        # Create independent buoyancy state per fish
        buoyancy_sys = self.base_env.physics_system.buoyancy_system
        if buoyancy_sys is not None:
            for fish in self.ai_fish:
                fish._buoyancy_state = create_buoyancy_state()
                buoyancy_sys.initialize(
                    fish._buoyancy_state,
                    fish.body_mass,
                    fish.total_length
                )
                fish.relative_density = fish._buoyancy_state.relative_density
                fish.net_buoyancy_force = fish._buoyancy_state.net_buoyancy_force
                fish.swimbladder_volume = fish._buoyancy_state.swimbladder_volume

        # Initialize NPC target locks
        for idx, npc in enumerate(self.base_env.interaction_state.other_fish):
            if npc.behavior_type in [FishBehaviorType.AGGRESSIVE, FishBehaviorType.SURFACE_PREDATOR]:
                self.npc_target_locks[idx] = NPCTargetLock()

        # Clear food list_spawn_env_food
        self.base_env.feeding_state.food_items = []
        total_mass = self.get_total_mass()

        ext = self.tank_geometry.get_extents()
        print(f"[OK] Initialized {self.num_fish} AI fish (v3.1 Training-Aligned)")
        print(f"     Curriculum: Stage 7 (Real Difficulty)")
        print(f"     Env seed={reset_seed}, tank={self.tank_geometry.shape_name}, obstacles={self.obstacle_field.count}")
        print(f"     Extents: {ext}")
        print(f"     Total mass: {total_mass:.1f}g")

    def update_config(self, num_fish: int, initial_mass: float, model=None, include_npc: bool = True):
        self.num_fish = num_fish
        self.initial_mass = initial_mass
        self.include_npc = include_npc
        if model is not None:
            self.model = model

        self.colors = AI_FISH_COLORS[:num_fish] if num_fish <= len(AI_FISH_COLORS) else \
            [AI_FISH_COLORS[i % len(AI_FISH_COLORS)] for i in range(num_fish)]

    def get_total_mass(self) -> float:
        return sum(f.body_mass for f in self.ai_fish if f.is_alive)

    def _initial_feeding(self, total_mass: float):
        """Initial feeding - optimized: ensures minimum pellets per fish."""
        self.base_env.feeding_state.food_items = []

        fc = CONFIG.feeding
        alive_count = sum(1 for f in self.ai_fish if f.is_alive)

        daily_total = total_mass * fc.daily_feeding_rate
        per_feeding = daily_total / fc.feedings_per_day

        base_floating = (fc.floating_pellets_min + fc.floating_pellets_max) // 2
        base_sinking = (fc.sinking_pellets_min + fc.sinking_pellets_max) // 2

        # 2-3
        min_pellets_per_fish = 2.5
        min_total_pellets = int(alive_count * min_pellets_per_fish)

        base_mass = 20.0
        if total_mass > base_mass * 1.5:
            mass_factor = min(3.0, total_mass / base_mass)
            num_floating = int(base_floating * mass_factor)
            num_sinking = int(base_sinking * mass_factor)
        else:
            num_floating = base_floating
            num_sinking = base_sinking

        total_pellets = num_floating + num_sinking
        if total_pellets < min_total_pellets:
            ratio = num_floating / (num_floating + num_sinking) if total_pellets > 0 else 0.6
            num_floating = int(min_total_pellets * ratio)
            num_sinking = min_total_pellets - num_floating

        max_floating = 30
        max_sinking = 20
        num_floating = min(num_floating, max_floating)
        num_sinking = min(num_sinking, max_sinking)

        total_pellets = num_floating + num_sinking
        if total_pellets == 0:
            total_pellets = 1
            num_floating = 1

        pellet_mass = np.clip(per_feeding / total_pellets,
                              fc.pellet_mass_min, fc.pellet_mass_max)

        feed_input = FeedingInput(
            agent_position=np.array([0.0, -0.3, 0.0]),
            agent_mass=total_mass,
            agent_length=0.1,
            stomach_fullness=0.0,
            tank_geometry=self.tank_geometry,
            obstacle_field=self.obstacle_field
        )

        for _ in range(num_floating):
            food = self.base_env.feeding_system._create_floating_pellet(feed_input, pellet_mass)
            food.age = np.random.randint(0, 50)
            self.base_env.feeding_state.food_items.append(food)

        for _ in range(num_sinking):
            food = self.base_env.feeding_system._create_sinking_pellet(feed_input, pellet_mass)
            self.base_env.feeding_state.food_items.append(food)

        self.total_food_spawned = total_pellets

        print(f"[FEED] Initial: {num_floating} floating + {num_sinking} sinking = {total_pellets} pellets")
        print(f"       ({total_pellets / alive_count:.1f} pellets/fish)")

    def _spawn_food(self):
        """Scheduled feeding - optimized."""
        total_mass = self.get_total_mass()
        if total_mass <= 0:
            return 0

        alive_count = sum(1 for f in self.ai_fish if f.is_alive)
        if alive_count == 0:
            return 0

        fc = CONFIG.feeding
        daily_total = total_mass * fc.daily_feeding_rate
        per_feeding = daily_total / fc.feedings_per_day

        base_floating = np.random.randint(fc.floating_pellets_min, fc.floating_pellets_max + 1)
        base_sinking = np.random.randint(fc.sinking_pellets_min, fc.sinking_pellets_max + 1)

        min_pellets_per_fish = 2.0
        min_total_pellets = int(alive_count * min_pellets_per_fish)

        base_mass = 20.0
        if total_mass > base_mass * 1.5:
            mass_factor = min(3.0, total_mass / base_mass)
            num_floating = int(base_floating * mass_factor)
            num_sinking = int(base_sinking * mass_factor)
        else:
            num_floating = base_floating
            num_sinking = base_sinking

        total_pellets = num_floating + num_sinking
        if total_pellets < min_total_pellets:
            ratio = num_floating / (num_floating + num_sinking) if total_pellets > 0 else 0.6
            num_floating = int(min_total_pellets * ratio)
            num_sinking = min_total_pellets - num_floating

        num_floating = min(num_floating, 35)
        num_sinking = min(num_sinking, 25)

        total_pellets = num_floating + num_sinking
        if total_pellets == 0:
            return 0

        pellet_mass = np.clip(per_feeding / total_pellets,
                              fc.pellet_mass_min, fc.pellet_mass_max)

        feed_input = FeedingInput(
            agent_position=np.array([0.0, -0.3, 0.0]),
            agent_mass=total_mass,
            agent_length=0.1,
            stomach_fullness=0.0,
            tank_geometry=self.tank_geometry,
            obstacle_field=self.obstacle_field
        )

        for _ in range(num_floating):
            food = self.base_env.feeding_system._create_floating_pellet(feed_input, pellet_mass)
            self.base_env.feeding_state.food_items.append(food)

        for _ in range(num_sinking):
            food = self.base_env.feeding_system._create_sinking_pellet(feed_input, pellet_mass)
            self.base_env.feeding_state.food_items.append(food)

        self.total_food_spawned += total_pellets
        return total_pellets

    def _inject_other_ai_fish(self, current_fish_id: int):
        """Inject other AI fish into interaction system"""
        original_fish = self.base_env.interaction_state.other_fish.copy()

        for fish in self.ai_fish:
            if fish.id != current_fish_id and fish.is_alive:
                other_fish_obj = fish.to_other_fish()
                other_fish_obj._is_injected_ai = True  #
                self.base_env.interaction_state.other_fish.append(other_fish_obj)

        return original_fish

    def _restore_original_fish(self, original_fish: List):
        self.base_env.interaction_state.other_fish = original_fish

    def _sync_env_state(self, fish: AIFishState):
        """Sync AI fish state TO environment state - """
        self.base_env.physics_state.position = fish.position.copy()
        self.base_env.physics_state.velocity = fish.velocity.copy()
        self.base_env.physics_state.using_burst = fish.using_burst

        # Inertia System
        if hasattr(self.base_env.physics_state, 'heading'):
            self.base_env.physics_state.heading = fish.heading.copy()
        if hasattr(self.base_env.physics_state, 'smoothed_action'):
            self.base_env.physics_state.smoothed_action = fish.smoothed_action.copy()
        if hasattr(self.base_env.physics_state, 'smoothed_buoyancy_control'):
            self.base_env.physics_state.smoothed_buoyancy_control = fish.smoothed_buoyancy_control
        if hasattr(self.base_env.physics_state, 'current_turn_rate'):
            self.base_env.physics_state.current_turn_rate = fish.current_turn_rate
        if hasattr(self.base_env.physics_state, 'inertia_initialized'):
            self.base_env.physics_state.inertia_initialized = fish.inertia_initialized
        if hasattr(self.base_env.physics_state, 'previous_velocity'):
            self.base_env.physics_state.previous_velocity = fish.velocity.copy()

        ms = self.base_env.metabolism_state
        ms.energy = fish.energy
        ms.stomach_fullness = fish.stomach_fullness
        ms.stomach_content_mass = fish.stomach_content_mass
        ms.initial_meal_mass = fish.initial_meal_mass
        ms.digestion_buffer = fish.digestion_buffer
        ms.energy_from_digestion = fish.energy_from_digestion
        ms.activity_state = fish.activity_state
        ms.rest_duration_steps = fish.rest_duration_steps
        ms.current_metabolism_factor = fish.current_metabolism_factor
        ms.current_growth_bonus = fish.current_growth_bonus
        ms.fatigue = fish.fatigue
        ms.stress_level = fish.stress_level
        ms.is_digesting = fish.is_digesting
        ms.state_switch_cooldown = fish.state_switch_cooldown
        ms.last_state_switch_step = fish.last_state_switch_step
        ms.total_rest_steps = fish.total_rest_steps

        gs = self.base_env.growth_state
        gs.body_mass = fish.body_mass
        gs.total_length = fish.total_length
        gs.growth_accumulation = fish.growth_accumulation
        gs.total_growth_energy = fish.total_growth_energy
        gs.growth_count = fish.growth_count
        gs.initial_mass = fish.initial_mass

        if fish._buoyancy_state is not None:
            self.base_env.physics_state.buoyancy_state = fish._buoyancy_state
            self.base_env.physics_state.buoyancy_initialized = True

        # v3
        ps = self.base_env.physics_state
        ps.pitch_angle = fish.pitch_angle
        ps.target_pitch_angle = fish.target_pitch_angle
        ps.current_pitch_rate = fish.current_pitch_rate
        ps.is_coasting = fish.is_coasting
        ps.coast_steps_remaining = fish.coast_steps_remaining
        ps.current_reynolds = fish.current_reynolds
        ps.rest_transition_progress = fish.rest_transition_progress
        ps.active_transition_progress = fish.active_transition_progress
        ps.drift_direction = fish.drift_direction.copy()
        ps.drift_update_counter = fish.drift_update_counter
        ps.previous_velocity = fish.previous_velocity.copy()  #
        ps.activity_state = fish.activity_state
        ps.air_exposure_time = fish.air_exposure_time
        ps.in_air = fish.in_air
        ps.is_at_surface = fish.is_at_surface

    def _get_observation(self, fish: AIFishState) -> np.ndarray:
        self._sync_env_state(fish)
        original_fish = self._inject_other_ai_fish(fish.id)

        perception_input = self.base_env._create_perception_input()
        self.base_env.perception_system.update(
            self.base_env.perception_state, perception_input
        )

        obs = self.base_env._get_observation()
        self._restore_original_fish(original_fish)

        return obs

    def _update_npc_fish(self):
        """
        Update NPC fish AI - v3.1 Fix

        Fix
        1. ✅
        2. ✅ Using CONFIG parameters
        3. ✅ Aligned with training env
        """
        if not self.include_npc:
            return

        alive_ai = [f for f in self.ai_fish if f.is_alive]
        if not alive_ai:
            return

        tank_depth = self.tank_geometry.depth
        extents = self.tank_geometry.get_extents()
        tank_radius = extents.get('radius', extents.get('width', 3.0) / 2)
        dt = CONFIG.environment.time_step
        ic = CONFIG.interaction

        # ===== Fix1: ！=====
        for npc in self.base_env.interaction_state.other_fish:
            if npc.is_alive:
                npc.update_cooldown()  # ← ！

        for idx, npc in enumerate(self.base_env.interaction_state.other_fish):
            if not npc.is_alive:
                continue

            npc_behavior = npc.behavior_type.value if hasattr(npc.behavior_type, 'value') else str(npc.behavior_type)
            base_speed = calculate_sustained_speed(npc.total_length)

            # ============  ============
            if npc_behavior == 'surface':
                self._update_surface_predator_with_lock(
                    npc, idx, alive_ai, base_speed, dt, tank_radius, ic
                )

            # ============  ============
            elif npc_behavior == 'aggressive':
                self._update_aggressive_fish_with_lock(
                    npc, idx, alive_ai, base_speed, dt, tank_radius, tank_depth, ic
                )

            # ============ / ============
            else:
                self._update_passive_fish_fleeing(
                    npc, alive_ai, base_speed, dt, tank_radius, tank_depth, ic
                )

    def _update_surface_predator_with_lock(self, npc: OtherFish, npc_idx: int,
                                           alive_ai: List[AIFishState],
                                           base_speed: float, dt: float,
                                           tank_radius: float, ic):
        """
        Update surface predator - FixUsing CONFIG parameters
        """
        cfg = CONFIG.surface_predator  # ←  CONFIG

        # Target lock state
        if npc_idx not in self.npc_target_locks:
            self.npc_target_locks[npc_idx] = NPCTargetLock(
                max_chase_duration=200,
                give_up_distance=cfg.detection_range,
                target_switch_cooldown=30
            )
        lock = self.npc_target_locks[npc_idx]
        lock.update()

        # ===== Consistent with training env=====
        prey_detection_depth = cfg.prey_detection_depth  # ←  CONFIG
        give_up_depth = cfg.give_up_depth
        vertical_chase_limit = cfg.vertical_chase_limit
        direction_noise = cfg.direction_noise

        # ===== No chase during cooldown（Using training env logic）=====
        if not npc.can_attack():
            npc.is_chasing = False
            lock.release()
            self._patrol_surface(npc, base_speed, cfg.patrol_speed_multiplier)
            npc.position += npc.velocity * dt
            self._enforce_surface_predator_boundaries(npc, tank_radius, cfg)
            return

        # ===== Helper: horizontal (X-Z) distance =====
        def horizontal_distance(pos1, pos2):
            return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[2] - pos2[2]) ** 2)

        # ===== Find or validate target =====
        target_fish = None
        target_h_distance = float('inf')

        if lock.target_id is not None:
            for ai_fish in alive_ai:
                if ai_fish.id == lock.target_id and ai_fish.is_alive:
                    in_detection = ai_fish.position[1] > -prey_detection_depth
                    not_too_deep = ai_fish.position[1] > -give_up_depth

                    if not in_detection or not not_too_deep:
                        lock.release()
                        break

                    target_fish = ai_fish
                    target_h_distance = horizontal_distance(npc.position, ai_fish.position)

                    if target_h_distance > cfg.detection_range:
                        lock.release()
                        target_fish = None
                    break

            if lock.target_id is not None and target_fish is None:
                lock.release()

        # ===== Search for new target =====
        if target_fish is None and lock.lock_cooldown <= 0:
            best_target = None
            best_distance = float('inf')

            for ai_fish in alive_ai:
                if ai_fish.position[1] < -prey_detection_depth:
                    continue

                # size ratio
                size_ratio = npc.body_mass / ai_fish.body_mass
                if size_ratio < ic.threat_size_ratio:
                    continue

                h_dist = horizontal_distance(npc.position, ai_fish.position)

                if h_dist < cfg.detection_range and h_dist < best_distance:
                    best_distance = h_dist
                    best_target = ai_fish

            if best_target is not None:
                lock.lock_target(best_target.id)
                target_fish = best_target
                target_h_distance = best_distance

        # ===== Execute chase or patrol =====
        if target_fish is not None:
            npc.is_chasing = True

            # Calculate chase direction
            direction = target_fish.position - npc.position

            # Limit vertical chase amplitude（Consistent with training env）
            direction[1] = np.clip(direction[1], -vertical_chase_limit, vertical_chase_limit)

            # Normalize
            dist = np.linalg.norm(direction)
            if dist > 0.01:
                direction = direction / dist

            # Add random noise（Consistent with training env）
            if direction_noise > 0:
                noise = np.random.uniform(-direction_noise, direction_noise, 3)
                noise[1] *= 0.3
                direction = direction + noise
                direction = direction / (np.linalg.norm(direction) + 1e-6)

            chase_speed = base_speed * cfg.chase_speed_multiplier
            npc.velocity = direction * chase_speed
        else:
            npc.is_chasing = False
            self._patrol_surface(npc, base_speed, cfg.patrol_speed_multiplier)

        # Speed limit
        max_speed = base_speed * cfg.chase_speed_multiplier
        speed = np.linalg.norm(npc.velocity)
        if speed > max_speed:
            npc.velocity = npc.velocity / speed * max_speed

        npc.position += npc.velocity * dt
        self._enforce_surface_predator_boundaries(npc, tank_radius, cfg)

    def _update_aggressive_fish_with_lock(self, npc: OtherFish, npc_idx: int,
                                          alive_ai: List[AIFishState],
                                          base_speed: float, dt: float,
                                          tank_radius: float, tank_depth: float, ic):
        """
        Update aggressive fish - FixUsing CONFIG parameters
        """
        cfg = CONFIG.aggressive_behavior  # ←  CONFIG

        if npc_idx not in self.npc_target_locks:
            self.npc_target_locks[npc_idx] = NPCTargetLock(
                max_chase_duration=150,
                give_up_distance=cfg.give_up_range  # ←  CONFIG
            )
        lock = self.npc_target_locks[npc_idx]
        lock.update()

        # ===== Attack cooldown active（Using training env logic）=====
        if not npc.can_attack():
            npc.is_chasing = False
            lock.release()
            self._random_swim(npc, base_speed, cfg.random_speed_multiplier)
            npc.position += npc.velocity * dt
            self._enforce_fish_boundaries(npc, tank_radius, tank_depth)
            return

        # Find or validate target
        target_fish = None
        target_distance = float('inf')

        if lock.target_id is not None:
            for ai_fish in alive_ai:
                if ai_fish.id == lock.target_id and ai_fish.is_alive:
                    target_fish = ai_fish
                    target_distance = np.linalg.norm(npc.position - ai_fish.position)
                    break

            if target_fish is None or lock.should_switch_target(target_distance, target_fish is not None):
                lock.release()
                target_fish = None

        # Search for new target
        if target_fish is None and lock.lock_cooldown <= 0:
            best_target = None
            best_distance = float('inf')

            for ai_fish in alive_ai:
                size_ratio = npc.body_mass / ai_fish.body_mass
                if size_ratio < ic.threat_size_ratio:
                    continue

                dist = np.linalg.norm(npc.position - ai_fish.position)
                if dist < cfg.detection_range and dist < best_distance:
                    best_distance = dist
                    best_target = ai_fish

            if best_target is not None:
                lock.lock_target(best_target.id)
                target_fish = best_target
                target_distance = best_distance

        # Execute chase or random swim
        if target_fish is not None:
            npc.is_chasing = True

            direction = target_fish.position - npc.position
            dist = np.linalg.norm(direction)
            if dist > 0.01:
                direction = direction / dist

            # Add slight random noise
            direction += np.random.uniform(-cfg.direction_noise * 0.5, cfg.direction_noise * 0.5, 3)
            direction = direction / (np.linalg.norm(direction) + 1e-6)

            chase_speed = base_speed * cfg.chase_speed_multiplier
            npc.velocity = direction * chase_speed
        else:
            npc.is_chasing = False
            self._random_swim(npc, base_speed, cfg.random_speed_multiplier)

        # Speed limit
        max_speed = base_speed * cfg.chase_speed_multiplier
        speed = np.linalg.norm(npc.velocity)
        if speed > max_speed:
            npc.velocity = npc.velocity / speed * max_speed

        npc.position += npc.velocity * dt
        self._enforce_fish_boundaries(npc, tank_radius, tank_depth)

    def _update_passive_fish_fleeing(self, npc: OtherFish, alive_ai: List[AIFishState],
                                     base_speed: float, dt: float,
                                     tank_radius: float, tank_depth: float, ic):
        """Update passive fish (with fleeing)"""
        cfg = FLEEING_CONFIG

        # AI
        nearest_threat = None
        min_dist = float('inf')

        for ai_fish in alive_ai:
            size_ratio = ai_fish.body_mass / npc.body_mass
            if size_ratio < ic.min_predation_size_ratio:
                continue

            dist = np.linalg.norm(npc.position - ai_fish.position)
            if dist < min_dist:
                min_dist = dist
                nearest_threat = ai_fish

        if nearest_threat is not None and min_dist < cfg.detection_range:
            if not npc.is_fleeing:
                npc.is_fleeing = True
                npc.flee_duration = 0

            npc.flee_duration += 1

            direction = npc.position - nearest_threat.position
            dist = np.linalg.norm(direction)
            if dist > 0.01:
                direction = direction / dist
            else:
                direction = np.random.uniform(-1, 1, 3)
                direction = direction / (np.linalg.norm(direction) + 1e-6)

            direction += np.random.uniform(-cfg.direction_noise, cfg.direction_noise, 3)
            direction = direction / (np.linalg.norm(direction) + 1e-6)

            flee_speed = base_speed * cfg.flee_speed_multiplier
            npc.velocity = direction * flee_speed

        elif npc.is_fleeing:
            should_stop = (
                    min_dist > cfg.safe_distance or
                    npc.flee_duration > cfg.flee_duration_max or
                    nearest_threat is None
            )
            if should_stop:
                npc.is_fleeing = False
                npc.flee_duration = 0
        else:
            npc.velocity += np.random.uniform(-0.005, 0.005, 3).astype(np.float32)
            npc.velocity[1] *= 0.5
            speed = np.linalg.norm(npc.velocity)
            if speed > base_speed:
                npc.velocity = npc.velocity / speed * base_speed

        npc.position += npc.velocity * dt
        self._enforce_fish_boundaries(npc, tank_radius, tank_depth)

    def _random_swim(self, npc: OtherFish, base_speed: float, speed_mult: float):
        """Random swimming"""
        npc.random_direction_timer = getattr(npc, 'random_direction_timer', 0) + 1
        if npc.random_direction_timer > 50:
            npc.random_direction_timer = 0
            angle = np.random.uniform(0, 2 * np.pi)
            npc.velocity = np.array([
                np.cos(angle),
                np.random.uniform(-0.1, 0.1),
                np.sin(angle)
            ], dtype=np.float32)
            npc.velocity *= base_speed * speed_mult

        npc.velocity += np.random.uniform(-0.003, 0.003, 3).astype(np.float32)

    def _patrol_surface(self, npc: OtherFish, base_speed: float, speed_mult: float):
        """Surface patrol"""
        npc.random_direction_timer = getattr(npc, 'random_direction_timer', 0) + 1
        if npc.random_direction_timer > 60:
            npc.random_direction_timer = 0
            angle = np.random.uniform(0, 2 * np.pi)
            npc.velocity = np.array([
                np.cos(angle),
                np.random.uniform(-0.05, 0.05),
                np.sin(angle)
            ], dtype=np.float32)
            npc.velocity *= base_speed * speed_mult

        npc.velocity *= 0.98

    def _enforce_fish_boundaries(self, npc: OtherFish, tank_radius: float, tank_depth: float):
        tank_geo = getattr(self, 'tank_geometry', None)
        obs_field = getattr(self, 'obstacle_field', None)

        if tank_geo is not None:
            npc.position, npc.velocity, _ = tank_geo.enforce_boundary(
                npc.position, npc.velocity
            )
        else:
            # Fallback logic
            fc = CONFIG.feeding
            horizontal_dist = np.sqrt(npc.position[0] ** 2 + npc.position[2] ** 2)
            max_radius = tank_radius - fc.boundary_buffer
            if horizontal_dist > max_radius:
                factor = max_radius / horizontal_dist
                npc.position[0] *= factor
                npc.position[2] *= factor
                npc.velocity[0] *= -0.5
                npc.velocity[2] *= -0.5

            bottom_limit = -tank_depth + fc.bottom_buffer
            if npc.position[1] < bottom_limit:
                npc.position[1] = bottom_limit
                npc.velocity[1] = abs(npc.velocity[1]) * 0.5
            surface_limit = -fc.surface_buffer
            if npc.position[1] > surface_limit:
                npc.position[1] = surface_limit
                npc.velocity[1] = -abs(npc.velocity[1]) * 0.5

        # Obstacle collision
        if obs_field is not None:
            col = obs_field.check_collision(npc.position)
            if col.collided:
                npc.position = col.pushed_position.copy()
                npc.velocity = obs_field.resolve_collision_velocity(
                    npc.velocity, col.normal
                )

    def _enforce_surface_predator_boundaries(self, npc: OtherFish, tank_radius: float, cfg):
        tank_geo = getattr(self, 'tank_geometry', None)
        obs_field = getattr(self, 'obstacle_field', None)

        if tank_geo is not None:
            npc.position, npc.velocity, _ = tank_geo.enforce_boundary(
                npc.position, npc.velocity
            )
        else:
            horizontal_dist = np.sqrt(npc.position[0] ** 2 + npc.position[2] ** 2)
            max_radius = tank_radius * 0.9
            if horizontal_dist > max_radius:
                factor = max_radius / horizontal_dist
                npc.position[0] *= factor
                npc.position[2] *= factor
                npc.velocity[0] *= -0.5
                npc.velocity[2] *= -0.5

        # Depth limit unchanged
        if npc.position[1] < -cfg.surface_zone_depth:
            npc.position[1] = -cfg.surface_zone_depth
            npc.velocity[1] = abs(npc.velocity[1]) * 0.3
        if npc.position[1] > cfg.surface_zone_max:
            npc.position[1] = cfg.surface_zone_max
            npc.velocity[1] = -abs(npc.velocity[1]) * 0.3

        if obs_field is not None:
            col = obs_field.check_collision(npc.position)
            if col.collided:
                npc.position = col.pushed_position.copy()
                npc.velocity = obs_field.resolve_collision_velocity(
                    npc.velocity, col.normal
                )

    def _update_food_movement(self):
        """Update food movement"""
        tank_depth = self.tank_geometry.depth if self.tank_geometry is not None else CONFIG.environment.tank_depth
        tank_radius = CONFIG.environment.tank_radius
        if self.tank_geometry is not None:
            extents = self.tank_geometry.get_extents()
            if self.tank_geometry.shape_name == 'circular':
                tank_radius = extents['radius']
            else:
                tank_radius = max(extents['width'], extents['length']) * 0.5

        feed_input = FeedingInput(
            agent_position=np.array([0.0, -0.5, 0.0]),
            agent_mass=self.get_total_mass(),
            agent_length=0.1,
            stomach_fullness=0.0,
            tank_radius=tank_radius,
            tank_depth=tank_depth,
            tank_geometry=self.tank_geometry,
            obstacle_field=self.obstacle_field
        )

        feeding_system = self.base_env.feeding_system
        state = self.base_env.feeding_state

        feeding_system._update_food_movement(state, feed_input)
        feeding_system._spawn_env_food(state, feed_input)

    def _execute_action(self, fish: AIFishState, action: np.ndarray):
        """Execute action - training-aligned version v3.1"""
        movement_action = action[:3].copy()
        state_action = action[3] if len(action) > 3 else 0.0
        raw_buoyancy_action = action[4] if len(action) > 4 else 0.0
        buoyancy_action = self._discretize_buoyancy_action_for_fish(fish, raw_buoyancy_action)

        fish.last_action = action.copy()
        fish.buoyancy_control = buoyancy_action
        fish.raw_buoyancy_control = float(raw_buoyancy_action)

        # action[0]clip[0,1]environment.py
        movement_action[0] = float(np.clip(movement_action[0], 0.0, 1.0))
        # Normalizeenvironment.py
        fish.last_action_normalized = False

        # ===== Fix2: Activity state threshold aligned (< 0 ) =====
        if state_action < 0:  #  < -0.3
            requested_state = ActivityState.RESTING
        else:
            requested_state = ActivityState.ACTIVE

        # Physics update
        physics_output = None
        try:
            physics_output = self.base_env._update_physics(
                movement_action, requested_state, buoyancy_action
            )
            if hasattr(physics_output, 'propulsion_magnitude'):
                fish.last_propulsion_force = physics_output.propulsion_magnitude
            else:
                fish.last_propulsion_force = np.linalg.norm(movement_action) * 0.1
        except Exception as e:
            print(f"[physics fallback] fish#{fish.id}: {e}")
            self._simple_physics_update(fish, movement_action, requested_state)

        # NPC
        self._check_npc_predation(fish)

        # Metabolism update
        metabolism_output = None
        try:
            buoyancy_energy = 0.0
            if physics_output and hasattr(physics_output, 'buoyancy_energy_consumed'):
                buoyancy_energy = physics_output.buoyancy_energy_consumed

            from systems import MetabolismInput
            metabolism_input = MetabolismInput(
                body_mass=self.base_env.growth_state.body_mass,
                action_magnitude=np.linalg.norm(movement_action),
                is_burst_swimming=self.base_env.physics_state.using_burst,
                velocity_magnitude=np.linalg.norm(self.base_env.physics_state.velocity),
                requested_activity_state=requested_state,
                current_step=self.current_step,
                buoyancy_energy_cost=buoyancy_energy
            )
            metabolism_output = self.base_env.metabolism_system.update(
                self.base_env.metabolism_state,
                metabolism_input,
                curriculum_multiplier=self.curriculum_config['energy_cost_multiplier'],
                growth_state=self.base_env.growth_state
            )
        except Exception as e:
            speed = np.linalg.norm(self.base_env.physics_state.velocity)
            if requested_state == ActivityState.ACTIVE:
                energy_cost = 0.015 + speed * 0.05
            else:
                energy_cost = 0.003
            self.base_env.metabolism_state.energy -= energy_cost

        # Growth update
        if metabolism_output and hasattr(metabolism_output, 'growth_energy') and metabolism_output.growth_energy > 0:
            try:
                self.base_env.growth_system.add_growth_energy(
                    self.base_env.growth_state, metabolism_output.growth_energy
                )
            except:
                pass

        try:
            growth_output = self.base_env.growth_system.process_growth(self.base_env.growth_state)
            if growth_output and growth_output.grew:
                print(
                    f"[GROWTH] AI#{fish.id} grew! Mass: {fish.body_mass:.1f}g -> {self.base_env.growth_state.body_mass:.1f}g")
        except:
            pass

        # Sync physics state
        fish.position = self.base_env.physics_state.position.copy()
        fish.velocity = self.base_env.physics_state.velocity.copy()
        fish.using_burst = self.base_env.physics_state.using_burst
        ps = self.base_env.physics_state
        fish.pitch_angle = ps.pitch_angle
        fish.target_pitch_angle = ps.target_pitch_angle
        fish.current_pitch_rate = ps.current_pitch_rate
        fish.is_coasting = ps.is_coasting
        fish.coast_steps_remaining = ps.coast_steps_remaining
        fish.current_reynolds = ps.current_reynolds
        fish.rest_transition_progress = ps.rest_transition_progress
        fish.active_transition_progress = ps.active_transition_progress
        fish.drift_direction = ps.drift_direction.copy()
        fish.drift_update_counter = ps.drift_update_counter
        fish.previous_velocity = ps.previous_velocity.copy()
        fish.air_exposure_time = ps.air_exposure_time
        fish.in_air = ps.in_air
        fish.is_at_surface = ps.is_at_surface

        # Inertia System
        if hasattr(self.base_env.physics_state, 'heading'):
            fish.heading = self.base_env.physics_state.heading.copy()
        if hasattr(self.base_env.physics_state, 'smoothed_action'):
            fish.smoothed_action = self.base_env.physics_state.smoothed_action.copy()
        if hasattr(self.base_env.physics_state, 'smoothed_buoyancy_control'):
            fish.smoothed_buoyancy_control = self.base_env.physics_state.smoothed_buoyancy_control
        if hasattr(self.base_env.physics_state, 'current_turn_rate'):
            fish.current_turn_rate = self.base_env.physics_state.current_turn_rate
        if hasattr(self.base_env.physics_state, 'inertia_initialized'):
            fish.inertia_initialized = self.base_env.physics_state.inertia_initialized

        if physics_output:
            if hasattr(physics_output, 'smoothed_action_magnitude'):
                fish.smoothed_action_magnitude = physics_output.smoothed_action_magnitude
            if hasattr(physics_output, 'turn_rate_deg_s'):
                fish.turn_rate = physics_output.turn_rate_deg_s

        #  physics.py Speed Limiting
        fish.speed_clamped = False
        fish.current_speed = np.linalg.norm(fish.velocity)

        gs = self.base_env.growth_state
        fish.body_mass = gs.body_mass
        fish.total_length = gs.total_length
        fish.growth_accumulation = gs.growth_accumulation
        fish.total_growth_energy = gs.total_growth_energy
        fish.growth_count = gs.growth_count

        ms = self.base_env.metabolism_state
        fish.energy = ms.energy
        fish.stomach_fullness = ms.stomach_fullness
        fish.stomach_content_mass = ms.stomach_content_mass
        fish.initial_meal_mass = ms.initial_meal_mass
        fish.digestion_buffer = ms.digestion_buffer
        fish.energy_from_digestion = ms.energy_from_digestion
        fish.activity_state = ms.activity_state
        fish.rest_duration_steps = ms.rest_duration_steps
        fish.current_metabolism_factor = ms.current_metabolism_factor
        fish.current_growth_bonus = ms.current_growth_bonus
        fish.fatigue = ms.fatigue
        fish.stress_level = ms.stress_level
        fish.is_digesting = ms.is_digesting
        fish.state_switch_cooldown = ms.state_switch_cooldown
        fish.last_state_switch_step = ms.last_state_switch_step
        fish.total_rest_steps = ms.total_rest_steps

        if fish._buoyancy_state is not None:
            fish.relative_density = fish._buoyancy_state.relative_density
            fish.net_buoyancy_force = fish._buoyancy_state.net_buoyancy_force
            fish.swimbladder_volume = fish._buoyancy_state.swimbladder_volume

        fish.position_history.append(fish.position.copy())
        fish.update_history()

        # Death detection
        if fish.energy <= 0:
            fish.is_alive = False
            fish.death_reason = 'starvation'
            print(f"[DEATH] AI#{fish.id} starved (energy=0)")

    def _calculate_capture_radius(self, body_length: float) -> float:
        """Calculate capture radius - using training env function"""
        return calculate_capture_radius(body_length)

    def _check_food_capture(self, fish: AIFishState):
        """Check food capture with LoS + mouth cone + sweep hit."""
        if fish.activity_state != ActivityState.ACTIVE:
            return

        food_items = self.base_env.feeding_state.food_items
        capture_radius = self._calculate_capture_radius(fish.total_length)
        stomach_capacity = fish.body_mass * CONFIG.feeding.stomach_capacity_ratio
        available_space = stomach_capacity - fish.stomach_content_mass
        dt = CONFIG.environment.time_step
        prev_pos = fish.position - fish.velocity * dt
        heading = fish.heading if np.linalg.norm(fish.heading) > 1e-6 else None
        feeding_system = self.base_env.feeding_system
        obs_field = getattr(self, 'obstacle_field', None)

        consumed_indices = []
        mass_consumed = 0.0

        for i, food in enumerate(food_items):
            if available_space - mass_consumed < food.mass:
                continue

            if obs_field is not None and not obs_field.check_line_of_sight(fish.position, food.position):
                continue

            distance = np.linalg.norm(food.position - fish.position)
            in_mouth_cone = feeding_system._is_in_mouth_cone(
                fish.position, food.position, heading
            )
            direct_hit = distance <= capture_radius and in_mouth_cone

            sweep_dist = feeding_system._distance_point_to_segment_3d(
                food.position, prev_pos, fish.position
            )
            sweep_hit = sweep_dist <= capture_radius * 0.85 and in_mouth_cone

            if direct_hit or sweep_hit:
                consumed_indices.append(i)
                mass_consumed += food.mass
                fish.food_eaten += 1
                self.total_food_eaten += 1

        for i in reversed(consumed_indices):
            self.base_env.feeding_state.food_items.pop(i)

        if mass_consumed > 0:
            fish.stomach_content_mass += mass_consumed
            fish.initial_meal_mass += mass_consumed
            fish.is_digesting = True

            stomach_capacity = fish.body_mass * CONFIG.feeding.stomach_capacity_ratio
            fish.stomach_fullness = (fish.stomach_content_mass / stomach_capacity) * 100

            # Sync to base env metabolism state
            ms = self.base_env.metabolism_state
            ms.stomach_content_mass = fish.stomach_content_mass
            ms.stomach_fullness = fish.stomach_fullness
            ms.initial_meal_mass = fish.initial_meal_mass
            ms.is_digesting = True

    def _check_npc_predation(self, fish: AIFishState) -> Dict:
        """AI fish preying on NPC - """
        result = {'success': 0, 'mass_gained': 0.0}

        if fish.activity_state != ActivityState.ACTIVE:
            return result

        capture_radius = self._calculate_capture_radius(fish.total_length)
        strike_range = calculate_strike_range(fish.total_length)
        stomach_capacity = fish.body_mass * CONFIG.feeding.stomach_capacity_ratio
        available_space = stomach_capacity - fish.stomach_content_mass

        if available_space < 0.5:
            return result

        min_ratio = CONFIG.interaction.min_predation_size_ratio

        for npc in self.base_env.interaction_state.other_fish[:]:
            if not npc.is_alive:
                continue

            if npc.behavior_type.value in ['aggressive', 'surface']:
                continue

            # Skip injected AI fishcannibalism handled by _check_ai_predation
            if hasattr(npc, '_is_injected_ai'):
                continue

            size_ratio = fish.body_mass / npc.body_mass
            if size_ratio < min_ratio:
                continue

            distance = np.linalg.norm(npc.position - fish.position)
            if distance > strike_range:
                continue

            digestible_mass = npc.body_mass * 0.7
            if digestible_mass > available_space:
                continue

            # ===== Aligned with training env =====
            success_prob = self._calculate_predation_success_prob(
                size_ratio, distance, fish, npc
            )

            # Apply curriculum multiplier
            success_prob *= self.curriculum_config['predation_multiplier']

            if np.random.random() < success_prob:
                npc.is_alive = False

                fish.stomach_content_mass += digestible_mass
                fish.stomach_fullness = (fish.stomach_content_mass / stomach_capacity) * 100
                available_space -= digestible_mass

                fish.fish_eaten += 1
                result['success'] += 1
                result['mass_gained'] += digestible_mass

                self.base_env.metabolism_state.stomach_content_mass = fish.stomach_content_mass
                self.base_env.metabolism_state.stomach_fullness = fish.stomach_fullness

        self.base_env.interaction_state.other_fish = [
            f for f in self.base_env.interaction_state.other_fish if f.is_alive
        ]

        return result

    def _calculate_predation_success_prob(self, size_ratio: float, distance: float,
                                          predator: AIFishState, prey: OtherFish) -> float:
        """Calculate predation success probability - training-aligned"""
        # Consistent with training env
        if size_ratio >= 3.0:
            base_prob = 0.8
        elif size_ratio >= 2.0:
            base_prob = 0.6
        elif size_ratio >= 1.5:
            base_prob = 0.3
        else:
            base_prob = 0.1

        strike_range = calculate_strike_range(predator.total_length)
        distance_factor = 1.0 - (distance / strike_range) * 0.5

        predator_speed = np.linalg.norm(predator.velocity)
        prey_speed = np.linalg.norm(prey.velocity)
        if prey_speed > 0.01:
            speed_factor = min(predator_speed / prey_speed, 1.5)
        else:
            speed_factor = 1.5

        fatigue_factor = 1.0 - predator.fatigue / 200.0

        stress_factor = 1.0 - predator.stress_level * 0.3

        if prey.is_fleeing:
            base_prob *= 0.9

        final_prob = base_prob * distance_factor * speed_factor * fatigue_factor * stress_factor
        return np.clip(final_prob, 0.0, 0.8)

    def _simple_physics_update(self, fish: AIFishState, movement: np.ndarray, requested_state):
        """Fallback — heading"""
        dt = CONFIG.environment.time_step
        ps = self.base_env.physics_state

        if requested_state == ActivityState.ACTIVE:
            max_speed = 0.12
            # action[0]heading
            forward_mag = float(movement[0]) * max_speed
            forward_vel = ps.heading * forward_mag
            forward_vel[1] = ps.velocity[1]  #
            ps.velocity = ps.velocity * 0.6 + forward_vel * 0.4
        else:
            ps.velocity *= 0.95

        ps.position += ps.velocity * dt
        self._enforce_boundaries_on_env()

    def _enforce_boundaries_on_env(self):
        tank_geo = getattr(self, 'tank_geometry', None)
        pos = self.base_env.physics_state.position
        vel = self.base_env.physics_state.velocity

        if tank_geo is not None:
            pos[:], vel[:], _ = tank_geo.enforce_boundary(pos, vel)
        else:
            tank_radius = CONFIG.environment.tank_radius
            tank_depth = CONFIG.environment.tank_depth
            horizontal_dist = np.sqrt(pos[0] ** 2 + pos[2] ** 2)
            if horizontal_dist > tank_radius * 0.95:
                factor = tank_radius * 0.95 / horizontal_dist
                pos[0] *= factor
                pos[2] *= factor
                vel[0] *= -0.5
                vel[2] *= -0.5
            if pos[1] < -tank_depth + 0.05:
                pos[1] = -tank_depth + 0.05
                vel[1] = abs(vel[1]) * 0.5
            if pos[1] > -0.02:
                pos[1] = -0.02
                vel[1] = -abs(vel[1]) * 0.5

    def _check_ai_predation(self):
        """AI-to-AI predation"""
        ic = CONFIG.interaction  #  InteractionConfig
        min_ratio = ic.min_predation_size_ratio

        for predator in self.ai_fish:
            if not predator.is_alive:
                continue

            capture_distance = calculate_capture_radius(predator.total_length)
            stomach_capacity = predator.body_mass * CONFIG.feeding.stomach_capacity_ratio
            available_space = stomach_capacity - predator.stomach_content_mass

            if available_space < 0.5:
                continue

            for prey in self.ai_fish:
                if prey.id == predator.id or not prey.is_alive:
                    continue

                size_ratio = predator.body_mass / prey.body_mass
                if size_ratio < min_ratio:
                    continue

                distance = np.linalg.norm(predator.position - prey.position)
                if distance > capture_distance:
                    continue

                prey_digestible_mass = prey.body_mass * 0.7
                if prey_digestible_mass > available_space:
                    continue

                if size_ratio >= 3.0:
                    success_prob = 0.7
                elif size_ratio >= 2.0:
                    success_prob = 0.5
                else:
                    success_prob = 0.25

                if np.random.random() < success_prob:
                    prey.is_alive = False
                    prey.death_reason = 'eaten_by_ai'
                    prey.killed_by = predator.id

                    digestible_mass = prey.body_mass * 0.7
                    predator.stomach_content_mass += digestible_mass
                    predator.initial_meal_mass += digestible_mass
                    predator.stomach_fullness = (predator.stomach_content_mass / stomach_capacity) * 100
                    predator.ai_fish_eaten += 1
                    predator.fish_eaten += 1
                    available_space = stomach_capacity - predator.stomach_content_mass

                    self.ai_predation_events.append({
                        'step': self.current_step,
                        'predator_id': predator.id,
                        'prey_id': prey.id,
                        'prey_mass': prey.body_mass
                    })

                    print(f"[PREDATION] AI#{predator.id}({predator.body_mass:.1f}g) "
                          f"ate AI#{prey.id}({prey.body_mass:.1f}g)")

                    if available_space < 0.5:
                        break

    def _check_npc_attacks_on_ai(self):
        """
        Detect NPC attacks on AI - v3.1.1 FixAligned with training env

        Fix
        1. ✅ mass loss _process_damage
        2. ✅ size_ratio >= 5
        3. ✅ damage
        """
        ic = CONFIG.interaction

        for idx, npc in enumerate(self.base_env.interaction_state.other_fish):
            if not npc.is_alive:
                continue

            if not npc.is_chasing:
                continue

            if not npc.can_attack():
                continue

            if npc.behavior_type == FishBehaviorType.SURFACE_PREDATOR:
                attack_range = CONFIG.surface_predator.attack_range
                cooldown_steps = CONFIG.surface_predator.attack_cooldown_steps
            elif npc.behavior_type == FishBehaviorType.AGGRESSIVE:
                attack_range = CONFIG.aggressive_behavior.attack_range
                cooldown_steps = CONFIG.aggressive_behavior.attack_cooldown_steps
            else:
                continue

            for ai_fish in self.ai_fish:
                if not ai_fish.is_alive:
                    continue

                size_ratio = npc.body_mass / ai_fish.body_mass
                if size_ratio < ic.threat_size_ratio:
                    continue

                distance = np.linalg.norm(npc.position - ai_fish.position)
                if distance > attack_range:
                    continue

                # ===== Escape check=====
                if size_ratio >= 4.0:
                    escape_prob = 0.1
                elif size_ratio >= 3.0:
                    escape_prob = 0.15
                elif size_ratio >= 2.0:
                    escape_prob = 0.2
                else:
                    escape_prob = 0.25

                if np.random.random() < escape_prob:
                    print(f"💨 AI#{ai_fish.id} NPC#{idx}！")
                    continue

                # ===== 🔥 newAligned with training env =====
                old_mass = ai_fish.body_mass
                mass_loss = 0.0

                if size_ratio >= 5:
                    # 0.15% mass loss
                    mass_loss = old_mass * 0.015

                elif size_ratio >= 3.5:
                    # 0.15% mass loss
                    mass_loss = old_mass * 0.007

                elif size_ratio >= 2:
                    # 0.7% mass loss
                    mass_loss = old_mass * 0.003

                elif size_ratio >= 1.5:
                    # 0.4% mass loss
                    mass_loss = old_mass * 0.001

                else:
                    # 0.1% mass loss
                    mass_loss = old_mass * 0.0001

                # mass loss
                if ai_fish.is_alive and mass_loss > 0:
                    ai_fish.body_mass = max(1.0, old_mass - mass_loss)
                    ai_fish.total_length = mass_to_length(ai_fish.body_mass)

                    # Aligned with training env
                    stress_increase = min(0.3 * (size_ratio / 5), 0.5)
                    ai_fish.stress_level = min(1.0, ai_fish.stress_level + stress_increase)

                    print(f"⚔️ NPC#{idx}({npc.body_mass:.0f}g)  AI#{ai_fish.id} "
                          f"size ratio={size_ratio:.1f}x, mass loss={mass_loss:.2f}g "
                          f"({old_mass:.1f}→{ai_fish.body_mass:.1f}g)")

                # Enter cooldown
                npc.start_cooldown(cooldown_steps)
                break  # Single target per attack

    def step(self) -> Dict:
        """Execute one simulation step."""
        self.current_step += 1
        self.steps_since_feeding += 1

        self._update_food_movement()
        self._update_npc_fish()

        for fish in self.ai_fish:
            if not fish.is_alive:
                continue

            fish.steps_alive += 1
            self._sync_env_state(fish)
            original_fish = self._inject_other_ai_fish(fish.id)

            obs = self._get_observation_simple(fish)

            self._restore_original_fish(original_fish)

            if self.model is not None:
                action, _ = self.model.predict(obs, deterministic=False)
            else:
                action = np.random.uniform(-1, 1, 5).astype(np.float32)

            self._execute_action(fish, action)

            if fish.activity_state == ActivityState.ACTIVE:
                self._check_food_capture(fish)



        self._check_ai_predation()
        self._check_npc_attacks_on_ai()

        return {
            'step': self.current_step,
            'alive_count': sum(1 for f in self.ai_fish if f.is_alive),
            'food_count': len(self.base_env.feeding_state.food_items)
        }

    def _get_observation_simple(self, fish: AIFishState) -> np.ndarray:
        perception_input = self.base_env._create_perception_input()
        self.base_env.perception_system.update(
            self.base_env.perception_state, perception_input
        )
        return self.base_env._get_observation()

    def get_statistics(self) -> Dict:
        alive_fish = [f for f in self.ai_fish if f.is_alive]
        dead_fish = [f for f in self.ai_fish if not f.is_alive]

        stats = {
            'total_fish': self.num_fish,
            'alive_count': len(alive_fish),
            'dead_count': len(dead_fish),
            'current_step': self.current_step,
            'food_remaining': len(self.base_env.feeding_state.food_items),
            'total_food_spawned': self.total_food_spawned,
            'total_food_eaten': self.total_food_eaten,
            'ai_predation_events': len(self.ai_predation_events),
        }

        if alive_fish:
            stats['avg_mass'] = np.mean([f.body_mass for f in alive_fish])
            stats['avg_energy'] = np.mean([f.energy for f in alive_fish])
            stats['total_mass'] = sum(f.body_mass for f in alive_fish)
            stats['resting_count'] = sum(1 for f in alive_fish if f.activity_state == ActivityState.RESTING)

        return stats


# ============================================================
# PyGame Academic Visualizer - Same as original version，Minor adjustments
# ============================================================

class AcademicVisualizer:
    """Academic-style visualization interface v3.1."""

    def __init__(self, settings: dict = None, window_size: tuple = (1500, 900)):
        self.window_size = window_size
        self.settings = settings or {}

        model_path = self.settings.get('model_path')
        num_fish = self.settings.get('num_fish', 10)
        initial_mass = self.settings.get('initial_mass', 20.0)
        self.include_npc = self.settings.get('include_npc', True)

        self.model = None
        self.model_path = model_path
        if model_path and HAS_SB3 and os.path.exists(model_path):
            try:
                self.model = PPO.load(model_path)
                print(f"[OK] Model loaded: {model_path}")
            except Exception as e:
                print(f"[ERROR] Failed to load model: {e}")
                self.model = None
        else:
            print("[INFO] No model loaded, using random actions")

        self.env = BassEnvironment({
            'verbose': 0,
            'course': settings.get('course', 'course4')
        })
        self.manager = MultiAIFishManager(
            self.env, self.model, num_fish, initial_mass, self.include_npc
        )

        self.tank_radius = CONFIG.environment.tank_radius  #
        self.tank_depth = CONFIG.environment.tank_depth
        self.tank_shape = 'circular'  #
        self.tank_half_x = self.tank_radius  # X_sync_tank_info
        self.time_step = CONFIG.environment.time_step
        self.time_acceleration = CONFIG.environment.time_acceleration
        self.realtime_mode = False

        self.paused = False
        self.running = True
        self.speed_multiplier = 1
        self.fullscreen = False
        self.need_reset_dialog = False

        self.selected_fish_id = 0
        self.fish_list_rects = []
        self.fish_list_scroll = 0
        self.fish_list_max_visible = 8

        pygame.init()
        pygame.font.init()

        self.screen = pygame.display.set_mode(window_size, pygame.RESIZABLE)
        pygame.display.set_caption("Bass Fish Simulation v3.1 - Training Aligned")

        self.clock = pygame.time.Clock()

        self.font_title = pygame.font.SysFont('Arial', 20, bold=True)
        self.font_header = pygame.font.SysFont('Arial', 14, bold=True)
        self.font_normal = pygame.font.SysFont('Consolas', 13)
        self.font_small = pygame.font.SysFont('Consolas', 11)
        self.font_tiny = pygame.font.SysFont('Consolas', 10)

        self._update_layout()

    def _update_layout(self):
        w, h = self.window_size
        self.margin = 15
        self.left_col_width = min(450, int(w * 0.32))
        self.top_view_top = 45
        self.top_view_height = min(380, int(h * 0.45))
        self.top_view_center = (self.margin + self.left_col_width // 2,
                                self.top_view_top + self.top_view_height // 2)
        self.top_view_radius = min(180, self.top_view_height // 2 - 30)
        self.top_view_scale = self.top_view_radius / self.tank_radius
        self.side_view_top = self.top_view_top + self.top_view_height + 15
        self.side_view_height = min(180, int(h * 0.22))
        self.side_view_left = self.margin
        self.side_view_width = self.left_col_width
        self.legend_top = self.side_view_top + self.side_view_height + 50
        self.legend_height = 80
        self.panel_left = self.margin + self.left_col_width + 20
        self.panel_width = w - self.panel_left - self.margin
        self.fish_list_width = min(210, int(self.panel_width * 0.23))
        self.fish_list_height = min(450, int(h * 0.52))
        self.details_left = self.panel_left + self.fish_list_width + 10
        self.details_width = min(280, int(self.panel_width * 0.32))
        self.details_height = self.fish_list_height
        self.charts_left = self.details_left + self.details_width + 10
        self.charts_width = max(300, w - self.charts_left - self.margin)
        self.charts_height = self.fish_list_height
        self.status_top = max(self.fish_list_height + 60, self.legend_top)
        self.status_height = h - self.status_top - self.margin
        self.status_width = self.panel_width

    def show_reset_dialog(self) -> bool:
        pygame.display.iconify()
        result = show_settings_dialog(
            default_num_fish=self.manager.num_fish,
            default_mass=self.manager.initial_mass,
            default_model_path=self.model_path or ""
        )
        pygame.display.set_mode(self.window_size, pygame.RESIZABLE)

        if result:
            self.settings = result
            new_model_path = result.get('model_path')
            if new_model_path != self.model_path:
                self.model_path = new_model_path
                if new_model_path and HAS_SB3 and os.path.exists(new_model_path):
                    try:
                        self.model = PPO.load(new_model_path)
                        print(f"[OK] Model loaded: {new_model_path}")
                    except Exception as e:
                        print(f"[ERROR] Failed to load model: {e}")
                        self.model = None
                else:
                    self.model = None

            self.manager.update_config(
                num_fish=result['num_fish'],
                initial_mass=result['initial_mass'],
                model=self.model,
                include_npc=result['include_npc']
            )
            self.include_npc = result['include_npc']
            self.manager.reset()
            self._sync_tank_info()
            self.selected_fish_id = 0
            self.fish_list_scroll = 0
            return True
        return False

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    self.need_reset_dialog = True
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self.speed_multiplier = min(10, self.speed_multiplier + 1)
                elif event.key == pygame.K_MINUS:
                    self.speed_multiplier = max(1, self.speed_multiplier - 1)
                elif event.key == pygame.K_f:
                    self.fullscreen = not self.fullscreen
                    if self.fullscreen:
                        self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
                        self.window_size = self.screen.get_size()
                    else:
                        self.window_size = (1500, 900)
                        self.screen = pygame.display.set_mode(self.window_size, pygame.RESIZABLE)
                    self._update_layout()
                elif event.key == pygame.K_t:
                    self.realtime_mode = not self.realtime_mode
                elif event.key in [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5,
                                   pygame.K_6, pygame.K_7, pygame.K_8, pygame.K_9]:
                    self.speed_multiplier = event.key - pygame.K_0
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self._handle_click(event.pos)
                elif event.button == 4:
                    self._handle_scroll(-1)
                elif event.button == 5:
                    self._handle_scroll(1)
            elif event.type == pygame.MOUSEWHEEL:
                self._handle_scroll(-event.y)
            elif event.type == pygame.VIDEORESIZE:
                self.window_size = event.size
                self.screen = pygame.display.set_mode(self.window_size, pygame.RESIZABLE)
                self._update_layout()

    def _handle_click(self, pos):
        for fish_id, rect in self.fish_list_rects:
            if rect.collidepoint(pos):
                self.selected_fish_id = fish_id
                return

    def _handle_scroll(self, direction):
        mouse_pos = pygame.mouse.get_pos()
        fish_list_rect = pygame.Rect(self.panel_left, 40, self.fish_list_width, self.fish_list_height)
        if fish_list_rect.collidepoint(mouse_pos):
            max_scroll = max(0, len(self.manager.ai_fish) - self.fish_list_max_visible)
            self.fish_list_scroll = max(0, min(max_scroll, self.fish_list_scroll + direction))

    def get_selected_fish(self) -> Optional[AIFishState]:
        for fish in self.manager.ai_fish:
            if fish.id == self.selected_fish_id:
                return fish
        for fish in self.manager.ai_fish:
            if fish.is_alive:
                self.selected_fish_id = fish.id
                return fish
        return None

    def draw_panel_background(self, x, y, width, height, title=None):
        rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.screen, AcademicColors.PANEL_BG, rect)
        pygame.draw.rect(self.screen, AcademicColors.PANEL_BORDER, rect, 1)
        if title:
            title_rect = pygame.Rect(x, y, width, 24)
            pygame.draw.rect(self.screen, AcademicColors.PANEL_HEADER, title_rect)
            title_text = truncate_text(title, self.font_header, width - 20)
            text_surface = self.font_header.render(title_text, True, AcademicColors.TEXT_PRIMARY)
            self.screen.blit(text_surface, (x + 8, y + 4))
            return y + 28
        return y + 5

    def draw_tank_top(self):
        cx, cy = self.top_view_center
        panel_y = self.draw_panel_background(
            self.margin, self.top_view_top - 5,
            self.left_col_width, self.top_view_height,
            f"Top View (X-Z) [{self.tank_shape}]"
        )

        if self.tank_shape == 'circular':
            radius = int(self.tank_radius * self.top_view_scale)
            pygame.draw.circle(self.screen, AcademicColors.WATER_DEEP, (cx, cy), radius)
            pygame.draw.circle(self.screen, AcademicColors.TANK_BORDER, (cx, cy), radius, 2)
            for dist in [0.5, 1.0, 1.5]:
                if dist <= self.tank_radius:
                    r = int(dist * self.top_view_scale)
                    pygame.draw.circle(self.screen, AcademicColors.CHART_GRID, (cx, cy), r, 1)
        elif self.tank_shape == 'irregular_polygon':
            geo = getattr(self.manager, 'tank_geometry', None)
            drawn = False
            if geo is not None and hasattr(geo, 'vertices'):
                try:
                    pts = [world_to_screen_top(
                        np.array([x, 0.0, z]), self.top_view_center, self.top_view_scale
                    ) for x, z in geo.vertices]
                    pygame.draw.polygon(self.screen, AcademicColors.WATER_DEEP, pts)
                    pygame.draw.polygon(self.screen, AcademicColors.TANK_BORDER, pts, 2)
                    drawn = True
                except Exception:
                    pass
            if not drawn:
                half_w = int(self.tank_width / 2 * self.top_view_scale)
                half_l = int(self.tank_length / 2 * self.top_view_scale)
                rect = pygame.Rect(cx - half_w, cy - half_l, half_w * 2, half_l * 2)
                pygame.draw.rect(self.screen, AcademicColors.WATER_DEEP, rect)
                pygame.draw.rect(self.screen, AcademicColors.TANK_BORDER, rect, 2)
        else:
            half_w = int(self.tank_width / 2 * self.top_view_scale)
            half_l = int(self.tank_length / 2 * self.top_view_scale)
            rect = pygame.Rect(cx - half_w, cy - half_l, half_w * 2, half_l * 2)
            pygame.draw.rect(self.screen, AcademicColors.WATER_DEEP, rect)
            pygame.draw.rect(self.screen, AcademicColors.TANK_BORDER, rect, 2)
            # 0.5m
            step = max(1, int(0.5 * self.top_view_scale))
            for gx in range(cx - half_w + step, cx + half_w, step):
                pygame.draw.line(self.screen, AcademicColors.CHART_GRID,
                                 (gx, cy - half_l), (gx, cy + half_l), 1)
            for gz in range(cy - half_l + step, cy + half_l, step):
                pygame.draw.line(self.screen, AcademicColors.CHART_GRID,
                                 (cx - half_w, gz), (cx + half_w, gz), 1)

        axis_len = int(self.tank_radius * self.top_view_scale) + 10
        pygame.draw.line(self.screen, AcademicColors.TEXT_DIM,
                         (cx - axis_len, cy), (cx + axis_len, cy), 1)
        pygame.draw.line(self.screen, AcademicColors.TEXT_DIM,
                         (cx, cy - axis_len), (cx, cy + axis_len), 1)
        x_label = self.font_tiny.render("X", True, AcademicColors.TEXT_SECONDARY)
        z_label = self.font_tiny.render("Z", True, AcademicColors.TEXT_SECONDARY)
        self.screen.blit(x_label, (cx + axis_len + 2, cy - 6))
        self.screen.blit(z_label, (cx - 4, cy - axis_len - 14))

    def draw_obstacles(self):
        """"""
        obs_field = getattr(self.manager, 'obstacle_field', None)
        if obs_field is None or obs_field.count == 0:
            return

        for obs in obs_field.obstacles:
            pos = obs.center

            if hasattr(obs, 'radius'):
                # Rock -
                r = obs.radius
                screen_pos = world_to_screen_top(pos, self.top_view_center, self.top_view_scale)
                screen_r = max(3, int(r * self.top_view_scale))
                pygame.draw.circle(self.screen, (100, 90, 80), screen_pos, screen_r)
                pygame.draw.circle(self.screen, (140, 130, 120), screen_pos, screen_r, 1)

                screen_pos = world_to_screen_side(
                    pos, self.side_view_left, self.side_view_top,
                    self.side_view_width, self.side_view_height,
                    self.tank_half_x, self.tank_depth
                )
                screen_r_side = max(3, int(r * self.side_view_height / self.tank_depth))
                pygame.draw.circle(self.screen, (140, 130, 120), screen_pos, screen_r_side, 1)
            else:
                # Box -
                hx, hy, hz = obs.half_size
                x1 = pos[0] - hx
                x2 = pos[0] + hx
                z1 = pos[2] - hz
                z2 = pos[2] + hz
                p1 = world_to_screen_top(np.array([x1, 0, z1]), self.top_view_center, self.top_view_scale)
                p2 = world_to_screen_top(np.array([x2, 0, z2]), self.top_view_center, self.top_view_scale)
                rect = pygame.Rect(min(p1[0], p2[0]), min(p1[1], p2[1]), abs(p2[0]-p1[0]), abs(p2[1]-p1[1]))
                pygame.draw.rect(self.screen, (80, 70, 60), rect)
                pygame.draw.rect(self.screen, (120, 110, 100), rect, 2)

                x1 = pos[0] - hx
                x2 = pos[0] + hx
                y1 = pos[1] - hy
                y2 = pos[1] + hy
                p1 = world_to_screen_side(
                    np.array([x1, y1, 0]), self.side_view_left, self.side_view_top,
                    self.side_view_width, self.side_view_height,
                    self.tank_half_x, self.tank_depth
                )
                p2 = world_to_screen_side(
                    np.array([x2, y2, 0]), self.side_view_left, self.side_view_top,
                    self.side_view_width, self.side_view_height,
                    self.tank_half_x, self.tank_depth
                )
                rect = pygame.Rect(min(p1[0], p2[0]), min(p1[1], p2[1]), abs(p2[0]-p1[0]), abs(p2[1]-p1[1]))
                pygame.draw.rect(self.screen, (80, 70, 60), rect)
                pygame.draw.rect(self.screen, (120, 110, 100), rect, 2)

    def draw_tank_side(self):
        left = self.side_view_left
        top = self.side_view_top
        width = self.side_view_width
        height = self.side_view_height
        panel_y = self.draw_panel_background(left, top - 30, width, height + 40, "Side View (X-Y Depth)")
        view_top = top
        rect = pygame.Rect(left, view_top, width, height)
        pygame.draw.rect(self.screen, AcademicColors.WATER_DEEP, rect)
        danger_ratio = 0.15 / self.tank_depth
        danger_height = int(height * danger_ratio)
        danger_rect = pygame.Rect(left, view_top, width, danger_height)
        pygame.draw.rect(self.screen, AcademicColors.WATER_DANGER, danger_rect)
        pygame.draw.rect(self.screen, AcademicColors.TANK_BORDER, rect, 2)
        pygame.draw.line(self.screen, AcademicColors.WATER_SURFACE,
                         (left, view_top), (left + width, view_top), 3)
        scale_x = left + width + 3
        for depth_cm in [0, 20, 40, 60, 80]:
            depth_m = depth_cm / 100.0
            if depth_m <= self.tank_depth:
                y = int(view_top + depth_m / self.tank_depth * height)
                pygame.draw.line(self.screen, AcademicColors.TEXT_DIM,
                                 (left + width, y), (scale_x + 3, y), 1)
                label = self.font_tiny.render(f"{depth_cm}cm", True, AcademicColors.TEXT_DIM)
                self.screen.blit(label, (scale_x + 5, y - 6))

    def draw_food(self):
        food_items = self.env.feeding_state.food_items
        for food in food_items:
            pos = food.position
            if food.is_settling:
                color = AcademicColors.FOOD_SETTLING
            elif food.food_type.value == 'floating':
                color = AcademicColors.FOOD_FLOATING
            elif food.food_type.value == 'ambient':
                color = (100, 200, 255)      #
            elif food.food_type.value == 'surface_env':
                color = (255, 255, 100)      #
            elif food.food_type.value == 'benthic':
                color = (180, 120, 60)       #
            elif food.food_type.value == 'attached':
                color = (100, 255, 150)      #
            else:
                color = AcademicColors.FOOD_SINKING
            screen_pos = world_to_screen_top(pos, self.top_view_center, self.top_view_scale)
            pygame.draw.circle(self.screen, color, screen_pos, 3)
            screen_pos = world_to_screen_side(
                pos, self.side_view_left, self.side_view_top,
                self.side_view_width, self.side_view_height,
                self.tank_half_x, self.tank_depth
            )
            pygame.draw.circle(self.screen, color, screen_pos, 3)

    def draw_npc_fish(self):
        if not self.include_npc:
            return
        fish_states = self.env.interaction_system.get_fish_states(self.env.interaction_state)
        for fish in fish_states:
            pos = fish['position']
            behavior = fish.get('behavior_type', 'passive')
            size_cat = fish.get('size_category', 'small')
            is_chasing = fish.get('is_chasing', False)
            if behavior == 'surface':
                color = AcademicColors.NPC_PREDATOR
                size = 10
            elif behavior == 'aggressive':
                color = AcademicColors.NPC_CHASING if is_chasing else AcademicColors.NPC_LARGE
                size = 8
            elif size_cat == 'small':
                color = AcademicColors.NPC_SMALL
                size = 3
            elif size_cat == 'medium':
                color = AcademicColors.NPC_MEDIUM
                size = 5
            else:
                color = AcademicColors.NPC_LARGE
                size = 7
            screen_pos = world_to_screen_top(pos, self.top_view_center, self.top_view_scale)
            pygame.draw.circle(self.screen, color, screen_pos, size)
            if is_chasing:
                pygame.draw.circle(self.screen, (255, 0, 0), screen_pos, size + 3, 2)
            screen_pos = world_to_screen_side(
                pos, self.side_view_left, self.side_view_top,
                self.side_view_width, self.side_view_height,
                self.tank_half_x, self.tank_depth
            )
            pygame.draw.circle(self.screen, color, screen_pos, size)

    def _draw_fish_shape(self, screen_pos, heading_2d, size, color, is_selected):
        """heading"""
        if heading_2d is None or np.linalg.norm(heading_2d) < 0.01:
            heading_2d = np.array([1.0, 0.0])
        hx, hy = heading_2d[0], -heading_2d[1]  # Y

        # /
        body_len = size * 2.2
        body_wid = size * 0.9

        # heading
        sx, sy = -hy, hx

        # ()
        cx, cy = screen_pos
        head   = (cx + hx * body_len * 0.55, cy + hy * body_len * 0.55)
        mid_l  = (cx + sx * body_wid * 0.5 - hx * body_len * 0.05,
                  cy + sy * body_wid * 0.5 - hy * body_len * 0.05)
        tail_l = (cx + sx * body_wid * 0.7 - hx * body_len * 0.55,
                  cy + sy * body_wid * 0.7 - hy * body_len * 0.55)
        tail_r = (cx - sx * body_wid * 0.7 - hx * body_len * 0.55,
                  cy - sy * body_wid * 0.7 - hy * body_len * 0.55)
        mid_r  = (cx - sx * body_wid * 0.5 - hx * body_len * 0.05,
                  cy - sy * body_wid * 0.5 - hy * body_len * 0.05)

        points = [head, mid_l, tail_l, tail_r, mid_r]
        points_int = [(int(p[0]), int(p[1])) for p in points]

        if is_selected:
            glow_pts = [(int(cx + (p[0]-cx)*1.35), int(cy + (p[1]-cy)*1.35)) for p in points_int]
            pygame.draw.polygon(self.screen, AcademicColors.SELECTED_GLOW, glow_pts, 2)

        pygame.draw.polygon(self.screen, color, points_int)
        pygame.draw.polygon(self.screen, (0, 0, 0), points_int, 1)

        eye_x = int(cx + hx * body_len * 0.38 + sx * body_wid * 0.28)
        eye_y = int(cy + hy * body_len * 0.38 + sy * body_wid * 0.28)
        pygame.draw.circle(self.screen, (0, 0, 0), (eye_x, eye_y), max(1, size // 4))

    def draw_ai_fish(self):
        for fish in self.manager.ai_fish:
            if not fish.is_alive:
                continue
            pos = fish.position
            color = fish.color
            size = max(5, int(fish.body_mass / 4))
            is_selected = (fish.id == self.selected_fish_id)

            # ===== heading =====
            screen_pos = world_to_screen_top(pos, self.top_view_center, self.top_view_scale)
            heading = fish.heading
            heading_2d = np.array([heading[0], heading[2]])
            heading_norm = np.linalg.norm(heading_2d)
            if heading_norm > 0.01:
                heading_2d = heading_2d / heading_norm
            else:
                heading_2d = np.array([1.0, 0.0])

            self._draw_fish_shape(screen_pos, heading_2d, size, color, is_selected)

            # ID
            id_label = self.font_tiny.render(str(fish.id), True, (255, 255, 255))
            self.screen.blit(id_label, (screen_pos[0] - 3, screen_pos[1] - 4))

            # ===== headingX+pitch =====
            screen_pos = world_to_screen_side(
                pos, self.side_view_left, self.side_view_top,
                self.side_view_width, self.side_view_height,
                self.tank_half_x, self.tank_depth
            )
            if is_selected:
                pygame.draw.circle(self.screen, AcademicColors.SELECTED_GLOW, screen_pos, size + 5, 2)
            pygame.draw.circle(self.screen, color, screen_pos, size)
            pygame.draw.circle(self.screen, (0, 0, 0), screen_pos, size, 1)

    def draw_fish_list_panel(self):
        panel_x = self.panel_left
        panel_y = 40
        panel_width = self.fish_list_width
        panel_height = self.fish_list_height
        stats = self.manager.get_statistics()
        resting_count = stats.get('resting_count', 0)
        content_y = self.draw_panel_background(
            panel_x, panel_y, panel_width, panel_height,
            f"Fish ({stats['alive_count']}/{self.manager.num_fish}) R:{resting_count}"
        )
        self.fish_list_rects = []
        item_height = 38
        visible_height = panel_height - 35
        self.fish_list_max_visible = visible_height // item_height
        clip_rect = pygame.Rect(panel_x + 3, content_y, panel_width - 6, visible_height)
        self.screen.set_clip(clip_rect)
        y = content_y + 3
        start_idx = self.fish_list_scroll
        end_idx = min(start_idx + self.fish_list_max_visible + 1, len(self.manager.ai_fish))
        for fish in self.manager.ai_fish[start_idx:end_idx]:
            is_selected = (fish.id == self.selected_fish_id)
            item_rect = pygame.Rect(panel_x + 4, y, panel_width - 8, item_height - 2)
            if is_selected:
                pygame.draw.rect(self.screen, AcademicColors.PANEL_HEADER, item_rect)
                pygame.draw.rect(self.screen, AcademicColors.SELECTED_GLOW, item_rect, 2)
            self.fish_list_rects.append((fish.id, item_rect))
            pygame.draw.circle(self.screen, fish.color, (panel_x + 16, y + item_height // 2), 6)
            if fish.is_alive:
                status_color = AcademicColors.TEXT_PRIMARY
                state_indicator = "😴" if fish.activity_state == ActivityState.RESTING else ""
                status_text = f"#{fish.id}{state_indicator}: {fish.body_mass:.1f}g"
                energy_text = f"E:{fish.energy:.0f}%"
                energy_color = AcademicColors.TEXT_SUCCESS if fish.energy > 50 else AcademicColors.TEXT_WARNING
            else:
                status_color = AcademicColors.TEXT_DIM
                status_text = f"#{fish.id}: DEAD"
                energy_text = fish.death_reason[:8] if fish.death_reason else ""
                energy_color = AcademicColors.TEXT_DANGER
            text = self.font_small.render(status_text, True, status_color)
            self.screen.blit(text, (panel_x + 28, y + 3))
            energy_label = self.font_tiny.render(energy_text, True, energy_color)
            self.screen.blit(energy_label, (panel_x + 28, y + 18))
            y += item_height
        self.screen.set_clip(None)
        total_items = len(self.manager.ai_fish)
        if total_items > self.fish_list_max_visible:
            scrollbar_x = panel_x + panel_width - 10
            scrollbar_y = content_y + 2
            scrollbar_height = visible_height - 4
            pygame.draw.rect(self.screen, AcademicColors.SCROLLBAR_BG,
                             (scrollbar_x, scrollbar_y, 6, scrollbar_height))
            thumb_height = max(20, int(scrollbar_height * self.fish_list_max_visible / total_items))
            max_scroll = total_items - self.fish_list_max_visible
            thumb_y = scrollbar_y + int((
                                                    scrollbar_height - thumb_height) * self.fish_list_scroll / max_scroll) if max_scroll > 0 else scrollbar_y
            pygame.draw.rect(self.screen, AcademicColors.SCROLLBAR_THUMB,
                             (scrollbar_x, thumb_y, 6, thumb_height), border_radius=3)

    def draw_selected_fish_details(self):
        fish = self.get_selected_fish()
        if not fish:
            return
        panel_x = self.details_left
        panel_y = 40
        panel_width = self.details_width
        panel_height = self.details_height
        content_y = self.draw_panel_background(
            panel_x, panel_y, panel_width, panel_height,
            f"Fish #{fish.id} Details"
        )
        y = content_y + 3
        line_height = 16
        max_width = panel_width - 25

        section_label = self.font_small.render("─ Basic State ─", True, AcademicColors.TEXT_ACCENT)
        self.screen.blit(section_label, (panel_x + 8, y))
        y += line_height + 3

        basic_info = [
            (f"Mass: {fish.body_mass:.2f} g", AcademicColors.TEXT_PRIMARY),
            (f"Length: {fish.total_length * 100:.1f} cm", AcademicColors.TEXT_PRIMARY),
            (f"Energy: {fish.energy:.1f} %",
             AcademicColors.TEXT_SUCCESS if fish.energy > 50 else AcademicColors.TEXT_WARNING),
            (f"Stomach: {fish.stomach_fullness:.1f} %", AcademicColors.TEXT_PRIMARY),
            (f"Activity: {'RESTING' if fish.activity_state == ActivityState.RESTING else 'ACTIVE'}",
             AcademicColors.TEXT_DIM if fish.activity_state == ActivityState.RESTING else AcademicColors.TEXT_ACCENT),
        ]
        for text, color in basic_info:
            label = self.font_small.render(truncate_text(text, self.font_small, max_width), True, color)
            self.screen.blit(label, (panel_x + 12, y))
            y += line_height
        y += 6

        section_label = self.font_small.render("─ Motion ─", True, AcademicColors.TEXT_ACCENT)
        self.screen.blit(section_label, (panel_x + 8, y))
        y += line_height + 3

        speed_mps = fish.current_speed
        speed_bls = speed_mps / fish.total_length if fish.total_length > 0 else 0
        speed_color = AcademicColors.CHART_SPEED
        if speed_bls > 3.0:
            speed_color = AcademicColors.TEXT_DANGER
        elif speed_bls > 1.5:
            speed_color = AcademicColors.TEXT_WARNING

        motion_info = [
            (f"Speed: {speed_mps:.3f} m/s ({speed_bls:.1f} BL/s)", speed_color),
            (f"Heading: ({fish.heading[0]:.2f}, {fish.heading[2]:.2f})", AcademicColors.TEXT_PRIMARY),
            (f"Turn Rate: {fish.turn_rate:.1f} °/s", AcademicColors.TEXT_PRIMARY),
            (f"Pitch: {fish.pitch_angle:+.1f}°", AcademicColors.TEXT_PRIMARY),
        ]
        for text, color in motion_info:
            label = self.font_small.render(truncate_text(text, self.font_small, max_width), True, color)
            self.screen.blit(label, (panel_x + 12, y))
            y += line_height
        y += 6

        section_label = self.font_small.render("─ Buoyancy ─", True, AcademicColors.TEXT_ACCENT)
        self.screen.blit(section_label, (panel_x + 8, y))
        y += line_height + 3

        buoyancy_ctrl = fish.buoyancy_control
        mode = int(getattr(fish, 'buoyancy_mode', 0))
        ctrl_direction = "UP" if mode < 0 else ("DOWN" if mode > 0 else "HOLD")
        current_depth = max(0, -fish.position[1])

        buoyancy_info = [
            (f"Mode: {mode:+d} ({ctrl_direction})",
             AcademicColors.TEXT_WARNING if mode != 0 else AcademicColors.TEXT_PRIMARY),
            (f"Raw Signal: {getattr(fish, 'raw_buoyancy_control', buoyancy_ctrl):+.2f}",
             AcademicColors.TEXT_PRIMARY),
            (f"Rel.Density: {fish.relative_density:.4f}",
             AcademicColors.TEXT_SUCCESS if 0.99 < fish.relative_density < 1.01 else AcademicColors.TEXT_WARNING),
            (f"Depth: {current_depth * 100:.1f} cm", AcademicColors.TEXT_PRIMARY),
        ]
        for text, color in buoyancy_info:
            label = self.font_small.render(truncate_text(text, self.font_small, max_width), True, color)
            self.screen.blit(label, (panel_x + 12, y))
            y += line_height
        y += 6

        section_label = self.font_small.render("─ Statistics ─", True, AcademicColors.TEXT_ACCENT)
        self.screen.blit(section_label, (panel_x + 8, y))
        y += line_height + 3

        stats_info = [
            (f"Food Eaten: {fish.food_eaten}", AcademicColors.TEXT_PRIMARY),
            (f"Fish Eaten: {fish.fish_eaten}", AcademicColors.TEXT_PRIMARY),
            (f"Steps Alive: {fish.steps_alive}", AcademicColors.TEXT_PRIMARY),
            (f"Growth: {fish.growth_count}x", AcademicColors.TEXT_SUCCESS),
        ]
        for text, color in stats_info:
            label = self.font_small.render(truncate_text(text, self.font_small, max_width), True, color)
            self.screen.blit(label, (panel_x + 12, y))
            y += line_height

    def draw_realtime_charts(self):
        fish = self.get_selected_fish()
        if not fish:
            return
        panel_x = self.charts_left
        panel_y = 40
        panel_width = self.charts_width
        panel_height = self.charts_height
        content_y = self.draw_panel_background(
            panel_x, panel_y, panel_width, panel_height,
            "Real-time Charts"
        )
        chart_width = panel_width - 70
        chart_height = (panel_height - 70) // 3 - 15
        chart_left = panel_x + 50

        self._draw_mini_chart(
            chart_left, content_y + 8, chart_width, chart_height,
            list(fish.energy_history), "Energy",
            AcademicColors.CHART_ENERGY, 0, 100
        )

        if fish.mass_history:
            min_mass = max(0, min(fish.mass_history) - 2)
            max_mass = max(fish.mass_history) + 2
        else:
            min_mass, max_mass = 15, 25

        self._draw_mini_chart(
            chart_left, content_y + chart_height + 25, chart_width, chart_height,
            list(fish.mass_history), "Mass(g)",
            AcademicColors.CHART_MASS, min_mass, max_mass
        )

        fish_length = fish.total_length
        max_bio_speed = fish_length * 3.5
        self._draw_speed_chart(
            chart_left, content_y + 2 * (chart_height + 17) + 8, chart_width, chart_height,
            list(fish.speed_history), "Speed",
            AcademicColors.CHART_SPEED, 0, max(0.15, max_bio_speed),
            fish_length
        )

    def _draw_mini_chart(self, x, y, width, height, data, label, color, y_min, y_max):
        chart_rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.screen, AcademicColors.CHART_BG, chart_rect)
        pygame.draw.rect(self.screen, AcademicColors.PANEL_BORDER, chart_rect, 1)
        label_text = self.font_tiny.render(label, True, color)
        self.screen.blit(label_text, (x - 45, y + height // 2 - 6))
        for i in range(5):
            gy = y + int(i * height / 4)
            pygame.draw.line(self.screen, AcademicColors.CHART_GRID,
                             (x, gy), (x + width, gy), 1)
        y_labels = [y_max, (y_max + y_min) / 2, y_min]
        for i, val in enumerate(y_labels):
            ly = y + int(i * height / 2)
            val_text = self.font_tiny.render(f"{val:.0f}", True, AcademicColors.TEXT_DIM)
            self.screen.blit(val_text, (x + width + 3, ly - 5))
        if len(data) > 1:
            points = []
            for i, val in enumerate(data):
                px = x + int(i / len(data) * width)
                val = max(y_min, min(y_max, val))
                py = y + height - int((val - y_min) / (y_max - y_min) * height)
                points.append((px, py))
            if len(points) >= 2:
                pygame.draw.lines(self.screen, color, False, points, 2)
                if points:
                    pygame.draw.circle(self.screen, color, points[-1], 3)

    def _draw_speed_chart(self, x, y, width, height, data, label, color, y_min, y_max, fish_length):
        chart_rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.screen, AcademicColors.CHART_BG, chart_rect)
        pygame.draw.rect(self.screen, AcademicColors.PANEL_BORDER, chart_rect, 1)
        label_text = self.font_tiny.render(label, True, color)
        self.screen.blit(label_text, (x - 45, y + height // 2 - 6))
        for i in range(5):
            gy = y + int(i * height / 4)
            pygame.draw.line(self.screen, AcademicColors.CHART_GRID,
                             (x, gy), (x + width, gy), 1)

        cruise_speed = fish_length * 1.0
        if y_min <= cruise_speed <= y_max:
            cruise_y = y + height - int((cruise_speed - y_min) / (y_max - y_min) * height)
            pygame.draw.line(self.screen, AcademicColors.TEXT_SUCCESS,
                             (x, cruise_y), (x + width, cruise_y), 1)
            cruise_label = self.font_tiny.render("1BL/s", True, AcademicColors.TEXT_SUCCESS)
            self.screen.blit(cruise_label, (x + 2, cruise_y - 10))

        burst_mult = CONFIG.physics.burst_speed_multiplier
        burst_speed = fish_length * burst_mult

        if y_min <= burst_speed <= y_max:
            burst_y = y + height - int((burst_speed - y_min) / (y_max - y_min) * height)
            pygame.draw.line(self.screen, AcademicColors.TEXT_DANGER,
                             (x, burst_y), (x + width, burst_y), 1)

            #  "6BL/s"
            label_text = f"{burst_mult:.0f}BL/s"
            burst_label = self.font_tiny.render(label_text, True, AcademicColors.TEXT_DANGER)
            self.screen.blit(burst_label, (x + 2, burst_y - 10))

        if len(data) > 1:
            points = []
            for i, val in enumerate(data):
                px = x + int(i / len(data) * width)
                val_clamped = max(y_min, min(y_max, val))
                py = y + height - int((val_clamped - y_min) / (y_max - y_min) * height)
                points.append((px, py))
            if len(points) >= 2:
                pygame.draw.lines(self.screen, color, False, points, 2)
                if points:
                    marker_color = color
                    if data and len(data) > 0:
                        current_val = data[-1]
                        if current_val > burst_speed:
                            marker_color = AcademicColors.TEXT_DANGER
                        elif current_val > fish_length * 1.5:
                            marker_color = AcademicColors.TEXT_WARNING
                    pygame.draw.circle(self.screen, marker_color, points[-1], 4)

    def draw_simulation_info(self):
        panel_x = self.panel_left
        panel_y = self.status_top
        panel_width = self.status_width
        panel_height = min(160, self.status_height)
        content_y = self.draw_panel_background(
            panel_x, panel_y, panel_width, panel_height,
            "Simulation Status (v3.1 Training-Aligned)"
        )
        stats = self.manager.get_statistics()
        y = content_y + 8
        col_width = panel_width // 4
        col1_x = panel_x + 15
        col2_x = col1_x + col_width
        col3_x = col2_x + col_width
        col4_x = col3_x + col_width

        if self.realtime_mode:
            sim_time = format_time(stats['current_step'], self.time_step, self.time_acceleration, True)
            time_mode = "REAL"
            time_color = AcademicColors.TEXT_SUCCESS
        else:
            sim_time = format_time(stats['current_step'], self.time_step, self.time_acceleration, False)
            time_mode = "ACC"
            time_color = AcademicColors.TEXT_WARNING

        time_label = self.font_normal.render(f"Time: {sim_time} [{time_mode}]", True, time_color)
        self.screen.blit(time_label, (col1_x, y))

        step_label = self.font_small.render(f"Step: {stats['current_step']}", True, AcademicColors.TEXT_SECONDARY)
        self.screen.blit(step_label, (col2_x, y))

        speed_label = self.font_small.render(f"Speed: x{self.speed_multiplier}", True, AcademicColors.TEXT_SECONDARY)
        self.screen.blit(speed_label, (col3_x, y))

        status_text = "PAUSED" if self.paused else "RUNNING"
        status_color = AcademicColors.TEXT_WARNING if self.paused else AcademicColors.TEXT_SUCCESS
        status_label = self.font_normal.render(status_text, True, status_color)
        self.screen.blit(status_label, (col4_x, y))

        y += 28
        info_items = [
            (f"Alive: {stats['alive_count']}/{stats['total_fish']}", AcademicColors.TEXT_PRIMARY),
            (f"Food: {stats['food_remaining']}", AcademicColors.FOOD_FLOATING),
            (f"Eaten: {stats['total_food_eaten']}", AcademicColors.TEXT_SUCCESS),
            (f"Predation: {stats['ai_predation_events']}", AcademicColors.TEXT_DANGER),
        ]
        for i, (text, color) in enumerate(info_items):
            x_pos = [col1_x, col2_x, col3_x, col4_x][i]
            label = self.font_small.render(text, True, color)
            self.screen.blit(label, (x_pos, y))

        y += 22
        if stats['alive_count'] > 0:
            info_items2 = [
                (f"Avg Mass: {stats.get('avg_mass', 0):.1f}g", AcademicColors.CHART_MASS),
                (f"Avg Energy: {stats.get('avg_energy', 0):.0f}%", AcademicColors.CHART_ENERGY),
                (f"Total Mass: {stats.get('total_mass', 0):.0f}g", AcademicColors.TEXT_PRIMARY),
            ]
            for i, (text, color) in enumerate(info_items2):
                x_pos = [col1_x, col2_x, col3_x][i]
                label = self.font_small.render(text, True, color)
                self.screen.blit(label, (x_pos, y))

        y += 28
        controls = "SPACE=Pause | R=Reset | T=Time Mode | +/-/1-9=Speed | ESC=Quit"
        help_label = self.font_tiny.render(controls, True, AcademicColors.TEXT_DIM)
        self.screen.blit(help_label, (col1_x, y))

    def draw_legend(self):
        panel_x = self.margin
        panel_y = self.legend_top
        panel_width = self.left_col_width
        panel_height = self.legend_height
        content_y = self.draw_panel_background(panel_x, panel_y, panel_width, panel_height, "Legend")
        y = content_y + 3
        col1_x = panel_x + 12
        col2_x = panel_x + panel_width // 3 + 10
        col3_x = panel_x + 2 * panel_width // 3
        legend_items = [
            [(AcademicColors.NPC_SMALL,   "Small NPC"),  (AcademicColors.NPC_LARGE,    "Large NPC")],
            [((100, 200, 255),             "Ambient"),    ((255, 255, 100),              "Surface")],
            [((180, 120,  60),             "Benthic"),    ((100, 255, 150),              "Attached")],
            [(AcademicColors.SELECTED_GLOW,"Selected"),  (AcademicColors.NPC_PREDATOR, "Predator")],
        ]
        col_xs = [col1_x, col2_x, col3_x, col1_x + (panel_width - 24)]
        # 43+341
        row_offset = [0, 0, 0, 36]
        for col_idx, column in enumerate(legend_items):
            x = [col1_x, col2_x, col3_x, col1_x][col_idx]
            y_off = row_offset[col_idx]
            for i, (color, name) in enumerate(column):
                cy = y + y_off + i * 18
                pygame.draw.circle(self.screen, color, (x + 6, cy + 6), 5)
                text = self.font_tiny.render(name, True, AcademicColors.TEXT_SECONDARY)
                self.screen.blit(text, (x + 18, cy))

    def draw_title(self):
        model_name = os.path.basename(self.model_path) if self.model_path else "Random"
        title = f"Bass Fish Simulation v3.1 (Training-Aligned) | Model: {model_name[:30]}"
        title_text = self.font_title.render(title, True, AcademicColors.TEXT_PRIMARY)
        self.screen.blit(title_text, (self.margin, 12))

    def _sync_tank_info(self):
        """Sync tank info from environment"""
        if hasattr(self.manager, 'tank_geometry') and self.manager.tank_geometry is not None:
            geo = self.manager.tank_geometry
            self.tank_shape = geo.shape_name
            self.tank_depth = geo.depth
            extents = geo.get_extents()
            if self.tank_shape == 'circular':
                self.tank_radius = extents['radius']
                self.tank_width = self.tank_radius * 2
                self.tank_length = self.tank_radius * 2
                self.tank_half_x = self.tank_radius
            else:
                self.tank_width = extents['width']
                self.tank_length = extents['length']
                self.tank_radius = max(self.tank_width, self.tank_length) / 2
                self.tank_half_x = self.tank_width / 2  # X
            self._update_layout()

    def run(self, target_fps: int = 60):
        print("[START] Academic visualization v3.1 (Training-Aligned)")
        print("        R=Open Settings | SPACE=Pause | T=Time Mode | ESC=Quit")
        print("        Fixes: activity threshold, speed limits, NPC target lock")

        self.manager.reset()
        self._sync_tank_info()

        while self.running:
            if self.need_reset_dialog:
                self.need_reset_dialog = False
                self.paused = True
                self.show_reset_dialog()
                self.paused = False

            self.handle_events()

            if not self.paused:
                if self.realtime_mode:
                    if not hasattr(self, 'realtime_frame_counter'):
                        self.realtime_frame_counter = 0
                    self.realtime_frame_counter += 1
                    frames_per_step = max(1, int(target_fps * 0.3))
                    if self.realtime_frame_counter >= frames_per_step:
                        for _ in range(self.speed_multiplier):
                            self.manager.step()
                        self.realtime_frame_counter = 0
                else:
                    for _ in range(self.speed_multiplier):
                        self.manager.step()

            self.screen.fill(AcademicColors.BACKGROUND)
            self.draw_tank_top()
            self.draw_tank_side()
            self.draw_obstacles()
            self.draw_food()
            self.draw_npc_fish()
            self.draw_ai_fish()
            self.draw_legend()
            self.draw_fish_list_panel()
            self.draw_selected_fish_details()
            self.draw_realtime_charts()
            self.draw_simulation_info()
            self.draw_title()

            pygame.display.flip()
            self.clock.tick(target_fps)

        pygame.quit()

        stats = self.manager.get_statistics()
        print("\n" + "=" * 60)
        print("Final Statistics")
        print("=" * 60)
        for key, value in stats.items():
            print(f"  {key}: {value}")

        return stats


# ============================================================
# Main Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Bass Fish Simulation v3.1 (Training-Aligned)')
    parser.add_argument('--model', type=str, help='Model file path (optional)')
    parser.add_argument('--num_fish', type=int, default=10, help='Number of AI fish')
    parser.add_argument('--mass', type=float, default=20.0, help='Initial mass (g)')
    parser.add_argument('--fps', type=int, default=60, help='Target FPS')
    parser.add_argument('--no_npc', action='store_true', help='Hide NPC fish')
    parser.add_argument('--no_dialog', action='store_true', help='Skip settings dialog')
    parser.add_argument('--course', type=str, default='course4',
                       choices=['course1', 'course2', 'course3', 'course4'],
                       help='Training course (course2=maze with obstacles)')
    parser.add_argument('--layout', type=str, default='vertical_maze',
                       choices=['vertical_maze', 'corridor', 'reef', 'random'],
                       help='Obstacle layout for course2')

    args = parser.parse_args()

    print("=" * 70)
    print("  Bass Fish Behavior Simulation v3.1 - Training Aligned Edition")
    print("=" * 70)
    print("\n  Fixes in this version:")
    print("    1. Activity state threshold: action[3] < 0 (was < -0.3)")
    print("    2. Rest speed limit: 0.15 BL/s (was missing)")
    print("    3. NPC target lock: prevents chaotic chasing in fish groups")
    print("    4. Curriculum: Stage 7 (real difficulty)")
    print("    5. Predation probability: aligned with training env")
    print()

    if not args.no_dialog:
        settings = show_settings_dialog(
            default_num_fish=args.num_fish,
            default_mass=args.mass,
            default_model_path=args.model or ""
        )
        if settings is None:
            print("[CANCELLED] User cancelled the settings dialog")
            return
    else:
        settings = {
            'model_path': args.model,
            'num_fish': args.num_fish,
            'initial_mass': args.mass,
            'include_npc': not args.no_npc,
            'course': args.course,
            'layout': args.layout
        }

    print(f"  AI Fish Count : {settings['num_fish']}")
    print(f"  Initial Mass  : {settings['initial_mass']}g each")
    print(f"  Model         : {settings['model_path'] or 'Random Actions'}")
    print(f"  Include NPC   : {settings['include_npc']}")
    print(f"  Course        : {settings.get('course', 'course4')}")
    print(f"  Layout        : {settings.get('layout', 'vertical_maze')}")
    print(f"  Target FPS    : {args.fps}")
    print("=" * 70)

    visualizer = AcademicVisualizer(settings=settings)
    visualizer.run(target_fps=args.fps)


if __name__ == "__main__":
    main()
