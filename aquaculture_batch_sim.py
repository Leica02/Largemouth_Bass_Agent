#!/usr/bin/env python3
"""
Aquaculture Batch Simulation
=============================
Simulates a cohort of largemouth bass (Micropterus salmoides) reared in a
circular recirculating aquaculture tank.

Setup
-----
- Tank  : circular RAS tank, no obstacles, water current only
- Fish  : N individuals, initial mass uniform ~ [INIT_MASS_MIN, INIT_MASS_MAX] g
- Feed  : 2x per day, daily ration = random [DR_MIN, DR_MAX] of total biomass
- Steps : TOTAL_STEPS

Outputs
-------
Figures saved to  aquaculture_results/
  1. growth_curves.png     - individual & mean body mass over time
  2. final_weight_hist.png - terminal weight distribution
  3. survival_curve.png    - Kaplan-Meier style survival over time
  4. fcr_sgr.png           - daily FCR and SGR
  5. summary_table.png     - key aquaculture metrics table
  6. biomass_trend.png     - total biomass trend

All text is in English for publication-ready figures.

Implementation note
-------------------
Uses MultiAIFishManager from multi_fish_visualizer for physics/feeding/metabolism.
This fixed version keeps long-lived per-fish physiological state on each
AIFishState object, and uses the shared BassEnvironment only as a temporary
calculation workspace during each individual fish step.  Custom feeding logic
overrides the manager's built-in feeder so daily ration is drawn from
[DR_MIN, DR_MAX] BW. Pellets are queued and released as floating pellets near
the school centroid, so agents compete for a low-density trickle-feed stream.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from environment import BassEnvironment
from config import CONFIG
from multi_fish_visualizer import MultiAIFishManager, AIFishState
from utils.biological_formulas import mass_to_length

# ── time-acceleration patch ────────────────────────────────────
# The metabolism system scales all biological rates by
# CONFIG.environment.time_acceleration.  Default is 10 (1 step = 1 s).
# We override to 300 so that 1 step = 30 biological seconds →
# STEPS_PER_DAY = 86400/30 = 2880 ≈ 2820 (chosen for round numbers).
# This must be done before ANY import that reads CONFIG, so it is placed
# here, immediately after the CONFIG import.
_ENV_CFG = CONFIG.environment
object.__setattr__(_ENV_CFG, 'time_acceleration', 300)
# ── tank geometry patch ─────────────────────────────────────
# Target tank: circular, diameter 1.5 m → radius 0.75 m, water depth 0.7 m.
# Must be set here so create_default_tank() reads the correct values.
object.__setattr__(_ENV_CFG, 'tank_radius', 2.25)   # diameter 1.5 m
object.__setattr__(_ENV_CFG, 'tank_depth',  1.65)   # water depth 0.7 m

# ── simulation parameters ──────────────────────────────────────
MODEL_PATH       = "models/bass_ppo_behn22.zip"
NUM_FISH         = 120
INIT_MASS_MIN    = 54.3        # g
INIT_MASS_MAX    = 285.9        # g
FEEDINGS_PER_DAY = 2
DR_MIN           = 0.015       # min daily ration as fraction of live BW (1.5%)
DR_MAX           = 0.020       # max daily ration as fraction of live BW (2.0%)
TOTAL_DAYS       = 157          # simulated days
SIM_SECONDS_PER_STEP = CONFIG.environment.time_step * CONFIG.environment.time_acceleration
STEPS_PER_DAY    = int(round(86400.0 / SIM_SECONDS_PER_STEP))  # 2880 at dt=0.1, accel=300
TOTAL_STEPS      = TOTAL_DAYS * STEPS_PER_DAY
LOG_INTERVAL     = STEPS_PER_DAY
OUT_DIR          = "aquaculture_results"

# Explicit aquaculture perception settings. BassEnvironment.reset() may
# dynamically patch CONFIG.perception according to a random training mass;
# for batch grow-out we restore the intended cohort-level ranges after reset.
AQUA_VISION_RANGE       = 3.0
AQUA_FOOD_DETECT_RANGE  = 3.0
AQUA_LATERAL_LINE_RANGE = 0.5

# ── cannibalism parameters ─────────────────────────────────────
# Body-length ratio thresholds (predator_length / prey_length)
# Based on literature for largemouth bass (Micropterus salmoides):
#   fry     (<6 cm):   min ratio 1.20  (soft jaws, small injury sufficient)
#   juvenile (6-10cm): min ratio 1.30  (fingerling stage)
#   subadult(10-15cm): min ratio 1.40  (hardened scales, larger gap needed)
#   adult   (>15 cm):  min ratio 1.50  (gape limitation dominates)
# _cann_min_ratio() implements this size-dependent threshold.
CANN_MAX_LENGTH_RATIO   = 2.0    # above this ratio → gape limit, whole-swallow only
CANN_PERCEPTION_RANGE   = 3.0    # m  – matches full vision_range; predator scans whole tank
CANN_FLEE_RANGE         = 3.0    # m  – obs-injection range, same as vision
# Base probability per step; scales linearly with (actual_ratio - MIN) / (MAX - MIN)
CANN_BASE_ATTACK_PROB   = 0.008  # raised from 0.003 → more realistic contact rate
CANN_MAX_ATTACK_PROB    = 0.030  # raised from 0.015
# Energy transfer on successful attack
CANN_ENERGY_STEAL_MIN   = 0.10   # fraction stolen at min size ratio
CANN_ENERGY_STEAL_MAX   = 0.45   # fraction stolen at max size ratio
# Bite / swallow thresholds (replaces flat CANN_PREY_MASS_LOSS)
CANN_PREY_MASS_LOSS     = 0.05   # kept for backward compat (unused in new logic)
CANN_SWALLOW_RATIO      = 1.5    # length ratio above which predator can swallow whole
CANN_BITE_MASS_LOSS_MIN = 0.10   # bite damage at ratio just above MIN (10 % body mass)
CANN_BITE_MASS_LOSS_MAX = 0.20   # bite damage at ratio = SWALLOW (20 % body mass)
CANN_SWALLOW_PROB_BASE  = 0.35   # prob of successful whole-swallow at ratio = SWALLOW_RATIO
CANN_SWALLOW_PROB_MAX   = 0.80   # prob of successful whole-swallow at ratio = MAX
CANN_BITE_DEATH_THRESH  = 0.60   # prey dies when body_mass < initial_mass * this
CANN_ATTACK_COOLDOWN    = 50     # steps between attacks on the same prey (~5 real seconds)

# ── chronic malnutrition parameters ──────────────────────────────────────────
CMAL_MIN_RATION_PCT  = 0.008   # 3-day avg intake < 0.6 % BW/day → undernourished
                                # (lowered from 0.008; only extreme under-eaters die)
CMAL_AVG_PELLET_MASS = 0.0225  # g per pellet (midpoint of 0.015-0.030 g range)
CMAL_GRACE_DAYS      = 5       # days of deficiency before death risk starts
CMAL_LETHAL_DAYS     = 20      # days to reach max death prob after grace period
CMAL_DEATH_PROB_MIN  = 0.02    # daily death probability at start of lethal window
CMAL_DEATH_PROB_MAX  = 0.35    # daily death probability after LETHAL_DAYS

# ── stocking acclimation ──────────────────────────────────────────────────────
ACCLIMATION_DAYS     = 0       # no feeding delay — fish are fed from Day 1

# Observation injection indices (verified: total_dim=89)
# obs[13:43] = 3 fish slots × 10 dims each
# obs[43:64] = 3 food slots × 7 dims each  (dir_x,dir_y,dir_z, vel_x,vel_y,vel_z, dist_norm)
_OBS_FISH_START  = 13
_OBS_FISH_STRIDE = 10   # dims per fish slot
_OBS_FISH_SLOTS  = 3
_OBS_FOOD_START  = 43
_OBS_FOOD_STRIDE = 7    # dims per food slot
_OBS_FOOD_SLOTS  = 3
_FOOD_DETECT_RANGE = 3.0   # CONFIG.perception.food_detection_range


def _capture_radius(body_length: float) -> float:
    """Dynamic capture radius, identical to MultiAIFishManager._calculate_capture_radius."""
    ic = CONFIG.interaction
    if body_length <= ic.capture_radius_small_threshold:
        coef = ic.capture_radius_small
    elif body_length <= ic.capture_radius_medium_threshold:
        coef = ic.capture_radius_medium
    else:
        coef = ic.capture_radius_large
    return max(body_length * coef, ic.capture_radius_min)


def _cann_min_ratio(pred_length_m: float) -> float:
    """
    Size-dependent minimum predator/prey length ratio for cannibalism.

    Smaller fish can be cannibalized at lower ratios because their jaws are
    weaker and a lighter bite causes proportionally more damage.  As fish
    grow, hardened scales and stronger escape responses require a larger
    size advantage before an attack is worthwhile.

    Thresholds (Micropterus salmoides literature):
      fry     (<6 cm):    1.20  – soft tissue, slight size advantage enough
      juvenile (6-10 cm): 1.30  – fingerling stage (initial sim range)
      subadult(10-15 cm): 1.40  – scales harden, bigger gap needed
      adult   (>15 cm):   1.50  – gape limitation dominates
    """
    L_cm = pred_length_m * 100.0
    if   L_cm <  6.0: return 1.20
    elif L_cm < 10.0: return 1.30
    elif L_cm < 15.0: return 1.40
    else:             return 1.50


# ─────────────────────────────────────────────────────────────
# Custom feeding: override MultiAIFishManager._spawn_food
# so daily ration is random [DR_MIN, DR_MAX] of live biomass
# ─────────────────────────────────────────────────────────────

# ── trickle-feed parameters ───────────────────────────────────────────────
# Pellets are released every step at a rate matched to fish consumption speed.
# From empirical measurement: 100 fish consume ~8 pellets/step during active feeding.
# Target: spread the meal over ~80 steps (40 bio-minutes), keeping surface count
# low (10-30 pellets) so all fish have equal access rather than a few monopolising
# a large surface pile.
TRICKLE_STEPS       = 80    # release meal over this many steps (~40 bio-minutes)
# Pellets-per-step = total_meal_pellets / TRICKLE_STEPS  (computed dynamically)

# Floating pellets begin sinking after SINK_AFTER_STEPS steps (consistent with
# the base feeding system). In the multi-fish batch path (env_food_only=True),
# the system's batch-settle trigger is bypassed, so uneaten floating pellets
# would persist indefinitely in food_items, causing O(N_fish × N_pellets)
# computation to grow unboundedly over simulation days (progressive slowdown).
# Re-introducing the sink logic ensures uneaten feed settles to the bottom
# and is auto-removed, bounding food_items length effectively.
SINK_AFTER_STEPS    = 400   # steps before floating pellets begin sinking (~200 bio-minutes)


def _aqua_spawn_food(manager: MultiAIFishManager, rng: np.random.Generator) -> int:
    """
    Queue one meal worth of floating pellets for trickle release.

    Instead of spawning all pellets at once (which lets a few nearby fish
    eat the entire meal before others arrive), we store the total count in
    manager._pending_pellets and release them in small handfuls each step
    via _aqua_trickle_feed().  This mirrors the way a farmer broadcasts
    feed by hand — a little at a time, refilling as the fish consume it.
    """
    alive = [f for f in manager.ai_fish if f.is_alive]
    if not alive:
        return 0

    alive_biomass = sum(f.body_mass for f in alive)

    dr         = float(rng.uniform(DR_MIN, DR_MAX))
    per_meal_g = alive_biomass * dr / FEEDINGS_PER_DAY

    pellet_mass = float(rng.uniform(0.015, 0.030))
    n_total     = max(1, int(round(per_meal_g / pellet_mass)))

    # Store in queue instead of spawning immediately
    manager._pending_pellets     = getattr(manager, '_pending_pellets', 0) + n_total
    manager._meal_total_pellets  = n_total
    manager._pending_pellet_mass = pellet_mass
    manager._meal_total_g        = getattr(manager, '_meal_total_g', 0.0) + per_meal_g
    manager._trickle_step        = 0   # reset step counter for this meal

    # Reset feeding state step counter
    state = manager.base_env.feeding_state
    state.steps_since_feeding = 0
    state.current_batch_settling = False

    # Offered feed mass, not necessarily consumed feed mass.
    manager._total_feed_offered_g = getattr(manager, '_total_feed_offered_g', 0.0) + per_meal_g
    manager._total_feed_mass_g = manager._total_feed_offered_g  # backward-compatible alias
    return 0   # actual spawning happens in _aqua_trickle_feed each step


def _aqua_trickle_feed(manager: MultiAIFishManager, sim_step: int = 0) -> int:
    """
    Release pellets every step at a rate matched to fish consumption speed.

    Spreads the meal over TRICKLE_STEPS steps so the surface pellet count
    stays low (matching fish capture rate), giving all fish equal access
    rather than a few monopolising a large pile.
    """
    pending = getattr(manager, '_pending_pellets', 0)
    if pending <= 0:
        return 0

    meal_total = getattr(manager, '_meal_total_pellets', pending)
    # Pellets to release this step = ceil(meal_total / TRICKLE_STEPS)
    per_step = max(1, int(np.ceil(meal_total / TRICKLE_STEPS)))
    release  = min(pending, per_step)

    pellet_mass = getattr(manager, '_pending_pellet_mass', 0.0225)
    fs    = manager.base_env.feeding_system
    state = manager.base_env.feeding_state

    from systems.feeding import FeedingInput
    alive = [f for f in manager.ai_fish if f.is_alive]
    ref_mass = sum(f.body_mass for f in alive) if alive else 1.0
    # Use fish school centroid so pellets land near where the fish actually are
    if alive:
        centroid = np.mean([f.position for f in alive], axis=0).astype(np.float32)
        centroid[1] = -0.05  # surface, just above fish depth
    else:
        centroid = np.array([0.0, -0.05, 0.0], dtype=np.float32)
    feed_input = FeedingInput(
        agent_position=centroid,
        agent_mass=ref_mass,
        agent_length=0.10,
        stomach_fullness=0.0,
        tank_geometry=getattr(manager.base_env, 'tank_geometry', None),
        obstacle_field=getattr(manager.base_env, 'obstacle_field', None),
    )

    # Spread feed over a patch that scales with tank size, so the feed disk
    # covers a constant fraction (~28%) of tank area regardless of tank radius.
    # 0.53 * R reproduces the old 0.40m patch in the 0.75m training tank,
    # but widens to ~1.19m in the 2.25m tank so small fish are no longer
    # shut out of a tiny central feeding spot.
    fs_cfg = manager.base_env.feeding_system.c
    _tank_geo = getattr(manager.base_env, 'tank_geometry', None)
    if _tank_geo is not None:
        _tank_r = float(_tank_geo.get_extents().get('radius', CONFIG.environment.tank_radius))
    else:
        _tank_r = float(CONFIG.environment.tank_radius)
    _spread = 0.53 * _tank_r
    # floor at 0.40m (preserve small-tank behaviour), keep pellets inside the wall
    _spread = min(max(_spread, 0.40), _tank_r - fs_cfg.boundary_buffer)
    _orig_spread = fs_cfg.spread_radius
    object.__setattr__(fs_cfg, 'spread_radius', _spread)

    spawned = 0
    for _ in range(release):
        food = fs._create_floating_pellet(feed_input, pellet_mass)
        # Record spawn timestamp for SINK_AFTER_STEPS sink trigger.
        # Uses the batch main-loop step counter (not food.age, which may be
        # incremented multiple times per step in multi-fish path).
        food._aqua_birth_step = sim_step
        state.food_items.append(food)
        spawned += 1

    object.__setattr__(fs_cfg, 'spread_radius', _orig_spread)  # restore

    manager._pending_pellets -= spawned
    manager.total_food_spawned += spawned
    return spawned


def _aqua_sink_aged_pellets(manager: MultiAIFishManager, sim_step: int,
                            sink_after: int = SINK_AFTER_STEPS) -> int:
    “””Sink floating pellets that have exceeded sink_after steps without capture.

    Reuses the base system's settle mechanism (is_settling=True, food_type=SETTLING,
    downward velocity). Once pellets reach the tank bottom, feeding_system's
    _update_food_movement auto-removes them (counted as total_food_wasted).

    Effect: uneaten feed does not persist on the surface indefinitely; food_items
    length is bounded, preventing O(N_fish x N_pellets) from growing over time.
    Only affects trickle-spawned FLOATING pellets; environment food is unaffected.
    “””
    from systems.feeding import FoodType
    state = manager.base_env.feeding_state
    cfg   = manager.base_env.feeding_system.c
    settle_speed = float(getattr(cfg, 'settle_speed_floating', 0.002))

    sunk = 0
    for food in state.food_items:
        if (food.food_type == FoodType.FLOATING
                and not food.is_settling
                and (sim_step - getattr(food, '_aqua_birth_step', sim_step)) >= sink_after):
            # Identical to _trigger_all_food_settling sink initialization
            food.is_settling  = True
            food.food_type    = FoodType.SETTLING
            food.settle_speed = settle_speed
            food.velocity = np.array([
                food.velocity[0] * 0.3,
                -settle_speed,
                food.velocity[2] * 0.3,
            ], dtype=np.float32)
            sunk += 1
    return sunk


# ─────────────────────────────────────────────────────────────
# Per-fish state isolation patches
# ─────────────────────────────────────────────────────────────

def _activity_state_for_growth(growth_state, fish_activity_state):
    """Convert the fish's ActivityState enum to the GrowthState enum class.

    growth.py and metabolism.py may define separate ActivityState classes.
    Comparing enum members from different classes returns False even when
    their names are identical. This helper prevents rest-growth accounting
    from silently failing.
    """
    if not hasattr(growth_state, 'current_activity_state'):
        return fish_activity_state
    enum_cls = type(growth_state.current_activity_state)
    state_name = getattr(fish_activity_state, 'name', str(fish_activity_state).split('.')[-1])
    if hasattr(enum_cls, state_name):
        return getattr(enum_cls, state_name)
    return growth_state.current_activity_state


def _ensure_growth_aux_state(fish) -> None:
    """Attach GrowthState auxiliary fields to each fish if absent.

    MultiAIFishManager already stores core growth fields per fish, but
    GrowthState also contains threshold/accounting fields used by
    sync_length_from_mass(). They must be stored per individual too.
    """
    defaults = {
        'last_length_update_mass': float(getattr(fish, 'body_mass', 0.0)),
        'length_shrink_count': 0,
        'growth_during_rest': 0.0,
        'growth_count_during_rest': 0,
        'energy_added_during_rest': 0.0,
    }
    for name, value in defaults.items():
        if not hasattr(fish, name):
            setattr(fish, name, value)
    if getattr(fish, 'last_length_update_mass', 0.0) <= 0.0:
        fish.last_length_update_mass = float(getattr(fish, 'body_mass', 0.0))


def _sync_growth_aux_to_env(manager: MultiAIFishManager, fish) -> None:
    """Copy per-fish GrowthState auxiliary fields into the shared env workspace."""
    _ensure_growth_aux_state(fish)
    gs = manager.base_env.growth_state
    if hasattr(gs, 'last_length_update_mass'):
        gs.last_length_update_mass = float(fish.last_length_update_mass)
    if hasattr(gs, 'length_shrink_count'):
        gs.length_shrink_count = int(fish.length_shrink_count)
    if hasattr(gs, 'growth_during_rest'):
        gs.growth_during_rest = float(fish.growth_during_rest)
    if hasattr(gs, 'growth_count_during_rest'):
        gs.growth_count_during_rest = int(fish.growth_count_during_rest)
    if hasattr(gs, 'energy_added_during_rest'):
        gs.energy_added_during_rest = float(fish.energy_added_during_rest)
    if hasattr(gs, 'current_activity_state'):
        gs.current_activity_state = _activity_state_for_growth(gs, fish.activity_state)


def _sync_growth_aux_from_env(manager: MultiAIFishManager, fish) -> None:
    """Copy GrowthState auxiliary fields back from env workspace to this fish."""
    _ensure_growth_aux_state(fish)
    gs = manager.base_env.growth_state
    for name in (
        'last_length_update_mass',
        'length_shrink_count',
        'growth_during_rest',
        'growth_count_during_rest',
        'energy_added_during_rest',
    ):
        if hasattr(gs, name):
            setattr(fish, name, getattr(gs, name))


def _sync_metabolism_state_back_to_fish(manager: MultiAIFishManager, fish) -> None:
    """Fully persist the shared metabolism workspace back to one fish.

    This is especially important immediately after food capture, because
    add_food_to_stomach() updates diet composition and ADC fields in
    metabolism_state. If these are not copied back, the next fish step can
    overwrite them.
    """
    ms = manager.base_env.metabolism_state
    fields = (
        'energy', 'stomach_fullness', 'stomach_content_mass', 'initial_meal_mass',
        'digestion_buffer', 'energy_from_digestion', 'activity_state',
        'rest_duration_steps', 'current_metabolism_factor', 'current_growth_bonus',
        'fatigue', 'stress_level', 'is_digesting', 'stomach_protein_fraction',
        'stomach_lipid_fraction', 'stomach_carbohydrate_fraction',
        'stomach_adc_protein', 'stomach_adc_lipid', 'stomach_adc_carbohydrate',
        'stomach_include_carbohydrate_energy', 'state_switch_cooldown',
        'last_state_switch_step', 'total_rest_steps', 'initial_body_mass',
        'lipid_reserve', 'protein_reserve', 'death_mass_loss_threshold',
    )
    for name in fields:
        if hasattr(ms, name):
            setattr(fish, name, getattr(ms, name))


def _patch_manager_state_isolation(mgr: MultiAIFishManager) -> None:
    """Patch manager methods without editing multi_fish_visualizer.py.

    The manager uses one BassEnvironment as a calculation workspace. These
    wrappers make the additional growth and feeding-related metabolism fields
    round-trip per fish.
    """
    import types

    original_sync = mgr._sync_env_state.__func__
    original_execute = mgr._execute_action.__func__
    original_capture = mgr._check_food_capture.__func__

    def patched_sync(self_mgr, fish):
        original_sync(self_mgr, fish)
        _sync_growth_aux_to_env(self_mgr, fish)

    def patched_execute(self_mgr, fish, action):
        _sync_growth_aux_to_env(self_mgr, fish)
        original_execute(self_mgr, fish, action)
        _sync_growth_aux_from_env(self_mgr, fish)
        _sync_metabolism_state_back_to_fish(self_mgr, fish)

    def patched_capture(self_mgr, fish):
        original_capture(self_mgr, fish)
        _sync_metabolism_state_back_to_fish(self_mgr, fish)

    mgr._sync_env_state = types.MethodType(patched_sync, mgr)
    mgr._execute_action = types.MethodType(patched_execute, mgr)
    mgr._check_food_capture = types.MethodType(patched_capture, mgr)


# ─────────────────────────────────────────────────────────────
# Observation injection helpers
# ─────────────────────────────────────────────────────────────

def _inject_cannibalism_obs(obs: np.ndarray,
                             ego_fish,
                             all_fish,
                             rng: np.random.Generator) -> np.ndarray:
    """
    Post-process the observation vector of one fish to encode
    cannibalism-relevant conspecific information.

    FOR THE PREDATOR (ego_fish is larger):
      A nearby, sufficiently smaller conspecific is injected into the
      *first available food slot* of the observation, making the model
      perceive it as a nearby food pellet.  The direction and distance
      are encoded identically to a real pellet, so the model naturally
      steers toward it.

    FOR THE PREY (ego_fish is smaller):
      A nearby, sufficiently larger conspecific is injected into the
      first fish slot with is_threat=1.0, which the model was trained
      to avoid.  (The perception system already does this via
      _inject_other_ai_fish, but this function ensures the threat flag
      is correctly set even when the size gap is large enough to trigger
      cannibalism concern.)

    Only the *closest* qualifying conspecific is injected to avoid
    overwriting multiple real stimuli.
    """
    obs = obs.copy()
    ego_pos = ego_fish.position
    ego_L   = ego_fish.total_length
    ego_m   = ego_fish.body_mass
    min_ratio = _cann_min_ratio(ego_L)   # dynamic threshold for this fish's size

    # --- collect candidates ---
    prey_candidates    = []   # (dist, ratio, fish) where ego is predator
    threat_candidates  = []   # (dist, fish) where ego is prey

    for other in all_fish:
        if other is ego_fish or not other.is_alive:
            continue
        dist = float(np.linalg.norm(other.position - ego_pos))
        if dist > CANN_FLEE_RANGE:   # = vision_range = 3.0m, whole-tank detection
            continue
        ratio = ego_L / max(other.total_length, 1e-6)
        if ratio >= min_ratio:       # ego is predator
            prey_candidates.append((dist, ratio, other))
        inv_ratio = other.total_length / max(ego_L, 1e-6)
        if inv_ratio >= _cann_min_ratio(other.total_length):   # other is predator
            threat_candidates.append((dist, other))

    # --- inject nearest prey as fake food pellet (probability ∝ size advantage) ---
    if prey_candidates:
        prey_candidates.sort(key=lambda x: x[0])
        dist, length_ratio, prey = prey_candidates[0]

        # Injection probability: larger size gap → more likely to "mistake" prey for food
        # At min_ratio → 0.2 (rare misidentification), at MAX_ratio → 1.0 (obvious prey)
        t_ratio = min(1.0, (length_ratio - min_ratio) /
                      max(1e-6, CANN_MAX_LENGTH_RATIO - min_ratio))
        inject_prob = 0.20 + t_ratio * 0.80

        if rng.random() < inject_prob:
            rel = (prey.position - ego_pos)
            d   = float(np.linalg.norm(rel)) + 1e-8
            dir_vec  = rel / d
            dist_n   = float(np.clip(d / _FOOD_DETECT_RANGE, 0.0, 1.0))

            # find first food slot that is empty (dist_norm == 0) or furthest
            best_slot = 0
            best_dist = -1.0
            for slot in range(_OBS_FOOD_SLOTS):
                s = _OBS_FOOD_START + slot * _OBS_FOOD_STRIDE
                slot_dist = float(obs[s + 6])    # dist_norm is last field
                if slot_dist == 0.0:             # empty slot
                    best_slot = slot
                    break
                if slot_dist > best_dist:
                    best_dist = slot_dist
                    best_slot = slot

            if best_slot >= 0:
                s = _OBS_FOOD_START + best_slot * _OBS_FOOD_STRIDE
                obs[s:s+3] = dir_vec.astype(np.float32)
                obs[s+3:s+6] = np.zeros(3, dtype=np.float32)   # prey velocity looks zero
                obs[s+6] = np.float32(dist_n)

    # --- ensure nearest threat has is_threat flag set correctly ---
    if threat_candidates:
        threat_candidates.sort(key=lambda x: x[0])
        dist, threat = threat_candidates[0]

        rel    = (threat.position - ego_pos)
        d      = float(np.linalg.norm(rel)) + 1e-8
        dir_vec = rel / d
        dist_n  = float(np.clip(d / float(CONFIG.perception.vision_range), 0.0, 1.0))
        mass_ratio = float(np.clip(
            (threat.body_mass - ego_m) / max(ego_m, 1e-6) / 5.0, -0.4, 1.0))

        # Use first fish slot (threat priority slot)
        s = _OBS_FISH_START
        obs[s:s+3]   = dir_vec.astype(np.float32)
        obs[s+3:s+6] = np.zeros(3, dtype=np.float32)
        obs[s+6]     = np.float32(dist_n)
        obs[s+7]     = np.float32(mass_ratio)
        obs[s+8]     = np.float32(1.0)   # is_threat = 1
        obs[s+9]     = np.float32(0.0)   # is_prey   = 0

    return obs


def _patch_manager_obs_injection(mgr: MultiAIFishManager,
                                  rng: np.random.Generator) -> None:
    """
    Monkey-patch MultiAIFishManager so that after the normal observation
    is built, cannibalism-relevant conspecific information is injected
    before the model makes its decision.
    """
    original_get_obs = mgr._get_observation_simple.__func__  # unbound

    def patched_get_obs(self_mgr, fish):
        obs = original_get_obs(self_mgr, fish)
        return _inject_cannibalism_obs(obs, fish, self_mgr.ai_fish, rng)

    # bind as method
    import types
    mgr._get_observation_simple = types.MethodType(patched_get_obs, mgr)


# ─────────────────────────────────────────────────────────────
# Cannibalism module
# ─────────────────────────────────────────────────────────────

def _apply_bite_damage(prey, t: float, rng: np.random.Generator) -> bool:
    """
    Apply bite damage to prey.  t = 0 at MIN ratio, 1 at MAX ratio.
    Returns True if the prey dies from this bite.
    """
    mass_loss_frac = (CANN_BITE_MASS_LOSS_MIN
                      + t * (CANN_BITE_MASS_LOSS_MAX - CANN_BITE_MASS_LOSS_MIN))
    prey.body_mass = max(0.0, prey.body_mass * (1.0 - mass_loss_frac))
    if prey.body_mass < prey.initial_mass * CANN_BITE_DEATH_THRESH:
        prey.is_alive = False
        prey.death_reason = 'cannibalism_wounded'
        return True
    return False


def _process_cannibalism(manager: MultiAIFishManager,
                         rng: np.random.Generator) -> dict:
    """
    Simulate intraspecific predation (cannibalism) among AI fish.

    Fixed version:
      - checks every living fish as a potential predator each step;
      - returns a consistent swallows counter;
      - updates body length after bite or swallow mass changes;
      - keeps attack cooldowns per predator-prey pair.
    """
    alive = [f for f in manager.ai_fish if f.is_alive]
    if len(alive) < 2:
        return {'attacks': 0, 'successes': 0, 'deaths': 0, 'swallows': 0}

    attacks = 0
    successes = 0
    deaths = 0
    swallows = 0

    # Tick down all cooldown counters first
    for f in alive:
        f.attack_cooldowns = {k: v - 1 for k, v in f.attack_cooldowns.items() if v > 1}

    for predator in alive:
        if not predator.is_alive:
            continue

        pred_L = predator.total_length  # metres
        min_ratio = _cann_min_ratio(pred_L)  # size-dependent threshold

        for prey in alive:
            if prey is predator or not prey.is_alive:
                continue

            prey_L = prey.total_length
            if prey_L <= 0:
                continue

            length_ratio = pred_L / prey_L

            # Size check: predator must be meaningfully larger.
            if length_ratio < min_ratio:
                continue

            # Attack cooldown: skip if this predator recently attacked this prey.
            if predator.attack_cooldowns.get(prey.id, 0) > 0:
                continue

            # Distance check: prey must be within perception range.
            dist = float(np.linalg.norm(predator.position - prey.position))
            if dist > CANN_PERCEPTION_RANGE:
                continue

            # Attack probability scales with size advantage above the dynamic threshold.
            t = min(1.0, (length_ratio - min_ratio) /
                    max(1e-6, CANN_MAX_LENGTH_RATIO - min_ratio))
            attack_prob = CANN_BASE_ATTACK_PROB + t * (CANN_MAX_ATTACK_PROB - CANN_BASE_ATTACK_PROB)

            if rng.random() > attack_prob:
                continue

            attacks += 1

            # Strike range: same dynamic capture radius as normal feeding.
            if dist > _capture_radius(predator.total_length):
                continue

            successes += 1
            predator.attack_cooldowns[prey.id] = CANN_ATTACK_COOLDOWN

            if length_ratio >= CANN_SWALLOW_RATIO:
                # Whole-swallow path.
                t_sw = min(1.0, (length_ratio - CANN_SWALLOW_RATIO) /
                           max(1e-6, CANN_MAX_LENGTH_RATIO - CANN_SWALLOW_RATIO))
                swallow_prob = (CANN_SWALLOW_PROB_BASE
                                + t_sw * (CANN_SWALLOW_PROB_MAX - CANN_SWALLOW_PROB_BASE))
                if rng.random() < swallow_prob:
                    prey.is_alive = False
                    prey.death_reason = 'cannibalism_swallowed'
                    deaths += 1
                    swallows += 1

                    # Route cannibalism through the digestion pipeline instead of
                    # directly adding mass.  Prey tissue is protein-rich animal flesh.
                    _FISH_PREY_PROFILE = {
                        'protein_fraction': 0.70,
                        'lipid_fraction': 0.12,
                        'carbohydrate_fraction': 0.00,
                        'adc_protein': 0.90,
                        'adc_lipid': 0.85,
                        'adc_carbohydrate': 0.30,
                        'include_carbohydrate_energy': False,
                    }
                    # ~80 % of prey body mass is edible (excludes skeleton, scales, etc.)
                    edible_mass = prey.body_mass * 0.80
                    stomach_capacity = predator.body_mass * CONFIG.feeding.stomach_capacity_ratio
                    available_space = max(0.0, stomach_capacity
                                         - getattr(predator, 'stomach_content_mass', 0.0))
                    food_added = min(edible_mass, available_space)

                    if food_added > 0:
                        predator.stomach_content_mass = (
                            getattr(predator, 'stomach_content_mass', 0.0) + food_added
                        )
                        predator.stomach_fullness = min(
                            100.0,
                            predator.stomach_content_mass / stomach_capacity * 100
                        )
                        # Set prey-flesh diet profile on predator stomach
                        predator.stomach_protein_fraction     = _FISH_PREY_PROFILE['protein_fraction']
                        predator.stomach_lipid_fraction       = _FISH_PREY_PROFILE['lipid_fraction']
                        predator.stomach_carbohydrate_fraction = _FISH_PREY_PROFILE['carbohydrate_fraction']
                        predator.stomach_adc_protein          = _FISH_PREY_PROFILE['adc_protein']
                        predator.stomach_adc_lipid            = _FISH_PREY_PROFILE['adc_lipid']
                        predator.stomach_adc_carbohydrate     = _FISH_PREY_PROFILE['adc_carbohydrate']
                        predator.stomach_include_carbohydrate_energy = False

                    # Small immediate energy boost (excitement / blood ingestion)
                    predator.energy = min(100.0, predator.energy + prey.energy * 0.15)
                    continue

                # Swallow fails: fall back to bite injury.
                died = _apply_bite_damage(prey, t, rng)
                prey.total_length = mass_to_length(max(prey.body_mass, 1.0))
                if died:
                    deaths += 1
            else:
                # Bite path.
                steal_frac = CANN_ENERGY_STEAL_MIN + t * (CANN_ENERGY_STEAL_MAX - CANN_ENERGY_STEAL_MIN)
                predator.energy = min(100.0, predator.energy + prey.energy * steal_frac * 0.3)
                died = _apply_bite_damage(prey, t, rng)
                prey.total_length = mass_to_length(max(prey.body_mass, 1.0))
                if died:
                    deaths += 1

    return {'attacks': attacks, 'successes': successes, 'deaths': deaths, 'swallows': swallows}


# ─────────────────────────────────────────────────────────────
# Main simulation class
# ─────────────────────────────────────────────────────────────

class AquacultureSimulation:

    def __init__(self, model_path=MODEL_PATH, num_fish=NUM_FISH,
                 total_steps=TOTAL_STEPS, seed=43):
        self.num_fish    = num_fish
        self.total_steps = total_steps
        self.rng         = np.random.default_rng(seed)

        print(f"Loading model: {model_path}")
        self.model = PPO.load(model_path)

        # Base environment: no obstacles, no predators (course1)
        self.env = BassEnvironment({
            'verbose'           : 0,
            'disable_obstacles' : True,
            'force_default_tank': True,
            'training_phase'    : 'course1',
        })
        self.env.reset(seed=seed)
        object.__setattr__(CONFIG.perception, 'vision_range', AQUA_VISION_RANGE)
        object.__setattr__(CONFIG.perception, 'food_detection_range', AQUA_FOOD_DETECT_RANGE)
        object.__setattr__(CONFIG.perception, 'lateral_line_range', AQUA_LATERAL_LINE_RANGE)

        # Build the manager with randomised initial masses
        masses = self.rng.uniform(INIT_MASS_MIN, INIT_MASS_MAX, num_fish)
        self.manager = self._build_manager(masses)

        # time-series data
        self.ts_step        : List[int]   = []
        self.ts_day         : List[float] = []
        self.ts_alive       : List[int]   = []
        self.ts_mean_mass   : List[float] = []
        self.ts_total_biomass: List[float] = []
        self.mass_snapshots : List[np.ndarray] = []  # shape (T, N)

        # daily metrics
        self.daily_fcr : List[float] = []
        self.daily_sgr : List[float] = []
        self.daily_feed_offered_g : List[float] = []
        self.daily_feed_eaten_g : List[float] = []
        self.daily_positive_gain_g : List[float] = []
        self.daily_biomass_delta_g : List[float] = []
        self._day_biomass_start  = sum(f.body_mass for f in self.manager.ai_fish if f.is_alive)
        self._day_alive_start    = sum(1 for f in self.manager.ai_fish if f.is_alive)
        self._day_feed_start_g   = 0.0
        self._day_mass_start     = {f.id: f.body_mass for f in self.manager.ai_fish}

        # per-fish daily records (one dict per fish per day)
        self._fish_daily_rows : List[dict] = []
        self._day_eaten_start : Dict[int, int] = {
            f.id: f.food_eaten for f in self.manager.ai_fish
        }

        # cannibalism counters
        self._cann_attacks_total   = 0
        self._cann_successes_total = 0
        self._cann_deaths_total    = 0
        self._cann_swallows_total  = 0
        self._daily_cann_attacks   : List[int] = []
        self._daily_cann_successes : List[int] = []
        self._day_cann_attacks     = 0
        self._day_cann_successes   = 0

        # malnutrition death counter
        self._mal_deaths_total     = 0

        # action recording (opt-in via enable_action_recording())
        self._action_recording : bool = False
        self._action_interval  : int  = 10
        self._action_rows      : List[dict] = []

        self._record_snapshot()

    def _build_manager(self, masses: np.ndarray) -> MultiAIFishManager:
        """
        Create a MultiAIFishManager then override each fish's mass
        with our randomised draw.
        """
        mean_mass = float(masses.mean())
        mgr = MultiAIFishManager(
            base_env     = self.env,
            model        = self.model,
            num_fish     = self.num_fish,
            initial_mass = mean_mass,
            include_npc  = False,   # no NPC fish, pure grow-out
            verbose      = False,   # suppress per-step growth/feed/death prints
        )
        # reset() populates ai_fish and calls base_env.reset() + _initial_feeding()
        mgr.reset()
        object.__setattr__(CONFIG.perception, 'vision_range', AQUA_VISION_RANGE)
        object.__setattr__(CONFIG.perception, 'food_detection_range', AQUA_FOOD_DETECT_RANGE)
        object.__setattr__(CONFIG.perception, 'lateral_line_range', AQUA_LATERAL_LINE_RANGE)

        # Clear the food spawned by _initial_feeding() — our custom feeder
        # will supply all meals so we don't want a free pre-loaded batch that
        # inflates Day-1 pellet counts and gives an unfair head-start.
        mgr.base_env.feeding_state.food_items = []
        mgr.total_food_spawned = 0
        mgr.total_food_eaten   = 0
        mgr._total_feed_offered_g = 0.0
        mgr._total_feed_mass_g = 0.0

        # Override each fish's mass with our randomised draw
        # Also reset stomach to empty — fish just stocked, not mid-digestion
        # And re-calibrate swimbladder gas to match INIT_DEPTH so the buoyancy
        # system starts in equilibrium (avoids the random-depth → wrong gas →
        # physics overcorrection → fish stuck at bottom problem).
        INIT_DEPTH = 0.30   # m below surface; matches position[1] = -0.30
        _bc = CONFIG.buoyancy  # buoyancy config constants
        _GAS_CONSTANT = 8.314  # J/(mol·K)
        for fish, m in zip(mgr.ai_fish, masses):
            fish.body_mass             = float(m)
            fish.initial_mass          = float(m)
            fish.total_length          = mass_to_length(float(m))
            fish.position[1]           = -INIT_DEPTH
            fish.stomach_content_mass  = 0.0
            fish.stomach_fullness      = 0.0
            fish.initial_meal_mass     = 0.0
            fish.is_digesting          = False
            fish.smr_individual_factor = float(
                np.clip(self.rng.normal(1.0, 0.12), 0.70, 1.35)
            )

            # Recalculate swimbladder gas for INIT_DEPTH neutral buoyancy.
            # Without this the gas was calibrated to the random reset() depth,
            # causing a large buoyancy error on step 1 that drives fish to the bottom.
            mass_kg       = float(m) / 1000.0
            tissue_vol    = mass_kg / _bc.fish_tissue_density          # m³
            neutral_vol   = max(mass_kg / _bc.water_density - tissue_vol,
                                tissue_vol * _bc.min_volume_ratio)
            pressure_atm  = _bc.atmospheric_pressure + INIT_DEPTH * _bc.pressure_per_meter
            pressure_pa   = pressure_atm * 101325.0
            temp_k        = 25.0 + 273.15
            gas_amount    = (pressure_pa * neutral_vol) / (_GAS_CONSTANT * temp_k)

            # IMPORTANT: update this fish's private buoyancy state, not the
            # environment's last-used shared pointer. _sync_env_state() will
            # later reference-swap ps.buoyancy_state to fish._buoyancy_state.
            bs = getattr(fish, '_buoyancy_state', None)
            if bs is None:
                bs = getattr(mgr.base_env.physics_state, 'buoyancy_state', None)
            if bs is not None:
                bs.neutral_volume     = neutral_vol
                bs.swimbladder_volume = neutral_vol
                bs.gas_amount         = gas_amount
                bs.current_depth      = INIT_DEPTH
                bs.relative_density   = 1.0
                bs.net_buoyancy_force = 0.0
                bs.is_neutral         = True
            # Mirror onto AIFishState so the first sync-back is also clean
            fish.swimbladder_volume = neutral_vol
            fish.relative_density   = 1.0
            fish.net_buoyancy_force = 0.0

            # Reset per-fish digestive nutrient profile and growth auxiliary
            # fields so the shared env workspace cannot leak previous fish state.
            fish.digestion_buffer = 0.0
            fish.energy_from_digestion = 0.0
            fish.stomach_protein_fraction = 0.0
            fish.stomach_lipid_fraction = 0.0
            fish.stomach_carbohydrate_fraction = 0.0
            fish.stomach_adc_protein = 0.0
            fish.stomach_adc_lipid = 0.0
            fish.stomach_adc_carbohydrate = 0.0
            fish.stomach_include_carbohydrate_energy = False
            fish.initial_body_mass = 0.0
            fish.lipid_reserve = 0.0
            fish.protein_reserve = 0.0
            fish.death_mass_loss_threshold = 0.275
            fish.last_length_update_mass = float(m)
            fish.length_shrink_count = 0
            fish.growth_during_rest = 0.0
            fish.growth_count_during_rest = 0
            fish.energy_added_during_rest = 0.0

        # Set feeding interval BEFORE touching steps_since_feeding so the
        # trigger threshold is consistent from the very first step.
        feed_interval = STEPS_PER_DAY // FEEDINGS_PER_DAY
        mgr.feeding_interval    = feed_interval
        # Delay first feeding by ACCLIMATION_DAYS (fish adapt to tank before fed)
        mgr.steps_since_feeding = -ACCLIMATION_DAYS * STEPS_PER_DAY
        _rng = self.rng  # capture for closure

        def _custom_spawn(self_mgr=mgr, rng=_rng):
            return _aqua_spawn_food(self_mgr, rng)

        mgr._spawn_food = _custom_spawn

        # Patch manager methods so growth auxiliary state and the complete
        # post-capture metabolism state round-trip per fish.
        _patch_manager_state_isolation(mgr)

        # ── Observation injection patch ──────────────────────────────────
        _patch_manager_obs_injection(mgr, self.rng)

        return mgr

    def _record_snapshot(self):
        step  = self.manager.current_step
        alive = [f for f in self.manager.ai_fish if f.is_alive]
        masses = np.array([f.body_mass for f in alive]) if alive else np.array([0.0])

        self.ts_step.append(step)
        self.ts_day.append(step / STEPS_PER_DAY)
        self.ts_alive.append(len(alive))
        self.ts_mean_mass.append(float(masses.mean()) if alive else 0.0)
        self.ts_total_biomass.append(float(masses.sum()))

        snap = np.full(self.num_fish, np.nan)
        for f in self.manager.ai_fish:
            if f.is_alive:
                snap[f.id] = f.body_mass
        self.mass_snapshots.append(snap)

    def _calc_smr_kj_day(self, fish) -> float:
        """Calculate daily SMR (kJ/day) for this fish at current body mass (for logging)."""
        import math
        c = CONFIG.metabolism
        factor = getattr(fish, 'smr_individual_factor', 1.0)
        T = CONFIG.environment.water_temp
        smr_per_g = (c.smr_coefficient
                     * (fish.body_mass ** c.smr_exponent)
                     * math.exp(c.smr_temp_coeff * T + c.smr_swim_coeff * c.smr_swim_speed))
        return smr_per_g * fish.body_mass * c.oxycalorific_coeff * 24 * factor

    # ── daily FCR / SGR ───────────────────────────────────────
    def _record_daily(self):
        alive = [f for f in self.manager.ai_fish if f.is_alive]
        end_biomass = sum(f.body_mass for f in alive)
        biomass_delta = end_biomass - self._day_biomass_start

        # Feed offered is retained for ration accounting, but FCR for tiny fish
        # is calculated from actual pellets eaten. Offered-feed deltas can be
        # misleading when meals are extremely small or uneaten pellets remain.
        total_feed_g = getattr(self.manager, '_total_feed_offered_g',
                               getattr(self.manager, '_total_feed_mass_g', 0.0))
        feed_offered_g = total_feed_g - self._day_feed_start_g

        # Actual feed eaten this day = per-fish pellet count increment × mean pellet mass.
        # This matches the per-fish daily CSV field pellets_eaten_today and is the
        # robust FCR numerator for small juveniles.
        feed_eaten_g = sum(
            (f.food_eaten - self._day_eaten_start.get(f.id, 0)) * CMAL_AVG_PELLET_MASS
            for f in self.manager.ai_fish
        )

        # Positive individual gain avoids treating mortality biomass loss or
        # weight loss in underfed fish as negative growth in FCR.
        positive_gain_g = sum(
            max(0.0, f.body_mass - self._day_mass_start.get(f.id, f.body_mass))
            for f in alive
        )

        fcr = (feed_eaten_g / positive_gain_g
               if (positive_gain_g > 0.01 and feed_eaten_g > 0) else np.nan)

        n_end = len(alive)
        if n_end > 0 and self._day_alive_start > 0 and self._day_biomass_start > 0:
            m_start = self._day_biomass_start / max(1, self._day_alive_start)
            m_end   = end_biomass / max(1, n_end)
            sgr = (np.log(m_end) - np.log(m_start)) * 100 if m_end > 0 and m_start > 0 else np.nan
        else:
            sgr = np.nan

        self.daily_fcr.append(fcr)
        self.daily_sgr.append(sgr)
        self.daily_feed_offered_g.append(feed_offered_g)
        self.daily_feed_eaten_g.append(feed_eaten_g)
        self.daily_positive_gain_g.append(positive_gain_g)
        self.daily_biomass_delta_g.append(biomass_delta)

        # ── per-fish daily rows ────────────────────────────────
        day_num = len(self.daily_fcr)

        # ── chronic malnutrition death check ──────────────────
        for f in self.manager.ai_fish:
            if not f.is_alive:
                continue
            pellets_today = f.food_eaten - self._day_eaten_start.get(f.id, 0)
            intake_pct_bw = (pellets_today * CMAL_AVG_PELLET_MASS) / max(f.body_mass, 1.0)
            # Maintain 3-day rolling window
            f.recent_intake_pct.append(intake_pct_bw)
            if len(f.recent_intake_pct) > 3:
                f.recent_intake_pct.pop(0)
            avg_intake = sum(f.recent_intake_pct) / len(f.recent_intake_pct)
            if avg_intake < CMAL_MIN_RATION_PCT:
                f.underfed_days += 1
            else:
                f.underfed_days = max(0, f.underfed_days - 1)
            # Larger fish tolerate starvation longer: +1 grace day per 10 g body mass
            size_grace = CMAL_GRACE_DAYS + int(f.body_mass / 10.0)
            if f.underfed_days <= size_grace:
                continue
            days_past_grace = f.underfed_days - size_grace
            t = min(1.0, days_past_grace / CMAL_LETHAL_DAYS)
            death_prob = CMAL_DEATH_PROB_MIN + t * (CMAL_DEATH_PROB_MAX - CMAL_DEATH_PROB_MIN)
            if self.rng.random() < death_prob:
                f.is_alive = False
                f.death_reason = 'chronic_malnutrition'
                self._mal_deaths_total += 1

        for f in self.manager.ai_fish:
            pellets_today = f.food_eaten - self._day_eaten_start.get(f.id, 0)
            self._fish_daily_rows.append({
                'day'                : day_num,
                'fish_id'            : f.id,
                'is_alive'           : int(f.is_alive),
                'body_mass_g'        : round(f.body_mass, 4)        if f.is_alive else np.nan,
                'initial_mass_g'     : round(f.initial_mass, 4),
                'total_length_m'     : round(f.total_length, 5)     if f.is_alive else np.nan,
                'energy_pct'         : round(f.energy, 2)           if f.is_alive else np.nan,
                'stomach_full_pct'   : round(f.stomach_fullness, 2) if f.is_alive else np.nan,
                'pellets_eaten_today': pellets_today,
                'pellets_eaten_cum'  : f.food_eaten,
                'activity_state'     : str(f.activity_state.name)   if f.is_alive else 'DEAD',
                'pos_x'              : round(float(f.position[0]), 4) if f.is_alive else np.nan,
                'pos_y'              : round(float(f.position[1]), 4) if f.is_alive else np.nan,
                'pos_z'              : round(float(f.position[2]), 4) if f.is_alive else np.nan,
                'death_reason'       : f.death_reason if not f.is_alive else '',
                'smr_individual_factor': round(getattr(f, 'smr_individual_factor', 1.0), 4),
                'smr_kj_per_day': round(self._calc_smr_kj_day(f), 4),
                'last_length_update_mass_g': round(getattr(f, 'last_length_update_mass', np.nan), 4),
                'length_shrink_count': getattr(f, 'length_shrink_count', 0),
            })

        self._day_eaten_start  = {f.id: f.food_eaten for f in self.manager.ai_fish}
        self._day_biomass_start = sum(f.body_mass for f in self.manager.ai_fish if f.is_alive)
        self._day_alive_start = sum(1 for f in self.manager.ai_fish if f.is_alive)
        self._day_mass_start = {f.id: f.body_mass for f in self.manager.ai_fish}
        self._day_feed_start_g  = getattr(self.manager, '_total_feed_offered_g',
                                          getattr(self.manager, '_total_feed_mass_g', 0.0))

        # flush daily cannibalism counters
        self._daily_cann_attacks.append(self._day_cann_attacks)
        self._daily_cann_successes.append(self._day_cann_successes)
        self._day_cann_attacks   = 0
        self._day_cann_successes = 0

    # ── main loop ─────────────────────────────────────────────
    def enable_action_recording(self, interval: int = 10):
        """Record each fish's policy action + state every `interval` steps.

        Must be called after construction (manager already built) and before
        run().  Wraps the manager's _execute_action so the latest action vector
        is stashed per fish id; the actual rows are assembled in the run loop
        where the current step/day are known.
        """
        self._action_recording = True
        self._action_interval  = max(1, int(interval))
        self._action_rows      = []

        mgr = self.manager
        if not getattr(mgr, '_action_stash_installed', False):
            mgr._last_actions = {}
            inner = mgr._execute_action   # already-bound (possibly patched) method

            def _stashing_execute(fish, action, _inner=inner, _mgr=mgr):
                # stash this step's action vector keyed by fish id, then run
                # the real (patched) execute. Keyed by id() avoids needing the
                # fish object to allow new attributes (it may use __slots__).
                try:
                    _mgr._last_actions[fish.id] = np.asarray(action,
                                                             dtype=float).ravel()
                except Exception:
                    pass
                return _inner(fish, action)

            # _execute_action is an instance attribute (set via MethodType),
            # so assigning a plain function here is called as (fish, action)
            # without an implicit self — matching the manager's call site.
            mgr._execute_action = _stashing_execute
            mgr._action_stash_installed = True

        print(f"  Action recording enabled: every {self._action_interval} steps")

    def _record_actions(self, step: int):
        """Append one row per alive fish for the current step."""
        day  = step / STEPS_PER_DAY
        last = getattr(self.manager, '_last_actions', {})
        for f in self.manager.ai_fish:
            if not f.is_alive:
                continue
            act = last.get(f.id, None)
            a   = (list(act) + [None] * 5)[:5] if act is not None else [None] * 5
            pos = f.position
            self._action_rows.append({
                'day'             : round(day, 4),
                'step'            : step,
                'fish_id'         : f.id,
                'action_0'        : round(float(a[0]), 5) if a[0] is not None else '',
                'action_1'        : round(float(a[1]), 5) if a[1] is not None else '',
                'action_2'        : round(float(a[2]), 5) if a[2] is not None else '',
                'action_3'        : round(float(a[3]), 5) if a[3] is not None else '',
                'action_4'        : round(float(a[4]), 5) if a[4] is not None else '',
                'pos_x'           : round(float(pos[0]), 4),
                'pos_y'           : round(float(pos[1]), 4),
                'pos_z'           : round(float(pos[2]), 4),
                'body_mass_g'     : round(float(f.body_mass), 4),
                'energy_pct'      : round(float(getattr(f, 'energy', float('nan'))), 3),
                'stomach_full_pct': round(float(getattr(f, 'stomach_fullness',
                                                        float('nan'))), 3),
            })

    def run(self):
        print(f"\nStarting simulation: {self.num_fish} fish x {self.total_steps:,} steps")
        print(f"  = ~{self.total_steps / STEPS_PER_DAY:.1f} simulated days")
        print(f"  Feeding: {FEEDINGS_PER_DAY}x/day, ration [{DR_MIN*100:.1f}-{DR_MAX*100:.1f}]% BW offered")
        print(f"  Log interval: every {LOG_INTERVAL / STEPS_PER_DAY:.1f} day(s)")
        print()

        # feeding_interval already set in _build_manager; no override needed here

        _debug_steps = 20  # print per-step detail for first N steps then stop

        # feeding window tracker: measure how many steps each meal takes
        _feed_window_start = None   # step when meal queue was first loaded
        _feed_window_peak  = 0      # max food_items seen during this meal

        for step in range(1, self.total_steps + 1):

            food_before = len(self.manager.base_env.feeding_state.food_items)
            eaten_before = sum(f.food_eaten for f in self.manager.ai_fish)
            pending_before_step = getattr(self.manager, '_pending_pellets', 0)

            # one full step: physics + metabolism + food movement + capture
            self.manager.step()

            # Detect newly queued meal before trickle release changes the queue.
            pending_after_step = getattr(self.manager, '_pending_pellets', 0)
            if pending_before_step == 0 and pending_after_step > 0:
                _feed_window_start = step
                _feed_window_peak  = pending_after_step

            # trickle-feed: release next handful
            _aqua_trickle_feed(self.manager, step)
            # Floating pellets older than SINK_AFTER_STEPS begin sinking;
            # settled pellets are auto-removed by feeding_system (consistent with base system)
            _aqua_sink_aged_pellets(self.manager, step)
            pending_after  = getattr(self.manager, '_pending_pellets', 0)

            # action recording (every _action_interval steps, if enabled)
            if self._action_recording and step % self._action_interval == 0:
                self._record_actions(step)

            # feeding window tracking
            if _feed_window_start is not None:
                current_food = len(self.manager.base_env.feeding_state.food_items)
                _feed_window_peak = max(_feed_window_peak, current_food)
                if pending_after == 0 and current_food == 0:
                    # meal fully consumed
                    window_steps = step - _feed_window_start
                    window_min   = window_steps * SIM_SECONDS_PER_STEP / 60.0   # biological minutes
                    print(f"  [FEED_WINDOW] meal consumed in {window_steps} steps "
                          f"= {window_min:.1f} bio-min  "
                          f"(limit=180 steps=90 bio-min)  "
                          f"{'OK' if window_steps <= 180 else 'SLOW - increase TRICKLE_BATCH_FRAC'}")
                    _feed_window_start = None
                    _feed_window_peak  = 0

            if step <= _debug_steps:
                food_after  = len(self.manager.base_env.feeding_state.food_items)
                eaten_after = sum(f.food_eaten for f in self.manager.ai_fish)
                spawned_this_step = food_after - food_before + (eaten_after - eaten_before)
                print(f"  [DBG] step={step:4d}  food_items={food_after:4d}"
                      f"  spawned~={spawned_this_step:4d}"
                      f"  eaten_this_step={eaten_after - eaten_before:3d}"
                      f"  steps_since_feed={self.manager.steps_since_feeding}")

            # intraspecific predation (cannibalism)
            cann = _process_cannibalism(self.manager, self.rng)
            self._cann_attacks_total   += cann['attacks']
            self._cann_successes_total += cann['successes']
            self._cann_deaths_total    += cann['deaths']
            self._cann_swallows_total  += cann.get('swallows', 0)
            self._day_cann_attacks     += cann['attacks']
            self._day_cann_successes   += cann['successes']

            # daily record
            if step % STEPS_PER_DAY == 0:
                self._record_daily()

            # log snapshot
            if step % LOG_INTERVAL == 0 or step == self.total_steps:
                self._record_snapshot()
                alive_n   = self.ts_alive[-1]
                pct       = alive_n / self.num_fish * 100
                mean_m    = self.ts_mean_mass[-1]
                day       = step / STEPS_PER_DAY
                eaten_tot = sum(f.food_eaten for f in self.manager.ai_fish)
                print(f"  Day {day:6.1f}  |  alive={alive_n:3d} ({pct:.0f}%)  "
                      f"|  mean mass={mean_m:.1f} g  |  pellets eaten={eaten_tot:,}"
                      f"  |  mal_deaths={self._mal_deaths_total}"
                      f"  cann_deaths={self._cann_deaths_total}"
                      f"  (attacks={self._cann_attacks_total} hits={self._cann_successes_total} swallows={self._cann_swallows_total})")

        print("\nSimulation complete.")

    # ── summary stats ─────────────────────────────────────────
    def summary(self) -> dict:
        alive = [f for f in self.manager.ai_fish if f.is_alive]
        final_m = np.array([f.body_mass for f in alive]) if alive else np.array([0.0])
        init_m  = np.array([f.initial_mass for f in self.manager.ai_fish])

        surv    = len(alive) / self.num_fish * 100
        wg      = final_m.mean() - init_m.mean() if alive else 0.0
        n_days  = self.total_steps / STEPS_PER_DAY
        sgr     = ((np.log(final_m.mean()) - np.log(init_m.mean())) / n_days * 100
                   if alive and init_m.mean() > 0 and final_m.mean() > 0 else 0.0)
        total_positive_gain = sum(max(0.0, f.body_mass - f.initial_mass) for f in alive)
        total_feed_offered_g = getattr(self.manager, '_total_feed_offered_g',
                                       getattr(self.manager, '_total_feed_mass_g', 0.0))
        total_feed_eaten_g = sum(f.food_eaten for f in self.manager.ai_fish) * CMAL_AVG_PELLET_MASS
        fcr = (total_feed_eaten_g / total_positive_gain
               if total_positive_gain > 0 and total_feed_eaten_g > 0 else np.nan)
        cv  = (final_m.std() / final_m.mean() * 100
               if alive and final_m.mean() > 0 else np.nan)

        return {
            'n_initial'        : self.num_fish,
            'n_final_alive'    : len(alive),
            'survival_rate_pct': surv,
            'mean_init_mass_g' : float(init_m.mean()),
            'mean_final_mass_g': float(final_m.mean()) if alive else 0.0,
            'std_final_mass_g' : float(final_m.std())  if alive else 0.0,
            'min_final_mass_g' : float(final_m.min())  if alive else 0.0,
            'max_final_mass_g' : float(final_m.max())  if alive else 0.0,
            'cv_pct'           : cv,
            'weight_gain_g'    : wg,
            'total_positive_gain_g': total_positive_gain,
            'total_feed_offered_g' : total_feed_offered_g,
            'total_feed_eaten_g'   : total_feed_eaten_g,
            'sgr_pct_day'           : sgr,
            'fcr_total'             : fcr,
            'sim_days'              : n_days,
            'cann_attacks_total'    : self._cann_attacks_total,
            'cann_successes_total'  : self._cann_successes_total,
            'cann_deaths_total'     : self._cann_deaths_total,
            'cann_swallows_total'   : self._cann_swallows_total,
            'malnutrition_deaths_total': self._mal_deaths_total,
            'cann_success_rate_pct' : (self._cann_successes_total / self._cann_attacks_total * 100
                                       if self._cann_attacks_total > 0 else 0.0),
        }


# ─────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────

def _style():
    plt.rcParams.update({
        'font.family'      : 'DejaVu Sans',
        'font.size'        : 10,
        'axes.titlesize'   : 11,
        'axes.labelsize'   : 10,
        'axes.spines.top'  : False,
        'axes.spines.right': False,
        'figure.dpi'       : 150,
    })


def plot_growth_curves(sim: AquacultureSimulation, out_dir: str):
    _style()
    days  = sim.ts_day
    snaps = np.array(sim.mass_snapshots)

    fig, ax = plt.subplots(figsize=(9, 5))
    for i in range(sim.num_fish):
        col   = snaps[:, i]
        valid = ~np.isnan(col)
        if valid.sum() > 1:
            ax.plot(np.array(days)[valid], col[valid],
                    color='steelblue', alpha=0.15, linewidth=0.6)

    pop_mean = np.nanmean(snaps, axis=1)
    pop_std  = np.nanstd(snaps,  axis=1)
    ax.fill_between(days, pop_mean - pop_std, pop_mean + pop_std,
                    alpha=0.25, color='navy', label='Mean +/- 1 SD')
    ax.plot(days, pop_mean, color='navy', linewidth=2, label='Population mean')
    ax.set_xlabel("Simulated day")
    ax.set_ylabel("Body mass (g)")
    ax.set_title(f"Growth Curves - Largemouth Bass Cohort (N = {sim.num_fish})")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    path = os.path.join(out_dir, "growth_curves.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_final_weight_hist(sim: AquacultureSimulation, out_dir: str):
    _style()
    masses = [f.body_mass for f in sim.manager.ai_fish if f.is_alive]
    if not masses:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(masses, bins=20, color='steelblue', edgecolor='white', linewidth=0.6)
    ax.axvline(np.mean(masses), color='navy', linestyle='--', linewidth=1.5,
               label=f'Mean = {np.mean(masses):.1f} g')
    ax.axvline(np.median(masses), color='orangered', linestyle=':', linewidth=1.5,
               label=f'Median = {np.median(masses):.1f} g')
    ax.set_xlabel("Terminal body mass (g)")
    ax.set_ylabel("Number of fish")
    ax.set_title(f"Terminal Weight Distribution  (n = {len(masses)} survivors)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = os.path.join(out_dir, "final_weight_hist.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_survival_curve(sim: AquacultureSimulation, out_dir: str):
    _style()
    days = sim.ts_day
    surv = [c / sim.num_fish * 100 for c in sim.ts_alive]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.step(days, surv, where='post', color='seagreen', linewidth=2)
    ax.fill_between(days, surv, alpha=0.15, color='seagreen', step='post')
    ax.set_ylim(0, 105)
    ax.set_xlabel("Simulated day")
    ax.set_ylabel("Survival (%)")
    ax.set_title("Survival Curve - Largemouth Bass Cohort")
    ax.axhline(surv[-1], color='grey', linestyle='--', linewidth=1,
               label=f'Final survival = {surv[-1]:.1f}%')
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = os.path.join(out_dir, "survival_curve.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_fcr_sgr(sim: AquacultureSimulation, out_dir: str):
    _style()
    days    = list(range(1, len(sim.daily_fcr) + 1))
    fcr_arr = np.array(sim.daily_fcr, dtype=float)
    sgr_arr = np.array(sim.daily_sgr, dtype=float)

    fig, ax1 = plt.subplots(figsize=(9, 4))
    c1, c2   = 'steelblue', 'orangered'
    valid_f  = ~np.isnan(fcr_arr)
    ax1.plot(np.array(days)[valid_f], fcr_arr[valid_f],
             color=c1, linewidth=1.5, label='Daily FCR')
    ax1.set_xlabel("Simulated day")
    ax1.set_ylabel("FCR", color=c1)
    ax1.tick_params(axis='y', labelcolor=c1)

    ax2 = ax1.twinx()
    valid_s = ~np.isnan(sgr_arr)
    ax2.plot(np.array(days)[valid_s], sgr_arr[valid_s],
             color=c2, linewidth=1.5, linestyle='--', label='Daily SGR (%/d)')
    ax2.set_ylabel("SGR (%/day)", color=c2)
    ax2.tick_params(axis='y', labelcolor=c2)

    l1, b1 = ax1.get_legend_handles_labels()
    l2, b2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, b1 + b2, fontsize=9, loc='upper right')
    ax1.set_title("Daily Feed Conversion Ratio and Specific Growth Rate")
    plt.tight_layout()
    path = os.path.join(out_dir, "fcr_sgr.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_biomass(sim: AquacultureSimulation, out_dir: str):
    _style()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sim.ts_day, sim.ts_total_biomass, color='darkorange', linewidth=2)
    ax.fill_between(sim.ts_day, sim.ts_total_biomass, alpha=0.2, color='darkorange')
    ax.set_xlabel("Simulated day")
    ax.set_ylabel("Total biomass (g)")
    ax.set_title("Total Cohort Biomass Over Time")
    plt.tight_layout()
    path = os.path.join(out_dir, "biomass_trend.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_summary_table(stats: dict, out_dir: str):
    _style()
    rows = [
        ["Parameter",                          "Value"],
        ["Initial fish count (N)",             str(stats['n_initial'])],
        ["Final survivors",                    str(stats['n_final_alive'])],
        ["Survival rate",                      f"{stats['survival_rate_pct']:.1f}%"],
        ["Simulation duration (days)",         f"{stats['sim_days']:.0f}"],
        ["Mean initial body mass",             f"{stats['mean_init_mass_g']:.1f} g"],
        ["Mean final body mass",               f"{stats['mean_final_mass_g']:.1f} g"],
        ["SD final body mass",                 f"{stats['std_final_mass_g']:.1f} g"],
        ["Min / Max final body mass",          f"{stats['min_final_mass_g']:.1f} / {stats['max_final_mass_g']:.1f} g"],
        ["CV of final body mass",              f"{stats['cv_pct']:.1f}%"
                                                if not np.isnan(stats['cv_pct']) else "N/A"],
        ["Mean weight gain",                   f"{stats['weight_gain_g']:.1f} g/fish"],
        ["Total positive gain",              f"{stats['total_positive_gain_g']:.1f} g"],
        ["Total feed offered",               f"{stats['total_feed_offered_g']:.1f} g"],
        ["Overall SGR",                        f"{stats['sgr_pct_day']:.3f} %/day"],
        ["Overall FCR (eaten/gain)",          f"{stats['fcr_total']:.2f}"
                                                if not np.isnan(stats['fcr_total']) else "N/A"],
        ["Feeding regime",                     f"{FEEDINGS_PER_DAY}x / day"],
        ["Daily ration range (% BW)",          f"{DR_MIN*100:.1f}-{DR_MAX*100:.1f}%"],
    ]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.axis('off')
    tbl = ax.table(cellText=rows[1:], colLabels=rows[0],
                   colWidths=[0.62, 0.38], cellLoc='center', loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.55)
    for j in range(2):
        tbl[0, j].set_facecolor('#2c5f8a')
        tbl[0, j].set_text_props(color='white', fontweight='bold')
    for i in range(1, len(rows)):
        bg = '#f0f4f8' if i % 2 == 0 else 'white'
        for j in range(2):
            tbl[i, j].set_facecolor(bg)
    ax.set_title(
        "Largemouth Bass (Micropterus salmoides) - Aquaculture Batch Simulation\n"
        "Summary Statistics",
        fontsize=11, fontweight='bold', pad=12
    )
    plt.tight_layout()
    path = os.path.join(out_dir, "summary_table.png")
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────
# CSV export
# ─────────────────────────────────────────────────────────────

def export_fish_daily_csv(sim: 'AquacultureSimulation', out_dir: str) -> str:
    """Per-fish daily data: body mass, energy, pellets eaten, position, etc."""
    import csv
    path = os.path.join(out_dir, "fish_daily.csv")
    if not sim._fish_daily_rows:
        print(f"  [WARN] No per-fish daily rows to export.")
        return path
    fieldnames = list(sim._fish_daily_rows[0].keys())
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sim._fish_daily_rows)
    print(f"  Saved: {path}  ({len(sim._fish_daily_rows)} rows, "
          f"{sim.num_fish} fish × {len(sim.daily_fcr)} days)")
    return path


def export_population_daily_csv(sim: 'AquacultureSimulation', out_dir: str) -> str:
    """Population-level daily summary: survival, mean mass, FCR, SGR."""
    import csv
    path = os.path.join(out_dir, "population_daily.csv")
    n_days = len(sim.daily_fcr)
    rows = []
    for i in range(n_days):
        ts_idx = i + 1
        rows.append({
            'day'                 : i + 1,
            'alive_count'         : sim.ts_alive[ts_idx]     if ts_idx < len(sim.ts_alive)    else '',
            'survival_pct'        : round(sim.ts_alive[ts_idx] / sim.num_fish * 100, 2)
                                    if ts_idx < len(sim.ts_alive) else '',
            'mean_mass_g'         : round(sim.ts_mean_mass[ts_idx], 4)
                                    if ts_idx < len(sim.ts_mean_mass) else '',
            'total_biomass_g'     : round(sim.ts_total_biomass[ts_idx], 4)
                                    if ts_idx < len(sim.ts_total_biomass) else '',
            'feed_offered_g'      : round(sim.daily_feed_offered_g[i], 4)
                                    if i < len(sim.daily_feed_offered_g) else '',
            'feed_eaten_g'        : round(sim.daily_feed_eaten_g[i], 4)
                                    if i < len(sim.daily_feed_eaten_g) else '',
            'positive_gain_g'     : round(sim.daily_positive_gain_g[i], 4)
                                    if i < len(sim.daily_positive_gain_g) else '',
            'biomass_delta_g'     : round(sim.daily_biomass_delta_g[i], 4)
                                    if i < len(sim.daily_biomass_delta_g) else '',
            'daily_fcr'           : round(sim.daily_fcr[i], 4)
                                    if not np.isnan(sim.daily_fcr[i]) else '',
            'daily_sgr_pct_day'   : round(sim.daily_sgr[i], 4)
                                    if not np.isnan(sim.daily_sgr[i]) else '',
            'cann_attacks_today'  : sim._daily_cann_attacks[i]   if i < len(sim._daily_cann_attacks)   else '',
            'cann_successes_today': sim._daily_cann_successes[i] if i < len(sim._daily_cann_successes) else '',
        })
    if not rows:
        return path
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {path}  ({len(rows)} rows)")
    return path


def export_fish_actions_csv(sim: 'AquacultureSimulation', out_dir: str) -> str:
    """Per-fish action log: day, step, fish_id, action_0..4, position,
    body mass, energy %, stomach %.  Populated only if action recording was
    enabled via sim.enable_action_recording()."""
    import csv
    path = os.path.join(out_dir, "fish_actions.csv")
    rows = getattr(sim, '_action_rows', None)
    if not rows:
        print("  [WARN] No action rows to export "
              "(did you call sim.enable_action_recording() before run()?).")
        return path
    fieldnames = list(rows[0].keys())
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {path}  ({len(rows)} rows, "
          f"every {getattr(sim, '_action_interval', '?')} steps)")
    return path

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Aquaculture batch simulation")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed (default: 42)")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="output directory "
                             "(default: aquaculture_results_seed<SEED>)")
    args = parser.parse_args()

    out_dir = args.out_dir or f"{OUT_DIR}_seed{args.seed}"

    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)

    sim = AquacultureSimulation(seed=args.seed)
    sim.run()

    print("\nComputing summary statistics...")
    stats = sim.summary()

    print("\n" + "=" * 55)
    print(f"  AQUACULTURE BATCH SIMULATION - SUMMARY (seed={args.seed})")
    print("=" * 55)
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k:<30s}: {v:.3f}")
        else:
            print(f"  {k:<30s}: {v}")
    print("=" * 55)

    print(f"\nGenerating figures -> {out_dir}/")
    plot_growth_curves(sim, out_dir)
    plot_final_weight_hist(sim, out_dir)
    plot_survival_curve(sim, out_dir)
    plot_fcr_sgr(sim, out_dir)
    plot_biomass(sim, out_dir)
    plot_summary_table(stats, out_dir)

    print(f"\nExporting CSV data -> {out_dir}/")
    export_fish_daily_csv(sim, out_dir)
    export_population_daily_csv(sim, out_dir)

    print(f"\n[DONE] All figures and CSV files saved to {out_dir}/")


if __name__ == "__main__":
    import sys as _sys
    _sys.stdout.reconfigure(line_buffering=True)
    main()