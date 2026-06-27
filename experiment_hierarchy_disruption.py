#!/usr/bin/env python3
"""
Experiment 2: Hierarchy Disruption by External Stocking
========================================================
Evaluates how an established dominance hierarchy in juvenile largemouth bass
(Micropterus salmoides) responds to the sudden introduction of intruder fish
of varying relative sizes.

Experimental Design
-------------------
  Phase 1 (day 1-50):   Establish hierarchy in a resident cohort (10-19 g).
  Phase 2 (day 51-100): Introduce 10 intruder fish under one of four conditions:
    A. LARGER   -- intruder mass = 1.6x current cohort mean (dominant challengers)
    B. SMALLER  -- intruder mass = 0.4x current cohort mean (new subordinates)
    C. EQUAL    -- intruder mass = 1.0x current cohort mean (stranger effect only)
    D. CONTROL  -- no intruders added (baseline comparison)

Scientific Questions
--------------------
1. Does introducing larger fish collapse the existing hierarchy?
   (Are resident dominants "dethroned" by the intruders?)
2. Does introducing smaller fish "rescue" the lowest-ranking residents?
   (Does a new bottom tier reduce mortality pressure on original low-ranks?)
3. Does introducing equal-sized strangers reset the hierarchy through the
   winner/loser effect alone, without size advantage?
4. What does this tell us about how solitary predators adapt to forced group
   living under aquaculture conditions?

Output
------
  aquaculture_results/hierarchy_disruption/
    fish_daily_<CONDITION>.csv       (with 'phase' column: 1=pre, 2=post)
    rank_stability_<CONDITION>.png   (rho trajectory with phase-2 boundary)
    rank_recovery_matrix.png         (quartile transition pre->post disruption)
    intruder_vs_resident.png         (intruder rank at end vs resident ranks)
    summary_comparison.png
    summary.txt
"""

import os, sys, copy, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aquaculture_batch_sim import (
    AquacultureSimulation, STEPS_PER_DAY, MODEL_PATH,
    _aqua_trickle_feed, _process_cannibalism,
    AQUA_VISION_RANGE, AQUA_FOOD_DETECT_RANGE, AQUA_LATERAL_LINE_RANGE,
    CMAL_AVG_PELLET_MASS, CMAL_MIN_RATION_PCT, CMAL_GRACE_DAYS,
    CMAL_LETHAL_DAYS, CMAL_DEATH_PROB_MIN, CMAL_DEATH_PROB_MAX,
)
from config import CONFIG
from utils.biological_formulas import mass_to_length
from multi_fish_visualizer import AIFishState

OUT_DIR = "aquaculture_results/hierarchy_disruption"

# Simulation parameters
INIT_MASS_MIN    = 10.0
INIT_MASS_MAX    = 19.0
NUM_FISH_COHORT  = 40       # resident cohort size
NUM_INTRUDERS    = 10       # fish added at phase transition
PHASE1_DAYS      = 50       # days to establish hierarchy
PHASE2_DAYS      = 50       # days after disruption
SEED             = 42

# Intruder size multipliers relative to cohort mean at day 50
INTRUDER_SIZE = {
    'LARGER'  : 1.6,    # 1.6x mean = clear size advantage
    'SMALLER' : 0.4,    # 0.4x mean = clearly subordinate
    'EQUAL'   : 1.0,    # same mean, but strangers
    'CONTROL' : None,   # no intruders
}
CONDITION_COLORS = {
    'LARGER' : 'firebrick',
    'SMALLER': 'steelblue',
    'EQUAL'  : 'darkorange',
    'CONTROL': 'dimgrey',
}


def add_intruders(sim: AquacultureSimulation,
                  intruder_masses: np.ndarray,
                  rng: np.random.Generator) -> list:
    """Inject new fish into a running simulation mid-experiment.

    Constructs AIFishState objects with appropriate buoyancy and metabolic
    initialisation, then appends them to the simulation's active fish list.
    Uses the same construction pattern as MultiAIFishManager.reset().

    Args:
        sim: The running AquacultureSimulation to inject fish into.
        intruder_masses: Array of body masses (grams) for the new fish.
        rng: NumPy random generator for position and state randomisation.

    Returns:
        List of newly created AIFishState objects marked as intruders.
    """
    mgr = sim.manager
    _bc = CONFIG.buoyancy
    INIT_DEPTH = 0.30
    _GAS_CONSTANT = 8.314

    existing_ids = {f.id for f in mgr.ai_fish}
    next_id = max(existing_ids) + 1 if existing_ids else 0

    tank_radius = CONFIG.environment.tank_radius

    new_fish = []
    for i, m in enumerate(intruder_masses):
        # Random position within tank (same pattern as reset())
        angle    = float(rng.uniform(0, 2 * np.pi))
        radius   = float(rng.uniform(0.2, tank_radius * 0.7))
        position = np.array([radius * np.cos(angle), -INIT_DEPTH,
                              radius * np.sin(angle)], dtype=np.float32)
        velocity = rng.uniform(-0.02, 0.02, 3).astype(np.float32)

        fish = AIFishState(
            id                   = next_id + i,
            position             = position,
            velocity             = velocity,
            body_mass            = float(m),
            total_length         = mass_to_length(float(m)),
            energy               = float(rng.uniform(70, 90)),
            stomach_fullness     = float(rng.uniform(20, 40)),
            stomach_content_mass = 0.0,
            initial_meal_mass    = 0.0,
            digestion_buffer     = 0.0,
            energy_from_digestion= 0.0,
            growth_accumulation  = 0.0,
            total_growth_energy  = 0.0,
            growth_count         = 0,
            initial_mass         = float(m),
            state_switch_cooldown= 0.0,
            last_state_switch_step= 0,
            total_rest_steps     = 0,
        )
        fish.position_history.append(position.copy())

        # Buoyancy: create independent state (same as reset())
        try:
            from systems.buoyancy import create_buoyancy_state
            buoyancy_sys = getattr(
                getattr(mgr.base_env, 'physics_system', None), 'buoyancy_system', None)
            if buoyancy_sys is not None:
                fish._buoyancy_state = create_buoyancy_state()
                buoyancy_sys.initialize(fish._buoyancy_state, fish.body_mass, fish.total_length)
                fish.relative_density   = fish._buoyancy_state.relative_density
                fish.net_buoyancy_force = fish._buoyancy_state.net_buoyancy_force
                fish.swimbladder_volume = fish._buoyancy_state.swimbladder_volume
        except Exception:
            # Fallback: manual buoyancy initialisation
            mass_kg = float(m) / 1000.0
            tv  = mass_kg / _bc.fish_tissue_density
            nv  = max(mass_kg / _bc.water_density - tv, tv * _bc.min_volume_ratio)
            pp  = (_bc.atmospheric_pressure + INIT_DEPTH * _bc.pressure_per_meter) * 101325.0
            ga  = pp * nv / (_GAS_CONSTANT * 298.15)
            fish.swimbladder_volume = nv
            fish.relative_density   = 1.0
            fish.net_buoyancy_force = 0.0

        # Additional fields used by aquaculture_batch_sim patches
        fish.smr_individual_factor       = float(np.clip(rng.normal(1.0, 0.12), 0.70, 1.35))
        fish.initial_body_mass           = 0.0
        fish.lipid_reserve               = 0.0
        fish.protein_reserve             = 0.0
        fish.death_mass_loss_threshold   = 0.275
        fish.last_length_update_mass     = float(m)
        fish.length_shrink_count         = 0
        fish.growth_during_rest          = 0.0
        fish.growth_count_during_rest    = 0
        fish.energy_added_during_rest    = 0.0

        # Mark as intruder for downstream analysis
        fish._is_intruder         = True
        fish._phase2_initial_mass = float(m)

        mgr.ai_fish.append(fish)
        sim.num_fish += 1   # keep survival percentage consistent
        new_fish.append(fish)

    print(f"  Injected {len(new_fish)} intruders: "
          f"mean={np.mean(intruder_masses):.1f}g, "
          f"range={intruder_masses.min():.1f}-{intruder_masses.max():.1f}g")
    return new_fish


def run_two_phase(condition_name: str, seed: int = SEED) -> tuple:
    """Run the two-phase hierarchy disruption experiment for one condition.

    Phase 1 establishes the dominance hierarchy among the resident cohort.
    Phase 2 optionally injects intruder fish and tracks hierarchy response.

    Args:
        condition_name: One of 'LARGER', 'SMALLER', 'EQUAL', or 'CONTROL'.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (sim, fish_daily_rows) where fish_daily_rows includes
        'phase' (1 or 2) and 'is_intruder' (0 or 1) columns.
    """
    cond = CONDITION_COLORS  # for naming only
    intruder_ratio = INTRUDER_SIZE[condition_name]

    total_days  = PHASE1_DAYS + PHASE2_DAYS
    total_steps = total_days * STEPS_PER_DAY

    # Build sim for Phase 1 only initially
    sim = AquacultureSimulation(
        model_path  = MODEL_PATH,
        num_fish    = NUM_FISH_COHORT,
        total_steps = PHASE1_DAYS * STEPS_PER_DAY,
        seed        = seed,
    )
    sim._condition_name = condition_name

    # Override initial masses with fixed range
    rng_init = np.random.default_rng(seed)
    init_masses = rng_init.uniform(INIT_MASS_MIN, INIT_MASS_MAX, NUM_FISH_COHORT)
    _bc = CONFIG.buoyancy
    INIT_DEPTH = 0.30
    for fish, m in zip(sim.manager.ai_fish, init_masses):
        fish.body_mass    = float(m)
        fish.initial_mass = float(m)
        fish.total_length = mass_to_length(float(m))
        fish.position[1]  = -INIT_DEPTH
        fish.stomach_content_mass = 0.0
        fish.stomach_fullness = 0.0
        fish.is_digesting = False
        mass_kg = float(m)/1000.0
        tv = mass_kg/_bc.fish_tissue_density
        nv = max(mass_kg/_bc.water_density - tv, tv*_bc.min_volume_ratio)
        pp = (_bc.atmospheric_pressure + INIT_DEPTH*_bc.pressure_per_meter)*101325.0
        ga = pp*nv/(8.314*298.15)
        bs = getattr(fish,'_buoyancy_state',None) or getattr(
            sim.manager.base_env.physics_state,'buoyancy_state',None)
        if bs is not None:
            bs.neutral_volume=nv; bs.swimbladder_volume=nv; bs.gas_amount=ga
            bs.current_depth=INIT_DEPTH; bs.relative_density=1.0
            bs.net_buoyancy_force=0.0; bs.is_neutral=True
        fish.swimbladder_volume=nv; fish.relative_density=1.0; fish.net_buoyancy_force=0.0
        fish._is_intruder = False

    print(f"\n{'='*60}")
    print(f"  CONDITION: {condition_name}")
    print(f"  Phase 1: {PHASE1_DAYS} days, {NUM_FISH_COHORT} resident fish")
    print(f"{'='*60}")

    all_daily_rows = []   # accumulates rows from both phases

    # Phase 1 loop: hierarchy establishment
    for step in range(1, PHASE1_DAYS * STEPS_PER_DAY + 1):
        sim.manager.step()
        _aqua_trickle_feed(sim.manager)
        cann = _process_cannibalism(sim.manager, sim.rng)
        sim._cann_attacks_total   += cann['attacks']
        sim._cann_successes_total += cann['successes']
        sim._cann_deaths_total    += cann['deaths']
        sim._cann_swallows_total  += cann.get('swallows', 0)
        sim._day_cann_attacks     += cann['attacks']
        sim._day_cann_successes   += cann['successes']

        if step % STEPS_PER_DAY == 0:
            sim._record_daily()
            sim._record_snapshot()
            day = step // STEPS_PER_DAY
            # Tag rows with phase=1 and is_intruder=0
            for row in sim._fish_daily_rows[-NUM_FISH_COHORT:]:
                row['phase']       = 1
                row['is_intruder'] = 0
                all_daily_rows.append(row)
            sim._fish_daily_rows = []   # flush buffer (keep all_daily_rows)

            if day % 10 == 0:
                alive_n = sum(1 for f in sim.manager.ai_fish if f.is_alive)
                mean_m  = np.mean([f.body_mass for f in sim.manager.ai_fish if f.is_alive])
                print(f"  [P1 Day {day:3d}] alive={alive_n}, mean={mean_m:.1f}g")

    # Measure rank at end of Phase 1
    alive_p1 = [f for f in sim.manager.ai_fish if f.is_alive]
    mean_mass_p1 = np.mean([f.body_mass for f in alive_p1])
    rank_p1 = {f.id: i+1 for i, f in enumerate(
        sorted(alive_p1, key=lambda x: x.body_mass))}
    print(f"\n  Phase 1 complete: {len(alive_p1)} alive, mean={mean_mass_p1:.1f}g")

    # Compute rank fixation (Spearman rho) at end of Phase 1
    p1_masses = pd.DataFrame([
        {'fish_id': f.id, 'init_mass': f.initial_mass, 'p1_mass': f.body_mass}
        for f in sim.manager.ai_fish if f.is_alive
    ])
    if len(p1_masses) >= 5:
        rho_p1, _ = spearmanr(p1_masses['init_mass'], p1_masses['p1_mass'])
        print(f"  Rank fixation at end of Phase 1: Spearman ρ = {rho_p1:.4f}")

    # Inject intruders (Phase 2 start)
    intruder_fish = []
    if intruder_ratio is not None:
        intruder_target_mass = mean_mass_p1 * intruder_ratio
        # Small spread around target mass (+/-10%)
        intruder_masses = sim.rng.uniform(
            intruder_target_mass * 0.90,
            intruder_target_mass * 1.10,
            NUM_INTRUDERS
        )
        print(f"\n  Phase 2: injecting {NUM_INTRUDERS} intruders "
              f"(type={condition_name}, target={intruder_target_mass:.1f}g)")
        intruder_fish = add_intruders(sim, intruder_masses, sim.rng)

        # Reset day tracking to include intruders
        sim._day_eaten_start  = {f.id: f.food_eaten for f in sim.manager.ai_fish}
        sim._day_mass_start   = {f.id: f.body_mass  for f in sim.manager.ai_fish}
    else:
        print(f"\n  Phase 2: CONTROL -- no intruders added")

    # Phase 2 loop: post-disruption dynamics
    n_total_p2 = NUM_FISH_COHORT + len(intruder_fish)
    print(f"  Phase 2: {PHASE2_DAYS} days, {len([f for f in sim.manager.ai_fish if f.is_alive])} alive fish")

    for step in range(1, PHASE2_DAYS * STEPS_PER_DAY + 1):
        sim.manager.step()
        _aqua_trickle_feed(sim.manager)
        cann = _process_cannibalism(sim.manager, sim.rng)
        sim._cann_attacks_total   += cann['attacks']
        sim._cann_successes_total += cann['successes']
        sim._cann_deaths_total    += cann['deaths']
        sim._cann_swallows_total  += cann.get('swallows', 0)
        sim._day_cann_attacks     += cann['attacks']
        sim._day_cann_successes   += cann['successes']

        if step % STEPS_PER_DAY == 0:
            sim._record_daily()
            sim._record_snapshot()
            day_global = PHASE1_DAYS + step // STEPS_PER_DAY

            # Tag phase-2 rows with intruder flag
            n_fish_tracked = len(sim.manager.ai_fish)
            for row in sim._fish_daily_rows[-n_fish_tracked:]:
                row['phase'] = 2
                fid = row['fish_id']
                row['is_intruder'] = 1 if any(f.id == fid and f._is_intruder
                                               for f in sim.manager.ai_fish) else 0
                all_daily_rows.append(row)
            sim._fish_daily_rows = []

            if (step // STEPS_PER_DAY) % 10 == 0:
                alive_n = sum(1 for f in sim.manager.ai_fish if f.is_alive)
                mean_m  = np.mean([f.body_mass for f in sim.manager.ai_fish if f.is_alive])
                print(f"  [P2 Day {day_global:3d}] alive={alive_n}, mean={mean_m:.1f}g")

    print(f"\n  [{condition_name}] Complete.")
    return sim, all_daily_rows


# ── Analysis ──────────────────────────────────────────────────────────────────


def rho_from_rows(rows: list, reference_day: int = 1) -> list[tuple[int, float]]:
    """Compute Spearman rho of current mass versus mass at a reference day.

    For each recorded day, correlates the mass ranking at that day with the
    ranking at the reference day to measure rank stability over time.

    Args:
        rows: List of per-fish daily record dicts (from run_two_phase).
        reference_day: Day to use as the baseline for rank comparison.

    Returns:
        List of (day, rho) tuples sorted chronologically.
    """
    df = pd.DataFrame(rows)
    df = df[df['is_alive'] == 1]
    pivot = df.pivot_table(index='fish_id', columns='day', values='body_mass_g')
    all_days = sorted(pivot.columns.tolist())
    if reference_day not in pivot.columns:
        reference_day = all_days[0]
    result = []
    for d in all_days:
        sub = pivot[[reference_day, d]].dropna()
        if len(sub) < 5:
            continue
        rho = float(spearmanr(sub.iloc[:, 0], sub.iloc[:, 1])[0])
        result.append((int(d), rho))
    return result


# ── Plotting ──────────────────────────────────────────────────────────────────


def plot_rank_stability_all(all_results: dict, out_dir: str) -> None:
    """Plot rank stability (Spearman rho) trajectories for all conditions.

    Shows how the hierarchy evolves before and after the disruption event,
    with a vertical line marking the Phase 1/2 boundary.

    Args:
        all_results: Dict mapping condition names to (sim, rows) tuples.
        out_dir: Directory to write the output PNG.
    """
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.dpi': 150})
    fig, ax = plt.subplots(figsize=(10, 5))

    for cond_name, (sim, rows) in all_results.items():
        rho_traj = rho_from_rows(rows, reference_day=1)
        days, rhos = zip(*rho_traj) if rho_traj else ([], [])
        ax.plot(days, rhos,
                color=CONDITION_COLORS[cond_name],
                linewidth=2,
                label=cond_name,
                linestyle='-' if cond_name == 'CONTROL' else '--')

    ax.axvline(PHASE1_DAYS, color='black', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.text(PHASE1_DAYS + 1, 0.55, '← Intruders\n   added', fontsize=9, color='black')
    ax.axhline(0.9, color='grey', linestyle=':', linewidth=1, alpha=0.5)
    ax.set_xlabel("Simulated day")
    ax.set_ylabel("Spearman ρ  (day 1 → day X)")
    ax.set_ylim(0.3, 1.05)
    ax.set_title("Rank Stability Trajectory After Hierarchy Disruption\n"
                 "(ρ drop = hierarchy destabilised by intruders)")
    ax.legend(fontsize=9, loc='lower left')
    plt.tight_layout()
    path = os.path.join(out_dir, "rank_stability_comparison.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_mass_distribution_snapshots(all_results: dict, out_dir: str) -> None:
    """Plot box plots of mass distribution pre- and post-disruption.

    Shows the mass distribution at day 50 (pre-disruption) and day 100
    (post-disruption), separating residents from intruders in the post-
    disruption snapshot.

    Args:
        all_results: Dict mapping condition names to (sim, rows) tuples.
        out_dir: Directory to write the output PNG.
    """
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.dpi': 150})
    fig, axes = plt.subplots(1, 4, figsize=(14, 5), sharey=False)

    for ax, (cond_name, (sim, rows)) in zip(axes, all_results.items()):
        df = pd.DataFrame(rows)
        df_alive = df[df['is_alive'] == 1]

        pre_masses  = df_alive[df_alive['day'] == PHASE1_DAYS]['body_mass_g'].dropna()
        post_masses = df_alive[df_alive['day'] == PHASE1_DAYS + PHASE2_DAYS]['body_mass_g'].dropna()

        # Separate intruder vs resident in post phase
        intruder_ids = {r['fish_id'] for r in rows if r.get('is_intruder', 0) == 1}
        post_resident = df_alive[
            (df_alive['day'] == PHASE1_DAYS + PHASE2_DAYS) &
            (~df_alive['fish_id'].isin(intruder_ids))
        ]['body_mass_g'].dropna()
        post_intruder = df_alive[
            (df_alive['day'] == PHASE1_DAYS + PHASE2_DAYS) &
            (df_alive['fish_id'].isin(intruder_ids))
        ]['body_mass_g'].dropna()

        data = [pre_masses]
        labels = [f'Day {PHASE1_DAYS}\n(pre)']
        colors_bp = ['lightblue']

        if len(post_resident) > 0:
            data.append(post_resident)
            labels.append(f'Day {PHASE1_DAYS+PHASE2_DAYS}\nResidents')
            colors_bp.append('steelblue')

        if len(post_intruder) > 0:
            data.append(post_intruder)
            labels.append(f'Day {PHASE1_DAYS+PHASE2_DAYS}\nIntruders')
            colors_bp.append('orangered')

        bp = ax.boxplot(data, patch_artist=True, widths=0.5)
        for patch, color in zip(bp['boxes'], colors_bp):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("Body mass (g)")
        ax.set_title(f"Condition: {cond_name}", fontsize=10, fontweight='bold')

    fig.suptitle("Mass Distribution Before and After Intruder Introduction",
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(out_dir, "mass_distribution_snapshots.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_intruder_rank_penetration(all_results: dict, out_dir: str) -> None:
    """Plot final rank position of intruders relative to residents.

    For each non-CONTROL condition, shows where intruder fish end up in
    the final mass-based rank order, indicating whether they penetrated
    the existing hierarchy.

    Args:
        all_results: Dict mapping condition names to (sim, rows) tuples.
        out_dir: Directory to write the output PNG.
    """
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.dpi': 150})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    ax_idx = 0
    for cond_name, (sim, rows) in all_results.items():
        if cond_name == 'CONTROL':
            continue
        ax = axes[ax_idx]; ax_idx += 1

        df = pd.DataFrame(rows)
        df_final = df[(df['day'] == PHASE1_DAYS + PHASE2_DAYS) & (df['is_alive'] == 1)]
        if len(df_final) == 0:
            ax.set_title(f"{cond_name} – no data"); continue

        df_final = df_final.sort_values('body_mass_g').reset_index(drop=True)
        df_final['rank_pct'] = np.arange(1, len(df_final)+1) / len(df_final) * 100
        intruder_ids = {r['fish_id'] for r in rows if r.get('is_intruder', 0) == 1}

        residents = df_final[~df_final['fish_id'].isin(intruder_ids)]
        intruders = df_final[df_final['fish_id'].isin(intruder_ids)]

        ax.scatter(residents['rank_pct'], residents['body_mass_g'],
                   color='steelblue', alpha=0.7, s=40, label='Residents')
        ax.scatter(intruders['rank_pct'],  intruders['body_mass_g'],
                   color='orangered', s=80, zorder=5, marker='*', label='Intruders')

        ax.set_xlabel("Rank percentile (%)")
        ax.set_ylabel("Body mass (g)")
        ax.set_title(f"{cond_name} intruders\n(★ = intruder position in final rank)")
        ax.legend(fontsize=8)

    fig.suptitle("Intruder Rank Penetration at End of Phase 2",
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(out_dir, "intruder_rank_penetration.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    """Execute the hierarchy disruption experiment across all conditions.

    Runs the two-phase protocol for CONTROL, LARGER, SMALLER, and EQUAL
    conditions sequentially, exports per-fish daily CSVs, and generates
    comparative analysis plots and a text summary.
    """
    os.makedirs(OUT_DIR, exist_ok=True)

    all_results = {}
    all_rows    = {}

    for cond_name in ['CONTROL', 'LARGER', 'SMALLER', 'EQUAL']:
        sim, rows = run_two_phase(cond_name, seed=SEED)
        all_results[cond_name] = (sim, rows)
        all_rows[cond_name]    = rows

        # Export CSV
        csv_path = os.path.join(OUT_DIR, f"fish_daily_{cond_name}.csv")
        if rows:
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            print(f"  Saved CSV: {csv_path}")

    # Analysis
    print("\n=== ANALYSIS ===\n")
    summary_lines = []
    for cond_name in ['CONTROL', 'LARGER', 'SMALLER', 'EQUAL']:
        rows = all_rows[cond_name]
        df   = pd.DataFrame(rows)

        # Rho at day 50 (end of P1) and day 100 (end of P2), day-1 as reference
        rho_traj = rho_from_rows(rows, reference_day=1)
        rho_dict = {d: r for d, r in rho_traj}
        rho_p1   = rho_dict.get(PHASE1_DAYS, float('nan'))
        rho_p2   = rho_dict.get(PHASE1_DAYS + PHASE2_DAYS, float('nan'))
        delta    = rho_p2 - rho_p1

        # Intruder survival
        intruder_ids = {r['fish_id'] for r in rows if r.get('is_intruder', 0) == 1}
        if intruder_ids:
            df_intruder_final = df[
                (df['fish_id'].isin(intruder_ids)) &
                (df['day'] == PHASE1_DAYS + PHASE2_DAYS)
            ]
            intruder_surv = df_intruder_final['is_alive'].mean() * 100
            intruder_mean_mass = df_intruder_final[df_intruder_final['is_alive']==1]['body_mass_g'].mean()
        else:
            intruder_surv, intruder_mean_mass = float('nan'), float('nan')

        line = (f"{cond_name:8s}: ρ(P1_end)={rho_p1:.3f}, ρ(P2_end)={rho_p2:.3f}, "
                f"Δρ={delta:+.3f} | "
                f"intruder_surv={intruder_surv:.0f}%, intruder_mass={intruder_mean_mass:.1f}g")
        print(f"  {line}")
        summary_lines.append(line)

    # Plots
    plot_rank_stability_all(all_results, OUT_DIR)
    plot_mass_distribution_snapshots(all_results, OUT_DIR)
    plot_intruder_rank_penetration(all_results, OUT_DIR)

    with open(os.path.join(OUT_DIR, "summary.txt"), 'w') as f:
        f.write("Hierarchy Disruption Experiment\n")
        f.write("=" * 60 + "\n")
        f.write(f"Phase 1: {PHASE1_DAYS} days (hierarchy formation)\n")
        f.write(f"Phase 2: {PHASE2_DAYS} days (post-disruption)\n")
        f.write(f"Resident cohort: {NUM_FISH_COHORT} fish, {INIT_MASS_MIN}-{INIT_MASS_MAX}g\n")
        f.write(f"Intruders added: {NUM_INTRUDERS} fish\n\n")
        for line in summary_lines:
            f.write(line + "\n")
        f.write("\n\nExpected patterns:\n")
        f.write("  LARGER  : Δρ strongly negative → existing hierarchy collapses\n")
        f.write("  SMALLER : Δρ near zero or slight negative → hierarchy preserved, new bottom tier\n")
        f.write("  EQUAL   : Δρ moderately negative → stranger effect resets some ranks\n")
        f.write("  CONTROL : Δρ near zero → hierarchy self-maintains without disruption\n")

    print(f"\n[DONE] Results saved to {OUT_DIR}/")


if __name__ == "__main__":
    import sys as _sys
    _sys.stdout.reconfigure(line_buffering=True)
    main()
