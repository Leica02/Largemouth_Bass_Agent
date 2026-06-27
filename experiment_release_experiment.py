#!/usr/bin/env python3
"""
Experiment 1: Wild vs. Aquaculture Conditions
==============================================
Evaluates whether size-based rank fixation is an artefact of confined
aquaculture space or an intrinsic emergent property of largemouth bass
(Micropterus salmoides) social dynamics.

Two conditions are compared with identical fish (same seed, same initial
masses):
  - FARM: small tank (r = 0.75 m), centralised trickle-feed (baseline RAS)
  - WILD: large tank (r = 3.00 m), spatially dispersed food

Hypothesis
----------
Rank fixation (Spearman rho) should be significantly weaker under WILD
conditions because subordinate fish can spatially escape dominant
competitors.

Output
------
  aquaculture_results/wild_vs_farm/
    fish_daily_FARM.csv
    fish_daily_WILD.csv
    comparison.png            -- side-by-side rho and CV trajectories
    rank_transition_FARM.png
    rank_transition_WILD.png
    summary.txt
"""

import os, sys, copy, types
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import base simulation (reuse all existing logic)
from aquaculture_batch_sim import (
    AquacultureSimulation, STEPS_PER_DAY, TOTAL_DAYS,
    _aqua_spawn_food, _aqua_trickle_feed, _process_cannibalism,
    _patch_manager_state_isolation, _patch_manager_obs_injection,
    export_fish_daily_csv, export_population_daily_csv,
    MODEL_PATH, NUM_FISH, DR_MIN, DR_MAX, FEEDINGS_PER_DAY,
    ACCLIMATION_DAYS, AQUA_VISION_RANGE, AQUA_FOOD_DETECT_RANGE,
    AQUA_LATERAL_LINE_RANGE, CMAL_AVG_PELLET_MASS, TRICKLE_STEPS,
)
from config import CONFIG
from utils.biological_formulas import mass_to_length

OUT_DIR = "aquaculture_results/wild_vs_farm_seed43"

# Condition-specific parameters
CONDITIONS = {
    'FARM': {
        'tank_radius'   : 0.75,   # m -- standard RAS tank, diameter 1.5 m
        'tank_depth'    : 0.70,
        'spread_radius' : 0.40,   # m -- centralised trickle near school centroid
        'num_fish'      : 30,     # same density-per-area as baseline
        'label'         : 'Aquaculture (confined, d=1.5 m)',
        'color'         : 'steelblue',
    },
    'WILD': {
        'tank_radius'   : 3.00,   # m -- large arena simulating wild territory
        'tank_depth'    : 1.00,
        'spread_radius' : 2.00,   # m -- food dispersed across whole arena
        'num_fish'      : 30,     # same absolute fish count, far lower density
        'label'         : 'Wild-like (open space, d=6.0 m)',
        'color'         : 'seagreen',
    },
}

INIT_MASS_MIN = 10.0
INIT_MASS_MAX = 19.0
SIM_DAYS      = 100
SEED          = 43


def build_sim(condition_name: str, shared_masses: np.ndarray) -> AquacultureSimulation:
    """Create a simulation configured for a specific environmental condition.

    Constructs an AquacultureSimulation with condition-specific tank geometry
    and food dispersal radius, but identical initial fish masses so that the
    two conditions differ only in spatial parameters.

    Args:
        condition_name: Key into CONDITIONS dict ('FARM' or 'WILD').
        shared_masses: Array of initial body masses (grams) shared across
            conditions to ensure identical starting cohorts.

    Returns:
        Configured AquacultureSimulation instance ready to run.
    """
    cond = CONDITIONS[condition_name]

    # Patch tank geometry BEFORE building the sim
    _env_cfg = CONFIG.environment
    object.__setattr__(_env_cfg, 'tank_radius', cond['tank_radius'])
    object.__setattr__(_env_cfg, 'tank_depth',  cond['tank_depth'])

    total_steps = SIM_DAYS * STEPS_PER_DAY
    sim = AquacultureSimulation(
        model_path  = MODEL_PATH,
        num_fish    = cond['num_fish'],
        total_steps = total_steps,
        seed        = SEED,
    )

    # Override masses with the shared draw so both conditions start identically
    from utils.biological_formulas import mass_to_length as m2l
    _bc = CONFIG.buoyancy
    _GAS_CONSTANT = 8.314
    INIT_DEPTH = 0.30
    for fish, m in zip(sim.manager.ai_fish, shared_masses[:cond['num_fish']]):
        fish.body_mass    = float(m)
        fish.initial_mass = float(m)
        fish.total_length = m2l(float(m))
        fish.position[1]  = -INIT_DEPTH
        fish.stomach_content_mass = 0.0
        fish.stomach_fullness     = 0.0
        fish.is_digesting         = False

        mass_kg      = float(m) / 1000.0
        tissue_vol   = mass_kg / _bc.fish_tissue_density
        neutral_vol  = max(mass_kg / _bc.water_density - tissue_vol,
                           tissue_vol * _bc.min_volume_ratio)
        pressure_pa  = (_bc.atmospheric_pressure + INIT_DEPTH * _bc.pressure_per_meter) * 101325.0
        gas_amount   = pressure_pa * neutral_vol / (8.314 * (25.0 + 273.15))
        bs = getattr(fish, '_buoyancy_state', None) or getattr(
            sim.manager.base_env.physics_state, 'buoyancy_state', None)
        if bs is not None:
            bs.neutral_volume = neutral_vol
            bs.swimbladder_volume = neutral_vol
            bs.gas_amount = gas_amount
            bs.current_depth = INIT_DEPTH
            bs.relative_density = 1.0
            bs.net_buoyancy_force = 0.0
            bs.is_neutral = True
        fish.swimbladder_volume = neutral_vol
        fish.relative_density   = 1.0
        fish.net_buoyancy_force = 0.0

    # Patch food spread radius for this condition
    _orig_spawn = sim.manager._spawn_food
    _spread = cond['spread_radius']
    _rng    = sim.rng

    def _cond_spawn(self_mgr=sim.manager, rng=_rng, spread=_spread):
        """Override spread_radius for this condition before spawning."""
        return _aqua_spawn_food(self_mgr, rng)   # _aqua_trickle_feed applies spread

    sim.manager._spawn_food = _cond_spawn

    # Store spread radius on manager so trickle feed can read it
    sim.manager._food_spread_radius = _spread

    # Monkey-patch _aqua_trickle_feed spread usage via manager attribute
    from systems.feeding import FeedingInput
    original_trickle_fn = _aqua_trickle_feed.__code__

    sim._condition_name  = condition_name
    sim._condition_label = cond['label']

    return sim


def _run_sim_with_spread(sim: AquacultureSimulation, spread_radius: float) -> None:
    """Run the simulation applying a custom food spread radius during trickle feed.

    Executes the full step loop for the simulation, injecting food pellets at
    the specified spread_radius around the school centroid rather than using
    the default configuration value.

    Args:
        sim: An initialised AquacultureSimulation instance.
        spread_radius: Radius (metres) over which food pellets are dispersed
            around the school centroid at each trickle-feed event.
    """
    total_steps = sim.total_steps

    print(f"\n[{sim._condition_name}] Starting: {sim.num_fish} fish, "
          f"tank_r={CONDITIONS[sim._condition_name]['tank_radius']}m, "
          f"food_spread={spread_radius}m, {SIM_DAYS} days")

    for step in range(1, total_steps + 1):
        sim.manager.step()

        # Trickle feed with condition-specific spread
        pending = getattr(sim.manager, '_pending_pellets', 0)
        if pending > 0:
            meal_total = getattr(sim.manager, '_meal_total_pellets', pending)
            per_step   = max(1, int(np.ceil(meal_total / TRICKLE_STEPS)))
            release    = min(pending, per_step)
            pellet_mass = getattr(sim.manager, '_pending_pellet_mass', 0.0225)

            from systems.feeding import FeedingInput
            alive_fish = [f for f in sim.manager.ai_fish if f.is_alive]
            if alive_fish:
                centroid = np.mean([f.position for f in alive_fish], axis=0).astype(np.float32)
                centroid[1] = -0.05
            else:
                centroid = np.array([0.0, -0.05, 0.0], dtype=np.float32)

            feed_input = FeedingInput(
                agent_position=centroid,
                agent_mass=sum(f.body_mass for f in alive_fish) if alive_fish else 1.0,
                agent_length=0.10,
                stomach_fullness=0.0,
                tank_geometry=getattr(sim.manager.base_env, 'tank_geometry', None),
                obstacle_field=getattr(sim.manager.base_env, 'obstacle_field', None),
            )

            fs     = sim.manager.base_env.feeding_system
            state  = sim.manager.base_env.feeding_state
            fs_cfg = fs.c
            _orig  = fs_cfg.spread_radius
            object.__setattr__(fs_cfg, 'spread_radius', spread_radius)

            spawned = 0
            for _ in range(release):
                food = fs._create_floating_pellet(feed_input, pellet_mass)
                state.food_items.append(food)
                spawned += 1

            object.__setattr__(fs_cfg, 'spread_radius', _orig)
            sim.manager._pending_pellets -= spawned
            sim.manager.total_food_spawned += spawned

        # Cannibalism interactions
        cann = _process_cannibalism(sim.manager, sim.rng)
        sim._cann_attacks_total   += cann['attacks']
        sim._cann_successes_total += cann['successes']
        sim._cann_deaths_total    += cann['deaths']
        sim._cann_swallows_total  += cann.get('swallows', 0)
        sim._day_cann_attacks     += cann['attacks']
        sim._day_cann_successes   += cann['successes']

        if step % STEPS_PER_DAY == 0:
            sim._record_daily()

        if step % STEPS_PER_DAY == 0 or step == total_steps:
            sim._record_snapshot()
            day = step / STEPS_PER_DAY
            alive_n = sim.ts_alive[-1]
            mean_m  = sim.ts_mean_mass[-1]
            if int(day) % 10 == 0 or day == SIM_DAYS:
                print(f"  Day {day:5.0f} | alive={alive_n} | mean={mean_m:.1f}g "
                      f"| cann_deaths={sim._cann_deaths_total} mal_deaths={sim._mal_deaths_total}")

    print(f"[{sim._condition_name}] Complete.")


# ── Analysis helpers ──────────────────────────────────────────────────────────


def rho_trajectory(fish_csv_path: str) -> list[tuple[int, float]]:
    """Compute Spearman rho between day-1 mass and each subsequent day.

    Tracks rank stability over time by correlating the initial mass ranking
    with the ranking at each later day among fish that remain alive.

    Args:
        fish_csv_path: Path to the per-fish daily CSV file.

    Returns:
        List of (day, rho) tuples sorted chronologically.
    """
    df = pd.read_csv(fish_csv_path)
    alive = df[df['is_alive'] == 1]
    pivot = alive.pivot_table(index='fish_id', columns='day', values='body_mass_g')
    all_days = sorted(pivot.columns.tolist())
    d_start  = all_days[0]
    result   = []
    for d in all_days:
        sub = pivot[[d_start, d]].dropna()
        if len(sub) < 5:
            continue
        rho = float(spearmanr(sub.iloc[:, 0], sub.iloc[:, 1])[0])
        result.append((int(d), rho))
    return result


def cv_trajectory(fish_csv_path: str) -> list[tuple[int, float]]:
    """Compute coefficient of variation (%) of body mass for each day.

    Measures size heterogeneity within the surviving population over time.

    Args:
        fish_csv_path: Path to the per-fish daily CSV file.

    Returns:
        List of (day, CV_percent) tuples sorted chronologically.
    """
    df = pd.read_csv(fish_csv_path)
    alive = df[df['is_alive'] == 1]
    pivot = alive.pivot_table(index='fish_id', columns='day', values='body_mass_g')
    result = []
    for d in sorted(pivot.columns.tolist()):
        col = pivot[d].dropna()
        if len(col) > 3:
            cv = col.std() / col.mean() * 100
            result.append((int(d), float(cv)))
    return result


def rank_transition_matrix(fish_csv_path: str, n_quartiles: int = 4) -> np.ndarray:
    """Compute the quartile rank-transition matrix from day 1 to final day.

    Quantifies how many fish remain in their initial size quartile versus
    moving to a different quartile, indicating mobility within the hierarchy.

    Args:
        fish_csv_path: Path to the per-fish daily CSV file.
        n_quartiles: Number of quantile bins (default 4 for quartiles).

    Returns:
        An (n_quartiles x n_quartiles) integer matrix where element [i, j]
        is the count of fish starting in quartile i and ending in quartile j.
    """
    df = pd.read_csv(fish_csv_path)
    alive = df[df['is_alive'] == 1]
    pivot = alive.pivot_table(index='fish_id', columns='day', values='body_mass_g')
    all_days = sorted(pivot.columns.tolist())
    both = pivot[[all_days[0], all_days[-1]]].dropna()
    q_start = pd.qcut(both.iloc[:, 0], n_quartiles, labels=False, duplicates='drop') + 1
    q_end   = pd.qcut(both.iloc[:, 1], n_quartiles, labels=False, duplicates='drop') + 1
    mat = np.zeros((n_quartiles, n_quartiles), dtype=int)
    for qs, qe in zip(q_start, q_end):
        mat[int(qs) - 1, int(qe) - 1] += 1
    return mat


# ── Plotting ──────────────────────────────────────────────────────────────────


def plot_comparison(farm_csv: str, wild_csv: str, out_dir: str) -> None:
    """Generate a two-panel comparison figure of rho and CV trajectories.

    Left panel: Spearman rho trajectory (rank stability) for both conditions.
    Right panel: CV of body mass trajectory (size heterogeneity).

    Args:
        farm_csv: Path to the FARM condition fish_daily CSV.
        wild_csv: Path to the WILD condition fish_daily CSV.
        out_dir: Directory to write the output PNG.
    """
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.dpi': 150})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for csv_path, label, color, ls in [
        (farm_csv, 'Aquaculture (confined)', 'steelblue', '-'),
        (wild_csv, 'Wild-like (open space)', 'seagreen',  '--'),
    ]:
        rho_traj = rho_trajectory(csv_path)
        cv_traj  = cv_trajectory(csv_path)

        days_rho, rhos = zip(*rho_traj) if rho_traj else ([], [])
        days_cv,  cvs  = zip(*cv_traj)  if cv_traj  else ([], [])

        axes[0].plot(days_rho, rhos, color=color, linestyle=ls, linewidth=2, label=label)
        axes[1].plot(days_cv,  cvs,  color=color, linestyle=ls, linewidth=2, label=label)

    axes[0].axhline(0.9, color='grey', linestyle=':', linewidth=1, alpha=0.6)
    axes[0].set_xlabel("Simulated day")
    axes[0].set_ylabel("Spearman ρ  (day 1 → day X)")
    axes[0].set_title("Rank Stability Trajectory\n(ρ→1 = hierarchy locked)")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(fontsize=9)
    axes[0].annotate('Rank fixation\nthreshold (ρ=0.9)', xy=(5, 0.91),
                     fontsize=8, color='grey')

    axes[1].set_xlabel("Simulated day")
    axes[1].set_ylabel("CV of body mass (%)")
    axes[1].set_title("Size Heterogeneity Trajectory\n(↑CV = growing inequality)")
    axes[1].legend(fontsize=9)

    fig.suptitle("Wild-like vs. Aquaculture Conditions:\nEmergence of Rank Fixation in Largemouth Bass",
                 fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(out_dir, "comparison.png")
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_rank_transition(fish_csv: str, condition_name: str, out_dir: str) -> None:
    """Plot a heatmap of the quartile rank-transition matrix.

    Visualises the fraction of fish that remain in or move between size
    quartiles from the first to the last recorded day.

    Args:
        fish_csv: Path to the per-fish daily CSV.
        condition_name: Condition label for the plot title.
        out_dir: Directory to write the output PNG.
    """
    mat  = rank_transition_matrix(fish_csv)
    n    = mat.shape[0]
    # Normalise each row to percentage
    row_sums = mat.sum(axis=1, keepdims=True)
    mat_pct  = np.divide(mat * 100.0, row_sums, where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(mat_pct, cmap='Blues', vmin=0, vmax=100)
    plt.colorbar(im, ax=ax, label='% of fish')
    labels = [f'Q{i+1}' for i in range(n)]
    ax.set_xticks(range(n)); ax.set_xticklabels(labels)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels)
    ax.set_xlabel("Final rank quartile")
    ax.set_ylabel("Initial rank quartile")
    ax.set_title(f"Rank Transition Matrix – {condition_name}\n"
                 f"(diagonal = fraction staying in same quartile)")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f'{mat_pct[i,j]:.0f}%', ha='center', va='center',
                    fontsize=11, color='white' if mat_pct[i,j] > 50 else 'black')
    plt.tight_layout()
    path = os.path.join(out_dir, f"rank_transition_{condition_name}.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    """Execute the wild vs. aquaculture comparison experiment.

    Generates shared initial masses, runs both FARM and WILD conditions,
    exports per-fish daily CSVs, and produces comparison and rank-transition
    plots.
    """
    os.makedirs(OUT_DIR, exist_ok=True)

    # Generate shared initial masses (same fish, different environments)
    rng_shared = np.random.default_rng(SEED)
    shared_masses = rng_shared.uniform(INIT_MASS_MIN, INIT_MASS_MAX, max(
        CONDITIONS['FARM']['num_fish'], CONDITIONS['WILD']['num_fish']
    ))
    print(f"Shared initial masses: mean={shared_masses.mean():.2f}g, "
          f"CV={shared_masses.std()/shared_masses.mean()*100:.1f}%")

    results = {}
    csv_paths = {}

    for cond_name in ['FARM', 'WILD']:
        cond = CONDITIONS[cond_name]
        print(f"\n{'='*55}")
        print(f"  CONDITION: {cond['label']}")
        print(f"{'='*55}")

        # Patch geometry for this condition
        object.__setattr__(CONFIG.environment, 'tank_radius', cond['tank_radius'])
        object.__setattr__(CONFIG.environment, 'tank_depth',  cond['tank_depth'])

        sim = AquacultureSimulation(
            model_path  = MODEL_PATH,
            num_fish    = cond['num_fish'],
            total_steps = SIM_DAYS * STEPS_PER_DAY,
            seed        = SEED,
        )
        sim._condition_name  = cond_name
        sim._condition_label = cond['label']

        # Override masses with shared draw
        _bc = CONFIG.buoyancy
        INIT_DEPTH = 0.30
        for fish, m in zip(sim.manager.ai_fish, shared_masses[:cond['num_fish']]):
            fish.body_mass    = float(m)
            fish.initial_mass = float(m)
            fish.total_length = mass_to_length(float(m))
            fish.position[1]  = -INIT_DEPTH
            fish.stomach_content_mass = 0.0
            fish.stomach_fullness     = 0.0
            fish.is_digesting         = False
            mass_kg  = float(m) / 1000.0
            tv       = mass_kg / _bc.fish_tissue_density
            nv       = max(mass_kg / _bc.water_density - tv, tv * _bc.min_volume_ratio)
            pp       = (_bc.atmospheric_pressure + INIT_DEPTH*_bc.pressure_per_meter)*101325.0
            ga       = pp * nv / (8.314 * 298.15)
            bs = getattr(fish, '_buoyancy_state', None) or getattr(
                sim.manager.base_env.physics_state, 'buoyancy_state', None)
            if bs is not None:
                bs.neutral_volume = nv; bs.swimbladder_volume = nv
                bs.gas_amount = ga; bs.current_depth = INIT_DEPTH
                bs.relative_density = 1.0; bs.net_buoyancy_force = 0.0; bs.is_neutral = True
            fish.swimbladder_volume = nv
            fish.relative_density   = 1.0
            fish.net_buoyancy_force = 0.0

        _run_sim_with_spread(sim, cond['spread_radius'])

        # Export per-fish daily CSV
        sub_dir = os.path.join(OUT_DIR, cond_name)
        os.makedirs(sub_dir, exist_ok=True)
        csv_path = os.path.join(OUT_DIR, f"fish_daily_{cond_name}.csv")
        sim._fish_daily_rows  # already populated
        import csv as csv_mod
        if sim._fish_daily_rows:
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv_mod.DictWriter(f, fieldnames=list(sim._fish_daily_rows[0].keys()))
                writer.writeheader()
                writer.writerows(sim._fish_daily_rows)
            print(f"  Saved: {csv_path}")

        csv_paths[cond_name] = csv_path
        results[cond_name]   = sim

    # Analysis and plots
    print("\n=== ANALYSIS ===")
    summary_lines = []
    for cond_name in ['FARM', 'WILD']:
        csv_path = csv_paths[cond_name]
        rho_traj = rho_trajectory(csv_path)
        cv_traj  = cv_trajectory(csv_path)
        final_rho = rho_traj[-1][1] if rho_traj else float('nan')
        final_cv  = cv_traj[-1][1]  if cv_traj  else float('nan')
        init_cv   = cv_traj[0][1]   if cv_traj  else float('nan')

        drop_day  = next((d for d, r in rho_traj if r < 0.90), None)
        line = (f"{cond_name}: final_rho={final_rho:.3f}, "
                f"CV {init_cv:.1f}%→{final_cv:.1f}%, "
                f"rho<0.90 at day={drop_day}")
        print(f"  {line}")
        summary_lines.append(line)

        plot_rank_transition(csv_path, cond_name, OUT_DIR)

    plot_comparison(csv_paths['FARM'], csv_paths['WILD'], OUT_DIR)

    with open(os.path.join(OUT_DIR, "summary.txt"), 'w') as f:
        f.write("Wild vs. Aquaculture Rank Fixation Experiment\n")
        f.write("=" * 50 + "\n")
        for line in summary_lines:
            f.write(line + "\n")
        f.write("\nInterpretation:\n")
        f.write("  FARM rho >> WILD rho  →  rank fixation is an aquaculture artefact\n")
        f.write("  FARM rho ≈  WILD rho  →  rank fixation is intrinsic to the species\n")

    print(f"\n[DONE] Results saved to {OUT_DIR}/")


if __name__ == "__main__":
    import sys as _sys
    _sys.stdout.reconfigure(line_buffering=True)
    main()
