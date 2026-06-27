#!/usr/bin/env python3
"""
PPO Action-Recording Simulation
=================================
Runs a 30-day aquaculture batch simulation (8-22 g initial mass, seed=42,
60 fish) with per-step action logging.  Every SAMPLE_EVERY steps, the raw
PPO action vector and associated state variables are recorded for each
living fish, enabling post-hoc behavioural analysis of the learned policy.

Output
------
  aquaculture_results/8-22g_actions_seed=42/
    fish_actions.csv       -- day, step, fish_id, action_0..4, pos_x/y/z,
                              body_mass_g, energy_pct, stomach_full_pct
    fish_daily.csv         -- standard daily per-fish log
    population_daily.csv   -- daily population-level summary
"""
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(line_buffering=True)

import aquaculture_batch_sim as _sim_mod
# Override globals before instantiating the simulation
_sim_mod.INIT_MASS_MIN = 8.0
_sim_mod.INIT_MASS_MAX = 22.0
_sim_mod.NUM_FISH      = 60

from aquaculture_batch_sim import (
    AquacultureSimulation, STEPS_PER_DAY,
    export_fish_daily_csv, export_population_daily_csv, export_fish_actions_csv,
    MODEL_PATH,
)

OUT_DIR: str      = "aquaculture_results/8-22g_actions_seed=42"
MASS_MIN: float   = 8.0
MASS_MAX: float   = 22.0
SEED: int         = 42
N_DAYS: int       = 30
N_FISH: int       = 60
SAMPLE_EVERY: int = 10   # record actions every N simulation steps

os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 60)
print("  PPO action-recording run")
print("  Group: 8-22g  seed=%d  %d days  %d fish" % (SEED, N_DAYS, N_FISH))
print("  Action sample: every %d steps" % SAMPLE_EVERY)
print("  Output: %s/" % OUT_DIR)
print("=" * 60)

sim = AquacultureSimulation(
    seed=SEED,
    num_fish=N_FISH,
    total_steps=N_DAYS * STEPS_PER_DAY,
)
sim.enable_action_recording(interval=SAMPLE_EVERY)
sim.run()

print("\nExporting CSVs...")
export_fish_daily_csv(sim, OUT_DIR)
export_population_daily_csv(sim, OUT_DIR)
export_fish_actions_csv(sim, OUT_DIR)

# Sanity check on recorded action data
import numpy as np
if sim._action_rows:
    import pandas as pd
    df = pd.DataFrame(sim._action_rows[:5])
    print("\nFirst 5 action rows:")
    print(df.to_string(index=False))
    print("\nTotal rows: %d" % len(sim._action_rows))
    expected = N_DAYS * STEPS_PER_DAY // SAMPLE_EVERY * N_FISH
    print("Expected ~%d rows (assuming 100%% survival)" % expected)

print("\n[DONE]")
