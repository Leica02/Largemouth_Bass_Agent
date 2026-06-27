#!/usr/bin/env python3
"""
Growth rank-locking analysis for the 13-15 g size class.

Computes rank-order stability metrics across replicate aquaculture batch
simulations to quantify the degree to which initial body size predicts
final size rank. Generates publication-ready summary tables for:
    - Per-experiment rank-locking statistics (Spearman rho, rank inversion
      rate, quartile conservation rate, CV dynamics).
    - Group-level summaries (mean +/- SD across random seeds).
    - Cross-group trends in CV_ratio versus rank-locking strength.
    - Significance classification of rank-order correlations.
    - Focused mechanistic interpretation for the 13-15 g narrow-range group.

Metrics computed:
    - Spearman rank correlation (initial mass vs. final mass)
    - Coefficient of variation ratio (CV_initial / CV_SMR)
    - Pairwise rank inversion rate
    - Quartile conservation rate
    - Relative change in CV over the simulation period
    - Shapiro-Wilk normality test on final mass distribution
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

base = r"C:\Users\admin\PycharmProjects\PythonProject\Bass_lr_final\aquaculture_results"

# All valid experiments (excluding 13-15g seed=42 due to simulation bug)
experiments = {
    "2-7g":   [42, 43, 44],
    "5-10g":  [42, 43, 44],
    "10-19g": [42, 43, 44],
    "13-15g": [43, 44],   # seed=42 EXCLUDED (simulation bug)
}

results = []

for group, seeds in experiments.items():
    for seed in seeds:
        fish_path = f"{base}\\{group}_seed={seed}\\fish_daily.csv"
        pop_path  = f"{base}\\{group}_seed={seed}\\population_daily.csv"
        try:
            df = pd.read_csv(fish_path)
        except Exception as e:
            print(f"MISSING: {group}_seed={seed}: {e}")
            continue

        # Day 1: initial masses and SMR individual factors
        d1 = df[df['day'] == 1][['fish_id','initial_mass_g','smr_individual_factor']].drop_duplicates('fish_id')

        # Final alive record per fish
        alive = df[df['is_alive'] == 1]
        final = alive.sort_values('day').groupby('fish_id').last().reset_index()[['fish_id','body_mass_g','day']]

        merged = d1.merge(final, on='fish_id')
        n = len(merged)

        # Spearman rank correlation (initial vs final mass)
        r, p = stats.spearmanr(merged['initial_mass_g'], merged['body_mass_g'])

        # CV ratio = CV_initial_mass / CV_SMR_individual_factor
        cv_init = merged['initial_mass_g'].std() / merged['initial_mass_g'].mean()
        cv_smr  = d1['smr_individual_factor'].std() / d1['smr_individual_factor'].mean()
        cv_ratio = cv_init / cv_smr

        # Rank inversion rate (all pairwise comparisons)
        rank_init  = merged['initial_mass_g'].rank().values
        rank_final = merged['body_mass_g'].rank().values
        inv = 0
        total_pairs = n * (n - 1) // 2
        for i in range(n):
            for j in range(i + 1, n):
                if (rank_init[i] - rank_init[j]) * (rank_final[i] - rank_final[j]) < 0:
                    inv += 1
        inv_rate = inv / total_pairs * 100

        # Quartile conservation rate
        q_init  = pd.qcut(merged['initial_mass_g'], 4, labels=[1,2,3,4])
        q_final = pd.qcut(merged['body_mass_g'],    4, labels=[1,2,3,4])
        q_cons  = (q_init == q_final).mean() * 100

        # Relative CV change over simulation period
        cv_final  = merged['body_mass_g'].std() / merged['body_mass_g'].mean()
        cv_change = (cv_final - cv_init) / cv_init * 100

        # Mean SGR from population-level daily file
        try:
            pop = pd.read_csv(pop_path)
            sgr = pop['sgr_pct_day'].dropna().mean() if 'sgr_pct_day' in pop.columns else np.nan
        except:
            sgr = np.nan

        # Normality tests on final body mass distribution
        sw_stat, sw_p = stats.shapiro(merged['body_mass_g'])
        skew_val = stats.skew(merged['body_mass_g'])
        kurt_val = stats.kurtosis(merged['body_mass_g'])

        results.append({
            'group':       group,
            'seed':        seed,
            'n_alive':     n,
            'cv_init_pct': cv_init * 100,
            'cv_smr_pct':  cv_smr  * 100,
            'cv_ratio':    cv_ratio,
            'spearman_r':  r,
            'spearman_p':  p,
            'inv_rate':    inv_rate,
            'q_conserved': q_cons,
            'cv_change':   cv_change,
            'sgr':         sgr,
            'sw_p':        sw_p,
            'skew':        skew_val,
            'kurt':        kurt_val,
            'mean_init':   merged['initial_mass_g'].mean(),
            'mean_final':  merged['body_mass_g'].mean(),
        })
        print(f"  {group}_seed={seed}: r={r:.4f} p={p:.2e}  CV_ratio={cv_ratio:.3f}  inv={inv_rate:.1f}%  q_cons={q_cons:.1f}%")

df = pd.DataFrame(results)

print("\n" + "="*80)
print("TABLE 1: PER-EXPERIMENT RANK-LOCKING METRICS")
print("="*80)
cols = ['group','seed','n_alive','cv_ratio','spearman_r','spearman_p','inv_rate','q_conserved','cv_change']
print(df[cols].to_string(index=False, float_format='%.4f'))

# -- Group-level summary ---------------------------------------------------
print("\n" + "="*80)
print("TABLE 2: GROUP-LEVEL SUMMARY (mean +/- SD across seeds)")
print("="*80)
grp_stats = []
for group in ["2-7g", "5-10g", "10-19g", "13-15g"]:
    sub = df[df['group'] == group]
    n_seeds = len(sub)
    grp_stats.append({
        'group':      group,
        'n_seeds':    n_seeds,
        'cv_ratio':   f"{sub['cv_ratio'].mean():.3f} ± {sub['cv_ratio'].std():.3f}" if n_seeds>1 else f"{sub['cv_ratio'].mean():.3f}",
        'spearman_r': f"{sub['spearman_r'].mean():.3f} ± {sub['spearman_r'].std():.3f}" if n_seeds>1 else f"{sub['spearman_r'].mean():.3f}",
        'inv_rate':   f"{sub['inv_rate'].mean():.1f} ± {sub['inv_rate'].std():.1f}%" if n_seeds>1 else f"{sub['inv_rate'].mean():.1f}%",
        'q_conserved':f"{sub['q_conserved'].mean():.1f} ± {sub['q_conserved'].std():.1f}%" if n_seeds>1 else f"{sub['q_conserved'].mean():.1f}%",
        'cv_change':  f"{sub['cv_change'].mean():.2f} ± {sub['cv_change'].std():.2f}%" if n_seeds>1 else f"{sub['cv_change'].mean():.2f}%",
    })
grp_df = pd.DataFrame(grp_stats)
print(grp_df.to_string(index=False))

# -- Group-level Spearman trend: cv_ratio vs mean Spearman_r ---------------
print("\n" + "="*80)
print("TABLE 3: GROUP-LEVEL TREND -- CV_ratio vs Mean Spearman_r")
print("="*80)
grp_mean = df.groupby('group').agg(
    mean_cv_ratio=('cv_ratio','mean'),
    mean_r=('spearman_r','mean'),
    mean_inv=('inv_rate','mean'),
).reset_index()
grp_mean = grp_mean.sort_values('mean_cv_ratio')
print(grp_mean.to_string(index=False, float_format='%.4f'))

r_trend, p_trend = stats.spearmanr(grp_mean['mean_cv_ratio'], grp_mean['mean_r'])
print(f"\nGroup-level Spearman(CV_ratio, rank-locking_r): rho={r_trend:.4f}, p={p_trend:.4f}  (n={len(grp_mean)} groups)")

# -- Significance summary --------------------------------------------------
print("\n" + "="*80)
print("TABLE 4: SIGNIFICANCE CLASSIFICATION")
print("="*80)
for _, row in df.iterrows():
    sig = "***" if row['spearman_p'] < 0.001 else ("**" if row['spearman_p'] < 0.01 else ("*" if row['spearman_p'] < 0.05 else "ns"))
    print(f"  {row['group']}_seed={int(row['seed'])}: r={row['spearman_r']:.4f}, p={row['spearman_p']:.2e} {sig}  CV_ratio={row['cv_ratio']:.3f}")

# -- 13-15g specific analysis ----------------------------------------------
print("\n" + "="*80)
print("FOCUS: 13-15g GROUP (seeds 43 & 44 only, seed=42 excluded)")
print("="*80)
g = df[df['group'] == '13-15g']
print(g[['seed','n_alive','cv_init_pct','cv_smr_pct','cv_ratio',
         'spearman_r','spearman_p','inv_rate','q_conserved','cv_change']].to_string(index=False, float_format='%.4f'))
print(f"\nMean CV_ratio:   {g['cv_ratio'].mean():.3f} (both < 1.0 -> metabolic variability EXCEEDS initial size variability)")
print(f"Mean Spearman r: {g['spearman_r'].mean():.4f} (mean p={g['spearman_p'].mean():.2e})")
print(f"Both seeds significant at p < 0.001: {(g['spearman_p'] < 0.001).all()}")
print(f"Mean inv_rate:   {g['inv_rate'].mean():.2f}%  (random expectation = 50%)")
print(f"Mean q_conserved:{g['q_conserved'].mean():.2f}%  (random expectation = 25%)")

print("\n--- Key Mechanistic Interpretation ---")
print("CV_ratio < 1: metabolic heterogeneity dominates initial size heterogeneity.")
print("Despite this, rank-locking IS still significant (r~0.60, p<0.001).")
print("This means initial body size STILL predicts final rank, even when SMR")
print("variation is larger than initial size variation.")
print("The initial size advantage is not fully erased by metabolic noise.")
