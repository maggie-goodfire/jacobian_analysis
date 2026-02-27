#!/usr/bin/env python
"""
NB03 Figure generation — runs in conda Python 3.10 environment.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

OUT_DIR = '/mnt/polished-lake/home/mbeheleramass/jacobian_analysis'

# ─── Load data ────────────────────────────────────────────────────────────────
df      = pd.read_parquet(f'{OUT_DIR}/nb03_jacobian_results.parquet')
df_dist = pd.read_parquet(f'{OUT_DIR}/nb03_distribution_hspa5_arfgef2.parquet')

print(f"Panel data: {df.shape}")
print(f"Distribution data: {df_dist.shape}")

sec_kds      = ['TRAPPC3', 'TRAPPC5', 'SRP54', 'HSPA5']
neg_ctrl_kds = ['MYC', 'BRCA1', 'CHEK1']
ribo_kds     = ['RPL10', 'RPS6', 'RPL12', 'RPS8']

# ─── Figure 1: Secretory Jacobian Heatmap ────────────────────────────────────
print("\nCreating secretory heatmap...")
df_sec = df[df['pathway'] == 'secretory'].copy()

# Create pair label
df_sec['pair'] = df_sec['regulator'] + '→' + df_sec['target']
all_kds_sec    = sec_kds + neg_ctrl_kds
pairs_sec      = sorted(df_sec['pair'].unique())

# Build matrix: rows=KDs, cols=pairs
mat_sec = np.zeros((len(all_kds_sec), len(pairs_sec)))
for i, kd in enumerate(all_kds_sec):
    for j, pair in enumerate(pairs_sec):
        row = df_sec[(df_sec['kd'] == kd) & (df_sec['pair'] == pair)]
        if len(row) > 0:
            mat_sec[i, j] = row['J'].values[0]

fig, ax = plt.subplots(figsize=(14, 5))
im = ax.imshow(mat_sec, cmap='RdBu_r', aspect='auto',
               vmin=-np.percentile(np.abs(mat_sec), 95),
               vmax=np.percentile(np.abs(mat_sec), 95))
ax.set_xticks(range(len(pairs_sec)))
ax.set_xticklabels(pairs_sec, rotation=45, ha='right', fontsize=7)
ax.set_yticks(range(len(all_kds_sec)))
ax.set_yticklabels(all_kds_sec, fontsize=9)

# Draw separator between pathway KDs and negative controls
sep_y = len(sec_kds) - 0.5
ax.axhline(sep_y, color='black', linewidth=2, linestyle='--')

# Color-code y-axis labels
for i, lbl in enumerate(all_kds_sec):
    color = '#d62728' if lbl in sec_kds else '#1f77b4'
    ax.get_yticklabels()[i].set_color(color)

plt.colorbar(im, ax=ax, label='J = dY[target]/dX[regulator]')
ax.set_title('Secretory Pathway Targeted Jacobian\n'
             'Red labels = secretory KDs, Blue = negative controls\n'
             'Rows = KD condition, Cols = (regulator, target) gene pair',
             fontsize=10)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/nb03_secretory_jacobian_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: nb03_secretory_jacobian_heatmap.png")


# ─── Figure 2: Ribosome Jacobian Heatmap ─────────────────────────────────────
print("Creating ribosome heatmap...")
df_ribo = df[df['pathway'] == 'ribosome'].copy()
df_ribo['pair'] = df_ribo['regulator'] + '→' + df_ribo['target']
all_kds_ribo    = ribo_kds + neg_ctrl_kds
pairs_ribo      = sorted(df_ribo['pair'].unique())

mat_ribo = np.zeros((len(all_kds_ribo), len(pairs_ribo)))
for i, kd in enumerate(all_kds_ribo):
    for j, pair in enumerate(pairs_ribo):
        row = df_ribo[(df_ribo['kd'] == kd) & (df_ribo['pair'] == pair)]
        if len(row) > 0:
            mat_ribo[i, j] = row['J'].values[0]

fig, ax = plt.subplots(figsize=(14, 5))
im = ax.imshow(mat_ribo, cmap='RdBu_r', aspect='auto',
               vmin=-np.percentile(np.abs(mat_ribo), 95),
               vmax=np.percentile(np.abs(mat_ribo), 95))
ax.set_xticks(range(len(pairs_ribo)))
ax.set_xticklabels(pairs_ribo, rotation=45, ha='right', fontsize=7)
ax.set_yticks(range(len(all_kds_ribo)))
ax.set_yticklabels(all_kds_ribo, fontsize=9)

sep_y = len(ribo_kds) - 0.5
ax.axhline(sep_y, color='black', linewidth=2, linestyle='--')
for i, lbl in enumerate(all_kds_ribo):
    color = '#d62728' if lbl in ribo_kds else '#1f77b4'
    ax.get_yticklabels()[i].set_color(color)

plt.colorbar(im, ax=ax, label='J = dY[target]/dX[regulator]')
ax.set_title('Ribosome Pathway Targeted Jacobian\n'
             'Red labels = ribosomal KDs, Blue = negative controls\n'
             'Rows = KD condition, Cols = (regulator, target) gene pair',
             fontsize=10)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/nb03_ribosome_jacobian_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: nb03_ribosome_jacobian_heatmap.png")


# ─── Figure 3: Distribution across ALL KDs (HSPA5 -> ARFGEF2) ────────────────
print("Creating distribution figure...")
df_dist_sorted = df_dist.sort_values('J').reset_index(drop=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: histogram with KD categories highlighted
ax = axes[0]
all_J   = df_dist['J'].values
sec_J   = df_dist[df_dist['is_secretory_kd']]['J'].values
neg_J   = df_dist[df_dist['is_neg_ctrl']]['J'].values
ribo_J  = df_dist[df_dist['is_ribo_kd']]['J'].values

ax.hist(all_J, bins=60, alpha=0.5, color='gray', label=f'All KDs (n={len(all_J)})')
for j_val, lbl in zip(sec_J, sec_kds):
    ax.axvline(j_val, color='#d62728', alpha=0.8, linewidth=2, label=f'Sec: {lbl}' if j_val == sec_J[0] else None)
for j_val, lbl in zip(neg_J, neg_ctrl_kds):
    ax.axvline(j_val, color='#1f77b4', alpha=0.8, linewidth=2, linestyle='--', label=f'Neg: {lbl}' if j_val == neg_J[0] else None)

# Label individual lines
for j_val, lbl in zip(list(sec_J) + list(neg_J), sec_kds + neg_ctrl_kds):
    ax.annotate(lbl, xy=(j_val, 0), xytext=(j_val, ax.get_ylim()[1]*0.05 if ax.get_ylim()[1] > 0 else 5),
                fontsize=7, rotation=90, ha='center', va='bottom',
                color='#d62728' if lbl in sec_kds else '#1f77b4')

ax.set_xlabel('J = dY[ARFGEF2]/dX[HSPA5]')
ax.set_ylabel('Count')
ax.set_title('Distribution of Targeted Jacobian\nacross all 2024 KDs\n(HSPA5 → ARFGEF2)')
ax.legend(fontsize=7)

# Right: rank plot — rank of secretory and neg ctrl KDs
ax2 = axes[1]
n_total = len(df_dist)
sec_ranks = []
for kd in sec_kds:
    row = df_dist[df_dist['kd'] == kd]
    if len(row) > 0:
        rank_pct = (df_dist_sorted[df_dist_sorted['kd'] == kd].index[0] + 1) / n_total * 100
        sec_ranks.append((kd, rank_pct, row['J'].values[0]))
neg_ranks = []
for kd in neg_ctrl_kds:
    row = df_dist[df_dist['kd'] == kd]
    if len(row) > 0:
        rank_pct = (df_dist_sorted[df_dist_sorted['kd'] == kd].index[0] + 1) / n_total * 100
        neg_ranks.append((kd, rank_pct, row['J'].values[0]))

# Scatter: pathway vs neg ctrl positions
x_labels = [r[0] for r in sec_ranks + neg_ranks]
x_values = list(range(len(x_labels)))
y_values = [r[2] for r in sec_ranks + neg_ranks]
colors   = ['#d62728'] * len(sec_ranks) + ['#1f77b4'] * len(neg_ranks)

ax2.scatter(x_values, y_values, c=colors, s=100, zorder=5)
ax2.axhline(np.mean(all_J), color='gray', linestyle='--', alpha=0.7, label=f'Mean J = {np.mean(all_J):.4f}')
for xi, yi, lbl in zip(x_values, y_values, x_labels):
    rk_pct = (sec_ranks + neg_ranks)[[r[0] for r in sec_ranks + neg_ranks].index(lbl)][1]
    ax2.annotate(f'{lbl}\n(rank {rk_pct:.0f}%ile)', (xi, yi), textcoords='offset points',
                 xytext=(0, 10), ha='center', fontsize=8)
ax2.set_xticks([])
ax2.set_ylabel('J = dY[ARFGEF2]/dX[HSPA5]')
ax2.set_title('J values for secretory (red) vs\nneg ctrl (blue) KDs\n(rank percentile from bottom)')
ax2.legend()

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/nb03_jacobian_distribution_all_kds.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: nb03_jacobian_distribution_all_kds.png")


# ─── Statistical tests ────────────────────────────────────────────────────────
print("\n=== Statistical Tests ===")

for pathway_name in ['secretory', 'ribosome']:
    df_p = df[df['pathway'] == pathway_name]
    pathway_J = df_p[df_p['pathway_kd'] == True]['J'].values
    neg_J     = df_p[df_p['pathway_kd'] == False]['J'].values

    if len(pathway_J) > 0 and len(neg_J) > 0:
        stat, pval = mannwhitneyu(pathway_J, neg_J, alternative='greater')
        print(f"\n{pathway_name.upper()} panel:")
        print(f"  Pathway KDs  n={len(pathway_J)}, mean={np.mean(pathway_J):.6f}, std={np.std(pathway_J):.6f}")
        print(f"  Neg ctrl KDs n={len(neg_J)},  mean={np.mean(neg_J):.6f}, std={np.std(neg_J):.6f}")
        print(f"  Mann-Whitney U (pathway > neg), p={pval:.4f}")
        if pval < 0.05:
            print(f"  => SIGNIFICANT at p<0.05 (pathway KDs have higher J)")
        else:
            print(f"  => NOT significant (no evidence for pathway-specific routing)")


# ─── Bonus scatter: per-regulator-target pair, pathway KDs vs neg ctrl KDs ───
print("\nCreating bonus scatter figure...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, pathway_name in zip(axes, ['secretory', 'ribosome']):
    df_p    = df[df['pathway'] == pathway_name].copy()
    df_p['pair'] = df_p['regulator'] + '→' + df_p['target']
    pairs   = sorted(df_p['pair'].unique())

    # For each pair: mean J under pathway KDs vs mean J under neg KDs
    xs, ys, lbls = [], [], []
    for pair in pairs:
        sub = df_p[df_p['pair'] == pair]
        mean_pathway = sub[sub['pathway_kd'] == True]['J'].mean()
        mean_neg     = sub[sub['pathway_kd'] == False]['J'].mean()
        xs.append(mean_neg)
        ys.append(mean_pathway)
        lbls.append(pair)

    ax.scatter(xs, ys, s=50, alpha=0.8)
    # Diagonal reference line
    mn = min(min(xs), min(ys))
    mx = max(max(xs), max(ys))
    ax.plot([mn, mx], [mn, mx], 'k--', alpha=0.5, label='y=x')
    for xi, yi, lbl in zip(xs, ys, lbls):
        ax.annotate(lbl, (xi, yi), fontsize=6, alpha=0.8)
    ax.set_xlabel('Mean J under neg ctrl KDs')
    ax.set_ylabel('Mean J under pathway KDs')
    ax.set_title(f'{pathway_name.capitalize()} pathway\nPathway KDs vs Neg Ctrl KDs\n(one dot per gene pair)')
    ax.legend()

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/nb03_pathway_vs_negctrl_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: nb03_pathway_vs_negctrl_scatter.png")
print("\nAll figures saved.")
