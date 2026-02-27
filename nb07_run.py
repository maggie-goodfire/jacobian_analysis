
import numpy as np
import pandas as pd
import pickle
from scipy.stats import pearsonr, spearmanr, sem
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTDIR = '/mnt/polished-lake/home/mbeheleramass/jacobian_analysis'

jac5 = np.load(f'{OUTDIR}/nb05_jacobians.npy')
with open(f'{OUTDIR}/nb05_jacobian_kd_names.pkl', 'rb') as f:
    kd5 = pickle.load(f)
dy5 = pd.read_parquet(f'{OUTDIR}/nb05_delta_y.parquet')

jac6 = np.load(f'{OUTDIR}/nb06_jacobians.npy')
with open(f'{OUTDIR}/nb06_jacobian_kd_names.pkl', 'rb') as f:
    kd6 = pickle.load(f)
dy6 = pd.read_parquet(f'{OUTDIR}/nb06_delta_y.parquet')

def pairwise_sims(jac, kd_names, dy_df):
    n = len(kd_names)
    jac_flat = jac.reshape(n, -1).astype(np.float32)
    jac_sim  = cosine_similarity(jac_flat)
    dy_vals  = dy_df.reindex(kd_names).values.astype(np.float32)
    dy_sim   = cosine_similarity(dy_vals)
    idx = np.triu_indices(n, k=1)
    return jac_sim[idx], dy_sim[idx]

jac_sim5, dy_sim5 = pairwise_sims(jac5, kd5, dy5)
jac_sim6, dy_sim6 = pairwise_sims(jac6, kd6, dy6)

pr5, _ = pearsonr(jac_sim5, dy_sim5)
sr5, sp5 = spearmanr(jac_sim5, dy_sim5)
pr6, _ = pearsonr(jac_sim6, dy_sim6)
sr6, sp6 = spearmanr(jac_sim6, dy_sim6)
print(f"Fewshot  Pearson={pr5:.4f} Spearman={sr5:.4f}")
print(f"Zeroshot Pearson={pr6:.4f} Spearman={sr6:.4f}")

N_BINS = 10

def decile_means(dy_sims, jac_sims, n_bins=N_BINS):
    rank_jac  = np.argsort(np.argsort(jac_sims)) / len(jac_sims)
    bin_edges = np.percentile(dy_sims, np.linspace(0, 100, n_bins + 1))
    bin_edges[-1] += 1e-9
    labels = np.clip(np.digitize(dy_sims, bin_edges) - 1, 0, n_bins - 1)
    centres, means, errors = [], [], []
    for b in range(n_bins):
        mask = labels == b
        vals = rank_jac[mask]
        centres.append(dy_sims[mask].mean())
        means.append(vals.mean())
        errors.append(sem(vals))
    return np.array(centres), np.array(means), np.array(errors)

c5, m5, e5 = decile_means(dy_sim5, jac_sim5)
c6, m6, e6 = decile_means(dy_sim6, jac_sim6)

COL_FS = '#2563EB'
COL_ZS = '#DC2626'

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle('Routing Coherence: Pearson vs Spearman  —  Fewshot vs Zeroshot',
             fontsize=13, fontweight='bold', y=1.01)

# Panel A: bar chart
ax = axes[0]
x = np.array([0, 1])
w = 0.3
b1 = ax.bar(x - w/2, [pr5, sr5], w, color=COL_FS, alpha=0.85, label='Fewshot/HepG2 (NB05)')
b2 = ax.bar(x + w/2, [pr6, sr6], w, color=COL_ZS, alpha=0.85, label='Zeroshot/HepG2 (NB06)')
for bar, val, col in zip(list(b1)+list(b2), [pr5, sr5, pr6, sr6],
                         [COL_FS, COL_FS, COL_ZS, COL_ZS]):
    offset = 0.005 if val >= 0 else -0.005
    va = 'bottom' if val >= 0 else 'top'
    ax.text(bar.get_x() + bar.get_width()/2, val + offset, f'{val:.3f}',
            ha='center', va=va, fontsize=10, fontweight='bold', color=col)
ax.annotate('p = 4e-50', xy=(x[1] - w/2, sr5), xytext=(x[1] + 0.05, sr5 + 0.03),
            fontsize=8.5, color=COL_FS, arrowprops=dict(arrowstyle='->', color=COL_FS, lw=0.8))
ax.annotate('p = 0.66 (NS)', xy=(x[1] + w/2, sr6), xytext=(x[1] + 0.3, sr6 - 0.03),
            fontsize=8.5, color=COL_ZS, arrowprops=dict(arrowstyle='->', color=COL_ZS, lw=0.8))
ax.axhline(0, color='black', lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(['Pearson r', 'Spearman r'], fontsize=12)
ax.set_ylabel('Correlation (Jacobian sim ~ ΔY sim)', fontsize=10)
ax.set_ylim(-0.05, 0.22)
ax.legend(fontsize=9, loc='upper left')
ax.grid(axis='y', alpha=0.3, lw=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_title('A   Summary statistics', fontsize=11, loc='left', fontweight='bold')

# Panel B: decile line plot
ax2 = axes[1]
ax2.errorbar(c5, m5, yerr=e5, color=COL_FS, marker='o', ms=6, lw=2, capsize=3,
             label=f'Fewshot   Spearman r = {sr5:.3f} (p=4e-50)', zorder=3)
ax2.errorbar(c6, m6, yerr=e6, color=COL_ZS, marker='s', ms=6, lw=2, capsize=3,
             label=f'Zeroshot  Spearman r = {sr6:.3f} (NS)', zorder=3)
ax2.axhline(0.5, color='gray', lw=1.2, linestyle='--', alpha=0.6, label='Flat (no correlation)')
ax2.set_xlabel('ΔY cosine similarity (binned into deciles)', fontsize=10)
ax2.set_ylabel('Mean rank of Jacobian similarity\n(normalised 0-1, mean +/- SE per bin)', fontsize=10)
ax2.set_title('B   Decile-binned mean Jacobian rank', fontsize=11, loc='left', fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3, lw=0.5)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
outpath = f'{OUTDIR}/nb07_pearson_vs_spearman.png'
fig.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"Saved: {outpath}")
