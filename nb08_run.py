
import numpy as np
import pandas as pd
import pickle
import torch
from scipy.stats import pearsonr, spearmanr, sem
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTDIR = '/mnt/polished-lake/home/mbeheleramass/jacobian_analysis'
CKPT   = '/mnt/polished-lake/artifacts/fellows-shared/life-sciences/state/models/ST-HVG-Replogle/fewshot/hepg2/checkpoints/best.ckpt'

# ── Extract pert_encoder weight from checkpoint (no model instantiation needed) ──
print("[1] Extracting pert_encoder weight matrix from checkpoint...")
sd = torch.load(CKPT, map_location='cpu', weights_only=False)
state = sd.get('state_dict', sd)
# key is e.g. 'pert_encoder.0.weight'
W = state['pert_encoder.0.weight'].numpy()   # [328, 2024]
print(f"  W shape: {W.shape}")

# ── Load NB05 Jacobians [225, 2000, 328] ─────────────────────────────────────
print("[2] Loading NB05 Jacobians...")
jac5 = np.load(f'{OUTDIR}/nb05_jacobians.npy')           # [225, 2000, 328]
with open(f'{OUTDIR}/nb05_jacobian_kd_names.pkl', 'rb') as f:
    kd5 = pickle.load(f)
dy5 = pd.read_parquet(f'{OUTDIR}/nb05_delta_y.parquet')
print(f"  jac5: {jac5.shape}")

# ── Chain rule: ∂Y/∂x_onehot = (∂Y/∂pert_repr) @ W ─────────────────────────
print("[3] Chain rule multiply: [225,2000,328] @ [328,2024] -> [225,2000,2024]...")
jac_onehot = jac5 @ W    # [225, 2000, 2024]
print(f"  jac_onehot: {jac_onehot.shape}, {jac_onehot.nbytes/1e9:.2f} GB")
np.save(f'{OUTDIR}/nb08_jacobians_onehot.npy', jac_onehot)
print("  Saved nb08_jacobians_onehot.npy")

# ── Pairwise cosine similarities ─────────────────────────────────────────────
print("[4] Pairwise cosine similarities...")
n = len(kd5)
idx = np.triu_indices(n, k=1)

jac_flat    = jac_onehot.reshape(n, -1).astype(np.float32)
jac_sim_oh  = cosine_similarity(jac_flat)[idx]

dy_vals     = dy5.reindex(kd5).values.astype(np.float32)
dy_sim      = cosine_similarity(dy_vals)[idx]

# also pert_repr pairs for comparison
jac_flat_pr = jac5.reshape(n, -1).astype(np.float32)
jac_sim_pr  = cosine_similarity(jac_flat_pr)[idx]

pr_oh, _ = pearsonr(jac_sim_oh, dy_sim)
sr_oh, sp_oh = spearmanr(jac_sim_oh, dy_sim)
pr_pr, _ = pearsonr(jac_sim_pr, dy_sim)
sr_pr, sp_pr = spearmanr(jac_sim_pr, dy_sim)
print(f"  One-hot:   Pearson={pr_oh:.4f}, Spearman={sr_oh:.4f} (p={sp_oh:.2e})")
print(f"  Pert-repr: Pearson={pr_pr:.4f}, Spearman={sr_pr:.4f} (p={sp_pr:.2e})")

# ── Decile binning ────────────────────────────────────────────────────────────
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

c_oh, m_oh, e_oh = decile_means(dy_sim, jac_sim_oh)
c_pr, m_pr, e_pr = decile_means(dy_sim, jac_sim_pr)

# ── Figure ────────────────────────────────────────────────────────────────────
COL_OH = '#7C3AED'
COL_PR = '#2563EB'

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle('One-hot vs Pert-repr Jacobian  —  Routing Coherence',
             fontsize=13, fontweight='bold', y=1.01)

# Panel A: bar chart
ax = axes[0]
x = np.array([0, 1])
w = 0.3
b1 = ax.bar(x - w/2, [pr_oh, sr_oh], w, color=COL_OH, alpha=0.85, label='One-hot (NB08)')
b2 = ax.bar(x + w/2, [pr_pr, sr_pr], w, color=COL_PR, alpha=0.85, label='Pert-repr (NB05)')
for bar, val, col in zip(list(b1)+list(b2),
                         [pr_oh, sr_oh, pr_pr, sr_pr],
                         [COL_OH, COL_OH, COL_PR, COL_PR]):
    offset = 0.005 if val >= 0 else -0.007
    va = 'bottom' if val >= 0 else 'top'
    ax.text(bar.get_x() + bar.get_width()/2, val + offset,
            f'{val:.3f}', ha='center', va=va, fontsize=10, fontweight='bold', color=col)
ax.axhline(0, color='black', lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(['Pearson r', 'Spearman r'], fontsize=12)
ax.set_ylabel('Correlation (Jacobian sim ~ ΔY sim)', fontsize=10)
ax.set_ylim(-0.05, 0.22)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3, lw=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_title('A   Summary statistics', fontsize=11, loc='left', fontweight='bold')

# Panel B: decile line plot
ax2 = axes[1]
ax2.errorbar(c_oh, m_oh, yerr=e_oh, color=COL_OH, marker='o', ms=6, lw=2,
             capsize=3, label=f'One-hot   Spearman r={sr_oh:.3f} (p={sp_oh:.1e})', zorder=3)
ax2.errorbar(c_pr, m_pr, yerr=e_pr, color=COL_PR, marker='s', ms=6, lw=2,
             capsize=3, label=f'Pert-repr Spearman r={sr_pr:.3f} (p={sp_pr:.1e})', zorder=3)
ax2.axhline(0.5, color='gray', lw=1.2, linestyle='--', alpha=0.6, label='Flat (no correlation)')
ax2.set_xlabel('ΔY cosine similarity (binned into deciles)', fontsize=10)
ax2.set_ylabel('Mean rank of Jacobian similarity (normalised 0-1, mean +/- SE)', fontsize=10)
ax2.set_title('B   Decile-binned mean Jacobian rank', fontsize=11, loc='left', fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3, lw=0.5)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
outpath = f'{OUTDIR}/nb08_onehot_jacobian.png'
fig.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"Saved: {outpath}")
