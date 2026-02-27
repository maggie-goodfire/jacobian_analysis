
import numpy as np
import pandas as pd
import pickle
import torch
from scipy.stats import pearsonr, spearmanr, sem
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTDIR    = '/mnt/polished-lake/home/mbeheleramass/jacobian_analysis'
CKPT_FS   = '/mnt/polished-lake/artifacts/fellows-shared/life-sciences/state/models/ST-HVG-Replogle/fewshot/hepg2/checkpoints/best.ckpt'
CKPT_ZS   = '/mnt/polished-lake/artifacts/fellows-shared/life-sciences/state/models/ST-HVG-Replogle/zeroshot/hepg2/checkpoints/best.ckpt'

# ── Extract pert_encoder W from both checkpoints ──────────────────────────────
print("[1] Extracting pert_encoder weights...")
def get_W(ckpt_path):
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = sd.get('state_dict', sd)
    return state['pert_encoder.0.weight'].numpy()   # [328, 2024]

W_fs = get_W(CKPT_FS)
W_zs = get_W(CKPT_ZS)
print(f"  W_fewshot:  {W_fs.shape}")
print(f"  W_zeroshot: {W_zs.shape}")

# ── Load pert_repr Jacobians ──────────────────────────────────────────────────
print("[2] Loading NB05/NB06 pert_repr Jacobians...")
jac5 = np.load(f'{OUTDIR}/nb05_jacobians.npy')   # [225, 2000, 328]
with open(f'{OUTDIR}/nb05_jacobian_kd_names.pkl', 'rb') as f:
    kd5 = pickle.load(f)
dy5 = pd.read_parquet(f'{OUTDIR}/nb05_delta_y.parquet')

jac6 = np.load(f'{OUTDIR}/nb06_jacobians.npy')   # [225, 2000, 328]
with open(f'{OUTDIR}/nb06_jacobian_kd_names.pkl', 'rb') as f:
    kd6 = pickle.load(f)
dy6 = pd.read_parquet(f'{OUTDIR}/nb06_delta_y.parquet')

# ── Chain rule → one-hot Jacobians ───────────────────────────────────────────
print("[3] Chain rule: pert_repr Jacobian @ W -> one-hot Jacobian...")
jac_oh_fs = jac5 @ W_fs   # [225, 2000, 2024]
jac_oh_zs = jac6 @ W_zs   # [225, 2000, 2024]
print(f"  fewshot  one-hot Jacobian: {jac_oh_fs.shape}")
print(f"  zeroshot one-hot Jacobian: {jac_oh_zs.shape}")

# ── Pairwise cosine similarities ─────────────────────────────────────────────
print("[4] Pairwise cosine similarities...")
n = 225
idx = np.triu_indices(n, k=1)

def get_pairs(jac, kd_names, dy_df):
    jac_sim = cosine_similarity(jac.reshape(n, -1).astype(np.float32))[idx]
    dy_sim  = cosine_similarity(dy_df.reindex(kd_names).values.astype(np.float32))[idx]
    return jac_sim, dy_sim

jac_pairs_fs, dy_pairs_fs = get_pairs(jac_oh_fs, kd5, dy5)
jac_pairs_zs, dy_pairs_zs = get_pairs(jac_oh_zs, kd6, dy6)

pr_fs, _     = pearsonr(jac_pairs_fs, dy_pairs_fs)
sr_fs, sp_fs = spearmanr(jac_pairs_fs, dy_pairs_fs)
pr_zs, _     = pearsonr(jac_pairs_zs, dy_pairs_zs)
sr_zs, sp_zs = spearmanr(jac_pairs_zs, dy_pairs_zs)
print(f"  Fewshot  one-hot: Pearson={pr_fs:.4f}, Spearman={sr_fs:.4f} (p={sp_fs:.2e})")
print(f"  Zeroshot one-hot: Pearson={pr_zs:.4f}, Spearman={sr_zs:.4f} (p={sp_zs:.2e})")

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

c_fs, m_fs, e_fs = decile_means(dy_pairs_fs, jac_pairs_fs)
c_zs, m_zs, e_zs = decile_means(dy_pairs_zs, jac_pairs_zs)

# ── Figure ────────────────────────────────────────────────────────────────────
COL_FS = '#2563EB'
COL_ZS = '#DC2626'

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle('One-hot Jacobian: Fewshot vs Zeroshot  —  Routing Coherence',
             fontsize=13, fontweight='bold', y=1.01)

# Panel A: bar chart
ax = axes[0]
x = np.array([0, 1])
w = 0.3
b1 = ax.bar(x - w/2, [pr_fs, sr_fs], w, color=COL_FS, alpha=0.85, label='Fewshot/HepG2')
b2 = ax.bar(x + w/2, [pr_zs, sr_zs], w, color=COL_ZS, alpha=0.85, label='Zeroshot/HepG2')
for bar, val, col in zip(list(b1)+list(b2),
                         [pr_fs, sr_fs, pr_zs, sr_zs],
                         [COL_FS, COL_FS, COL_ZS, COL_ZS]):
    offset = 0.005 if val >= 0 else -0.007
    va = 'bottom' if val >= 0 else 'top'
    ax.text(bar.get_x() + bar.get_width()/2, val + offset,
            f'{val:.3f}', ha='center', va=va, fontsize=10, fontweight='bold', color=col)

# p-value callouts on Spearman bars
ax.annotate(f'p={sp_fs:.1e}', xy=(x[1] - w/2, sr_fs), xytext=(x[1] + 0.05, sr_fs + 0.03),
            fontsize=8.5, color=COL_FS, arrowprops=dict(arrowstyle='->', color=COL_FS, lw=0.8))
p_label = f'p={sp_zs:.2f} (NS)' if sp_zs > 0.05 else f'p={sp_zs:.1e}'
ax.annotate(p_label, xy=(x[1] + w/2, sr_zs), xytext=(x[1] + 0.3, sr_zs - 0.03),
            fontsize=8.5, color=COL_ZS, arrowprops=dict(arrowstyle='->', color=COL_ZS, lw=0.8))

ax.axhline(0, color='black', lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(['Pearson r', 'Spearman r'], fontsize=12)
ax.set_ylabel('Correlation (Jacobian sim ~ ΔY sim)', fontsize=10)
ax.set_ylim(-0.05, 0.28)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3, lw=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_title('A   Summary statistics', fontsize=11, loc='left', fontweight='bold')

# Panel B: decile line plot
ax2 = axes[1]
ax2.errorbar(c_fs, m_fs, yerr=e_fs, color=COL_FS, marker='o', ms=6, lw=2, capsize=3,
             label=f'Fewshot   Spearman r={sr_fs:.3f} (p={sp_fs:.1e})', zorder=3)
ax2.errorbar(c_zs, m_zs, yerr=e_zs, color=COL_ZS, marker='s', ms=6, lw=2, capsize=3,
             label=f'Zeroshot  Spearman r={sr_zs:.3f} (p={sp_zs:.2f})', zorder=3)
ax2.axhline(0.5, color='gray', lw=1.2, linestyle='--', alpha=0.6, label='Flat (no correlation)')
ax2.set_xlabel('ΔY cosine similarity (binned into deciles)', fontsize=10)
ax2.set_ylabel('Mean rank of Jacobian similarity (normalised 0-1, mean +/- SE)', fontsize=10)
ax2.set_title('B   Decile-binned mean Jacobian rank', fontsize=11, loc='left', fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3, lw=0.5)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
outpath = f'{OUTDIR}/nb08_onehot_fewshot_vs_zeroshot.png'
fig.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"Saved: {outpath}")
