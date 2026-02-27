import numpy as np
import pandas as pd
import pickle
import torch
from scipy.stats import pearsonr, spearmanr, sem
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTDIR  = '/mnt/polished-lake/home/mbeheleramass/jacobian_analysis'
CKPT_FS = '/mnt/polished-lake/artifacts/fellows-shared/life-sciences/state/models/ST-HVG-Replogle/fewshot/hepg2/checkpoints/best.ckpt'
CKPT_ZS = '/mnt/polished-lake/artifacts/fellows-shared/life-sciences/state/models/ST-HVG-Replogle/zeroshot/hepg2/checkpoints/best.ckpt'

def get_W(path):
    sd = torch.load(path, map_location='cpu', weights_only=False)
    return sd.get('state_dict', sd)['pert_encoder.0.weight'].numpy()

W_fs = get_W(CKPT_FS)
W_zs = get_W(CKPT_ZS)

jac5 = np.load(f'{OUTDIR}/nb05_jacobians.npy')
with open(f'{OUTDIR}/nb05_jacobian_kd_names.pkl', 'rb') as f: kd5 = pickle.load(f)
dy5 = pd.read_parquet(f'{OUTDIR}/nb05_delta_y.parquet')

jac6 = np.load(f'{OUTDIR}/nb06_jacobians.npy')
with open(f'{OUTDIR}/nb06_jacobian_kd_names.pkl', 'rb') as f: kd6 = pickle.load(f)
dy6 = pd.read_parquet(f'{OUTDIR}/nb06_delta_y.parquet')

def get_pairs(jac, kd_names, dy_df):
    n = len(kd_names)
    idx = np.triu_indices(n, k=1)
    jac_sim = cosine_similarity(jac.reshape(n, -1).astype(np.float32))[idx]
    dy_sim  = cosine_similarity(dy_df.reindex(kd_names).values.astype(np.float32))[idx]
    return jac_sim, dy_sim

# one-hot pairs
jac_oh_fs = jac5 @ W_fs
jac_oh_zs = jac6 @ W_zs
js_oh_fs, dy_fs = get_pairs(jac_oh_fs, kd5, dy5)
js_oh_zs, dy_zs = get_pairs(jac_oh_zs, kd6, dy6)

# pert-repr pairs
js_pr_fs, _ = get_pairs(jac5, kd5, dy5)
js_pr_zs, _ = get_pairs(jac6, kd6, dy6)

def rs(j, d): return pearsonr(j,d)[0], spearmanr(j,d)[0]

pr_oh_fs, sr_oh_fs = rs(js_oh_fs, dy_fs)
pr_oh_zs, sr_oh_zs = rs(js_oh_zs, dy_zs)
pr_pr_fs, sr_pr_fs = rs(js_pr_fs, dy_fs)
pr_pr_zs, sr_pr_zs = rs(js_pr_zs, dy_zs)

print('           Pearson   Spearman')
print(f'FS one-hot  {pr_oh_fs:.3f}    {sr_oh_fs:.3f}')
print(f'ZS one-hot  {pr_oh_zs:.3f}    {sr_oh_zs:.3f}')
print(f'FS pert-rep {pr_pr_fs:.3f}    {sr_pr_fs:.3f}')
print(f'ZS pert-rep {pr_pr_zs:.3f}    {sr_pr_zs:.3f}')

# decile means for Panel B
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

c_oh_fs,m_oh_fs,e_oh_fs = decile_means(dy_fs, js_oh_fs)
c_oh_zs,m_oh_zs,e_oh_zs = decile_means(dy_zs, js_oh_zs)
c_pr_fs,m_pr_fs,e_pr_fs = decile_means(dy_fs, js_pr_fs)
c_pr_zs,m_pr_zs,e_pr_zs = decile_means(dy_zs, js_pr_zs)

# colours: fewshot=blue family, zeroshot=red family
# one-hot=solid, pert-repr=hatched/lighter
C_FS_OH = '#2563EB'
C_ZS_OH = '#DC2626'
C_FS_PR = '#93C5FD'
C_ZS_PR = '#FCA5A5'

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('One-hot vs Pert-repr Jacobian: Routing Coherence (Fewshot vs Zeroshot)',
             fontsize=13, fontweight='bold', y=1.01)

# Panel A: grouped bar chart — 4 bars per metric
ax = axes[0]
metrics = ['Pearson r', 'Spearman r']
vals = {
    'FS one-hot':  [pr_oh_fs, sr_oh_fs],
    'ZS one-hot':  [pr_oh_zs, sr_oh_zs],
    'FS pert-repr': [pr_pr_fs, sr_pr_fs],
    'ZS pert-repr': [pr_pr_zs, sr_pr_zs],
}
colors  = [C_FS_OH, C_ZS_OH, C_FS_PR, C_ZS_PR]
hatches = ['', '', '///', '///']
x = np.array([0, 1])
w = 0.18
offsets = [-1.5*w, -0.5*w, 0.5*w, 1.5*w]

for i, (label, vals_pair) in enumerate(vals.items()):
    bars = ax.bar(x + offsets[i], vals_pair, w,
                  color=colors[i], hatch=hatches[i],
                  edgecolor='white', linewidth=0.5,
                  alpha=0.9, label=label)
    for bar, v in zip(bars, vals_pair):
        offset = 0.004 if v >= 0 else -0.006
        va = 'bottom' if v >= 0 else 'top'
        ax.text(bar.get_x() + bar.get_width()/2, v + offset,
                f'{v:.3f}', ha='center', va=va,
                fontsize=7.5, fontweight='bold', color=colors[i])

ax.axhline(0, color='black', lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=12)
ax.set_ylabel('Correlation (Jacobian sim ~ ΔY sim)', fontsize=10)
ax.set_ylim(-0.05, 0.27)
ax.legend(fontsize=8.5, loc='upper left', ncol=2)
ax.grid(axis='y', alpha=0.3, lw=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_title('A   Summary statistics', fontsize=11, loc='left', fontweight='bold')

# Panel B: decile lines — all 4 conditions
ax2 = axes[1]
ax2.errorbar(c_oh_fs, m_oh_fs, yerr=e_oh_fs, color=C_FS_OH, marker='o', ms=5, lw=2,
             capsize=3, label=f'FS one-hot   r={sr_oh_fs:.3f}', zorder=4)
ax2.errorbar(c_oh_zs, m_oh_zs, yerr=e_oh_zs, color=C_ZS_OH, marker='o', ms=5, lw=2,
             capsize=3, label=f'ZS one-hot   r={sr_oh_zs:.3f}', zorder=4)
ax2.errorbar(c_pr_fs, m_pr_fs, yerr=e_pr_fs, color=C_FS_PR, marker='s', ms=5, lw=1.5,
             capsize=3, linestyle='--', label=f'FS pert-repr r={sr_pr_fs:.3f}', zorder=3)
ax2.errorbar(c_pr_zs, m_pr_zs, yerr=e_pr_zs, color=C_ZS_PR, marker='s', ms=5, lw=1.5,
             capsize=3, linestyle='--', label=f'ZS pert-repr r={sr_pr_zs:.3f}', zorder=3)
ax2.axhline(0.5, color='gray', lw=1.2, linestyle=':', alpha=0.6, label='Flat (no correlation)')
ax2.set_xlabel('ΔY cosine similarity (binned into deciles)', fontsize=10)
ax2.set_ylabel('Mean rank of Jacobian similarity (normalised 0-1)', fontsize=10)
ax2.set_title('B   Decile-binned mean Jacobian rank', fontsize=11, loc='left', fontweight='bold')
ax2.legend(fontsize=8.5)
ax2.grid(alpha=0.3, lw=0.5)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
outpath = f'{OUTDIR}/nb08_onehot_jacobian.png'
fig.savefig(outpath, dpi=150, bbox_inches='tight')
print(f'Saved: {outpath}')