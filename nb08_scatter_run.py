import numpy as np
import pandas as pd
import pickle
import torch
from scipy.stats import pearsonr, spearmanr, mannwhitneyu
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
clusters5 = pd.read_csv(f'{OUTDIR}/nb05_clusters.csv', index_col=0)

jac6 = np.load(f'{OUTDIR}/nb06_jacobians.npy')
with open(f'{OUTDIR}/nb06_jacobian_kd_names.pkl', 'rb') as f: kd6 = pickle.load(f)
dy6 = pd.read_parquet(f'{OUTDIR}/nb06_delta_y.parquet')
clusters6 = pd.read_csv(f'{OUTDIR}/nb06_clusters.csv', index_col=0)

jac_oh_fs = jac5 @ W_fs
jac_oh_zs = jac6 @ W_zs

def build_pairs(jac_oh, kd_names, dy_df, clusters_df):
    n = len(kd_names)
    idx = np.triu_indices(n, k=1)
    ii, jj = idx
    jac_sim = cosine_similarity(jac_oh.reshape(n, -1).astype(np.float32))[idx]
    dy_sim  = cosine_similarity(dy_df.reindex(kd_names).values.astype(np.float32))[idx]
    clust = clusters_df['cluster'].reindex(kd_names)
    same  = (clust.values[ii] == clust.values[jj]).astype(bool)
    return jac_sim, dy_sim, same

js_fs, dy_fs, same_fs = build_pairs(jac_oh_fs, kd5, dy5, clusters5)
js_zs, dy_zs, same_zs = build_pairs(jac_oh_zs, kd6, dy6, clusters6)

def get_stats(jac_sim, dy_sim, same):
    pr, _ = pearsonr(jac_sim, dy_sim)
    sr, _ = spearmanr(jac_sim, dy_sim)
    s_vals = jac_sim[same]; d_vals = jac_sim[~same]
    _, wp = mannwhitneyu(s_vals, d_vals, alternative='greater')
    ratio = s_vals.mean() / d_vals.mean()
    return pr, sr, s_vals, d_vals, wp, ratio

pr_fs, sr_fs, s_fs, d_fs, wp_fs, rat_fs = get_stats(js_fs, dy_fs, same_fs)
pr_zs, sr_zs, s_zs, d_zs, wp_zs, rat_zs = get_stats(js_zs, dy_zs, same_zs)
print(f'Fewshot:  Pearson={pr_fs:.3f}, Spearman={sr_fs:.3f}, ratio={rat_fs:.2f}x (p={wp_fs:.2e})')
print(f'Zeroshot: Pearson={pr_zs:.3f}, Spearman={sr_zs:.3f}, ratio={rat_zs:.2f}x (p={wp_zs:.2e})')

COL_SAME = '#DC2626'
COL_DIFF = '#94A3B8'
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle('One-hot Jacobian: Routing vs Output Similarity', fontsize=14, fontweight='bold')

def scatter_panel(ax, dy_sim, jac_sim, same, pr, sr, title, fs_ref_r=None):
    rng = np.random.default_rng(42)
    d_idx = np.where(~same)[0]
    s_idx = np.where(same)[0]
    d_sub = rng.choice(d_idx, size=min(5000, len(d_idx)), replace=False)
    ax.scatter(dy_sim[d_sub], jac_sim[d_sub], c=COL_DIFF, alpha=0.15, s=4,
               rasterized=True, label=f'Diff cluster (n={len(d_idx):,})')
    ax.scatter(dy_sim[s_idx], jac_sim[s_idx], c=COL_SAME, alpha=0.4, s=6,
               rasterized=True, label=f'Same cluster (n={len(s_idx):,})')
    m, b = np.polyfit(dy_sim, jac_sim, 1)
    xs = np.linspace(dy_sim.min(), dy_sim.max(), 200)
    ax.plot(xs, m*xs + b, color='#1D4ED8', lw=2)
    lines_txt = [f'Pearson r={pr:.3f}', f'Spearman r={sr:.3f}']
    if fs_ref_r is not None: lines_txt.append(f'Fewshot ref r={fs_ref_r:.3f}')
    ax.text(0.04, 0.96, chr(10).join(lines_txt), transform=ax.transAxes, fontsize=9,
            va='top', color='#1D4ED8',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='#1D4ED8'))
    ax.set_xlabel('DeltaY cosine similarity', fontsize=10)
    ax.set_ylabel('d(Y)/d(x_onehot) cosine similarity', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right', markerscale=2)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

def violin_panel(ax, s_vals, d_vals, wp, ratio, fs_ratio, title):
    parts = ax.violinplot([s_vals, d_vals], positions=[0, 1], showmedians=True, showextrema=False)
    for pc, col in zip(parts['bodies'], [COL_SAME, COL_DIFF]):
        pc.set_facecolor(col); pc.set_alpha(0.6)
    parts['cmedians'].set_color('black'); parts['cmedians'].set_linewidth(2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f'Same cluster (n={len(s_vals):,})', f'Diff cluster (n={len(d_vals):,})'], fontsize=10)
    ax.set_ylabel('d(Y)/d(x_onehot) cosine similarity', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    note = f'MWU p={wp:.2e}  |  ratio={ratio:.2f}x  (fewshot: {fs_ratio:.2f}x)'
    ax.text(0.5, 0.97, note, transform=ax.transAxes, ha='center', va='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9, edgecolor='gray'))
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

scatter_panel(axes[0,0], dy_fs, js_fs, same_fs, pr_fs, sr_fs, 'Fewshot -- Routing vs Output similarity')
violin_panel(axes[0,1], s_fs, d_fs, wp_fs, rat_fs, rat_fs, 'Fewshot -- Same vs Diff cluster routing')
scatter_panel(axes[1,0], dy_zs, js_zs, same_zs, pr_zs, sr_zs, 'Zeroshot -- Routing vs Output similarity', fs_ref_r=pr_fs)
violin_panel(axes[1,1], s_zs, d_zs, wp_zs, rat_zs, rat_fs, 'Zeroshot -- Same vs Diff cluster routing')

plt.tight_layout()
outpath = f'{OUTDIR}/nb08_onehot_routing_vs_output.png'
fig.savefig(outpath, dpi=150, bbox_inches='tight')
print(f'Saved: {outpath}')