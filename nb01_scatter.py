import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy import stats
import re

WORKDIR = '/mnt/polished-lake/home/mbeheleramass'
OUTDIR  = '/mnt/polished-lake/home/mbeheleramass/jacobian_analysis'

hepg2  = pd.read_parquet(f"{WORKDIR}/pert_mean_features_hepg2.parquet")
jurkat = pd.read_parquet(f"{WORKDIR}/pert_mean_features_jurkat.parquet")
feat_scores = pd.read_parquet(f"{WORKDIR}/feat_scores.parquet")

genes = list(hepg2.index)

# ---- Compute per-perturbation cross-cell-line cosine similarity ----
# Normalize each row vector then take dot product
h_vals = hepg2.values.astype(np.float64)
j_vals = jurkat.values.astype(np.float64)

# L2-normalize each perturbation vector
h_norm = h_vals / (np.linalg.norm(h_vals, axis=1, keepdims=True) + 1e-12)
j_norm = j_vals / (np.linalg.norm(j_vals, axis=1, keepdims=True) + 1e-12)

cross_sim = (h_norm * j_norm).sum(axis=1)  # dot product of unit vectors = cosine sim
print("Cross-cell-line cosine similarity stats:")
print(f"  mean={cross_sim.mean():.4f}  std={cross_sim.std():.4f}  min={cross_sim.min():.4f}  max={cross_sim.max():.4f}")

# ---- Compute per-perturbation effective Cohen's d ----
# Strategy: for each perturbation, weight features by their activation in HepG2 or Jurkat
# and compute the mean |Cohen's d| of the top-activated features.
# This gives a perturbation-level measure of how much it preferentially activates
# cell-line-differential features.

cohens_d = feat_scores['cohens_d'].values  # shape (2624,) — per feature
# Positive = HepG2 > Jurkat

# Per perturbation: compute weighted mean Cohen's d
# weight = activation fraction of each feature for that perturbation
# Use HepG2 activations as weights (most activated features for this pert)
h_total = h_vals.sum(axis=1, keepdims=True)
h_total = np.where(h_total == 0, 1, h_total)
h_weights = h_vals / h_total  # (1035, 2624) normalized activation profile

pert_cohens_d = (h_weights * cohens_d[np.newaxis, :]).sum(axis=1)  # (1035,)
print(f"\nPer-perturbation weighted Cohen's d stats:")
print(f"  mean={pert_cohens_d.mean():.4f}  std={pert_cohens_d.std():.4f}")
print(f"  min={pert_cohens_d.min():.4f}  max={pert_cohens_d.max():.4f}")

# ---- Correlation ----
r, p = stats.spearmanr(cross_sim, pert_cohens_d)
print(f"\nSpearman r(cross_sim, pert_cohens_d) = {r:.4f}  p = {p:.4g}")

r2, p2 = stats.pearsonr(cross_sim, pert_cohens_d)
print(f"Pearson  r(cross_sim, pert_cohens_d) = {r2:.4f}  p = {p2:.4g}")

# ---- Pathway labels for coloring ----
pathway_genes = {
    'Ribosome (cytoplasmic)': [g for g in genes if re.match(r'^RP[SL]\d', g)],
    'Mitoribosome':            [g for g in genes if re.match(r'^MRP[SL]\d', g)],
    'OXPHOS / mito':           [g for g in genes if re.match(r'^NDUF', g)] + ['SDHC'],
    'ER / Secretory':          [g for g in genes if re.match(r'^TRAPPC', g)] +
                                [g for g in genes if re.match(r'^SRP', g)] + ['SRPRB'],
    'TF / Oncogene':           ['MYC','MAX','BRCA1','BRCA2'],
}
colors_pw = {
    'Ribosome (cytoplasmic)': '#E41A1C',
    'Mitoribosome':            '#FF7F00',
    'OXPHOS / mito':           '#984EA3',
    'ER / Secretory':          '#4DAF4A',
    'TF / Oncogene':           '#377EB8',
    'Other':                   '#CCCCCC',
}
gene_to_pw = {g: 'Other' for g in genes}
for pw, members in pathway_genes.items():
    for g in members:
        if g in gene_to_pw and gene_to_pw[g] == 'Other':
            gene_to_pw[g] = pw
point_colors = [colors_pw[gene_to_pw[g]] for g in genes]
point_labels = [gene_to_pw[g] for g in genes]

# ---- Plot ----
fig, ax = plt.subplots(figsize=(9, 7))

# Background (Other)
mask_other = np.array([l == 'Other' for l in point_labels])
ax.scatter(pert_cohens_d[mask_other], cross_sim[mask_other],
           c='#CCCCCC', s=10, alpha=0.3, zorder=1, label='Other')

for pw in list(pathway_genes.keys()):
    mask = np.array([l == pw for l in point_labels])
    if mask.sum() == 0:
        continue
    ax.scatter(pert_cohens_d[mask], cross_sim[mask],
               c=colors_pw[pw], s=50, alpha=0.85, zorder=3,
               label=f"{pw} (n={mask.sum()})", edgecolors='white', linewidths=0.5)

# Identify outliers: low cross-sim or extreme Cohen's d
# Low cross-sim outliers (bottom 5%)
low_sim_thresh = np.percentile(cross_sim, 5)
high_cd_thresh = np.percentile(np.abs(pert_cohens_d), 95)
annotate_mask = (cross_sim < low_sim_thresh) | (np.abs(pert_cohens_d) > high_cd_thresh)
# Also annotate known interesting genes
always_annotate = ['MYC','MAX','TRAPPC3','TRAPPC5','SRP54','RPL10','NDUFB8','MRPL1']
for i, g in enumerate(genes):
    if annotate_mask[i] or g in always_annotate:
        ax.annotate(g, (pert_cohens_d[i], cross_sim[i]),
                    fontsize=7, fontweight='bold',
                    xytext=(4, 4), textcoords='offset points',
                    path_effects=[pe.withStroke(linewidth=2, foreground='white')])

# Regression line
z_fit = np.polyfit(pert_cohens_d, cross_sim, 1)
x_line = np.linspace(pert_cohens_d.min(), pert_cohens_d.max(), 200)
ax.plot(x_line, np.polyval(z_fit, x_line), 'k--', lw=1.5, alpha=0.6,
        label=f"OLS fit")

ax.axvline(0, color='gray', lw=0.8, ls=':', alpha=0.5)
ax.set_xlabel("Per-perturbation weighted Cohen's d (HepG2 > Jurkat when positive)", fontsize=11)
ax.set_ylabel("Cross-cell-line cosine similarity\n(HepG2 vs Jurkat SAE feature vectors)", fontsize=11)
ax.set_title(
    f"Cross-cell-line SAE feature similarity vs differential Cohen's d\n"
    f"Spearman r={r:.3f}, p={p:.3g}  |  n=1035 perturbations",
    fontsize=12)
ax.legend(loc='lower right', fontsize=8.5, framealpha=0.9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f"{OUTDIR}/nb01_cross_cellline_scatter.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved nb01_cross_cellline_scatter.png")

# Print top/bottom outliers
df_out = pd.DataFrame({
    'gene': genes,
    'cross_sim': cross_sim,
    'pert_cohens_d': pert_cohens_d,
    'pathway': point_labels,
})
print("\nLowest cross-cell-line similarity (most cell-line-divergent):")
print(df_out.nsmallest(15, 'cross_sim')[['gene','cross_sim','pert_cohens_d','pathway']].to_string(index=False))
print("\nHighest Cohen's d (most HepG2-biased feature activation):")
print(df_out.nlargest(10, 'pert_cohens_d')[['gene','cross_sim','pert_cohens_d','pathway']].to_string(index=False))
