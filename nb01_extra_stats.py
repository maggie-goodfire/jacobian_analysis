"""Extra stats: cross-sim distribution + Wilcoxon within vs across pathway."""
import numpy as np
import pandas as pd
from scipy import stats
import re

WORKDIR = '/mnt/polished-lake/home/mbeheleramass'
OUTDIR  = '/mnt/polished-lake/home/mbeheleramass/jacobian_analysis'

hepg2  = pd.read_parquet(f"{WORKDIR}/pert_mean_features_hepg2.parquet")
jurkat = pd.read_parquet(f"{WORKDIR}/pert_mean_features_jurkat.parquet")
sim_h  = np.load(f"{OUTDIR}/nb01_sim_h.npy")
sim_j  = np.load(f"{OUTDIR}/nb01_sim_j.npy")
genes  = list(hepg2.index)

# L2-normalize
h_norm = hepg2.values / (np.linalg.norm(hepg2.values, axis=1, keepdims=True) + 1e-12)
j_norm = jurkat.values / (np.linalg.norm(jurkat.values, axis=1, keepdims=True) + 1e-12)
cross_sim = (h_norm * j_norm).sum(axis=1)

gene_to_idx = {g: i for i, g in enumerate(genes)}

pathway_genes = {
    'Ribosome (cytoplasmic)': [g for g in genes if re.match(r'^RP[SL]\d', g)],
    'Mitoribosome':            [g for g in genes if re.match(r'^MRP[SL]\d', g)],
    'OXPHOS / mito':           [g for g in genes if re.match(r'^NDUF', g)] + ['SDHC'],
    'ER / Secretory':          [g for g in genes if re.match(r'^TRAPPC', g)] +
                                [g for g in genes if re.match(r'^SRP', g)] + ['SRPRB'],
}

print("Cross-cell-line cosine similarity by pathway:")
print(f"{'Pathway':30s}  {'n':>4}  {'mean':>8}  {'median':>8}  {'std':>8}  {'vs_all p':>10}")
all_other_idx = list(range(len(genes)))
for pw, members in pathway_genes.items():
    valid = [g for g in members if g in gene_to_idx]
    if not valid:
        continue
    idx = [gene_to_idx[g] for g in valid]
    pw_sims = cross_sim[idx]
    other_sims = np.delete(cross_sim, idx)
    stat, p = stats.mannwhitneyu(pw_sims, other_sims, alternative='two-sided')
    print(f"{pw:30s}  {len(valid):>4}  {pw_sims.mean():>8.4f}  {np.median(pw_sims):>8.4f}  {pw_sims.std():>8.4f}  {p:>10.4g}")

# Also: Wilcoxon within-pathway vs across-pathway cosine sim for HepG2
print("\n\nWilcoxon within-pathway vs across-pathway cosine similarity (HepG2):")
print(f"{'Pathway':30s}  {'n_within':>8}  {'n_across':>8}  {'within_mean':>12}  {'across_mean':>12}  {'p_value':>10}")
for pw, members in pathway_genes.items():
    valid = [g for g in members if g in gene_to_idx]
    if len(valid) < 2:
        continue
    idx = [gene_to_idx[g] for g in valid]
    idx_arr = np.array(idx)
    # within-pathway pairs
    sub = sim_h[np.ix_(idx_arr, idx_arr)]
    within = sub[np.triu_indices(len(idx), k=1)]
    # across-pathway pairs (same gene vs all non-pathway genes)
    all_idx = np.arange(len(genes))
    other_idx = np.array([i for i in all_idx if i not in set(idx)])
    cross_pairs = sim_h[np.ix_(idx_arr, other_idx)].flatten()
    stat, p = stats.mannwhitneyu(within, cross_pairs, alternative='greater')
    print(f"{pw:30s}  {len(within):>8}  {len(cross_pairs):>8}  {within.mean():>12.4f}  {cross_pairs.mean():>12.4f}  {p:>10.4g}")
