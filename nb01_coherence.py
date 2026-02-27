import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import re

WORKDIR = '/mnt/polished-lake/home/mbeheleramass'
OUTDIR  = '/mnt/polished-lake/home/mbeheleramass/jacobian_analysis'

hepg2 = pd.read_parquet(f"{WORKDIR}/pert_mean_features_hepg2.parquet")
jurkat = pd.read_parquet(f"{WORKDIR}/pert_mean_features_jurkat.parquet")
sim_h = np.load(f"{OUTDIR}/nb01_sim_h.npy")
sim_j = np.load(f"{OUTDIR}/nb01_sim_j.npy")

genes = list(hepg2.index)
gene_to_idx = {g: i for i, g in enumerate(genes)}
n_genes = len(genes)
rng = np.random.default_rng(42)
N_BOOT = 5000

pathway_genes = {
    'Ribosome\n(cytoplasmic)': [g for g in genes if re.match(r'^RP[SL]\d', g)],
    'Mitoribosome':              [g for g in genes if re.match(r'^MRP[SL]\d', g)],
    'OXPHOS / mito':             [g for g in genes if re.match(r'^NDUF', g)] +
                                  [g for g in genes if g in ['SDHC']],
    'ER / Secretory':            [g for g in genes if re.match(r'^TRAPPC', g)] +
                                  [g for g in genes if re.match(r'^SRP', g)] +
                                  [g for g in genes if re.match(r'^SEC6', g)] + ['SRPRB'],
    'Proteasome':                [g for g in genes if re.match(r'^PSM', g)],
    'Splicing':                  [g for g in genes if re.match(r'^PRPF', g)] +
                                  [g for g in genes if g in ['DHX15','DHX16','DDX46','HNRNPU']],
}

def within_pathway_sim(sim_matrix, idx_list):
    if len(idx_list) < 2:
        return np.nan
    sub = sim_matrix[np.ix_(idx_list, idx_list)]
    upper = sub[np.triu_indices(len(idx_list), k=1)]
    return upper.mean()

def bootstrap_null(sim_matrix, n_members, n_boot=N_BOOT, rng=rng):
    null_sims = []
    all_idx = np.arange(n_genes)
    for _ in range(n_boot):
        rand_idx = rng.choice(all_idx, size=n_members, replace=False)
        null_sims.append(within_pathway_sim(sim_matrix, rand_idx))
    return np.array(null_sims)

results = []
for pw, members in pathway_genes.items():
    valid = [g for g in members if g in gene_to_idx]
    if len(valid) < 2:
        continue
    idx_list = [gene_to_idx[g] for g in valid]

    obs_h = within_pathway_sim(sim_h, idx_list)
    obs_j = within_pathway_sim(sim_j, idx_list)

    null_h = bootstrap_null(sim_h, len(idx_list))
    null_j = bootstrap_null(sim_j, len(idx_list))

    p_h = (null_h >= obs_h).mean()
    p_j = (null_j >= obs_j).mean()

    z_h = (obs_h - null_h.mean()) / null_h.std()
    z_j = (obs_j - null_j.mean()) / null_j.std()

    results.append({
        'pathway': pw, 'n': len(valid),
        'obs_hepg2': obs_h, 'null_mean_hepg2': null_h.mean(), 'null_std_hepg2': null_h.std(),
        'z_hepg2': z_h, 'p_hepg2': p_h,
        'obs_jurkat': obs_j, 'null_mean_jurkat': null_j.mean(), 'null_std_jurkat': null_j.std(),
        'z_jurkat': z_j, 'p_jurkat': p_j,
    })
    print(f"{pw:30s} n={len(valid):2d}  HepG2: obs={obs_h:.4f} null={null_h.mean():.4f} z={z_h:.2f} p={p_h:.4f}  |  Jurkat: obs={obs_j:.4f} null={null_j.mean():.4f} z={z_j:.2f} p={p_j:.4f}")

df = pd.DataFrame(results)

# ---- Plot ----
pw_labels = df['pathway'].tolist()
x = np.arange(len(pw_labels))
width = 0.35

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

specs = [
    (axes[0], 'HepG2',  'obs_hepg2',  'null_mean_hepg2',  'null_std_hepg2',  'z_hepg2',  'p_hepg2'),
    (axes[1], 'Jurkat', 'obs_jurkat', 'null_mean_jurkat', 'null_std_jurkat', 'z_jurkat', 'p_jurkat'),
]

for ax, cell, col_obs, col_null, col_std, col_z, col_p in specs:
    obs_vals  = df[col_obs].values
    null_vals = df[col_null].values
    null_stds = df[col_std].values
    z_vals    = df[col_z].values
    p_vals    = df[col_p].values

    ax.bar(x - width/2, obs_vals,  width, label='Within-pathway', color='steelblue', alpha=0.85)
    ax.bar(x + width/2, null_vals, width, label='Random null (mean)', color='#BBBBBB', alpha=0.85,
           yerr=null_stds*1.96, capsize=4, error_kw={'linewidth': 1.5})

    for i, (z, p) in enumerate(zip(z_vals, p_vals)):
        star = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
        ypos = max(obs_vals[i], null_vals[i] + null_stds[i]*1.96) + 0.005
        ax.text(x[i] - width/2, ypos + 0.002, star, ha='center', fontsize=9, color='steelblue', fontweight='bold')
        ax.text(x[i], ypos + 0.013, f"z={z:.1f}", ha='center', fontsize=7.5, color='#444444')

    ax.set_xticks(x)
    ax.set_xticklabels(pw_labels, fontsize=9)
    ax.set_ylabel("Mean pairwise cosine similarity", fontsize=11)
    ax.set_title(f"Pathway coherence - {cell}\nSAE layer-4 feature space", fontsize=12)
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)

plt.suptitle("Within-pathway vs random pairwise cosine similarity\n(* p<0.05, ** p<0.01, *** p<0.001, bootstrap n=5000)",
             fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/nb01_pathway_coherence.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved nb01_pathway_coherence.png")

print("\nSummary:")
print(df[['pathway','n','obs_hepg2','z_hepg2','p_hepg2','obs_jurkat','z_jurkat','p_jurkat']].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
