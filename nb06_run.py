"""
NB06 — Zeroshot Model: Output Clustering + Perturbation Encoder Jacobian
Mirrors NB05 exactly but uses the zeroshot/hepg2 model.
Part 1: Compute ΔY for all KDs → cluster → UMAP → ARI comparison with NB05
Part 2: ∂Y/∂pert_repr Jacobian for representative KDs
Part 3: Routing similarity (zeroshot) vs fewshot reference (r=0.119)
Part 4: Comparison figures (side-by-side UMAP, ΔY magnitude scatter)
"""

import sys
sys.path.insert(0, '/mnt/polished-lake/home/mbeheleramass/state/src')

import torch
import pickle
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import wilcoxon, pearsonr, spearmanr
from sklearn.metrics import adjusted_rand_score
import json
import warnings
warnings.filterwarnings('ignore')

print("=== NB06: Zeroshot Model — Output Clustering + Pert Encoder Jacobian ===")
print(f"Torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")

OUTDIR = '/mnt/polished-lake/home/mbeheleramass/jacobian_analysis'

# ─────────────────────────────────────────────────────────────────────
# SETUP: Load ZEROSHOT model and data
# ─────────────────────────────────────────────────────────────────────
print("\n[1] Loading ZEROSHOT model and data...")

BASE = '/mnt/polished-lake/artifacts/fellows-shared/life-sciences/state/models/ST-HVG-Replogle/zeroshot/hepg2'

from state.tx.models.state_transition import StateTransitionPerturbationModel

model = StateTransitionPerturbationModel.load_from_checkpoint(
    f'{BASE}/checkpoints/best.ckpt',
    map_location='cuda' if torch.cuda.is_available() else 'cpu'
)
model.eval()
device = next(model.parameters()).device
print(f"Zeroshot model loaded on: {device}")

var_dims = pickle.load(open(f'{BASE}/var_dims.pkl', 'rb'))
gene_names = var_dims['gene_names']
gene_name_list = list(gene_names)
print(f"Gene names: {len(gene_name_list)} HVGs")

pert_map = torch.load(f'{BASE}/pert_onehot_map.pt', weights_only=False)
# Normalise keys to str
pert_map = {str(k): v for k, v in pert_map.items()}
print(f"Zeroshot pert map size: {len(pert_map)}")

# Load fewshot pert_map to identify intersection
BASE_FS = '/mnt/polished-lake/artifacts/fellows-shared/life-sciences/state/models/ST-HVG-Replogle/fewshot/hepg2'
pert_map_fs = torch.load(f'{BASE_FS}/pert_onehot_map.pt', weights_only=False)
pert_map_fs = {str(k): v for k, v in pert_map_fs.items()}
print(f"Fewshot pert map size: {len(pert_map_fs)}")

ctrl_keys = {'non-targeting', 'NT', 'Non-targeting', 'non_targeting'}
zs_kds = set(pert_map.keys()) - ctrl_keys
fs_kds = set(pert_map_fs.keys()) - ctrl_keys
shared_kds = zs_kds & fs_kds
print(f"Zeroshot-only KDs: {len(zs_kds)}")
print(f"Fewshot-only KDs: {len(fs_kds)}")
print(f"Shared KDs: {len(shared_kds)}")
print(f"KDs in fewshot but NOT in zeroshot: {len(fs_kds - zs_kds)} (not in Jurkat/K562/RPE1 training)")
print(f"KDs in zeroshot but NOT in fewshot: {len(zs_kds - fs_kds)}")

# ─────────────────────────────────────────────────────────────────────
# Load HepG2 control cells (same as NB05: seed=42, n=64)
# ─────────────────────────────────────────────────────────────────────
print("\nLoading HepG2 control cells (same seed=42 as NB05)...")
adata = ad.read_h5ad('/mnt/polished-lake/artifacts/fellows-shared/life-sciences/state/data/Replogle-Nadig-Preprint/replogle_matched_hvg.h5ad')
ctrl_mask = (adata.obs['cell_line'] == 'hepg2') & (adata.obs['gene'] == 'non-targeting')
ctrl_adata = adata[ctrl_mask]
print(f"HepG2 control cells: {ctrl_adata.n_obs}")

ctrl_mat = ctrl_adata.obsm['X_hvg']
if hasattr(ctrl_mat, 'toarray'):
    ctrl_mat = ctrl_mat.toarray()
print(f"ctrl_mat shape: {ctrl_mat.shape}")

np.random.seed(42)
CELL_SET_LEN = 64
idx = np.random.choice(len(ctrl_mat), size=CELL_SET_LEN, replace=False)
ctrl_cells = torch.tensor(ctrl_mat[idx], dtype=torch.float32).to(device)
print(f"ctrl_cells tensor: {ctrl_cells.shape}")

BATCH_IDX = 0
batch_t = torch.full((CELL_SET_LEN,), BATCH_IDX, dtype=torch.long).to(device)

# ─────────────────────────────────────────────────────────────────────
# Helper: run model forward
# ─────────────────────────────────────────────────────────────────────
def run_model(pert_key):
    """Run model forward for a given KD. Returns mean predicted output [2000]."""
    pert_oh = pert_map[pert_key].float().to(device)
    S = ctrl_cells.shape[0]
    pert_exp = pert_oh.unsqueeze(0).expand(S, -1)
    batch_dict = {
        'ctrl_cell_emb': ctrl_cells,
        'pert_emb': pert_exp,
        'pert_name': [pert_key] * S,
        'batch': batch_t,
    }
    with torch.no_grad():
        out = model.predict_step(batch_dict, batch_idx=0, padded=False)
    preds = out['preds'].reshape(-1, out['preds'].shape[-1])
    return preds.mean(0).cpu().numpy()

# ─────────────────────────────────────────────────────────────────────
# PART 1: Compute ΔY for all zeroshot KDs
# ─────────────────────────────────────────────────────────────────────
print("\n[PART 1] Computing ΔY for all zeroshot KDs...")

# Control baseline (use zeroshot model's non-targeting)
# Check which ctrl key exists
available_ctrl = [k for k in ctrl_keys if k in pert_map]
print(f"Available ctrl keys: {available_ctrl}")
ctrl_key_use = available_ctrl[0]
ctrl_mean = run_model(ctrl_key_use)
print(f"ctrl_mean shape: {ctrl_mean.shape}, range: [{ctrl_mean.min():.3f}, {ctrl_mean.max():.3f}]")

# All zeroshot perturbations (non-control)
all_perts = sorted([(k, v) for k, v in pert_map.items() if k not in ctrl_keys], key=lambda x: x[0])
print(f"KDs to process: {len(all_perts)}")

import time
delta_y = {}
t0_all = time.time()
for i, (pert_name, _) in enumerate(all_perts):
    pert_mean = run_model(pert_name)
    delta_y[pert_name] = pert_mean - ctrl_mean
    if (i+1) % 200 == 0:
        elapsed = time.time() - t0_all
        rate = (i+1) / elapsed
        eta = (len(all_perts) - i - 1) / rate
        print(f"  Processed {i+1}/{len(all_perts)} KDs... ({rate:.1f}/s, ETA {eta:.0f}s)")

delta_y_df = pd.DataFrame(delta_y).T  # [N_perts, 2000]
delta_y_df.columns = gene_name_list
print(f"\ndelta_y_df shape: {delta_y_df.shape}")
print(f"Mean |ΔY|: {delta_y_df.abs().values.mean():.4f}")

delta_y_df.to_parquet(f'{OUTDIR}/nb06_delta_y.parquet')
print("Saved nb06_delta_y.parquet")

# ─────────────────────────────────────────────────────────────────────
# Hierarchical clustering of ΔY
# ─────────────────────────────────────────────────────────────────────
print("\n[1b] Hierarchical clustering of ΔY vectors...")

dy_mat = delta_y_df.values.astype(np.float32)  # [N_perts, 2000]
pert_names_arr = np.array(delta_y_df.index.tolist())
N_perts = len(pert_names_arr)

cos_dist = pdist(dy_mat, metric='cosine')
Z = linkage(cos_dist, method='ward')

n_clusters = 15  # Same as NB05 for direct comparison
cluster_labels = fcluster(Z, n_clusters, criterion='maxclust')
print(f"Cluster sizes (n={n_clusters}): {np.unique(cluster_labels, return_counts=True)[1].tolist()}")

cluster_df = pd.DataFrame({'pert_name': pert_names_arr, 'cluster': cluster_labels})
cluster_df.to_csv(f'{OUTDIR}/nb06_clusters.csv', index=False)

# ─────────────────────────────────────────────────────────────────────
# ARI comparison with NB05 fewshot clustering
# ─────────────────────────────────────────────────────────────────────
print("\n[1b-ARI] Comparing cluster structure with NB05 fewshot...")

nb05_clusters = pd.read_csv(f'{OUTDIR}/nb05_clusters.csv', index_col='pert_name')
nb06_clusters_df = cluster_df.set_index('pert_name')

# Only KDs present in BOTH analyses
shared_for_ari = sorted(set(nb05_clusters.index) & set(nb06_clusters_df.index))
print(f"KDs for ARI comparison: {len(shared_for_ari)}")

if len(shared_for_ari) > 10:
    labels05 = nb05_clusters.loc[shared_for_ari, 'cluster'].values
    labels06 = nb06_clusters_df.loc[shared_for_ari, 'cluster'].values
    ari = adjusted_rand_score(labels05, labels06)
    print(f"Adjusted Rand Index (fewshot vs zeroshot clustering): {ari:.4f}")
    print(f"  (ARI=1.0 = identical structure; ARI~0 = random; ARI<0 = less than random)")
else:
    ari = None
    print("Not enough shared KDs for ARI")

# ─────────────────────────────────────────────────────────────────────
# UMAP of ΔY
# ─────────────────────────────────────────────────────────────────────
print("\n[1c] Computing UMAP of ΔY vectors...")

try:
    import umap as umap_lib
    reducer = umap_lib.UMAP(metric='cosine', n_neighbors=15, min_dist=0.1, random_state=42, verbose=False)
    umap_emb = reducer.fit_transform(dy_mat)
    dim_method = 'UMAP'
except ImportError:
    print("  umap not available, using PCA fallback")
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=42)
    umap_emb = pca.fit_transform(dy_mat)
    dim_method = 'PCA(2)'

print(f"{dim_method} embedding: {umap_emb.shape}")
np.save(f'{OUTDIR}/nb06_umap_emb.npy', umap_emb)

# ─────────────────────────────────────────────────────────────────────
# Pathway annotation (same patterns as NB05)
# ─────────────────────────────────────────────────────────────────────
print("\n[1d] Pathway enrichment per cluster...")

pathway_patterns = {
    'ribosome':    [g for g in gene_name_list if g.startswith('RPL') or g.startswith('RPS')],
    'proteasome':  [g for g in gene_name_list if g.startswith('PSM')],
    'ER_secretory':[g for g in gene_name_list if any(g.startswith(p) for p in ['SEC', 'SRP', 'TRAPPC', 'SSR', 'DERL', 'CANX', 'CALR', 'HSPA5'])],
    'mito_complex':[g for g in gene_name_list if g.startswith('MT-') or g.startswith('MRPL') or g.startswith('MRPS')],
    'splicing':    [g for g in gene_name_list if any(g.startswith(p) for p in ['SNRP', 'SF3', 'U2AF', 'SRSF'])],
    'DNA_repair':  [g for g in gene_name_list if any(g.startswith(p) for p in ['BRCA', 'RAD', 'PARP', 'MLH', 'MSH'])],
    'chromatin':   [g for g in gene_name_list if any(g.startswith(p) for p in ['HIST', 'H2A', 'H2B', 'H3C', 'SMARCA', 'ARID'])],
    'MYC_targets': [g for g in ['MYC', 'MAX', 'MYCN', 'MYB', 'E2F1', 'E2F2', 'CDK4', 'CCND1', 'CCND2'] if g in gene_name_list],
}
print("Pathway gene counts:", {k: len(v) for k, v in pathway_patterns.items()})

cluster_pathway_scores = {}
for c in range(1, n_clusters+1):
    mask = cluster_labels == c
    cluster_dy = dy_mat[mask].mean(0)
    scores = {}
    for pname, pgenes in pathway_patterns.items():
        if len(pgenes) == 0:
            scores[pname] = 0.0
            continue
        pidx = [gene_name_list.index(g) for g in pgenes]
        scores[pname] = np.abs(cluster_dy[pidx]).mean()
    cluster_pathway_scores[c] = scores

pathway_score_df = pd.DataFrame(cluster_pathway_scores).T

cluster_top_pathway = {}
cluster_top_genes = {}
for c in range(1, n_clusters+1):
    top = pathway_score_df.loc[c].idxmax()
    cluster_top_pathway[c] = top
    mask = cluster_labels == c
    mean_dy = dy_mat[mask].mean(0)
    top_up = np.argsort(mean_dy)[-5:][::-1]
    top_down = np.argsort(mean_dy)[:5]
    cluster_top_genes[c] = {
        'top_up': [gene_name_list[i] for i in top_up],
        'top_down': [gene_name_list[i] for i in top_down],
        'mean_abs_dy': float(np.abs(mean_dy).mean()),
        'n': int(mask.sum()),
    }

print("\nCluster summary:")
for c in sorted(cluster_top_pathway.keys()):
    info = cluster_top_genes[c]
    print(f"  C{c:2d} (n={info['n']:3d}, mean|ΔY|={info['mean_abs_dy']:.4f}): "
          f"top={cluster_top_pathway[c]}, up={info['top_up'][:3]}")

# Pathway color map
pathway_color_map = {
    'ribosome':    '#e41a1c',
    'proteasome':  '#377eb8',
    'ER_secretory':'#4daf4a',
    'mito_complex':'#984ea3',
    'splicing':    '#ff7f00',
    'DNA_repair':  '#a65628',
    'chromatin':   '#f781bf',
    'MYC_targets': '#999999',
}

# ─────────────────────────────────────────────────────────────────────
# Figure 1: UMAP (zeroshot only) — saved as nb06_zeroshot_delta_y_umap.png
# ─────────────────────────────────────────────────────────────────────
print("\n[1e] Plotting zeroshot UMAP...")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
cmap = plt.cm.get_cmap('tab20', n_clusters)

ax = axes[0]
for c in range(1, n_clusters+1):
    mask = cluster_labels == c
    ax.scatter(umap_emb[mask, 0], umap_emb[mask, 1],
               c=[cmap(c-1)], s=12, alpha=0.7, label=f'C{c} (n={mask.sum()})')
ax.set_title(f'Zeroshot ΔY {dim_method} — colored by cluster\n(model trained on Jurkat/K562/RPE1 only)', fontsize=11)
ax.set_xlabel(f'{dim_method} 1'); ax.set_ylabel(f'{dim_method} 2')
ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=7, markerscale=1.5, ncol=1)

ax2 = axes[1]
plotted_pathways = set()
for c in range(1, n_clusters+1):
    mask = cluster_labels == c
    pname = cluster_top_pathway[c]
    color = pathway_color_map.get(pname, '#cccccc')
    label = pname if pname not in plotted_pathways else None
    ax2.scatter(umap_emb[mask, 0], umap_emb[mask, 1],
                c=color, s=12, alpha=0.7, label=label)
    plotted_pathways.add(pname)
    cx, cy = umap_emb[mask, 0].mean(), umap_emb[mask, 1].mean()
    ax2.text(cx, cy, str(c), fontsize=7, ha='center', va='center',
             fontweight='bold', color='white',
             bbox=dict(boxstyle='round,pad=0.15', facecolor='black', alpha=0.4, edgecolor='none'))

ax2.set_title('Zeroshot ΔY — colored by top pathway', fontsize=11)
ax2.set_xlabel(f'{dim_method} 1'); ax2.set_ylabel(f'{dim_method} 2')
ax2.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9)

plt.suptitle(f'ARI vs NB05 fewshot clustering: {ari:.3f}' if ari is not None else 'ARI: N/A',
             fontsize=10, y=1.01)
plt.tight_layout()
plt.savefig(f'{OUTDIR}/nb06_zeroshot_delta_y_umap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved nb06_zeroshot_delta_y_umap.png")

# ─────────────────────────────────────────────────────────────────────
# Figure 2: Pathway enrichment (nb06_pathway_enrichment.png)
# ─────────────────────────────────────────────────────────────────────
print("\n[1f] Plotting pathway enrichment...")

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

score_arr = pathway_score_df.values
score_norm = score_arr / (score_arr.max(axis=0, keepdims=True) + 1e-8)

ax = axes[0]
im = ax.imshow(score_norm.T, aspect='auto', cmap='YlOrRd')
ax.set_xticks(range(n_clusters))
ax.set_xticklabels([f'C{c}' for c in range(1, n_clusters+1)], fontsize=8)
ax.set_yticks(range(len(pathway_patterns)))
ax.set_yticklabels(list(pathway_patterns.keys()), fontsize=10)
ax.set_xlabel('Cluster')
ax.set_title('Zeroshot pathway enrichment score per cluster\n(normalized per pathway)', fontsize=11)
plt.colorbar(im, ax=ax, label='Normalized mean |ΔY|')

ax2 = axes[1]
sizes = [(cluster_labels == c).sum() for c in range(1, n_clusters+1)]
colors = [pathway_color_map.get(cluster_top_pathway[c], '#cccccc') for c in range(1, n_clusters+1)]
ax2.bar(range(1, n_clusters+1), sizes, color=colors, edgecolor='black', linewidth=0.5)
ax2.set_xlabel('Cluster'); ax2.set_ylabel('Number of KDs')
ax2.set_title('Zeroshot cluster sizes (colored by top pathway)', fontsize=11)
ax2.set_xticks(range(1, n_clusters+1))
ax2.set_xticklabels([f'C{c}' for c in range(1, n_clusters+1)], fontsize=8)

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=v, edgecolor='black', label=k)
                   for k, v in pathway_color_map.items() if k in set(cluster_top_pathway.values())]
ax2.legend(handles=legend_elements, loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig(f'{OUTDIR}/nb06_pathway_enrichment.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved nb06_pathway_enrichment.png")

# ─────────────────────────────────────────────────────────────────────
# PART 2: Perturbation encoder Jacobian ∂Y/∂pert_repr (zeroshot)
# ─────────────────────────────────────────────────────────────────────
print("\n[PART 2] Computing ∂Y/∂pert_repr Jacobians (zeroshot model)...")

# Check pert_repr dim
with torch.no_grad():
    test_oh = list(pert_map.values())[0].float().to(device)
    test_repr = model.pert_encoder(test_oh.unsqueeze(0))
    PERT_REPR_DIM = test_repr.shape[-1]
    print(f"pert_repr dim: {PERT_REPR_DIM}")

# Select representative KDs: top 15 per cluster by mean |ΔY|
# BUT: prioritize shared KDs (in NB05 Jacobian set) for direct comparison
print("\n[2a] Selecting representative KDs (top 15 per cluster)...")

# Load NB05 Jacobian KD names for intersection
with open(f'{OUTDIR}/nb05_jacobian_kd_names.pkl', 'rb') as f:
    nb05_jac_kds = set(pickle.load(f))

pert_name_to_idx = {p: i for i, p in enumerate(pert_names_arr)}

rep_kds = []
for c in range(1, n_clusters+1):
    mask = cluster_labels == c
    cluster_perts = pert_names_arr[mask]
    cluster_dy_abs = np.abs(dy_mat[mask]).mean(1)
    top_n = min(15, len(cluster_perts))
    top_idx_local = np.argsort(cluster_dy_abs)[-top_n:][::-1]
    rep_kds.extend(cluster_perts[top_idx_local].tolist())

rep_kds = list(dict.fromkeys(rep_kds))
print(f"Representative KDs: {len(rep_kds)}")
print(f"  Overlap with NB05 Jacobian KDs: {len(set(rep_kds) & nb05_jac_kds)}")

# If total KDs <= 600, use all
if N_perts <= 600:
    rep_kds = list(pert_names_arr)
    print(f"Using all {len(rep_kds)} KDs (total <= 600)")
print(f"Final KD set for Jacobians: {len(rep_kds)}")

# ─────────────────────────────────────────────────────────────────────
# Jacobian computation helpers
# ─────────────────────────────────────────────────────────────────────

def get_pert_repr(pert_key):
    oh = pert_map[pert_key].float().to(device)
    with torch.no_grad():
        pr = model.pert_encoder(oh.unsqueeze(0))
    return pr.detach()


def forward_from_pert_repr(pert_repr_1d):
    """
    Forward pass from pert_repr [PERT_REPR_DIM] to mean output [2000].
    """
    S = ctrl_cells.shape[0]
    basal_enc = model.basal_encoder(ctrl_cells)
    pert_repr_expanded = pert_repr_1d.unsqueeze(0).expand(S, -1)
    combined = basal_enc + pert_repr_expanded

    if hasattr(model, 'batch_encoder') and model.batch_encoder is not None:
        batch_idx = torch.zeros(S, dtype=torch.long, device=device)
        batch_emb = model.batch_encoder(batch_idx)
        combined = combined + batch_emb

    seq_input = combined.unsqueeze(0)
    outputs = model.transformer_backbone(inputs_embeds=seq_input)
    transformer_out = outputs.last_hidden_state

    res_pred = transformer_out.squeeze(0)
    out_pred = model.project_out(res_pred)
    out_pred = model.relu(out_pred)
    return out_pred.mean(0)


def compute_pert_jacobian_vectorized(pert_key):
    pr = get_pert_repr(pert_key).squeeze(0)
    J = torch.autograd.functional.jacobian(
        forward_from_pert_repr, pr, vectorize=True, create_graph=False,
    )
    return J.detach().cpu().numpy()


def compute_pert_jacobian_manual(pert_key):
    pr = get_pert_repr(pert_key).squeeze(0).requires_grad_(True)
    mean_preds = forward_from_pert_repr(pr)
    n_genes = len(mean_preds)
    J = np.zeros((n_genes, PERT_REPR_DIM), dtype=np.float32)
    for j in range(n_genes):
        grad = torch.autograd.grad(
            mean_preds[j], pr,
            retain_graph=(j < n_genes - 1),
            create_graph=False,
        )[0]
        J[j] = grad.detach().cpu().numpy()
    return J


# Test on one KD
print("\n[2b] Testing Jacobian computation...")
test_kd = rep_kds[0]
jacobian_works = False
use_vectorized = True

try:
    J_test = compute_pert_jacobian_vectorized(test_kd)
    print(f"Vectorized Jacobian OK: shape={J_test.shape}, range=[{J_test.min():.4f}, {J_test.max():.4f}]")
    jacobian_works = True
    use_vectorized = True
except Exception as e:
    print(f"Vectorized failed: {e}")
    try:
        J_test = compute_pert_jacobian_manual(test_kd)
        print(f"Manual Jacobian OK: shape={J_test.shape}")
        jacobian_works = True
        use_vectorized = False
    except Exception as e2:
        print(f"Manual also failed: {e2}")

# ─────────────────────────────────────────────────────────────────────
# Compute Jacobians
# ─────────────────────────────────────────────────────────────────────
if jacobian_works:
    compute_fn = compute_pert_jacobian_vectorized if use_vectorized else compute_pert_jacobian_manual
    print(f"\n[2c] Computing Jacobians for {len(rep_kds)} KDs "
          f"({'vectorized' if use_vectorized else 'manual'})...")

    jacobians = {}
    failed_kds = []
    t0 = time.time()

    for i, kd_name in enumerate(rep_kds):
        try:
            J = compute_fn(kd_name)
            jacobians[kd_name] = J
        except Exception as e:
            failed_kds.append((kd_name, str(e)))

        if (i+1) % 25 == 0:
            elapsed = time.time() - t0
            rate = (i+1) / elapsed
            eta = (len(rep_kds) - i - 1) / rate
            print(f"  {i+1}/{len(rep_kds)} done, {len(failed_kds)} failed, {rate:.1f}/s, ETA {eta:.0f}s")

    print(f"\nJacobians computed: {len(jacobians)}, failed: {len(failed_kds)}")

    jac_kd_names = list(jacobians.keys())
    jac_mat = np.stack([jacobians[k] for k in jac_kd_names])
    print(f"jac_mat shape: {jac_mat.shape}")

    np.save(f'{OUTDIR}/nb06_jacobians.npy', jac_mat)
    with open(f'{OUTDIR}/nb06_jacobian_kd_names.pkl', 'wb') as f:
        pickle.dump(jac_kd_names, f)
    print("Saved nb06_jacobians.npy and nb06_jacobian_kd_names.pkl")
else:
    print("Jacobian computation failed — skipping Parts 2 and 3.")
    jac_kd_names = []
    jac_mat = None

# ─────────────────────────────────────────────────────────────────────
# PART 3: Routing similarity comparison
# ─────────────────────────────────────────────────────────────────────
print("\n[PART 3] Routing similarity analysis (zeroshot vs fewshot r=0.119)...")

FEWSHOT_PEARSON_R = 0.11920101195573807  # from NB05

r_pearson = None
r_spearman = None

if jac_mat is not None and len(jac_kd_names) > 10:
    from sklearn.metrics.pairwise import cosine_similarity

    # Align ΔY and cluster labels to jac_kd_names
    rep_dy_indices = [pert_name_to_idx[k] for k in jac_kd_names if k in pert_name_to_idx]
    jac_kd_names_aligned = [k for k in jac_kd_names if k in pert_name_to_idx]
    jac_mat_aligned = np.stack([jacobians[k] for k in jac_kd_names_aligned])

    rep_dy_mat = dy_mat[rep_dy_indices]
    rep_cluster_labels = cluster_labels[rep_dy_indices]
    N = len(jac_kd_names_aligned)
    print(f"Aligned KDs: {N}")

    jac_flat = jac_mat_aligned.reshape(N, -1).astype(np.float32)
    print(f"Flattened Jacobians: {jac_flat.shape}")

    print("Computing pairwise cosine similarities...")
    jac_cos_sim = cosine_similarity(jac_flat)
    dy_cos_sim = cosine_similarity(rep_dy_mat)

    print(f"Jacobian cosine sim: [{jac_cos_sim.min():.3f}, {jac_cos_sim.max():.3f}]")
    print(f"ΔY cosine sim: [{dy_cos_sim.min():.3f}, {dy_cos_sim.max():.3f}]")

    triu_idx = np.triu_indices(N, k=1)
    jac_pairs = jac_cos_sim[triu_idx]
    dy_pairs = dy_cos_sim[triu_idx]
    same_cluster = rep_cluster_labels[triu_idx[0]] == rep_cluster_labels[triu_idx[1]]

    print(f"\nTotal pairs: {len(jac_pairs)}")
    print(f"Same-cluster pairs: {same_cluster.sum()} ({same_cluster.mean():.1%})")

    r_pearson, p_pearson = pearsonr(dy_pairs, jac_pairs)
    r_spearman, p_spearman = spearmanr(dy_pairs, jac_pairs)
    print(f"\nZeroshot — Jacobian sim ~ ΔY sim:")
    print(f"  Pearson r = {r_pearson:.4f}, p = {p_pearson:.2e}")
    print(f"  Spearman r = {r_spearman:.4f}, p = {p_spearman:.2e}")
    print(f"\nFewshot reference (NB05): Pearson r = {FEWSHOT_PEARSON_R:.4f}")
    print(f"Difference: {r_pearson - FEWSHOT_PEARSON_R:+.4f}")

    same_jac = jac_pairs[same_cluster]
    diff_jac = jac_pairs[~same_cluster]

    np.random.seed(42)
    n_sample = min(len(same_jac), 5000)
    same_jac_s = np.random.choice(same_jac, size=n_sample, replace=False)
    diff_jac_s = np.random.choice(diff_jac, size=n_sample, replace=False)

    stat, p_wilcoxon = wilcoxon(same_jac_s, diff_jac_s)
    same_diff_ratio = same_jac.mean() / (diff_jac.mean() + 1e-8)
    print(f"\nWilcoxon (same vs diff cluster Jacobian sim):")
    print(f"  Same-cluster: mean={same_jac.mean():.4f}")
    print(f"  Diff-cluster: mean={diff_jac.mean():.4f}")
    print(f"  Ratio (same/diff): {same_diff_ratio:.2f}x")
    print(f"  p = {p_wilcoxon:.2e}")
    print(f"\nFewshot reference ratio: 1.92x")

    # SVD
    print("\n[3b] SVD of Jacobian matrix...")
    from sklearn.decomposition import TruncatedSVD

    n_components = min(10, N - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    kd_scores = svd.fit_transform(jac_flat)
    explained_var = svd.explained_variance_ratio_
    print(f"SVD explained variance (top {n_components}): {explained_var.round(4).tolist()}")

    n_genes_out = jac_mat_aligned.shape[1]
    n_pert_dim = jac_mat_aligned.shape[2]
    components_3d = svd.components_[:3].reshape(3, n_genes_out, n_pert_dim)
    gene_loadings = np.abs(components_3d).sum(axis=2)

    top_gene_names_per_comp = {}
    for comp_i in range(3):
        top_idx = np.argsort(gene_loadings[comp_i])[-20:][::-1]
        top_gene_names_per_comp[comp_i+1] = [gene_name_list[j] for j in top_idx]
        print(f"\n  SVD component {comp_i+1} top genes: {top_gene_names_per_comp[comp_i+1][:10]}")

    # ─────────────────────────────────────────────────────────────────────
    # Figure 3: Routing vs output similarity scatter (nb06_jacobian_vs_output_similarity.png)
    # ─────────────────────────────────────────────────────────────────────
    print("\n[3c] Plotting Jacobian vs ΔY similarity...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    n_plot = min(5000, len(jac_pairs))
    np.random.seed(42)
    plot_idx = np.random.choice(len(jac_pairs), size=n_plot, replace=False)

    ax = axes[0]
    scatter_colors = np.where(same_cluster[plot_idx], '#e41a1c', '#aaaaaa')
    ax.scatter(dy_pairs[plot_idx], jac_pairs[plot_idx],
               c=scatter_colors, s=8, alpha=0.35)

    slope = r_pearson * jac_pairs.std() / (dy_pairs.std() + 1e-8)
    intercept = jac_pairs.mean() - slope * dy_pairs.mean()
    xfit = np.linspace(dy_pairs.min(), dy_pairs.max(), 100)
    ax.plot(xfit, slope * xfit + intercept, 'b-', linewidth=2, label=f'Zeroshot r={r_pearson:.3f}')

    # Annotate fewshot reference r
    ax.axhline(jac_pairs.mean(), color='gray', linestyle=':', alpha=0.5)
    ax.text(0.05, 0.92, f'Fewshot reference r={FEWSHOT_PEARSON_R:.3f}',
            transform=ax.transAxes, fontsize=9, color='darkred',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

    from matplotlib.patches import Patch
    handles = [Patch(facecolor='#e41a1c', label=f'Same cluster (n={same_cluster.sum()})'),
               Patch(facecolor='#aaaaaa', label=f'Diff cluster (n={(~same_cluster).sum()})')]
    ax.legend(handles=handles, fontsize=9)
    ax.set_xlabel('ΔY cosine similarity', fontsize=11)
    ax.set_ylabel('∂Y/∂pert_repr cosine similarity', fontsize=11)
    ax.set_title(f'Zeroshot — Routing vs Output similarity\n'
                 f'Pearson r={r_pearson:.3f} (p={p_pearson:.1e}) | '
                 f'Fewshot r={FEWSHOT_PEARSON_R:.3f}', fontsize=10)

    ax2 = axes[1]
    n_violin = min(len(same_jac), len(diff_jac), 2000)
    np.random.seed(42)
    violin_data = [
        np.random.choice(same_jac, n_violin, replace=False),
        np.random.choice(diff_jac, n_violin, replace=False)
    ]
    parts = ax2.violinplot(violin_data, positions=[1, 2], showmedians=True, showextrema=True)
    parts['cmedians'].set_color('black')
    parts['bodies'][0].set_facecolor('#e41a1c')
    parts['bodies'][1].set_facecolor('#aaaaaa')
    for b in parts['bodies']:
        b.set_alpha(0.7)
    ax2.set_xticks([1, 2])
    ax2.set_xticklabels([f'Same cluster\n(n={len(same_jac)})',
                          f'Diff cluster\n(n={len(diff_jac)})'], fontsize=10)
    ax2.set_ylabel('∂Y/∂pert_repr cosine similarity', fontsize=10)
    ax2.set_title(f'Wilcoxon: same vs diff cluster routing\n'
                  f'p={p_wilcoxon:.2e}, ratio={same_diff_ratio:.2f}x (fewshot: 1.92x)', fontsize=10)
    ax2.axhline(0, color='black', linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(f'{OUTDIR}/nb06_jacobian_vs_output_similarity.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved nb06_jacobian_vs_output_similarity.png")

else:
    print("Skipping Part 3 — no Jacobians available.")
    r_pearson = None
    p_pearson = None
    r_spearman = None
    p_spearman = None
    p_wilcoxon = None
    same_diff_ratio = None
    explained_var = []
    top_gene_names_per_comp = {}
    N = 0
    jac_kd_names_aligned = []
    same_jac = np.array([])
    diff_jac = np.array([])
    kd_scores = np.zeros((0, 1))
    gene_loadings = None

# ─────────────────────────────────────────────────────────────────────
# PART 4: Side-by-side comparison figure
# ─────────────────────────────────────────────────────────────────────
print("\n[PART 4] Comparison figures (fewshot vs zeroshot)...")

# Load fewshot ΔY for shared KDs
nb05_delta_y = pd.read_parquet(f'{OUTDIR}/nb05_delta_y.parquet')
nb05_umap_emb = np.load(f'{OUTDIR}/nb05_umap_emb.npy')
nb05_clusters_loaded = pd.read_csv(f'{OUTDIR}/nb05_clusters.csv', index_col='pert_name')

# ─── Fig 4a: Side-by-side UMAP ────────────────────────────────────────
print("[4a] Side-by-side UMAP comparison...")

# Use consistent pathway color across both panels
fig, axes = plt.subplots(2, 2, figsize=(20, 16))

# Load NB05 pathway top assignments (from summary)
with open(f'{OUTDIR}/nb05_summary.json') as f:
    nb05_summary = json.load(f)
nb05_cluster_top_pathway = {int(k): v for k, v in nb05_summary['cluster_top_pathway'].items()}

# Fewshot UMAP — cluster colors
ax = axes[0, 0]
n_fs_clusters = max(nb05_clusters_loaded['cluster'])
cmap_fs = plt.cm.get_cmap('tab20', n_fs_clusters)
for c in range(1, n_fs_clusters+1):
    mask_fs = nb05_clusters_loaded['cluster'].values == c
    pname_fs = nb05_cluster_top_pathway.get(c, 'other')
    col = pathway_color_map.get(pname_fs, '#cccccc')
    ax.scatter(nb05_umap_emb[mask_fs, 0], nb05_umap_emb[mask_fs, 1],
               c=col, s=8, alpha=0.6)
    cx, cy = nb05_umap_emb[mask_fs, 0].mean(), nb05_umap_emb[mask_fs, 1].mean()
    ax.text(cx, cy, str(c), fontsize=6, ha='center', va='center',
            fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.1', facecolor='black', alpha=0.4, edgecolor='none'))
ax.set_title('Fewshot/HepG2 ΔY UMAP\n(trained on HepG2 perturbations)', fontsize=12, fontweight='bold')
ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2')

# Zeroshot UMAP — cluster colors
ax2 = axes[0, 1]
for c in range(1, n_clusters+1):
    mask_zs = cluster_labels == c
    pname_zs = cluster_top_pathway[c]
    col = pathway_color_map.get(pname_zs, '#cccccc')
    ax2.scatter(umap_emb[mask_zs, 0], umap_emb[mask_zs, 1],
                c=col, s=8, alpha=0.6)
    cx, cy = umap_emb[mask_zs, 0].mean(), umap_emb[mask_zs, 1].mean()
    ax2.text(cx, cy, str(c), fontsize=6, ha='center', va='center',
             fontweight='bold', color='white',
             bbox=dict(boxstyle='round,pad=0.1', facecolor='black', alpha=0.4, edgecolor='none'))
ax2.set_title(f'Zeroshot/HepG2 ΔY UMAP\n(trained on Jurkat/K562/RPE1 only; ARI={ari:.3f})', fontsize=12, fontweight='bold')
ax2.set_xlabel('UMAP 1'); ax2.set_ylabel('UMAP 2')

# Shared pathway legend
from matplotlib.patches import Patch
all_pathways_used = set(nb05_cluster_top_pathway.values()) | set(cluster_top_pathway.values())
legend_elements = [Patch(facecolor=pathway_color_map.get(p, '#cccccc'), edgecolor='black', label=p)
                   for p in sorted(all_pathways_used)]
axes[0, 0].legend(handles=legend_elements, loc='lower left', fontsize=7, title='Pathway')

# ─── Fig 4b: ΔY magnitude scatter ─────────────────────────────────────
print("[4b] ΔY magnitude scatter (fewshot vs zeroshot)...")

# Compute magnitudes for shared KDs
shared_sorted = sorted(shared_kds)
fs_magnitudes = {}
zs_magnitudes = {}

for kd in shared_sorted:
    if kd in nb05_delta_y.index:
        fs_magnitudes[kd] = float(np.linalg.norm(nb05_delta_y.loc[kd].values))
    if kd in delta_y_df.index:
        zs_magnitudes[kd] = float(np.linalg.norm(delta_y_df.loc[kd].values))

common_kds = sorted(set(fs_magnitudes.keys()) & set(zs_magnitudes.keys()))
print(f"Common KDs for magnitude scatter: {len(common_kds)}")

fs_mag_arr = np.array([fs_magnitudes[k] for k in common_kds])
zs_mag_arr = np.array([zs_magnitudes[k] for k in common_kds])

# Annotate per-KD pathway for color
pathway_for_kd = {}
for kd in common_kds:
    # Use NB05 cluster assignment → pathway label
    if kd in nb05_clusters_loaded.index:
        c = nb05_clusters_loaded.loc[kd, 'cluster']
        pathway_for_kd[kd] = nb05_cluster_top_pathway.get(int(c), 'other')
    else:
        pathway_for_kd[kd] = 'other'

kd_colors = [pathway_color_map.get(pathway_for_kd[k], '#cccccc') for k in common_kds]

ax3 = axes[1, 0]
ax3.scatter(fs_mag_arr, zs_mag_arr, c=kd_colors, s=12, alpha=0.6)
max_mag = max(fs_mag_arr.max(), zs_mag_arr.max())
ax3.plot([0, max_mag], [0, max_mag], 'k--', alpha=0.4, label='y=x')

# Annotate outliers (fewshot >> zeroshot: HepG2-memorized)
ratio = fs_mag_arr / (zs_mag_arr + 1e-6)
top_outliers_idx = np.argsort(ratio)[-10:][::-1]
for idx in top_outliers_idx[:7]:
    ax3.annotate(common_kds[idx],
                 (fs_mag_arr[idx], zs_mag_arr[idx]),
                 fontsize=6, xytext=(3, 3), textcoords='offset points', alpha=0.9)

# Annotate bottom outliers (zeroshot >> fewshot: backbone-specific)
bottom_outliers_idx = np.argsort(ratio)[:5]
for idx in bottom_outliers_idx[:3]:
    ax3.annotate(common_kds[idx],
                 (fs_mag_arr[idx], zs_mag_arr[idx]),
                 fontsize=6, xytext=(3, 3), textcoords='offset points', alpha=0.9, color='blue')

r_mag, _ = pearsonr(np.log1p(fs_mag_arr), np.log1p(zs_mag_arr))
ax3.set_xlabel('||ΔY|| fewshot/HepG2', fontsize=11)
ax3.set_ylabel('||ΔY|| zeroshot/HepG2', fontsize=11)
ax3.set_title(f'ΔY magnitude per KD: Fewshot vs Zeroshot\n(n={len(common_kds)} shared KDs; log-space r={r_mag:.3f})', fontsize=11)
ax3.legend(fontsize=9)

# Add pathway legend
legend_elements2 = [Patch(facecolor=v, edgecolor='black', label=k)
                    for k, v in pathway_color_map.items() if k in set(pathway_for_kd.values())]
ax3.legend(handles=legend_elements2, loc='upper left', fontsize=7, title='KD pathway (fewshot)')

# ─── Fig 4c: Summary routing comparison bar ───────────────────────────
ax4 = axes[1, 1]

# Summary bar chart: fewshot vs zeroshot metrics
labels_bar = ['Pearson r\n(routing~output)', 'Same/Diff ratio\n(Jacobian sim)']
if r_pearson is not None:
    fewshot_vals = [0.119, 1.92]
    zeroshot_vals = [r_pearson, same_diff_ratio if same_diff_ratio is not None else 0]
    x = np.arange(len(labels_bar))
    width = 0.35
    bars1 = ax4.bar(x - width/2, fewshot_vals, width, label='Fewshot/HepG2\n(NB05)', color='#2171b5', alpha=0.8)
    bars2 = ax4.bar(x + width/2, zeroshot_vals, width, label='Zeroshot/HepG2\n(NB06)', color='#cb181d', alpha=0.8)
    for bar, val in zip(bars1, fewshot_vals):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#2171b5')
    for bar, val in zip(bars2, zeroshot_vals):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#cb181d')
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels_bar, fontsize=11)
    ax4.set_ylabel('Value', fontsize=11)
    ax4.set_title('Routing coherence: Fewshot vs Zeroshot', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.set_ylim(0, max(max(fewshot_vals), max(zeroshot_vals)) * 1.3)
    ax4.axhline(0, color='black', linewidth=0.5)

    if ari is not None:
        ax4.text(0.5, 0.05, f'Cluster ARI (fewshot vs zeroshot): {ari:.3f}',
                 transform=ax4.transAxes, ha='center', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
else:
    ax4.text(0.5, 0.5, 'Jacobian computation failed', ha='center', va='center', transform=ax4.transAxes)

plt.tight_layout()
plt.savefig(f'{OUTDIR}/nb06_fewshot_vs_zeroshot_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved nb06_fewshot_vs_zeroshot_comparison.png")

# ─────────────────────────────────────────────────────────────────────
# Per-KD ΔY correlation: fewshot vs zeroshot
# ─────────────────────────────────────────────────────────────────────
print("\n[4c] Per-KD ΔY vector correlations (fewshot vs zeroshot)...")

kd_corr = {}
for kd in common_kds:
    if kd in nb05_delta_y.index and kd in delta_y_df.index:
        fs_vec = nb05_delta_y.loc[kd].values.astype(float)
        zs_vec = delta_y_df.loc[kd].values.astype(float)
        r, _ = pearsonr(fs_vec, zs_vec)
        kd_corr[kd] = r

kd_corr_s = pd.Series(kd_corr).sort_values(ascending=False)
print(f"\nPer-KD ΔY correlation (fewshot vs zeroshot):")
print(f"  Mean: {kd_corr_s.mean():.3f}")
print(f"  Median: {kd_corr_s.median():.3f}")
print(f"  Std: {kd_corr_s.std():.3f}")
print(f"  Top 10 (most similar): {kd_corr_s.head(10).to_dict()}")
print(f"  Bottom 10 (most different): {kd_corr_s.tail(10).to_dict()}")

kd_corr_s.to_csv(f'{OUTDIR}/nb06_kd_fewshot_zeroshot_corr.csv', header=['pearson_r'])
print("Saved nb06_kd_fewshot_zeroshot_corr.csv")

# ─────────────────────────────────────────────────────────────────────
# Save summary JSON
# ─────────────────────────────────────────────────────────────────────
summary = {
    'model': 'zeroshot/hepg2',
    'n_kds_total': int(N_perts),
    'n_zeroshot_kds': int(len(zs_kds)),
    'n_fewshot_kds': int(len(fs_kds)),
    'n_shared_kds': int(len(shared_kds)),
    'n_clusters': n_clusters,
    'cluster_ari_vs_fewshot': float(ari) if ari is not None else None,
    'n_rep_kds_jacobian': int(N) if N else 0,
    'fewshot_pearson_r': float(FEWSHOT_PEARSON_R),
    'pearson_r_jac_dy': float(r_pearson) if r_pearson is not None else None,
    'pearson_p_jac_dy': float(p_pearson) if p_pearson is not None else None,
    'spearman_r_jac_dy': float(r_spearman) if r_spearman is not None else None,
    'spearman_p_jac_dy': float(p_spearman) if p_spearman is not None else None,
    'wilcoxon_p': float(p_wilcoxon) if p_wilcoxon is not None else None,
    'same_cluster_jac_mean': float(same_jac.mean()) if len(same_jac) > 0 else None,
    'diff_cluster_jac_mean': float(diff_jac.mean()) if len(diff_jac) > 0 else None,
    'same_diff_ratio': float(same_diff_ratio) if same_diff_ratio is not None else None,
    'fewshot_same_diff_ratio': 1.92,
    'jac_pert_repr_dim': int(PERT_REPR_DIM),
    'jac_output_dim': 2000,
    'svd_explained_var': explained_var[:5].tolist() if len(explained_var) >= 5 else list(explained_var),
    'svd_top_genes': {str(k): v[:10] for k, v in top_gene_names_per_comp.items()},
    'cluster_top_pathway': {str(k): v for k, v in cluster_top_pathway.items()},
    'per_kd_corr_mean': float(kd_corr_s.mean()),
    'per_kd_corr_median': float(kd_corr_s.median()),
    'per_kd_corr_std': float(kd_corr_s.std()),
}

# Verdict
if r_pearson is not None:
    if r_pearson > 0.3:
        verdict = "STRONG POSITIVE — backbone has internalized circuits"
        interp = "Zeroshot routing coherence exceeds fewshot — backbone structure drives routing"
    elif r_pearson > 0.1:
        verdict = "WEAK POSITIVE — partial backbone circuit structure"
        interp = "Similar to fewshot; backbone contributes routing coherence pre-fine-tuning"
    elif r_pearson > 0.05:
        verdict = "MARGINAL — weak backbone structure"
        interp = "Zeroshot r < fewshot — fine-tuning adds most of the routing coherence"
    else:
        verdict = "NEGATIVE — fine-tuning artifact"
        interp = "No backbone routing structure; fewshot r=0.119 is HepG2 memorization"
    summary['verdict'] = verdict
    summary['interpretation'] = interp
    print(f"\nVERDICT: {verdict}")
    print(f"Interpretation: {interp}")

with open(f'{OUTDIR}/nb06_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("\nSaved nb06_summary.json")

print("\n=== NB06 COMPLETE ===")
print(f"\nKey comparison:")
print(f"  Fewshot routing r  = {FEWSHOT_PEARSON_R:.4f}")
print(f"  Zeroshot routing r = {r_pearson:.4f}" if r_pearson is not None else "  Zeroshot routing r = N/A")
print(f"  ARI (cluster agreement) = {ari:.4f}" if ari is not None else "  ARI = N/A")
print(f"  Per-KD ΔY corr (mean) = {kd_corr_s.mean():.4f}")
