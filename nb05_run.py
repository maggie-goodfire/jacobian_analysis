"""
NB05 — Output Clustering + Perturbation Encoder Jacobian
Part 1: Compute ΔY for all KDs, cluster, UMAP, pathway enrichment
Part 2: ∂Y/∂pert_repr Jacobian for representative KDs
Part 3: Jacobian similarity vs output similarity
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
import json
import warnings
warnings.filterwarnings('ignore')

print("=== NB05: Output Clustering + Pert Encoder Jacobian ===")
print(f"Torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")

OUTDIR = '/mnt/polished-lake/home/mbeheleramass/jacobian_analysis'

# ─────────────────────────────────────────────────────────────────────
# SETUP: Load model and data
# ─────────────────────────────────────────────────────────────────────
print("\n[1] Loading model and data...")

BASE = '/mnt/polished-lake/artifacts/fellows-shared/life-sciences/state/models/ST-HVG-Replogle/fewshot/hepg2'

from state.tx.models.state_transition import StateTransitionPerturbationModel

model = StateTransitionPerturbationModel.load_from_checkpoint(
    f'{BASE}/checkpoints/best.ckpt',
    map_location='cuda' if torch.cuda.is_available() else 'cpu'
)
model.eval()
device = next(model.parameters()).device
print(f"Model loaded on: {device}")

var_dims = pickle.load(open(f'{BASE}/var_dims.pkl', 'rb'))
gene_names = var_dims['gene_names']
gene_name_list = list(gene_names)
print(f"Gene names: {len(gene_name_list)} HVGs")

pert_map = torch.load(f'{BASE}/pert_onehot_map.pt', weights_only=False)
# Normalise keys to str
pert_map = {str(k): v for k, v in pert_map.items()}
print(f"Perturbations in pert_map: {len(pert_map)}")

print("\nModel top-level modules:")
for name, mod in model.named_children():
    print(f"  {name}: {type(mod).__name__}")
print(f"pert_dim: {model.pert_dim}")
print(f"hidden_dim (hparams): {model.hparams.get('hidden_dim', 'N/A')}")

# Load HepG2 control cells
print("\nLoading HepG2 control cells...")
adata = ad.read_h5ad('/mnt/polished-lake/artifacts/fellows-shared/life-sciences/state/data/Replogle-Nadig-Preprint/replogle_matched_hvg.h5ad')
ctrl_mask = (adata.obs['cell_line'] == 'hepg2') & (adata.obs['gene'] == 'non-targeting')
ctrl_adata = adata[ctrl_mask]
print(f"HepG2 control cells: {ctrl_adata.n_obs}")

ctrl_mat = ctrl_adata.obsm['X_hvg']  # [N, 2000]
if hasattr(ctrl_mat, 'toarray'):
    ctrl_mat = ctrl_mat.toarray()
print(f"ctrl_mat shape: {ctrl_mat.shape}")

np.random.seed(42)
CELL_SET_LEN = 64
idx = np.random.choice(len(ctrl_mat), size=CELL_SET_LEN, replace=False)
ctrl_cells = torch.tensor(ctrl_mat[idx], dtype=torch.float32).to(device)
print(f"ctrl_cells tensor: {ctrl_cells.shape}")

# Batch index for HepG2 (using 0 as default, same pattern as prior notebooks)
BATCH_IDX = 0
batch_t = torch.full((CELL_SET_LEN,), BATCH_IDX, dtype=torch.long).to(device)

# ─────────────────────────────────────────────────────────────────────
# Helper: run model forward for a given KD
# ─────────────────────────────────────────────────────────────────────
def run_model(pert_key, ctrl_cells_in=None, batch_idx_t=None):
    """Run model.predict_step for a given KD. Returns mean predicted output [2000]."""
    S = ctrl_cells_in.shape[0] if ctrl_cells_in is not None else CELL_SET_LEN
    cells = ctrl_cells if ctrl_cells_in is None else ctrl_cells_in
    bi = batch_t if batch_idx_t is None else batch_idx_t
    pert_oh = pert_map[pert_key].float().to(device)
    pert_exp = pert_oh.unsqueeze(0).expand(S, -1)
    batch_dict = {
        'ctrl_cell_emb': cells,
        'pert_emb': pert_exp,
        'pert_name': [pert_key] * S,
        'batch': bi,
    }
    with torch.no_grad():
        out = model.predict_step(batch_dict, batch_idx=0, padded=False)
    preds = out['preds'].reshape(-1, out['preds'].shape[-1])  # [S, 2000]
    return preds.mean(0).cpu().numpy()

# ─────────────────────────────────────────────────────────────────────
# PART 1: Compute ΔY for all KDs
# ─────────────────────────────────────────────────────────────────────
print("\n[PART 1] Computing ΔY for all KDs...")

# Ctrl baseline
ctrl_mean = run_model('non-targeting')
print(f"ctrl_mean shape: {ctrl_mean.shape}, range: [{ctrl_mean.min():.3f}, {ctrl_mean.max():.3f}]")

# Compute ΔY for all KDs
ctrl_keys = {'non-targeting', 'NT', 'Non-targeting', 'non_targeting'}
all_perts = [(k, v) for k, v in pert_map.items() if k not in ctrl_keys]
print(f"KDs to process: {len(all_perts)}")

delta_y = {}
for i, (pert_name, _) in enumerate(all_perts):
    pert_mean = run_model(pert_name)
    delta_y[pert_name] = pert_mean - ctrl_mean
    if (i+1) % 100 == 0:
        print(f"  Processed {i+1}/{len(all_perts)} KDs...")

delta_y_df = pd.DataFrame(delta_y).T  # [N_perts, 2000]
delta_y_df.columns = gene_name_list
print(f"\ndelta_y_df shape: {delta_y_df.shape}")
print(f"Mean |ΔY|: {delta_y_df.abs().values.mean():.4f}")

delta_y_df.to_parquet(f'{OUTDIR}/nb05_delta_y.parquet')
print("Saved nb05_delta_y.parquet")

# ─────────────────────────────────────────────────────────────────────
# Hierarchical clustering of ΔY
# ─────────────────────────────────────────────────────────────────────
print("\n[1b] Hierarchical clustering of ΔY vectors...")

dy_mat = delta_y_df.values.astype(np.float32)  # [N_perts, 2000]
pert_names_arr = np.array(delta_y_df.index.tolist())
N_perts = len(pert_names_arr)

# Cosine distance
cos_dist = pdist(dy_mat, metric='cosine')
Z = linkage(cos_dist, method='ward')

n_clusters = 15
cluster_labels = fcluster(Z, n_clusters, criterion='maxclust')
print(f"Cluster sizes (n={n_clusters}): {np.unique(cluster_labels, return_counts=True)[1].tolist()}")

cluster_df = pd.DataFrame({'pert_name': pert_names_arr, 'cluster': cluster_labels})
cluster_df.to_csv(f'{OUTDIR}/nb05_clusters.csv', index=False)

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
np.save(f'{OUTDIR}/nb05_umap_emb.npy', umap_emb)

# ─────────────────────────────────────────────────────────────────────
# Pathway annotation
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
print("\nCluster pathway score matrix:")
print(pathway_score_df.round(3))

# Print top ΔY genes per cluster and top pathway
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
        'mean_abs_dy': np.abs(mean_dy).mean(),
        'n': int(mask.sum()),
    }

print("\nCluster summary:")
for c in sorted(cluster_top_pathway.keys()):
    info = cluster_top_genes[c]
    print(f"  C{c:2d} (n={info['n']:3d}, mean|ΔY|={info['mean_abs_dy']:.4f}): "
          f"top={cluster_top_pathway[c]}, up={info['top_up'][:3]}")

# ─────────────────────────────────────────────────────────────────────
# Try gseapy enrichr for top clusters
# ─────────────────────────────────────────────────────────────────────
enrichr_results = {}
try:
    import gseapy
    print("\nRunning gseapy enrichr for each cluster (top 50 up-regulated genes)...")
    for c in range(1, n_clusters+1):
        mask = cluster_labels == c
        cluster_dy = dy_mat[mask].mean(0)
        top_idx = np.argsort(cluster_dy)[-50:][::-1]
        top_genes = [gene_name_list[i] for i in top_idx]
        try:
            enr = gseapy.enrichr(
                gene_list=top_genes,
                gene_sets='KEGG_2021_Human',
                organism='Human',
                outdir=None,
                verbose=False
            )
            if enr.results is not None and len(enr.results) > 0:
                top5 = enr.results.sort_values('Adjusted P-value').head(5)
                enrichr_results[c] = top5[['Term', 'Adjusted P-value', 'Overlap']].to_dict('records')
                print(f"  C{c}: {top5['Term'].iloc[0]} (p={top5['Adjusted P-value'].iloc[0]:.2e})")
            else:
                enrichr_results[c] = []
        except Exception as e:
            enrichr_results[c] = []
    with open(f'{OUTDIR}/nb05_enrichr_results.json', 'w') as f:
        json.dump(enrichr_results, f, indent=2)
    print("gseapy enrichr done — saved nb05_enrichr_results.json")
except ImportError:
    print("  gseapy not available — using hardcoded pathway enrichment only")

# ─────────────────────────────────────────────────────────────────────
# Pathway color map
# ─────────────────────────────────────────────────────────────────────
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
# Figure 1: UMAP colored by cluster + pathway label
# ─────────────────────────────────────────────────────────────────────
print("\n[1e] Plotting UMAP...")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
cmap = plt.cm.get_cmap('tab20', n_clusters)

ax = axes[0]
for c in range(1, n_clusters+1):
    mask = cluster_labels == c
    ax.scatter(umap_emb[mask, 0], umap_emb[mask, 1],
               c=[cmap(c-1)], s=12, alpha=0.7, label=f'C{c} (n={mask.sum()})')
ax.set_title(f'ΔY {dim_method} — colored by cluster', fontsize=12)
ax.set_xlabel(f'{dim_method} 1'); ax.set_ylabel(f'{dim_method} 2')
ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=7, markerscale=1.5,
          ncol=1 if n_clusters <= 20 else 2)

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

ax2.set_title('ΔY — colored by top pathway', fontsize=12)
ax2.set_xlabel(f'{dim_method} 1'); ax2.set_ylabel(f'{dim_method} 2')
ax2.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9)

plt.tight_layout()
plt.savefig(f'{OUTDIR}/nb05_delta_y_umap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved nb05_delta_y_umap.png")

# ─────────────────────────────────────────────────────────────────────
# Figure 2: Pathway enrichment heatmap
# ─────────────────────────────────────────────────────────────────────
print("\n[1f] Plotting pathway enrichment...")

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

score_arr = pathway_score_df.values  # [n_clusters, n_pathways]
score_norm = score_arr / (score_arr.max(axis=0, keepdims=True) + 1e-8)

ax = axes[0]
im = ax.imshow(score_norm.T, aspect='auto', cmap='YlOrRd')
ax.set_xticks(range(n_clusters))
ax.set_xticklabels([f'C{c}' for c in range(1, n_clusters+1)], fontsize=8)
ax.set_yticks(range(len(pathway_patterns)))
ax.set_yticklabels(list(pathway_patterns.keys()), fontsize=10)
ax.set_xlabel('Cluster')
ax.set_title('Pathway enrichment score per cluster\n(normalized per pathway)', fontsize=11)
plt.colorbar(im, ax=ax, label='Normalized mean |ΔY|')

ax2 = axes[1]
sizes = [(cluster_labels == c).sum() for c in range(1, n_clusters+1)]
colors = [pathway_color_map.get(cluster_top_pathway[c], '#cccccc') for c in range(1, n_clusters+1)]
ax2.bar(range(1, n_clusters+1), sizes, color=colors, edgecolor='black', linewidth=0.5)
ax2.set_xlabel('Cluster')
ax2.set_ylabel('Number of KDs')
ax2.set_title('Cluster sizes (colored by top pathway)', fontsize=11)
ax2.set_xticks(range(1, n_clusters+1))
ax2.set_xticklabels([f'C{c}' for c in range(1, n_clusters+1)], fontsize=8)

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=v, edgecolor='black', label=k)
                   for k, v in pathway_color_map.items() if k in set(cluster_top_pathway.values())]
ax2.legend(handles=legend_elements, loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig(f'{OUTDIR}/nb05_delta_y_pathway_enrichment.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved nb05_delta_y_pathway_enrichment.png")

# ─────────────────────────────────────────────────────────────────────
# PART 2: Perturbation encoder Jacobian ∂Y/∂pert_repr
# ─────────────────────────────────────────────────────────────────────
print("\n[PART 2] Computing ∂Y/∂pert_repr Jacobians...")

# Determine pert_repr dimension (output of pert_encoder)
with torch.no_grad():
    test_oh = list(pert_map.values())[0].float().to(device)
    test_repr = model.pert_encoder(test_oh.unsqueeze(0))
    PERT_REPR_DIM = test_repr.shape[-1]
    print(f"pert_repr dim: {PERT_REPR_DIM}")

# Select representative KDs: top ~15 per cluster by mean |ΔY| (ensures coverage)
print("\n[2a] Selecting representative KDs (top 15 per cluster)...")
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

# If total KDs <= 600, use all of them
if N_perts <= 600:
    rep_kds = list(pert_names_arr)
    print(f"Using all {len(rep_kds)} KDs (total <= 600)")
print(f"Final KD set for Jacobians: {len(rep_kds)}")

# ─────────────────────────────────────────────────────────────────────
# Jacobian function: ∂mean_output / ∂pert_repr
# Strategy: get pert_repr, make differentiable, inject directly into
#           the model's internal computation path
# ─────────────────────────────────────────────────────────────────────

def get_pert_repr(pert_key):
    """Get the pert_encoder output for a given KD. Returns [1, PERT_REPR_DIM]."""
    oh = pert_map[pert_key].float().to(device)
    with torch.no_grad():
        pr = model.pert_encoder(oh.unsqueeze(0))  # [1, PERT_REPR_DIM]
    return pr.detach()

def forward_from_pert_repr(pert_repr_1d):
    """
    Forward pass from pert_repr [PERT_REPR_DIM] to mean output [2000].
    pert_repr_1d: [PERT_REPR_DIM] — treated as differentiable input.
    """
    S = ctrl_cells.shape[0]

    # Encode basal cells: [S, PERT_REPR_DIM]
    basal_enc = model.basal_encoder(ctrl_cells)  # [S, PERT_REPR_DIM]

    # pert_repr broadcast: [1, PERT_REPR_DIM] → [S, PERT_REPR_DIM]
    pert_repr_expanded = pert_repr_1d.unsqueeze(0).expand(S, -1)  # [S, H]

    combined = basal_enc + pert_repr_expanded  # [S, H]

    # Add batch encoding if batch_encoder exists
    if hasattr(model, 'batch_encoder') and model.batch_encoder is not None:
        batch_idx = torch.zeros(S, dtype=torch.long, device=device)
        batch_emb = model.batch_encoder(batch_idx)  # [S, H]
        combined = combined + batch_emb

    # Transformer backbone expects [B, S, H] but we have [S, H]
    # Reshape to [1, S, H]
    seq_input = combined.unsqueeze(0)  # [1, S, H]

    # Pass through transformer
    outputs = model.transformer_backbone(inputs_embeds=seq_input)
    transformer_out = outputs.last_hidden_state  # [1, S, H]

    # project_out
    res_pred = transformer_out.squeeze(0)  # [S, H]
    out_pred = model.project_out(res_pred)  # [S, 2000]
    out_pred = model.relu(out_pred)

    return out_pred.mean(0)  # [2000]


def compute_pert_jacobian_vectorized(pert_key):
    """
    Returns [2000, PERT_REPR_DIM] Jacobian using torch.autograd.functional.jacobian.
    """
    pr = get_pert_repr(pert_key).squeeze(0)  # [PERT_REPR_DIM]

    J = torch.autograd.functional.jacobian(
        forward_from_pert_repr,
        pr,
        vectorize=True,
        create_graph=False,
    )  # [2000, PERT_REPR_DIM]
    return J.detach().cpu().numpy()


def compute_pert_jacobian_manual(pert_key):
    """
    Fallback: manual backward loop over output genes.
    Returns [2000, PERT_REPR_DIM] Jacobian.
    """
    pr = get_pert_repr(pert_key).squeeze(0)  # [PERT_REPR_DIM]
    pr = pr.requires_grad_(True)

    mean_preds = forward_from_pert_repr(pr)  # [2000]

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
    print("Trying manual backward...")
    try:
        J_test = compute_pert_jacobian_manual(test_kd)
        print(f"Manual Jacobian OK: shape={J_test.shape}, range=[{J_test.min():.4f}, {J_test.max():.4f}]")
        jacobian_works = True
        use_vectorized = False
    except Exception as e2:
        print(f"Manual also failed: {e2}")
        jacobian_works = False

# ─────────────────────────────────────────────────────────────────────
# Compute Jacobians for representative KDs
# ─────────────────────────────────────────────────────────────────────
if jacobian_works:
    compute_fn = compute_pert_jacobian_vectorized if use_vectorized else compute_pert_jacobian_manual
    print(f"\n[2c] Computing Jacobians for {len(rep_kds)} KDs "
          f"({'vectorized' if use_vectorized else 'manual'})...")

    jacobians = {}
    failed_kds = []
    import time

    for i, kd_name in enumerate(rep_kds):
        t0 = time.time()
        try:
            J = compute_fn(kd_name)
            jacobians[kd_name] = J
        except Exception as e:
            failed_kds.append((kd_name, str(e)))

        if (i+1) % 25 == 0:
            dt = time.time() - t0
            print(f"  {i+1}/{len(rep_kds)} done, {len(failed_kds)} failed, last took {dt:.1f}s")

    print(f"\nJacobians computed: {len(jacobians)}, failed: {len(failed_kds)}")
    if failed_kds:
        print(f"  Failed examples: {failed_kds[:3]}")

    jac_kd_names = list(jacobians.keys())
    jac_mat = np.stack([jacobians[k] for k in jac_kd_names])  # [N, 2000, PERT_REPR_DIM]
    print(f"jac_mat shape: {jac_mat.shape}")

    np.save(f'{OUTDIR}/nb05_jacobians.npy', jac_mat)
    with open(f'{OUTDIR}/nb05_jacobian_kd_names.pkl', 'wb') as f:
        pickle.dump(jac_kd_names, f)
    print("Saved nb05_jacobians.npy and nb05_jacobian_kd_names.pkl")
else:
    print("Jacobian computation failed.")
    jac_kd_names = []
    jac_mat = None

# ─────────────────────────────────────────────────────────────────────
# PART 3: Does Jacobian similarity predict output similarity?
# ─────────────────────────────────────────────────────────────────────
print("\n[PART 3] Jacobian similarity vs output similarity...")

if jac_mat is not None and len(jac_kd_names) > 10:
    from sklearn.metrics.pairwise import cosine_similarity

    # Align ΔY and cluster labels to jac_kd_names
    pert_name_to_idx = {p: i for i, p in enumerate(pert_names_arr)}
    rep_dy_indices = [pert_name_to_idx[k] for k in jac_kd_names if k in pert_name_to_idx]
    jac_kd_names_aligned = [k for k in jac_kd_names if k in pert_name_to_idx]
    jac_mat_aligned = np.stack([jacobians[k] for k in jac_kd_names_aligned])

    rep_dy_mat = dy_mat[rep_dy_indices]
    rep_cluster_labels = cluster_labels[rep_dy_indices]
    N = len(jac_kd_names_aligned)
    print(f"Aligned KDs: {N}")

    # Flatten Jacobians: [N, 2000*PERT_REPR_DIM]
    jac_flat = jac_mat_aligned.reshape(N, -1).astype(np.float32)
    print(f"Flattened Jacobians: {jac_flat.shape}")

    # Pairwise cosine similarities
    print("Computing pairwise cosine similarities...")
    jac_cos_sim = cosine_similarity(jac_flat)   # [N, N]
    dy_cos_sim = cosine_similarity(rep_dy_mat)  # [N, N]

    print(f"Jacobian cosine sim: [{jac_cos_sim.min():.3f}, {jac_cos_sim.max():.3f}]")
    print(f"ΔY cosine sim: [{dy_cos_sim.min():.3f}, {dy_cos_sim.max():.3f}]")

    # Upper triangle pairs
    triu_idx = np.triu_indices(N, k=1)
    jac_pairs = jac_cos_sim[triu_idx]
    dy_pairs = dy_cos_sim[triu_idx]
    same_cluster = rep_cluster_labels[triu_idx[0]] == rep_cluster_labels[triu_idx[1]]

    print(f"\nTotal pairs: {len(jac_pairs)}")
    print(f"Same-cluster pairs: {same_cluster.sum()} ({same_cluster.mean():.1%})")

    # Pearson / Spearman
    r_pearson, p_pearson = pearsonr(dy_pairs, jac_pairs)
    r_spearman, p_spearman = spearmanr(dy_pairs, jac_pairs)
    print(f"\nJacobian sim ~ ΔY sim:")
    print(f"  Pearson r = {r_pearson:.4f}, p = {p_pearson:.2e}")
    print(f"  Spearman r = {r_spearman:.4f}, p = {p_spearman:.2e}")

    # Wilcoxon: same vs diff cluster Jacobian similarity
    same_jac = jac_pairs[same_cluster]
    diff_jac = jac_pairs[~same_cluster]

    np.random.seed(42)
    n_sample = min(len(same_jac), 5000)
    same_jac_s = np.random.choice(same_jac, size=n_sample, replace=False)
    diff_jac_s = np.random.choice(diff_jac, size=n_sample, replace=False)

    stat, p_wilcoxon = wilcoxon(same_jac_s, diff_jac_s)
    print(f"\nWilcoxon (same vs diff cluster Jacobian sim):")
    print(f"  Same-cluster: mean={same_jac.mean():.4f} ± {same_jac.std():.4f}")
    print(f"  Diff-cluster: mean={diff_jac.mean():.4f} ± {diff_jac.std():.4f}")
    print(f"  statistic={stat:.2f}, p={p_wilcoxon:.2e}")

    # ─────────────────────────────────────────────────────────────────────
    # SVD of stacked Jacobian matrix
    # ─────────────────────────────────────────────────────────────────────
    print("\n[3b] SVD of Jacobian matrix...")
    from sklearn.decomposition import TruncatedSVD

    n_components = min(10, N - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    kd_scores = svd.fit_transform(jac_flat)  # [N, n_components]
    explained_var = svd.explained_variance_ratio_
    print(f"SVD explained variance (top {n_components}): {explained_var.round(4).tolist()}")

    # Gene loadings per component
    n_genes_out = jac_mat_aligned.shape[1]  # 2000
    n_pert_dim = jac_mat_aligned.shape[2]   # PERT_REPR_DIM

    components_3d = svd.components_[:3].reshape(3, n_genes_out, n_pert_dim)
    gene_loadings = np.abs(components_3d).sum(axis=2)  # [3, 2000]

    top_gene_names_per_comp = {}
    for comp_i in range(3):
        top_idx = np.argsort(gene_loadings[comp_i])[-20:][::-1]
        top_gene_names_per_comp[comp_i+1] = [gene_name_list[j] for j in top_idx]
        print(f"\n  SVD component {comp_i+1} top genes: {top_gene_names_per_comp[comp_i+1][:10]}")

    # Top KDs per component
    for comp_i in range(3):
        top_kd_idx = np.argsort(np.abs(kd_scores[:, comp_i]))[-10:][::-1]
        top_kds = [(jac_kd_names_aligned[j], round(float(kd_scores[j, comp_i]), 3))
                   for j in top_kd_idx]
        print(f"  Component {comp_i+1} top KDs: {top_kds[:5]}")

    # ─────────────────────────────────────────────────────────────────────
    # Figure 3: Scatter + Wilcoxon violin
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

    # Trend line
    slope = r_pearson * jac_pairs.std() / (dy_pairs.std() + 1e-8)
    intercept = jac_pairs.mean() - slope * dy_pairs.mean()
    xfit = np.linspace(dy_pairs.min(), dy_pairs.max(), 100)
    ax.plot(xfit, slope * xfit + intercept, 'b-', linewidth=2)

    from matplotlib.patches import Patch
    handles = [Patch(facecolor='#e41a1c', label=f'Same cluster (n={same_cluster.sum()})'),
               Patch(facecolor='#aaaaaa', label=f'Diff cluster (n={(~same_cluster).sum()})')]
    ax.legend(handles=handles, fontsize=9)
    ax.set_xlabel('ΔY cosine similarity', fontsize=11)
    ax.set_ylabel('∂Y/∂pert_repr cosine similarity', fontsize=11)
    ax.set_title(f'Routing vs Output similarity\n'
                 f'Pearson r={r_pearson:.3f} (p={p_pearson:.1e}), '
                 f'Spearman r={r_spearman:.3f}', fontsize=10)

    # Wilcoxon violin
    ax2 = axes[1]
    n_violin = min(len(same_jac), len(diff_jac), 2000)
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
    ax2.set_xticklabels([f'Same cluster\n(ΔY, n={len(same_jac)})',
                          f'Diff cluster\n(ΔY, n={len(diff_jac)})'], fontsize=10)
    ax2.set_ylabel('∂Y/∂pert_repr cosine similarity', fontsize=10)
    ax2.set_title(f'Wilcoxon: same vs diff cluster routing\np = {p_wilcoxon:.2e}', fontsize=11)
    ax2.axhline(0, color='black', linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(f'{OUTDIR}/nb05_jacobian_vs_output_similarity.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved nb05_jacobian_vs_output_similarity.png")

    # ─────────────────────────────────────────────────────────────────────
    # Figure 4: SVD components
    # ─────────────────────────────────────────────────────────────────────
    print("\n[3d] Plotting SVD components...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    pathway_gene_color = {}
    for g in gene_name_list:
        if g.startswith('RPL') or g.startswith('RPS'):
            pathway_gene_color[g] = '#e41a1c'
        elif g.startswith('PSM'):
            pathway_gene_color[g] = '#377eb8'
        elif any(g.startswith(p) for p in ['SEC', 'SRP', 'TRAPPC', 'HSPA']):
            pathway_gene_color[g] = '#4daf4a'
        elif g.startswith('MT-') or g.startswith('MRPL') or g.startswith('MRPS'):
            pathway_gene_color[g] = '#984ea3'
        else:
            pathway_gene_color[g] = '#888888'

    for comp_i in range(3):
        # Top gene loadings
        ax = axes[0, comp_i]
        loadings = gene_loadings[comp_i]
        top_idx = np.argsort(loadings)[-20:][::-1]
        top_genes_plot = [gene_name_list[j] for j in top_idx]
        top_loadings = loadings[top_idx]
        bar_colors = [pathway_gene_color.get(g, '#888888') for g in top_genes_plot]

        ax.barh(range(len(top_genes_plot)), top_loadings[::-1],
                color=bar_colors[::-1])
        ax.set_yticks(range(len(top_genes_plot)))
        ax.set_yticklabels(top_genes_plot[::-1], fontsize=8)
        ax.set_xlabel('|Loading| (summed over pert_repr dims)')
        ax.set_title(f'SVD Component {comp_i+1} — top output genes\n'
                     f'({explained_var[comp_i]:.1%} var explained)')

        # Pathway legend
        from matplotlib.patches import Patch
        leg = [Patch(facecolor='#e41a1c', label='ribosome'),
               Patch(facecolor='#377eb8', label='proteasome'),
               Patch(facecolor='#4daf4a', label='ER/secretory'),
               Patch(facecolor='#984ea3', label='mito'),
               Patch(facecolor='#888888', label='other')]
        ax.legend(handles=leg, fontsize=7, loc='lower right')

        # KD scores on UMAP
        ax2 = axes[1, comp_i]
        rep_umap_idx = [pert_name_to_idx[k] for k in jac_kd_names_aligned]
        rep_umap = umap_emb[rep_umap_idx]
        scores_comp = kd_scores[:, comp_i]
        vmax = np.percentile(np.abs(scores_comp), 95)
        sc = ax2.scatter(rep_umap[:, 0], rep_umap[:, 1],
                         c=scores_comp, cmap='RdBu_r',
                         vmin=-vmax, vmax=vmax, s=15, alpha=0.85)
        plt.colorbar(sc, ax=ax2, label='SVD score')
        ax2.set_title(f'SVD Comp. {comp_i+1} scores on {dim_method}')
        ax2.set_xlabel(f'{dim_method} 1')
        ax2.set_ylabel(f'{dim_method} 2')

    plt.tight_layout()
    plt.savefig(f'{OUTDIR}/nb05_jacobian_svd.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved nb05_jacobian_svd.png")

    # ─────────────────────────────────────────────────────────────────────
    # Save summary
    # ─────────────────────────────────────────────────────────────────────
    summary = {
        'n_kds_total': int(N_perts),
        'n_clusters': n_clusters,
        'n_rep_kds_jacobian': int(N),
        'pearson_r_jac_dy': float(r_pearson),
        'pearson_p_jac_dy': float(p_pearson),
        'spearman_r_jac_dy': float(r_spearman),
        'spearman_p_jac_dy': float(p_spearman),
        'wilcoxon_p': float(p_wilcoxon),
        'same_cluster_jac_mean': float(same_jac.mean()),
        'diff_cluster_jac_mean': float(diff_jac.mean()),
        'jac_pert_repr_dim': int(PERT_REPR_DIM),
        'jac_output_dim': int(n_genes_out),
        'svd_explained_var': explained_var[:5].tolist(),
        'svd_top_genes': {str(k): v[:10] for k, v in top_gene_names_per_comp.items()},
        'cluster_top_pathway': {str(k): v for k, v in cluster_top_pathway.items()},
    }

    with open(f'{OUTDIR}/nb05_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("\nSaved nb05_summary.json")

    # ─────────────────────────────────────────────────────────────────────
    # Final interpretation
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("FINAL INTERPRETATION")
    print("="*60)
    print(f"\nPearson r (Jacobian sim ~ ΔY sim) = {r_pearson:.4f}")
    print(f"Spearman r = {r_spearman:.4f}")
    print(f"Same-cluster mean Jacobian sim = {same_jac.mean():.4f}")
    print(f"Diff-cluster mean Jacobian sim = {diff_jac.mean():.4f}")
    print(f"Wilcoxon p = {p_wilcoxon:.2e}")

    if r_pearson > 0.3 and p_wilcoxon < 0.05:
        verdict = "POSITIVE — Model B (internalized circuits)"
        interp = ("pert_encoder structure is causally active; "
                  "related KDs share transformer routing")
    elif r_pearson > 0.1 or p_wilcoxon < 0.05:
        verdict = "WEAK POSITIVE — partial circuit sharing"
        interp = "Some routing structure but not strongly predictive"
    else:
        verdict = "NEGATIVE — Model A (statistical compression)"
        interp = ("pert_encoder geometry is decorative; "
                  "transformer routes KDs idiosyncratically (fine-tuned memorization)")

    print(f"\nVERDICT: {verdict}")
    print(f"Interpretation: {interp}")

    summary['verdict'] = verdict
    summary['interpretation'] = interp
    with open(f'{OUTDIR}/nb05_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
else:
    print("Skipping Part 3 (insufficient Jacobians)")
    # Placeholder figures
    for figname in ['nb05_jacobian_vs_output_similarity.png', 'nb05_jacobian_svd.png']:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, 'Jacobian computation failed\nSee logs',
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        plt.savefig(f'{OUTDIR}/{figname}', dpi=100, bbox_inches='tight')
        plt.close()

print("\n=== NB05 COMPLETE ===")
