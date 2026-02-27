#!/usr/bin/env python
"""
NB03 Targeted Jacobian Analysis — Compute Jacobians (revised gene panels)

Gene panel design notes:
- TRAPPC3/TRAPPC5/SRP54/SEC24A are NOT in the 2000-gene HVG index (cannot be input regulators)
- But TRAPPC3/TRAPPC5/SRP54 ARE in the pert_map (can be KD conditions)
- For secretory pathway: use HSPA5/ATF4/XBP1/SEC24A as regulators and
  ARFGEF2/MFGE8/PDIA3/CALR/ERO1A as targets; test under TRAPPC3/TRAPPC5/SRP54/HSPA5
  KDs vs MYC/BRCA1/CHEK1 negative controls
- For ribosome: use RPL/RPS genes in both HVG and pert_map

Runs in conda env (Python 3.10 + pytorch/state).
Saves results to jacobian_analysis/nb03_*.parquet
"""
import sys
sys.path.insert(0, '/mnt/polished-lake/home/mbeheleramass/state/src')

import torch
import pickle
import numpy as np
import pandas as pd
import anndata as ad
import time

# ─── Paths ────────────────────────────────────────────────────────────────────
MODEL_BASE = '/mnt/polished-lake/artifacts/fellows-shared/life-sciences/state/models/ST-HVG-Replogle'
CKPT_PATH  = f'{MODEL_BASE}/fewshot/hepg2/checkpoints/best.ckpt'
VAR_DIMS   = f'{MODEL_BASE}/fewshot/hepg2/var_dims.pkl'
PERT_MAP   = f'{MODEL_BASE}/fewshot/hepg2/pert_onehot_map.pt'
DATA_PATH  = ('/mnt/polished-lake/artifacts/fellows-shared/life-sciences/state/data/'
              'Replogle-Nadig-Preprint/replogle_matched_hvg.h5ad')
OUT_DIR    = '/mnt/polished-lake/home/mbeheleramass/jacobian_analysis'

# ─── Device ───────────────────────────────────────────────────────────────────
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# ─── Load model ───────────────────────────────────────────────────────────────
from state.tx.models.state_transition import StateTransitionPerturbationModel
model = StateTransitionPerturbationModel.load_from_checkpoint(CKPT_PATH, map_location=device)
model.eval()
print("Model loaded.")

# ─── Load auxiliary data ──────────────────────────────────────────────────────
var_dims   = pickle.load(open(VAR_DIMS, 'rb'))
gene_names = var_dims['gene_names']   # list of 2000 gene names
gene2idx   = {g: i for i, g in enumerate(gene_names)}

pert_map   = torch.load(PERT_MAP, map_location=device, weights_only=False)
# Normalise keys to str
pert_map   = {str(k): v for k, v in pert_map.items()}
print(f"Gene index size: {len(gene_names)}, KDs available: {len(pert_map)}")

# ─── Load control cells ───────────────────────────────────────────────────────
adata      = ad.read_h5ad(DATA_PATH)
ctrl_mask  = (adata.obs['cell_line'] == 'hepg2') & (adata.obs['gene'] == 'non-targeting')
ctrl_mat   = adata.obsm['X_hvg'][ctrl_mask]   # [N, 2000]
print(f"Control cells: {ctrl_mat.shape}")

np.random.seed(42)
n_cells    = 64  # model cell_set_len
idx        = np.random.choice(len(ctrl_mat), size=n_cells, replace=False)
ctrl_cells = torch.tensor(ctrl_mat[idx], dtype=torch.float32).to(device)

# ─── Targeted Jacobian function ───────────────────────────────────────────────
def targeted_jacobian(model, ctrl_cells, pert_key, input_gene_idx, output_gene_idx):
    """
    Compute scalar dY[output] / dX[input] under given KD.
    Uses mean-cell expansion (spec note: "compute dY_i/dX_i while holding other cells fixed").
    Returns: scalar float
    """
    S = len(ctrl_cells)
    # Use mean control cell as representative
    X_mean = ctrl_cells.mean(0).detach().clone()   # [2000]
    X_mean.requires_grad_(True)

    # Expand to S-cell set (same cell repeated — off-diagonal gradients cancel)
    X_expanded = X_mean.unsqueeze(0).expand(S, -1)  # [S, 2000]

    pert_vec  = pert_map[pert_key].float().to(device)
    pert_exp  = pert_vec.unsqueeze(0).expand(S, -1)
    batch_t   = torch.zeros(S, dtype=torch.long, device=device)

    batch_dict = {
        'ctrl_cell_emb': X_expanded,
        'pert_emb':      pert_exp,
        'pert_name':     [pert_key] * S,
        'batch':         batch_t,
    }
    out    = model.predict_step(batch_dict, batch_idx=0, padded=False)
    preds  = out['preds'].reshape(-1, out['preds'].shape[-1])  # [S, 2000]
    target = preds[:, output_gene_idx].mean()
    target.backward()

    J = X_mean.grad[input_gene_idx].item()
    return J


# ─── Pathway panels (revised for HVG availability) ───────────────────────────
#
# SECRETORY PANEL
# Biological rationale: 
#   - TRAPPC3/TRAPPC5/SRP54 KDs disrupt ER/Golgi trafficking → expected to change
#     sensitivity of ER chaperone targets to ER stress sensors
#   - Regulators: HSPA5 (BiP/GRP78, master ER chaperone), ATF4 (UPR TF),
#     SEC24A (COPII coat component)
#   - Targets: ARFGEF2 (GBF1, Golgi tethering), MFGE8 (secretory protein),
#     PDIA3 (ER protein disulfide isomerase), CALR (calreticulin),
#     ERO1A (ER oxidoreductase)
#   - NOTE: TRAPPC3/TRAPPC5/SRP54 themselves are KD conditions, NOT input regulators
#     (they are not in the 2000-gene HVG index)
sec_regulators = [g for g in ['HSPA5', 'ATF4', 'XBP1', 'SEC24A'] if g in gene2idx]
sec_targets    = [g for g in ['ARFGEF2', 'MFGE8', 'PDIA3', 'CALR', 'ERO1A'] if g in gene2idx]
# Use TRAPPC3/TRAPPC5/SRP54 as KD conditions (they ARE in pert_map even though not in HVG)
sec_kds        = [g for g in ['TRAPPC3', 'TRAPPC5', 'SRP54', 'HSPA5'] if g in pert_map]
neg_ctrl_kds   = [g for g in ['MYC', 'BRCA1', 'CHEK1'] if g in pert_map]

print(f"\nSecretory panel:")
print(f"  Regulators: {sec_regulators}")
print(f"  Targets:    {sec_targets}")
print(f"  Pathway KDs: {sec_kds}")
print(f"  Neg ctrl KDs: {neg_ctrl_kds}")

#
# RIBOSOME PANEL
# Biological rationale:
#   - RPL/RPS KDs disrupt ribosome assembly → expected to change co-sensitivity
#     between ribosomal protein expression levels
#   - Regulators (in HVG): RPL10, RPS6, RPL12, RPS8 (core ribosomal proteins)
#   - Targets (different set): RPL13, RPL14, RPS10, RPS18, RPS27 
#   - Pathway KDs = regulators themselves
ribo_regulators = [g for g in ['RPL10', 'RPS6', 'RPL12', 'RPS8'] if g in gene2idx]
ribo_targets    = [g for g in ['RPL13', 'RPL14', 'RPS10', 'RPS18', 'RPS27'] if g in gene2idx]
ribo_kds        = [g for g in ['RPL10', 'RPS6', 'RPL12', 'RPS8'] if g in pert_map]

print(f"\nRibosome panel:")
print(f"  Regulators: {ribo_regulators}")
print(f"  Targets:    {ribo_targets}")
print(f"  Pathway KDs: {ribo_kds}")
print(f"  Neg ctrl KDs: {neg_ctrl_kds}")


# ─── Compute Jacobians for both panels ────────────────────────────────────────
def compute_panel(regulators, targets, pathway_kds, neg_kds, pathway_name):
    records = []
    all_kds = pathway_kds + neg_kds
    total   = len(regulators) * len(targets) * len(all_kds)
    done    = 0
    for reg in regulators:
        reg_idx = gene2idx[reg]
        for tgt in targets:
            tgt_idx = gene2idx[tgt]
            for kd in all_kds:
                t0 = time.time()
                J  = targeted_jacobian(model, ctrl_cells, kd, reg_idx, tgt_idx)
                dt = time.time() - t0
                done += 1
                records.append({
                    'pathway':    pathway_name,
                    'regulator':  reg,
                    'target':     tgt,
                    'kd':         kd,
                    'pathway_kd': kd in pathway_kds,
                    'J':          J,
                })
                print(f"  [{pathway_name}] {done}/{total} | {reg}->{tgt} | KD={kd} | J={J:.6f} | {dt:.1f}s")
    return records

print("\n=== Computing SECRETORY panel ===")
sec_records = compute_panel(sec_regulators, sec_targets, sec_kds, neg_ctrl_kds, 'secretory')

print("\n=== Computing RIBOSOME panel ===")
ribo_records = compute_panel(ribo_regulators, ribo_targets, ribo_kds, neg_ctrl_kds, 'ribosome')

# Save panel results
df = pd.DataFrame(sec_records + ribo_records)
df.to_parquet(f'{OUT_DIR}/nb03_jacobian_results.parquet', index=False)
print(f"\nSaved {len(df)} rows to nb03_jacobian_results.parquet")
print(df.groupby(['pathway','pathway_kd'])['J'].describe().round(6).to_string())


# ─── Distribution: ALL KDs for HSPA5 -> ARFGEF2 ─────────────────────────────
print("\n=== Computing J distribution across ALL KDs: HSPA5 -> ARFGEF2 ===")

reg_gene = 'HSPA5'
tgt_gene = 'ARFGEF2'

if reg_gene in gene2idx and tgt_gene in gene2idx:
    reg_idx = gene2idx[reg_gene]
    tgt_idx = gene2idx[tgt_gene]

    all_kd_keys = sorted(pert_map.keys())
    dist_records = []
    for i, kd in enumerate(all_kd_keys):
        J = targeted_jacobian(model, ctrl_cells, kd, reg_idx, tgt_idx)
        dist_records.append({'kd': str(kd), 'J': J})
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(all_kd_keys)} KDs | last J={J:.6f}")

    df_dist = pd.DataFrame(dist_records)

    # Add pathway labels
    df_dist['is_secretory_kd'] = df_dist['kd'].isin(sec_kds)
    df_dist['is_neg_ctrl']     = df_dist['kd'].isin(neg_ctrl_kds)
    df_dist['is_ribo_kd']      = df_dist['kd'].isin(ribo_kds)

    df_dist.to_parquet(f'{OUT_DIR}/nb03_distribution_hspa5_arfgef2.parquet', index=False)
    print(f"Saved {len(df_dist)} rows to nb03_distribution_hspa5_arfgef2.parquet")

    # Quick stats
    print(f"\nJ statistics:")
    print(f"  All KDs:     mean={df_dist['J'].mean():.6f}, std={df_dist['J'].std():.6f}")
    print(f"  Sec KDs:     mean={df_dist[df_dist['is_secretory_kd']]['J'].mean():.6f}")
    print(f"  Neg KDs:     mean={df_dist[df_dist['is_neg_ctrl']]['J'].mean():.6f}")
    print(f"  Ribo KDs:    mean={df_dist[df_dist['is_ribo_kd']]['J'].mean():.6f}")

    # Rank of secretory KDs
    df_dist_sorted = df_dist.sort_values('J', ascending=False).reset_index(drop=True)
    for kd in sec_kds + neg_ctrl_kds:
        rank_row = df_dist_sorted[df_dist_sorted['kd'] == kd]
        if len(rank_row) > 0:
            rank = rank_row.index[0] + 1
            J_val = rank_row['J'].values[0]
            print(f"  {kd}: rank={rank}/{len(df_dist)}, J={J_val:.6f}")
else:
    print("Gene not in index, skipping")

print("\nDONE.")
