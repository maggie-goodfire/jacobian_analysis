# Option 3 — Targeted Jacobian with Known Regulatory Pairs

## Core idea

Rather than comparing full 2000×2000 Jacobian matrices across KDs (which is noisy and
dominated by fine-tune artifacts), ask scalar targeted questions: for KDs within the same
pathway, is ∂E[Y_target] / ∂X_regulator elevated compared to cross-pathway controls?

**Example**: for all secretory pathway KDs (TRAPPC3, TRAPPC5, SRP54, SEC24A...), is the
sensitivity of predicted secretory output genes (MFGE8, SEC24A, ARFGEF2) to variation in
input secretory regulator expression higher than the same sensitivity under unrelated KDs
(MYC, MAX)?

This converts the uninterpretable matrix comparison into a hypothesis-driven pathway test.

## Why this fixes the NB05 problem

NB05 computed `||ΔJ_h − ΔJ_j||` — the norm of the full Jacobian difference matrix. This
aggregates signal across all 2000×2000 elements. Any fine-tune difference that affects many
elements (which is most of them) will dominate.

The targeted version computes a scalar: J_ij = ∂Y_j / ∂X_i for a specific (i, j) gene pair
— one output gene j (pathway target) and one input gene i (pathway regulator). This is:

1. **Interpretable**: "does changing input expression of regulator gene i shift predicted
   expression of target gene j under this KD?"
2. **Pathway-specific**: you are testing a specific biological hypothesis about which input
   genes should sensitize which output genes under which KDs
3. **Comparable to null**: you can compare J_ij under pathway-relevant KDs vs pathway-
   irrelevant KDs — the null is well-defined

## What a positive result looks like

For a specific pathway (secretory, ribosomal, etc.):
- J(target_gene | pathway_KD, regulator_gene) >> J(target_gene | unrelated_KD, regulator_gene)
- The elevated sensitivity is consistent across multiple pathway members (not one KD outlier)
- The pattern holds in both fewshot and zeroshot models (backbone-level)
- The effect is specific: J(unrelated_target | pathway_KD, regulator_gene) is not elevated

## What a negative result looks like

- J_ij is the same regardless of which KD conditions it — the model's prediction for target
  gene j is equally sensitive to input regulator i whether or not the KD is pathway-relevant
- High J_ij for unrelated KDs (MYC) just as much as for secretory KDs → not pathway-specific
  routing, just high-variance input/output gene pairs dominating

## Practical implementation

```python
import torch
import numpy as np

# Single-pair Jacobian: ∂E[Y_j] / ∂X_i under KD p
# X is a single cell's HVG expression vector [1, 2000]
# Y is the predicted perturbed expression vector [1, 2000]

def compute_targeted_jacobian(model, ctrl_cells, pert_onehot, input_gene_idx, output_gene_idx):
    """
    Compute scalar ∂Y[output_gene_idx] / ∂X[input_gene_idx] for a given KD.

    ctrl_cells: [S, 2000] — the set of control cells (requires_grad on the gene of interest)
    pert_onehot: [S, pert_dim]
    """
    ctrl_cells = ctrl_cells.clone().detach().requires_grad_(True)  # [S, 2000]

    batch = {
        "ctrl_cell_emb": ctrl_cells,
        "pert_emb": pert_onehot,
    }
    output = model.predict_step(batch, batch_idx=0, padded=False)
    preds = output["preds"]  # [S, 2000]

    # Mean output for target gene across cells
    target = preds[:, output_gene_idx].mean()
    target.backward()

    # Gradient w.r.t. input gene, averaged across cells
    J_ij = ctrl_cells.grad[:, input_gene_idx].mean().item()
    return J_ij


# Define pathway gene panels (use model gene basis from var_dims.pkl)
import pickle
var_dims = pickle.load(open(
    "/mnt/polished-lake/artifacts/fellows-shared/life-sciences/state/models/"
    "ST-HVG-Replogle/fewshot/hepg2/var_dims.pkl", "rb"
))
gene_names = var_dims["gene_names"]  # authoritative HVG index basis

# Example: secretory pathway
secretory_regulators = ["TRAPPC3", "TRAPPC5", "SRP54"]
secretory_targets = ["SEC24A", "ARFGEF2", "MFGE8"]
unrelated_kds = ["MYC", "MAX", "PAXBP1"]
secretory_kds = ["TRAPPC3", "TRAPPC5", "SRP54"]

# For each (regulator, target, KD) triple, compute J_ij
# Compare distribution across pathway vs non-pathway KDs
results = []
for reg in secretory_regulators:
    reg_idx = gene_names.index(reg)
    for tgt in secretory_targets:
        tgt_idx = gene_names.index(tgt)
        for kd in secretory_kds + unrelated_kds:
            J = compute_targeted_jacobian(model, ctrl_cells, pert_map[kd], reg_idx, tgt_idx)
            results.append({
                "regulator": reg, "target": tgt, "kd": kd,
                "pathway_kd": kd in secretory_kds,
                "J": J
            })
```

## Scaling up: pathway panel test

For a proper statistical test:
1. Define N_pathway pathway (regulator, target) gene pairs from curated databases
2. Define N_null matched null pairs (similar expression, different pathway)
3. For each KD, compute mean J_pathway and mean J_null
4. Plot per-KD J_pathway / J_null ratio — does this ratio correlate with pathway membership?

This gives a per-KD "pathway routing score" that is directly interpretable and comparable
to the zeroshot/fewshot Pearson r from NB08.

## Important caveat: cell-set Jacobian vs single-cell Jacobian

The ST model takes S cells as a SET with bidirectional attention. When computing
∂Y_i / ∂X_i (same cell), you're getting the diagonal of the full cross-cell Jacobian.
When all cells in the set have similar expression (as in a ctrl batch), off-diagonal
cross-cell gradients may cancel, making the diagonal a reasonable approximation.

Best practice: compute ∂Y_i / ∂X_i while holding all other cells in the set fixed at
their actual expression values. Do not perturb all cells simultaneously.

## Connection to other analyses

- NB05 computed ΔJ as the full matrix difference — this option extracts the meaningful
  scalar from that noisy matrix by conditioning on known biology
- NB07 (activation patching) tested causal effect of SAE features on output genes — this
  tests causal sensitivity of input genes on output predictions
- Together these form a stronger probe: if both ∂Y_target/∂X_regulator (Option 3) and
  patching of the relevant SAE feature (NB07) are pathway-specific, you have convergent
  evidence of a real circuit
