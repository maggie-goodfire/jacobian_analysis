# Option 5 — Output Clustering + Perturbation Encoder Jacobian

## Motivation

Prior analyses revealed a key asymmetry:
- **NB02**: `pert_encoder` clusters by pathway geometry (TRAPPC3's NN is TRAPPC5, ribosomal
  genes form a tight island) — the model *represents* perturbation biology correctly
- **NB04**: In-silico KO mean cross-cell-line Spearman r = 0.057 — the model does NOT route
  input signal coherently across cell lines

The missing test: are the pert_encoder clusters *functionally active* in the forward pass?
Do KDs that cluster together in pert_repr space actually produce similar outputs AND route
through similar transformer circuits?

The earlier Jacobian analyses (NB05, NB03) used ∂Y/∂X_ctrl — the wrong variable.
**X_ctrl is a distribution of context, not the regulatory signal. The perturbation is.**

The correct Jacobian is ∂E[Y] / ∂pert_repr — how predicted output changes as a function of
the perturbation representation. This tells you which directions in the 328d perturbation
space drive which output genes, and whether mechanistically related KDs share that routing.

---

## Two-part analysis

### Part 1 — Output similarity clustering and pathway enrichment

Compute actual ΔY = E[Y_pert] − E[Y_ctrl] vectors in 2000-gene output space for all 1,035
KDs by running the fewshot/hepg2 model. Cluster by cosine similarity. Test pathway enrichment
within clusters.

This is distinct from NB01 which used SAE feature space (internal 2624d activations).
Here we use the actual predicted gene output — the external, interpretable signal.

### Part 2 — ∂Y/∂pert_repr Jacobian

For each KD, with pert_repr.requires_grad_(True), compute:

```
pert_repr = model.pert_encoder(pert_onehot)         # [1, 328]
cell_repr = model.basal_encoder(ctrl_cells)          # [S, 328]
combined  = cell_repr + pert_repr                    # [S, 328] (broadcast)
output    = model.project_out(
              model.transformer_backbone(combined))  # [S, 2000]

# J[:,k] = ∂mean_output[:,j] / ∂pert_repr[k]
# Full matrix: [2000 output genes × 328 pert dims] per KD
```

Use `torch.autograd.functional.jacobian` for efficiency.

### Part 3 — Does routing similarity predict output similarity?

Compute pairwise Jacobian cosine similarity (after flattening [2000,328] → [655360]) across
KDs. Correlate with pairwise output cosine similarity. Test: do same-cluster KDs (from Part 1)
share routing (higher Jacobian similarity)?

This directly distinguishes:
- **Model A (statistical compression)**: pert_encoder clusters by co-occurrence in training
  data, but transformer ignores that structure. Jacobian similarity ≈ uncorrelated with output
  similarity.
- **Model B (internalized circuits)**: pert_encoder clusters are used by transformer; same-
  cluster KDs route through same directions. Jacobian similarity correlates with output
  similarity.

---

## What a positive result looks like

- Same-cluster KDs (from output similarity) have higher pairwise Jacobian similarity than
  cross-cluster KDs (Wilcoxon, p < 0.05)
- Pearson r between per-pair output cosine similarity and Jacobian cosine similarity > 0.3
- SVD of the stacked Jacobian matrix reveals interpretable biological axes (e.g., PC1 loads
  on ribosomal output genes; KDs with high PC1 scores are ribosomal KDs)

## What a negative result looks like

- Jacobian similarity is orthogonal to output similarity: two KDs can produce similar outputs
  via completely different routing through pert_repr dimensions
- SVD axes don't align with biological pathway annotations
- Consistent with Model A: the model memorized per-KD responses during fine-tuning; the
  pert_encoder geometry is decorative, not mechanistically active

---

## Connection to prior work

| Prior finding | What this tests |
|---|---|
| NB02: pert_encoder clusters by pathway | Are those clusters functionally active (not just geometric)? |
| NB01: SAE feature clusters — mitoribosome PASS, ER/secretory FAIL | Do gene-output clusters match? Does routing also differ? |
| NB08: MYC generalizes (r=0.355), TRAPPC3 doesn't (r=0.001) | Do MYC and ribosomal KDs share routing directions? |
| NB04: in-silico KO mean r=0.057 | Confirms input routing is not backbone-level; does pert routing compensate? |
