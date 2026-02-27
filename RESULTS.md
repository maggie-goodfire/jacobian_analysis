# Jacobian Analysis — Results Summary

**Date**: 2026-02-27
**Model family**: ST-HVG-Replogle (Replogle-Nadig perturb-seq, 1,035 shared KDs, HepG2 + Jurkat)
**Core question**: Does the ST perturbation model encode gene regulatory network structure, or does it compress statistical co-variation from training data?

---

## Origin of the analysis

The starting idea was to use the Jacobian ∂E[Y_pert] / ∂E[X_ctrl] as a GRN probe: if two KDs use similar biological mechanisms, their Jacobians should look similar. This was already attempted in NB05 (perturbation mechanism matching) and produced an uninterpretable result — TRAPPC3/SRP54 (ER/Golgi) showed equally high cross-cell-line Jacobian divergence as MYC/MAX/PAXBP1, which have no ER biology.

**Why it failed**: X_ctrl is a distribution of control cell states, not a gene-regulatory input. ∂Y/∂X_ctrl captures how natural expression variation shifts predicted KD response — sensitivity to context, not causal regulatory flow. The full 2000×2000 Jacobian matrix comparison is dominated by fine-tune differences broadly, not pathway-specific routing.

**The fix**: shift the independent variable from X_ctrl to the perturbation itself. The perturbation one-hot is the actual causal intervention in this model. ∂Y/∂pert_repr is the more meaningful Jacobian.

---

## Notebooks

| Notebook | Analysis | Model |
|---|---|---|
| `nb01_output_similarity.ipynb` | SAE feature space clustering of 1035 KDs | fewshot/hepg2 |
| `nb02_pert_embedding_space.ipynb` | pert_encoder geometry + pathway coherence | fewshot/hepg2 + fewshot/jurkat |
| `nb03_targeted_jacobian.ipynb` | ∂Y[target] / ∂X[regulator] for secretory + ribosome panels | fewshot/hepg2 |
| `nb04_insilico_ko.ipynb` | Cross-cell-line KO response correlation (200 genes × 2000 outputs) | fewshot/hepg2 + fewshot/jurkat |
| `nb05_pert_jacobian_output_clusters.ipynb` | ΔY output clustering + ∂Y/∂pert_repr routing coherence | fewshot/hepg2 |
| `nb06_zeroshot_pert_jacobian.ipynb` | Same as NB05 on zeroshot model — backbone isolation | zeroshot/hepg2 |

---

## Results by notebook

### NB01 — Output Similarity Clustering (SAE feature space)

KDs clustered by per-perturbation mean SAE layer-4 feature activations (2624d). Pathway coherence tested via bootstrap (n=5000) within-pathway vs random cosine similarity.

| Pathway | HepG2 z | p | Jurkat z | p | Verdict |
|---|---|---|---|---|---|
| Mitoribosome | 5.58 | <0.001 | 5.71 | <0.001 | Backbone-level |
| OXPHOS / mito | 2.23 | 0.004 | 1.89 | 0.016 | Backbone-level |
| Cytoplasmic ribosome | 1.64 | 0.045 | 1.96 | 0.018 | Marginal |
| ER / Secretory | 0.19 | 0.45 | 0.49 | 0.33 | **Null** |
| Proteasome / Splicing | — | ns | — | ns | Null |

Cross-cell-line scatter: Spearman r = -0.41 between per-perturbation Cohen's d and cross-cell-line cosine similarity. SRP54 is among the lowest cross-cell-line similarities — consistent with NB08's TRAPPC3 zeroshot r=0.001.

**Conclusion**: mitochondrial programs are backbone-level; ER/secretory programs are cell-line-specific artifacts.

---

### NB02 — Perturbation Representation Space

`pert_encoder` is a single linear layer (Linear 2024→328) — a learned lookup table. Pathway coherence in pert_encoder output space:

| Pathway | Mean within-pathway sim | p |
|---|---|---|
| Splicing | 0.581 | 4e-25 |
| Ribosome | 0.497 | ~0 |
| ER/Golgi | 0.364 | 2e-7 |
| Proteasome | 0.348 | 2e-7 |
| Background mean | 0.267 | — |

Nearest neighbors: TRAPPC3 → TRAPPC5 (0.82), TRAPPC11, STX5 (Golgi SNARE), NSF. MYC → MAX (0.69). RPL5 → exclusively ribosomal proteins.

Cross-model (HepG2 vs Jurkat) pert_encoder similarity: mean 0.33. The model encodes perturbation pathway identity correctly but fine-tuning partially reorganizes the space.

**Conclusion**: strongest positive result. The model has genuinely learned biologically coherent perturbation representations. This is a necessary but not sufficient condition for GRN encoding.

---

### NB03 — Targeted Jacobian ∂Y[target] / ∂X[regulator]

Scalar Jacobians for specific known regulatory pairs under pathway-concordant vs negative-control KDs.

| Test | p | Fold | Verdict |
|---|---|---|---|
| Secretory panel (pathway vs neg ctrl) | 0.947 | 0.53× (reversed) | **Null** |
| Ribosome panel (pathway vs neg ctrl) | 0.559 | 0.94× | **Null** |
| CHEK1 rank for HSPA5→ARFGEF2 | 15/2024 (top 0.7%) | — | DNA damage gene outranks secretory KDs |
| SRP54 rank for HSPA5→ARFGEF2 | 1888/2024 (bottom 7%) | — | |

**Conclusion**: ∂Y/∂X_ctrl is not pathway-specific regardless of KD conditioning. Confirms the fundamental problem with the original Jacobian approach.

---

### NB04 — In-Silico KO Comparison

Pre-computed KO matrices (200 input genes × 2000 output genes) compared across HepG2 and Jurkat models.

- **Mean cross-cell-line Spearman r = 0.057**, median = 0.049
- Range: [−0.07, +0.26]. **Zero genes pass r > 0.4 backbone threshold.**
- 22/200 genes (11%) are actively anti-correlated — opposite routing directions across cell lines
- Divergence concentrates in lipid/ER metabolism outputs (DGAT1, CEPT1, TMX3) — biologically coherent with HepG2 hepatocyte identity
- No pathway achieves significant within-pathway KO profile coherence in HepG2

**Conclusion**: the model routes input signals in a cell-line-specific way throughout. This is not noise — it's real cell-type biology encoded by fine-tuning. But it means no backbone-level input→output regulatory routing.

---

### NB05 — ΔY Output Clustering + ∂Y/∂pert_repr (fewshot/hepg2)

**Part 1 — ΔY clustering (gene output space)**: 15 clusters from 2023 KDs. Key enriched clusters: C7 ribosome (p=9e-26), C12 ER protein processing (p=1e-9), C15 steroid biosynthesis (p=2e-5), C8 largest cluster (n=512, p53/stress).

**Part 2 — ∂Y/∂pert_repr Jacobians**: 225 KDs × [2000, 328] Jacobians via `torch.autograd.functional.jacobian`. SVD component 1 explains 62.8% of variance.

**Part 3 — Routing coherence**:
- Pearson r (Jacobian sim vs ΔY sim) = **0.119** (p=2e-80)
- Spearman r = 0.094 (p=4e-50)
- Same-cluster Jacobian similarity = **1.92×** diff-cluster (Wilcoxon p=1.2e-15)

**Conclusion**: weak positive. The model partially routes mechanistically related KDs through similar pert_repr directions, but the effect (r=0.12) is far below the r>0.4 threshold for internalized GRN circuits. Consistent with fine-tuned statistical compression rather than learned regulatory logic.

---

### NB06 — ΔY Clustering + ∂Y/∂pert_repr (zeroshot/hepg2) — backbone isolation

Same analysis as NB05 on the zeroshot model (trained on Jurkat/K562/RPE1 only — zero HepG2 perturbation data).

| Metric | Fewshot NB05 | Zeroshot NB06 | Interpretation |
|---|---|---|---|
| Pearson r (routing ~ output) | 0.119 | **0.131** | Backbone property — exists without HepG2 data |
| Spearman r | 0.094 (p=4e-50) | **-0.003 (p=0.66, NS)** | Rank-order signal is HepG2 fine-tune artifact |
| Same/diff-cluster Jacobian ratio | 1.92× | **2.99×** | Zeroshot clusters more tightly... |
| Cluster ARI (fewshot vs zeroshot) | — | 0.112 | ...but in completely different groupings |
| Per-KD ΔY mean correlation | — | 0.277 | Most KD predictions diverge between models |

**Key finding**: The r≈0.12 routing coherence is a backbone property — the zeroshot model achieves 0.131 without HepG2 data. But the Spearman collapse (0.094 → -0.003) reveals that the rank-order routing structure in fewshot is a HepG2 memorization artifact layered on top of backbone outlier effects.

**Biologically**:
- ER/secretory cluster absent from zeroshot — confirms Feature 1069 and secretory programs are HepG2 fine-tune memorization
- Top backbone-generalized KDs: RNA processing (SMU1, UPF1/2, SNRPD1), mitochondrial RNA (LRPPRC, POLRMT, MTPAP) — deeply conserved programs
- Most anti-correlated KDs: FOXO1 (liver insulin TF), FDPS (cholesterol biosynthesis) — hepatocyte-specific biology the backbone cannot generalize

---

## Overall conclusions

### What the model does encode (representation level)
The ST model has learned biologically coherent perturbation representations. The `pert_encoder` clusters KDs by pathway — TRAPPC3's nearest neighbor is TRAPPC5, ribosomal genes form a tight island, MYC and MAX co-locate. This structure exists in both fewshot and zeroshot models and is statistically significant across all major pathway groups (NB02).

### What the model does NOT encode (routing / causal level)
The pathway representations in `pert_encoder` are not translated into consistent causal routing circuits:
- ∂Y/∂X_ctrl is not pathway-specific (NB03)
- In-silico KO cross-cell-line r = 0.057 — no backbone input→output routing (NB04)
- ∂Y/∂pert_repr routing coherence r = 0.12 — statistically present but far below GRN threshold (NB05/06)
- The Spearman rank-order routing signal collapses in the zeroshot model (NB06)

### The backbone vs fine-tune decomposition
| Program type | Backbone (zeroshot) | Fine-tune adds |
|---|---|---|
| Mitoribosome / OXPHOS | Backbone-level clustering | Sharpens magnitude |
| RNA processing | Backbone-level generalization | Sharpens magnitude |
| Cytoplasmic ribosome | Marginal backbone signal | Sharpens specificity |
| ER / Secretory | **Absent from backbone** | Fully HepG2-memorized |
| Hepatocyte-specific (FOXO1, FDPS) | **Anti-correlated** | Fully HepG2-memorized |

### Final interpretation
The ST model is a high-quality perturbation response predictor whose `pert_encoder` has learned to organize perturbation identity by biological pathway. However, that organization is **representational, not mechanistic** — the transformer does not consistently route mechanistically related perturbations through shared circuits. The r=0.12 routing coherence reflects a backbone-level property (present in zeroshot) but the effect size is consistent with statistical compression of co-variation from training data, not internalized causal gene regulatory logic.

Universal programs (mitoribosome, RNA processing) show backbone-level evidence of generalization. Cell-type-specific programs (ER/secretory, hepatocyte metabolism) are fine-tune memorization artifacts with no backbone generalization and no cross-cell-line routing consistency.

> The fewshot fine-tune creates cell-line-specific ΔY clusters and SAE feature directions
> that were not present in the backbone. These encode real HepG2 biology but represent
> compressed statistical co-variation, not transferable regulatory circuits.
