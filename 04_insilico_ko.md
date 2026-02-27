# Option 4 — In-Silico KO Comparison

## Core idea

You already have `insilico_ko_hepg2.npy` and `insilico_ko_jurkat.npy`. These are causal
interventions — specific input gene dimensions are zeroed out — which is much closer to a
true GRN probe than the Jacobian ∂Y/∂X_ctrl.

This tests: **when the model is causally deprived of a specific input gene's expression,
does the predicted output change in a pathway-coherent way? And does this differ across
cell lines in a biologically meaningful way?**

## Why this is architecturally closer to a GRN

The Jacobian ∂Y/∂X_ctrl measures sensitivity to natural variation in input expression.
In-silico KO sets X_i = 0 (or X_i = mean) for gene i and measures the resulting shift in
predicted output. The distinction matters because:

- Natural variation in X_i confounds regulatory signal with co-expression structure —
  genes that are co-expressed will have correlated Jacobian entries regardless of causality
- Setting X_i = 0 is a harder intervention: it removes gene i's contribution to the model's
  input representation, forcing the model to predict without that signal

The resulting matrix KO_ij = ΔY_j when X_i = 0 is interpretable as: "how much does gene j's
predicted KD response change when the model cannot see gene i's expression?" This is closer
to asking whether the model routes gene i's signal through to gene j's output.

## What the existing data contains

```
insilico_ko_hepg2.npy   — shape likely [N_genes, N_output_genes] or [N_genes, N_perts, N_output_genes]
insilico_ko_jurkat.npy  — same structure for Jurkat
insilico_ko_gene_idx.npy — gene indices that were knocked out
```

First step: read the actual shapes and confirm the axis interpretation.

```python
import numpy as np

ko_h = np.load("insilico_ko_hepg2.npy")
ko_j = np.load("insilico_ko_jurkat.npy")
gene_idx = np.load("insilico_ko_gene_idx.npy")

print(f"HepG2 KO shape: {ko_h.shape}")
print(f"Jurkat KO shape: {ko_j.shape}")
print(f"KO gene indices: {gene_idx.shape}, range [{gene_idx.min()}, {gene_idx.max()}]")
```

## Analysis plan once data structure is confirmed

### 4a. Per-gene KO response profiles
For each knocked-out input gene i, the KO response vector = change in predicted output.
This is the model's implicit "what does this gene regulate" answer.

Compare to known regulatory relationships: does zeroing out TRAPPC3 expression specifically
suppress secretory pathway output genes (SEC24A, ARFGEF2)? Does zeroing out MYC suppress
MYC target genes?

### 4b. Cross-cell-line comparison of KO response
Compute Spearman r between ko_h[i, :] and ko_j[i, :] for each knocked-out gene i.
- High r: the same input gene i influences the same output genes in both cell lines →
  backbone-level regulatory logic, not fine-tune artifact
- Low r: gene i's regulatory routing is cell-line specific → could be genuine cell-type
  specific regulation or fine-tune artifact (distinguish using zeroshot model)

This is the in-silico KO equivalent of the zeroshot/fewshot comparison from NB08.

### 4c. Pathway coherence test
Group knocked-out genes by KEGG pathway. Does within-pathway KO response similarity
exceed across-pathway?

```python
import pandas as pd
from scipy.stats import spearmanr

ko_h = np.load("insilico_ko_hepg2.npy")  # [N_ko_genes, N_output_genes]
ko_j = np.load("insilico_ko_jurkat.npy")

# Per-gene cross-cell-line correlation
cross_cl_r = np.array([
    spearmanr(ko_h[i], ko_j[i]).statistic
    for i in range(len(ko_h))
])

# Plot distribution: genes with high r are backbone-level regulators
# Genes with low r are cell-line-specific (may be fine-tune artifact)

# Pathway coherence: pairwise cosine similarity within vs across pathway
from sklearn.metrics.pairwise import cosine_similarity
sim_h = cosine_similarity(ko_h)  # [N_ko_genes, N_ko_genes]

# For each KEGG pathway, compute mean within-pathway similarity
```

### 4d. Comparison to output similarity (Option 1)
Options 1 and 4 are complementary but distinct:
- Option 1 asks: do similar-mechanism KDs produce similar predicted outputs?
- Option 4 asks: does zeroing out a specific input gene produce pathway-specific output changes?

If both show the same pathway groupings, it's convergent evidence. If they disagree, the
disagreement is informative: a pathway that clusters in Option 1 but not Option 4 might
mean the model learned output co-variation but not input sensitivity.

## Interpretation caveats

**Input zeros vs input mean.** Setting X_i = 0 is a strong intervention that may push the
model out of distribution (no cell naturally has zero expression of any gene). Setting
X_i = mean_ctrl is a softer intervention. The existing data uses one of these — check which.

**KD identity vs input KO.** The in-silico KO here is zeroing out input gene i's expression
in the control cell representation, while the perturbation identity (pert_onehot) is held
fixed at some KD (or non-targeting ctrl). This is different from running the model with
pert = TRAPPC3 KD. The question becomes: "given that the model is told perturbation X is
happening, how does removing input gene i's expression change its prediction?" which is a
state-conditioning question, not a regulatory one, unless pert = non-targeting ctrl.

Best use: run in-silico KO with pert = non-targeting ctrl. Then KO_ij = how much does
removing gene i from the input affect the model's baseline prediction for gene j. This is
a pure input-output gene routing question without any KD conditioning.

## Connection to other analyses

- This is the causal complement to the Jacobian: Jacobian = sensitivity (linear, local),
  in-silico KO = hard intervention (nonlinear, global)
- Cross-cell-line KO correlation directly parallels the NB08 zeroshot/fewshot comparison
  — both ask whether the same biological signal is present in both models
- If cross-cell-line KO correlation is high for ribosomal genes but low for secretory genes,
  this confirms the NB08 finding that universal programs generalize (MYC, ribosomal) but
  cell-type-specific programs do not (TRAPPC3, ER trafficking)
