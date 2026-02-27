# Jacobian Analysis — Overview

## The Original Idea and Why It's Complicated

The starting idea: for each KD we can compute a Jacobian ∂E[Y_pert] / ∂E[X_ctrl] — how the
model's predicted output changes as a function of input control gene expression. If two
perturbations use similar biological mechanisms, their Jacobians should look similar; if the
model routes them differently, the Jacobians should diverge.

This is appealing because it treats the ST model as a gene regulatory network in disguise:
J_ij ≈ "how much does input gene i influence output gene j under perturbation k."

### Why it's harder than it looks

**The Jacobian is not a GRN.** In a true GRN, J_ij would mean "knocking down gene i causally
changes gene j." But X_ctrl is not a perturbation of gene i — it's the natural expression
state of a control cell. So ∂Y/∂X_ctrl captures how natural variation in the control cell's
expression state shifts the model's predicted KD response. That is real biology (state-
dependent perturbation effects), but it is not gene regulatory causality.

**The perturbation enters as a one-hot, not a mechanism.** The model sees TRAPPC3 as an index
in a learned embedding — not as a mechanistic description. Two KDs with similar biology can
have completely unrelated one-hot IDs. The Jacobian will differ not because the model routes
them differently biologically, but because the two pert_encoder offsets are different.

**The architecture is set-to-set.** The model takes S cells that all attend to each other
before making predictions. Cross-cell attention effects contaminate the per-cell gradient.
The "right" Jacobian would be per-cell-diagonal, but this is expensive and not what NB05
computed.

**The NB05 result confirmed the problem.** TRAPPC3/TRAPPC5/SRP54 (ER/Golgi) showed ~3.3×
higher cross-cell-line Jacobian divergence than neutral controls — but so did MYC, MAX, and
PAXBP1, which have no ER/Golgi biology. High Jacobian divergence reflected fine-tune
differences broadly, not pathway-specific routing.

---

## Four Alternative Probes

Rather than abandoning the question, we designed four probes that address the same underlying
question — does the model encode shared mechanistic structure for related KDs? — using
approaches better matched to the model architecture.

---

### Option 1 — Output Similarity Clustering
**File**: `01_output_similarity.md` | **Notebook**: `nb01_output_similarity.ipynb`

**Question**: Does the model's internal representation space cluster mechanistically related
KDs together?

**Approach**: Compute per-perturbation mean SAE feature vectors (layer 4 activations) for all
1,035 shared KDs in both HepG2 and Jurkat. Cluster by cosine similarity. Compare clusters to
known pathway memberships (ribosome, proteasome, splicing, ER/secretory, TFs).

**Why it works**: This bypasses gradient computation entirely. If the model has learned that
ribosomal KDs produce similar internal states, they will cluster regardless of whether fine-
tuning re-organized the Jacobian. The signal is in the representation, not the gradient.

**Key comparisons**:
- Within-pathway vs across-pathway cosine similarity (pathway coherence score)
- HepG2 vs Jurkat clustering consistency (backbone-level vs fine-tune)
- Fewshot vs zeroshot: does the backbone cluster without HepG2 fine-tuning?

---

### Option 2 — Perturbation Representation Space
**File**: `02_perturbation_embedding_space.md` | **Notebook**: `nb02_pert_embedding_space.ipynb`

**Question**: Has the model learned to represent similar-mechanism KDs as similar vectors,
before any interaction with cell state?

**Approach**: Extract `pert_encoder` outputs (328d) for every KD from both the HepG2 and
Jurkat models. These are the model's learned "meaning" of each perturbation identity, computed
before the transformer sees any cell expression. Cluster and measure pathway coherence.

**Why it's architecturally direct**: The `pert_encoder` output is the model's pre-cell-state
encoding of perturbation identity. If the model has internalized biology at training, related
perturbations should be neighbors here — independent of any specific cell's state, free of
set-to-set attention confounds, and not a gradient approximation.

**Key comparisons**:
- Nearest neighbors in pert_encoder space: are TRAPPC3's neighbors ER genes? MYC's neighbors TFs?
- Cross-model consistency: do HepG2 and Jurkat models encode the same perturbation neighborhoods?
- Pathway coherence: does within-pathway pert_repr similarity exceed random background?

---

### Option 3 — Targeted Jacobian with Known Regulatory Pairs
**File**: `03_targeted_jacobian.md` | **Notebook**: `nb03_targeted_jacobian.ipynb`

**Question**: For KDs within the same pathway, is ∂E[Y_target] / ∂X_regulator elevated
compared to cross-pathway controls?

**Approach**: Rather than comparing full 2000×2000 Jacobian matrices (which is dominated by
fine-tune noise), extract a scalar for a specific (input gene i, output gene j) pair under a
specific KD k. Test whether this scalar is elevated for pathway-concordant KDs vs controls.

**Example**: is ∂Y[SEC24A] / ∂X[TRAPPC3] higher under secretory pathway KDs (TRAPPC3, SRP54)
than under unrelated KDs (MYC, MAX)?

**Why this fixes the NB05 problem**: NB05 computed the norm of the full Jacobian difference
matrix, which aggregates signal across all 2000×2000 elements. Any fine-tune difference that
affects many elements (most of them) dominates. The targeted version asks a specific hypothesis-
driven question and has a well-defined null model.

**Key comparisons**:
- J(target | pathway KD, regulator) vs J(target | non-pathway KD, regulator)
- Secretory panel: (TRAPPC3/SRP54 regulator genes) → (SEC24A/ARFGEF2/MFGE8 target genes)
- Ribosome panel: (RPL5/RPS6 regulator genes) → (RPL24/RPS15 target genes)

---

### Option 4 — In-Silico KO Comparison
**File**: `04_insilico_ko.md` | **Notebook**: `nb04_insilico_ko.ipynb`

**Question**: When the model is causally deprived of a specific input gene's expression, does
the predicted output change in a pathway-coherent way — and consistently across cell lines?

**Approach**: Use pre-computed in-silico KO matrices (`insilico_ko_hepg2.npy`,
`insilico_ko_jurkat.npy`, each shape 200×2000): for 200 input genes, we have the predicted
output shift when that gene's expression is zeroed in the control cells. Compare these response
profiles cross-cell-line and against known pathway annotations.

**Why this is closer to a GRN than the Jacobian**: Setting X_i = 0 is a hard causal
intervention. It removes gene i's contribution to the model's input representation and forces
the model to predict without that signal. Unlike the Jacobian (sensitivity to natural variation),
this is an out-of-distribution intervention that more directly asks "what does the model think
gene i regulates?"

**Key comparisons**:
- Per-gene Spearman r between ko_h[i,:] and ko_j[i,:]: high r = backbone-level program,
  low r = cell-line specific (echoes NB08: MYC r=0.355, TRAPPC3 r=0.001)
- Within-pathway KO profile similarity vs random background
- UMAP of 200-gene KO response profiles, colored by pathway

---

## What a Positive Result Requires

Consistent with the dual-probe framework (project_overview.md), a single analysis passing is
insufficient. The thresholds for meaningful signal are:

| Probe | Metric | Threshold |
|---|---|---|
| Output / internal clustering | Within-pathway cosine sim > background | p < 0.05, effect size > 0.1 |
| Pert embedding coherence | Within-pathway NN similarity > random | p < 0.05 |
| Targeted Jacobian | Pathway KD J > non-pathway KD J | Wilcoxon p < 0.05 |
| In-silico KO cross-cell r | Mean within-pathway r | > 0.4 |

A perturbation program that passes Options 1+2 (representation-level) but fails Options 3+4
(causal/intervention-level) would suggest the model has organized its representations around
biology but does not causally route information through those circuits. The reverse would be
a different failure mode.

---

## Relationship to Prior Work

| Prior notebook | What it showed | How these build on it |
|---|---|---|
| NB05 (ΔJ divergence) | TRAPPC3/MYC both show high ΔJ — non-specific | Options 3+4 fix this with targeted/causal probes |
| NB07 (activation patching) | Feature 1069 sec_specific ≈ 0 after normalization | Option 3 tests input sensitivity rather than feature ablation |
| NB08 (zeroshot comparison) | MYC r=0.355, TRAPPC3 r=0.001 | Option 4 cross-cell-line KO r should mirror this pattern |
| NB09 (conserved features) | 211 SAE features with Spearman r > 0.5 | Option 1 and 2 should show coherent clusters for the KDs that drive those features |
