# Option 2 — Perturbation Representation Space

## Core idea

After `pert_encoder`, each KD's one-hot identity is mapped to a d_model=328 vector. Extract
these representations and ask: are mechanistically related KDs neighbors in perturbation
embedding space? This probes the model's internal encoding of perturbation identity before
any interaction with cell state.

This tests: **has the model learned to represent similar-mechanism KDs as similar vectors?**

## Why this is architecturally direct

The ST model processes perturbations as:

```
pert_onehot  →  pert_encoder (MLP)  →  328d pert_repr
                                              ↓
                               element-wise add with cell repr
                                              ↓
                               LLaMA transformer
```

The `pert_encoder` output is the model's learned "meaning" of each perturbation, independent
of any specific cell's state. If the model has learned that TRAPPC3 and SRP54 are both
ER/Golgi genes, they should be close in pert_encoder output space. If it has not, they will
be arbitrary neighbors determined by which index they happened to occupy in the one-hot.

This is a pre-cell-state probe — it isolates what the model "knows" about perturbation
identity before looking at any cells, and is free of the Jacobian confounds (no set-to-set
attention effects, no gradient approximation noise).

## What a positive result looks like

- Cosine similarity between pert_encoder outputs clusters by pathway (KEGG/Reactome overlap)
- Within-pathway pert_repr similarity > cross-pathway (Wilcoxon rank-sum, enrichment analysis)
- The clustering is consistent across fewshot/hepg2 and fewshot/jurkat models → not cell-line
  specific, backbone-level
- Known co-regulators (MYC/MAX, TRAPPC3/TRAPPC5) are nearest neighbors

## What a negative result looks like

- pert_encoder outputs are random with respect to biology — the model has not organized its
  perturbation representations by mechanism, only by prediction accuracy
- Clusters correspond to KD magnitude or essentiality rather than pathway membership
- HepG2 and Jurkat models have completely different perturbation clustering → fine-tune
  reorganizes perturbation representations

## Practical implementation

```python
import torch
import numpy as np
import pandas as pd
from state.tx.models.state_transition import StateTransitionPerturbationModel

# Load fewshot/hepg2 model
model = StateTransitionPerturbationModel.load_from_checkpoint(
    "/mnt/polished-lake/artifacts/fellows-shared/life-sciences/state/models/"
    "ST-HVG-Replogle/fewshot/hepg2/checkpoints/best.ckpt",
    map_location="cuda",
)
model.eval()

# Load perturbation one-hot map
pert_map = torch.load(
    "/mnt/polished-lake/artifacts/fellows-shared/life-sciences/state/models/"
    "ST-HVG-Replogle/fewshot/hepg2/pert_onehot_map.pt"
)

# Extract pert_encoder representations for all KDs
pert_reprs = {}
with torch.no_grad():
    for pert_name, onehot in pert_map.items():
        onehot_t = onehot.unsqueeze(0).cuda()  # [1, pert_dim]
        repr_vec = model.pert_encoder(onehot_t)  # [1, 328]
        pert_reprs[pert_name] = repr_vec.squeeze(0).cpu().numpy()

# Stack into matrix
pert_names = list(pert_reprs.keys())
repr_matrix = np.stack([pert_reprs[p] for p in pert_names])  # [N_perts, 328]

# Cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
sim = cosine_similarity(repr_matrix)  # [N_perts, N_perts]

# Cross-model comparison: repeat for fewshot/jurkat
# Spearman r between per-KD cosine similarity rows → does the model organize
# perturbations the same way across cell lines?
```

## Key comparison: fewshot vs zeroshot pert_encoder

Run the same extraction on zeroshot/hepg2. If the zeroshot backbone encodes mechanistic
similarity in pert_encoder space, that's strong evidence of backbone-level generalization.
If fewshot reorganizes that space substantially, fine-tuning is re-learning perturbation
representations from scratch.

This also explains the NB08 finding: MYC zeroshot r=0.355 vs TRAPPC3 r=0.001 could be
because in zeroshot, MYC's pert_encoder representation is close to its fewshot representation
(backbone learned MYC's mechanism from other cell lines), while TRAPPC3's pert_encoder
representation shifts substantially after fewshot fine-tuning.

## Connection to other analyses

- If pert_encoder clustering by pathway is high, this supports Hypothesis A (model learned
  generalizable regulatory logic). If it's low, supports Hypothesis B (statistical compression
  from training data only).
- The 211 conserved SAE features (NB09) should correspond to perturbations whose
  pert_encoder representations are also conserved across cell lines — a cross-level
  consistency check.
- Downstream: if pert_encoder representations cluster by pathway, you can use them as
  mechanistic similarity proxies for unseen perturbations (zero-shot mechanism transfer).
