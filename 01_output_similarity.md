# Option 1 — Output Similarity Clustering

## Core idea

Compute the mean predicted response vector ΔY = E[Y_pert] − E[Y_ctrl] for all 1,035 shared
KDs, then cluster KDs by cosine similarity in the 2,000-gene output space. Compare clusters
to known pathway annotations (KEGG, Reactome, GO).

This directly tests: **does the model's output space encode mechanistic groupings?**

## Why this is the cleanest probe

The Jacobian approach (NB05) failed because Jacobian divergence was dominated by fine-tune
differences — MYC/MAX showed equally high ΔJ as TRAPPC3/5 despite having no shared biology.
Output clustering sidesteps this entirely. If the model has learned that ribosomal KDs all
produce similar downstream expression changes, those KDs will cluster together regardless of
which fine-tune epoch produced their representation. The signal is in the predictions, not the
gradients.

No architecture confounds: this works the same way regardless of set-to-set attention or
one-hot perturbation identity. The ΔY vector is interpretable in gene space.

## What a positive result looks like

- KDs of genes in the same KEGG pathway have higher pairwise cosine similarity than
  cross-pathway KDs (test: Wilcoxon rank-sum on within- vs across-pathway similarities)
- Hierarchical clustering or UMAP of the 1,035 × 2,000 ΔY matrix shows coherent pathway
  islands (ribosome, proteasome, splicing, mitochondria, etc.)
- The clustering structure is similar between HepG2 and Jurkat models → backbone-level
  grouping, not fine-tune artifact
- The clustering structure is similar between fewshot and zeroshot models → backbone-level
  generalization

## What a negative result looks like

- Clusters are dominated by KD magnitude (highly expressed KD targets cluster together
  regardless of pathway) rather than pathway identity
- Clusters differ substantially between fewshot and zeroshot → model has memorized KD
  responses rather than learned regulatory logic
- Known pathway members are spread across many clusters (low pathway coherence score)

## Practical implementation

```python
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, dendrogram

# Load pre-computed mean perturbation features
hepg2 = pd.read_parquet("pert_mean_features_hepg2.parquet")  # [1035, 2000]
jurkat = pd.read_parquet("pert_mean_features_jurkat.parquet")  # [1035, 2000]

# Compute ΔY = mean pert features - ctrl mean
# (pert_mean_features already contains per-perturbation means in HVG space)

# Cosine similarity matrix
sim_h = cosine_similarity(hepg2.values)  # [1035, 1035]
sim_j = cosine_similarity(jurkat.values)

# Pathway enrichment: for each pair of KDs in the same KEGG pathway,
# is their pairwise cosine similarity higher than background?

# UMAP for visualization
from umap import UMAP
reducer = UMAP(metric='cosine', n_neighbors=15, min_dist=0.1)
emb_h = reducer.fit_transform(hepg2.values)
```

## Key comparison

Run this for both fewshot/hepg2 and zeroshot/hepg2 ΔY vectors. If pathway clusters appear
in fewshot but not zeroshot, the grouping is fine-tune specific. If they appear in both,
the backbone has learned generalizable regulatory groupings.

## Connection to other analyses

- NB08 (zeroshot comparison) showed MYC has r=0.355 fewshot/zeroshot agreement while
  TRAPPC3 has r=0.001. Scaling this across all 1,035 KDs gives a per-perturbation
  "backbone generalizability score" — the distribution of this score across pathways
  tells you which biological programs the backbone has actually internalized.
- NB09 (conserved features) identified 211 SAE features with high HepG2/Jurkat Spearman r.
  The KDs that most activate conserved features should cluster coherently here.
