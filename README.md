# Jacobian Analysis

Probes whether the ST perturbation model (ST-HVG-Replogle) encodes gene regulatory network structure, using four complementary approaches across HepG2 and Jurkat cell lines (1,035 shared knockdowns).

## Notebooks

| Notebook | What it does |
|---|---|
| `nb01_output_similarity.ipynb` | Clusters KDs by mean SAE layer-4 activations; tests pathway coherence via bootstrap |
| `nb02_pert_embedding_space.ipynb` | Extracts `pert_encoder` (328d) outputs; measures within-pathway cosine similarity and nearest-neighbor biology |
| `nb03_targeted_jacobian.ipynb` | Computes scalar `∂Y[target] / ∂X[regulator]` for known secretory and ribosome gene pairs under pathway-concordant vs control KDs |
| `nb04_insilico_ko.ipynb` | Compares in-silico KO response profiles (200 genes × 2000 outputs) across HepG2 and Jurkat via Spearman r |
| `nb05_pert_jacobian_output_clusters.ipynb` | Clusters KDs by ΔY output; computes `∂Y/∂pert_repr` Jacobians and tests routing coherence (fewshot model) |
| `nb06_zeroshot_pert_jacobian.ipynb` | Same as NB05 on zeroshot model to isolate backbone-level vs fine-tune effects |
| `nb07_spearman_visualization.ipynb` | Compares Pearson vs Spearman routing correlation metrics |
| `nb08_onehot_jacobian.ipynb` | Computes Jacobians with respect to the perturbation one-hot; compares fewshot vs zeroshot routing |
| `nb09_random_stratified_jacobian.ipynb` | Tests sensitivity of Jacobian estimates to random vs top-magnitude cell sampling strategies |
| `nb10_256cells_jacobian.ipynb` | Examines how Jacobian estimates scale with number of cells |

## Key Results

- **pert_encoder** clusters KDs by biological pathway (splicing 0.581, ribosome 0.497, ER/Golgi 0.364 mean within-pathway cosine sim vs 0.267 background)
- **Mitochondrial programs** show backbone-level output clustering in both cell lines (z > 2, p < 0.05)
- **ER/secretory programs** show no output clustering — HepG2-specific fine-tune artifact
- **In-silico KO** cross-cell-line Spearman r = 0.057 mean; no genes pass r > 0.4
- **∂Y/∂pert_repr** routing coherence r = 0.12 (fewshot and zeroshot alike) — backbone property but well below GRN threshold
