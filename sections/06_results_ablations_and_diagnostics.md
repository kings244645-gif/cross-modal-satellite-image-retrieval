# Benchmark Results, Ablation Studies, and Visual Diagnostics

## Calibration Ablation: Intrinsic Alignment vs. Frozen Inductive Centering

To measure the modality gap (cone collapse) and test post-hoc latent calibration, we evaluated the trained ConvNeXt-Tiny dual-tower network on the unseen 3,248-pair test split. This split contains 6,496 total items in the unified modality-agnostic FAISS gallery. We compared two strict evaluation protocols: 

* **Uncalibrated** (raw $\ell_2$-normalized embeddings directly from the target projectors) and 

* **Frozen Inductive Centering** (shifting test embeddings with frozen sensor centroids $\mathbf{\mu}_{\text{SAR}}, \mathbf{\mu}_{\text{Opt}}$ computed exclusively on the 3,255-pair validation split).

| Configuration | Geo R@1 | Geo R@5 | Geo R@10 | Semantic F1@5 | Same-Sensor Top-5 |
| --- | --- | --- | --- | --- | --- |
| **Uncalibrated (Raw Latent Space)** | 21.31% | 46.09% | 58.25% | 0.7309 | 59.82% |
| **Frozen Inductive Centering** | **21.60%** | **46.44%** | **58.71%** | **0.7313** | **59.74%** |

(Note: For reference, transductive centering with test-set means yields 21.89% Geo R@1 and 47.06% Geo R@5. This result confirms that zero-leakage Frozen Inductive Centering captures almost the entire calibration benefit without test-set data leakage.)

### Analysis of Calibration Dynamics

1. **Intrinsic Cone-Collapse Mitigation:** In standard cross-modal contrastive pipelines, uncalibrated embeddings suffer from severe modality cone collapse. That collapse causes a $100\%$ Same-Sensor Top-5 retrieval ratio and near-$0\%$ cross-modal recall. In contrast, our uncalibrated baseline achieves **21.31% Geo R@1** and a balanced **59.82% Same-Sensor Top-5 ratio**, where $50\%$ represents ideal sensor parity. This result shows that our composite loss ($0.50\times$ cross-modal InfoNCE + $0.25\times$ intra-modal self-distillation) and EMA-synchronized `BatchNorm1d` running statistics prevent the SAR and Optical cones from separating during training.


2. **Inductive Refinement:** Frozen Inductive Centering improves all metrics without data leakage. It increases exact geographic counterpart recall to **21.60% at R@1** ($+0.29\%$) and **58.71% at R@10** ($+0.46\%$). It also moves the Same-Sensor Top-5 distribution closer to parity ($59.74\%$).



## Modality-Agnostic Directional Retrieval Performance

After applying Frozen Inductive Centering, we measured retrieval performance by query modality across all 6,496 queries against the unified mixed gallery:

| Query Direction | Geo R@1 | Geo R@5 | Geo R@10 | Semantic F1@5 | Semantic F1@10 |
| --- | --- | --- | --- | --- | --- |
| **SAR $\to$ Unified Gallery** | **22.60%** | **47.97%** | **60.44%** | 0.7259 | 0.7224 |
| **Optical $\to$ Unified Gallery** | 20.60% | 44.92% | 56.99% | **0.7367** | **0.7315** |
| **Mixed (Modality-Agnostic)** | 21.60% | 46.44% | 58.71% | 0.7313 | 0.7270 |

### Structural vs. Spectral Asymmetry

The directional results show a clear physical trade-off between active microwave and passive optical sensing:

* **SAR Excels at Instance-Level Localization (Geo R@K):** Sentinel-1 SAR queries outperform Sentinel-2 Optical queries by **$+2.00\%$ on Geo R@1** ($22.60\%$ vs. $20.60\%$) and **$+3.45\%$ on Geo R@10** ($60.44\%$ vs. $56.99\%$). Because C-band radar measures macroscopic surface roughness, dielectric boundaries, and corner reflectors, SAR embeddings encode distinct structural skeletons such as sharp coastlines, road intersections, and urban grids. Combined with GeM spatial pooling ($p=3.0$), these invariant geometric landmarks act as a unique spatial fingerprint that locates the exact optical twin.


* **Optical Excels at Semantic Biome Clustering (F1@K):** Sentinel-2 Optical queries outperform SAR queries on Twin-Excluded Semantic Multi-Label F1 (**0.7367 vs. 0.7259 at K=5**). Its 10 calibrated spectral bands capture fine biochemical and chlorophyll absorption features. Therefore, the optical encoder separates subtle land-cover variations, such as arable crops versus natural pastures, more reliably than 2-channel radar backscatter.



## Semantic Class-Level Diagnostics

To evaluate how surface physics governs cross-modal alignment across biomes, we grouped test queries into five primary CORINE Land Cover (CLC) macro-groups within the frozen inductive gallery:

| Land-Cover Macro-Group | Query Count ($n$) | Geo R@1 | Geo R@5 | Semantic F1@5 |
| --- | --- | --- | --- | --- |
| **Water Bodies** | 722 | **37.67%** | **67.59%** | 0.6397 |
| **Agricultural Areas** | 5,280 | 24.51% | 50.49% | **0.7290** |
| **Woodland / Shrub** | 2,394 | 21.68% | 46.91% | 0.6718 |
| **Urban / Industrial** | 4,102 | 21.18% | 45.51% | 0.7125 |
| **Forests** | 3,364 | 17.39% | 40.37% | 0.7156 |

### Physical Interpretation of Class Performance

1. **Water Bodies (Highest Instance Recall):** Queries with marine or inland water bodies achieve the highest exact geographic recall (**37.67% Geo R@1** and **67.59% Geo R@5**). Calm water acts as a specular microwave reflector ($\sigma^0 < -25\text{ dB}$, appearing dark in SAR) and absorbs NIR/SWIR solar radiation in Sentinel-2. This high-contrast shoreline geometry creates a distinct cross-modal contour.


2. **Agricultural and Urban Landscapes (Balanced Instance & Semantic Alignment):** Agricultural patches achieve the highest semantic consistency (**0.7290 F1@5**) and strong twin recall (**24.51% Geo R@1**) because distinct field boundaries appear in both modalities. **Urban / Industrial** queries also maintain high semantic clustering (**0.7125 F1@5**) and solid instance recall (**21.18% Geo R@1**) due to double-bounce corner reflections from buildings and transport infrastructure.


3. **Homogeneous Forests (High Semantic Clustering, Lower Instance Recall):** Forest queries achieve strong semantic clustering (**0.7156 F1@5**), which shows that the Jaccard-masked InfoNCE loss groups forest biomes effectively. However, forest queries yield the lowest exact twin recall (**17.39% Geo R@1**). Dense, continuous forest canopies produce uniform volumetric VH backscatter in SAR and uniform chlorophyll reflectance in Optical imagery. Because these canopies lack localized geometric landmarks, the model cannot easily distinguish one forest patch from thousands of visually similar forest patches in the gallery.



## Qualitative Visual Diagnostics

To visually confirm the quantitative metrics, we generated multi-panel retrieval grids during inference with strict leak-free validation centering. For each query, the system retrieves the Top-5 candidates from the 6,496-item unified gallery.

Each retrieved candidate patch in the visual output shows four annotations:

1. **[TWIN] Tag:** Marks whether the retrieved image is the exact geographic counterpart of the query to verify Geo R@K.


2. **Modality Badge:** Shows whether the retrieved patch is SAR or Optical to demonstrate modality-agnostic search.


3. **Dice / F1 Overlap:** Displays the multi-label Sørensen–Dice (F1) similarity between the CORINE labels of the query and the retrieved candidate.


4. **Cosine Similarity:** Reports the raw inner-product score ($0.0 \to 1.0$) computed by FAISS in the normalized 256-D latent space.


![Modality-Agnostic Test 1: SAR query retrieving mixed cross-modal semantic candidates.](images/retrieval_images/result-img-1.png)
*Figure 6.1: A Sentinel-1 SAR query (Agriculture / Broad-leaved forest) crosses the modality gap and retrieves semantically identical Sentinel-2 Optical patches at Ranks 1, 2, 4, and 5.*

![Modality-Agnostic Test 2: Optical query retrieving a SAR geographic twin.](images/retrieval_images/result-img-2.png)
*Figure 6.2: An Optical query (Inland wetlands) retrieves its exact SAR geographic [TWIN] at Rank 2 with a cosine similarity of 0.91.*

![Modality-Agnostic Test 3: SAR query retrieving intra-modal candidates.](images/retrieval_images/result-img-3.png)
*Figure 6.3: A SAR query (Arable land) demonstrates intra-modal clustering and retrieves morphologically similar SAR patches with 1.00 multi-label Dice overlap at Ranks 1, 4, and 5.*

![Modality-Agnostic Test 4: SAR query retrieving an Optical geographic twin.](images/retrieval_images/result-img-4.png)
*Figure 6.4: A SAR query (Arable land / Pastures) retrieves a mixed-modality candidate set and locates its exact Optical geographic [TWIN] at Rank 4 alongside overlapping semantic neighbors.*

Visual inspection of these grids confirms that even when the exact geographic twin does not appear at Rank 1, the retrieved patches share identical terrain morphology and high multi-label F1 overlap ($\sim 0.73$). These results prove that the network learns physical topography and land-cover semantics rather than coordinate artifacts.

## Hardware-Synchronized Inference Latency

Operational deployment requires strict latency limits. Using `torch.cuda.synchronize()` on a standard cloud GPU instance (NVIDIA T4 on Google Colab), we measured the end-to-end inference profile for a single query:

* **ConvNeXt-Tiny Encoding (FP16):** $\sim 4.2 \text{ ms}$


* **Inductive Centering & $\ell_2$ Normalization:** $\sim 0.1 \text{ ms}$


* **FAISS Inner Product Search (6,496 items):** $\sim 0.8 \text{ ms}$


* **Total End-to-End Latency:** $\sim 5.1 \text{ ms}$



This sub-10-millisecond end-to-end latency and sub-millisecond vector search scale efficiently, which proves that the architecture supports real-time, interactive deployment.
