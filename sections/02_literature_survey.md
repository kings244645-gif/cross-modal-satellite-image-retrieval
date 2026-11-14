# Literature Survey and Theoretical Foundations

## Taxonomy of Remote Sensing Image Retrieval

The rapid growth of Earth observation data has moved Remote Sensing Image Retrieval (RSIR) from simple metadata queries to Content-Based Remote Sensing Image Retrieval (CBRSIR). Recent reviews group modern RSIR methods into two main branches based on the link between the query and the database: uni-source retrieval and cross-source retrieval[@zhou2023].

![RSIR Hierarchical Taxonomy](./images/rsir_network.svg)

### Uni-Source Retrieval

In uni-source retrieval, the same sensor type collects both the query and gallery images[@zhou2023]. Label complexity divides this branch into two groups:

* **Single-Label RSIR (SLRSIR):** Each image has one dominant land-cover class[@zhou2023]. Benchmark datasets such as UC Merced, WHU-RS19, and PatternNet drive this domain[@zhou2023]. However, SLRSIR does not represent the mixed land cover found in real remote sensing scenes.


* **Multi-Label RSIR (MLRSIR):** Each image has multiple primitive labels, such as buildings, roads, and water[@zhou2023]. Algorithms in this group must model how these labels appear together. Datasets like DLRSD, WHDLD, and MLRSNet support this task[@zhou2023]. Even so, defining reliable similarity metrics for multi-label ranking remains an open research problem.



### Cross-Source Retrieval

Cross-source retrieval handles cases where the query and database images come from different physical domains, sensors, or viewpoints[@zhou2023].

* **Retrieval Across Different Sensors (RASRSIR):** This task uses images from the same modality but different sensors or resolutions. For example, a system retrieves high-resolution panchromatic images with a multispectral query[@zhou2023].


* **Cross-View RSIR (CVRSIR):** Also called geolocalization, this task matches images from very different viewpoints. For example, a system matches a ground-level panorama against an overhead drone or satellite database[@zhou2023].


* **Cross-Modal RSIR (CMRSIR):** This project focuses on CMRSIR, which bridges completely different data representations[@zhou2023]. CMRSIR includes sketch-to-image, text-to-image, audio-to-image, and Optical-to-SAR retrieval[@zhou2023]. To align active microwave (SAR) and passive optical imagery, the model must overcome a large physical modality gap because these sensors measure different Earth surface properties.



## Evolution of Deep Metric Learning

To bridge the modality gap, modern RSIR uses Deep Metric Learning (DML). DML learns a parameterized embedding space that places similar samples close together and pushes dissimilar samples apart.

### From Pairwise to Triplet Formulations

Early metric learning used the Siamese architecture and the **Contrastive Loss** (Hadsell et al., 2006)[@hadsell2006]. This loss reduces the Euclidean distance between positive pairs and penalizes negative pairs inside a set margin. Later work introduced the **Triplet Margin Loss** (Schroff et al., 2015)[@schroff2015] to optimize relative distances. It anchors a query image, pulls a positive sample closer, and pushes a negative sample away by a set margin. However, triplet loss has cubic sampling complexity ($O(N^3)$) and requires costly hard-negative mining.

### Information Noise Contrastive Estimation (InfoNCE)

The introduction of **InfoNCE** (Oord et al., 2018)[@oord2018] shifted the field toward maximizing mutual information between representations. Instead of comparing one positive and one negative sample, InfoNCE solves a multi-class classification task over a batch. It compares the query against one positive and $K$ negative samples at the same time. InfoNCE uses $\ell_2$-normalization to map vectors onto a hypersphere and scales similarities with a temperature parameter $\tau$. This design produces uniform and linearly separable feature spaces.

### Asymmetric Momentum Self-Distillation

In cross-modal tasks, rapid network updates make it hard to keep stable representation targets. Architectures such as **MoCo (Momentum Contrast)** (He et al., 2020)[@he2020] and **BYOL (Bootstrap Your Own Latent)** (Grill et al., 2020)[@grill2020] introduced the asymmetric dual-tower design. Backpropagation updates only the online network, while an Exponential Moving Average (EMA) of the online weights updates the target network. This mechanism provides a stable, slowly changing target for the contrastive loss when the model aligns different domains like SAR and Optical imagery.

## The Modality Gap and Cone Collapse in Cross-Modal Retrieval

When a model applies contrastive learning to dual-modality inputs—such as SAR in one encoder and Optical in another—it risks a geometric failure called the **modality gap** or **cone collapse** (Liang et al., 2022)[@liang2022].

Even when InfoNCE aligns paired SAR and Optical patches, different sensor data distributions split the hypersphere. The network maps all SAR embeddings into one narrow cone and all Optical embeddings into a separate cone. Although paired vectors stay relatively close across the gap, querying a SAR image against a mixed gallery retrieves only SAR images. This failure occurs because intra-modality similarity outweighs cross-modality similarity.

### Transductive vs. Inductive Calibration

Recent methods fix this gap by centering the embeddings. They subtract the mean of the SAR vectors and the mean of the Optical vectors. However, computing these centroids on the *test set* is transductive learning. That approach introduces severe data leakage and invalidates the evaluation.

An operationally sound system must use **Frozen Inductive Centering**. The pipeline computes calibration centroids ($\boldsymbol{\mu}_{\text{SAR}}$ and $\boldsymbol{\mu}_{\text{Opt}}$) exclusively on a held-out validation set. During inference, the system shifts incoming test queries with these frozen, pre-computed centroids. This method prevents data leakage, removes modality cone collapse, and enables true modality-agnostic unified retrieval.
