# Neural Architecture and Mathematical Formulation

\begin{center}
\includegraphics[width=\textwidth,keepaspectratio]{images/training_pipeline_final.pdf}
\captionof{figure}{End-to-end asymmetric dual-tower training pipeline. Co-registered Sentinel-1 SAR and Sentinel-2 optical image pairs undergo synchronized $D_4$ dihedral augmentations. The system routes these pairs into online encoders ($\theta_q$) and momentum-updated target encoders ($\theta_k$, $m=0.996$). A composite multi-label Jaccard-masked InfoNCE loss ($\tau_{\text{IoU}}=0.40$) optimizes the latent projections on the 256-dimensional unit hypersphere. This loss executes cross-modal alignment and intra-modal self-distillation while it suppresses false negatives.}
\label{fig:training-pipeline}
\end{center}

## Sensor-Specific Stem Adaptation

The system uses ConvNeXt-Tiny[@liu2022] as the backbone for both the SAR and Optical branches to extract hierarchical features. The standard ConvNeXt architecture starts with a "patchify" stem. This stem is a convolutional layer with a $4 \times 4$ kernel and a stride of 4 (`Conv2d(3, 96, 4, 4)`), which ImageNet-1K pre-trained for 3-channel RGB inputs.

Remote sensing tensors use different channel depths ($C=2$ for SAR and $C=10$ for Optical). Therefore, the system must adapt the first convolutional layer to accept these modalities while it preserves the pre-trained variance and energy of the ImageNet weights $\mathbf{W}_{\text{RGB}} \in \mathbb{R}^{96 \times 3 \times 4 \times 4}$.

### SAR 2-Channel Initialization

For the 2-channel Sentinel-1 input ($\text{VV}, \text{VH}$), the system computes the mean across the 3 RGB channels of the pre-trained weights. It scales this mean to conserve the original energy and repeats it across the 2 SAR channels to create the new stem weights $\mathbf{W}_{\text{S1}}$:

$$\mathbf{W}_{\text{S1}} = \text{repeat}_2\left(\frac{1}{3}\sum_{c=1}^{3} \mathbf{W}_{\text{RGB}}[:, c, :, :]\right) \in \mathbb{R}^{96 \times 2 \times 4 \times 4}$$



### Optical 10-Channel Initialization

For the 10-channel Sentinel-2 input, the system tiles and scales the pre-trained weights. It tiles the original 3-channel weights to cover 10 dimensions and scales the output down by a factor of $10/3 \approx 3.33$. This step ensures that the sum of the activations matches the expected ImageNet pre-trained magnitude:

$$\mathbf{W}_{\text{S2}} = \frac{3}{10} \left(\text{repeat}_4(\mathbf{W}_{\text{RGB}})\right)[:, :10, :, :] \in \mathbb{R}^{96 \times 10 \times 4 \times 4}$$



\begin{figure}[p]
\centering
\begin{subfigure}[b]{0.48\textwidth}
\centering
\includegraphics[height=0.76\textheight,keepaspectratio]{images/sar_encoder_compact.pdf}
\caption{Sentinel-1 SAR Target Encoder (2-ch)}
\label{fig:sar-encoder}
\end{subfigure}\hfill
\begin{subfigure}[b]{0.48\textwidth}
\centering
\includegraphics[height=0.76\textheight,keepaspectratio]{images/opt_encoder_compact.pdf}
\caption{Sentinel-2 Optical Target Encoder (10-ch)}
\label{fig:opt-encoder}
\end{subfigure}
\caption{Computational graph topologies for the target encoders generated via \texttt{torchview} ($\text{depth}=2$). Both backbones map multi-channel rasters to a shared 256-d normalized latent space.}
\label{fig:target-encoders}
\end{figure}

## Generalized Mean (GeM) Spatial Pooling

Standard convolutional architectures usually end feature extraction with Global Average Pooling (GAP). In image retrieval, localized terrain features such as a small bridge or an isolated building are critical. GAP dilutes these strong localized activations into the background noise.

To prevent this signal loss, the system uses Generalized Mean (GeM) spatial pooling[@radenovic2018]. From the final convolutional feature map $\mathbf{F} \in \mathbb{R}^{768 \times H' \times W'}$, the system computes the GeM descriptor $\mathbf{g} \in \mathbb{R}^{768}$:

$$\mathbf{g} = \left( \frac{1}{H'W'} \sum_{u=1}^{H'} \sum_{v=1}^{W'} \max(\mathbf{F}_{:, u, v}, \epsilon)^p \right)^{\frac{1}{p}}$$


Here, $p$ is a learnable pooling parameter initialized at $p=3.0$, and $\epsilon=10^{-6}$ maintains numerical stability during backpropagation. The GeM equation provides a continuous generalization. It reduces to GAP as $p \to 1$, and it approaches Global Max Pooling (GMP) as $p \to \infty$. Because the network learns $p$, the system dynamically balances background context with salient object activations.

## Asymmetric Momentum Dual-Tower Dynamics

To align the heterogeneous SAR and Optical domains, the system uses an asymmetric online–target framework (`CrossModalSAROpticalNet`). This design follows self-supervised momentum architectures such as MoCo[@he2020] and BYOL[@grill2020].

### Online and Target Encoders

The architecture maintains two pairs of networks:

* **Online Networks ($\theta_q$):** These networks process heavily augmented views of the SAR and Optical batches $(\tilde{\mathbf{X}}_{\text{S1}}^{\text{aug}}, \tilde{\mathbf{X}}_{\text{S2}}^{\text{aug}})$. Standard gradient descent updates their parameters dynamically.


* **Target Networks ($\theta_k$):** These networks process the original, unaugmented views $(\tilde{\mathbf{X}}_{\text{S1}}^{\text{orig}}, \tilde{\mathbf{X}}_{\text{S2}}^{\text{orig}})$. They do not receive gradients. Instead, an Exponential Moving Average (EMA) of the online network parameters updates the target parameters:



$$\theta_k \leftarrow m\theta_k + (1 - m)\theta_q$$


where $m = 0.996$ is the momentum coefficient.

### Projection Head and Buffer Synchronization

| Stage / Component | Operation | Input Shape | Output Shape | Spatial / Feature Details |
| --- | --- | --- | --- | --- |
| **SAR Stem** | Conv2d ($4\times4, s=4$) | $(B, 2, 120, 120)$ | $(B, 96, 30, 30)$ | Patch projection (VV, VH)
| **Optical Stem** | Conv2d ($4\times4, s=4$) | $(B, 10, 120, 120)$ | $(B, 96, 30, 30)$ | Patch projection (10 MSI bands)
| **Stage 1** | ConvNeXt Block $\times 3$ | $(B, 96, 30, 30)$ | $(B, 96, 30, 30)$ | Depthwise $7\times 7$, GELU, LN
| **Downsample 1** | LayerNorm + Conv2d ($2\times 2, s=2$) | $(B, 96, 30, 30)$ | $(B, 192, 15, 15)$ | Spatial reduction
| **Stage 2** | ConvNeXt Block $\times 3$ | $(B, 192, 15, 15)$ | $(B, 192, 15, 15)$ | Mid-level spatial features
| **Downsample 2** | LayerNorm + Conv2d ($2\times 2, s=2$) | $(B, 192, 15, 15)$ | $(B, 384, 7, 7)$ | Spatial reduction
| **Stage 3** | ConvNeXt Block $\times 9$ | $(B, 384, 7, 7)$ | $(B, 384, 7, 7)$ | Core semantic feature extraction
| **Downsample 3** | LayerNorm + Conv2d ($2\times 2, s=2$) | $(B, 384, 7, 7)$ | $(B, 768, 3, 3)$ | Deep semantic bottleneck
| **Stage 4** | ConvNeXt Block $\times 3$ | $(B, 768, 3, 3)$ | $(B, 768, 3, 3)$ | Final deep spatial maps
| **GeM Pooling** | Generalized Mean ($p=3.0$) | $(B, 768, 3, 3)$ | $(B, 768, 1, 1)$ | Learnable spatial focus
| **Flatten** | Reshape | $(B, 768, 1, 1)$ | $(B, 768)$ | Feature vectorization
| **Projector** | Linear $\rightarrow$ BN $\rightarrow$ GELU $\rightarrow$ Dropout $\rightarrow$ Linear | $(B, 768)$ | $(B, 256)$ | Dimensionality compression
| **L2 Norm** | Normalization ($p=2$) | $(B, 256)$ | $(B, 256)$ | Unit hypersphere projection

After the GeM pooling layer, a two-layer Multi-Layer Perceptron (MLP) projects the features into a normalized latent space. This MLP follows the sequence $\text{Linear}(768 \to 512) \to \text{BatchNorm1d}(512) \to \text{GELU} \to \text{Dropout}(0.1) \to \text{Linear}(512 \to 256)$. The system then $\ell_2$-normalizes the final vector onto the hypersphere $\mathcal{S}^{255}$:

$$\mathbf{z} = \frac{\phi(\mathbf{g})}{\Vert{}\phi(\mathbf{g})\Vert{}_2} \in \mathbb{R}^{256}$$



To keep target representations stable over long training runs, the system must track both the convolutional weights and the floating-point running statistics of the `BatchNorm1d` layers. The system applies the EMA update directly to the batch normalization running mean and variance buffers:

$$\mathbf{\mu}_{\text{target}} \leftarrow m\mathbf{\mu}_{\text{target}} + (1 - m)\mathbf{\mu}_{\text{online}}, \quad \mathbf{\sigma}^2_{\text{target}} \leftarrow m\mathbf{\sigma}^2_{\text{target}} + (1 - m)\mathbf{\sigma}^2_{\text{online}}$$


If the system does not synchronize these buffers, severe representation drift occurs between the online and target spaces.

## Multi-Label Semantic InfoNCE Objective

Standard InfoNCE loss functions treat every non-matching sample in a mini-batch as a strict negative. In remote sensing, this rule creates a false-negative problem. For example, two distinct geographic patches in the same batch can both contain "Coniferous Forest" and "River". Pushing these semantically similar patches apart degrades the semantic manifold.

### Pairwise Jaccard (IoU) Masking

To add semantic awareness to the contrastive objective, the system uses CORINE land-cover multi-labels. For any two samples $i$ and $j$ in a mini-batch of size $B$ with label sets $Y_i$ and $Y_j$, the system calculates the pairwise Jaccard index (Intersection over Union):

$$\text{IoU}_{i,j} = \frac{\vert{}Y_i \cap Y_j\vert{}}{\vert{}Y_i \cup Y_j\vert{} + 10^{-8}}$$


The system builds a binary mask $\mathcal{M}$ to suppress false negatives. When the semantic overlap exceeds the threshold $\tau_{\text{IoU}} = 0.40$, the mask removes that pair from the contrastive denominator:

$$\mathcal{M}_{i,j} = \mathbb{I}(\text{IoU}_{i,j} > 0.40 \land i \neq j)$$



### Masked Symmetric InfoNCE

Let $\mathbf{u}_i$ represent the online embedding of sample $i$ (e.g., SAR), and let $\mathbf{v}_j$ represent the target embedding of sample $j$ (e.g., Optical). The system defines the masked logits as:

$$\tilde{\ell}_{i,j} = \begin{cases} -\infty \ (-10^9) & \text{if } \mathcal{M}_{i,j} = 1 \\ \frac{\mathbf{u}_i^\top \mathbf{v}_j}{\tau} & \text{otherwise} \end{cases}$$


where $\tau = 0.1$ is the temperature hyperparameter. The asymmetric masked InfoNCE loss for a batch is defined as:

$$\mathcal{L}_{\text{InfoNCE}}(\mathbf{U}, \mathbf{V}) = \frac{1}{2B} \sum_{i=1}^{B} \left[ \left( -\frac{\mathbf{u}_i^\top \mathbf{v}_i}{\tau} + \log \sum_{j=1}^{B} \exp(\tilde{\ell}_{i,j}) \right) + \left( -\frac{\mathbf{v}_i^\top \mathbf{u}_i}{\tau} + \log \sum_{j=1}^{B} \exp(\tilde{\ell}_{j,i}) \right) \right]$$



### Composite Objective

To balance cross-modal alignment (sensor-to-sensor) with intra-modal semantic clustering (sensor-to-itself), the system calculates the final loss as a weighted sum of four contrastive pairs:

$$\mathcal{L}_{\text{total}} = \frac{1}{2}\Big(\mathcal{L}_{\text{InfoNCE}}(\mathbf{Z}_{\text{S1}}, \mathbf{T}_{\text{S2}}) + \mathcal{L}_{\text{InfoNCE}}(\mathbf{Z}_{\text{S2}}, \mathbf{T}_{\text{S1}})\Big) + \frac{1}{4}\Big(\mathcal{L}_{\text{InfoNCE}}(\mathbf{Z}_{\text{S1}}, \mathbf{T}_{\text{S1}}) + \mathcal{L}_{\text{InfoNCE}}(\mathbf{Z}_{\text{S2}}, \mathbf{T}_{\text{S2}})\Big)$$

where $\mathbf{Z}$ denotes the online predictions and $\mathbf{T}$ denotes the target projections.
