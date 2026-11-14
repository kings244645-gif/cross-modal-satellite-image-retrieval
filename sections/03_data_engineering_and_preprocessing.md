# Data Engineering, Radiometric Calibration & Pipeline Architecture

## Dataset Curation and Integrity Verification

This project uses the BigEarthNet-14K dataset from Kaggle (`narendraaironi/bigearthnet-14k`)[@sumbul2019]. The archive contains paired Sentinel-1 Synthetic Aperture Radar (SAR) and Sentinel-2 optical imagery[@sumbul2019]. To align the two modalities, the pipeline enforces a strict 1:1 filename mapping convention. This mapping links each Sentinel-1 TIFF basename (`s1_name`) to its corresponding Sentinel-2 TIFF basename (`patch_id`)[@sumbul2019].

A programmatic disk-mapping protocol verifies dataset integrity and produces a compiled `verified_paired_manifest.csv` file. The verified corpus contains 13,683 co-registered S1–S2 patch pairs. To prevent data leakage, the dataset partitions these pairs into static, non-overlapping geographic splits: 7,180 training pairs, 3,255 validation pairs, and 3,248 test pairs[@sumbul2019].

## Multi-Label Semantic Taxonomy

Unlike standard single-label benchmarks, BigEarthNet-14K annotations come from the CORINE Land Cover (CLC) database. This database provides a dense, multi-label semantic framework[@sumbul2019]. The metadata file (`metadata.parquet`) assigns one or more labels from a 43-class taxonomy to each image patch[@sumbul2019]. The retrieval pipeline parses these multi-label annotations to evaluate semantic similarity. This approach calculates continuous Jaccard (IoU) overlap between different geographic scenes instead of binary match or no-match criteria.

## Radiometric Calibration and Normalization

Raw geospatial rasters contain physical units that fall outside the optimal activation ranges of standard neural network backbones. The pipeline calibrates each modality independently before tensor fusion.

### Sentinel-1 SAR Calibration

The Sentinel-1 inputs form $2 \times 120 \times 120$ spatial tensors that represent the VV and VH polarizations. Raw backscatter intensity, measured in decibels (dB), shows a long-tailed distribution. The pipeline clips these values to physical lower and upper bounds of $-35.0\text{ dB}$ (noise floor over calm water) and $0.0\text{ dB}$ (strong corner reflectors). Next, linear min-max scaling projects the clipped data into the $[0, 1]$ range:

$$\tilde{\mathbf{X}}_{\text{S1}} = \frac{\text{clip}(\mathbf{X}_{\text{S1}}, -35.0, 0.0) + 35.0}{35.0} \in [0, 1]^{2 \times 120 \times 120}$$



### Sentinel-2 Optical Calibration

The Sentinel-2 inputs form $10 \times 120 \times 120$ tensors across the visible, near-infrared, and shortwave-infrared bands. Raw pixel values represent bottom-of-atmosphere surface reflectance scaled by a factor of 10,000. The pipeline divides the pixel values by this scale factor and clamps the upper bound to $1.0$ to handle specular anomalies or saturated cloud pixels:

$$\tilde{\mathbf{X}}_{\text{S2}} = \text{clip}\left(\frac{\mathbf{X}_{\text{S2}}}{10000.0}, 0.0, 1.0\right) \in [0, 1]^{10 \times 120 \times 120}$$



## Co-Registered Geometric Augmentations

During training, the pipeline applies synchronized geometric augmentations to expand the data distribution while it preserves pixel-level spatial alignment between the sensors. The pipeline uses the dihedral group $D_4$, which includes horizontal flips, vertical flips, and $90^\circ k$ rotations ($k \in \{0, 1, 2, 3\}$).

When the pipeline samples a random transformation $T \in D_4$ for a mini-batch index $i$, it applies $T$ identically to both modalities:

$$(\tilde{\mathbf{X}}_{\text{S1}, i}^{\text{aug}}, \tilde{\mathbf{X}}_{\text{S2}, i}^{\text{aug}}) = (T(\tilde{\mathbf{X}}_{\text{S1}, i}), T(\tilde{\mathbf{X}}_{\text{S2}, i}))$$


This synchronization ensures that structural geometry—such as a river orientation or an urban grid layout—matches across the active and passive sensor inputs.
