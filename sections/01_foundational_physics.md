# Foundational Physics of Multi-Sensor Remote Sensing

## The Electromagnetic Spectrum and Earth Observation Regimes

Earth remote sensing collects and interprets electromagnetic radiation (EMR) that originates from or interacts with surface materials. The observed signal depends on three parameters: operational wavelength $\lambda$, frequency $\nu = c / \lambda$ (where $c \approx 3 \times 10^8 \text{ m/s}$ is the speed of light in vacuum), and active or passive sensing mode.

Modern spaceborne Earth observation systems operate across two distinct physical regimes:

1. **The Optical/Reflective Solar Regime:** This regime covers the visible (VIS, $0.4 - 0.7\,\mu\text{m}$), near-infrared (NIR, $0.7 - 1.1\,\mu\text{m}$), and shortwave-infrared (SWIR, $1.1 - 2.5\,\mu\text{m}$) spectra. Optical sensors measure passive solar radiance reflected from Earth surfaces.


2. **The Active Microwave Regime:** This regime covers centimeter-to-millimeter wavelengths ($1\,\text{mm} - 1\,\text{m}$, frequencies $0.3 - 300\,\text{GHz}$). Synthetic aperture radar (SAR) instruments transmit coherent microwave pulses and measure the amplitude and phase of the backscattered echo.



Optical sensors capture biochemical and electronic absorption features, whereas microwave sensors respond to geometric roughness, structural orientation, and complex dielectric permittivity. Because these sensors measure different physical properties, a large domain gap separates their raster data representations.

## Synthetic Aperture Radar (SAR) Physics: Sentinel-1

Sentinel-1 is a spaceborne C-band Synthetic Aperture Radar constellation. It operates at a center frequency $f_0 = 5.405\text{ GHz}$, which corresponds to a free-space wavelength:

$$\lambda = \frac{c}{f_0} \approx \frac{3 \times 10^8\text{ m/s}}{5.405 \times 10^9\text{ Hz}} \approx 5.55\text{ cm}$$



### Coherent Transmission and Synthetic Aperture Formulation

Diffraction limits the along-track (azimuth) spatial resolution $\delta_{\text{az}}$ of real-aperture radar antennas. For an antenna of physical length $L_a$ at orbital altitude $R$, the resolution is:

$$\delta_{\text{az, real}} = \frac{\lambda R}{L_a}$$


In low-Earth orbit ($R \approx 693\text{ km}$) at C-band, a $10\text{ m}$ azimuth resolution requires an antenna several kilometers long. SAR solves this problem by transmitting a continuous train of coherent frequency-modulated chirps as the satellite moves along its flight path. The processor synthesizes the Doppler phase history of the reflected echoes over the synthetic aperture length $L_{\text{syn}} = \frac{\lambda R}{L_a}$. As a result, the theoretical along-track resolution becomes independent of range:

$$\delta_{\text{az, SAR}} = \frac{L_a}{2}$$


Sentinel-1’s Interferometric Wide (IW) swath mode uses the Terrain Observation with Progressive Scans SAR (TOPSAR) technique. This mode produces a nominal ground resolution of $20\text{ m} \times 22\text{ m}$ in range and azimuth, which the system multi-looks and resamples to a $10\text{ m}$ pixel spacing.

### Dual Polarization and Scattering Matrix

Sentinel-1 operates in dual-polarization mode on transmit and receive. The antenna transmits linearly polarized electric fields horizontally ($\text{H}$) or vertically ($\text{V}$), and the receiver measures the backscattered wave in co-polarized and cross-polarized channels:

* $\mathbf{S}_{\text{VV}}$: Vertical transmit, Vertical receive.


* $\mathbf{S}_{\text{VH}}$: Vertical transmit, Horizontal receive.



The surface scattering matrix defines this scattering behavior:

$$\mathbf{S} = \begin{bmatrix} S_{\text{HH}} & S_{\text{HV}} \\ S_{\text{VH}} & S_{\text{VV}} \end{bmatrix}$$


The received backscatter intensity reflects three physical mechanisms:

* **Surface / Specular Scattering ($\text{VV}$ dominant):** Calm water bodies, paved runways, and smooth bare soils act as specular reflectors. They bounce the incident radar beam away from the sensor line-of-sight, so open water surfaces appear dark ($\sigma^0 < -25\text{ dB}$).


* **Volumetric Scattering ($\text{VH}$ dominant):** Dense vegetation canopies, agricultural crops, and forest branches cause multiple interior reflections and wave depolarization. The cross-polarized $\text{VH}$ signal provides a direct proxy for structural biomass, canopy volume, and leaf orientation.


* **Double-Bounce Corner Scattering:** Perpendicular geometries between vertical building walls and horizontal ground or asphalt create double-bounce corner reflectors. These structures return almost all pulse energy directly to the antenna, so urban fabrics and industrial ports appear as bright point and line clusters ($\sigma^0 > 0\text{ dB}$).



### Dielectric Permittivity and Soil Moisture Dynamics

The normalized radar cross-section $\sigma^0$ depends directly on Fresnel reflection coefficients. These coefficients depend on the complex relative dielectric permittivity of the target material:

$$\varepsilon_r = \varepsilon' - j\varepsilon''$$


Dry soil has a dielectric constant $\varepsilon_r \approx 3 - 5$, whereas liquid water has $\varepsilon_r \approx 80$ at C-band frequencies. When soil moisture increases, the dielectric permittivity of the soil-water matrix rises sharply. This change turns the soil into an efficient reflector and increases backscatter returns by several decibels without altering surface geometry.

### Geometric Distortions in Radar Geometry

Radar measures range from the two-way signal travel time $\Delta t = 2R/c$. Because Sentinel-1 acquires SAR imagery at an off-nadir incidence angle $\theta_i \in [29^\circ, 46^\circ]$, three geometric distortions occur:

* **Foreshortening:** Slopes facing toward the radar sensor appear compressed in the slant-range plane.


* **Layover:** Steep mountain summits sit closer to the antenna in slant range than the mountain base, which inverts the topography.


* **Radar Shadow:** Terrain slopes steeper than the radar depression angle receive no radar illumination, which causes a complete loss of backscatter data.

## Passive Optical Spectroscopy: Sentinel-2

Sentinel-2 carries the Multi-Spectral Instrument (MSI), an optical pushbroom sensor. The MSI measures top-of-atmosphere (TOA) and bottom-of-atmosphere (BOA) surface reflectance across 13 spectral bands in the visible, near-infrared, and shortwave-infrared spectra.

### Radiometric Calibration and Reflectance

Passive optical sensors measure spectral radiance $L_\lambda$ ($\text{W}\cdot\text{m}^{-2}\cdot\text{sr}^{-1}\cdot\mu\text{m}^{-1}$). Atmospheric correction routines such as Sen2Cor convert radiance to bottom-of-atmosphere surface reflectance $\rho_\lambda \in [0, 1]$ by compensating for Rayleigh scattering and aerosol absorption:

$$\rho_\lambda = \frac{\pi \cdot L_{\lambda,\text{surface}}}{E_{0,\lambda} \cdot \cos\theta_s}$$


where $E_{0,\lambda}$ is the exo-atmospheric solar spectral irradiance and $\theta_s$ is the solar zenith angle.

### Spectral Bands and Physical Targets

Our paired retrieval pipeline uses the 10 core surface reflectance bands at a common $10\text{ m}$ spatial resolution and excludes the $60\text{ m}$ atmospheric bands (B01, B09, and B10):

1. **B02 (Blue, $490\text{ nm}$):** Distinguishes soil from vegetation and monitors atmospheric scattering.


2. **B03 (Green, $560\text{ nm}$):** Captures the green reflection peak of chlorophyll in vegetation.


3. **B04 (Red, $665\text{ nm}$):** Measures strong chlorophyll-a and chlorophyll-b absorption ($650 - 680\text{ nm}$).


4. **B05, B06, B07 (Vegetation Red Edge, $705 - 783\text{ nm}$):** Captures the steep transition between visible red absorption and NIR scattering, which indicates vegetation stress, chlorophyll content, and canopy structure.


5. **B08 (Broad NIR, $842\text{ nm}$) & B8A (Narrow NIR, $865\text{ nm}$):** Measures leaf mesophyll cell wall scattering ($40 - 60\%$ reflectance in healthy canopies) to compute the Normalized Difference Vegetation Index (NDVI).


6. **B11 ($1610\text{ nm}$) & B12 ($2190\text{ nm}$) (SWIR):** Detects plant canopy moisture, soil hydration, and mineral composition through hydroxyl ion absorption.

## Synthesis: The Cross-Modal Sensor Domain Gap

Sentinel-1 SAR and Sentinel-2 MSI rely on complementary but distinct physical mechanisms:

| Physical / Operational Dimension | Sentinel-1 SAR (Microwave) | Sentinel-2 MSI (Optical) |
| --- | --- | --- |
| **Energy Source** | Active coherent transmission (C-band) | Passive extraterrestrial solar irradiance
| **Primary Interaction** | Surface roughness, geometric structure, dielectric constant| Electronic transitions, molecular vibrations, pigments
| **Atmospheric Sensitivity** | Completely transparent to clouds, haze, rain, and smoke| Severely attenuated or blocked by clouds, fog, and shadow
| **Diurnal Independence** | Operational day and night| Constrained strictly to clear daylight conditions
| **Topographic Sensitivity** | Slant-range distortions (foreshortening, layover, shadow)| Optical cast shadows, terrain slope illumination variation
| **Dimensionality** | 2 channels ($\text{VV}, \text{VH}$) in decibel scale| 10 spectral bands scaled in surface reflectance $[0, 1]$

Because the radiometric properties of these two sensors diverge, pixel-wise comparisons and shallow statistical correlations fail. To align these observations, the system uses deep contrastive metric learning to map surface geometry and spectral reflectance into a single, modality-agnostic latent space.
