# FAST Spectral Ray Tracing

Python tools for real-ray-tracing analysis of the **FAST low-resolution spectrograph** using **Zemax OpticStudio** and **PyZDDE**.

## Overview

FAST Spectral Ray Tracing is a collection of Python scripts and reusable utilities for studying field acceptance, slit imaging, and spectral performance in the FAST-JAAD optical model (JAAD refers to the initials of José Antonio Araiza Durán), based on the published design of the Fully Automatic Spectrograph for Transient (FAST) by [Araiza-Durán et al. (2020)](https://doi.org/10.1117/12.2562659). It automates ray tracing over pupil coordinates, field positions, and wavelengths, then analyzes image-plane ray distributions and their centroids. The calculations cover projected slit dimensions, spectral image positions, linear dispersion, spectral purity, and resolving power. Both discrete wavelength sets and a densely sampled wavelength range are examined.

Two related models are provided for the resolving-power estimate:

- **Constant slit model:** Uses a wavelength-independent slit dimension in the spectral-purity calculation.
- **Wavelength-dependent projected slit model:** Ray-traces the upper and lower slit-edge fields at each wavelength, measures their two-dimensional centroid separation, and interpolates that projected dimension to the midpoint wavelengths used for the spectral calculations.

The models use the same two-dimensional centroid displacements for spectral dispersion. Their results can therefore be compared to isolate the effect of the slit-size assumption. The wavelength-dependent model is a geometric estimate; using the full two-dimensional slit-edge separation rather than its projection onto the local dispersion direction remains a modeling assumption to evaluate.

## Repository structure

```text
FAST-Spectral-Ray-Tracing/
├── scripts/
│   ├── FAST-JAAD_Slit_Field_Determination.py
│   ├── FAST-JAAD_Spectral_Analysis_2D.py
│   ├── FAST-JAAD_Spectral_Analysis_Y.py
│   ├── FAST-JAAD_Spectral_Analysis_Continuous_2D.py
│   ├── FAST-JAAD_EFFL_variable.py
│   └── FAST-JAAD_Compare_Spectral_Resolution.py
├── utils/
│   ├── __init__.py
│   ├── fast_utils.py
│   └── fast_plots.py
├── results/
│   ├── data/
│   └── figures/
├── Zemax/
│   └── .gitkeep
├── .gitignore
├── requirements.txt
└── README.md
```

The `Zemax/` directory is reserved for an **authorized, locally supplied** optical model. The optical model is not part of the intended public distribution. Generated figures and numerical outputs are written under `results/`.

## Analysis scripts

| Script | Purpose |
| --- | --- |
| `FAST-JAAD_Slit_Field_Determination.py` | Determines the admissible slit-related field limits through ray tracing and binary searches; saves the configuration used by subsequent analyses. |
| `FAST-JAAD_Spectral_Analysis_2D.py` | Analyzes image-plane footprints and two-dimensional spectral centroid separations for a discrete wavelength set. |
| `FAST-JAAD_Spectral_Analysis_Y.py` | Performs a discrete-wavelength analysis using centroid displacement along the detector Y axis. |
| `FAST-JAAD_Spectral_Analysis_Continuous_2D.py` | Samples 100 wavelengths continuously over 0.37–0.95 µm and computes two-dimensional dispersion, spectral purity, resolving power, and cumulative spectral path length with a constant slit-size assumption. |
| `FAST-JAAD_EFFL_variable.py` | Samples the same wavelength range, measures the wavelength-dependent projected slit-edge separation, and uses it to estimate spectral purity and resolving power. |
| `FAST-JAAD_Compare_Spectral_Resolution.py` | Reads the constant- and variable-slit CSV outputs and compares resolving power and its relative percentage difference without opening Zemax. |

The shared modules provide pupil sampling, real-ray tracing, centroid calculations, field-search helpers, plot styling, and figure export. 

**Execution order:** Run `FAST-JAAD_Slit_Field_Determination.py` first to generate `results/data/max_field_config3.json`, which contains the field limits required by the subsequent spectral analyses. This file must be available before running the scripts that depend on it.

## Methodology

### Pupil sampling

The entrance pupil is sampled using a uniform random distribution over the unit disk. The normalized pupil coordinates are generated as

\[
\rho=\sqrt{u}, \qquad \theta=2\pi v,
\]

\[
P_x=\rho\cos\theta, \qquad P_y=\rho\sin\theta,
\]

where \(u\) and \(v\) are independent random variables uniformly distributed between 0 and 1. The square-root transformation ensures uniform sampling over the pupil area.

### Real-ray tracing

For each field position and wavelength, rays are traced through the FAST-JAAD optical model using PyZDDE to communicate with Zemax OpticStudio.

Field coordinates are converted to the normalized coordinates required by the optical model. Each ray is defined by its field coordinates \((H_x,H_y)\), normalized pupil coordinates \((P_x,P_y)\), and wavelength index.

The ray-tracing procedure returns the intersection coordinates \((x_i,y_i)\) at the selected optical surface. Rays that do not produce valid intersections are excluded from the centroid calculation.

The image-plane centroid is calculated from the successfully traced rays:

\[
\bar{x}=\frac{1}{N}\sum_{i=1}^{N}x_i,
\qquad
\bar{y}=\frac{1}{N}\sum_{i=1}^{N}y_i,
\]

where \(N\) is the number of valid rays.

These centroids are used to determine the positions of spectral images and the projected separation between slit-edge fields.

### Spectral centroid displacement

For consecutive wavelengths \(\lambda_i\) and \(\lambda_{i+1}\), the two-dimensional centroid displacement is

\[
\Delta r_i =
\sqrt{
(\bar{x}_{i+1}-\bar{x}_i)^2+
(\bar{y}_{i+1}-\bar{y}_i)^2
}.
\]

The cumulative spectral path length is obtained by summing these displacements along the sampled wavelength sequence.

### Spectral dispersion

The wavelength interval and corresponding detector displacement are

\[
\Delta\lambda_i=\lambda_{i+1}-\lambda_i,
\qquad
\Delta p_i=\frac{\Delta r_i}{s_{\mathrm{pix}}},
\]

where \(s_{\mathrm{pix}}\) is the detector pixel size in millimeters.

The reciprocal linear dispersion is estimated as

\[
D_i=\frac{\Delta\lambda_i}{\Delta p_i},
\]

expressed in micrometers per pixel.

### Spectral purity and resolving power

Spectral purity is estimated from the detector dispersion and the projected slit dimension:

\[
\delta\lambda_i=w_iD_i,
\]

where \(w_i\) is the assumed slit dimension in pixels.

The resolving power is then

\[
R_i=\frac{\lambda_{\mathrm{mid},i}}{\delta\lambda_i},
\qquad
\lambda_{\mathrm{mid},i}=
\frac{\lambda_i+\lambda_{i+1}}{2}.
\]

Two slit-size assumptions are implemented:

- **Constant slit model:** Uses a fixed slit dimension across the sampled wavelength range.
- **Wavelength-dependent projected slit model:** Traces the upper and lower slit-edge fields at each wavelength and calculates their two-dimensional centroid separation. The resulting projected slit dimensions are interpolated to the midpoint wavelengths.

In the wavelength-dependent model, the full two-dimensional separation is used as the slit dimension. Its projection onto the local spectral dispersion direction is not currently applied and remains a modeling limitation.

### Model comparison

The resolving-power estimates from both models are compared using

\[
\frac{\Delta R}{R}(\%)=
100\frac{R_{\mathrm{variable}}-R_{\mathrm{constant}}}
{R_{\mathrm{constant}}}.
\]

Because both models use the same spectral centroid displacements and detector pixel size, the comparison isolates the effect of the adopted slit-size model.

## Installation and Requirements

### Requirements

The project requires:

- Python 3
- Zemax OpticStudio
- PyZDDE
- NumPy
- SciPy
- Pandas
- Matplotlib

Zemax OpticStudio must be installed and running with the FAST-JAAD optical model loaded in **configuration 3** before executing the ray-tracing scripts. PyZDDE provides the communication interface between Python and Zemax.


## References

Araiza-Durán, J. A., Pignata, G., & Brucalassi, A. (2020). "Final design and performance of the Fully Automatic Spectrograph for Transient (FAST)." *Proceedings of SPIE*, **11447**, 114475R. Ground-based and Airborne Instrumentation for Astronomy VIII.

https://doi.org/10.1117/12.2562659