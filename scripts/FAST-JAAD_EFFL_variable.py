# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 12:16:25 2026

@author: Morgan
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import pyzdde.zdde as pyz
from matplotlib.colors import Normalize

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
ZEMAX_FILE = PROJECT_ROOT / "Zemax" / "FAST - Model 2023 - Final.zmx"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
DATA_DIR = RESULTS_DIR / "data"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


from utils.fast_utils import (
    sample_random_pupil,
    trace_real_field,
    trace_field_centroid,
)

from utils.fast_plots import (
    apply_scientific_style,
    apply_colorbar_style,
    save_figure,
)


# ============================================================
# Load saved maximum-field results
# ============================================================

with open(DATA_DIR / "max_field_config3.json", "r", encoding="utf-8") as f:
    results = json.load(f)

# ============================================================
# Simulation configuration
# ============================================================

# Zemax configuration
config = results["config"]

# Optical surfaces
final_surf = results["final_surf"]

# Field normalization
Fr = results["Fr"]

# Pupil sampling
pupil_seed = results["pupil_seed"]
n_pupil_rays = results["n_pupil_rays"]

pupil_points = sample_random_pupil(
    n_pupil_rays,
    seed=pupil_seed
)

# Slit geometry (mm)
slit_width = results["slit_width_mm"]
slit_height = results["slit_height_mm"]

# Maximum admissible fields (degrees)
fx_max = results["fx_max_deg"]
fy_max = results["fy_max_deg"]

# Fields within the admissible region
fields_to_plot = [
    (-fx_max, fy_max),
    (-fx_max / 2.0, fy_max),
    (fx_max, fy_max),
    (0.0, 0.0),
    (fx_max, -fy_max),
    (fx_max / 2.0, fy_max),
    (-fx_max, -fy_max),
]

# Detector and optical scales
pixel_size = 0.0135  # mm/pixel

# Spectral sampling
wavelengths_um = np.linspace(0.37, 0.95, 100)
wave_index = 1

# Open Zemax
ln = pyz.createLink()
ln.zLoadFile(str(ZEMAX_FILE))
ln.zSetConfig(config)


# ============================================================
# Measure projected field rectangle at image plane
# ============================================================

test_lambda = 0.65

ln.zSetWave(wave_index, test_lambda, 1.0)
ln.zGetUpdate()

fields_to_measure = [
    (-fx_max, fy_max),
    (-fx_max, -fy_max),
    (fx_max, fy_max),
]

centers = {
    (fx, fy): trace_field_centroid(
        ln=ln,
        fx=fx,
        fy=fy,
        Fr=Fr,
        pupil_points=pupil_points,
        surf=final_surf,
        wave=wave_index
    )
    for fx, fy in fields_to_measure
}

# Projected slit dimensions
x_top_left, y_top_left = centers[(-fx_max, fy_max)]
x_bot_left, y_bot_left = centers[(-fx_max, -fy_max)]
x_top_right, y_top_right = centers[(fx_max, fy_max)]

projected_height_y = abs(y_top_left - y_bot_left)
projected_width_x = abs(x_top_right - x_top_left)

# Geometrical magnification
magnification_y = projected_height_y / slit_height
magnification_x = projected_width_x / slit_width

# ------------------------------------------------------------
# Image-plane footprint
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 8))

norm = Normalize(
    vmin=wavelengths_um.min(),
    vmax=wavelengths_um.max()
)

cmap_obj = mpl.colormaps.get_cmap("turbo")

for fx, fy in fields_to_plot:

    X_all = []
    Y_all = []
    lambda_all = []

    for wavelength in wavelengths_um:

        ln.zSetWave(
            wave_index,
            float(wavelength),
            1.0
        )

        ln.zGetUpdate()

        X, Y = trace_real_field(
            ln=ln,
            fx=fx,
            fy=fy,
            Fr=Fr,
            pupil_points=pupil_points,
            surf=final_surf,
            wave=wave_index
        )

        X_all.extend(X)
        Y_all.extend(Y)
        lambda_all.extend([wavelength] * len(X))

    sc = ax.scatter(
        X_all,
        Y_all,
        c=lambda_all,
        cmap=cmap_obj,
        norm=norm,
        s=5,
        alpha=1.0
    )

apply_scientific_style(
    ax,
    xlabel=r"$X\;(\mathrm{mm})$",
    ylabel=r"$Y\;(\mathrm{mm})$"
)

ax.set_aspect("equal")
ax.set_xlim(-11, 11)
ax.set_ylim(-11, 11)

cbar = fig.colorbar(
    sc,
    ax=ax,
    shrink=0.80,
    pad=0.03
)

apply_colorbar_style(
    cbar,
    r"$\lambda\;(\mu\mathrm{m})$"
)

fig.tight_layout()

save_figure(
    fig,
    filename="Image_plane_footprint_variable_slit",
    results_dir=FIGURES_DIR,
    file_format="both"
)

plt.show()


# ============================================================
# Central field centroids
# ============================================================

centroids_by_wave = {}

for wavelength in wavelengths_um:

    ln.zSetWave(wave_index, float(wavelength), 1.0)
    ln.zGetUpdate()

    centroids_by_wave[float(wavelength)] = trace_field_centroid(
        ln=ln,
        fx=0.0,
        fy=0.0,
        Fr=Fr,
        pupil_points=pupil_points,
        surf=final_surf,
        wave=wave_index
    )

# ============================================================
# Projected slit size vs wavelength
# ============================================================

projected_slit_by_wave = {}

for wavelength in wavelengths_um:

    ln.zSetWave(wave_index, float(wavelength), 1.0)
    ln.zGetUpdate()

    # Slit edge centroids
    xc_top, yc_top = trace_field_centroid(
        ln=ln,
        fx=-fx_max,
        fy=fy_max,
        Fr=Fr,
        pupil_points=pupil_points,
        surf=final_surf,
        wave=wave_index
    )

    xc_bottom, yc_bottom = trace_field_centroid(
        ln=ln,
        fx=-fx_max,
        fy=-fy_max,
        Fr=Fr,
        pupil_points=pupil_points,
        surf=final_surf,
        wave=wave_index
    )

    # Projected slit size in the XY image plane
    projected_slit_mm = np.hypot(
        xc_top - xc_bottom,
        yc_top - yc_bottom
    )

    projected_slit_by_wave[float(wavelength)] = projected_slit_mm

ln.close()

# ============================================================
# Spectral calculations
# ============================================================

# Wavelength array
wavelengths = np.array(sorted(centroids_by_wave.keys()))

# Consecutive wavelength intervals
delta_wave_um = np.diff(wavelengths)
lambda_mid = wavelengths[:-1] + delta_wave_um / 2.0


# ============================================================
# Wavelength-dependent projected slit size
# ============================================================

projected_slit_mm = np.array([
    projected_slit_by_wave[lam]
    for lam in wavelengths
])

# Convert projected slit size to pixels
projected_slit_pix = projected_slit_mm / pixel_size

# Interpolate slit size at midpoint wavelengths
projected_slit_pix_mid = np.interp(
    lambda_mid,
    wavelengths,
    projected_slit_pix
)


# ============================================================
# Spectral centroid positions
# ============================================================

x_position_mm = np.array([
    centroids_by_wave[lam][0]
    for lam in wavelengths
])

y_position_mm = np.array([
    centroids_by_wave[lam][1]
    for lam in wavelengths
])


# ============================================================
# Spectral displacement in the image plane
# ============================================================

delta_x_mm = np.diff(x_position_mm)
delta_y_mm = np.diff(y_position_mm)

# Y-only displacement
delta_y_abs_mm = np.abs(delta_y_mm)

# Total XY displacement
delta_r_mm = np.hypot(delta_x_mm, delta_y_mm)


# ============================================================
# Cumulative spectral positions
# ============================================================

# Y-only cumulative position
y_spectral_position_mm = np.concatenate((
    [0.0],
    np.cumsum(delta_y_abs_mm)
))

# XY cumulative spectral path
r_position_mm = np.concatenate((
    [0.0],
    np.cumsum(delta_r_mm)
))

# Verify consistency
assert np.allclose(
    np.diff(r_position_mm),
    delta_r_mm
), "Inconsistent cumulative spectral path."


# ============================================================
# Linear spectral dispersion
# ============================================================

delta_r_pix = delta_r_mm / pixel_size

dlambda_per_pix = delta_wave_um / delta_r_pix


# ============================================================
# Spectral purity and resolving power
# ============================================================

delta_lambda_purity = (
    projected_slit_pix_mid * dlambda_per_pix
)

Resolution = lambda_mid / delta_lambda_purity

spectral_data = pd.DataFrame({
    "wavelength_mid_um": lambda_mid,
    "delta_wavelength_um": delta_wave_um,
    "delta_x_mm": delta_x_mm,
    "delta_y_mm": delta_y_mm,
    "delta_xy_mm": delta_r_mm,
    "delta_xy_pix": delta_r_pix,
    "projected_slit_pix": projected_slit_pix_mid,
    "dispersion_um_per_pix": dlambda_per_pix,
    "spectral_purity_um": delta_lambda_purity,
    "resolving_power": Resolution,
})

csv_file = DATA_DIR / "FAST_JAAD_Variable_Slit_spectral_properties.csv"

spectral_data.to_csv(
    csv_file,
    index=False,
    float_format="%.10g"
)

centroid_data = pd.DataFrame({
    "wavelength_um": wavelengths,
    "x_position_mm": x_position_mm,
    "y_position_mm": y_position_mm,
    "spectral_path_mm": r_position_mm,
    "projected_slit_mm": projected_slit_mm,
    "projected_slit_pix": projected_slit_pix,
})

centroid_csv_file = (
    DATA_DIR / "FAST_JAAD_Variable_Slit_centroid_positions.csv"
)

centroid_data.to_csv(
    centroid_csv_file,
    index=False,
    float_format="%.10g"
)

# ============================================================
# Final plots
# ============================================================

# ------------------------------------------------------------
# 1) Y-axis vs 2D centroid displacement
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 7))

ax.plot(
    lambda_mid,
    np.abs(delta_y_mm),
    "-",
    color="black",
    linewidth=2,
    label=r"$|\Delta Y_c|$"
)

ax.plot(
    lambda_mid,
    delta_r_mm,
    "--",
    color="gray",
    linewidth=2,
    label=r"$\Delta R_c=\sqrt{\Delta X_c^2+\Delta Y_c^2}$"
)

apply_scientific_style(
    ax,
    xlabel=r"$\lambda\;(\mu\mathrm{m})$",
    ylabel=r"$\Delta r\;(\mathrm{mm})$"
)

ax.legend(
    fontsize=12,
    frameon=False
)

fig.tight_layout()

save_figure(
    fig,
    filename="Centroid_displacement_comparison_variable_slit",
    results_dir=FIGURES_DIR,
    file_format="both"
)

plt.show()

# ------------------------------------------------------------
# 2) Wavelength vs projected slit size
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 7))

ax.plot(
    wavelengths,
    projected_slit_mm,
    "-",
    color="black",
    linewidth=2
)

apply_scientific_style(
    ax,
    xlabel=r"$\lambda\;(\mu\mathrm{m})$",
    ylabel=r"$h_{\mathrm{proj}}\;(\mathrm{mm})$"
)

fig.tight_layout()

save_figure(
    fig,
    filename="Projected_slit_size_vs_wavelength",
    results_dir=FIGURES_DIR,
    file_format="both"
)

plt.show()


# ------------------------------------------------------------
# 3) Wavelength vs cumulative spectral position
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 7))

ax.plot(
    y_spectral_position_mm,
    wavelengths,
    "-",
    color="gray",
    linewidth=2,
    label=r"$Y_c$ only"
)

ax.plot(
    r_position_mm,
    wavelengths,
    "--",
    color="black",
    linewidth=2,
    label=r"$X_c$ and $Y_c$"
)

apply_scientific_style(
    ax,
    xlabel=r"$R_c\;(\mathrm{mm})$",
    ylabel=r"$\lambda\;(\mu\mathrm{m})$"
)

ax.legend(
    fontsize=12,
    frameon=False
)

fig.tight_layout()

save_figure(
    fig,
    filename="Wavelength_vs_spectral_position_variable_slit",
    results_dir=FIGURES_DIR,
    file_format="both"
)

plt.show()

# ------------------------------------------------------------
# 4) Linear spectral dispersion
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 7))

ax.plot(
    lambda_mid,
    dlambda_per_pix,
    "-",
    color="black",
    linewidth=2
)

apply_scientific_style(
    ax,
    xlabel=r"$\lambda\;(\mu\mathrm{m})$",
    ylabel=r"$d\lambda/dl\;(\mu\mathrm{m/pixel})$"
)

fig.tight_layout()

save_figure(
    fig,
    filename="Spectral_dispersion_variable_slit",
    results_dir=FIGURES_DIR,
    file_format="both"
)

plt.show()

# ------------------------------------------------------------
# 5) Spectral resolving power
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 7))

ax.plot(
    lambda_mid,
    Resolution,
    "-",
    color="black",
    linewidth=2
)

apply_scientific_style(
    ax,
    xlabel=r"$\lambda\;(\mu\mathrm{m})$",
    ylabel=r"$R=\lambda/\delta\lambda$"
)

fig.tight_layout()

save_figure(
    fig,
    filename="Spectral_resolution_variable_slit",
    results_dir=FIGURES_DIR,
    file_format="both"
)

plt.show()

# ============================================================
# Simulation summary
# ============================================================

print("\n" + "=" * 55)
print("FAST-JAAD | Spectral Analysis (Variable Slit)")
print("=" * 55)

print(f"Zemax configuration : {config}")
print(f"Pupil rays          : {len(pupil_points)}")
print(f"Sampled wavelengths : {len(wavelengths)}")
print(
    f"Wavelength range    : "
    f"{wavelengths.min():.3f} - "
    f"{wavelengths.max():.3f} um"
)

print("\nProjected slit geometry at reference wavelength:")
print(f"  Reference wavelength : {test_lambda:.3f} um")
print(f"  Width                : {projected_width_x:.6f} mm")
print(f"  Height Y             : {projected_height_y:.6f} mm")
print(f"  Magnification X      : {magnification_x:.4f}")
print(f"  Magnification Y      : {magnification_y:.4f}")

print("\nWavelength-dependent slit projection:")
print(
    f"  Projected slit size : "
    f"{np.min(projected_slit_mm):.6f} - "
    f"{np.max(projected_slit_mm):.6f} mm"
)
print(
    f"  Projected slit size : "
    f"{np.min(projected_slit_pix):.3f} - "
    f"{np.max(projected_slit_pix):.3f} pixels"
)

print("\nDetector:")
print(f"  Pixel size          : {pixel_size * 1000:.2f} um")

print("\nSpectral performance:")
print(
    f"  Resolving power     : "
    f"{np.min(Resolution):.2f} - "
    f"{np.max(Resolution):.2f}"
)
print(
    f"  Spectral purity     : "
    f"{np.min(delta_lambda_purity):.6f} - "
    f"{np.max(delta_lambda_purity):.6f} um"
)
print(
    f"  Dispersion          : "
    f"{np.min(dlambda_per_pix):.6f} - "
    f"{np.max(dlambda_per_pix):.6f} um/pixel"
)

print("\nOutput:")
print(f"  Figures             : {FIGURES_DIR}")
print(f"  Spectral data       : {csv_file}")
print(f"  Centroid data       : {centroid_csv_file}")

print("=" * 55)