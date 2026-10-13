# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 16:49:51 2026

@author: Morgan
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyzdde.zdde as pyz
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


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
    trace_real_field_multiray,
    spot_centroid,
)

from utils.fast_plots import (
    plot_interpolated_multicolor_spot,
    apply_scientific_style,
    apply_colorbar_style,
    save_figure,
)

with open(DATA_DIR / "max_field_config3.json", "r", encoding="utf-8") as f:
    results = json.load(f)

# ============================================================
# Simulation configuration
# ============================================================

# Zemax configuration
config = results["config"]

# Optical surfaces
slit_surf = results["slit_surf"]
final_surf = results["final_surf"]

# Field normalization
Fr = results["Fr"]

# Pupil sampling
pupil_seed = results["pupil_seed"]
n_pupil_rays = results["n_pupil_rays"]

# Slit geometry (mm)
x0 = results["x0"]
y0 = results["y0"]

slit_width = results["slit_width_mm"]
slit_height = results["slit_height_mm"]

# Maximum admissible fields (degrees)
fx_max = results["fx_max_deg"]
fy_max = results["fy_max_deg"]

# Detector and optical scales
pixel_size = 0.0135  # mm/pixel
EFFL_mean = -5520.0  # mm

# Wavelengths defined in Zemax
waves = list(range(1, 10))

wavelengths_um = np.array([
    0.38, 0.4475, 0.515,
    0.5825, 0.65, 0.717,
    0.785, 0.8525, 0.92
])


pupil_points = sample_random_pupil(n_pupil_rays, seed=pupil_seed)

cmap = plt.get_cmap("turbo")
colors = [cmap(i / (len(waves) - 1)) for i in range(len(waves))]

fields_to_plot = [
    (-fx_max, fy_max),
    (-fx_max / 2, fy_max),
    ( fx_max, fy_max),
    (0.0, 0.0),
    ( fx_max, -fy_max),
    ( fx_max / 2, fy_max),
    (-fx_max, -fy_max),
]

# Open Zemax link

ln = pyz.createLink()
ln.zLoadFile(str(ZEMAX_FILE))
ln.zSetConfig(config)


# ============================================================
# Plot surface 20: slit plane
# ============================================================

plt.figure(figsize=(8, 8))

slit_label = f"Slit ({slit_width*1e3:.0f} × {slit_height*1e3:.0f} µm)"
slit_rect = Rectangle(
    (x0 - slit_width / 2.0, y0 - slit_height / 2.0),
    slit_width,
    slit_height,
    fill=False,
    linewidth=2,
    linestyle="--",
    label=slit_label
)
plt.gca().add_patch(slit_rect)

labeled = set()

for fx, fy in fields_to_plot:
    for i, wave in enumerate(waves):
        X, Y = trace_real_field(
            ln=ln,
            fx=fx,
            fy=fy,
            Fr=Fr,
            pupil_points=pupil_points,
            surf=slit_surf,
            wave=wave
        )

        label = f"{wavelengths_um[i]:.2f} µm" if i not in labeled else None
        labeled.add(i)

        plt.scatter(X, Y, s=5, color=colors[i], label=label)

ax = plt.gca()

apply_scientific_style(
    ax,
    xlabel=r"$X\;(\mathrm{mm})$",
    ylabel=r"$Y\;(\mathrm{mm})$"
)

ax.set_aspect("equal")
ax.set_xlim(-0.6, 0.6)
ax.set_ylim(-0.6, 0.6)

ax.legend(
    fontsize=11,
    loc="upper left",
    frameon=False
)

plt.gcf().tight_layout()

save_figure(
    plt.gcf(),
    filename="Slit_footprint_results",
    results_dir=FIGURES_DIR,
    file_format="both"
)

plt.show()


# ============================================================
# Plot surface 57: image plane
# ============================================================

plt.figure(figsize=(8, 8))
ax = plt.gca()

for fx, fy in fields_to_plot:
    x_by_wave, y_by_wave = trace_real_field_multiray(
        ln=ln,
        fx=fx,
        fy=fy,
        Fr=Fr,
        pupil_points=pupil_points,
        surf=final_surf,
        waves=waves
    )

    sm = plot_interpolated_multicolor_spot(
        ax=ax,
        x_by_wave=x_by_wave,
        y_by_wave=y_by_wave,
        wavelengths_um=wavelengths_um,
        n_interp=80,
        cmap="turbo",
        lw=0.4,
        alpha=0.5
    )

apply_scientific_style(
    ax,
    xlabel=r"$X\;(\mathrm{mm})$",
    ylabel=r"$Y\;(\mathrm{mm})$"
)

ax.set_aspect("equal")
ax.set_xlim(-11, 11)
ax.set_ylim(-11, 11)

cbar = plt.colorbar(
    sm,
    ax=ax,
    shrink=0.80,
    pad=0.03
)

apply_colorbar_style(
    cbar,
    r"$\lambda\;(\mu\mathrm{m})$"
)

plt.gcf().tight_layout()

save_figure(
    plt.gcf(),
    filename="Image_plane_footprint_results",
    results_dir=FIGURES_DIR,
    file_format="both"
)

plt.show()


# ============================================================
# Central field centroids
# ============================================================

centroids_by_wave = {}

for wave, wavelength in zip(waves, wavelengths_um):
    X, Y = trace_real_field(
        ln=ln,
        fx=0.0,
        fy=0.0,
        Fr=Fr,
        pupil_points=pupil_points,
        surf=final_surf,
        wave=wave
    )

    xc, yc = spot_centroid(X, Y)
    centroids_by_wave[wavelength] = (xc, yc)

ln.close()


# ============================================================
# Spectral calculations
# ============================================================

plate_scale_mm = 206265 / abs(EFFL_mean)        # arcsec/mm
plate_scale_pix = plate_scale_mm * pixel_size   # arcsec/pix

slit_height_pix = slit_height / pixel_size

wavelengths = np.array(sorted(centroids_by_wave.keys()))
y_position_mm = np.array([centroids_by_wave[lam][1] for lam in wavelengths])

delta_wave_um = np.diff(wavelengths)
lambda_mid = wavelengths[:-1] + delta_wave_um / 2.0

delta_pos_mm = np.diff(y_position_mm)
delta_pos_pix = delta_pos_mm / pixel_size
delta_pos_pix_abs = np.abs(delta_pos_pix)

delta_lambda_purity = (slit_height_pix * delta_wave_um) / delta_pos_pix_abs
Resolution = lambda_mid / delta_lambda_purity

dlambda_per_pix = delta_wave_um / delta_pos_pix_abs
dlambda_per_mm = dlambda_per_pix / pixel_size

spectral_results = pd.DataFrame({
    "wavelength_mid_um": lambda_mid,
    "delta_lambda_um": delta_lambda_purity,
    "resolving_power": Resolution,
    "dispersion_um_per_pixel": dlambda_per_pix,
    "dispersion_um_per_mm": dlambda_per_mm,
    "centroid_displacement_y_mm": delta_pos_mm,
    "centroid_displacement_y_pixel": delta_pos_pix,
})

output_file = DATA_DIR / "FAST_JAAD_Analysis_Y_spectral_properties.csv"

spectral_results.to_csv(
    output_file,
    index=False,
    float_format="%.10g"
)

# ============================================================
# Final plots
# ============================================================

# ============================================================
# Wavelength vs detector position
# ============================================================

fig, ax = plt.subplots(figsize=(9, 7))

ax.plot(
    y_position_mm,
    wavelengths,
    "-",
    color="black",
    linewidth=2
)

apply_scientific_style(
    ax,
    xlabel=r"$Y\;(\mathrm{mm})$",
    ylabel=r"$\lambda\;(\mu\mathrm{m})$"
)

fig.tight_layout()

save_figure(
    fig,
    filename="Wavelength_vs_detector_position_results",
    results_dir=FIGURES_DIR,
    file_format="both"
)

plt.show()


# ============================================================
# Spectral resolution
# ============================================================

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
    filename="Spectral_resolution_results",
    results_dir=FIGURES_DIR,
    file_format="both"
)

plt.show()


# ============================================================
# Linear spectral dispersion
# ============================================================

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
    ylabel=r"$\Delta\lambda\;(\mu\mathrm{m/pixel})$"
)

fig.tight_layout()

save_figure(
    fig,
    filename="Spectral_dispersion_results",
    results_dir=FIGURES_DIR,
    file_format="both"
)

plt.show()
   
# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 48)
print("FAST-JAAD | Spectral Analysis (Y-axis)")
print("=" * 48)

print(f"Zemax configuration : {config}")
print(f"Pupil rays          : {len(pupil_points)}")
print(
    f"Wavelength range    : "
    f"{np.min(wavelengths_um):.3f} - "
    f"{np.max(wavelengths_um):.3f} um"
)

print("\nMaximum accepted fields:")
print(f"  fx = {fx_max:.8f} deg ({fx_max * 3600:.4f} arcsec)")
print(f"  fy = {fy_max:.8f} deg ({fy_max * 3600:.4f} arcsec)")

print("\nOptical scales:")
print(f"  Plate scale       : {plate_scale_mm:.4f} arcsec/mm")
print(f"  Pixel scale       : {plate_scale_pix:.4f} arcsec/pixel")
print(f"  Slit height       : {slit_height_pix:.3f} pixel")

print("\nSpectral performance:")
print(
    f"  Resolving power R : "
    f"{np.min(Resolution):.2f} - {np.max(Resolution):.2f}"
)
print(
    f"  Spectral purity   : "
    f"{np.min(delta_lambda_purity):.6f} - "
    f"{np.max(delta_lambda_purity):.6f} um"
)
print(
    f"  Dispersion        : "
    f"{np.min(dlambda_per_pix):.6f} - "
    f"{np.max(dlambda_per_pix):.6f} um/pixel"
)

print(f"\nFigures directory   : {FIGURES_DIR}")
print(f"Spectral data       : {output_file}")
print("=" * 48)