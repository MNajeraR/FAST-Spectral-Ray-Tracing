# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 12:00:31 2026

@author: Morgan
"""

import sys
from pathlib import Path
import numpy as np
import pyzdde.zdde as pyz
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


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
    binary_search_fx_max_full_spot,
    binary_search_fy_max_full_spot,
    spot_centroid,
    smooth_curve,
)

from utils.fast_plots import (
    plot_interpolated_multicolor_spot,
    apply_scientific_style,
    save_figure, 
    apply_colorbar_style,
)

# ============================================================
# Simulation configuration
# ============================================================

# Zemax configuration
config = 3

# Optical surfaces
slit_surf = 20
final_surf = 57

# System fields (degrees)
system_fields = [
    (0.000, 0.000),
    (0.014, 0.000),
    (0.028, 0.000),
    (0.042, 0.000),
    (0.056, 0.000),
    (0.070, 0.000),
    (0.084, 0.000),
]

Fr = max(np.hypot(fx, fy) for fx, fy in system_fields)

# Wavelengths (um)
waves = list(range(1, 10))

wavelengths_um = [
    0.42, 0.45, 0.52,
    0.58, 0.65, 0.72,
    0.79, 0.85, 0.90
]

# Pupil sampling
n_pupil_rays = 100
pupil_seed = 123

# Slit geometry (mm)
x0 = 0.0
y0 = 0.0

slit_width = 1.000
slit_height = 0.040

# Detector and optical scales
pixel_size = 0.0135  # mm/pixel
EFFL_mean = -5520.0  # mm

# Binary-search parameters (degrees)
fx_low = -1.0e-3
fx_high = -1.0e-2

fy_low = 0.0
fy_high = 3.0e-4

# ============================================================
# Initialize Zemax
# ============================================================

ln = pyz.createLink()
ln.zLoadFile(str(ZEMAX_FILE))
ln.zSetConfig(config)

pupil_points = sample_random_pupil(
    n_pupil_rays,
    seed=pupil_seed
)

# ============================================================
# Spectral colormap
# ============================================================

cmap = plt.get_cmap("turbo")

norm = Normalize(
    vmin=min(wavelengths_um),
    vmax=max(wavelengths_um)
)

colors = [cmap(norm(wavelength)) for wavelength in wavelengths_um]

# ============================================================
# 1) Find maximum admissible field at surface 20
# ============================================================

fx_max = binary_search_fx_max_full_spot(
    ln, Fr, pupil_points, slit_surf, waves,
    x0, y0, slit_width, slit_height,
    fx_low=fx_low,
    fx_high=fx_high
)

fy_max = binary_search_fy_max_full_spot(
    ln, Fr, pupil_points, slit_surf, waves,
    x0, y0, slit_width, slit_height,
    fy_low=fy_low,
    fy_high=fy_high
)

# ============================================================
# 2) Generate fields within the admissible region
# ============================================================

fields_to_plot = [
    (-fx_max, fy_max),
    (-fx_max/2 , fy_max),
    ( fx_max, fy_max),
    ( 0.00  ,  0.00),
    ( fx_max, -fy_max),
    (fx_max/2, fy_max),
    (-fx_max, -fy_max)]

# ============================================================
# 3) Slit footprint - Surface 20
# ============================================================

fig, ax = plt.subplots(figsize=(9, 7))

slit_label = (
    f"Slit ({slit_width*1e3:.0f} × "
    f"{slit_height*1e3:.0f} µm)"
)

slit_rect = Rectangle(
    (x0 - slit_width / 2.0, y0 - slit_height / 2.0),
    slit_width,
    slit_height,
    fill=False,
    edgecolor="black",
    linewidth=2,
    linestyle="--",
    label=slit_label
)

ax.add_patch(slit_rect)
central_spots = {}
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
        if np.isclose(fx, 0.0) and np.isclose(fy, 0.0):
            central_spots[wave] = (X, Y)
        ax.scatter(
            X,
            Y,
            s=5,
            color=colors[i],
            alpha=0.6,
            linewidths=0
        )

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

axins = inset_axes(
    ax,
    width="35%",
    height="35%",
    loc="lower right",
    borderpad=1.5
)

for i, wave in enumerate(waves):

    X, Y = central_spots[wave]

    axins.scatter(
        X,
        Y,
        s=8,
        color=colors[i],
        alpha=0.8,
        linewidths=0
    )

X_all = np.concatenate(
    [central_spots[wave][0] for wave in waves]
)

Y_all = np.concatenate(
    [central_spots[wave][1] for wave in waves]
)

x_min, x_max = np.min(X_all), np.max(X_all)
y_min, y_max = np.min(Y_all), np.max(Y_all)

dx = max(x_max - x_min, 1e-6)
dy = max(y_max - y_min, 1e-6)

margin = 0.15

axins.set_xlim(x_min - margin * dx, x_max + margin * dx)
axins.set_ylim(y_min - margin * dy, y_max + margin * dy)

axins.set_aspect("equal", adjustable="box")

axins.set_xticks([])
axins.set_yticks([])
axins.tick_params(
    bottom=False,
    top=False,
    left=False,
    right=False
)
axins.grid(False)

for spine in axins.spines.values():
    spine.set_color("black")
    spine.set_linewidth(1.2)

axins.set_facecolor("white")

sm = plt.cm.ScalarMappable(
    cmap=cmap,
    norm=norm
)
sm.set_array([])

cbar = fig.colorbar(
    sm,
    ax=ax,
    shrink=0.80,
    pad=0.03
)

apply_colorbar_style(
    cbar,
    r"$\lambda\;(\mu\mathrm{m})$"
)

fig.subplots_adjust(
    left=0.12,
    right=0.88,
    bottom=0.12,
    top=0.95
)

save_figure(
    fig,
    filename="Slit_footprint",
    results_dir=FIGURES_DIR,
    file_format="both"
)

plt.show()

# ============================================================
# 4) Image-plane footprint - Surface 57
# ============================================================

fig, ax = plt.subplots(figsize=(9, 7))

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

cbar = fig.colorbar(
    sm,
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
    filename="Image_plane_footprint",
    results_dir=FIGURES_DIR,
    file_format="both"
)

plt.show()

# ============================================================
# 5) Central field at different wavelengths
# ============================================================
central_field = (0.0, 0.0)
fx, fy = central_field

centroids_by_wave = {}
spots_by_wave = {}

for wave, wavelength in zip(waves, wavelengths_um):
    X, Y = trace_real_field(
        ln=ln,
        fx=fx,
        fy=fy,
        Fr=Fr,
        pupil_points=pupil_points,
        surf=final_surf,
        wave=wave
    )
    spots_by_wave[wavelength] = (X, Y)

    xc, yc = spot_centroid(X, Y)
    centroids_by_wave[wavelength] = (xc, yc)
    
# ============================================================
# 6) Central-field footprint
# ============================================================

fig, ax = plt.subplots(figsize=(9, 7))

for color, wavelength in zip(colors, wavelengths_um):

    X, Y = spots_by_wave[wavelength]
    xc, yc = centroids_by_wave[wavelength]

    # Ray footprint
    ax.scatter(
        X,
        Y,
        s=8,
        color=color,
        alpha=0.35,
        linewidths=0
    )

    # Spot centroid
    ax.scatter(
        xc,
        yc,
        s=75,
        color=color,
        marker="x",
        linewidths=2
    )

apply_scientific_style(
    ax,
    xlabel=r"$X\;(\mathrm{mm})$",
    ylabel=r"$Y\;(\mathrm{mm})$"
)

ax.set_aspect("equal")
ax.set_xlim(-11, 11)
ax.set_ylim(-11, 11)

sm = plt.cm.ScalarMappable(
    cmap=cmap,
    norm=norm
)
sm.set_array([])

cbar = fig.colorbar(
    sm,
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
    filename="Central_field_footprint",
    results_dir=FIGURES_DIR,
    file_format="both"
)

plt.show()
    

# ============================================================
# 7) Spectral purity, resolution, and dispersion analysis
# ============================================================

# ------------------------------------------------------------
# 7.1 Derived optical scales
# ------------------------------------------------------------

plate_scale_mm = 206265 / abs(EFFL_mean)        # arcsec/mm
plate_scale_pix = plate_scale_mm * pixel_size   # arcsec/pix

slit_width_pix = slit_width / pixel_size
slit_height_pix = slit_height / pixel_size

slit_width_arcsec = slit_width * plate_scale_mm
slit_height_arcsec = slit_height * plate_scale_mm

# ------------------------------------------------------------
# 7.2 Wavelength intervals and centroid separations
# ------------------------------------------------------------

wave_keys = sorted(centroids_by_wave.keys())
wavelengths = np.array(wave_keys)

x_vals = np.array([centroids_by_wave[lam][0] for lam in wave_keys])
y_vals = np.array([centroids_by_wave[lam][1] for lam in wave_keys])

delta_lambda = np.diff(wavelengths)          # µm
lambda_mid = 0.5 * (wavelengths[:-1] + wavelengths[1:])

delta_x = np.diff(x_vals)                    # mm
delta_y = np.diff(y_vals)                    # mm

delta_pos_mm = np.sqrt(delta_x**2 + delta_y**2)
delta_pos_pix = delta_pos_mm / pixel_size


# ------------------------------------------------------------
# 7.3 Spectral purity and resolving power
# ------------------------------------------------------------

# Use the slit dimension that limits the spectral resolution.
# For your current configuration, slit_height = 0.040 mm.
slit_resolution_mm = slit_height
slit_resolution_pix = slit_resolution_mm / pixel_size

delta_lambda_purity = (slit_resolution_pix * delta_lambda) / delta_pos_pix
R = lambda_mid / delta_lambda_purity

# ------------------------------------------------------------
# 7.4 Spectral dispersion in wavelength per pixel
# ------------------------------------------------------------

dlambda_per_pix = delta_lambda / delta_pos_pix


lambda_R_smooth, R_smooth = smooth_curve(lambda_mid, R)
lambda_purity_smooth, purity_smooth = smooth_curve(lambda_mid, delta_lambda_purity)
lambda_disp_smooth, dlambda_pix_smooth = smooth_curve(lambda_mid, dlambda_per_pix)
y_smooth, wavelength_smooth = smooth_curve(y_vals, wavelengths)

# ============================================================
# 7.5 Export spectral properties
# ============================================================

spectral_results = pd.DataFrame({
    "wavelength_um": lambda_mid,
    "delta_lambda_um": delta_lambda_purity,
    "resolving_power": R,
    "dispersion_um_per_pixel": dlambda_per_pix,
    "centroid_separation_mm": delta_pos_mm,
    "centroid_separation_pixel": delta_pos_pix,
})

spectral_results.to_csv(
    DATA_DIR / "FAST_JAAD_Analysis_2D_spectral_properties.csv",
    index=False,
    float_format="%.10g"
)

# ============================================================
# 8) Plots
# ============================================================

# ------------------------------------------------------------
# 8.1 Spectral resolution
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 7))

ax.scatter(lambda_mid, R, c=lambda_mid,
           cmap=cmap, norm=norm, marker="x", s=60)
ax.plot(lambda_R_smooth, R_smooth, "-", color = "black",
        linewidth=2)

apply_scientific_style(
    ax,
    xlabel=r"$\lambda\;(\mu\mathrm{m})$",
    ylabel=r"$R=\lambda/\delta\lambda$"
)

fig.tight_layout()

save_figure(
    fig,
    filename="Spectral_resolution",
    results_dir=FIGURES_DIR,
    file_format="both"
)

plt.show()


# ============================================================
# 8.2 Spectral purity
# ============================================================

fig, ax = plt.subplots(figsize=(9, 7))

ax.scatter(lambda_mid, delta_lambda_purity, c=lambda_mid,
           cmap=cmap, norm=norm, marker="x", s=60)
ax.plot(lambda_purity_smooth, purity_smooth, "-", color = "black",
        linewidth=2)

apply_scientific_style(
    ax,
    xlabel=r"$\lambda\;(\mu\mathrm{m})$",
    ylabel=r"$\delta\lambda\;(\mu\mathrm{m})$"
)

fig.tight_layout()

save_figure(
    fig,
    filename="Spectral_purity",
    results_dir=FIGURES_DIR,
    file_format="both"
)

plt.show()


# ============================================================
# 8.3 Spectral dispersion
# ============================================================

fig, ax = plt.subplots(figsize=(9, 7))

ax.scatter(
    lambda_mid,
    dlambda_per_pix,
    c=lambda_mid,
    cmap=cmap,
    norm=norm,
    marker="x",
    s=60
)
ax.plot(lambda_disp_smooth, dlambda_pix_smooth,
        "-", color = "black",
        linewidth=2)

apply_scientific_style(
    ax,
    xlabel=r"$\lambda\;(\mu\mathrm{m})$",
    ylabel=r"$\Delta\lambda\;(\mu\mathrm{m/pixel})$"
)

fig.tight_layout()

save_figure(
    fig,
    filename="Spectral_dispersion",
    results_dir=FIGURES_DIR,
    file_format="both"
)

plt.show()


# ============================================================
# 8.4 Wavelength vs detector position
# ============================================================

fig, ax = plt.subplots(figsize=(9, 7))

ax.scatter(y_vals, wavelengths, c=wavelengths,
           cmap=cmap, norm=norm, marker="x", s=60)
ax.plot(y_smooth, wavelength_smooth, "-", color = "black", 
        linewidth=2)

apply_scientific_style(
    ax,
    xlabel=r"$Y\;(\mathrm{mm})$",
    ylabel=r"$\lambda\;(\mu\mathrm{m})$"
)

fig.tight_layout()

save_figure(
    fig,
    filename="Wavelength_vs_detector_position",
    results_dir=FIGURES_DIR,
    file_format="both"
)

plt.show()

print("\n" + "=" * 48)
print("FAST-JAAD | Spectral Analysis")
print("=" * 48)

print(f"Zemax configuration : {config}")
print(f"Pupil rays          : {len(pupil_points)}")
print(f"Wavelength range    : {min(wavelengths_um):.3f} - {max(wavelengths_um):.3f} um")

print("\nMaximum accepted fields:")
print(f"  fx = {fx_max:.8f} deg ({fx_max * 3600:.4f} arcsec)")
print(f"  fy = {fy_max:.8f} deg ({fy_max * 3600:.4f} arcsec)")

print("\nOptical scales:")
print(f"  Plate scale       : {plate_scale_mm:.4f} arcsec/mm")
print(f"  Pixel scale       : {plate_scale_pix:.4f} arcsec/pixel")

print("\nSpectral performance:")
print(f"  Resolving power R : {np.min(R):.2f} - {np.max(R):.2f}")
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
print(f"Spectral data       : {DATA_DIR / 'FAST_JAAD_Analysis_2D_spectral_properties.csv'}")
print("=" * 48)
