# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 14:46:22 2026

@author: Morgan
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "results" / "data"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

from utils.fast_plots import (
    apply_scientific_style,
    save_figure,
)

# ============================================================
# Load spectral results
# ============================================================

constant_data = pd.read_csv(
    DATA_DIR / "FAST_JAAD_Continuous_2D_spectral_properties.csv"
)

variable_data = pd.read_csv(
    DATA_DIR / "FAST_JAAD_Variable_Slit_spectral_properties.csv"
)

# Verify wavelength sampling
wavelengths = constant_data["wavelength_mid_um"].to_numpy()

assert np.allclose(
    wavelengths,
    variable_data["wavelength_mid_um"].to_numpy()
), "Wavelength grids do not match."

# Spectral resolving power
R_constant = constant_data["resolving_power"].to_numpy()
R_variable = variable_data["resolving_power"].to_numpy()

# Relative difference
relative_difference = (
    (R_variable - R_constant) / R_constant * 100.0
)

# ============================================================
# Compare spectral resolving power
# ============================================================

fig, ax = plt.subplots(figsize=(9, 7))

ax.plot(
    wavelengths,
    R_constant,
    "-",
    color="black",
    linewidth=2,
    label="Constant slit"
)

ax.plot(
    wavelengths,
    R_variable,
    "--",
    color="gray",
    linewidth=2,
    label="Wavelength-dependent slit"
)

apply_scientific_style(
    ax,
    xlabel=r"$\lambda\;(\mu\mathrm{m})$",
    ylabel=r"$R=\lambda/\delta\lambda$"
)

ax.legend(fontsize=12, frameon=False)

fig.tight_layout()

save_figure(
    fig,
    filename="Spectral_resolution_model_comparison",
    results_dir=FIGURES_DIR,
    file_format="both"
)

plt.show()

# ============================================================
# Relative difference in spectral resolving power
# ============================================================

fig, ax = plt.subplots(figsize=(9, 7))

ax.plot(
    wavelengths,
    relative_difference,
    "-",
    color="black",
    linewidth=2
)

ax.axhline(
    y=0.0,
    color="gray",
    linestyle="--",
    linewidth=1
)

apply_scientific_style(
    ax,
    xlabel=r"$\lambda\;(\mu\mathrm{m})$",
    ylabel=r"$\Delta R/R\;(\%)$"
)

fig.tight_layout()

save_figure(
    fig,
    filename="Spectral_resolution_relative_difference",
    results_dir=FIGURES_DIR,
    file_format="both"
)

plt.show()

# ============================================================
# Comparison summary
# ============================================================

print("\n" + "=" * 55)
print("FAST-JAAD | Spectral Resolution Comparison")
print("=" * 55)

print(f"Spectral intervals : {len(wavelengths)}")

print("\nConstant slit:")
print(f"  Resolving power  : {R_constant.min():.2f} - {R_constant.max():.2f}")

print("\nVariable slit:")
print(f"  Resolving power  : {R_variable.min():.2f} - {R_variable.max():.2f}")

print("\nRelative difference:")
print(f"  Minimum          : {relative_difference.min():.2f}%")
print(f"  Maximum          : {relative_difference.max():.2f}%")
print(f"  Mean             : {relative_difference.mean():.2f}%")

print("=" * 55)