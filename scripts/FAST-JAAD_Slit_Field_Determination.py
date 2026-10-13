# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 14:58:58 2026

@author: Morgan
"""

import sys
from pathlib import Path
import json
import numpy as np
import pyzdde.zdde as pyz
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter

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
    binary_search_fx_max,
    binary_search_fy_max_with_history,)
from utils.fast_plots import plot_binary_search_frame




ln = pyz.createLink()
ln.zLoadFile(str(ZEMAX_FILE))
ln.zSetConfig(3)

system_fields = [
    (0.00,  0.00),
    (0.014, 0.00),
    (0.028, 0.00),
    (0.042, 0.00),
    (0.056, 0.00),
    (0.070, 0.00),
    (0.084, 0.00),
]

Fr = max(np.hypot(fx, fy) for fx, fy in system_fields)

slit_surf = 20
final_surf = 57

waves = [1, 2, 3, 4, 5, 6, 7, 8, 9]
wavelengths_um = [0.38, 0.4475, 0.515, 0.5825, 0.65, 
                  0.717, 0.785, 0.8525, 0.92]

pupil_seed = 123
n_pupil_rays = 100
pupil_points = sample_random_pupil(n_pupil_rays, seed=pupil_seed)

x0 = 0.0
y0 = 0.0
slit_width = 1.000
slit_height = 0.040

fx_max = binary_search_fx_max(
    ln, Fr, pupil_points, slit_surf, waves,
    x0, y0, slit_width, slit_height,
    fx_low=-1.0e-3,
    fx_high=-1.0e-2
)

fy_max, fy_history = binary_search_fy_max_with_history(
    ln, Fr, pupil_points, slit_surf, waves,
    x0, y0, slit_width, slit_height,
    fy_low=0.0,
    fy_high=3.0e-4
)

# ============================================================
# GIF: Binary search for fy_max
# ============================================================


gif_path = FIGURES_DIR / "binary_search_fy.gif"

fig, ax = plt.subplots(figsize=(8, 6))

cmap = plt.get_cmap("turbo")
colors = [
    cmap(i / (len(waves) - 1))
    for i in range(len(waves))
]

writer = PillowWriter(fps=2)

with writer.saving(fig, str(gif_path), dpi=100):
    for frame in range(len(fy_history)):
        plot_binary_search_frame(
            ax=ax,
            frame=frame,
            fy_history=fy_history,
            x0=x0,
            y0=y0,
            slit_width=slit_width,
            slit_height=slit_height,
            colors=colors,
        )
        writer.grab_frame()

plt.close(fig)

print(f"GIF saved: {gif_path}")

ln.close()

results = {
    "config": 3,
    "Fr": Fr,
    "system_fields": system_fields,
    "slit_surf": slit_surf,
    "final_surf": final_surf,
    "waves": waves,
    "wavelengths_um": wavelengths_um,
    "pupil_seed": pupil_seed,
    "n_pupil_rays": n_pupil_rays,
    "x0": x0,
    "y0": y0,
    "slit_width_mm": slit_width,
    "slit_height_mm": slit_height,
    "fx_max_deg": fx_max,
    "fy_max_deg": fy_max,
    "fx_max_arcsec": fx_max * 3600,
    "fy_max_arcsec": fy_max * 3600,
}

json_path = DATA_DIR / "max_field_config3.json"

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)

print(f"Saved {json_path}")
print("fx_max =", fx_max)
print("fy_max =", fy_max)