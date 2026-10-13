# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 15:10:44 2026

@author: Morgan
"""

import numpy as np
import matplotlib as mpl
from pathlib import Path

from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from matplotlib.patches import Rectangle


def plot_interpolated_multicolor_spot(ax, x_by_wave, y_by_wave, wavelengths_um,
                                      n_interp=60, cmap='turbo', lw=0.6, alpha=0.8):
    """
    Interpolate each ray trajectory in wavelength and draw it with a continuous colormap.
    """
    wavelengths_um = np.array(wavelengths_um)
    lam_dense = np.linspace(wavelengths_um.min(), wavelengths_um.max(), n_interp)

    norm = Normalize(vmin=wavelengths_um.min(), vmax=wavelengths_um.max())
    cmap_obj = mpl.colormaps.get_cmap(cmap)

    n_rays = x_by_wave.shape[1]

    for ir in range(n_rays):
        x_dense = np.interp(lam_dense, wavelengths_um, x_by_wave[:, ir])
        y_dense = np.interp(lam_dense, wavelengths_um, y_by_wave[:, ir])

        points = np.array([x_dense, y_dense]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(
            segments,
            cmap=cmap_obj,
            norm=norm,
            linewidths=lw,
            alpha=alpha
        )
        lambda_segments = 0.5 * (lam_dense[:-1] + lam_dense[1:])
        lc.set_array(lambda_segments)
        ax.add_collection(lc)

    sm = mpl.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    return sm

def plot_binary_search_frame(
    ax,
    frame,
    fy_history,
    x0,
    y0,
    slit_width,
    slit_height,
    colors,
):
    ax.clear()

    data = fy_history[frame]

    iteration = data["iteration"]
    fy = data["fy"]
    accepted = data["accepted"]
    traces = data["traces"]

    slit_rect = Rectangle(
        (x0 - slit_width / 2.0, y0 - slit_height / 2.0),
        slit_width,
        slit_height,
        fill=False,
        linewidth=2,
        linestyle="--",
    )
    ax.add_patch(slit_rect)

    for i, (X, Y, xc, yc) in enumerate(traces):
        if len(X) == 0:
            continue

        ax.scatter(
            X, Y,
            s=6,
            color=colors[i],
            alpha=0.30,
        )

        ax.scatter(
            xc, yc,
            s=70,
            marker="x",
            color=colors[i],
            linewidth=2,
        )

    ax.axhline(
        y0 + slit_height / 2.0,
        linewidth=1,
        linestyle=":",
    )

    ax.axhline(
        y0 - slit_height / 2.0,
        linewidth=1,
        linestyle=":",
    )

    status = "ACCEPTED" if accepted else "REJECTED"

    ax.set_title(
    f"Iteration {iteration} | "
    f"$f_y$ = {fy:.6e}° "
    f"({fy * 3600:.4f} arcsec) | {status}",
    fontsize=13
    )
    
    ax.set_xlabel("Slit-plane X (mm)", fontsize=12)
    ax.set_ylabel("Slit-plane Y (mm)", fontsize=12)

    ax.set_xlim(-0.08, 0.08)
    ax.set_ylim(-0.030, 0.030)

    ax.set_aspect("equal")
    ax.tick_params(labelsize=11)

    return []

def apply_scientific_style(
    ax,
    xlabel=None,
    ylabel=None,
    labelsize=18,
    tick_labelsize=16
):
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=labelsize)

    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=labelsize)

    ax.set_title("")

    ax.minorticks_on()

    ax.tick_params(
        which="major",
        direction="in",
        top=True,
        right=True,
        length=6,
        width=2.2,
        labelsize=tick_labelsize
    )

    ax.tick_params(
        which="minor",
        direction="in",
        top=True,
        right=True,
        length=3,
        width=1.2
    )


def save_figure(
    fig,
    filename,
    results_dir,
    file_format="both",
    dpi=300
):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if file_format not in ("png", "pdf", "both"):
        raise ValueError("file_format must be png, pdf, or both")

    formats = ["png", "pdf"] if file_format == "both" else [file_format]

    saved_files = []

    for fmt in formats:
        output_file = results_dir / f"{filename}.{fmt}"

        fig.savefig(
            output_file,
            dpi=dpi,
            bbox_inches="tight"
        )

        saved_files.append(output_file)

    return saved_files

def apply_colorbar_style(cbar, label):
    cbar.set_label(label, fontsize=18)
    cbar.ax.minorticks_on()

    cbar.ax.tick_params(
        which="major",
        direction="in",
        length=6,
        width=2.2,
        labelsize=16
    )

    cbar.ax.tick_params(
        which="minor",
        direction="in",
        length=3,
        width=1.2
    )