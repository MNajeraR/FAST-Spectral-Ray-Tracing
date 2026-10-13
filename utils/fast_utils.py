# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 15:00:33 2026

@author: Morgan
"""

import numpy as np
from scipy.interpolate import make_interp_spline

def sample_random_pupil(n_rays, seed=None):
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 1.0, n_rays)
    v = rng.uniform(0.0, 1.0, n_rays)

    r = np.sqrt(u)
    theta = 2.0 * np.pi * v

    px = r * np.cos(theta)
    py = r * np.sin(theta)

    return np.column_stack((px, py))


def trace_real_field(ln, fx, fy, Fr, pupil_points, surf=20, wave=1):
    hx = fx / Fr
    hy = fy / Fr

    X = []
    Y = []

    for px, py in pupil_points:
        ray = ln.zGetTrace(wave, 0, surf, hx, hy, float(px), float(py))
        error = ray[0]
        vig = ray[1]

        if error == 0 and vig == 0:
            X.append(ray[2])
            Y.append(ray[3])

    return np.array(X), np.array(Y)

def trace_real_field_multiray(ln, fx, fy, Fr, pupil_points, surf, waves):
    """
    Trace the same pupil rays for multiple wavelengths.
    Returns:
        x_by_wave: array shape (n_waves, n_rays)
        y_by_wave: array shape (n_waves, n_rays)
    """
    hx = fx / Fr
    hy = fy / Fr

    n_waves = len(waves)
    n_rays = len(pupil_points)

    x_by_wave = np.full((n_waves, n_rays), np.nan)
    y_by_wave = np.full((n_waves, n_rays), np.nan)
    valid_mask = np.ones(n_rays, dtype=bool)

    for iw, wave in enumerate(waves):
        for ir, (px, py) in enumerate(pupil_points):
            ray = ln.zGetTrace(wave, 0, surf, hx, hy, float(px), float(py))
            error = ray[0]
            vig = ray[1]

            if error == 0 and vig == 0:
                x_by_wave[iw, ir] = ray[2]
                y_by_wave[iw, ir] = ray[3]
            else:
                valid_mask[ir] = False

    # Keep only rays that are valid at all wavelengths
    x_by_wave = x_by_wave[:, valid_mask]
    y_by_wave = y_by_wave[:, valid_mask]

    return x_by_wave, y_by_wave

def spot_center_fits_in_slit(X, Y, x0, y0, slit_width, slit_height):
    """
    Check whether the center of the spot lies inside the slit.

    This criterion is used to determine the geometrical field
    corresponding to the slit edges, without requiring the full
    spot to remain inside the slit.
    """

    if len(X) == 0:
        return False

    xc = np.mean(X)
    yc = np.mean(Y)

    inside_x = abs(xc - x0) <= slit_width / 2.0
    inside_y = abs(yc - y0) <= slit_height / 2.0

    return inside_x and inside_y


def fits_field_multiwave(
    ln, fx, fy, Fr, pupil_points, surf, waves,
    x0, y0, slit_width, slit_height
):
    """
    A field is accepted when the center of the spot for every
    wavelength remains inside the slit.
    """

    for wave in waves:

        X, Y = trace_real_field(
            ln=ln,
            fx=fx,
            fy=fy,
            Fr=Fr,
            pupil_points=pupil_points,
            surf=surf,
            wave=wave
        )

        if len(X) == 0:
            return False

        if not spot_center_fits_in_slit(
            X, Y,
            x0, y0,
            slit_width,
            slit_height
        ):
            return False

    return True


def binary_search_fx_max(
    ln, Fr, pupil_points, surf, waves,
    x0, y0, slit_width, slit_height,
    fx_low, fx_high, tol=1e-8, max_iter=50
):
    for _ in range(max_iter):
        fx_mid = 0.5 * (fx_low + fx_high)

        if fits_field_multiwave(ln, fx_mid, 0.0, Fr, pupil_points, surf, waves, x0, y0, slit_width, slit_height):
            fx_low = fx_mid
        else:
            fx_high = fx_mid

        if abs(fx_high - fx_low) < tol:
            break

    return fx_low


def binary_search_fy_max(
    ln, Fr, pupil_points, surf, waves,
    x0, y0, slit_width, slit_height,
    fy_low, fy_high, tol=1e-8, max_iter=50
):
    for _ in range(max_iter):
        fy_mid = 0.5 * (fy_low + fy_high)

        if fits_field_multiwave(ln, 0.0, fy_mid, Fr, pupil_points, surf, waves, x0, y0, slit_width, slit_height):
            fy_low = fy_mid
        else:
            fy_high = fy_mid

        if abs(fy_high - fy_low) < tol:
            break

    return fy_low

def evaluate_field_multiwave(
    ln, fx, fy, Fr, pupil_points, surf, waves,
    x0, y0, slit_width, slit_height
):
    """
    Evaluate a field at all wavelengths and store the
    ray footprints for subsequent animation.
    """

    traces = []
    field_accepted = True

    for wave in waves:

        X, Y = trace_real_field(
            ln=ln,
            fx=fx,
            fy=fy,
            Fr=Fr,
            pupil_points=pupil_points,
            surf=surf,
            wave=wave
        )

        if len(X) == 0:
            field_accepted = False
            traces.append((X, Y, np.nan, np.nan))
            continue

        xc, yc = spot_centroid(X, Y)

        traces.append((X, Y, xc, yc))

        inside_x = abs(xc - x0) <= slit_width / 2.0
        inside_y = abs(yc - y0) <= slit_height / 2.0

        if not (inside_x and inside_y):
            field_accepted = False

    return field_accepted, traces

def binary_search_fy_max_with_history(
    ln, Fr, pupil_points, surf, waves,
    x0, y0, slit_width, slit_height,
    fy_low, fy_high, tol=1e-8, max_iter=50
):
    """
    Find the maximum admissible Y field using binary search.
    
    Store the results of each iteration for GIF animation.
    """

    history = []

    for iteration in range(max_iter):

        fy_mid = 0.5 * (fy_low + fy_high)

        accepted, traces = evaluate_field_multiwave(
            ln=ln,
            fx=0.0,
            fy=fy_mid,
            Fr=Fr,
            pupil_points=pupil_points,
            surf=surf,
            waves=waves,
            x0=x0,
            y0=y0,
            slit_width=slit_width,
            slit_height=slit_height
        )

        # Store the results of the current iteration
        history.append({
            "iteration": iteration + 1,
            "fy": fy_mid,
            "accepted": accepted,
            "traces": traces
        })

        if accepted:
            fy_low = fy_mid
        else:
            fy_high = fy_mid

        if abs(fy_high - fy_low) < tol:
            break

    return fy_low, history

def spot_fits_in_slit(X, Y, x0, y0, slit_width, slit_height):
    if len(X) == 0:
        return False

    xmin, xmax = X.min(), X.max()
    ymin, ymax = Y.min(), Y.max()

    inside_x = (xmin >= x0 - slit_width / 2.0) and (xmax <= x0 + slit_width / 2.0)
    inside_y = (ymin >= y0 - slit_height / 2.0) and (ymax <= y0 + slit_height / 2.0)

    return inside_x and inside_y


def fits_field_multiwave_full_spot(
    ln, fx, fy, Fr, pupil_points, surf, waves,
    x0, y0, slit_width, slit_height
):
    all_X = []
    all_Y = []

    for wave in waves:
        X, Y = trace_real_field(
            ln=ln,
            fx=fx,
            fy=fy,
            Fr=Fr,
            pupil_points=pupil_points,
            surf=surf,
            wave=wave
        )

        if len(X) > 0:
            all_X.extend(X)
            all_Y.extend(Y)

    all_X = np.array(all_X)
    all_Y = np.array(all_Y)

    return spot_fits_in_slit(
        all_X, all_Y, x0, y0, slit_width, slit_height
    )

def binary_search_fx_max_full_spot(
    ln, Fr, pupil_points, surf, waves,
    x0, y0, slit_width, slit_height,
    fx_low, fx_high, tol=1e-8, max_iter=50
):
    for _ in range(max_iter):
        fx_mid = 0.5 * (fx_low + fx_high)

        if fits_field_multiwave_full_spot(
            ln, fx_mid, 0.0, Fr, pupil_points, surf, waves,
            x0, y0, slit_width, slit_height
        ):
            fx_low = fx_mid
        else:
            fx_high = fx_mid

        if abs(fx_high - fx_low) < tol:
            break

    return fx_low


def binary_search_fy_max_full_spot(
    ln, Fr, pupil_points, surf, waves,
    x0, y0, slit_width, slit_height,
    fy_low, fy_high, tol=1e-8, max_iter=50
):
    for _ in range(max_iter):
        fy_mid = 0.5 * (fy_low + fy_high)

        if fits_field_multiwave_full_spot(
            ln, 0.0, fy_mid, Fr, pupil_points, surf, waves,
            x0, y0, slit_width, slit_height
        ):
            fy_low = fy_mid
        else:
            fy_high = fy_mid

        if abs(fy_high - fy_low) < tol:
            break

    return fy_low

def spot_centroid(X, Y):
    return np.mean(X), np.mean(Y)

def smooth_curve(x, y, n_points=200, k=3):
    """
    Smooth a curve using cubic spline interpolation.
    Falls back to linear interpolation if there are too few points.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]

    x_smooth = np.linspace(x_sorted.min(), x_sorted.max(), n_points)

    if len(x_sorted) > k:
        spline = make_interp_spline(x_sorted, y_sorted, k=k)
        y_smooth = spline(x_smooth)
    else:
        y_smooth = np.interp(x_smooth, x_sorted, y_sorted)

    return x_smooth, y_smooth


def trace_field_centroid(
    ln,
    fx,
    fy,
    Fr,
    pupil_points,
    surf,
    wave
):
    """
    Trace a field and calculate its spot centroid.

    Parameters
    ----------
    ln : PyZDDE link
        Active Zemax connection.
    fx, fy : float
        Field coordinates in degrees.
    Fr : float
        Field normalization factor.
    pupil_points : array-like
        Normalized pupil coordinates.
    surf : int
        Target optical surface.
    wave : int
        Zemax wavelength index.

    Returns
    -------
    xc, yc : float
        Spot centroid coordinates in millimeters.
    """

    X, Y = trace_real_field(
        ln=ln,
        fx=fx,
        fy=fy,
        Fr=Fr,
        pupil_points=pupil_points,
        surf=surf,
        wave=wave
    )

    return spot_centroid(X, Y)