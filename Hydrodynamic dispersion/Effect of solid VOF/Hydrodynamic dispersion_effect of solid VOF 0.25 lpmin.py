

# =====================================================================
# HYDRODYNAMIC DISPERSION FROM STB TRACKS — SVF SWEEP (MINIMAL CHANGES)
# =====================================================================
# Purpose:
#   - Read DaVis 3D particle tracks (x,y,z) for several solid VOF (SVF) cases
#   - Compute ensemble MSD (axial y; radial x+z) and fluctuation-corrected MSD
#   - Fit dispersion D from linear tail over the last 'fit_window_duration' sec
#   - Save per-case summary (Excel) and figures (SVG+PNG)
#
# Key change from baseline:
#   - The comparison axis is now SOLID VOF (%) at a fixed gas rate (0.05 L/min).
#     Legends, plot titles, and Excel columns reflect SVF instead of gas flow.
#
# Outputs:
#   - Axial/Radial MSD (normal & fluctuation-corrected), sci y-axis
#   - Dispersion vs SVF (normal) — dashed lines (axial & radial)
#   - Dispersion vs SVF (fluctuation-corrected) — dashed lines (axial & radial)
#   - MSD statistics (increments per lag): semi-log-Y
#   - Excel summary + compact "Dispersion_vs_SVF.xlsx"
#
# Author: Hooman Eslami  |  Last edit: 2025-10-27
# =====================================================================

# ----------------------------- IMPORTS --------------------------------
import os                                  # filesystem utils
from pathlib import Path                    # robust paths
from collections import Counter             # track length histogram
import numpy as np                          # numerics
import pandas as pd                         # Excel I/O
import matplotlib.pyplot as plt             # plotting
from matplotlib.ticker import ScalarFormatter  # scientific axis
import lvpyio as lv                         # DaVis particle I/O

# ----------------------------- CONFIG ---------------------------------
fps = 198.4                     # [Hz] camera frame rate
dt = 1.0 / fps                  # [s] time between frames
analysis_window_s = 1.0         # [s] max lag τ to analyze
fit_window_duration = 0.2       # [s] tail length for linear fit (baseline kept)

px_per_mm = 5.84                # [px/mm] calibration
mm_per_px = 1.0 / px_per_mm     # [mm/px]
mm2_per_px2 = mm_per_px**2      # [mm^2/px^2]
MM2_TO_M2 = 1e-6                # [m^2/mm^2] conversion

# --- Fixed gas rate for this comparison (informational) ---
gas_rate_Lmin = 0.25            # [L/min] same for all SVF cases

# --- Dataset folders: SVF sweep at 0.05 L/min ---
data_paths = [
    r"E:\Three-phase Hydrogels\track data_dispersion\Three-phase\0%\0.25 lpermin\dipersion calculation_0% (images 500-1000)_0.25 lpmin",
    r"E:\Three-phase Hydrogels\track data_dispersion\Three-phase\10%\0.25 lpermin\dipersion calculation_10% (images 500-1000)_0.25 lpmin",
    r"E:\Three-phase Hydrogels\track data_dispersion\Three-phase\30%\0.25 lpermin\dipersion calculation_30% (images 500-1000)_0.25 lpmin",
    r"E:\Three-phase Hydrogels\track data_dispersion\Three-phase\50%\0.25 lpermin\dipersion calculation_50% (images 500-1000)_0.25 lpmin",

]
svf_levels_pct = [0, 10, 30, 50]           # [%] legend/x-axis for SVF

# --- Output directory (specific to SVF sweep and gas rate) ---
output_dir = r"C:\python files\Hydrogel_Three-phase\Figures\New results\Hydrodynamic dispersions\Effect of solid VOF\0.25 Lpmin"
Path(output_dir).mkdir(parents=True, exist_ok=True)  # ensure folder exists

# --- Fast testing (limit number of tracks per dataset); set to None for full run ---
FAST_DEBUG_MAX_TRACKS = None   # e.g., 150 for quick test; None to use all tracks

# --- Sliding-window α selection (unweighted) ---------------------------
ALPHA_WINDOW_S = 0.2            # [s] window length for local α fit (log–log)
ALPHA_TOL = 0.10                # accept |α-1| <= ALPHA_TOL as "diffusive"
MIN_WIN_PTS = 6                 # minimum samples inside a window
START_FIT_TAU = 0.4             # [s] do NOT consider windows that start before this τ
PREFER_LATE = True              # tie-break: prefer spans closer to the tail

# ------------------------- PLOT STYLE ---------------------------------
# - Sizes/fonts tuned for A4 two-up (~85–90 mm wide per figure)
plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300,          # hi-res export
    "font.family": "sans-serif",
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,  # slightly larger
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.9, "grid.linewidth": 0.4
})
FIG_W, FIG_H = 3.54, 2.50       # ~90 mm × 63 mm (works side-by-side on A4)
LINE_W = 1.4                    # line width for solid curves
MARKER_SZ = 4.0                 # marker size for SVF scatter

# ---------------------------- HELPERS ---------------------------------
def tukey_box_from_hist(length_counter: Counter):
    """
    Compute compact Tukey stats (quartiles, whiskers) from a histogram
    of track lengths. Returns dict with counts and stats.
    """
    if not length_counter:  # if empty, return NaNs/zeros
        return dict(n_tracks=0, n_particles=0, min=np.nan, q1=np.nan, median=np.nan,
                    q3=np.nan, max=np.nan, whisker_low=np.nan, whisker_high=np.nan,
                    mean=np.nan, std=np.nan)
    Ls = np.array(sorted(length_counter.keys()), dtype=np.int64)   # unique lengths
    Cs = np.array([length_counter[L] for L in Ls], dtype=np.int64) # their counts
    n_tracks = int(Cs.sum())                                       # total tracks
    n_particles = int((Ls * Cs).sum())                              # total points
    mn, mx = int(Ls[0]), int(Ls[-1])                               # min/max length
    mean = (Ls * Cs).sum() / n_tracks                              # mean length
    mean2 = (Ls**2 * Cs).sum() / n_tracks                          # mean square
    std = float(np.sqrt(max(0.0, mean2 - mean**2)))                # std dev
    cdf = np.cumsum(Cs)                                            # cumulative
    def qtile(q):                                                  # quantile helper
        target = q * n_tracks
        idx = np.searchsorted(cdf, target, side="left")
        idx = min(idx, len(Ls) - 1)
        return int(Ls[idx])
    q1, med, q3 = qtile(0.25), qtile(0.50), qtile(0.75)            # quartiles
    iqr = q3 - q1                                                  # interquartile range
    wl_thr, wh_thr = q1 - 1.5*iqr, q3 + 1.5*iqr                    # Tukey whiskers
    wl_idx = np.searchsorted(Ls, wl_thr, side="left")              # whisker low idx
    wh_idx = np.searchsorted(Ls, wh_thr, side="right") - 1         # whisker high idx
    wl_idx = min(max(wl_idx, 0), len(Ls)-1)                        # clamp
    wh_idx = min(max(wh_idx, 0), len(Ls)-1)                        # clamp
    wlow, whigh = int(Ls[wl_idx]), int(Ls[wh_idx])                 # whisker values
    return dict(n_tracks=n_tracks, n_particles=n_particles,        # package stats
                min=mn, q1=q1, median=med, q3=q3, max=mx,
                whisker_low=wlow, whisker_high=whigh,
                mean=float(mean), std=std)

def _best_span_alpha_unweighted(tau, msd, win_s=0.2, alpha_tol=0.10,
                                min_pts=6, start_tau=0.0, prefer_late=True):
    """
    Slide a window of length win_s along (tau, msd) and in each window fit:
        log(MSD) = c + α * log(tau)
    Mark windows where |α - 1| <= alpha_tol, then choose the best span
    (prefer later and longer). Return selected span and slope on linear scale.
    """
    m = np.isfinite(tau) & np.isfinite(msd) & (tau > 0) & (msd > 0)  # valid positive data
    if m.sum() < min_pts:                                           # require enough points
        return dict(ok=False)
    t = tau[m]; y = msd[m]                                          # filtered series
    dt_local = np.median(np.diff(t))                                # local Δτ
    if not np.isfinite(dt_local) or dt_local <= 0:                  # sanity check
        return dict(ok=False)
    nwin = max(int(round(win_s / dt_local)), min_pts)               # points/window
    if nwin > t.size:                                               # fallback (too short)
        a, b = np.polyfit(t, y, 1)
        return dict(ok=True, tau0=float(t[0]), tau1=float(t[-1]), slope=float(a))
    alphas = []                                                     # local α list
    spans = []                                                      # window index pairs
    for i0 in range(0, t.size - nwin + 1):                          # slide windows
        i1 = i0 + nwin                                             # exclusive end
        if t[i0] < start_tau:                                      # enforce startτ threshold
            continue
        xt = np.log(t[i0:i1]); yt = np.log(y[i0:i1])               # log–log arrays
        p = np.polyfit(xt, yt, 1)                                  # α = slope
        alphas.append(float(p[0])); spans.append((i0, i1))         # cache α and span
    if not alphas:                                                  # no eligible windows
        return dict(ok=False)
    alphas = np.asarray(alphas)                                     # ndarray
    good = np.abs(alphas - 1.0) <= alpha_tol                        # diffusive mask
    best = None                                                     # best run cache
    run_start = None                                                # current run start
    for k, g in enumerate(good):                                    # scan runs
        if g and run_start is None:
            run_start = k
        if (not g or k == len(good)-1) and run_start is not None:   # run ended
            k_end = k if not g else k                               # inclusive end
            i0 = spans[run_start][0]; i1 = spans[k_end][1]          # union indices
            length_s = t[i1-1] - t[i0]                              # run length
            end_time = t[i1-1]                                      # right edge time
            score = (end_time if prefer_late else 0.0, length_s)    # score
            if (best is None) or (score > best[0]):                 # keep best
                best = (score, (i0, i1))
            run_start = None
    if best is None:                                                # no α≈1 run found
        idx = int(np.argmin(np.abs(alphas - 1.0)))                  # closest-to-1
        i0, i1 = spans[idx]
    else:
        i0, i1 = best[1]
    a, b = np.polyfit(t[i0:i1], y[i0:i1], 1)                        # slope on span
    return dict(ok=True, tau0=float(t[i0]), tau1=float(t[i1-1]), slope=float(a))

# --------------------------- RUNTIME LOGS ------------------------------
max_time_intervals = int(analysis_window_s / dt)  # number of lag steps
print(f"✅ Frame interval: {dt:.6f} s")                       # report dt
print(f"✅ Number of time intervals to analyze: {max_time_intervals}")  # report count
print(f"✅ Calibration: {px_per_mm:.2f} px/mm  (1 px = {mm_per_px:.6f} mm)")  # calib
print(f"✅ Fixed gas rate for this sweep: {gas_rate_Lmin:.2f} L/min")         # gas rate

# ------------------------- PER-CASE STORAGE ---------------------------
all_time_sec = []        # τ arrays (s)
all_msd_ax_mm2 = []      # axial MSD (normal)
all_msd_ra_mm2 = []      # radial MSD (normal)
all_msd_ax_fl_mm2 = []   # axial MSD (fluctuation-corrected)
all_msd_ra_fl_mm2 = []   # radial MSD (fluctuation-corrected)
all_counts = []          # increments per lag
summary = []             # rows for Excel

# Selected spans for dashed overlay (per curve)
sel_ax = []; sel_ra = []; sel_ax_fl = []; sel_ra_fl = []  # span caches

# ============================== CORE LOOP ==============================
for svf, path in zip(svf_levels_pct, data_paths):          # loop SVF cases
    # ---- Context log ----
    print("\n------------------------------------------------------------")
    print(f"🔎 Solid VOF (SVF): {svf:d}%   |   Gas: {gas_rate_Lmin:.2f} L/min")
    print(f"📂 Folder: {path}")

    # ---- Load tracks ----
    tr = lv.read_particles(path)             # read the LV particles folder
    tracks = tr.tracks()                      # get list of tracks
    print(f"   Tracks found: {len(tracks)}")  # log count

    # ---- Accumulators (normal) ----
    msd_ax_sum_px2 = np.zeros(max_time_intervals)  # ∑(Δy)^2 per lag
    msd_ra_sum_px2 = np.zeros(max_time_intervals)  # ∑[(Δx)^2+(Δz)^2] per lag

    # ---- Accumulators (fluctuation-corrected) ----
    msd_ax_fl_sum_px2 = np.zeros(max_time_intervals)  # ∑(Δy - kdt*vȳ)^2
    msd_ra_fl_sum_px2 = np.zeros(max_time_intervals)  # ∑[(Δx - kdt*vx̄)^2 + (Δz - kdt*vz̄)^2]

    # ---- Counts per lag ----
    counts    = np.zeros(max_time_intervals, dtype=int)  # normal counts
    counts_fl = np.zeros(max_time_intervals, dtype=int)  # mean-free counts

    # ---- Track length stats ----
    length_hist = Counter()              # histogram of track lengths
    particles_used = 0                   # total used particles
    MIN_LEN = 5                          # minimum track length to accept

    # ---- Loop tracks ----
    used = 0                             # how many tracks processed (fast mode)
    for t_i, tck in enumerate(tracks):   # iterate tracks
        if (FAST_DEBUG_MAX_TRACKS is not None) and (used >= FAST_DEBUG_MAX_TRACKS):
            break                        # stop early in fast testing mode
        P = tck.particles                # structured array with x,y,z
        x = P["x"]; y = P["y"]; z = P["z"]     # position arrays (px)
        n = len(x)                       # points in this track
        if n <= MIN_LEN:                 # skip too-short tracks
            continue
        length_hist[n] += 1              # update histogram
        particles_used += n              # update particle count
        used += 1                        # count this track

        total_T = (n - 1) * dt           # total time span of this track
        if total_T <= 0:                 # guard
            continue
        vx_bar = (x[-1] - x[0]) / total_T  # mean velocities for mean-free MSD
        vy_bar = (y[-1] - y[0]) / total_T
        vz_bar = (z[-1] - z[0]) / total_T

        max_dt_here = min(max_time_intervals, n)  # max lag for this track
        for k in range(1, max_dt_here):           # loop lags (k=1..)
            dx = x[k:] - x[:-k]                   # Δx over lag k
            dy = y[k:] - y[:-k]                   # Δy over lag k
            dz = z[k:] - z[:-k]                   # Δz over lag k

            # Normal MSD terms
            dy2 = dy**2                           # axial squared increment
            dr2 = dx**2 + dz**2                   # radial squared increment

            # Mean-free increments (remove drift)
            kdt = k * dt                          # lag time
            dx_fl = dx - kdt * vx_bar             # x mean-free
            dy_fl = dy - kdt * vy_bar             # y mean-free
            dz_fl = dz - kdt * vz_bar             # z mean-free

            dy_fl2 = dy_fl**2                     # axial mean-free squared
            dr_fl2 = dx_fl**2 + dz_fl**2          # radial mean-free squared

            # Accumulate sums for ensemble average
            msd_ax_sum_px2[k]    += dy2.sum()     # sum axial squared increments
            msd_ra_sum_px2[k]    += dr2.sum()     # sum radial squared increments
            msd_ax_fl_sum_px2[k] += dy_fl2.sum()  # sum axial mean-free squared
            msd_ra_fl_sum_px2[k] += dr_fl2.sum()  # sum radial mean-free squared

            ninc = dy2.size                        # number of increments at this lag
            counts[k]    += ninc                   # update counts (normal)
            counts_fl[k] += ninc                   # update counts (mean-free)

    # ---- Finalize ensemble means ----
    valid    = counts    > 0                           # lags with data
    valid_fl = counts_fl > 0                           # lags with data (mean-free)

    msd_ax_px2    = np.full_like(msd_ax_sum_px2,    np.nan, dtype=float)  # init NaNs
    msd_ra_px2    = np.full_like(msd_ra_sum_px2,    np.nan, dtype=float)
    msd_ax_fl_px2 = np.full_like(msd_ax_fl_sum_px2, np.nan, dtype=float)
    msd_ra_fl_px2 = np.full_like(msd_ra_fl_sum_px2, np.nan, dtype=float)

    msd_ax_px2[valid]       = msd_ax_sum_px2[valid]       / counts[valid]        # ⟨Δy²⟩
    msd_ra_px2[valid]       = msd_ra_sum_px2[valid]       / counts[valid]        # ⟨Δx²+Δz²⟩
    msd_ax_fl_px2[valid_fl] = msd_ax_fl_sum_px2[valid_fl] / counts_fl[valid_fl]  # mean-free axial
    msd_ra_fl_px2[valid_fl] = msd_ra_fl_sum_px2[valid_fl] / counts_fl[valid_fl]  # mean-free radial

    # ---- Units: px^2 -> mm^2 ----
    msd_ax_mm2    = msd_ax_px2    * mm2_per_px2          # axial in mm^2
    msd_ra_mm2    = msd_ra_px2    * mm2_per_px2          # radial in mm^2
    msd_ax_fl_mm2 = msd_ax_fl_px2 * mm2_per_px2          # axial' in mm^2
    msd_ra_fl_mm2 = msd_ra_fl_px2 * mm2_per_px2          # radial' in mm^2

    # ---- Lag axis τ (s) ----
    tau = np.arange(max_time_intervals) * dt             # τ = k*dt

    # ---- Store for plotting ----
    all_time_sec.append(tau)                              # keep τ
    all_msd_ax_mm2.append(msd_ax_mm2)                     # store axial
    all_msd_ra_mm2.append(msd_ra_mm2)                     # store radial
    all_msd_ax_fl_mm2.append(msd_ax_fl_mm2)               # store axial'
    all_msd_ra_fl_mm2.append(msd_ra_fl_mm2)               # store radial'
    all_counts.append(counts.copy())                       # store counts

    # ---- Sliding 0.2 s, α≈1 selection; simple slope over selected span ----
    info_ax  = _best_span_alpha_unweighted(tau, msd_ax_mm2,    win_s=ALPHA_WINDOW_S, alpha_tol=ALPHA_TOL,
                                           min_pts=MIN_WIN_PTS, start_tau=START_FIT_TAU, prefer_late=PREFER_LATE)
    info_ra  = _best_span_alpha_unweighted(tau, msd_ra_mm2,    win_s=ALPHA_WINDOW_S, alpha_tol=ALPHA_TOL,
                                           min_pts=MIN_WIN_PTS, start_tau=START_FIT_TAU, prefer_late=PREFER_LATE)
    info_axF = _best_span_alpha_unweighted(tau, msd_ax_fl_mm2, win_s=ALPHA_WINDOW_S, alpha_tol=ALPHA_TOL,
                                           min_pts=MIN_WIN_PTS, start_tau=START_FIT_TAU, prefer_late=PREFER_LATE)
    info_raF = _best_span_alpha_unweighted(tau, msd_ra_fl_mm2, win_s=ALPHA_WINDOW_S, alpha_tol=ALPHA_TOL,
                                           min_pts=MIN_WIN_PTS, start_tau=START_FIT_TAU, prefer_late=PREFER_LATE)

    # Save spans for plotting overlays
    sel_ax.append((info_ax.get("tau0", None),  info_ax.get("tau1", None)))     # axial
    sel_ra.append((info_ra.get("tau0", None),  info_ra.get("tau1", None)))     # radial
    sel_ax_fl.append((info_axF.get("tau0", None), info_axF.get("tau1", None))) # axial'
    sel_ra_fl.append((info_raF.get("tau0", None), info_raF.get("tau1", None))) # radial'

    # Convert slope (mm^2/s) -> dispersion D (m^2/s)
    D_ax_m2_s    = (info_ax.get("slope", np.nan)  / 2.0) * MM2_TO_M2
    D_ra_m2_s    = (info_ra.get("slope", np.nan)  / 2.0) * MM2_TO_M2
    D_ax_fl_m2_s = (info_axF.get("slope", np.nan) / 2.0) * MM2_TO_M2
    D_ra_fl_m2_s = (info_raF.get("slope", np.nan) / 2.0) * MM2_TO_M2

    # ---- Track length stats ----
    bx = tukey_box_from_hist(length_hist)                 # compute Tukey stats

    # ---- Append summary row (indexed by SVF) ----
    summary.append({
        "SVF_percent": svf,
        "gas_L_per_min": gas_rate_Lmin,
        "tracks_used": bx["n_tracks"],
        "particles_used": bx["n_particles"],
        "tracklen_min": bx["min"],
        "tracklen_q1": bx["q1"],
        "tracklen_median": bx["median"],
        "tracklen_q3": bx["q3"],
        "tracklen_max": bx["max"],
        "tracklen_whisker_low": bx["whisker_low"],
        "tracklen_whisker_high": bx["whisker_high"],
        "tracklen_mean": bx["mean"],
        "tracklen_std": bx["std"],
        "D_axial_m2_per_s": D_ax_m2_s,
        "D_radial_m2_per_s": D_ra_m2_s,
        "D_axial_fluct_m2_per_s": D_ax_fl_m2_s,
        "D_radial_fluct_m2_per_s": D_ra_fl_m2_s
    })

    # ---- Diagnostic print ----
    def _span_str(info):                                       # span summary
        return "span: n/a" if not info.get("ok", False) else f"span: [{info['tau0']:.3f}, {info['tau1']:.3f}] s"
    print(f"   Tracks used: {bx['n_tracks']} | Particles used: {bx['n_particles']}")
    print(f"   Track length Q1/Med/Q3: {bx['q1']} / {bx['median']} / {bx['q3']}")
    print(f"   Axial D  = {D_ax_m2_s:.4e} m²/s ({_span_str(info_ax)})")
    print(f"   Radial D = {D_ra_m2_s:.4e} m²/s ({_span_str(info_ra)})")
    print(f"   Axial D' = {D_ax_fl_m2_s:.4e} m²/s ({_span_str(info_axF)})")
    print(f"   Radial D'= {D_ra_fl_m2_s:.4e} m²/s ({_span_str(info_raF)})")

# ============================== SAVE EXCEL =============================
excel_summary_path = os.path.join(output_dir, "PTV_dispersion_summary.xlsx")  # path
pd.DataFrame(summary).to_excel(excel_summary_path, index=False)               # write file
print(f"\n📗 Excel summary saved: {excel_summary_path}")                       # log

dispersion_only_path = os.path.join(output_dir, "Dispersion_vs_SVF.xlsx")    # compact file
pd.DataFrame({
    "SVF_percent": [row["SVF_percent"] for row in summary],
    "D_axial_m2_per_s": [row["D_axial_m2_per_s"] for row in summary],
    "D_radial_m2_per_s": [row["D_radial_m2_per_s"] for row in summary],
    "D_axial_fluct_m2_per_s": [row["D_axial_fluct_m2_per_s"] for row in summary],
    "D_radial_fluct_m2_per_s": [row["D_radial_fluct_m2_per_s"] for row in summary],
}).to_excel(dispersion_only_path, index=False)
print(f"📘 Dispersion-only Excel saved: {dispersion_only_path}")              # log

# ================================ FIGURES ==============================
# NOTE: MSD arrays are in mm^2; multiply by 1e-6 to show m^2 on plots.

def _overlay_selected_span(tau, msd_mm2, span, color):
    """
    Dashed overlay on top of the selected time span.
    - Re-draw the solid segment beneath at alpha=0.2 (80% invisible) for visibility.
    - Draw a slightly bolder dashed line on top.
    """
    tau0, tau1 = span                                         # unpack span
    if tau0 is None or tau1 is None or not np.isfinite(tau0) or not np.isfinite(tau1):
        return                                                # nothing to draw
    mask = np.isfinite(msd_mm2) & (tau >= tau0) & (tau <= tau1)  # segment mask
    if mask.sum() < 2:                                        # need 2 points
        return
    plt.plot(tau[mask], msd_mm2[mask] * MM2_TO_M2, '-',       # faded solid underlay
             linewidth=LINE_W, color=color, alpha=0.2, zorder=3)
    plt.plot(tau[mask], msd_mm2[mask] * MM2_TO_M2, '--',      # bold dashed overlay
             linewidth=LINE_W + 0.9, color=color, alpha=1.0, zorder=4)

# --- RADIAL MSD (NORMAL) — sci y-axis; legend by SVF ---
plt.figure(figsize=(FIG_W, FIG_H))                            # new fig
for svf, tau, msd_mm2, span in zip(svf_levels_pct, all_time_sec, all_msd_ra_mm2, sel_ra):
    line, = plt.plot(tau, msd_mm2 * MM2_TO_M2,                # main solid curve
                     label=f"{svf:d}% SVF", linewidth=LINE_W, zorder=1)
    _overlay_selected_span(tau, msd_mm2, span, line.get_color())  # dashed overlay
plt.xlabel("Time interval, $\\tau$ (s)")                      # x label
plt.ylabel("Radial MSD (m$^2$)")                              # y label
plt.title(f"Radial MSD vs $\\tau$  (Gas = {gas_rate_Lmin:.2f} L/min)")  # title
ax = plt.gca()                                                # axis handle


# --- NEW: Scalar formatter with one decimal, keeps ×10^n --------------
class OneDecScalarFormatter(ScalarFormatter):
    """ScalarFormatter that keeps scientific notation and uses 1 decimal."""
    def _set_format(self, vmin=None, vmax=None):
        self.format = '%.1f'               # one decimal in mantissa
        
sf = OneDecScalarFormatter(useMathText=True)                  # scalar formatter (sci)
sf.set_powerlimits((-2, 2))                                   # power limits for ×10^n
sf.set_useOffset(False)                                       # no offset term
ax.yaxis.set_major_formatter(sf)                              # apply formatter

# ---- Y-axis formatter: sci notation with ×10ⁿ and 1 decimal ----
plt.grid(True, alpha=0.3)                                     # grid
plt.legend(frameon=False, title="Solid VOF", loc="best")      # legend
plt.tight_layout()                                            # layout
base = os.path.join(output_dir, "Radial_MSD_vs_Time_SVF")     # base path
plt.savefig(base + ".svg", bbox_inches="tight")               # save svg
plt.savefig(base + ".png", bbox_inches="tight", dpi=300)      # save png
plt.close()                                                   # close
print(f"🖼️ Saved: {base}.svg, {base}.png")                    # log

# --- AXIAL MSD (NORMAL) — sci y-axis; legend by SVF ---
plt.figure(figsize=(FIG_W, FIG_H))                            # new fig
for svf, tau, msd_mm2, span in zip(svf_levels_pct, all_time_sec, all_msd_ax_mm2, sel_ax):
    line, = plt.plot(tau, msd_mm2 * MM2_TO_M2, label=f"{svf:d}% SVF", linewidth=LINE_W, zorder=1)
    _overlay_selected_span(tau, msd_mm2, span, line.get_color())
plt.xlabel("Time interval, $\\tau$ (s)")                      # x label
plt.ylabel("Axial MSD (m$^2$)")                               # y label
plt.title(f"Axial MSD vs $\\tau$  (Gas = {gas_rate_Lmin:.2f} L/min)")  # title
ax = plt.gca()                                                # axis handle

# ---- Y-axis formatter: sci notation with ×10ⁿ and 1 decimal ----
sf = OneDecScalarFormatter(useMathText=True)
sf.set_powerlimits((-2, 2))
sf.set_useOffset(False)
ax.yaxis.set_major_formatter(sf)

plt.grid(True, alpha=0.3)                                     # grid
plt.legend(frameon=False, title="Solid VOF", loc="best")      # legend
plt.tight_layout()                                            # layout
base = os.path.join(output_dir, "Axial_MSD_vs_Time_SVF")      # base path
plt.savefig(base + ".svg", bbox_inches="tight")               # save svg
plt.savefig(base + ".png", bbox_inches="tight", dpi=300)      # save png
plt.close()                                                   # close
print(f"🖼️ Saved: {base}.svg, {base}.png")                    # log

# --- RADIAL MSD′ (FLUCTUATION-CORRECTED) — sci y-axis ---
plt.figure(figsize=(FIG_W, FIG_H))                            # new fig
for svf, tau, msd_mm2, span in zip(svf_levels_pct, all_time_sec, all_msd_ra_fl_mm2, sel_ra_fl):
    line, = plt.plot(tau, msd_mm2 * MM2_TO_M2, label=f"{svf:d}% SVF", linewidth=LINE_W, zorder=1)
    _overlay_selected_span(tau, msd_mm2, span, line.get_color())
plt.xlabel("Time interval, $\\tau$ (s)")                      # x label
plt.ylabel("Radial MSD′ (m$^2$)")                             # y label
plt.title(f"Radial MSD′ (mean-free) vs $\\tau$  (Gas = {gas_rate_Lmin:.2f} L/min)")  # title
ax = plt.gca()                                                # axis handle

# ---- Y-axis formatter: sci notation with ×10ⁿ and 1 decimal ----
sf = OneDecScalarFormatter(useMathText=True)
sf.set_powerlimits((-2, 2))
sf.set_useOffset(False)
ax.yaxis.set_major_formatter(sf)

plt.grid(True, alpha=0.3)                                     # grid
plt.legend(frameon=False, title="Solid VOF", loc="best")      # legend
plt.tight_layout()                                            # layout
base = os.path.join(output_dir, "Radial_MSD_fluct_vs_Time_SVF")  # base path
plt.savefig(base + ".svg", bbox_inches="tight")               # save svg
plt.savefig(base + ".png", bbox_inches="tight", dpi=300)      # save png
plt.close()                                                   # close
print(f"🖼️ Saved: {base}.svg, {base}.png")                    # log

# --- AXIAL MSD′ (FLUCTUATION-CORRECTED) — sci y-axis ---
plt.figure(figsize=(FIG_W, FIG_H))                            # new fig
for svf, tau, msd_mm2, span in zip(svf_levels_pct, all_time_sec, all_msd_ax_fl_mm2, sel_ax_fl):
    line, = plt.plot(tau, msd_mm2 * MM2_TO_M2, label=f"{svf:d}% SVF", linewidth=LINE_W, zorder=1)
    _overlay_selected_span(tau, msd_mm2, span, line.get_color())
plt.xlabel("Time interval, $\\tau$ (s)")                      # x label
plt.ylabel("Axial MSD′ (m$^2$)")                              # y label
plt.title(f"Axial MSD′ (mean-free) vs $\\tau$  (Gas = {gas_rate_Lmin:.2f} L/min)")  # title
ax = plt.gca()                                                # axis handle

# ---- Y-axis formatter: sci notation with ×10ⁿ and 1 decimal ----
sf = OneDecScalarFormatter(useMathText=True)
sf.set_powerlimits((-2, 2))
sf.set_useOffset(False)
ax.yaxis.set_major_formatter(sf)

plt.grid(True, alpha=0.3)                                     # grid
plt.legend(frameon=False, title="Solid VOF", loc="best")      # legend
plt.tight_layout()                                            # layout
base = os.path.join(output_dir, "Axial_MSD_fluct_vs_Time_SVF")  # base path
plt.savefig(base + ".svg", bbox_inches="tight")               # save svg
plt.savefig(base + ".png", bbox_inches="tight", dpi=300)      # save png
plt.close()                                                   # close
print(f"🖼️ Saved: {base}.svg, {base}.png")                    # log

# --- DISPERSION vs SVF (NORMAL) — dashed lines (axial & radial) ---
plt.figure(figsize=(FIG_W, FIG_H))                            # new fig
Da = [row["D_axial_m2_per_s"] for row in summary]            # axial D list
Dr = [row["D_radial_m2_per_s"] for row in summary]           # radial D list
plt.plot(svf_levels_pct, Da, "o--", linewidth=LINE_W, markersize=MARKER_SZ, label="Axial $D$")
plt.plot(svf_levels_pct, Dr, "s--", linewidth=LINE_W, markersize=MARKER_SZ, label="Radial $D$")
plt.xlabel("Solid VOF (%)")                                   # x label
plt.ylabel("Dispersion coefficient (m$^2$/s)")                # y label
plt.title(f"Effective dispersion vs SVF (normal MSD)  —  Gas = {gas_rate_Lmin:.2f} L/min)")  # title
ax = plt.gca()                                                # axis handle
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))          # sci fmt (unchanged)
ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))            # sci ticks (unchanged)
plt.grid(True, alpha=0.3)                                     # grid
plt.legend(frameon=False, loc="best")                         # legend
plt.tight_layout()                                            # layout
base = os.path.join(output_dir, "Dispersion_vs_SVF_NORMAL")   # base path
plt.savefig(base + ".svg", bbox_inches="tight")               # save svg
plt.savefig(base + ".png", bbox_inches="tight", dpi=300)      # save png
plt.close()                                                   # close
print(f"🖼️ Saved: {base}.svg, {base}.png")                    # log

# --- DISPERSION vs SVF (FLUCTUATION-CORRECTED) — dashed lines ---
plt.figure(figsize=(FIG_W, FIG_H))                            # new fig
DaF = [row["D_axial_fluct_m2_per_s"] for row in summary]     # axial D' list
DrF = [row["D_radial_fluct_m2_per_s"] for row in summary]    # radial D' list
plt.plot(svf_levels_pct, DaF, "o--", linewidth=LINE_W, markersize=MARKER_SZ, label="Axial $D'$")
plt.plot(svf_levels_pct, DrF, "s--", linewidth=LINE_W, markersize=MARKER_SZ, label="Radial $D'$")
plt.xlabel("Solid VOF (%)")                                   # x label
plt.ylabel("Dispersion coefficient (m$^2$/s)")                # y label
plt.title(f"Effective dispersion vs SVF (mean-free MSD)  —  Gas = {gas_rate_Lmin:.2f} L/min)")  # title
ax = plt.gca()                                                # axis handle
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))          # sci fmt (unchanged)
ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))            # sci ticks (unchanged)
plt.grid(True, alpha=0.3)                                     # grid
plt.legend(frameon=False, loc="best")                         # legend
plt.tight_layout()                                            # layout
base = os.path.join(output_dir, "Dispersion_vs_SVF_FLUCT")    # base path
plt.savefig(base + ".svg", bbox_inches="tight")               # save svg
plt.savefig(base + ".png", bbox_inches="tight", dpi=300)      # save png
plt.close()                                                   # close
print(f"🖼️ Saved: {base}.svg, {base}.png")                    # log

# --- COUNTS per LAG (semi-log-Y; τ > 0) ---
plt.figure(figsize=(FIG_W, FIG_H))                            # new fig
for svf, tau, c in zip(svf_levels_pct, all_time_sec, all_counts):
    mask = tau > 0                                            # ignore τ=0
    plt.semilogy(tau[mask], c[mask], linewidth=LINE_W, label=f"{svf:d}% SVF")  # plot
plt.xlabel("Time interval, $\\tau$ (s)")                      # x label
plt.ylabel("Increments per lag (count) [log]")                # y label
plt.title("MSD statistics: increments vs $\\tau$")            # title
plt.grid(True, which="both", alpha=0.3)                       # grid
plt.legend(frameon=False, title="Solid VOF", loc="best")      # legend
plt.tight_layout()                                            # layout
base = os.path.join(output_dir, "Counts_per_lag_semilogy_SVF")# base path
plt.savefig(base + ".svg", bbox_inches="tight")               # save svg
plt.savefig(base + ".png", bbox_inches="tight", dpi=300)      # save png
plt.close()                                                   # close
print(f"🖼️ Saved: {base}.svg, {base}.png")                    # log

# --------------------------- FINAL REPORT -----------------------------
print("\n✅ FINAL DISPERSION COEFFICIENTS (tail fit "
      f"{fit_window_duration:.3f} s), units m²/s, by SVF:\n")        # header
for i, svf in enumerate(svf_levels_pct):                              # loop rows
    print(f"  SVF {svf:>2d}%  ->  "
          f"Axial D  = {Da[i]:.4e} | Radial D  = {Dr[i]:.4e}   ||   "
          f"Axial D' = {DaF[i]:.4e} | Radial D' = {DrF[i]:.4e}")     # values
print("\n🎯 Done.")                                                   # footer
