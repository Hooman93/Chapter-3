# =====================================================================
# HYDRODYNAMIC DISPERSION FROM STB TRACKS — BASELINE CORE + SI PLOTS
# =====================================================================
# Core kept the same as your baseline:
#   - Reads tracks from DaVis .set
#   - Ensemble (increment-pooled) MSD in pixel^2 -> mm^2
#   - Linear tail fit over last 'fit_window_duration' seconds:
#       MSD ≈ 2 D t + c  =>  D = slope / 2
#   - D initially in mm^2/s (baseline), converted to m^2/s for outputs
#
# Outputs (updated per your request):
#   - Figures saved as BOTH SVG and PNG for: Axial MSD (m^2), Radial MSD (m^2),
#     Dispersion vs Gas Flow (m^2/s, sci y, dashed), Counts vs τ
#   - One Excel summary with per-case stats + D (m^2/s)
#   - One SEPARATE Excel file containing only dispersion vs gas-flow values
#
# Author: Hooman Eslami  |  Last edit: 2025-09-30
# =====================================================================

# ----------------------------- IMPORTS --------------------------------
import os                               # filesystem paths
from pathlib import Path                # robust path handling
from collections import Counter         # compact histogram of track lengths
import numpy as np                      # numerics
import pandas as pd                     # Excel I/O
import matplotlib.pyplot as plt         # plotting
from matplotlib.ticker import ScalarFormatter  # scientific y-axis
import lvpyio as lv                     # DaVis particle I/O

# ----------------------------- CONFIG ---------------------------------
fps = 198.4                             # [Hz] acquisition frame rate
dt = 1.0 / fps                          # [s] frame interval
analysis_window_s = 1.0                 # [s] max lag to analyze (keep as you set)
fit_window_duration = 0.2               # [s] tail length for linear fit (baseline)

px_per_mm = 5.84                        # calibration: pixels per mm
mm_per_px = 1.0 / px_per_mm             # [mm] per pixel
mm2_per_px2 = mm_per_px ** 2            # convert pixel^2 -> mm^2
MM2_TO_M2 = 1e-6                        # convert mm^2 -> m^2

# --- NEW (selection controls; minimal additions) ---
ALPHA_WINDOW_S = 0.2                    # [s] sliding window length for α
ALPHA_TOL = 0.10                        # accept |α-1| <= ALPHA_TOL
MIN_WIN_PTS = 6                         # min samples inside a window
START_FIT_TAU = 0.4                     # [s] do not consider windows that start before this τ
PREFER_LATE = True                      # prefer spans closer to the tail (tie-break)

#non-median filtered data_2 seconds of flow (images 500-1000)
data_paths = [
    r"E:\Three-phase Hydrogels\track data_dispersion\Three-phase\0%\0.05 lpermin\dipersion calculation_0% (images 500-1000)_0.05 lpmin",
    r"E:\Three-phase Hydrogels\track data_dispersion\Three-phase\0%\0.25 lpermin\dipersion calculation_0% (images 500-1000)_0.25 lpmin",
    r"E:\Three-phase Hydrogels\track data_dispersion\Three-phase\0%\0.5 lpermin\dipersion calculation_0% (images 500-1000)_0.5 lpmin",
]
gas_flow_rates = [0.05, 0.25, 0.50]     # [L/min] legend/x-axis

# Output directory
output_dir = r"C:\python files\Hydrogel_Three-phase\Figures\New results\Hydrodynamic dispersions\Effect of gas phase\0%"
Path(output_dir).mkdir(parents=True, exist_ok=True)  # ensure path exists

# ------------------------- PLOT STYLE ---------------------------------
plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300,  # high-res
    "font.family": "sans-serif",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.8, "grid.linewidth": 0.4
})
FIG_W, FIG_H = 3.54, 2.50  # ~90 mm x 63 mm (A4 2-column)
LINE_W = 1.4               # line width for curves
MARKER_SZ = 4.0            # marker size for markers

# ---------------------------- HELPERS ---------------------------------
def tukey_box_from_hist(length_counter: Counter):
    """
    Compute compact Tukey statistics (quartiles, whiskers) from a histogram
    of track lengths. Returns a dict with counts and stats.
    """
    # Return NaNs if histogram empty
    if not length_counter:
        return dict(n_tracks=0, n_particles=0, min=np.nan, q1=np.nan, median=np.nan,
                    q3=np.nan, max=np.nan, whisker_low=np.nan, whisker_high=np.nan,
                    mean=np.nan, std=np.nan)
    # Unique lengths and counts
    Ls = np.array(sorted(length_counter.keys()), dtype=np.int64)
    Cs = np.array([length_counter[L] for L in Ls], dtype=np.int64)
    # Totals
    n_tracks = int(Cs.sum())
    n_particles = int((Ls * Cs).sum())
    # Bounds
    mn, mx = int(Ls[0]), int(Ls[-1])
    # Mean and std from histogram
    mean = (Ls * Cs).sum() / n_tracks
    mean2 = (Ls**2 * Cs).sum() / n_tracks
    std = float(np.sqrt(max(0.0, mean2 - mean**2)))
    # CDF for quantile lookup
    cdf = np.cumsum(Cs)
    # Quantile helper (nearest-left)
    def qtile(q):
        target = q * n_tracks
        idx = np.searchsorted(cdf, target, side="left")
        idx = min(idx, len(Ls) - 1)
        return int(Ls[idx])
    # Quartiles
    q1, med, q3 = qtile(0.25), qtile(0.50), qtile(0.75)
    # Tukey whisker thresholds
    iqr = q3 - q1
    wl_thr, wh_thr = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    # Map to observed indices
    wl_idx = np.searchsorted(Ls, wl_thr, side="left")
    wh_idx = np.searchsorted(Ls, wh_thr, side="right") - 1
    wl_idx = min(max(wl_idx, 0), len(Ls) - 1)
    wh_idx = min(max(wh_idx, 0), len(Ls) - 1)
    # Observed whiskers
    wlow, whigh = int(Ls[wl_idx]), int(Ls[wh_idx])
    # Pack and return
    return dict(n_tracks=n_tracks, n_particles=n_particles,
                min=mn, q1=q1, median=med, q3=q3, max=mx,
                whisker_low=wlow, whisker_high=whigh,
                mean=float(mean), std=std)

# --- NEW: select best α≈1 span (unweighted) and get slope on that span ---
def _best_span_alpha_unweighted(tau, msd, win_s=0.2, alpha_tol=0.10,
                                min_pts=6, start_tau=0.0, prefer_late=True):
    """
    Slide a window (win_s) along (tau, msd). In each window, fit:
        log(MSD) = c + α * log(tau)    (ordinary least squares)
    Mark windows with |α-1| <= alpha_tol. Choose the LONGEST contiguous run
    whose RIGHT EDGE is CLOSEST TO THE TAIL (prefer_late). Return [tau0,tau1]
    and the simple unweighted slope of MSD vs τ over that span.
    """
    m = np.isfinite(tau) & np.isfinite(msd) & (tau > 0) & (msd > 0)
    if m.sum() < min_pts:
        return dict(ok=False)
    t = tau[m]; y = msd[m]
    dt_local = np.median(np.diff(t))
    if not np.isfinite(dt_local) or dt_local <= 0:
        return dict(ok=False)
    nwin = max(int(round(win_s / dt_local)), min_pts)
    if nwin > t.size:
        a, b = np.polyfit(t, y, 1)
        return dict(ok=True, tau0=float(t[0]), tau1=float(t[-1]), slope=float(a))
    alphas, spans = [], []
    for i0 in range(0, t.size - nwin + 1):
        i1 = i0 + nwin
        if t[i0] < start_tau:
            continue
        xt = np.log(t[i0:i1]); yt = np.log(y[i0:i1])
        p = np.polyfit(xt, yt, 1)
        alphas.append(float(p[0]))
        spans.append((i0, i1))
    if not alphas:
        return dict(ok=False)
    alphas = np.asarray(alphas)
    good = np.abs(alphas - 1.0) <= alpha_tol
    best = None
    run_start = None
    for k, g in enumerate(good):
        if g and run_start is None:
            run_start = k
        if (not g or k == len(good)-1) and run_start is not None:
            k_end = k if not g else k
            i0 = spans[run_start][0]; i1 = spans[k_end][1]
            length_s = t[i1-1] - t[i0]
            end_time = t[i1-1]
            score = (end_time if prefer_late else 0.0, length_s)
            if (best is None) or (score > best[0]):
                best = (score, (i0, i1))
            run_start = None
    if best is None:
        idx = int(np.argmin(np.abs(alphas - 1.0)))
        i0, i1 = spans[idx]
    else:
        i0, i1 = best[1]
    a, b = np.polyfit(t[i0:i1], y[i0:i1], 1)
    return dict(ok=True, tau0=float(t[i0]), tau1=float(t[i1-1]), slope=float(a))

# --------------------------- RUNTIME LOGS ------------------------------
max_time_intervals = int(analysis_window_s / dt)  # number of lag steps (includes k=0)
print(f"✅ Frame interval: {dt:.6f} s")
print(f"✅ Number of time intervals to analyze: {max_time_intervals}")
print(f"✅ Calibration: {px_per_mm:.2f} px/mm  (1 px = {mm_per_px:.6f} mm)")

# ------------------------- PER-CASE STORAGE ---------------------------
all_time_sec = []        # list of τ arrays (s) per dataset
all_msd_ax_mm2 = []      # axial MSD (normal) per dataset
all_msd_ra_mm2 = []      # radial MSD (normal) per dataset
all_msd_ax_fl_mm2 = []   # axial MSD (fluctuation-corrected) per dataset
all_msd_ra_fl_mm2 = []   # radial MSD (fluctuation-corrected) per dataset
all_counts = []          # increments-per-lag counts per dataset
summary = []             # per-case rows for Excel

# --- NEW: selected spans for dashed overlay (per curve) ---
sel_ax = []              # (tau0, tau1) for axial normal
sel_ra = []              # (tau0, tau1) for radial normal
sel_ax_fl = []           # (tau0, tau1) for axial fluctuation-corrected
sel_ra_fl = []           # (tau0, tau1) for radial fluctuation-corrected

# ============================== CORE LOOP ==============================
for q, path in zip(gas_flow_rates, data_paths):
    # Log which case is being processed
    print("\n------------------------------------------------------------")
    print(f"🔎 Gas flow rate: {q:.2f} L/min")
    print(f"📂 Folder: {path}")

    # Read DaVis data and get list of tracks
    tr = lv.read_particles(path)
    tracks = tr.tracks()
    print(f"   Tracks found: {len(tracks)}")

    # Accumulators for normal MSD sums at each lag
    msd_ax_sum_px2 = np.zeros(max_time_intervals)  # ∑(Δy)^2 per lag
    msd_ra_sum_px2 = np.zeros(max_time_intervals)  # ∑[(Δx)^2+(Δz)^2] per lag

    # Accumulators for fluctuation-corrected MSD sums at each lag
    msd_ax_fl_sum_px2 = np.zeros(max_time_intervals)  # ∑(Δy - kdt*vȳ)^2
    msd_ra_fl_sum_px2 = np.zeros(max_time_intervals)  # ∑[(Δx - kdt*vx̄)^2 + (Δz - kdt*vz̄)^2]

    # Counters for number of increments contributing at each lag
    counts    = np.zeros(max_time_intervals, dtype=int)  # for normal MSD
    counts_fl = np.zeros(max_time_intervals, dtype=int)  # for fluctuation-corrected MSD

    # Track length histogram/counters
    length_hist = Counter()
    particles_used = 0
    MIN_LEN = 5  # ignore very short tracks

    # --------------- Iterate over all tracks ----------------
    for tck in tracks:
        # Extract coordinate arrays (pixels)
        P = tck.particles
        x = P["x"]; y = P["y"]; z = P["z"]
        n = len(x)  # number of frames in this track

        # Skip too-short tracks
        if n <= MIN_LEN:
            continue

        # Update track length histogram and counts
        length_hist[n] += 1
        particles_used += n

        # Mean velocity (px/s) from end-to-start displacement / total time
        total_T = (n - 1) * dt
        if total_T <= 0:
            continue
        vx_bar = (x[-1] - x[0]) / total_T
        vy_bar = (y[-1] - y[0]) / total_T
        vz_bar = (z[-1] - z[0]) / total_T

        # Max usable lag for this track
        max_dt_here = min(max_time_intervals, n)

        # Loop lags k = 1..max_dt_here-1
        for k in range(1, max_dt_here):
            # Raw lag-k increments (vectors length n-k)
            dx = x[k:] - x[:-k]
            dy = y[k:] - y[:-k]
            dz = z[k:] - z[:-k]

            # Normal MSD components
            dy2 = dy**2
            dr2 = dx**2 + dz**2

            # Fluctuation-corrected increments: Δr' = Δr - k*dt*<v>
            kdt = k * dt
            dx_fl = dx - kdt * vx_bar
            dy_fl = dy - kdt * vy_bar
            dz_fl = dz - kdt * vz_bar

            # Fluctuation-corrected squared magnitudes
            dy_fl2 = dy_fl**2
            dr_fl2 = dx_fl**2 + dz_fl**2

            # Accumulate sums (per lag k)
            msd_ax_sum_px2[k]    += dy2.sum()
            msd_ra_sum_px2[k]    += dr2.sum()
            msd_ax_fl_sum_px2[k] += dy_fl2.sum()
            msd_ra_fl_sum_px2[k] += dr_fl2.sum()

            # Count the number of increments contributing at lag k
            ninc = dy2.size
            counts[k]    += ninc
            counts_fl[k] += ninc

    # Determine which lags have data
    valid    = counts    > 0
    valid_fl = counts_fl > 0

    # Allocate per-lag mean MSD arrays (start as NaN)
    msd_ax_px2    = np.full_like(msd_ax_sum_px2,    np.nan, dtype=float)
    msd_ra_px2    = np.full_like(msd_ra_sum_px2,    np.nan, dtype=float)
    msd_ax_fl_px2 = np.full_like(msd_ax_fl_sum_px2, np.nan, dtype=float)
    msd_ra_fl_px2 = np.full_like(msd_ra_fl_sum_px2, np.nan, dtype=float)

    # Convert sums to means by dividing by counts
    msd_ax_px2[valid]       = msd_ax_sum_px2[valid]       / counts[valid]
    msd_ra_px2[valid]       = msd_ra_sum_px2[valid]       / counts[valid]
    msd_ax_fl_px2[valid_fl] = msd_ax_fl_sum_px2[valid_fl] / counts_fl[valid_fl]
    msd_ra_fl_px2[valid_fl] = msd_ra_fl_sum_px2[valid_fl] / counts_fl[valid_fl]

    # Convert pixel^2 -> mm^2
    msd_ax_mm2    = msd_ax_px2    * mm2_per_px2
    msd_ra_mm2    = msd_ra_px2    * mm2_per_px2
    msd_ax_fl_mm2 = msd_ax_fl_px2 * mm2_per_px2
    msd_ra_fl_mm2 = msd_ra_fl_px2 * mm2_per_px2

    # Build τ axis (seconds)
    tau = np.arange(max_time_intervals) * dt

    # Store arrays to plot after processing all cases
    all_time_sec.append(tau)
    all_msd_ax_mm2.append(msd_ax_mm2)
    all_msd_ra_mm2.append(msd_ra_mm2)
    all_msd_ax_fl_mm2.append(msd_ax_fl_mm2)
    all_msd_ra_fl_mm2.append(msd_ra_fl_mm2)
    all_counts.append(counts.copy())

    # --- NEW: sliding 0.2 s α≈1 selection (unweighted), slope on selected span ---
    info_ax  = _best_span_alpha_unweighted(tau, msd_ax_mm2,    win_s=ALPHA_WINDOW_S, alpha_tol=ALPHA_TOL,
                                           min_pts=MIN_WIN_PTS, start_tau=START_FIT_TAU, prefer_late=PREFER_LATE)
    info_ra  = _best_span_alpha_unweighted(tau, msd_ra_mm2,    win_s=ALPHA_WINDOW_S, alpha_tol=ALPHA_TOL,
                                           min_pts=MIN_WIN_PTS, start_tau=START_FIT_TAU, prefer_late=PREFER_LATE)
    info_axF = _best_span_alpha_unweighted(tau, msd_ax_fl_mm2, win_s=ALPHA_WINDOW_S, alpha_tol=ALPHA_TOL,
                                           min_pts=MIN_WIN_PTS, start_tau=START_FIT_TAU, prefer_late=PREFER_LATE)
    info_raF = _best_span_alpha_unweighted(tau, msd_ra_fl_mm2, win_s=ALPHA_WINDOW_S, alpha_tol=ALPHA_TOL,
                                           min_pts=MIN_WIN_PTS, start_tau=START_FIT_TAU, prefer_late=PREFER_LATE)

    # --- NEW: save spans for dashed overlays ---
    sel_ax.append((info_ax.get("tau0", None),  info_ax.get("tau1", None)))
    sel_ra.append((info_ra.get("tau0", None),  info_ra.get("tau1", None)))
    sel_ax_fl.append((info_axF.get("tau0", None), info_axF.get("tau1", None)))
    sel_ra_fl.append((info_raF.get("tau0", None), info_raF.get("tau1", None)))

    # --- NEW: compute D from slope/2 on selected span (mm^2/s -> m^2/s) ---
    D_ax_m2_s    = (info_ax.get("slope",  np.nan) / 2.0) * MM2_TO_M2
    D_ra_m2_s    = (info_ra.get("slope",  np.nan) / 2.0) * MM2_TO_M2
    D_ax_fl_m2_s = (info_axF.get("slope", np.nan) / 2.0) * MM2_TO_M2
    D_ra_fl_m2_s = (info_raF.get("slope", np.nan) / 2.0) * MM2_TO_M2

    # Track length stats for diagnostics
    bx = tukey_box_from_hist(length_hist)

    # Append this case to summary table
    summary.append({
        "gas_L_per_min": q,
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

    # Print quick diagnostics
    def _span_str(info):
        return "span: n/a" if not info.get("ok", False) else f"span: [{info['tau0']:.3f}, {info['tau1']:.3f}] s"
    print(f"   Tracks used: {bx['n_tracks']} | Particles used: {bx['n_particles']}")
    print(f"   Track length Q1/Med/Q3: {bx['q1']} / {bx['median']} / {bx['q3']}")
    print(f"   Axial D  = {D_ax_m2_s:.4e} m²/s ({_span_str(info_ax)}) | Radial D  = {D_ra_m2_s:.4e} m²/s ({_span_str(info_ra)})")
    print(f"   Axial D' = {D_ax_fl_m2_s:.4e} m²/s ({_span_str(info_axF)}) | Radial D' = {D_ra_fl_m2_s:.4e} m²/s ({_span_str(info_raF)})")

# ============================== SAVE EXCEL =============================
# Full summary (includes both normal and fluctuation-corrected D)
excel_summary_path = os.path.join(output_dir, "PTV_dispersion_summary.xlsx")
pd.DataFrame(summary).to_excel(excel_summary_path, index=False)
print(f"\n📗 Excel summary saved: {excel_summary_path}")

# Compact sheet: gas flow vs D (normal & fluct)
dispersion_only_path = os.path.join(output_dir, "Dispersion_vs_GasFlow.xlsx")
pd.DataFrame({
    "gas_L_per_min": [row["gas_L_per_min"] for row in summary],
    "D_axial_m2_per_s": [row["D_axial_m2_per_s"] for row in summary],
    "D_radial_m2_per_s": [row["D_radial_m2_per_s"] for row in summary],
    "D_axial_fluct_m2_per_s": [row["D_axial_fluct_m2_per_s"] for row in summary],
    "D_radial_fluct_m2_per_s": [row["D_radial_fluct_m2_per_s"] for row in summary],
}).to_excel(dispersion_only_path, index=False)
print(f"📘 Dispersion-only Excel saved: {dispersion_only_path}")

# ================================ FIGURES ==============================
# NOTE: MSD arrays are in mm^2; multiply by 1e-6 to show m^2 on plots.

# --- NEW: dashed overlay helper; keep base styling identical ---
def _overlay_selected_span(tau, msd_mm2, span, color):
    tau0, tau1 = span
    if tau0 is None or tau1 is None or not np.isfinite(tau0) or not np.isfinite(tau1):
        return
    mask = np.isfinite(msd_mm2) & (tau >= tau0) & (tau <= tau1)
    if mask.sum() < 2:
        return
    # faded solid underlay (alpha=0.2 -> ~80% invisible)
    plt.plot(tau[mask], msd_mm2[mask] * MM2_TO_M2, '-',
             linewidth=LINE_W, color=color, alpha=0.2, zorder=3)
    # slightly bolder dashed on top
    plt.plot(tau[mask], msd_mm2[mask] * MM2_TO_M2, '--',
             linewidth=LINE_W + 0.9, color=color, alpha=1.0, zorder=4)

# --- RADIAL MSD (NORMAL) — scientific y-axis formatting ---
plt.figure(figsize=(FIG_W, FIG_H))
for q, tau, msd_mm2, span in zip(gas_flow_rates, all_time_sec, all_msd_ra_mm2, sel_ra):
    line, = plt.plot(tau, msd_mm2 * MM2_TO_M2, label=f"{q:.2f} L/min", linewidth=LINE_W)
    _overlay_selected_span(tau, msd_mm2, span, line.get_color())
plt.xlabel("Time interval, $\\tau$ (s)")
plt.ylabel("Radial MSD (m$^2$)")
plt.title("Radial MSD vs $\\tau$")
ax = plt.gca()
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))  # show 10^−n scaling
ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))    # scientific ticks
plt.grid(True, alpha=0.3)
plt.legend(frameon=False, title="Gas flow rate", loc="best")
plt.tight_layout()
base = os.path.join(output_dir, "Radial_MSD_vs_Time")
plt.savefig(base + ".svg", bbox_inches="tight")
plt.savefig(base + ".png", bbox_inches="tight", dpi=300)
plt.close()
print(f"🖼️ Saved: {base}.svg, {base}.png")

# --- AXIAL MSD (NORMAL) — scientific y-axis formatting ---
plt.figure(figsize=(FIG_W, FIG_H))
for q, tau, msd_mm2, span in zip(gas_flow_rates, all_time_sec, all_msd_ax_mm2, sel_ax):
    line, = plt.plot(tau, msd_mm2 * MM2_TO_M2, label=f"{q:.2f} L/min", linewidth=LINE_W)
    _overlay_selected_span(tau, msd_mm2, span, line.get_color())
plt.xlabel("Time interval, $\\tau$ (s)")
plt.ylabel("Axial MSD (m$^2$)")
plt.title("Axial MSD vs $\\tau$")
ax = plt.gca()
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))  # show 10^−n scaling
ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))    # scientific ticks
plt.grid(True, alpha=0.3)
plt.legend(frameon=False, title="Gas flow rate", loc="best")
plt.tight_layout()
base = os.path.join(output_dir, "Axial_MSD_vs_Time")
plt.savefig(base + ".svg", bbox_inches="tight")
plt.savefig(base + ".png", bbox_inches="tight", dpi=300)
plt.close()
print(f"🖼️ Saved: {base}.svg, {base}.png")   

# --- RADIAL MSD′ (FLUCTUATION-CORRECTED) — scientific y-axis formatting ---
plt.figure(figsize=(FIG_W, FIG_H))
for q, tau, msd_mm2, span in zip(gas_flow_rates, all_time_sec, all_msd_ra_fl_mm2, sel_ra_fl):
    line, = plt.plot(tau, msd_mm2 * MM2_TO_M2, label=f"{q:.2f} L/min", linewidth=LINE_W)
    _overlay_selected_span(tau, msd_mm2, span, line.get_color())
plt.xlabel("Time interval, $\\tau$ (s)")
plt.ylabel("Radial MSD′ (m$^2$)")
plt.title("Radial MSD′ (mean-free) vs $\\tau$")
ax = plt.gca()
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))  # show 10^−n scaling
ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))    # scientific ticks
plt.grid(True, alpha=0.3)
plt.legend(frameon=False, title="Gas flow rate", loc="best")
plt.tight_layout()
base = os.path.join(output_dir, "Radial_MSD_fluct_vs_Time")
plt.savefig(base + ".svg", bbox_inches="tight")
plt.savefig(base + ".png", bbox_inches="tight", dpi=300)
plt.close()
print(f"🖼️ Saved: {base}.svg, {base}.png")

# --- AXIAL MSD′ (FLUCTUATION-CORRECTED) — scientific y-axis formatting ---
plt.figure(figsize=(FIG_W, FIG_H))
for q, tau, msd_mm2, span in zip(gas_flow_rates, all_time_sec, all_msd_ax_fl_mm2, sel_ax_fl):
    line, = plt.plot(tau, msd_mm2 * MM2_TO_M2, label=f"{q:.2f} L/min", linewidth=LINE_W)
    _overlay_selected_span(tau, msd_mm2, span, line.get_color())
plt.xlabel("Time interval, $\\tau$ (s)")
plt.ylabel("Axial MSD′ (m$^2$)")
plt.title("Axial MSD′ (mean-free) vs $\\tau$")
ax = plt.gca()
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))  # show 10^−n scaling
ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))    # scientific ticks
plt.grid(True, alpha=0.3)
plt.legend(frameon=False, title="Gas flow rate", loc="best")
plt.tight_layout()
base = os.path.join(output_dir, "Axial_MSD_fluct_vs_Time")
plt.savefig(base + ".svg", bbox_inches="tight")
plt.savefig(base + ".png", bbox_inches="tight", dpi=300)
plt.close()
print(f"🖼️ Saved: {base}.svg, {base}.png")

# --- DISPERSION vs GAS FLOW (NORMAL MSD ONLY) — dashed lines ---
plt.figure(figsize=(FIG_W, FIG_H))
Da = [row["D_axial_m2_per_s"] for row in summary]   # axial D (normal)
Dr = [row["D_radial_m2_per_s"] for row in summary]  # radial D (normal)
plt.plot(gas_flow_rates, Da, "o--", linewidth=LINE_W, markersize=MARKER_SZ, label="Axial $D$")
plt.plot(gas_flow_rates, Dr, "s--", linewidth=LINE_W, markersize=MARKER_SZ, label="Radial $D$")
plt.xlabel("Gas flow rate (L/min)")
plt.ylabel("Dispersion coefficient (m$^2$/s)")
plt.title("Effective dispersion vs gas rate (normal MSD)")
ax = plt.gca()
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))  # scientific y-axis
ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
plt.grid(True, alpha=0.3)
plt.legend(frameon=False, loc="best")
plt.tight_layout()
base = os.path.join(output_dir, "Dispersion_vs_GasFlow_NORMAL")
plt.savefig(base + ".svg", bbox_inches="tight")
plt.savefig(base + ".png", bbox_inches="tight", dpi=300)
plt.close()
print(f"🖼️ Saved: {base}.svg, {base}.png")

# --- DISPERSION vs GAS FLOW (FLUCTUATION-CORRECTED MSD ONLY) — dashed lines ---
plt.figure(figsize=(FIG_W, FIG_H))
DaF = [row["D_axial_fluct_m2_per_s"] for row in summary]  # axial D'
DrF = [row["D_radial_fluct_m2_per_s"] for row in summary] # radial D'
plt.plot(gas_flow_rates, DaF, "o--", linewidth=LINE_W, markersize=MARKER_SZ, label="Axial $D'$")
plt.plot(gas_flow_rates, DrF, "s--", linewidth=LINE_W, markersize=MARKER_SZ, label="Radial $D'$")
plt.xlabel("Gas flow rate (L/min)")
plt.ylabel("Dispersion coefficient (m$^2$/s)")
plt.title("Effective dispersion vs gas rate (mean-free MSD)")
ax = plt.gca()
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))  # scientific y-axis
ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
plt.grid(True, alpha=0.3)
plt.legend(frameon=False, loc="best")
plt.tight_layout()
base = os.path.join(output_dir, "Dispersion_vs_GasFlow_FLUCT")
plt.savefig(base + ".svg", bbox_inches="tight")
plt.savefig(base + ".png", bbox_inches="tight", dpi=300)
plt.close()
print(f"🖼️ Saved: {base}.svg, {base}.png")

# --- COUNTS per LAG (ONLY semi-log-Y; τ > 0) ---
plt.figure(figsize=(FIG_W, FIG_H))
for q, tau, c in zip(gas_flow_rates, all_time_sec, all_counts):
    mask = tau > 0                         # avoid plotting at τ=0 on a log axis
    plt.semilogy(tau[mask], c[mask], linewidth=LINE_W, label=f"{q:.2f} L/min")
plt.xlabel("Time interval, $\\tau$ (s)")
plt.ylabel("Increments per lag (count) [log]")
plt.title("MSD statistics: increments vs $\\tau$")
plt.grid(True, which="both", alpha=0.3)
plt.legend(frameon=False, title="Gas flow rate", loc="best")
plt.tight_layout()
base = os.path.join(output_dir, "Counts_per_lag_semilogy")
plt.savefig(base + ".svg", bbox_inches="tight")
plt.savefig(base + ".png", bbox_inches="tight", dpi=300)
plt.close()
print(f"🖼️ Saved: {base}.svg, {base}.png")

# --------------------------- FINAL REPORT -----------------------------
print("\n✅ FINAL DISPERSION COEFFICIENTS (tail fit "
      f"{fit_window_duration:.3f} s), units m²/s:\n")
for i, q in enumerate(gas_flow_rates):
    print(f"  {q:.2f} L/min  ->  "
          f"Axial D  = {Da[i]:.4e} | Radial D  = {Dr[i]:.4e}   ||   "
          f"Axial D' = {DaF[i]:.4e} | Radial D' = {DrF[i]:.4e}")
print("\n🎯 Done.")
