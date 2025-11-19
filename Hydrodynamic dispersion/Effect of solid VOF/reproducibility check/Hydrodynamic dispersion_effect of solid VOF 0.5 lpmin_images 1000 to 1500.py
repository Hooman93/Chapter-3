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
# Author: Hooman Eslami  |  Last edit: 2025-10-06
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
gas_rate_Lmin = 0.5            # [L/min] same for all SVF cases

# --- Dataset folders: SVF sweep at 0.05 L/min ---
data_paths = [
    r"E:\Three-phase Hydrogels\New track data_1000 images_spatialmedian filtered\New track data_1000 images_spatialmedian filtered\0.5 lpermin\0%\Median filtered_Images_1500-2500_For track analysis_0%_0.5 lpmin",
    r"E:\Three-phase Hydrogels\New track data_1000 images_spatialmedian filtered\New track data_1000 images_spatialmedian filtered\0.5 lpermin\10%\Median filtered_Images_1500-2500_For track analysis_10%_0.5 lpmin",
    r"E:\Three-phase Hydrogels\New track data_1000 images_spatialmedian filtered\New track data_1000 images_spatialmedian filtered\0.5 lpermin\30%\Median filtered_Images_1500-2500_For track analysis_30%_0.5 lpmin",
    r"E:\Three-phase Hydrogels\New track data_1000 images_spatialmedian filtered\New track data_1000 images_spatialmedian filtered\0.5 lpermin\50%\Median filtered_Images_1500-2500_For track analysis_50%_0.5 lpmin",
]

# data_paths = [
#     r"E:\Three-phase Hydrogels\New track data 1000 - 1500(reproducibility check\0.5 lpmin\0%\dipersion calculation 0% (images 1000-1500)_0.5 lpmin",
#     r"E:\Three-phase Hydrogels\New track data 1000 - 1500(reproducibility check\0.5 lpmin\10%\dipersion calculation 10% (images 1000-1500)_0.5 lpmin",
#     r"E:\Three-phase Hydrogels\New track data 1000 - 1500(reproducibility check\0.5 lpmin\30%\dipersion calculation 30% (images 1000-1500)_0.5 lpmin",
#     r"E:\Three-phase Hydrogels\New track data 1000 - 1500(reproducibility check\0.5 lpmin\50%\dipersion calculation 50% (images 1000-1500)_0.5 lpmin",
# ]

svf_levels_pct = [0, 10, 30, 50]           # [%] legend/x-axis for SVF

# --- Output directory (specific to SVF sweep and gas rate) ---
output_dir = r"C:\python files\Hydrogel_Three-phase\Figures\New results\Hydrodynamic dispersions\Effect of solid VOF\0.5 Lpmin\Reproducibility_1000 images_MFed"
Path(output_dir).mkdir(parents=True, exist_ok=True)

# ------------------------- PLOT STYLE ---------------------------------
plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300,  # hi-res
    "font.family": "sans-serif",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.8, "grid.linewidth": 0.4
})
FIG_W, FIG_H = 3.54, 2.50  # ~90 mm × 63 mm
LINE_W = 1.4               # line width
MARKER_SZ = 4.0            # marker size

# ---------------------------- HELPERS ---------------------------------
def tukey_box_from_hist(length_counter: Counter):
    """
    Compute compact Tukey stats (quartiles, whiskers) from a histogram
    of track lengths. Returns dict with counts and stats.
    """
    if not length_counter:
        return dict(n_tracks=0, n_particles=0, min=np.nan, q1=np.nan, median=np.nan,
                    q3=np.nan, max=np.nan, whisker_low=np.nan, whisker_high=np.nan,
                    mean=np.nan, std=np.nan)
    Ls = np.array(sorted(length_counter.keys()), dtype=np.int64)
    Cs = np.array([length_counter[L] for L in Ls], dtype=np.int64)
    n_tracks = int(Cs.sum())
    n_particles = int((Ls * Cs).sum())
    mn, mx = int(Ls[0]), int(Ls[-1])
    mean = (Ls * Cs).sum() / n_tracks
    mean2 = (Ls**2 * Cs).sum() / n_tracks
    std = float(np.sqrt(max(0.0, mean2 - mean**2)))
    cdf = np.cumsum(Cs)
    def qtile(q):
        target = q * n_tracks
        idx = np.searchsorted(cdf, target, side="left")
        idx = min(idx, len(Ls) - 1)
        return int(Ls[idx])
    q1, med, q3 = qtile(0.25), qtile(0.50), qtile(0.75)
    iqr = q3 - q1
    wl_thr, wh_thr = q1 - 1.5*iqr, q3 + 1.5*iqr
    wl_idx = np.searchsorted(Ls, wl_thr, side="left")
    wh_idx = np.searchsorted(Ls, wh_thr, side="right") - 1
    wl_idx = min(max(wl_idx, 0), len(Ls)-1)
    wh_idx = min(max(wh_idx, 0), len(Ls)-1)
    wlow, whigh = int(Ls[wl_idx]), int(Ls[wh_idx])
    return dict(n_tracks=n_tracks, n_particles=n_particles,
                min=mn, q1=q1, median=med, q3=q3, max=mx,
                whisker_low=wlow, whisker_high=whigh,
                mean=float(mean), std=std)

# --------------------------- RUNTIME LOGS ------------------------------
max_time_intervals = int(analysis_window_s / dt)  # number of lag steps (incl k=0)
print(f"✅ Frame interval: {dt:.6f} s")
print(f"✅ Number of time intervals to analyze: {max_time_intervals}")
print(f"✅ Calibration: {px_per_mm:.2f} px/mm  (1 px = {mm_per_px:.6f} mm)")
print(f"✅ Fixed gas rate for this sweep: {gas_rate_Lmin:.2f} L/min")

# ------------------------- PER-CASE STORAGE ---------------------------
all_time_sec = []        # τ arrays (s)
all_msd_ax_mm2 = []      # axial MSD (normal)
all_msd_ra_mm2 = []      # radial MSD (normal)
all_msd_ax_fl_mm2 = []   # axial MSD (fluctuation-corrected)
all_msd_ra_fl_mm2 = []   # radial MSD (fluctuation-corrected)
all_counts = []          # increments per lag
summary = []             # rows for Excel

# ============================== CORE LOOP ==============================
for svf, path in zip(svf_levels_pct, data_paths):
    # ---- Context log ----
    print("\n------------------------------------------------------------")
    print(f"🔎 Solid VOF (SVF): {svf:d}%   |   Gas: {gas_rate_Lmin:.2f} L/min")
    print(f"📂 Folder: {path}")

    # ---- Load tracks ----
    tr = lv.read_particles(path)
    tracks = tr.tracks()
    print(f"   Tracks found: {len(tracks)}")

    # ---- Accumulators (normal) ----
    msd_ax_sum_px2 = np.zeros(max_time_intervals)  # ∑(Δy)^2 per lag
    msd_ra_sum_px2 = np.zeros(max_time_intervals)  # ∑[(Δx)^2+(Δz)^2] per lag

    # ---- Accumulators (fluctuation-corrected) ----
    msd_ax_fl_sum_px2 = np.zeros(max_time_intervals)  # ∑(Δy - kdt*vȳ)^2
    msd_ra_fl_sum_px2 = np.zeros(max_time_intervals)  # ∑[(Δx - kdt*vx̄)^2 + (Δz - kdt*vz̄)^2]

    # ---- Counts per lag ----
    counts    = np.zeros(max_time_intervals, dtype=int)
    counts_fl = np.zeros(max_time_intervals, dtype=int)

    # ---- Track length stats ----
    length_hist = Counter()
    particles_used = 0
    MIN_LEN = 5

    # ---- Loop tracks ----
    for tck in tracks:
        P = tck.particles
        x = P["x"]; y = P["y"]; z = P["z"]
        n = len(x)
        if n <= MIN_LEN:
            continue
        length_hist[n] += 1
        particles_used += n

        total_T = (n - 1) * dt
        if total_T <= 0:
            continue
        vx_bar = (x[-1] - x[0]) / total_T
        vy_bar = (y[-1] - y[0]) / total_T
        vz_bar = (z[-1] - z[0]) / total_T

        max_dt_here = min(max_time_intervals, n)
        for k in range(1, max_dt_here):
            dx = x[k:] - x[:-k]
            dy = y[k:] - y[:-k]
            dz = z[k:] - z[:-k]

            # Normal MSD
            dy2 = dy**2
            dr2 = dx**2 + dz**2

            # Mean-free increments
            kdt = k * dt
            dx_fl = dx - kdt * vx_bar
            dy_fl = dy - kdt * vy_bar
            dz_fl = dz - kdt * vz_bar

            dy_fl2 = dy_fl**2
            dr_fl2 = dx_fl**2 + dz_fl**2

            # Accumulate sums
            msd_ax_sum_px2[k]    += dy2.sum()
            msd_ra_sum_px2[k]    += dr2.sum()
            msd_ax_fl_sum_px2[k] += dy_fl2.sum()
            msd_ra_fl_sum_px2[k] += dr_fl2.sum()

            ninc = dy2.size
            counts[k]    += ninc
            counts_fl[k] += ninc

    # ---- Finalize ensemble means ----
    valid    = counts    > 0
    valid_fl = counts_fl > 0

    msd_ax_px2    = np.full_like(msd_ax_sum_px2,    np.nan, dtype=float)
    msd_ra_px2    = np.full_like(msd_ra_sum_px2,    np.nan, dtype=float)
    msd_ax_fl_px2 = np.full_like(msd_ax_fl_sum_px2, np.nan, dtype=float)
    msd_ra_fl_px2 = np.full_like(msd_ra_fl_sum_px2, np.nan, dtype=float)

    msd_ax_px2[valid]       = msd_ax_sum_px2[valid]       / counts[valid]
    msd_ra_px2[valid]       = msd_ra_sum_px2[valid]       / counts[valid]
    msd_ax_fl_px2[valid_fl] = msd_ax_fl_sum_px2[valid_fl] / counts_fl[valid_fl]
    msd_ra_fl_px2[valid_fl] = msd_ra_fl_sum_px2[valid_fl] / counts_fl[valid_fl]

    # ---- Units: px^2 -> mm^2 ----
    msd_ax_mm2    = msd_ax_px2    * mm2_per_px2
    msd_ra_mm2    = msd_ra_px2    * mm2_per_px2
    msd_ax_fl_mm2 = msd_ax_fl_px2 * mm2_per_px2
    msd_ra_fl_mm2 = msd_ra_fl_px2 * mm2_per_px2

    # ---- Lag axis τ (s) ----
    tau = np.arange(max_time_intervals) * dt

    # ---- Store for plotting ----
    all_time_sec.append(tau)
    all_msd_ax_mm2.append(msd_ax_mm2)
    all_msd_ra_mm2.append(msd_ra_mm2)
    all_msd_ax_fl_mm2.append(msd_ax_fl_mm2)
    all_msd_ra_fl_mm2.append(msd_ra_fl_mm2)
    all_counts.append(counts.copy())

    # ---- Tail-fit helper: D = slope/2 (mm^2/s) ----
    def fit_tail_D(tau_arr, msd_arr_mm2):
        finite_mask = np.isfinite(msd_arr_mm2)
        if np.count_nonzero(finite_mask) < 2:
            return np.nan
        fit_end = tau_arr[finite_mask][-1]
        fit_start = max(0.0, fit_end - fit_window_duration)
        mask = (tau_arr >= fit_start) & (tau_arr <= fit_end) & finite_mask
        if mask.sum() < 2:
            idx = np.where(finite_mask)[0]
            idx = idx[-min(10, len(idx)):]
            x = tau_arr[idx]
            y = msd_arr_mm2[idx]
        else:
            x = tau_arr[mask]
            y = msd_arr_mm2[mask]
        a, b = np.polyfit(x, y, 1)
        return a / 2.0

    # ---- Dispersion (mm^2/s -> m^2/s) ----
    D_ax_m2_s    = fit_tail_D(tau, msd_ax_mm2)    * MM2_TO_M2
    D_ra_m2_s    = fit_tail_D(tau, msd_ra_mm2)    * MM2_TO_M2
    D_ax_fl_m2_s = fit_tail_D(tau, msd_ax_fl_mm2) * MM2_TO_M2
    D_ra_fl_m2_s = fit_tail_D(tau, msd_ra_fl_mm2) * MM2_TO_M2

    # ---- Track length stats ----
    bx = tukey_box_from_hist(length_hist)

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
    print(f"   Tracks used: {bx['n_tracks']} | Particles used: {bx['n_particles']}")
    print(f"   Track length Q1/Med/Q3: {bx['q1']} / {bx['median']} / {bx['q3']}")
    print(f"   Axial D  = {D_ax_m2_s:.4e} m²/s | Radial D  = {D_ra_m2_s:.4e} m²/s")
    print(f"   Axial D' = {D_ax_fl_m2_s:.4e} m²/s | Radial D' = {D_ra_fl_m2_s:.4e} m²/s")

# ============================== SAVE EXCEL =============================
excel_summary_path = os.path.join(output_dir, "PTV_dispersion_summary.xlsx")
pd.DataFrame(summary).to_excel(excel_summary_path, index=False)
print(f"\n📗 Excel summary saved: {excel_summary_path}")

dispersion_only_path = os.path.join(output_dir, "Dispersion_vs_SVF.xlsx")
pd.DataFrame({
    "SVF_percent": [row["SVF_percent"] for row in summary],
    "D_axial_m2_per_s": [row["D_axial_m2_per_s"] for row in summary],
    "D_radial_m2_per_s": [row["D_radial_m2_per_s"] for row in summary],
    "D_axial_fluct_m2_per_s": [row["D_axial_fluct_m2_per_s"] for row in summary],
    "D_radial_fluct_m2_per_s": [row["D_radial_fluct_m2_per_s"] for row in summary],
}).to_excel(dispersion_only_path, index=False)
print(f"📘 Dispersion-only Excel saved: {dispersion_only_path}")

# ================================ FIGURES ==============================
# NOTE: MSD arrays are in mm^2; multiply by 1e-6 to show m^2 on plots.

# --- RADIAL MSD (NORMAL) — sci y-axis; legend by SVF ---
plt.figure(figsize=(FIG_W, FIG_H))
for svf, tau, msd_mm2 in zip(svf_levels_pct, all_time_sec, all_msd_ra_mm2):
    plt.plot(tau, msd_mm2 * MM2_TO_M2, label=f"{svf:d}% SVF", linewidth=LINE_W)
plt.xlabel("Time interval, $\\tau$ (s)")
plt.ylabel("Radial MSD (m$^2$)")
plt.title(f"Radial MSD vs $\\tau$  (Gas = {gas_rate_Lmin:.2f} L/min)")
ax = plt.gca()
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
plt.grid(True, alpha=0.3)
plt.legend(frameon=False, title="Solid VOF", loc="best")
plt.tight_layout()
base = os.path.join(output_dir, "Radial_MSD_vs_Time_SVF")
plt.savefig(base + ".svg", bbox_inches="tight")
plt.savefig(base + ".png", bbox_inches="tight", dpi=300)
plt.close()
print(f"🖼️ Saved: {base}.svg, {base}.png")

# --- AXIAL MSD (NORMAL) — sci y-axis; legend by SVF ---
plt.figure(figsize=(FIG_W, FIG_H))
for svf, tau, msd_mm2 in zip(svf_levels_pct, all_time_sec, all_msd_ax_mm2):
    plt.plot(tau, msd_mm2 * MM2_TO_M2, label=f"{svf:d}% SVF", linewidth=LINE_W)
plt.xlabel("Time interval, $\\tau$ (s)")
plt.ylabel("Axial MSD (m$^2$)")
plt.title(f"Axial MSD vs $\\tau$  (Gas = {gas_rate_Lmin:.2f} L/min)")
ax = plt.gca()
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
plt.grid(True, alpha=0.3)
plt.legend(frameon=False, title="Solid VOF", loc="best")
plt.tight_layout()
base = os.path.join(output_dir, "Axial_MSD_vs_Time_SVF")
plt.savefig(base + ".svg", bbox_inches="tight")
plt.savefig(base + ".png", bbox_inches="tight", dpi=300)
plt.close()
print(f"🖼️ Saved: {base}.svg, {base}.png")

# --- RADIAL MSD′ (FLUCTUATION-CORRECTED) — sci y-axis ---
plt.figure(figsize=(FIG_W, FIG_H))
for svf, tau, msd_mm2 in zip(svf_levels_pct, all_time_sec, all_msd_ra_fl_mm2):
    plt.plot(tau, msd_mm2 * MM2_TO_M2, label=f"{svf:d}% SVF", linewidth=LINE_W)
plt.xlabel("Time interval, $\\tau$ (s)")
plt.ylabel("Radial MSD′ (m$^2$)")
plt.title(f"Radial MSD′ (mean-free) vs $\\tau$  (Gas = {gas_rate_Lmin:.2f} L/min)")
ax = plt.gca()
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
plt.grid(True, alpha=0.3)
plt.legend(frameon=False, title="Solid VOF", loc="best")
plt.tight_layout()
base = os.path.join(output_dir, "Radial_MSD_fluct_vs_Time_SVF")
plt.savefig(base + ".svg", bbox_inches="tight")
plt.savefig(base + ".png", bbox_inches="tight", dpi=300)
plt.close()
print(f"🖼️ Saved: {base}.svg, {base}.png")

# --- AXIAL MSD′ (FLUCTUATION-CORRECTED) — sci y-axis ---
plt.figure(figsize=(FIG_W, FIG_H))
for svf, tau, msd_mm2 in zip(svf_levels_pct, all_time_sec, all_msd_ax_fl_mm2):
    plt.plot(tau, msd_mm2 * MM2_TO_M2, label=f"{svf:d}% SVF", linewidth=LINE_W)
plt.xlabel("Time interval, $\\tau$ (s)")
plt.ylabel("Axial MSD′ (m$^2$)")
plt.title(f"Axial MSD′ (mean-free) vs $\\tau$  (Gas = {gas_rate_Lmin:.2f} L/min)")
ax = plt.gca()
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
plt.grid(True, alpha=0.3)
plt.legend(frameon=False, title="Solid VOF", loc="best")
plt.tight_layout()
base = os.path.join(output_dir, "Axial_MSD_fluct_vs_Time_SVF")
plt.savefig(base + ".svg", bbox_inches="tight")
plt.savefig(base + ".png", bbox_inches="tight", dpi=300)
plt.close()
print(f"🖼️ Saved: {base}.svg, {base}.png")

# --- DISPERSION vs SVF (NORMAL) — dashed lines (axial & radial) ---
plt.figure(figsize=(FIG_W, FIG_H))
Da = [row["D_axial_m2_per_s"] for row in summary]
Dr = [row["D_radial_m2_per_s"] for row in summary]
plt.plot(svf_levels_pct, Da, "o--", linewidth=LINE_W, markersize=MARKER_SZ, label="Axial $D$")
plt.plot(svf_levels_pct, Dr, "s--", linewidth=LINE_W, markersize=MARKER_SZ, label="Radial $D$")
plt.xlabel("Solid VOF (%)")
plt.ylabel("Dispersion coefficient (m$^2$/s)")
plt.title(f"Effective dispersion vs SVF (normal MSD)  —  Gas = {gas_rate_Lmin:.2f} L/min")
ax = plt.gca()
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
plt.grid(True, alpha=0.3)
plt.legend(frameon=False, loc="best")
plt.tight_layout()
base = os.path.join(output_dir, "Dispersion_vs_SVF_NORMAL")
plt.savefig(base + ".svg", bbox_inches="tight")
plt.savefig(base + ".png", bbox_inches="tight", dpi=300)
plt.close()
print(f"🖼️ Saved: {base}.svg, {base}.png")

# --- DISPERSION vs SVF (FLUCTUATION-CORRECTED) — dashed lines ---
plt.figure(figsize=(FIG_W, FIG_H))
DaF = [row["D_axial_fluct_m2_per_s"] for row in summary]
DrF = [row["D_radial_fluct_m2_per_s"] for row in summary]
plt.plot(svf_levels_pct, DaF, "o--", linewidth=LINE_W, markersize=MARKER_SZ, label="Axial $D'$")
plt.plot(svf_levels_pct, DrF, "s--", linewidth=LINE_W, markersize=MARKER_SZ, label="Radial $D'$")
plt.xlabel("Solid VOF (%)")
plt.ylabel("Dispersion coefficient (m$^2$/s)")
plt.title(f"Effective dispersion vs SVF (mean-free MSD)  —  Gas = {gas_rate_Lmin:.2f} L/min")
ax = plt.gca()
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
plt.grid(True, alpha=0.3)
plt.legend(frameon=False, loc="best")
plt.tight_layout()
base = os.path.join(output_dir, "Dispersion_vs_SVF_FLUCT")
plt.savefig(base + ".svg", bbox_inches="tight")
plt.savefig(base + ".png", bbox_inches="tight", dpi=300)
plt.close()
print(f"🖼️ Saved: {base}.svg, {base}.png")

# --- COUNTS per LAG (semi-log-Y; τ > 0) ---
plt.figure(figsize=(FIG_W, FIG_H))
for svf, tau, c in zip(svf_levels_pct, all_time_sec, all_counts):
    mask = tau > 0
    plt.semilogy(tau[mask], c[mask], linewidth=LINE_W, label=f"{svf:d}% SVF")
plt.xlabel("Time interval, $\\tau$ (s)")
plt.ylabel("Increments per lag (count) [log]")
plt.title("MSD statistics: increments vs $\\tau$")
plt.grid(True, which="both", alpha=0.3)
plt.legend(frameon=False, title="Solid VOF", loc="best")
plt.tight_layout()
base = os.path.join(output_dir, "Counts_per_lag_semilogy_SVF")
plt.savefig(base + ".svg", bbox_inches="tight")
plt.savefig(base + ".png", bbox_inches="tight", dpi=300)
plt.close()
print(f"🖼️ Saved: {base}.svg, {base}.png")

# --------------------------- FINAL REPORT -----------------------------
print("\n✅ FINAL DISPERSION COEFFICIENTS (tail fit "
      f"{fit_window_duration:.3f} s), units m²/s, by SVF:\n")
for i, svf in enumerate(svf_levels_pct):
    print(f"  SVF {svf:>2d}%  ->  "
          f"Axial D  = {Da[i]:.4e} | Radial D  = {Dr[i]:.4e}   ||   "
          f"Axial D' = {DaF[i]:.4e} | Radial D' = {DrF[i]:.4e}")
print("\n🎯 Done.")
