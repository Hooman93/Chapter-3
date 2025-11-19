# ============================================
# Particle-track statistics (PTV) � FULL SCRIPT
# ============================================
# Summary:
# - Loads 3D PTV tracks for multiple cases (folders).
# - Computes velocities and accelerations from particle positions.
# - Builds PDFs (linear + semi-log) for Vx, Vy, Vz, |V|, |a| and ay.
# - Computes per-track RMS(Vx, Vy, Vz, |V|) and plots their PDFs.
# - Creates |V| boxplots across cases (e.g. varying gas flow rate).
# - Saves all figures (PNG+SVG) and per-case statistics to Excel/CSV.
# ============================================


# -------------------------
# Imports and configuration
# -------------------------
import os                                   # Filesystem utilities
import re                                   # Regex for parsing labels from paths
from pathlib import Path                    # Safer path handling
import numpy as np                          # Numerical computing
import pandas as pd                         # Tabular data and file export
import matplotlib.pyplot as plt             # Plotting
import matplotlib.colors as mcolors         # Color utilities (for darker marker edges)
import lvpyio as lv                         # PTV track reader

# -------------------------
# User inputs (edit below)
# -------------------------

data_paths = [
    r"E:\Three-phase Hydrogels\track data_dispersion\Three-phase\0%\0.05 lpermin\dipersion calculation_0% (images 500-1000)_0.05 lpmin",
    r"E:\Three-phase Hydrogels\track data_dispersion\Three-phase\0%\0.25 lpermin\dipersion calculation_0% (images 500-1000)_0.25 lpmin",
    r"E:\Three-phase Hydrogels\track data_dispersion\Three-phase\0%\0.5 lpermin\dipersion calculation_0% (images 500-1000)_0.5 lpmin",
]

output_root = r"C:\python files\Hydrogel_Three-phase\Figures\New results\velocity distribution analysis_varying gas flow rate\0%"  # Where figures/results will be written
os.makedirs(output_root, exist_ok=True)  # Create the output folder tree if it doesn't exist; do nothing if it already does


# Acquisition / scaling parameters
scale = 1 / 5.84                            # [mm/pixel] → spatial scale to convert pixels to mm
fps = 198.4                                 # [1/s] frames per second
dt = 1.0 / fps                              # [s] sampling interval between frames
min_track_length = 5                        # Minimum raw points required to accept a track
max_track_number = 10000                    # Hard cap to avoid processing too many tracks

# Figure/Style parameters (publication-oriented)
FIG_W, FIG_H = 3.4, 3.4                     # ~86 mm x 86 mm (A4 two-column)  <<< per your request
DPI = 600                                   # High-resolution saving
FONTSIZE = 9                                # Compact font sizes (journal figures)
MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X']  # Marker shapes for series distinction
MS = 2.4                                    # Marker size
MEW = 0.6                                   # Marker edge width
ALPHA = 0.95                                # Marker face alpha

# Apply a consistent Matplotlib style globally (thin axes, compact labels)
plt.rcParams.update({
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "font.size": FONTSIZE,
    "axes.labelsize": FONTSIZE,
    "axes.titlesize": FONTSIZE,
    "xtick.labelsize": FONTSIZE,
    "ytick.labelsize": FONTSIZE,
    "legend.fontsize": FONTSIZE,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})

# ------------------------------------------------------------
# Semi-log plot scaling controls (top-of-file, easy to tweak)
# ------------------------------------------------------------

# Manual Y-axis limits for SEMI-LOG PDFs; set to None for auto
# One-sided tuples like (1e-4,) mean "set only ymin"
SEMILOG_YLIMS_V = (1e-4, 40)                # For velocity PDFs on semi-log Y
SEMILOG_YLIMS_A = (1e-5, 1)                 # For acceleration PDFs on semi-log Y
SEMILOG_YLIMS_RMS = (1e-2,)                 # For RMS(V) PDFs on semi-log Y

# --- NEW: Optional per-variable Y-limit overrides (set to None to use group limits above)
SEMILOG_YLIMS_VX   = (1e-4, 100)            # e.g., (1e-4, 25)
SEMILOG_YLIMS_VY   = None
SEMILOG_YLIMS_VZ   = None
SEMILOG_YLIMS_VMAG = None
SEMILOG_YLIMS_AY   = (1e-5, 10)             # e.g., (2e-5, 0.5)
SEMILOG_YLIMS_AMAG = None

# Manual X-axis limits for SEMI-LOG PDFs; set to None for auto
# Requested: Vx/Vz semi-log x-range = (-0.3, 0.3), keep earlier ranges for others
SEMILOG_XLIMS_VX   = (-0.3, 0.3)            # Semi-log Vx x-range
SEMILOG_XLIMS_VY   = (-0.25, 0.45)          # Semi-log Vy x-range
SEMILOG_XLIMS_VZ   = (-0.3, 0.3)            # Semi-log Vz x-range
SEMILOG_XLIMS_VMAG = (None, 0.5)            # Semi-log |V| x-range
SEMILOG_XLIMS_AY   = (-15.0, 15.0)          # Semi-log ay x-range
SEMILOG_XLIMS_AMAG = (None, 25.0)           # Semi-log |a| x-range

# Optional: X-limits for RMS on semi-log; keep None (auto) unless you want to constrain
SEMILOG_XLIMS_RMS_VX   = None
SEMILOG_XLIMS_RMS_VY   = None
SEMILOG_XLIMS_RMS_VZ   = None
SEMILOG_XLIMS_RMS_VMAG = None

# ------------------------------------------------------------
# Optional export of histogram curves for exact re-plotting later
# ------------------------------------------------------------
SAVE_PDF_TABLES = False                     # Set True to write histogram curves to CSV
PDF_TABLE_CSV_NAME = "pdf_tables_all_variables.csv"

# ------------------------------------------------------------
# Excel output filename for per-case track statistics
# ------------------------------------------------------------
TRACK_STATS_XLSX_NAME = "track_statistics.xlsx"

# ============================================
# (1) PTV CORE FUNCTIONS — pure numerics
# ============================================

def central_diff_velocity(positions_m, dt):
    """
    Central-difference velocity from positions.
    positions_m : (N,3) array in meters
    returns     : (N-2,3) velocities in m/s
    """
    return (positions_m[2:] - positions_m[:-2]) / (2.0 * dt)

def acceleration_cdiff_on_velocity(velocities, dt):
    """
    Central-difference acceleration from velocities.
    velocities : (M,3) array in m/s
    returns    : (M-2,3) accelerations in m/s^2
    """
    return (velocities[2:] - velocities[:-2]) / (2.0 * dt)

def per_track_rms_from_velocity_sequence(v_seq):
    """
    RMS(Vx), RMS(Vy), RMS(Vz), RMS(|V|) for one track's velocity sequence.
    v_seq : (M,3) array in m/s
    """
    if v_seq.shape[0] == 0:                 # Empty guard
        return 0.0, 0.0, 0.0, 0.0
    vx = v_seq[:, 0]                        # Vx time series
    vy = v_seq[:, 1]                        # Vy time series
    vz = v_seq[:, 2]                        # Vz time series
    vmag = np.linalg.norm(v_seq, axis=1)    # |V|(t)
    rms_vx = float(np.sqrt(np.mean(vx**2))) # RMS(Vx)
    rms_vy = float(np.sqrt(np.mean(vy**2))) # RMS(Vy)
    rms_vz = float(np.sqrt(np.mean(vz**2))) # RMS(Vz)
    rms_vmag = float(np.sqrt(np.mean(vmag**2))) # RMS(|V|)
    return rms_vx, rms_vy, rms_vz, rms_vmag

def compute_case_from_ptv(ptv, min_len, max_tracks, scale_mm_per_px, dt):
    """
    Process a single PTV dataset into pooled arrays and per-track sequences.
    Returns:
      - flattened velocity samples: Vx, Vy, Vz, |V|
      - per-track velocity sequences (for accelerations & RMS)
      - per-track position sequences (retained here for parity; not used for MSD anymore)
      - stats dict (velocity-sample counts, tracks used)
    """
    # Pooled flattened arrays (collect all samples from all accepted tracks)
    vx, vy, vz, vmag = [], [], [], []
    # Per-track sequences (used for acceleration/RMS computations)
    vel_sequences = []
    # Positions in meters per track (kept for parity; MSD removed)
    pos_sequences = []
    # Meta stats for Excel output
    track_lengths_vel = []
    tracks_used = 0

    # Iterate over all tracks in the dataset
    for tid in range(ptv.track_count):
        if tracks_used >= max_tracks:       # Stop when we reached the cap
            break
        tr = ptv.single_track(tid)          # Fetch one track object
        n_raw = len(tr.particles)           # Number of raw (x,y,z) points in this track
        if n_raw <= min_len:                # Enforce minimum length
            continue

        # Convert raw pixels → meters: pixels * [mm/pixel] / 1000
        pos_m = np.array([[p["x"], p["y"], p["z"]] for p in tr.particles],
                         dtype=float) * (scale_mm_per_px / 1000.0)
        if pos_m.shape[0] < 3:              # Need ≥3 samples to compute central-diff velocity
            continue

        # Velocity (central difference) in m/s
        v_seq = central_diff_velocity(pos_m, dt)

        # Apply the Vy sign correction once here (as requested earlier)
        v_seq[:, 1] *= -1.0

        if v_seq.shape[0] == 0:             # Guard against empty result
            continue

        # Pool the component samples and |V|
        vx.extend(v_seq[:, 0])
        vy.extend(v_seq[:, 1])
        vz.extend(v_seq[:, 2])
        vmag.extend(np.linalg.norm(v_seq, axis=1))

        # Keep full sequences for acceleration and RMS
        vel_sequences.append(v_seq)
        # Keep positions (not used for MSD anymore; retained for minimal change)
        pos_sequences.append(pos_m)

        # Record number of velocity samples contributed by this track
        track_lengths_vel.append(v_seq.shape[0])
        tracks_used += 1                     # Count accepted track

    # Build simple stats for this case (velocity-sample perspective only)
    stats = {
        "tracks_used": tracks_used,
        "total_velocity_samples": int(np.sum(track_lengths_vel)) if track_lengths_vel else 0,
        "track_lengths_vel": np.array(track_lengths_vel, dtype=int),
    }

    # Return pooled arrays and per-track collections
    return (np.asarray(vx), np.asarray(vy), np.asarray(vz), np.asarray(vmag),
            vel_sequences, pos_sequences, stats)

def pooled_acc_components_and_magnitude(vel_sequences, dt):
    """
    From per-track velocities, compute pooled accelerations:
    returns flat arrays for ax, ay, az, and |a|.
    """
    ax_list, ay_list, az_list, amag_list = [], [], [], []

    for v_seq in vel_sequences:
        if v_seq.shape[0] < 3:              # Need ≥3 velocity samples to compute acceleration
            continue
        a_seq = acceleration_cdiff_on_velocity(v_seq, dt)   # (K,3) acceleration
        if a_seq.size == 0:
            continue
        ax_list.append(a_seq[:, 0])         # Collect ax
        ay_list.append(a_seq[:, 1])         # Collect ay
        az_list.append(a_seq[:, 2])         # Collect az
        amag_list.append(np.linalg.norm(a_seq, axis=1))  # Collect |a|

    # Concatenate lists into arrays (or return empty arrays if none)
    ax = np.concatenate(ax_list) if ax_list else np.array([])
    ay = np.concatenate(ay_list) if ay_list else np.array([])
    az = np.concatenate(az_list) if az_list else np.array([])
    amag = np.concatenate(amag_list) if amag_list else np.array([])
    return ax, ay, az, amag

def per_track_rms_distributions(vel_sequences):
    """
    Compute per-track distributions of RMS(Vx), RMS(Vy), RMS(Vz), RMS(|V|).
    Returns dict of arrays keyed by 'Vx', 'Vy', 'Vz', '|V|'.
    """
    rms_vx, rms_vy, rms_vz, rms_vmag = [], [], [], []
    for v_seq in vel_sequences:
        r_vx, r_vy, r_vz, r_vmag = per_track_rms_from_velocity_sequence(v_seq)
        rms_vx.append(r_vx)
        rms_vy.append(r_vy)
        rms_vz.append(r_vz)
        rms_vmag.append(r_vmag)
    return {"Vx": np.array(rms_vx), "Vy": np.array(rms_vy), "Vz": np.array(rms_vz), "|V|": np.array(rms_vmag)}

# ============================================
# (2) PLOTTING / GRAPH FUNCTIONS
# ============================================

def short_label_from_path(path_str):
    """
    Build a compact legend label from the folder path: '{SVF%} | {gas L/min}'.
    Extracts 'xx%' and 'yy lpermin' if present; otherwise uses fallbacks.
    """
    m_svf = re.search(r'(\d+(?:\.\d+)?)\s*%', path_str)              # Find a percentage like '30%'
    svf = f"{m_svf.group(1)}%" if m_svf else "SVF?"
    m_gas = re.search(r'(\d+(?:\.\d+)?)\s*lpermin', path_str, flags=re.I)  # Find '0.5 lpermin'
    gas = f"{m_gas.group(1)} L/min" if m_gas else "Q_g ?"
    return f"{svf} | {gas}"

def extract_gas_from_label(label):
    """
    Extract only the gas flow rate from a full case label, e.g.
    '0% | 0.05 L/min' -> '0.05'.
    Used for boxplot x-axis labels.
    """
    m = re.search(r'(\d+(?:\.\d+)?)\s*L/min', label)
    return f"{m.group(1)}" if m else label

def darker(rgb, factor=0.75):
    """
    Darken a color toward black by 'factor' (<1 makes it darker).
    Accepts color names or (r,g,b) tuples.
    """
    r, g, b = mcolors.to_rgb(rgb)
    return (r * factor, g * factor, b * factor)

def density_histogram_line(x, bins):
    """
    Build a density histogram (integrates to 1) and return
    centers, densities, left edges, right edges (for CSV export).
    """
    h, e = np.histogram(x, bins=bins, density=True)  # density=True normalizes the area to 1
    c = 0.5 * (e[:-1] + e[1:])                       # Bin centers
    return c, h, e[:-1], e[1:]                       # Return centers, PDF, edges

def common_bins_from_pooled(data_list, nbins=60):
    """
    Create common linear bins spanning pooled min..max across all arrays (for fair comparisons).
    Falls back to [-1,1] if no data.
    """
    pooled = np.concatenate([d for d in data_list if d.size > 0]) if data_list else np.array([])
    if pooled.size == 0:
        return np.linspace(-1.0, 1.0, nbins)
    vmin, vmax = float(np.min(pooled)), float(np.max(pooled))
    if vmin == vmax:                            # Degenerate case: expand slightly
        eps = 1e-12 if vmin == 0 else abs(vmin) * 1e-6
        vmin -= eps
        vmax += eps
    return np.linspace(vmin, vmax, nbins)       # Uniform linear bins

def plot_pdf_dict(qdict, title, xlabel, out_stub, nbins=60, semilogy=False, y_limits=None, x_limits=None):
    """
    Plot PDFs for dict {label: array} with common bins.
    - Markers only (no connecting lines) for clean journal-ready scatter-curve look.
    - If 'semilogy' is True, use log scale on Y (PDF axis); X remains linear.
    - 'y_limits' and 'x_limits' can be (min, max) or one-sided tuples like (min,).
    - Saves both PNG and SVG; returns rows for optional CSV export of the curves.
    """
    rows = []                                                   # For optional CSV export
    bins = common_bins_from_pooled(list(qdict.values()), nbins) # Build common bins across all series

    # Create a figure with the requested size (NOT square anymore)
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))              # <<< uses FIG_W, FIG_H

    # Color cycle for consistent color assignment across plots
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0','C1','C2','C3','C4','C5','C6'])

    # Loop through each series (case) and plot its density histogram as markers
    for idx, (label, arr) in enumerate(qdict.items()):
        if arr.size == 0:                                       # Skip empty arrays
            continue
        centers, dens, lefts, rights = density_histogram_line(arr, bins)  # PDF points
        color = color_cycle[idx % len(color_cycle)]             # Get color for this series
        edge = darker(color, 0.7)                               # Slightly darker edges
        mk = MARKERS[idx % len(MARKERS)]                        # Marker shape

        # Semi-log Y when requested; otherwise linear axes
        if semilogy:
            ax.semilogy(centers, dens, linestyle='None', marker=mk,
                        markersize=MS, markerfacecolor=color, markeredgecolor=edge,
                        markeredgewidth=MEW, alpha=ALPHA, label=label)
        else:
            ax.plot(centers, dens, linestyle='None', marker=mk,
                    markersize=MS, markerfacecolor=color, markeredgecolor=edge,
                    markeredgewidth=MEW, alpha=ALPHA, label=label)

        # Accumulate exact curve points for later CSV export if needed
        for bl, br, bc, pdens in zip(lefts, rights, centers, dens):
            rows.append({
                "title": title,
                "label": label,
                "bin_left": bl,
                "bin_right": br,
                "bin_center": bc,
                "pdf_density": pdens
            })

    # Axis labels/titles — per your preferences
    ax.set_xlabel(xlabel)                       # Velocity labels use "m/s"; acceleration uses "m s^-2"
    ax.set_ylabel("PDF")                        # Always "PDF" on the Y-axis
    ax.set_title(title)                         # Descriptive title

    # Apply manual Y limits (semi-log Y) if provided
    if semilogy and (y_limits is not None):
        if (len(y_limits) == 1) or (y_limits[1] is None):  # One-sided tuple → set only bottom
            ax.set_ylim(bottom=y_limits[0])
        else:
            ax.set_ylim(*y_limits)

    # Apply manual X limits if provided
    if x_limits is not None:
        if (len(x_limits) == 1) or (x_limits[1] is None):  # One-sided tuple → set only left
            ax.set_xlim(left=x_limits[0])
        else:
            ax.set_xlim(*x_limits)

    # Subtle grid and clean legend
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(frameon=False)

    # Save with tight layout to avoid clipping
    plt.tight_layout()
    fig.savefig(os.path.join(output_root, f"{out_stub}.png"), dpi=DPI)  # PNG
    fig.savefig(os.path.join(output_root, f"{out_stub}.svg"))           # SVG

    # Show for interactive work (e.g., Spyder)
    plt.show()

    # Return the rows so caller can optionally write CSV
    return rows

def plot_vmag_boxplots(vmag_by, title, out_stub):
    """
    Plot boxplots of |V| (velocity magnitude) for each case.
    - One box per case label (effect of gas flow rate at given solid VOF).
    - X-axis shows only gas flow rate (L/min).
    - Draws two versions: with and without outliers; both show mean values.
    - Prints a suggested caption including mean and median |V| for each gas flow rate.
    """
    if not vmag_by:
        return

    # Ensure deterministic order of cases
    case_labels = list(vmag_by.keys())                     # Full case labels (e.g. '0% | 0.05 L/min')
    data = [np.asarray(vmag_by[k]) for k in case_labels]   # Corresponding |V| arrays

    # Extract only the gas flow part (e.g. '0.05', '0.25', '0.5') for x-axis
    display_labels = [extract_gas_from_label(lbl) for lbl in case_labels]

    # ---- Compute mean and median for caption ----
    means = [np.mean(d) if d.size > 0 else np.nan for d in data]
    medians = [np.median(d) if d.size > 0 else np.nan for d in data]

    gas_str = ", ".join(display_labels)
    mean_str = ", ".join(f"{m:.3f}" for m in means)
    median_str = ", ".join(f"{m:.3f}" for m in medians)

    caption = (
        r"|V| distribution for gas flow rates "
        f"{gas_str} L/min (at 0% SVF). "
        r"Mean |V| = [" + mean_str + r"] m/s and median |V| = [" +
        median_str + r"] m/s, for gas flow rates " + gas_str + r" L/min, respectively."
    )

    print("\nSuggested caption for boxplot figure:\n" + caption + "\n")

    # --- Version 1: boxplot with outliers and mean marker ---
    fig1, ax1 = plt.subplots(figsize=(FIG_W, FIG_H))
    ax1.boxplot(
        data,
        labels=display_labels,
        showfliers=True,   # Keep outliers
        showmeans=True     # Show mean value as a separate marker
    )

    ax1.set_xlabel("Gas flow rate (L/min)")               # X-axis = gas flow rate
    ax1.set_ylabel("|V| (m/s)")                           # Velocity magnitude in m/s
    ax1.set_title(title)

    ax1.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    fig1.savefig(os.path.join(output_root, f"{out_stub}.png"), dpi=DPI)
    fig1.savefig(os.path.join(output_root, f"{out_stub}.svg"))
    plt.show()

    # --- Version 2: boxplot without outliers, still showing mean ---
    fig2, ax2 = plt.subplots(figsize=(FIG_W, FIG_H))
    ax2.boxplot(
        data,
        labels=display_labels,
        showfliers=False,  # Remove outliers
        showmeans=True     # Still show mean
    )

    ax2.set_xlabel("Gas flow rate (L/min)")
    ax2.set_ylabel("|V| (m/s)")
    ax2.set_title(title + " (no outliers)",fontsize=6)

    ax2.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    fig2.savefig(os.path.join(output_root, f"{out_stub}_no_outliers.png"), dpi=DPI)
    fig2.savefig(os.path.join(output_root, f"{out_stub}_no_outliers.svg"))
    plt.show()

# ============================================
# (3) READ DIRECTORIES + MAIN PIPELINE
# ============================================

def process_all_cases(data_paths):
    """
    Iterate over provided case folders, read PTV data, compute pooled arrays and stats.
    Returns a tuple of dicts/structures needed for plotting and tabulation.
    """
    # Containers for pooled PDFs per case (key = case label)
    vx_by, vy_by, vz_by, vmag_by = {}, {}, {}, {}                # Velocity samples
    ax_by, ay_by, az_by, amag_by = {}, {}, {}, {}                # Acceleration samples
    rms_vx_by, rms_vy_by, rms_vz_by, rms_vmag_by = {}, {}, {}, {}# Per-track RMS samples
    pos_seqs_by_case = {}                                        # Positions per case (kept for parity)
    track_stats_rows = []                                        # Rows for Excel stats

    # Loop over case folders
    for case_path in data_paths:
        label = short_label_from_path(case_path)                 # Build compact legend label
        ptv = lv.read_particles(case_path)                       # Load DaVis track data

        # Compute pooled velocities + per-track sequences + stats for this case
        vx, vy, vz, vmag, vel_seqs, pos_seqs, stats = compute_case_from_ptv(
            ptv, min_track_length, max_track_number, scale, dt
        )

        # Compute pooled accelerations from per-track velocity sequences
        ax, ay, az, amag = pooled_acc_components_and_magnitude(vel_seqs, dt)

        # Compute per-track RMS distributions
        rms_dict = per_track_rms_distributions(vel_seqs)

        # Store arrays under this case label
        vx_by[label] = vx;   vy_by[label] = vy;   vz_by[label] = vz;   vmag_by[label] = vmag
        ax_by[label] = ax;   ay_by[label] = ay;   az_by[label] = az;   amag_by[label] = amag
        rms_vx_by[label] = rms_dict["Vx"]; rms_vy_by[label] = rms_dict["Vy"]
        rms_vz_by[label] = rms_dict["Vz"]; rms_vmag_by[label] = rms_dict["|V|"]

        # Keep positions (MSD removed, but we retain variable for minimal code changes)
        pos_seqs_by_case[label] = pos_seqs

        # Append a single row of track stats (velocity-sample centric)
        append_track_stats_rows_velocity_only(track_stats_rows, label, stats)

    # Return all structures used downstream
    return (vx_by, vy_by, vz_by, vmag_by,
            ax_by, ay_by, az_by, amag_by,
            rms_vx_by, rms_vy_by, rms_vz_by, rms_vmag_by,
            pos_seqs_by_case, track_stats_rows)

# ============================================
# (4) EXPORT UTILITIES (Excel + optional CSVs)
# ============================================

def boxplot_stats(values):
    """
    Return common descriptive statistics used in boxplots and summaries.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return dict(count=0, min=np.nan, q1=np.nan, median=np.nan, q3=np.nan,
                    max=np.nan, mean=np.nan, std=np.nan)
    q1, med, q3 = np.percentile(arr, [25, 50, 75])
    return dict(
        count=int(arr.size),
        min=float(np.min(arr)),
        q1=float(q1),
        median=float(med),
        q3=float(q3),
        max=float(np.max(arr)),
        mean=float(np.mean(arr)),
        std=float(np.std(arr, ddof=1))  # sample standard deviation
    )

def append_track_stats_rows_velocity_only(rows, case_label, stats_dict):
    """
    Append a concise velocity-based stats row for Excel export:
      - number of tracks used
      - total velocity samples
      - distribution of velocity-sample lengths per track
    """
    vel_stats = boxplot_stats(stats_dict["track_lengths_vel"])
    rows.append({
        "case": case_label,
        "metric": "velocity_samples_per_track",
        "tracks_used": stats_dict["tracks_used"],
        "total_velocity_samples": stats_dict["total_velocity_samples"],
        **vel_stats
    })

def robust_save_excel(df, path_no_ext):
    """
    Save DataFrame to .xlsx using openpyxl if available; else xlsxwriter; else fallback to CSV.
    Returns the actual path written (useful in logs).
    """
    xlsx_path = path_no_ext if path_no_ext.lower().endswith(".xlsx") else path_no_ext + ".xlsx"
    try:
        df.to_excel(xlsx_path, index=False, engine="openpyxl")
        return xlsx_path
    except Exception:
        try:
            df.to_excel(xlsx_path, index=False, engine="xlsxwriter")
            return xlsx_path
        except Exception:
            csv_fallback = path_no_ext.replace(".xlsx", "") + ".csv"
            df.to_csv(csv_fallback, index=False)
            return csv_fallback

# ============================================
# RUN: compute, plot (PDFs), save stats
# ============================================

# Run the processing pipeline across all cases
(vx_by, vy_by, vz_by, vmag_by,
 ax_by, ay_by, az_by, amag_by,
 rms_vx_by, rms_vy_by, rms_vz_by, rms_vmag_by,
 pos_seqs_by_case, track_stats_rows) = process_all_cases(data_paths)

# Will hold optional histogram curve rows (for CSV)
pdf_rows_master = []

# --------------------------
# Velocity PDFs (linear + semi-log)
# --------------------------
# NOTE: Per your unit preference, velocity labels use "m/s" (not m s^-1).
pdf_rows_master += plot_pdf_dict(
    vx_by, "Vx PDF across cases (linear)", "Vx (m/s)",
    "PDF_Vx_linear_markers", nbins=60, semilogy=False
)
pdf_rows_master += plot_pdf_dict(
    vx_by, "Vx PDF across cases (semi-log)", "Vx (m/s)",
    "PDF_Vx_semilog_markers", nbins=60, semilogy=True,
    y_limits=(SEMILOG_YLIMS_VX if SEMILOG_YLIMS_VX is not None else SEMILOG_YLIMS_V),
    x_limits=SEMILOG_XLIMS_VX
)

pdf_rows_master += plot_pdf_dict(
    vy_by, "Vy PDF across cases (linear)", "Vy (m/s)",
    "PDF_Vy_linear_markers", nbins=60, semilogy=False
)
pdf_rows_master += plot_pdf_dict(
    vy_by, "Vy PDF across cases (semi-log)", "Vy (m/s)",
    "PDF_Vy_semilog_markers", nbins=60, semilogy=True,
    y_limits=(SEMILOG_YLIMS_VY if SEMILOG_YLIMS_VY is not None else SEMILOG_YLIMS_V),
    x_limits=SEMILOG_XLIMS_VY
)

pdf_rows_master += plot_pdf_dict(
    vz_by, "Vz PDF across cases (linear)", "Vz (m/s)",
    "PDF_Vz_linear_markers", nbins=60, semilogy=False
)
pdf_rows_master += plot_pdf_dict(
    vz_by, "Vz PDF across cases (semi-log)", "Vz (m/s)",
    "PDF_Vz_semilog_markers", nbins=60, semilogy=True,
    y_limits=(SEMILOG_YLIMS_VZ if SEMILOG_YLIMS_VZ is not None else SEMILOG_YLIMS_V),
    x_limits=SEMILOG_XLIMS_VZ
)

pdf_rows_master += plot_pdf_dict(
    vmag_by, "|V| PDF across cases (linear)", "|V| (m/s)",
    "PDF_Vmag_linear_markers", nbins=60, semilogy=False
)
pdf_rows_master += plot_pdf_dict(
    vmag_by, "|V| PDF across cases (semi-log)", "|V| (m/s)",
    "PDF_Vmag_semilog_markers", nbins=60, semilogy=True,
    y_limits=(SEMILOG_YLIMS_VMAG if SEMILOG_YLIMS_VMAG is not None else SEMILOG_YLIMS_V),
    x_limits=SEMILOG_XLIMS_VMAG
)

# --------------------------
# Acceleration PDFs (|a| and a_y)
# --------------------------
# NOTE: For acceleration labels, keep classical SI formatting 'm s^-2'.
pdf_rows_master += plot_pdf_dict(
    amag_by, "|a| PDF across cases (linear)", "|a| (m/s$^{2}$)",
    "PDF_Amag_linear_markers", nbins=60, semilogy=False
)
pdf_rows_master += plot_pdf_dict(
    amag_by, "|a| PDF across cases (semi-log)", "|a| (m/s$^{2}$)",
    "PDF_Amag_semilog_markers", nbins=60, semilogy=True,
    y_limits=(SEMILOG_YLIMS_AMAG if SEMILOG_YLIMS_AMAG is not None else SEMILOG_YLIMS_A),
    x_limits=SEMILOG_XLIMS_AMAG
)

pdf_rows_master += plot_pdf_dict(
    ay_by, "a$_y$ PDF across cases (linear)", "a$_y$ (m/s$^{2}$)",
    "PDF_Ay_linear_markers", nbins=60, semilogy=False
)
pdf_rows_master += plot_pdf_dict(
    ay_by, "a$_y$ PDF across cases (semi-log)", "a$_y$ (m/s$^{2}$)",
    "PDF_Ay_semilog_markers", nbins=60, semilogy=True,
    y_limits=(SEMILOG_YLIMS_AY if SEMILOG_YLIMS_AY is not None else SEMILOG_YLIMS_A),
    x_limits=SEMILOG_XLIMS_AY
)

# --------------------------
# Per-track RMS PDFs (linear + semi-log)
# --------------------------
# NOTE: Velocity labels here also use "m/s".
pdf_rows_master += plot_pdf_dict(
    rms_vx_by, "Per-track RMS(Vx) PDF (linear)", "RMS Vx (m/s)",
    "PDF_RMS_Vx_pertrack_linear", nbins=50, semilogy=False
)
pdf_rows_master += plot_pdf_dict(
    rms_vx_by, "Per-track RMS(Vx) PDF (semi-log)", "RMS Vx (m/s)",
    "PDF_RMS_Vx_pertrack_semilog", nbins=50, semilogy=True,
    y_limits=SEMILOG_YLIMS_RMS, x_limits=SEMILOG_XLIMS_RMS_VX
)

pdf_rows_master += plot_pdf_dict(
    rms_vy_by, "Per-track RMS(Vy) PDF (linear)", "RMS Vy (m/s)",
    "PDF_RMS_Vy_pertrack_linear", nbins=50, semilogy=False
)
pdf_rows_master += plot_pdf_dict(
    rms_vy_by, "Per-track RMS(Vy) PDF (semi-log)", "RMS Vy (m/s)",
    "PDF_RMS_Vy_pertrack_semilog", nbins=50, semilogy=True,
    y_limits=SEMILOG_YLIMS_RMS, x_limits=SEMILOG_XLIMS_RMS_VY
)

pdf_rows_master += plot_pdf_dict(
    rms_vz_by, "Per-track RMS(Vz) PDF (linear)", "RMS Vz (m/s)",
    "PDF_RMS_Vz_pertrack_linear", nbins=50, semilogy=False
)
pdf_rows_master += plot_pdf_dict(
    rms_vz_by, "Per-track RMS(Vz) PDF (semi-log)", "RMS Vz (m/s)",
    "PDF_RMS_Vz_pertrack_semilog", nbins=50, semilogy=True,
    y_limits=SEMILOG_YLIMS_RMS, x_limits=SEMILOG_XLIMS_RMS_VZ
)

pdf_rows_master += plot_pdf_dict(
    rms_vmag_by, "Per-track RMS(|V|) PDF (linear)", "RMS |V| (m/s)",
    "PDF_RMS_Vmag_pertrack_linear", nbins=50, semilogy=False
)
pdf_rows_master += plot_pdf_dict(
    rms_vmag_by, "Per-track RMS(|V|) PDF (semi-log)", "RMS |V| (m/s)",
    "PDF_RMS_Vmag_pertrack_semilog", nbins=50, semilogy=True,
    y_limits=SEMILOG_YLIMS_RMS, x_limits=SEMILOG_XLIMS_RMS_VMAG
)

# --------------------------
# |V| boxplots per case (new analysis)
# --------------------------
plot_vmag_boxplots(
    vmag_by,
    "|V| distribution across gas flow rates (boxplots)",
    "Boxplot_Vmag_across_gas_rates"
)

# --------------------------
# Save per-case track statistics to Excel
# (velocity rows + |V| boxplot stats)
# --------------------------
# Add |V| boxplot statistics per case to the same table
for case_label, arr in vmag_by.items():
    stats_vmag = boxplot_stats(arr)
    track_stats_rows.append({
        "case": case_label,
        "metric": "vmag_boxplot",
        **stats_vmag
    })

df_stats = pd.DataFrame(track_stats_rows)                                    # Collect stats rows into DataFrame
stats_path = robust_save_excel(df_stats, os.path.join(output_root, TRACK_STATS_XLSX_NAME))  # Save with fallback logic
print("Saved track statistics file:", stats_path)                             # Log where it went

# --------------------------
# Optional: save histogram curves to CSV for re-plotting elsewhere
# --------------------------
if SAVE_PDF_TABLES:
    df_pdf = pd.DataFrame(
        pdf_rows_master,
        columns=["title", "label", "bin_left", "bin_right", "bin_center", "pdf_density"]
    )
    pdf_csv_path = os.path.join(output_root, PDF_TABLE_CSV_NAME)
    df_pdf.to_csv(pdf_csv_path, index=False)
    print("Saved PDF tables CSV:", pdf_csv_path)

# ============================================
# END — (MSD/dispersion analysis removed)
# ============================================

print("Done. Figures and stats are in:", output_root)
