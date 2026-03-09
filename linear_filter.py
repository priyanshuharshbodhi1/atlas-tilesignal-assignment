"""
ATLAS Tile Calorimeter - Linear Energy Reconstruction (Optimal Filter)
======================================================================
GSoC 2026 Evaluation Assignment
Author: Priyanshu

The ATLAS TileCal reads electrical pulses every 25 ns (one Bunch Crossing, BC).
Each BC gives a digital ADC count (sample_lo). Energy (ene_lo) cannot be read
directly -- it must be inferred from the shape of the pulse across 7 samples
centered on the BC of interest. The known, consistent pulse shape is what makes
a weighted linear sum work: the weights act as a matched filter that suppresses
pedestal noise and pile-up while extracting the pulse amplitude.

This technique is the classical "Optimal Filter" in calorimetry (Fullana et al.),
equivalent to the Wiener filter applied to a known signal shape in colored noise.
Fitting a LinearRegression to (7 samples -> energy) learns exactly these filter
coefficients from data.

The figure of merit is (E_reco - E_true) / E_true, evaluated on the test set.
- Mean close to 0  => unbiased reconstruction
- Small RMS        => precise reconstruction

References:
  E. Fullana, "Optimal Filtering in the ATLAS Hadronic Tile Calorimeter"
  D. Oliveira, "Energy reconstruction of ATLAS TileCal using the Wiener filter"
  F. Curcio, "Machine Learning-Based Energy Reconstruction for ATLAS TileCal at HL-LHC"
"""

import glob
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUT_DIR  = os.path.dirname(__file__)

# Physical energy threshold: skip near-zero-energy BCs to avoid 1/E divergence.
# Below this the signal is just pedestal noise and the ratio is meaningless.
ENERGY_THRESHOLD = 10.0   # physical units (same as ene_lo after denorm)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_split(split: str):
    """Load all .pt shards for a given split (train / val / test).

    Each shard is a dict with keys:
      X     : float32 [N, 2, 7]  -- 7-sample windows, channels (hi, lo)
      y     : float32 [N, 2]     -- true energies, channels (hi, lo), normalised
      y_OF  : float32 [N, 2]     -- pre-computed Optimal Filter estimates

    Returns X_lo [M, 7], y_lo_norm [M], y_OF_lo [M] concatenated across shards.
    """
    pattern = os.path.join(DATA_DIR, split, f"{split}_*.pt")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No shards found at {pattern}")

    X_list, y_list, yOF_list = [], [], []
    for f in files:
        shard = torch.load(f, weights_only=False)
        X_list.append(shard["X"].numpy())      # [N, 2, 7]
        y_list.append(shard["y"].numpy())      # [N, 2]
        yOF_list.append(shard["y_OF"].numpy()) # [N, 2]

    X   = np.concatenate(X_list,   axis=0)
    y   = np.concatenate(y_list,   axis=0)
    yOF = np.concatenate(yOF_list, axis=0)

    # Channel index 1 is the lo-gain channel used in this assignment
    X_lo    = X[:, 1, :]    # [M, 7]  -- the 7-sample window
    y_lo    = y[:, 1]       # [M]     -- normalised true energy
    yOF_lo  = yOF[:, 1]     # [M]     -- OF predicted energy (raw units)

    print(f"  [{split}] {len(files)} shards -> {X_lo.shape[0]:,} events")
    return X_lo, y_lo, yOF_lo


def load_y_stats():
    """Load normalisation statistics stored as .npy files."""
    mean = np.load(os.path.join(DATA_DIR, "y_stats", "mean.npy"))  # shape (1,2)
    std  = np.load(os.path.join(DATA_DIR, "y_stats", "std.npy"))   # shape (1,2)
    return float(mean[0, 1]), float(std[0, 1])  # lo-gain mean, std


def denormalise(y_norm, mean, std):
    return y_norm * std + mean


# ---------------------------------------------------------------------------
# Feature engineering -- timing proxy for the bonus task
# ---------------------------------------------------------------------------

def timing_proxy(X_lo):
    """Estimate pulse timing from the centre-of-mass of the 7-sample window.

    For a noise-free, on-time pulse the peak is at sample index 3 (centre).
    If the pulse arrived one BC early the peak shifts left; one BC late it
    shifts right.  The weighted centre-of-mass gives a continuous timing
    estimate; we round it to {-1, 0, +1} for the discrete label.

    Returns
    -------
    t_cont  : float [N]    continuous centre-of-mass offset from centre
    t_disc  : int   [N]    discretised to {-1, 0, +1}
    """
    indices   = np.arange(7, dtype=float)   # [0 .. 6]
    row_sum   = X_lo.sum(axis=1)
    safe_mask = row_sum > 1e-6              # skip all-zero rows

    t_cont = np.zeros(len(X_lo))
    t_cont[safe_mask] = (
        (X_lo[safe_mask] * indices).sum(axis=1) / row_sum[safe_mask] - 3.0
    )
    t_disc = np.clip(np.round(t_cont).astype(int), -1, 1)
    return t_cont, t_disc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("ATLAS TileCal -- Linear Energy Reconstruction (Optimal Filter)")
    print("=" * 65)

    # Load normalisation stats
    mean_lo, std_lo = load_y_stats()
    print(f"\nEnergy normalisation  mean={mean_lo:.4f}  std={std_lo:.4f}\n")

    # Load all splits
    print("Loading data ...")
    X_tr, y_tr_n, yOF_tr = load_split("train")
    X_va, y_va_n, yOF_va = load_split("val")
    X_te, y_te_n, yOF_te = load_split("test")

    # Denormalise to physical units
    y_tr  = denormalise(y_tr_n, mean_lo, std_lo)
    y_va  = denormalise(y_va_n, mean_lo, std_lo)
    y_te  = denormalise(y_te_n, mean_lo, std_lo)
    yOF_te_phys = yOF_te  # y_OF is already in physical units

    # ------------------------------------------------------------------
    # 1. Fit the linear model (Optimal Filter coefficients)
    #
    #    E_reco = w0*s[-3] + w1*s[-2] + ... + w6*s[+3] + bias
    #
    #    sklearn LinearRegression minimises sum of squared residuals.
    #    The learned weights w_i are the amplitude filter coefficients
    #    of the classical ATLAS Optimal Filter.
    # ------------------------------------------------------------------
    print("\nFitting LinearRegression on training data ...")
    model = LinearRegression()
    model.fit(X_tr, y_tr)

    # Validation R^2 as a sanity check
    val_r2 = model.score(X_va, y_va)
    print(f"  Validation R^2 = {val_r2:.6f}")

    # ------------------------------------------------------------------
    # 2. Evaluate on test set
    # ------------------------------------------------------------------
    print("\nEvaluating on test set ...")
    y_pred_te = model.predict(X_te)

    # Figure of merit: relative residual, filtered on non-zero energies
    mask = np.abs(y_te) > ENERGY_THRESHOLD
    y_true_f = y_te[mask]
    y_pred_f = y_pred_te[mask]

    residual = (y_pred_f - y_true_f) / y_true_f

    fom_mean = float(np.mean(residual))
    fom_rms  = float(np.std(residual))

    print(f"\n  Events passing |E_true| > {ENERGY_THRESHOLD}: {mask.sum():,} / {len(y_te):,}")
    print(f"\n  Figure of Merit (test set):")
    print(f"    Mean  [ (E_reco - E_true) / E_true ] = {fom_mean:+.6f}")
    print(f"    RMS   [ (E_reco - E_true) / E_true ] = {fom_rms:.6f}")

    # ------------------------------------------------------------------
    # 3. Learned filter weights
    # ------------------------------------------------------------------
    offsets = [-3, -2, -1, 0, +1, +2, +3]
    print("\n  Learned Optimal Filter coefficients:")
    print(f"  {'Sample offset':>15}  {'Weight':>12}")
    print("  " + "-" * 30)
    for off, w in zip(offsets, model.coef_):
        bar = "#" * int(abs(w) / max(abs(model.coef_)) * 20)
        print(f"  {'s[' + str(off) + ']':>15}  {w:+12.4f}  {bar}")
    print(f"  {'bias':>15}  {model.intercept_:+12.4f}")

    # The central sample s[0] should dominate (highest weight) because it is
    # proportional to the pulse peak amplitude.  Pre- and post-peak samples
    # carry shape information that helps cancel pile-up contributions.

    # ------------------------------------------------------------------
    # 4. Plot 1 -- histogram of relative residual
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5))

    # Clip outliers for display only (keep stat on full distribution)
    clip = 1.5
    res_plot = np.clip(residual, -clip, clip)

    n, bins, _ = ax.hist(
        res_plot, bins=150, range=(-clip, clip),
        color="#2166ac", alpha=0.85, edgecolor="none",
        label="Linear filter (test)"
    )

    ymax = n.max()
    ax.axvline(fom_mean, color="#d7191c", lw=1.8, ls="--",
               label=f"Mean = {fom_mean:+.4f}")
    ax.axvline(fom_mean - fom_rms, color="#fdae61", lw=1.2, ls=":",
               label=f"RMS  = {fom_rms:.4f}")
    ax.axvline(fom_mean + fom_rms, color="#fdae61", lw=1.2, ls=":")

    # Annotate mean and RMS on the plot
    ax.text(0.97, 0.95, f"Mean = {fom_mean:+.4f}\nRMS  = {fom_rms:.4f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.8))

    ax.set_xlabel(r"$(E_{\rm reco} - E_{\rm true})\;/\;E_{\rm true}$", fontsize=13)
    ax.set_ylabel("Events / bin", fontsize=13)
    ax.set_title("ATLAS TileCal -- Lo-gain Energy Residual Distribution (test)", fontsize=12)
    ax.legend(fontsize=11)
    ax.set_xlim(-clip, clip)
    plt.tight_layout()
    path1 = os.path.join(OUT_DIR, "plot1_residual_histogram.png")
    fig.savefig(path1, dpi=150)
    plt.close(fig)
    print(f"\n  Saved Plot 1 -> {path1}")

    # ------------------------------------------------------------------
    # 5. Plot 2 -- 2-D hexbin: residual vs true energy
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6))

    # Use log-scale color for the hexbin since the distribution spans orders
    hb = ax.hexbin(
        y_true_f, np.clip(residual, -clip, clip),
        gridsize=80, cmap="viridis",
        mincnt=1, bins="log",
        extent=[y_true_f.min(), min(y_true_f.max(), 2000), -clip, clip]
    )
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("log$_{10}$(counts)", fontsize=11)

    ax.axhline(0,         color="white",   lw=1.2, ls="--", alpha=0.7)
    ax.axhline(fom_mean,  color="#d7191c", lw=1.5, ls="--",
               label=f"Mean = {fom_mean:+.4f}")
    ax.axhline(fom_mean + fom_rms, color="#fdae61", lw=1.2, ls=":",
               label=f"±1 RMS ({fom_rms:.4f})")
    ax.axhline(fom_mean - fom_rms, color="#fdae61", lw=1.2, ls=":")

    ax.set_xlabel(r"$E_{\rm true}$ (physical units)", fontsize=13)
    ax.set_ylabel(r"$(E_{\rm reco} - E_{\rm true})\;/\;E_{\rm true}$", fontsize=13)
    ax.set_title("ATLAS TileCal -- Residual vs True Energy (test)", fontsize=12)
    ax.legend(fontsize=10, loc="upper right")
    plt.tight_layout()
    path2 = os.path.join(OUT_DIR, "plot2_residual_vs_energy.png")
    fig.savefig(path2, dpi=150)
    plt.close(fig)
    print(f"  Saved Plot 2 -> {path2}")

    # ------------------------------------------------------------------
    # BONUS -- Timing prediction
    #
    # The TileCal Optimal Filter estimates two quantities per BC: amplitude
    # and phase (time offset).  The phase estimator is also a linear
    # combination of the 7 samples, scaled by the amplitude:
    #   tau = sum(b_i * s_i) / A_reco
    # Here we learn a linear model to predict timing from the 7 samples.
    #
    # Ground-truth timing is derived from the centre-of-mass of the pulse
    # window (a physics-motivated proxy): CoM - 3 rounded to {-1, 0, +1}.
    # Time=0: peak at sample 3 (in-time)
    # Time=-1: peak shifted left (signal arrived one BC early)
    # Time=+1: peak shifted right (signal arrived one BC late)
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("BONUS: Timing prediction")
    print("=" * 65)

    # Use only events with visible pulse (energy above a modest threshold)
    TIME_ENERGY_THR = 50.0
    tr_mask = y_tr > TIME_ENERGY_THR
    te_mask = y_te > TIME_ENERGY_THR

    _, t_tr = timing_proxy(X_tr[tr_mask])
    _, t_te = timing_proxy(X_te[te_mask])

    # Logistic regression: 7 samples -> {-1, 0, +1}
    time_clf = LogisticRegression(max_iter=500, solver="lbfgs", C=1.0)
    time_clf.fit(X_tr[tr_mask], t_tr)

    t_pred_te = time_clf.predict(X_te[te_mask])
    acc = accuracy_score(t_te, t_pred_te)

    print(f"\n  Training events (E > {TIME_ENERGY_THR}): {tr_mask.sum():,}")
    print(f"  Test events     (E > {TIME_ENERGY_THR}): {te_mask.sum():,}")
    print(f"\n  Timing classifier accuracy on test set: {acc:.4f}")
    print("\n  Classification report:")
    print(classification_report(t_te, t_pred_te,
                                target_names=["t=-1", "t=0", "t=+1"],
                                digits=4))

    # Timing histogram
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (t_data, label) in zip(axes, [(t_te, "True (proxy)"),
                                           (t_pred_te, "Predicted")]):
        counts = [np.sum(t_data == v) for v in [-1, 0, 1]]
        ax.bar([-1, 0, 1], counts, color="#4393c3", edgecolor="k", width=0.5)
        ax.set_xticks([-1, 0, 1])
        ax.set_xticklabels(["t=-1", "t=0", "t=+1"])
        ax.set_xlabel("Timing offset (BCs)", fontsize=12)
        ax.set_ylabel("Events", fontsize=12)
        ax.set_title(f"Timing distribution -- {label}", fontsize=11)

    axes[0].set_title(f"True timing proxy (test, E>{TIME_ENERGY_THR})", fontsize=11)
    axes[1].set_title(f"Predicted timing (accuracy={acc:.3f})", fontsize=11)
    plt.tight_layout()
    path3 = os.path.join(OUT_DIR, "plot3_timing.png")
    fig.savefig(path3, dpi=150)
    plt.close(fig)
    print(f"\n  Saved Plot 3 -> {path3}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"  Algorithm   : LinearRegression (7-sample Optimal Filter)")
    print(f"  Training    : {X_tr.shape[0]:,} events  ({len(glob.glob(os.path.join(DATA_DIR,'train','*.pt')))} shards)")
    print(f"  Test        : {X_te.shape[0]:,} events  ({len(glob.glob(os.path.join(DATA_DIR,'test','*.pt')))} shards)")
    print(f"  Filtered    : |E_true| > {ENERGY_THRESHOLD} -> {mask.sum():,} events used for FoM")
    print(f"")
    print(f"  Mean  (E_reco-E_true)/E_true = {fom_mean:+.6f}")
    print(f"  RMS   (E_reco-E_true)/E_true = {fom_rms:.6f}")
    print(f"  Validation R^2               = {val_r2:.6f}")
    print(f"")
    print(f"  Filter coefficients (s[-3]..s[+3]):")
    print("  " + "  ".join(f"{w:+.4f}" for w in model.coef_))
    print(f"  Bias: {model.intercept_:+.4f}")
    print("=" * 65)

    return {
        "model": model,
        "fom_mean": fom_mean,
        "fom_rms": fom_rms,
        "val_r2": val_r2,
        "mask": mask,
        "y_true_filtered": y_true_f,
        "y_pred_filtered": y_pred_f,
        "residual": residual,
        "timing_accuracy": acc,
        "time_clf": time_clf,
        "coef": model.coef_,
        "bias": model.intercept_,
    }


if __name__ == "__main__":
    results = main()
