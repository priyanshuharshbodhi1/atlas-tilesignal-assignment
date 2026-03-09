"""
ATLAS TileCal - Linear Energy Reconstruction
GSoC 2026 Evaluation Assignment
Author: Priyanshu

The TileCal reads ADC samples every 25 ns (one Bunch Crossing, BC).
Energy cannot be read directly from a single sample -- it has to be
inferred from the pulse shape across 7 consecutive samples. A weighted
sum of those 7 samples is enough because the pulse shape is fixed and
well-known. This is the "Optimal Filter" approach used in ATLAS calorimetry.
Training a LinearRegression on (7 samples -> energy) learns these filter
weights from data automatically.

References:
  E. Fullana, "Optimal Filtering in the ATLAS Hadronic Tile Calorimeter"
  D. Oliveira, "Energy reconstruction of ATLAS TileCal using the Wiener filter"
  F. Curcio, "Machine Learning-Based Energy Reconstruction for ATLAS TileCal at HL-LHC"
"""

import glob
import os
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

# Skip BCs where the true energy is near zero -- the relative residual
# (pred - true)/true diverges there and doesn't reflect filter quality.
ENERGY_THRESHOLD = 10.0


def load_split(split: str):
    """Load and concatenate all .pt shards for a given split."""
    pattern = os.path.join(DATA_DIR, split, f"{split}_*.pt")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No shards found at {pattern}")

    X_list, y_list, yOF_list = [], [], []
    for f in files:
        shard = torch.load(f, weights_only=False)
        X_list.append(shard["X"].numpy())
        y_list.append(shard["y"].numpy())
        yOF_list.append(shard["y_OF"].numpy())

    X   = np.concatenate(X_list,   axis=0)
    y   = np.concatenate(y_list,   axis=0)
    yOF = np.concatenate(yOF_list, axis=0)

    # Channel index 1 is the lo-gain channel
    X_lo   = X[:, 1, :]
    y_lo   = y[:, 1]
    yOF_lo = yOF[:, 1]

    print(f"  [{split}] {len(files)} shards -> {X_lo.shape[0]:,} events")
    return X_lo, y_lo, yOF_lo


def load_y_stats():
    mean = np.load(os.path.join(DATA_DIR, "y_stats", "mean.npy"))
    std  = np.load(os.path.join(DATA_DIR, "y_stats", "std.npy"))
    return float(mean[0, 1]), float(std[0, 1])


def denormalise(y_norm, mean, std):
    return y_norm * std + mean


def timing_proxy(X_lo):
    """Estimate pulse timing from the centre-of-mass of the 7-sample window.

    For an in-time pulse the peak sits at sample index 3. If the pulse
    arrived one BC early the window shifts left; one BC late it shifts right.
    Centre-of-mass gives a continuous estimate; we round to {-1, 0, +1}.
    """
    indices  = np.arange(7, dtype=float)
    row_sum  = X_lo.sum(axis=1)
    safe     = row_sum > 1e-6

    t_cont = np.zeros(len(X_lo))
    t_cont[safe] = (
        (X_lo[safe] * indices).sum(axis=1) / row_sum[safe] - 3.0
    )
    t_disc = np.clip(np.round(t_cont).astype(int), -1, 1)
    return t_cont, t_disc


def main():
    print("=" * 65)
    print("ATLAS TileCal - Linear Energy Reconstruction (Optimal Filter)")
    print("=" * 65)

    mean_lo, std_lo = load_y_stats()
    print(f"\nEnergy normalisation  mean={mean_lo:.4f}  std={std_lo:.4f}\n")

    print("Loading data ...")
    X_tr, y_tr_n, _ = load_split("train")
    X_va, y_va_n, _ = load_split("val")
    X_te, y_te_n, _ = load_split("test")

    y_tr = denormalise(y_tr_n, mean_lo, std_lo)
    y_va = denormalise(y_va_n, mean_lo, std_lo)
    y_te = denormalise(y_te_n, mean_lo, std_lo)

    print("\nFitting LinearRegression on training data ...")
    model = LinearRegression()
    model.fit(X_tr, y_tr)

    val_r2 = model.score(X_va, y_va)
    print(f"  Validation R^2 = {val_r2:.6f}")

    print("\nEvaluating on test set ...")
    y_pred_te = model.predict(X_te)

    mask     = np.abs(y_te) > ENERGY_THRESHOLD
    y_true_f = y_te[mask]
    y_pred_f = y_pred_te[mask]
    residual = (y_pred_f - y_true_f) / y_true_f

    fom_mean = float(np.mean(residual))
    fom_rms  = float(np.std(residual))

    print(f"\n  Events passing |E_true| > {ENERGY_THRESHOLD}: {mask.sum():,} / {len(y_te):,}")
    print(f"\n  Figure of Merit (test set):")
    print(f"    Mean  [ (E_reco - E_true) / E_true ] = {fom_mean:+.6f}")
    print(f"    RMS   [ (E_reco - E_true) / E_true ] = {fom_rms:.6f}")

    offsets = [-3, -2, -1, 0, +1, +2, +3]
    print("\n  Learned Optimal Filter coefficients:")
    print(f"  {'Sample offset':>15}  {'Weight':>12}")
    print("  " + "-" * 30)
    for off, w in zip(offsets, model.coef_):
        bar = "#" * int(abs(w) / max(abs(model.coef_)) * 20)
        print(f"  {'s[' + str(off) + ']':>15}  {w:+12.4f}  {bar}")
    print(f"  {'bias':>15}  {model.intercept_:+12.4f}")

    # Plot 1: histogram of relative residual
    fig, ax = plt.subplots(figsize=(8, 5.5))
    clip     = 1.5
    res_plot = np.clip(residual, -clip, clip)

    n, _, _ = ax.hist(
        res_plot, bins=150, range=(-clip, clip),
        color="#2166ac", alpha=0.85, edgecolor="none",
        label="Linear filter (test)"
    )
    ax.axvline(fom_mean, color="#d7191c", lw=1.8, ls="--",
               label=f"Mean = {fom_mean:+.4f}")
    ax.axvline(fom_mean - fom_rms, color="#fdae61", lw=1.2, ls=":",
               label=f"RMS  = {fom_rms:.4f}")
    ax.axvline(fom_mean + fom_rms, color="#fdae61", lw=1.2, ls=":")
    ax.text(0.97, 0.95, f"Mean = {fom_mean:+.4f}\nRMS  = {fom_rms:.4f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.8))
    ax.set_xlabel(r"$(E_{\rm reco} - E_{\rm true})\;/\;E_{\rm true}$", fontsize=13)
    ax.set_ylabel("Events / bin", fontsize=13)
    ax.set_title("ATLAS TileCal - Lo-gain Energy Residual Distribution (test)", fontsize=12)
    ax.legend(fontsize=11)
    ax.set_xlim(-clip, clip)
    plt.tight_layout()
    path1 = os.path.join(OUT_DIR, "plot1_residual_histogram.png")
    fig.savefig(path1, dpi=150)
    plt.close(fig)
    print(f"\n  Saved Plot 1 -> {path1}")

    # Plot 2: 2D hexbin, residual vs true energy
    fig, ax = plt.subplots(figsize=(9, 6))
    hb = ax.hexbin(
        y_true_f, np.clip(residual, -clip, clip),
        gridsize=80, cmap="viridis",
        mincnt=1, bins="log",
        extent=[y_true_f.min(), min(y_true_f.max(), 2000), -clip, clip]
    )
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("log$_{10}$(counts)", fontsize=11)
    ax.axhline(0,        color="white",   lw=1.2, ls="--", alpha=0.7)
    ax.axhline(fom_mean, color="#d7191c", lw=1.5, ls="--",
               label=f"Mean = {fom_mean:+.4f}")
    ax.axhline(fom_mean + fom_rms, color="#fdae61", lw=1.2, ls=":",
               label=f"+/-1 RMS ({fom_rms:.4f})")
    ax.axhline(fom_mean - fom_rms, color="#fdae61", lw=1.2, ls=":")
    ax.set_xlabel(r"$E_{\rm true}$ (physical units)", fontsize=13)
    ax.set_ylabel(r"$(E_{\rm reco} - E_{\rm true})\;/\;E_{\rm true}$", fontsize=13)
    ax.set_title("ATLAS TileCal - Residual vs True Energy (test)", fontsize=12)
    ax.legend(fontsize=10, loc="upper right")
    plt.tight_layout()
    path2 = os.path.join(OUT_DIR, "plot2_residual_vs_energy.png")
    fig.savefig(path2, dpi=150)
    plt.close(fig)
    print(f"  Saved Plot 2 -> {path2}")

    # BONUS: timing prediction
    # The dataset does not include explicit timing labels, so we derive them
    # from the centre-of-mass of the 7-sample window (a physics proxy).
    # We then train a logistic regression to predict {-1, 0, +1} from the
    # same 7 input samples.
    print("\n" + "=" * 65)
    print("BONUS: Timing prediction")
    print("=" * 65)

    TIME_ENERGY_THR = 50.0
    tr_mask = y_tr > TIME_ENERGY_THR
    te_mask = y_te > TIME_ENERGY_THR

    _, t_tr = timing_proxy(X_tr[tr_mask])
    _, t_te = timing_proxy(X_te[te_mask])

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

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (t_data, label) in zip(axes, [(t_te, "True (proxy)"),
                                           (t_pred_te, "Predicted")]):
        counts = [np.sum(t_data == v) for v in [-1, 0, 1]]
        ax.bar([-1, 0, 1], counts, color="#4393c3", edgecolor="k", width=0.5)
        ax.set_xticks([-1, 0, 1])
        ax.set_xticklabels(["t=-1", "t=0", "t=+1"])
        ax.set_xlabel("Timing offset (BCs)", fontsize=12)
        ax.set_ylabel("Events", fontsize=12)

    axes[0].set_title(f"True timing proxy (test, E>{TIME_ENERGY_THR})", fontsize=11)
    axes[1].set_title(f"Predicted timing (accuracy={acc:.3f})", fontsize=11)
    plt.tight_layout()
    path3 = os.path.join(OUT_DIR, "plot3_timing.png")
    fig.savefig(path3, dpi=150)
    plt.close(fig)
    print(f"\n  Saved Plot 3 -> {path3}")

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
        "residual": residual,
        "y_true_filtered": y_true_f,
        "y_pred_filtered": y_pred_f,
        "timing_accuracy": acc,
    }


if __name__ == "__main__":
    results = main()
