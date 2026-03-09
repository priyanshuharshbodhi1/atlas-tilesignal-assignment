# ATLAS TileCal Linear Energy Reconstruction

GSoC 2026 Evaluation Assignment — HSF/ATLAS: AI-Accelerated Reconstruction for the ATLAS Tile Calorimeter at the HL-LHC

## What this is

The ATLAS Tile Calorimeter samples the detector signal every 25 ns (one Bunch Crossing, BC). Each sample is a digital ADC count (`sample_lo`). The actual deposited energy (`ene_lo`) cannot be read directly — it has to be inferred from the characteristic shape of the pulse across 7 consecutive samples.

This repository implements a **linear energy reconstruction** algorithm that learns to predict `ene_lo[n]` from the window `sample_lo[n-3], ..., sample_lo[n+3]` using linear regression. The learned weights are the amplitude filter coefficients of the classical **Optimal Filter** used in ATLAS calorimetry (Fullana et al.), which is equivalent to a Wiener filter matched to the known pulse shape.

## Results (test set)

| Metric | Value |
|---|---|
| Mean (E_reco - E_true) / E_true | -0.0416 |
| RMS  (E_reco - E_true) / E_true |  1.3963 |
| Validation R² | 0.9584 |
| Bonus — timing accuracy | 93.5% |

A mean near zero means the filter is unbiased. The RMS reflects intrinsic difficulty at low energies where pedestal noise is comparable to the signal.

## Files

| File | Description |
|---|---|
| `linear_filter.py` | Main script: loads data, trains model, evaluates, produces plots |
| `generate_report.py` | Builds the PDF report from plots and results |
| `plot1_residual_histogram.png` | Histogram of (E_reco - E_true)/E_true |
| `plot2_residual_vs_energy.png` | 2D hexbin of residual vs true energy |
| `plot3_timing.png` | Bonus: timing prediction distribution |
| `report_atlas_tilecal.pdf` | Final PDF report |

## Data

Download the data from the CernBox public link and place it as:

```
data/
  train/   train_00000.pt ... train_00260.pt
  val/     val_00000.pt   ... val_00055.pt
  test/    test_00000.pt  ... test_00055.pt
  y_stats/ mean.npy  std.npy
```

Each `.pt` shard is a dict with keys:
- `X`: float32 `[N, 2, 7]` — 7-sample windows for hi/lo gain channels
- `y`: float32 `[N, 2]` — normalised true energies (hi/lo)
- `y_OF`: float32 `[N, 2]` — pre-computed Optimal Filter estimates

## Setup and run

```bash
uv venv .venv
uv pip install --python .venv/bin/python \
    torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python \
    scikit-learn pandas matplotlib numpy reportlab

.venv/bin/python linear_filter.py
.venv/bin/python generate_report.py
```

## Physics context

The key insight is that the TileCal pulse shape is stable and well-defined. For a given deposited energy, the 7-sample pattern is predictable up to noise. The linear regression finds the coefficient vector `w` that minimises the mean squared reconstruction error, which is exactly what the Optimal Filter computes analytically using the noise autocorrelation matrix and the reference pulse shape. The data-driven approach learns these coefficients without requiring an explicit pulse model.

The central sample `s[0]` gets the largest positive weight (it is at the pulse peak for an in-time signal). Neighbouring samples get negative weights that cancel pile-up contributions from adjacent BCs.

For the **timing bonus**: the same 7 samples carry phase information. A logistic regression classifier on the sample window recovers the timing offset {-1, 0, +1} with 93.5% accuracy for high-energy events.

## References

- E. Fullana, *Optimal Filtering in the ATLAS Hadronic Tile Calorimeter*, ATL-TILECAL-PUB-2005-001
- D. Oliveira, *Energy reconstruction of ATLAS TileCal under high pile-up conditions using the Wiener filter*
- F. Curcio, *Machine Learning-Based Energy Reconstruction for ATLAS TileCal at HL-LHC*
