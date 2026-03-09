"""
Generate the PDF report for the ATLAS TileCal linear reconstruction assignment.
"""

import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    HRFlowable, PageBreak
)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

DIR = os.path.dirname(__file__)

FOM_MEAN   =  -0.041601
FOM_RMS    =   1.396269
VAL_R2     =   0.958359
TIMING_ACC =   0.9348
N_TRAIN    =  534337
N_TEST     =  114502
N_FILTERED =    7271
ENERGY_THR =  10.0
COEF       = [+2214.7488, -1435.2515, -1024.1498, +5244.2261,
              -2092.9458,  +784.2809,   -227.1638]
BIAS       =  -0.6323


def build_styles():
    base = getSampleStyleSheet()
    return {
        "title":   ParagraphStyle("title",   parent=base["Title"],
                                   fontSize=18, spaceAfter=6,
                                   textColor=colors.HexColor("#1a3a5c")),
        "sub":     ParagraphStyle("sub",     parent=base["Normal"],
                                   fontSize=11, spaceAfter=10,
                                   textColor=colors.HexColor("#2c6e9c"),
                                   alignment=TA_CENTER),
        "h2":      ParagraphStyle("h2",      parent=base["Heading2"],
                                   fontSize=13, spaceBefore=14, spaceAfter=4,
                                   textColor=colors.HexColor("#1a3a5c")),
        "body":    ParagraphStyle("body",    parent=base["Normal"],
                                   fontSize=10.5, leading=15,
                                   alignment=TA_JUSTIFY, spaceAfter=6),
        "caption": ParagraphStyle("caption", parent=base["Normal"],
                                   fontSize=9, textColor=colors.grey,
                                   alignment=TA_CENTER, spaceAfter=8),
        "small":   ParagraphStyle("small",   parent=base["Normal"],
                                   fontSize=9.5, leading=14, spaceAfter=4),
    }


def make_pdf(out_path):
    doc = SimpleDocTemplate(
        out_path, pagesize=A4,
        leftMargin=2.2*cm, rightMargin=2.2*cm,
        topMargin=2.5*cm,  bottomMargin=2.5*cm,
    )
    S = build_styles()
    story = []

    story.append(Paragraph("ATLAS Tile Calorimeter", S["title"]))
    story.append(Paragraph("Linear Energy Reconstruction - Optimal Filter", S["sub"]))
    story.append(Paragraph(
        "GSoC 2026 Evaluation Assignment | "
        "HSF - ATLAS: AI-Accelerated Reconstruction for the HL-LHC",
        S["sub"]
    ))
    story.append(HRFlowable(width="100%", thickness=1.2,
                             color=colors.HexColor("#2c6e9c"), spaceAfter=10))

    # Section 1
    story.append(Paragraph("1. Approach", S["h2"]))
    story.append(Paragraph(
        "The ATLAS Tile Calorimeter samples signals every 25 ns, one sample per "
        "Bunch Crossing (BC). Each BC gives a digital ADC count called sample_lo. "
        "The actual deposited energy ene_lo cannot be read directly from a single "
        "sample because each reading also contains contributions from nearby pulses "
        "(pile-up) and a noisy pedestal baseline.",
        S["body"]
    ))
    story.append(Paragraph(
        "The key observation is that the TileCal pulse shape is fixed and known. "
        "So if you look at 7 consecutive samples around a BC of interest, the "
        "relative pattern of those samples encodes both the energy and any timing "
        "offset. A simple weighted sum of those 7 samples turns out to be enough "
        "to reconstruct the energy:",
        S["body"]
    ))
    story.append(Paragraph(
        "E_reco[n] = w[-3]*s[n-3] + w[-2]*s[n-2] + ... + w[+3]*s[n+3] + bias",
        ParagraphStyle("eq", parent=S["body"], alignment=TA_CENTER,
                       fontName="Courier", fontSize=10, spaceAfter=8)
    ))
    story.append(Paragraph(
        "This is exactly the Optimal Filter (OF) approach used in ATLAS calorimetry "
        "(Fullana et al.). Instead of deriving the filter weights analytically from "
        "the pulse shape and noise model, I fitted a scikit-learn LinearRegression "
        "to the training data and let it learn the weights directly. Both approaches "
        "end up doing the same thing: finding the best linear combination of the "
        "7 samples to estimate the energy.",
        S["body"]
    ))
    story.append(Paragraph(
        f"Training used {N_TRAIN:,} events from 261 shards. The lo-gain channel "
        "(X[:, 1, :]) was used as input and the denormalised energy (y[:, 1] * std + mean) "
        "as the target.",
        S["body"]
    ))

    # Section 2
    story.append(Paragraph("2. Results", S["h2"]))
    story.append(Paragraph(
        f"The figure of merit is (E_reco - E_true) / E_true, computed on "
        f"{N_TEST:,} test-set events. BCs where |E_true| is below {ENERGY_THR} are "
        f"excluded because the relative error becomes meaninglessly large for near-zero "
        f"signals. This leaves {N_FILTERED:,} events.",
        S["body"]
    ))

    data = [
        ["Metric", "Value"],
        ["Mean (E_reco - E_true) / E_true", f"{FOM_MEAN:+.6f}"],
        ["RMS  (E_reco - E_true) / E_true", f"{FOM_RMS:.6f}"],
        ["Validation R\u00b2",              f"{VAL_R2:.6f}"],
        ["Training events",                  f"{N_TRAIN:,}"],
        ["Test events (after filter)",       f"{N_FILTERED:,}"],
        ["Bonus: timing accuracy",           f"{TIMING_ACC:.4f}"],
    ]
    t = Table(data, colWidths=[10*cm, 5.5*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0), colors.HexColor("#1a3a5c")),
        ("TEXTCOLOR",      (0, 0), (-1, 0), colors.white),
        ("FONTNAME",       (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",       (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor("#eaf2fb"), colors.white]),
        ("ALIGN",          (1, 0), (-1, -1), "CENTER"),
        ("GRID",           (0, 0), (-1, -1), 0.4, colors.HexColor("#aaaaaa")),
        ("TOPPADDING",     (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 4),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph(
        f"The mean of {FOM_MEAN:+.4f} is very close to zero, which means the filter "
        "is not systematically over- or under-estimating energy. The RMS of "
        f"{FOM_RMS:.4f} is mainly driven by low-energy BCs where noise is comparable "
        "to the signal. The R\u00b2 of 0.958 on validation shows the model generalises well.",
        S["body"]
    ))

    # Section 3
    story.append(Paragraph("3. Residual Distribution", S["h2"]))
    p1 = os.path.join(DIR, "plot1_residual_histogram.png")
    if os.path.exists(p1):
        story.append(Image(p1, width=14*cm, height=9.5*cm))
        story.append(Paragraph(
            f"Figure 1: Distribution of (E_reco - E_true) / E_true on the test set "
            f"(|E_true| > {ENERGY_THR}). Red dashed line = mean. "
            "Orange dotted lines = +/- 1 RMS.",
            S["caption"]
        ))
    story.append(Paragraph(
        "The distribution is roughly Gaussian and centred near zero. The slight "
        f"negative mean ({FOM_MEAN:+.4f}) suggests the filter underestimates energy "
        "by a small amount on average, likely because pile-up from neighbouring BCs "
        "adds a positive bias to the input samples that the filter partially corrects for.",
        S["body"]
    ))

    # Section 4
    story.append(Paragraph("4. Residual vs True Energy", S["h2"]))
    p2 = os.path.join(DIR, "plot2_residual_vs_energy.png")
    if os.path.exists(p2):
        story.append(Image(p2, width=14*cm, height=9.5*cm))
        story.append(Paragraph(
            "Figure 2: 2D hexbin of (E_reco - E_true) / E_true vs E_true. "
            "Color encodes log10 of event count per cell.",
            S["caption"]
        ))
    story.append(Paragraph(
        "The residual does not show any strong trend with energy, which is a good "
        "sign -- the filter is not biased towards high or low energies. The wider "
        "spread at lower energies is expected since a fixed-size pedestal fluctuation "
        "has a bigger relative impact on a small signal.",
        S["body"]
    ))

    # Section 5
    story.append(Paragraph("5. Learned Filter Coefficients", S["h2"]))
    story.append(Paragraph(
        "The table below shows the 7 weights the model learned. "
        "The in-time sample s[0] has by far the largest positive weight, "
        "which makes sense since the pulse peak is at that position for a "
        "perfectly-timed BC. The surrounding samples get smaller weights "
        "that help cancel out contributions from adjacent-BC pile-up.",
        S["body"]
    ))

    offsets = ["s[-3]", "s[-2]", "s[-1]", "s[0]", "s[+1]", "s[+2]", "s[+3]"]
    cdata = [["Sample", "Coefficient"]] + \
            [[off, f"{w:+.4f}"] for off, w in zip(offsets, COEF)] + \
            [["bias", f"{BIAS:+.4f}"]]
    ct = Table(cdata, colWidths=[4*cm, 4*cm])
    ct.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0), colors.HexColor("#2c6e9c")),
        ("TEXTCOLOR",      (0, 0), (-1, 0), colors.white),
        ("FONTNAME",       (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",       (0, 0), (-1, -1), 10),
        ("ALIGN",          (1, 0), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor("#eaf2fb"), colors.white]),
        ("GRID",           (0, 0), (-1, -1), 0.4, colors.HexColor("#aaaaaa")),
        ("TOPPADDING",     (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 4),
    ]))
    story.append(ct)
    story.append(Spacer(1, 0.2*cm))

    # Section 6
    story.append(PageBreak())
    story.append(Paragraph("6. Bonus: Timing Prediction", S["h2"]))
    story.append(Paragraph(
        "Beyond energy, I also tried to predict the timing offset of the signal. "
        "Time=0 means the pulse is aligned with the current BC. "
        "Time=-1 means it came one BC early; Time=+1 means one BC late.",
        S["body"]
    ))
    story.append(Paragraph(
        "The dataset does not include explicit timing labels, so I derived a proxy "
        "from the data itself: the centre-of-mass of the 7-sample window. "
        "If the pulse peak sits at the centre (index 3) the timing is 0. "
        "A left-shifted peak means Time=-1 and a right-shifted peak means Time=+1. "
        "This proxy is only reliable for events with visible energy (E > 50), "
        "so only those events were used.",
        S["body"]
    ))
    story.append(Paragraph(
        "A logistic regression classifier (same 7-sample input) was then trained "
        "to predict these timing labels. On the test set it got "
        f"<b>{TIMING_ACC:.1%} accuracy</b>. The 7 samples carry enough shape "
        "information to distinguish the three timing cases most of the time.",
        S["body"]
    ))

    p3 = os.path.join(DIR, "plot3_timing.png")
    if os.path.exists(p3):
        story.append(Image(p3, width=14.5*cm, height=6.5*cm))
        story.append(Paragraph(
            f"Figure 3: True timing proxy (left) and predicted timing (right) "
            f"for test events with E > 50 (N=368). Accuracy = {TIMING_ACC:.4f}.",
            S["caption"]
        ))

    # Section 7
    story.append(Paragraph("7. Summary", S["h2"]))
    story.append(Paragraph(
        "The linear regression approach works well because the TileCal pulse shape "
        "is stable and predictable. Once the shape is fixed, recovering the amplitude "
        "is just a matter of finding the right weighted sum of samples, which is "
        "exactly what linear regression does.",
        S["body"]
    ))
    story.append(Paragraph(
        f"The achieved mean residual of {FOM_MEAN:+.4f} and R\u00b2 of {VAL_R2:.3f} "
        "show that a simple linear model is already quite effective. More complex "
        "models (neural networks etc.) would likely reduce the RMS further, especially "
        "at low energies where pile-up is hardest to separate, which is what the "
        "full GSoC project is about.",
        S["body"]
    ))

    story.append(HRFlowable(width="100%", thickness=0.8,
                             color=colors.HexColor("#aaaaaa"), spaceBefore=12))
    story.append(Paragraph(
        "References: "
        "(1) E. Fullana, <i>Optimal Filtering in the ATLAS Hadronic Tile Calorimeter</i>. "
        "(2) D. Oliveira, <i>Energy reconstruction of ATLAS TileCal under high pile-up "
        "conditions using the Wiener filter</i>. "
        "(3) F. Curcio, <i>Machine Learning-Based Energy Reconstruction for ATLAS TileCal "
        "at HL-LHC</i>.",
        ParagraphStyle("refs", parent=S["small"],
                       textColor=colors.HexColor("#555555"))
    ))

    doc.build(story)
    print(f"PDF saved -> {out_path}")


if __name__ == "__main__":
    make_pdf(os.path.join(DIR, "report_atlas_tilecal.pdf"))
