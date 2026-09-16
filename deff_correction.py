#!/usr/bin/env python3
"""
deff_correction.py

Reproduces the design-effect (DEFF) correction reported in the manuscript
(Section 2.10 / Section 3.3 / Supplementary Table S3) from the pipeline's
own raw output files. This script does NOT re-run training or inference --
it takes the already-computed held-out bootstrap CIs and the Holm-adjusted
DeLong p-values as input, and applies a conservative correction for
possible within-cow clustering that could not be verified (Section 2.3).

WHY THIS CORRECTION EXISTS
    A verified per-image cow identifier could not be reconstructed for
    this archive, so the train/validation/test partitioning and five-fold
    cross-validation cannot be certified as cow-disjoint: pairs of images
    from the same cow (nominally, left- and right-side captures) may be
    treated as independent observations by the naive bootstrap and DeLong
    tests. Rather than presenting the naive intervals/p-values as if this
    were not a concern, we widen intervals and deflate test statistics
    under a design-effect model:

        DEFF = 1 + (m - 1) * ICC

    with the known fixed cluster size m = 2 (each cow contributes exactly
    two images) and a sensitivity range of intraclass correlation
    ICC in {0.3, 0.5, 0.7}, representative of paired anatomical thermal
    measurements. ICC = 0.7 is the most conservative value in this range
    and is what the manuscript's main text and Supplementary Table S3
    report as the headline conservative estimate; ICC = 0.3 and 0.5 are
    included here for the full sensitivity analysis.

INPUTS (produced earlier in the pipeline; included in this repo/example)
    - held_out_bootstrap_CIs_TableS1.csv
        columns: Model, Accuracy, AUC, Sensitivity, Specificity, Precision, F1
        each cell formatted as "point (lo-hi)" from the naive (unclustered)
        bootstrap over the held-out test set.
    - significance_holm.csv
        columns: comparison, dAUC, DeLong_p_raw, DeLong_p_Holm, sig_Holm,
                 ct_p_raw, ct_p_Holm
        Holm-Bonferroni-adjusted DeLong and corrected-resampled-t-test
        p-values for the six pre-specified DenseNet-121-vs-other
        comparisons (Section 2.10).

OUTPUTS
    - TableS3_DEFF_adjusted_CIs.csv   (one row per model, one column per
      metric per ICC level -- this is Supplementary Table S3)
    - DeLong_DEFF_adjusted_pvalues.csv (DEFF-deflated DeLong p-values for
      each pre-specified comparison, at each ICC level, with a robustness
      verdict across the whole sensitivity range)

USAGE
    python deff_correction.py \
        --bootstrap-csv held_out_bootstrap_CIs_TableS1.csv \
        --holm-csv significance_holm.csv \
        --out-dir ./deff_outputs
"""

import argparse
import re
import math
import os
import csv
from scipy.stats import norm


CLUSTER_SIZE_M = 2                 # each cow contributes exactly 2 images (left/right)
ICC_SENSITIVITY_RANGE = [0.3, 0.5, 0.7]
HEADLINE_ICC = 0.7                 # most conservative value; used in main-text claims


def deff(icc, m=CLUSTER_SIZE_M):
    """Design effect: DEFF = 1 + (m - 1) * ICC."""
    return 1.0 + (m - 1) * icc


def parse_point_ci(cell):
    """
    Parses a "0.822 (0.760-0.884)" style cell (also tolerates an en-dash)
    into (point, lo, hi) floats.
    """
    m = re.match(r"\s*([0-9.]+)\s*\(\s*([0-9.]+)\s*[-–—]\s*([0-9.]+)\s*\)\s*", cell)
    if not m:
        raise ValueError(f"Could not parse point/CI cell: {cell!r}")
    return float(m.group(1)), float(m.group(2)), float(m.group(3))


def widen_ci(point, lo, hi, icc):
    """
    Proportionally widens an asymmetric CI by sqrt(DEFF), preserving the
    original asymmetry around the point estimate rather than assuming a
    symmetric normal interval:

        new_lo = point - (point - lo) * sqrt(DEFF)
        new_hi = point + (hi   - point) * sqrt(DEFF)

    Clipped to [0, 1] since every metric here is a bounded proportion/AUC.
    """
    factor = math.sqrt(deff(icc))
    new_lo = point - (point - lo) * factor
    new_hi = point + (hi - point) * factor
    return max(0.0, new_lo), min(1.0, new_hi)


def deflate_pvalue(p_naive, icc, two_sided=True):
    """
    Converts a naive p-value to a z-statistic, deflates the z-statistic by
    sqrt(DEFF) (the standard design-effect adjustment for a test
    statistic under clustering), and converts back to a p-value.

    p_naive == 0.0 is treated as a very small floor (< 1e-16) so norm.isf
    doesn't blow up; report these as "<0.001" downstream regardless of the
    exact adjusted value, since the naive value itself was already at
    floating-point precision limits.
    """
    p_floor = max(p_naive, 1e-16)
    z_naive = norm.isf(p_floor / 2) if two_sided else norm.isf(p_floor)
    z_adj = z_naive / math.sqrt(deff(icc))
    p_adj = 2 * norm.sf(abs(z_adj)) if two_sided else norm.sf(z_adj)
    return min(1.0, p_adj)


def load_bootstrap_table(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        metric_cols = [c for c in reader.fieldnames if c != "Model"]
        for row in reader:
            parsed = {"Model": row["Model"]}
            for col in metric_cols:
                parsed[col] = parse_point_ci(row[col])
            rows.append(parsed)
    return rows, metric_cols


def build_deff_ci_table(bootstrap_rows, metric_cols, icc_levels):
    out_rows = []
    for row in bootstrap_rows:
        out = {"Model": row["Model"]}
        for col in metric_cols:
            point, lo, hi = row[col]
            for icc in icc_levels:
                new_lo, new_hi = widen_ci(point, lo, hi, icc)
                out[f"{col}_ICC{icc:.1f}"] = f"{point:.3f} ({new_lo:.3f}-{new_hi:.3f})"
        out_rows.append(out)
    return out_rows


def load_holm_comparisons(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "comparison": row["comparison"],
                "dAUC": float(row["dAUC"]),
                "DeLong_p_Holm": float(row["DeLong_p_Holm"]),
            })
    return rows


def build_deff_pvalue_table(holm_rows, icc_levels):
    out_rows = []
    for row in holm_rows:
        out = {
            "comparison": row["comparison"],
            "dAUC": row["dAUC"],
            "DeLong_p_Holm_naive": row["DeLong_p_Holm"],
        }
        still_significant_all_icc = True
        for icc in icc_levels:
            p_adj = deflate_pvalue(row["DeLong_p_Holm"], icc)
            out[f"DeLong_p_Holm_DEFF_ICC{icc:.1f}"] = round(p_adj, 6)
            if p_adj >= 0.05:
                still_significant_all_icc = False
        out["robust_across_full_ICC_range"] = still_significant_all_icc
        out_rows.append(out)
    return out_rows


def write_csv(rows, path):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bootstrap-csv", required=True, help="held_out_bootstrap_CIs_TableS1.csv")
    ap.add_argument("--holm-csv", required=True, help="significance_holm.csv")
    ap.add_argument("--out-dir", default="./deff_outputs")
    ap.add_argument("--icc", type=float, nargs="*", default=ICC_SENSITIVITY_RANGE,
                     help="ICC sensitivity range (default: 0.3 0.5 0.7)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    bootstrap_rows, metric_cols = load_bootstrap_table(args.bootstrap_csv)
    ci_table = build_deff_ci_table(bootstrap_rows, metric_cols, args.icc)
    write_csv(ci_table, os.path.join(args.out_dir, "TableS3_DEFF_adjusted_CIs.csv"))

    holm_rows = load_holm_comparisons(args.holm_csv)
    pvalue_table = build_deff_pvalue_table(holm_rows, args.icc)
    write_csv(pvalue_table, os.path.join(args.out_dir, "DeLong_DEFF_adjusted_pvalues.csv"))

    print(f"\nHeadline (ICC = {HEADLINE_ICC}) robustness summary:")
    for row in pvalue_table:
        headline_col = f"DeLong_p_Holm_DEFF_ICC{HEADLINE_ICC:.1f}"
        verdict = "robust" if row[headline_col] < 0.05 else "NOT robust to DEFF"
        print(f"  {row['comparison']:30s} naive Holm p={row['DeLong_p_Holm_naive']:.4f}"
              f"  ->  ICC=0.7 p={row[headline_col]:.4f}  [{verdict}]")


if __name__ == "__main__":
    main()
