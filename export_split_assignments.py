# =============================================================================
# export_split_assignments.py
#
# HONESTY NOTE: this cell cannot be run by anyone except you, in your own
# Colab notebook. The actual train/validation/test/fold assignments only
# exist as variables already sitting in your notebook's memory (from
# Section 2.3 / 2.7) -- they were never uploaded anywhere I can reach, so
# I cannot generate this file myself. What I *can* do is give you the
# exact code to export whatever split variables you already have to a
# clean, de-identified CSV, ready to drop into the GitHub repo.
#
# HOW TO USE
#   1. Paste this cell into your main Colab notebook, AFTER the cell(s)
#      that define your train/val/test split and your 5-fold CV splits
#      (Section 2.3 / 2.7), so the variables referenced below already
#      exist in memory.
#   2. Edit the "ADJUST THESE" block so the variable names match what
#      your notebook actually calls them.
#   3. Run it. It writes split_assignments.csv to ROI_OUT_DIR (or wherever
#      you point OUT_PATH) with one row per image, containing:
#        - a de-identified image ID (NOT the raw filename, to avoid
#          exposing any farm/owner-linked information -- see note below)
#        - which held-out split it fell into (train / val / test)
#        - which of the 5 CV folds it fell into
#      This directly answers the reviewer's request in comment 8 for
#      "de-identified cow-level split assignments."
#
# ON DE-IDENTIFICATION
#   Your raw filenames (e.g. 25050241.BMP) look like they may encode an
#   acquisition batch/date pattern rather than a real cow ID (this is the
#   same fact that made true cow-level grouping unrecoverable in the
#   first place -- see Section 2.3). To be safe, this script hashes each
#   filename to a short, stable, non-reversible ID instead of publishing
#   the raw filename. If your filenames are already just arbitrary IDs
#   with no embedded date/farm information, you can set HASH_FILENAMES =
#   False below to publish them as-is -- but if in doubt, leave hashing on.
# =============================================================================

import os
import hashlib
import csv

# ---- ADJUST THESE to match your notebook's actual variable names ----
# `all_images_df` (or equivalent): a DataFrame/list with one row per image,
# containing at least a filename/path and a label. If you built this in
# Section 2.3, reuse that object directly instead of re-deriving it.
#
# `train_idx`, `val_idx`, `test_idx`: whatever index/boolean arrays (or
# filename lists) mark which images went into each held-out split
# (Section 2.3 / 3.1).
#
# `cv_fold_assignments`: something that gives you, for every image, which
# of the 5 outer folds it belonged to as TEST (Section 2.7 / 3.2) -- e.g.
# a dict {filename: fold_index} or a column already sitting in a
# per-image results DataFrame you built for Section 3.2 / Table 5.
#
# If your notebook already has a single DataFrame with columns like
# ["fname", "label", "split", "cv_fold"], skip straight to the WRITE
# block below and just point SOURCE_DF at it.

SOURCE_DF = None          # e.g. all_images_df, if it already has split/cv_fold columns
FNAME_COL = "fname"
LABEL_COL = "label"
SPLIT_COL = "split"       # expected values: "train" / "val" / "test"
CV_FOLD_COL = "cv_fold"   # expected values: 0-4, or -1/NaN if not in CV test role that fold

HASH_FILENAMES = True
OUT_PATH = os.path.join(ROI_OUT_DIR if "ROI_OUT_DIR" in dir() else ".", "split_assignments.csv")


def deidentify(fname, salt="mastitis-archive-v1"):
    """Stable, non-reversible short ID so raw filenames are not published."""
    h = hashlib.sha256((salt + fname).encode()).hexdigest()
    return "img_" + h[:12]


def export(df, out_path):
    rows = []
    for _, row in df.iterrows():
        fname = row[FNAME_COL]
        rows.append({
            "image_id": deidentify(fname) if HASH_FILENAMES else fname,
            "label": row[LABEL_COL],
            "held_out_split": row.get(SPLIT_COL, ""),
            "cv_test_fold": row.get(CV_FOLD_COL, ""),
        })
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "label", "held_out_split", "cv_test_fold"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_path}")
    print("Copy this file into your GitHub repo (e.g. data/split_assignments.csv)")
    print("and reference it from the Data Availability Statement.")


if SOURCE_DF is not None:
    export(SOURCE_DF, OUT_PATH)
else:
    print("SOURCE_DF is None -- point it at the DataFrame in your notebook that")
    print("already has one row per image with filename/label/split/fold columns,")
    print("or build one from your existing train_idx/val_idx/test_idx and")
    print("cv_fold_assignments variables before calling export().")
