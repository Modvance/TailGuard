"""Fuse TailGuard coverage and reconciled image scores.

The full training run writes the reconciled score table. The coverage replay
writes a second table for the same test images. This module validates sample
identity, averages the two final image scores, and reports macro I-AUROC.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "typeDict"):
    np.typeDict = np.sctypeDict

from sklearn.metrics import roc_auc_score


KEY_COLUMNS = ("class_name", "test_suffix", "label")


def normalized_test_suffix(path):
    normalized = str(path).replace("\\", "/")
    marker = "/test/"
    if marker not in normalized:
        raise ValueError("test image path has no /test/ suffix: {}".format(path))
    return normalized.split(marker, 1)[1]


def keyed_scores(path, prefix):
    frame = pd.read_csv(path)
    required = {"class_name", "img_path", "label", "final_score"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError("{} is missing columns: {}".format(path, sorted(missing)))
    frame = frame.copy()
    frame["class_name"] = frame["class_name"].astype(str)
    frame["test_suffix"] = frame["img_path"].map(normalized_test_suffix)
    frame["label"] = frame["label"].astype(int)
    if frame.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("{} contains duplicate test identities".format(path))
    return frame[list(KEY_COLUMNS) + ["img_path", "final_score"]].rename(
        columns={"img_path": "{}_img_path".format(prefix), "final_score": "{}_score".format(prefix)}
    )


def load_class_roles(path):
    if path is None:
        return None
    frame = pd.read_csv(path)
    required = {"class_name", "is_gt_tail"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError("{} is missing columns: {}".format(path, sorted(missing)))
    counts = frame.groupby("class_name")["is_gt_tail"].nunique(dropna=False)
    if not (counts == 1).all():
        raise ValueError("class role file contains inconsistent labels")
    return {
        str(class_name): bool(int(group["is_gt_tail"].iloc[0]))
        for class_name, group in frame.groupby("class_name")
    }


def fuse(coverage_path, reconciled_path, output_dir, class_roles_path=None):
    coverage = keyed_scores(coverage_path, "coverage")
    reconciled = keyed_scores(reconciled_path, "reconciled")
    merged = coverage.merge(
        reconciled,
        on=list(KEY_COLUMNS),
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not (merged["_merge"] == "both").all():
        counts = merged["_merge"].value_counts().to_dict()
        raise ValueError("coverage and reconciled score identities differ: {}".format(counts))
    merged = merged.drop(columns=["_merge"])
    merged["dual_score"] = 0.5 * (
        merged["coverage_score"].astype(float) + merged["reconciled_score"].astype(float)
    )

    roles = load_class_roles(class_roles_path)
    rows = []
    for class_name, group in merged.groupby("class_name", sort=True):
        labels = group["label"].to_numpy(dtype=int)
        if set(labels.tolist()) != {0, 1}:
            raise ValueError("class {} does not contain both test labels".format(class_name))
        row = {
            "class_name": class_name,
            "I-AUROC": float(roc_auc_score(labels, group["dual_score"].to_numpy(dtype=float))),
        }
        if roles is not None:
            if class_name not in roles:
                raise ValueError("class role is missing for {}".format(class_name))
            row["role"] = "tail" if roles[class_name] else "head"
        rows.append(row)
    per_class = pd.DataFrame(rows)

    summary = {
        "fusion_rule": "0.5 * (coverage_score + reconciled_score)",
        "num_test_images": int(len(merged)),
        "num_classes": int(len(per_class)),
        "all_I-AUROC": float(per_class["I-AUROC"].mean()),
    }
    if roles is not None:
        for role in ("head", "tail"):
            values = per_class.loc[per_class["role"] == role, "I-AUROC"]
            summary["{}_I-AUROC".format(role)] = (
                None if values.empty else float(values.mean())
            )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    merged.to_csv(output / "dual_scores.csv", index=False)
    per_class.to_csv(output / "dual_per_class.csv", index=False)
    with (output / "dual_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    return summary
