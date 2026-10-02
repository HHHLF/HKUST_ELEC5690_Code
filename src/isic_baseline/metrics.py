"""Test metrics. Undefined ratios stay null instead of being filled with zero."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve

_TRAPZ = getattr(np, "trapezoid", np.trapz)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


def safe_div(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, class_names: list[str]) -> dict:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    if y_prob.ndim != 2 or y_prob.shape[1] != len(class_names):
        raise ValueError(f"Expected probabilities of shape (N, {len(class_names)}), got {y_prob.shape}")
    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob length mismatch")
    if len(y_true) == 0:
        raise ValueError("Cannot score an empty split.")

    y_pred = y_prob.argmax(axis=1)
    labels = list(range(len(class_names)))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    per_class = []
    notes = []
    for index, name in enumerate(class_names):
        tp = int(matrix[index, index])
        support = int(matrix[index].sum())
        predicted = int(matrix[:, index].sum())
        precision = safe_div(tp, predicted)
        recall = safe_div(tp, support)
        if precision is None or recall is None:
            f1 = None
        else:
            f1 = safe_div(2 * precision * recall, precision + recall)
            if f1 is None:
                f1 = 0.0
        binary = (y_true == index).astype(np.int64)
        n_pos = int(binary.sum())
        n_neg = int(len(binary) - n_pos)
        auroc = None
        fpr = tpr = None
        if n_pos == 0 or n_neg == 0:
            notes.append(
                f"{name} AUROC is undefined because the split has {n_pos} positive and {n_neg} negative samples."
            )
        else:
            fpr_i, tpr_i, _ = roc_curve(binary, y_prob[:, index])
            auroc = float(_TRAPZ(tpr_i, fpr_i))
            fpr = fpr_i.tolist()
            tpr = tpr_i.tolist()
        if support == 0:
            notes.append(f"{name} recall/F1 are undefined because support is 0.")
        if predicted == 0:
            notes.append(f"{name} precision/F1 are undefined because the model predicted this class 0 times.")
        per_class.append(
            {
                "class": name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
                "predicted_count": predicted,
                "auroc": auroc,
                "n_positive": n_pos,
                "n_negative": n_neg,
                "fpr": fpr,
                "tpr": tpr,
            }
        )

    accuracy = float((y_pred == y_true).mean())
    defined_p = [row["precision"] for row in per_class if row["precision"] is not None]
    defined_r = [row["recall"] for row in per_class if row["recall"] is not None]
    defined_f = [row["f1"] for row in per_class if row["f1"] is not None]
    defined_auc = [row["auroc"] for row in per_class if row["auroc"] is not None]

    def weighted(field: str) -> float | None:
        numerator = 0.0
        denominator = 0.0
        for row in per_class:
            value = row[field]
            if value is None or row["support"] == 0:
                continue
            numerator += value * row["support"]
            denominator += row["support"]
        return safe_div(numerator, denominator)

    excluded_auc = [row["class"] for row in per_class if row["auroc"] is None]
    if excluded_auc:
        notes.append("macro one-vs-rest AUROC averages only classes with a defined AUROC: excluded " + ", ".join(excluded_auc))

    summary = {
        "num_samples": int(len(y_true)),
        "accuracy": accuracy,
        "macro_precision": _mean(defined_p),
        "macro_recall": _mean(defined_r),
        "macro_f1": _mean(defined_f),
        "weighted_precision": weighted("precision"),
        "weighted_recall": weighted("recall"),
        "weighted_f1": weighted("f1"),
        "macro_ovr_auroc": _mean(defined_auc),
        "balanced_accuracy": _mean(defined_r),
        "auroc_defined_classes": len(defined_auc),
        "selection_metric": "validation accuracy; validation macro-F1 is recorded but does not select the checkpoint",
    }
    return {
        "summary": summary,
        "per_class": [{key: value for key, value in row.items() if key not in {"fpr", "tpr"}} for row in per_class],
        "roc": [
            {"class": row["class"], "auroc": row["auroc"], "fpr": row["fpr"], "tpr": row["tpr"]}
            for row in per_class
        ],
        "confusion_matrix": matrix.tolist(),
        "class_names": list(class_names),
        "notes": notes,
    }


def metrics_to_rows(metrics: dict) -> list[dict]:
    rows = []
    for key, value in metrics["summary"].items():
        rows.append({"section": "overall", "name": key, "class": "", "value": value, "note": ""})
    for row in metrics["per_class"]:
        for field in ("precision", "recall", "f1", "support", "auroc", "predicted_count"):
            note = ""
            if row[field] is None:
                note = "undefined (zero denominator or missing class)"
            rows.append(
                {"section": "per_class", "name": field, "class": row["class"], "value": row[field], "note": note}
            )
    for note in metrics["notes"]:
        rows.append({"section": "note", "name": "note", "class": "", "value": "", "note": note})
    return rows
