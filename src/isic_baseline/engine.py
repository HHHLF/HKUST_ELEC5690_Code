"""Train and evaluation loops.

The default criterion is unweighted cross-entropy. Pass another criterion to use
weighted cross-entropy or focal loss for both the training loss and the validation loss.
"""

from __future__ import annotations

import torch

from isic_baseline import CLASS_NAMES
from isic_baseline.metrics import compute_metrics


def train_one_epoch(model, loader, optimizer, scaler, device, use_amp: bool, criterion=None) -> dict:
    model.train()
    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    seen = 0
    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, labels)
        if scaler is not None and use_amp and device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        batch = labels.shape[0]
        total_loss += float(loss.detach()) * batch
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        seen += batch
    return {"loss": total_loss / seen, "accuracy": correct / seen, "num_samples": seen}


@torch.no_grad()
def evaluate_loader(model, loader, device, use_amp: bool = True, criterion=None) -> dict:
    model.eval()
    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    seen = 0
    labels_all = []
    probs_all = []
    ids_all = []
    for images, labels, image_ids in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, labels)
        probs = torch.softmax(logits.float(), dim=1)
        batch = labels.shape[0]
        total_loss += float(loss) * batch
        seen += batch
        labels_all.append(labels.cpu())
        probs_all.append(probs.cpu())
        ids_all.extend(list(image_ids))
    y_true = torch.cat(labels_all).numpy()
    y_prob = torch.cat(probs_all).numpy()
    summary = compute_metrics(y_true, y_prob, list(CLASS_NAMES))["summary"]
    return {
        "loss": total_loss / seen,
        "accuracy": float(summary["accuracy"]),
        "macro_f1": summary["macro_f1"],
        "num_samples": seen,
        "y_true": y_true,
        "y_prob": y_prob,
        "image_ids": ids_all,
        "class_names": list(CLASS_NAMES),
    }


def prediction_rows(result: dict) -> list[dict]:
    rows = []
    for image_id, true_index, probs in zip(result["image_ids"], result["y_true"], result["y_prob"]):
        pred_index = int(probs.argmax())
        row = {
            "image_id": image_id,
            "true_label": CLASS_NAMES[int(true_index)],
            "pred_label": CLASS_NAMES[pred_index],
        }
        for name, prob in zip(CLASS_NAMES, probs.tolist()):
            row[f"prob_{name}"] = float(prob)
        rows.append(row)
    return rows
