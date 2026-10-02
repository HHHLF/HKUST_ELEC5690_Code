"""Checks for the three losses. Does not train and does not write into existing runs."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from isic_baseline import CLASS_NAMES
from isic_baseline.data import load_split
from isic_baseline.losses import (
    FocalLoss,
    WeightedCrossEntropy,
    build_criterion,
    inverse_frequency_weights,
    prepare_loss_spec,
    training_class_counts,
)


def expect_close(left, right, message: str) -> None:
    if not torch.allclose(left, right, atol=1e-6, rtol=1e-5):
        raise SystemExit(f"{message}: {left} vs {right}")


def main() -> None:
    torch.manual_seed(0)
    logits = torch.randn(32, 7)
    target = torch.randint(0, 7, (32,))
    cross_entropy = nn.functional.cross_entropy(logits, target)
    focal = FocalLoss(gamma=0.0, alpha=None)(logits, target)
    expect_close(focal, cross_entropy, "gamma=0 focal does not match mean cross-entropy")

    huge = torch.zeros(4, 7)
    huge[:, 0] = 50
    expect_close(
        FocalLoss(gamma=0.0, alpha=None)(huge, torch.zeros(4, dtype=torch.long)),
        nn.functional.cross_entropy(huge, torch.zeros(4, dtype=torch.long)),
        "gamma=0 focal diverges when p_t is about 1",
    )

    baseline = yaml.safe_load((ROOT / "configs" / "baseline.yaml").read_text())
    train_records = load_split("train", baseline["train_image_dir"], baseline["train_csv"])["records"]
    counts = training_class_counts(train_records)
    expected = [1113, 6705, 514, 327, 1099, 115, 142]
    if counts != expected:
        raise SystemExit(f"Training counts {counts} do not match the audited counts {expected}.")
    beta = 0.5
    weights = inverse_frequency_weights(counts, beta)
    for i in range(len(counts)):
        for j in range(len(counts)):
            ratio = weights[i] / weights[j]
            expected_ratio = (counts[j] / counts[i]) ** beta
            if abs(ratio - expected_ratio) > 1e-6:
                raise SystemExit("Class weights are not n_c^(-beta).")
    hard = inverse_frequency_weights(counts, beta=1.0)
    if (max(weights) / min(weights)) >= (max(hard) / min(hard)):
        raise SystemExit("beta=0.5 is not softer than plain inverse frequency.")
    if CLASS_NAMES[weights.index(min(weights))] != "NV" or CLASS_NAMES[weights.index(max(weights))] != "DF":
        raise SystemExit(f"Weight order is wrong: {list(zip(CLASS_NAMES, weights))}")
    try:
        training_class_counts([record for record in train_records if record["label"] != CLASS_NAMES.index("DF")])
    except RuntimeError:
        pass
    else:
        raise SystemExit("Missing DF did not raise.")

    weighted_spec = prepare_loss_spec({"loss": {"name": "weighted_cross_entropy"}}, train_records)
    focal_spec = prepare_loss_spec({"loss": {"name": "focal", "gamma": 2.0, "alpha": None}}, train_records)
    plain_spec = prepare_loss_spec({"loss": {"name": "cross_entropy"}}, train_records)
    device = torch.device("cpu")
    plain = build_criterion(plain_spec, device)
    weighted = build_criterion(weighted_spec, device)
    focal_module = build_criterion(focal_spec, device)
    if not isinstance(plain, nn.CrossEntropyLoss) or plain.weight is not None:
        raise SystemExit("cross_entropy config did not build an unweighted CrossEntropyLoss.")
    if not isinstance(weighted, WeightedCrossEntropy) or list(weighted.class_weights.shape) != [7]:
        raise SystemExit("weighted cross-entropy did not receive a 7-class weight vector.")
    if not isinstance(focal_module, FocalLoss) or focal_module.gamma != 2.0 or focal_module.alpha is not None:
        raise SystemExit("focal config did not build FocalLoss(gamma=2, alpha=None).")
    if not torch.equal(weighted.class_weights.cpu(), torch.tensor(weighted_spec["class_weights"])):
        raise SystemExit("Saved class-weight order does not match the criterion.")
    batch_weights = weighted.class_weights[target]
    if abs(float((batch_weights / batch_weights.mean()).mean()) - 1.0) > 1e-6:
        raise SystemExit("Batch weight mean is not 1.")
    expect_close(
        weighted(logits, target),
        nn.functional.cross_entropy(logits, target, weight=weighted.class_weights),
        "Batch normalization does not match mean CrossEntropyLoss(weight=raw)",
    )

    for module in (weighted, focal_module):
        sample = torch.randn(8, 7, requires_grad=True)
        loss = module(sample, torch.randint(0, 7, (8,)))
        loss.backward()
        if sample.grad is None or not torch.isfinite(sample.grad).all() or float(sample.grad.abs().sum()) == 0:
            raise SystemExit(f"Non-finite or empty gradient for {type(module).__name__}.")

    configs = {}
    for name in ("baseline", "weighted_ce", "focal"):
        configs[name] = yaml.safe_load((ROOT / "configs" / f"{name}.yaml").read_text())
    outputs = {name: Path(cfg["output_dir"]) for name, cfg in configs.items()}
    if len(set(outputs.values())) != 3:
        raise SystemExit(f"Output directories collide: {outputs}")
    for name in ("weighted_ce", "focal"):
        cfg = configs[name]
        if Path(cfg["output_dir"]) == Path("runs/baseline") or "baseline_e30" in cfg["output_dir"]:
            raise SystemExit(f"{name} would write into an existing baseline run.")
        if cfg["resume"]:
            raise SystemExit(f"{name} resume is not empty, so it could load a finetuned checkpoint.")
        if "dinov2_vitb14_pretrain" not in cfg["weights"]:
            raise SystemExit(f"{name} is not starting from the DINOv2 pretrain file.")
        reference = configs["baseline"]
        for key in ("epochs", "batch_size", "seed", "lr_backbone", "lr_head", "image_size"):
            if cfg[key] != reference[key]:
                raise SystemExit(f"{name} {key}={cfg[key]} does not match baseline.yaml {key}={reference[key]}.")
    print("loss checks passed")
    print("class_order", ",".join(CLASS_NAMES))
    print("class_counts", counts)
    print("class_weights", [round(value, 8) for value in weights])


if __name__ == "__main__":
    main()
