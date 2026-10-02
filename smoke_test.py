"""One-batch train step, checkpoint round-trip, metric shapes, and a non-empty Grad-CAM."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from isic_baseline import CLASS_NAMES
from isic_baseline.data import audit_splits, load_split, make_loader
from isic_baseline.engine import evaluate_loader, train_one_epoch
from isic_baseline.gradcam import generate_gradcam
from isic_baseline.metrics import compute_metrics
from isic_baseline.model import build_model
from isic_baseline.utils import get_device, load_config, set_seed


def main() -> None:
    cfg = load_config(ROOT / "configs" / "baseline.yaml")
    set_seed(int(cfg["seed"]))
    device = get_device("cuda" if torch.cuda.is_available() else "cpu")
    train_split = load_split("train", cfg["train_image_dir"], cfg["train_csv"])
    val_split = load_split("val", cfg["val_image_dir"], cfg["val_csv"])
    test_split = load_split("test", cfg["test_image_dir"], cfg["test_csv"])
    audit = audit_splits({"train": train_split, "val": val_split, "test": test_split})
    for name, expected in (("train", 10015), ("val", 193), ("test", 1512)):
        got = audit[name]["num_samples"]
        if got != expected:
            raise SystemExit(f"{name} sample count {got} != assignment count {expected}")

    model = build_model("vitb14", cfg["weights"], load_pretrained=True)
    loaded = torch.load(cfg["weights"], map_location="cpu")
    reference = loaded["patch_embed.proj.weight"]
    if not torch.equal(model.backbone.patch_embed.proj.weight.detach().cpu(), reference):
        raise SystemExit("Backbone weight does not match the DINOv2 pretrain checkpoint.")
    if model.head.out_features != 7:
        raise SystemExit("Head is not 7-way.")
    small_weights = ROOT / "weights" / "dinov2_vits14_pretrain.pth"
    small = build_model("vits14", small_weights, load_pretrained=True)
    if small.backbone.embed_dim != 384 or small.head.out_features != 7:
        raise SystemExit("ViT-S/14 backbone or head shape is wrong.")
    del small

    model.to(device)
    optimizer = torch.optim.AdamW(model.param_groups(1e-5, 1e-3, 0.0))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    loader = make_loader(train_split["records"][: cfg["batch_size"]], cfg["image_size"], 4, True, 0, cfg["seed"])
    before = model.head.weight.detach().clone()
    stats = train_one_epoch(model, loader, optimizer, scaler, device, device.type == "cuda")
    if not np.isfinite(stats["loss"]):
        raise SystemExit(f"Non-finite loss {stats}")
    if torch.equal(before, model.head.weight.detach()):
        raise SystemExit("Classification head did not receive a gradient update.")
    backbone_grad = model.backbone.patch_embed.proj.weight.grad
    if backbone_grad is None or float(backbone_grad.abs().sum()) == 0:
        raise SystemExit("Backbone did not receive a non-zero gradient.")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ckpt.pt"
        torch.save({"model": model.state_dict(), "loss": stats["loss"]}, path)
        restored = build_model("vitb14", weights_path=None, load_pretrained=False).to(device)
        restored.load_state_dict(torch.load(path, map_location=device)["model"])
        for left, right in zip(model.state_dict().values(), restored.state_dict().values()):
            if not torch.equal(left, right):
                raise SystemExit("Checkpoint reload changed a tensor.")

    eval_loader = make_loader(val_split["records"][:8], cfg["image_size"], 4, False, 0, cfg["seed"])
    result = evaluate_loader(model, eval_loader, device, use_amp=device.type == "cuda")
    if result["y_prob"].shape[1] != len(CLASS_NAMES):
        raise SystemExit(f"Bad probability shape {result['y_prob'].shape}")
    metrics = compute_metrics(result["y_true"], result["y_prob"], list(CLASS_NAMES))
    if len(metrics["confusion_matrix"]) != 7 or len(metrics["confusion_matrix"][0]) != 7:
        raise SystemExit("Confusion matrix is not 7x7.")

    model.eval()
    image, label, image_id = eval_loader.dataset[0]
    heatmap, layer = generate_gradcam(model, image.unsqueeze(0).to(device), int(label))
    if heatmap.sum() <= 0 or heatmap.std() <= 0:
        raise SystemExit("Grad-CAM map is empty.")
    print(
        f"smoke ok device={device} loss={stats['loss']:.4f} probs={tuple(result['y_prob'].shape)} "
        f"gradcam={heatmap.shape} layer={layer} image={image_id}"
    )


if __name__ == "__main__":
    main()
