# ELEC5690 Problem 1(a)–(c): ISIC 2018 seven-class baseline

The image backbone is a pretrained DINOv2 ViT with a randomly initialized 7-class linear head. Both are fine-tuned end to end. The baseline loss is ordinary unweighted `CrossEntropyLoss`: no class weights, focal loss, or class resampling.

The checkpoint is selected only by validation **accuracy**. Validation macro-F1 is recorded and does not choose the checkpoint. The test set is evaluated once, after `best.pt` has been selected.

## Environment

The server conda environment `prism` already has PyTorch 2.8, torchvision, pandas, scikit-learn, matplotlib, Pillow, and PyYAML. Reuse it. Do not reinstall torch into that environment.

```bash
source /home/lheax/miniconda3/etc/profile.d/conda.sh
conda activate prism
cd /home/lheax/data/elec5690_isic2018
python -c "import torch,torchvision,pandas,sklearn,matplotlib,PIL,yaml,numpy; print(torch.__version__, torch.cuda.is_available())"
```

If one library is missing, install only that library, for example `conda install -n prism pyyaml -y`. Use `requirements.txt` only when creating a new environment, and install a PyTorch build that matches this machine's CUDA.

GPUs 0–3 are usually occupied. Pick a free GPU before training, for example 5, 7, or 8:

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
export CUDA_VISIBLE_DEVICES=5
```

## Data and weights

Local data, under `/home/lheax/data/ISIC2018_Task3/`:

| split | image directory | label CSV |
|---|---|---|
| train | `ISIC2018_Task3_Training_Input` | `ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv` |
| val | `ISIC2018_Task3_Validation_Input` | `ISIC2018_Task3_Validation_GroundTruth/ISIC2018_Task3_Validation_GroundTruth.csv` |
| test | `ISIC2018_Task3_Test_Input` | `ISIC2018_Task3_Test_GroundTruth/ISIC2018_Task3_Test_GroundTruth.csv` |

CSV columns are `image, MEL, NV, BCC, AKIEC, BKL, DF, VASC`. `image` is the id without an extension, and the file is `{id}.jpg`. The seven label columns are one-hot. Class order matches the column order: MEL=0, NV=1, BCC=2, AKIEC=3, BKL=4, DF=5, VASC=6.

At startup the code checks that every label has an image, that every label is one of these seven classes, and that image ids do not overlap across splits. It prints the sample count and class distribution. The assignment counts are 10,015 train, 193 validation, and 1,512 test. A mismatch is printed as DIFF. Samples are not dropped silently.

Backbone weights (no ImageNet 1000-way head):

- ViT-B/14: `weights/dinov2_vitb14_pretrain.pth`
- ViT-S/14: `weights/dinov2_vits14_pretrain.pth`

The program does not download weights at startup. Set the path in the config or with `--weights`.

Normalization is the ImageNet mean `(0.485, 0.456, 0.406)` and std `(0.229, 0.224, 0.225)` used by official DINOv2. The input side length must be a multiple of 14. The default is 224.

## Commands

```bash
source /home/lheax/miniconda3/etc/profile.d/conda.sh
conda activate prism
cd /home/lheax/data/elec5690_isic2018
export CUDA_VISIBLE_DEVICES=5

python smoke_test.py

python train.py --config configs/baseline.yaml
# python train.py --config configs/baseline.yaml --epochs 30 --output-dir runs/baseline_e30

python evaluate.py --config configs/baseline.yaml --checkpoint runs/baseline/best.pt

python visualize.py --config configs/baseline.yaml --checkpoint runs/baseline/best.pt

# python evaluate.py --config configs/baseline.yaml --checkpoint runs/baseline_e30/best.pt --output-dir runs/baseline_e30/test
# python visualize.py --config configs/baseline.yaml --checkpoint runs/baseline_e30/best.pt --predictions runs/baseline_e30/test/predictions.csv --history runs/baseline_e30/history.json --output-dir runs/baseline_e30/figures
```

Common overrides:

```bash
python train.py --config configs/baseline.yaml \
  --variant vits14 --weights weights/dinov2_vits14_pretrain.pth \
  --batch-size 32 --epochs 10 --image-size 224 \
  --lr-backbone 1e-5 --lr-head 1e-3 \
  --output-dir runs/vits14

python train.py --config configs/baseline.yaml
```

`train.py` reads only the training and validation CSVs. `evaluate.py` is what reads the test labels. Validation metrics are never written out as test metrics.

## Outputs

`runs/baseline/`

- `best.pt`: checkpoint with the highest validation accuracy; ties go to the higher validation macro-F1
- `last.pt`: resume checkpoint, including the optimizer and AMP scaler
- `history.json`, `history.csv`, `figures/loss_acc_curves.png` (and PDF)

`runs/baseline/test/`

- `predictions.csv`: image id, true class, predicted class, and seven class probabilities
- `metrics.json`, `metrics.csv`
- `confusion_matrix.csv`, `confusion_matrix.png`
- `roc.png`: seven one-vs-rest ROC curves, plus a macro-average ROC when every class has both positive and negative samples

If a denominator is zero or the test set lacks a class, precision, recall, F1, or AUROC is left empty and explained in `notes`. The missing value is not filled with 0.

`runs/baseline/figures/gradcam/`

- original, heatmap, and overlay for each case
- `panel.png`: the case panel
- `case_analysis.md`: a note template filled from heatmap coordinates and probability margins. Write the medical interpretation only after looking at the images. The blank line is for that note.

Grad-CAM uses `norm1` of the last block. Patch tokens at that layer affect the final CLS token through attention. Patch outputs of the same block do not affect that CLS token, so their gradients are zero and that map is not used. The CLS token is removed and the patch grid is resized to the input. An all-zero or constant heatmap raises an error instead of saving a blank figure.

## Problem 1(d): two class-imbalance losses

`loss.name` selects the loss. All three settings share the same training and evaluation code:

| config | `loss.name` | output directory |
|---|---|---|
| `configs/baseline.yaml` | `cross_entropy` | `runs/baseline` |
| `configs/weighted_ce.yaml` | `weighted_cross_entropy` | `outputs/weighted_ce` |
| `configs/focal.yaml` | `focal` | `outputs/focal` |

Plain cross-entropy is the mean `CrossEntropyLoss` with no class weights. Weighted cross-entropy counts **training-set** true labels only, in the fixed order MEL, NV, BCC, AKIEC, BKL, DF, VASC. `w_c = n_c^(-beta)`. `beta` defaults to 0.5, so the weight follows `1/sqrt(n_c)` instead of `1/n_c`. `beta: 1` recovers plain inverse frequency. Each batch divides those weights by their mean inside that batch, so the batch's weight mean is 1. If any training class has count 0, training raises an error. The counts and weights are printed and saved to that run's `loss_spec.json`. Validation and test labels are not used to compute weights.

Focal loss is the single-label multi-class form and takes logits:

\[
-\,(1-p_t)^{\gamma}\log p_t,\quad p_t=\exp(\log\_softmax(logits)_y)
\]

`configs/focal.yaml` sets `gamma` to 1.0 and leaves `alpha` null. If `gamma` is omitted, the code default is 2.0. Inverse-frequency weights are not multiplied into focal loss by default. With `gamma: 0` and `alpha: null`, focal loss matches the mean of ordinary cross-entropy.

The finished Problem 1(a) result is `runs/baseline`: 10 epochs and batch size 64. `configs/baseline.yaml` currently says `epochs: 50` and `batch_size: 256`, which was edited after that run finished. The weighted-CE and focal configs do not copy those two numbers. They match the training settings recorded in `runs/baseline/best.pt`. `runs/baseline_e30` used 30 epochs and batch size 256, so it is not included in the three-model table. Both new models start from `dinov2_vitb14_pretrain.pth` and a new 7-class head. They do not load a checkpoint already fine-tuned on ISIC.

A new training run overwrites `best.pt`, `last.pt`, and `history.json` in the output directory, and rewrites `<output_dir>_train.log` beside that directory. Pass `--resume <run>/last.pt` to continue that checkpoint instead. `evaluate.py` overwrites the test files and rewrites `<run>_eval.log`.

```bash
source /home/lheax/miniconda3/etc/profile.d/conda.sh
conda activate prism
cd /home/lheax/data/elec5690_isic2018
python check_losses.py

CUDA_VISIBLE_DEVICES=8 python train.py --config configs/baseline.yaml
CUDA_VISIBLE_DEVICES=5 python train.py --config configs/weighted_ce.yaml
CUDA_VISIBLE_DEVICES=7 python train.py --config configs/focal.yaml

CUDA_VISIBLE_DEVICES=8 python evaluate.py --config configs/baseline.yaml
CUDA_VISIBLE_DEVICES=5 python evaluate.py --config configs/weighted_ce.yaml
CUDA_VISIBLE_DEVICES=7 python evaluate.py --config configs/focal.yaml

python compare_losses.py
```

`outputs/weighted_ce_conflict/` is an invalid run left by two jobs writing the same directory. Do not put its checkpoint or test metrics in the report. The valid weighted-CE result is `outputs/weighted_ce/`.

The comparison table and side-by-side confusion matrices are written to `outputs/comparison/`. Each experiment directory also contains `config_snapshot.json`, `loss_spec.json`, `best.pt`, `history.json`, `figures/loss_acc_curves.png`, and predictions plus metrics under `test/`. Test metrics add balanced accuracy to the Problem 1(a) set: accuracy, precision, recall, F1, AUROC, per-class recall, and the 7×7 confusion matrix.
