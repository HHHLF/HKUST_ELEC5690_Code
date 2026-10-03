# ELEC5690 ISIC 2018 baseline

This repo fine-tunes a pretrained DINOv2 ViT-B/14 with a randomly initialized 7-class head on ISIC 2018 Task 3 (MEL, NV, BCC, AKIEC, BKL, DF, VASC). Three runs share the same training code and differ only in the loss: unweighted cross-entropy, class-weighted cross-entropy, and focal loss. The checkpoint is chosen by validation accuracy, and the test set is evaluated only after that choice.

## Environment

Reuse the server conda environment `prism`. It already has PyTorch 2.8 and the other libraries. Do not reinstall torch into it.

```bash
source /home/lheax/miniconda3/etc/profile.d/conda.sh
conda activate prism
cd /home/lheax/data/elec5690_isic2018
```

For a new environment, install a CUDA build of PyTorch that matches this machine, then `pip install -r requirements.txt`.

## Dataset

ISIC 2018 Challenge Task 3, from <https://challenge.isic-archive.com/data/#2018>. The local copy is `/home/lheax/data/ISIC2018_Task3` (train 10,015 / validation 193 / test 1,512). Paths are set in `configs/baseline.yaml`, `configs/weighted_ce.yaml`, and `configs/focal.yaml`.

## Run

```bash
CUDA_VISIBLE_DEVICES=8 python train.py --config configs/baseline.yaml
CUDA_VISIBLE_DEVICES=5 python train.py --config configs/weighted_ce.yaml
CUDA_VISIBLE_DEVICES=7 python train.py --config configs/focal.yaml

CUDA_VISIBLE_DEVICES=8 python evaluate.py --config configs/baseline.yaml
CUDA_VISIBLE_DEVICES=5 python evaluate.py --config configs/weighted_ce.yaml
CUDA_VISIBLE_DEVICES=7 python evaluate.py --config configs/focal.yaml
```

## Problem 2(a): 3D left-atrium segmentation

This part is separate from the ISIC classifier. Code is in `src/la_seg3d/`, the config is `configs/la_seg3d/unet3d_ce_dice.yaml`, and runs write to `outputs/segmentation3d/`.

### Data

The local release is `/home/lheax/data/atriaseg2018`. A recursive search finds no `.h5` files. Each case is a directory of NRRD volumes:

- `Training Set/<case_id>/lgemri.nrrd`: MRI, 100 cases
- `Testing Set/<case_id>/lgemri.nrrd`: MRI, 54 cases
- `laendo.nrrd`: left-atrium cavity. Raw values are 0 and 255, mapped to 0 and 1. Any other value raises.
- `lawall.nrrd` is the wall and is not the training target.

NRRD `sizes` are X Y Z with X fastest on disk. Arrays are stored as (X, Y, Z), which is the model order (D, H, W). The default patch `[112, 112, 80]` is `[D, H, W] = [X, Y, Z]`. Predictions are saved in that same order. There is no resampling.

Every header has `space directions: (1,0,0) (0,1,0) (0,0,1)` and no `space units`. ASD and HD95 are therefore in **voxels**, not millimetres. The challenge paper's 0.625 mm spacing is not encoded in these files and is not applied. NIfTI exports use that same unit spacing so ITK-SNAP can open them.

The official training and testing directories are the split. Validation is 20% of the training cases at case level (`split_seed: 42`). `outputs/segmentation3d/split.json` records the ids and the overlap check. The test set is not used to pick the checkpoint or change preprocessing.

Intensity normalization is a per-case z-score of the nonzero voxels. An empty or zero-variance nonzero region becomes zeros. Volumes smaller than the patch are zero-padded on the trailing side and cropped back after prediction.

### Model and loss

The network is a 3D U-Net: Conv3d, four max-pools, ConvTranspose3d, skip concatenations, and two 3×3×3 convolutions at each of the five resolutions (16, 32, 64, 128, 256). The head outputs two-class logits `[B, 2, D, H, W]`.

Normalization is **InstanceNorm3d** (GroupNorm is the config alternative). That is the standard choice for this 3D U-Net when the batch contains only two patches; BatchNorm would be a poor fit. The norm statistics are computed in float32 inside mixed precision. Upsampled maps are center-cropped or padded so they match the skip connection.

`torch.nn.CrossEntropyLoss` takes the raw logits and the integer labels. Dice loss is `1 - soft Dice` between the softmax foreground probability and the foreground mask, averaged over the batch after a per-patch Dice, with `dice_smooth`. It is not computed on argmax labels.

`loss = ce_weight * CE + dice_weight * DiceLoss` with both weights at 1.0.

### Metrics

For each full test volume, with P and G the binary foreground masks:

- Dice = `2|P∩G| / (|P|+|G|)`
- Jaccard = `|P∩G| / |P∪G|`
- Surfaces are the voxels removed by one 6-neighborhood erosion (`scipy.ndimage`, connectivity 1), the same construction as medpy.
- Distances use `distance_transform_edt` and the voxel spacing in (X, Y, Z) order.
- ASD is the mean of the concatenated prediction-to-ground-truth and ground-truth-to-prediction surface distances.
- HD95 is the 95th percentile of that same concatenated set.

Both masks empty: Dice = Jaccard = 1 and both distances = 0. Only one mask empty: Dice = Jaccard = 0 and both distances = inf. Empty predictions are kept. The reported mean is the unweighted mean of the per-case values. If any distance is non-finite, that official mean is inf; `finite_mean` is only a diagnostic.

### Commands

```bash
source /home/lheax/miniconda3/etc/profile.d/conda.sh
conda activate prism
cd /home/lheax/data/elec5690_isic2018

python prepare_la_cache.py --config configs/la_seg3d/unet3d_ce_dice.yaml
CUDA_VISIBLE_DEVICES=5 python smoke_test_seg3d.py --config configs/la_seg3d/unet3d_ce_dice.yaml
CUDA_VISIBLE_DEVICES=5 python train_seg3d.py --config configs/la_seg3d/unet3d_ce_dice.yaml
CUDA_VISIBLE_DEVICES=5 python train_seg3d.py --config configs/la_seg3d/unet3d_ce_dice.yaml --resume outputs/segmentation3d/last.pt
CUDA_VISIBLE_DEVICES=5 python evaluate_seg3d.py --config configs/la_seg3d/unet3d_ce_dice.yaml --split test
python visualize_problem2b.py --config configs/la_seg3d/problem2b.yaml
```

Problem 2(b) only reads `outputs/segmentation3d/history.csv`, `test/per_case_metrics.csv`, and `test/predictions/*.npz`. It writes `outputs/segmentation3d/problem2b/` and does not retrain or overwrite the Problem 2(a) checkpoint or test metrics.

Training saves `best.pt` by the mean Dice of the full validation volumes (sliding window, overlap 0.5, Gaussian fusion of softmax probabilities on the GPU, then argmax). Patch sampling for the next batch runs on a thread while the GPU trains; `num_workers` stays 0 because the volumes are already in memory and forking after CUDA starts is unsafe. It also saves `last.pt`, `config_snapshot.json`, `history.csv`, `history.json`, `train.log`, and `figures/loss_val_dice.png`. Test evaluation reads `best.pt` only. The random seed fixes the case split, weight initialization, and patch sampling. cuDNN benchmark is on, so a rerun is not bitwise identical.

Test outputs are under `outputs/segmentation3d/test/`: `per_case_metrics.csv`, `per_case_metrics.json`, `summary.json`, `summary.csv`, `predictions/*.npz`, `nifti/*.nii.gz`, and `previews/*.png`.
