# Problem 2(d): 2D U-Net

This file is the run guide for the 2D experiment. It does not replace `README.md` and it does not discuss results.

Use the `prism` environment. It has the PyTorch build used by the 3D run.

```bash
source /home/lheax/miniconda3/etc/profile.d/conda.sh
conda activate prism
cd /home/lheax/data/HKUST_elec5690_programming
```

## Data and axis order

The local release is `/home/lheax/data/atriaseg2018`. The 3D audit found no HDF5 files. Each case is NRRD: `lgemri.nrrd` (image) and `laendo.nrrd` (cavity label, raw values 0 and 255).

Array order is `(X, Y, Z)`:

- axis 0 = X
- axis 1 = Y
- axis 2 = Z

Z is axis 2 because the 3D loader keeps the NRRD size order. One model input is `image[:, :, z]`, shape `(X, Y)`. Cases are 576×576×88 or 640×640×88. A slice index is the index along axis 2.

The case split is the finished 3D split (`outputs/segmentation3d/split.json`): official Testing Set (54), and a case holdout from the Training Set with `split_seed` 42 and `val_ratio` 0.2 (80 train / 20 val). Slices from one case stay in that case's split.

Normalization is the 3D per-case z-score of nonzero voxels, read from the existing cache `/dev/shm/elec5690_la_seg3d` (`image_f32.npy`). Volumes are loaded once into memory (`cache_mode: case_memory`). Training does not re-read NRRD each epoch. Slices are taken after that normalization. There is no resampling.

## Commands

Smoke test (forward, backward, slice index, shape restore, split check):

```bash
CUDA_VISIBLE_DEVICES=1 python smoke_test_segmentation2d.py
```

Train. The checkpoint is chosen by mean full-volume validation Dice.

```bash
CUDA_VISIBLE_DEVICES=1 python train_segmentation2d.py --config configs/segmentation2d.yaml
```

Resume from the last checkpoint:

```bash
CUDA_VISIBLE_DEVICES=1 python train_segmentation2d.py --config configs/segmentation2d.yaml --resume outputs/segmentation2d/last.pt
```

Test the validation-selected `best.pt` on every test case, all Z slices:

```bash
CUDA_VISIBLE_DEVICES=1 python evaluate_segmentation2d.py --config configs/segmentation2d.yaml --split test
```

Comparison tables and figures (requires the 2D test metrics and the existing 3D test metrics):

```bash
python compare_segmentation2d3d.py --config configs/segmentation2d.yaml
```

`comparison_protocol.json` is written at the start of training. Running compare does not retrain either model.

## Settings aligned with the 3D run

Taken from `outputs/segmentation3d/config_snapshot.json`, `split.json`, `train_finished.json`, and `test/summary.json`.

- Seed 42, AdamW, learning rate 0.001, weight decay 0.0001.
- Cosine decay, `eta_min` 1e-6, `T_max` 300 scheduler steps. The 3D scheduler steps once per epoch, and one 3D epoch is 40 optimizer updates, so the 2D scheduler steps every 40 optimizer updates.
- AMP and gradient clip 1.0.
- Loss: CE weight 1, soft Dice weight 1, smooth 1. Dice is per-sample on the foreground softmax probability, then the batch mean. CE is mean cross-entropy.
- U-Net widths `[16, 32, 64, 128, 256]`, instance norm, two 3×3 convolutions per level. Spatial ops are 2D.
- Crop `[112, 112]`, the XY size of the 3D patch `[112, 112, 80]`.
- Foreground sampling probability 0.5. The other half draws a Z index uniformly from all slices, including slices with no left atrium. XY foreground jitter is `[16, 16]`.
- In-plane flips and 90-degree rotations use probability 0.5. Intensity scale is 0.9–1.1 and Gaussian noise std is 0.05, with probability 0.5.
- Checkpoint rule: strict improvement of the mean Dice on full validation volumes. Slice Dice is not used. The test set is not used to choose it. Training-time validation records that Dice and Jaccard. ASD and HD95 are computed on the test volumes with `binary_segmentation_metrics`.
- Post-processing: none.
- Validation every 400 optimizer updates, matching 3D validation every 10 epochs. Early stopping waits for 50 validations without improvement. The 3D run completed 300 epochs and did not stop early.

## Differences that are intentional

- A 2D input is one axial slice. The 3D Z flip has no in-plane effect on that slice and is not applied. Intensities are not spatially rescaled.
- 2D batch size is 64. The 3D batch size is 16.
- `samples_per_epoch` is 640 crops (80 cases × 8). One 2D epoch is not one 3D epoch in wall time or in FLOPs.
- `max_steps` is 12000, the computed 3D optimizer update count: 300 epochs × ceil(80×8 / 16) updates. The 3D history file does not store a step counter; the count is computed from the snapshot, the split, and `train_finished.json`. Equal update counts are not equal compute. A 3D sample is a 112×112×80 patch.
- A slice whose sides are within the 112 training crop is padded if needed and run as one image. Larger slices, including the 576 and 640 cases, use 2D sliding windows of that crop, overlap 0.5, and Gaussian probability fusion. The window matches the training crop because the network uses instance normalization. The 3D run uses 3D windows of 112×112×80 with the same overlap and fusion.
- Small slices are zero-padded on the trailing side, together with the label. The pad is removed after inference.

## Metrics and distance unit

Dice, Jaccard, ASD, and HD95 use `la_seg3d.metrics.binary_segmentation_metrics` on the reconstructed 3D mask.

The surface is the voxels removed by one 6-neighborhood erosion. Distances are Euclidean EDT distances. ASD is the mean of both directions. HD95 is the 95th percentile of that same set.

NRRD headers store spacing `(1, 1, 1)` and no space units. Both runs report distances in **voxels**, not millimetres.

If both masks are empty, Dice and Jaccard are 1 and both distances are 0. If only one mask is empty, Dice and Jaccard are 0 and both distances are inf. Those cases stay in the table. The official mean is inf when any distance is non-finite. Cases are averaged with equal weight. A failed case is not dropped.

## Outputs

Training:

- `outputs/segmentation2d/best.pt`
- `outputs/segmentation2d/last.pt`
- `outputs/segmentation2d/config_snapshot.json`
- `outputs/segmentation2d/comparison_protocol.json`
- `outputs/segmentation2d/history.csv`
- `outputs/segmentation2d/history.json`
- `outputs/segmentation2d/train.log`
- `outputs/segmentation2d/split.json`
- `outputs/segmentation2d/slice_manifest.json`
- `outputs/segmentation2d/figures/loss_val_dice.png` and `.pdf`

Test:

- `outputs/segmentation2d/test/per_case_metrics.csv`
- `outputs/segmentation2d/test/per_case_metrics.json`
- `outputs/segmentation2d/test/summary.csv`
- `outputs/segmentation2d/test/summary.json`
- `outputs/segmentation2d/test/predictions/*.npz` (same fields as the 3D npz: `prediction`, `image`, `label`, `spacing_xyz`, `shape_xyz`, `axis_order`, `distance_unit`)
- `outputs/segmentation2d/test/nifti/`

Comparison:

- `outputs/segmentation2d/comparison/comparison_protocol.json`
- `outputs/segmentation2d/comparison/comparison_summary.csv`
- `outputs/segmentation2d/comparison/comparison_summary.json`
- `outputs/segmentation2d/comparison/comparison_per_case.csv`
- `outputs/segmentation2d/comparison/comparison_per_case.json`
- `outputs/segmentation2d/comparison/figures/metric_boxplots.png` and `.pdf`
- `outputs/segmentation2d/comparison/figures/metric_paired.png` and `.pdf`
- `outputs/segmentation2d/comparison/figures/test_cases_4x4.png` and `.pdf`
- `outputs/segmentation2d/comparison/figures/test_cases_error_overlay.png` and `.pdf`
- `outputs/segmentation2d/comparison/figures/training_curves_2d.png` and `.pdf`
- `outputs/segmentation2d/comparison/figures/training_curves_steps.png` and `.pdf`

The four displayed cases and Z indices come from `outputs/segmentation3d/problem2b/selected_cases_metrics.json`. Dice and HD95 on those figures are full-volume scores.
