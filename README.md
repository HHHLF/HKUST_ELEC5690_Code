# ELEC5690

## Environment

Use the server conda environment `prism` (PyTorch 2.8). Do not reinstall torch into it.

```bash
source /home/lheax/miniconda3/etc/profile.d/conda.sh
conda activate prism
cd /home/lheax/data/HKUST_elec5690_programming
```

For a new environment, install a CUDA build of PyTorch for this machine, then `pip install -r requirements.txt`.

## Problem 1

ISIC 2018 Task 3, 7-class classification. One DINOv2 ViT-B/14 with a random head, three losses: cross-entropy, class-weighted cross-entropy, and focal loss. The checkpoint is chosen by validation accuracy. Test evaluation runs after that choice. `compare_losses.py` and `visualize.py` compare the three runs and write figures.

Data: `/home/lheax/data/ISIC2018_Task3`.

```bash
# train with cross-entropy
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/baseline.yaml
# train with class-weighted cross-entropy
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/weighted_ce.yaml
# train with focal loss
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/focal.yaml

# test the checkpoint chosen on validation
CUDA_VISIBLE_DEVICES=0 python evaluate.py --config configs/baseline.yaml
CUDA_VISIBLE_DEVICES=0 python evaluate.py --config configs/weighted_ce.yaml
CUDA_VISIBLE_DEVICES=0 python evaluate.py --config configs/focal.yaml

# compare the three test results
python compare_losses.py
# write the comparison figures
python visualize.py
```

## Problem 2

Left-atrium cavity segmentation on `/home/lheax/data/atriaseg2018` (NRRD, not HDF5). Arrays are `(X, Y, Z)`; Z is axis 2. Distances are in voxels. The case split is shared: 80 train / 20 val / 54 test.

- **2(a)** 3D U-Net, CE + soft Dice. Checkpoint by full-volume validation Dice. Test metrics: Dice, Jaccard, ASD, HD95.
- **2(b)** Training curve and four test-case figures from the saved 3D run. Does not retrain.
- **2(d)** 2D U-Net on axial slices, same split and metrics. Comparison tables and figures against 3D.

```bash
# build the volume cache used by training
python prepare_la_cache.py --config configs/la_seg3d/unet3d_ce_dice.yaml
# train the 3D U-Net; checkpoint by full-volume validation Dice
CUDA_VISIBLE_DEVICES=0 python train_seg3d.py --config configs/la_seg3d/unet3d_ce_dice.yaml
# resume 3D training from the last checkpoint
CUDA_VISIBLE_DEVICES=0 python train_seg3d.py --config configs/la_seg3d/unet3d_ce_dice.yaml --resume outputs/segmentation3d/last.pt
# test Dice, Jaccard, ASD, and HD95 on full volumes
CUDA_VISIBLE_DEVICES=0 python evaluate_seg3d.py --config configs/la_seg3d/unet3d_ce_dice.yaml --split test
# write the 3D training curve and four test-case figures
python visualize_problem2b.py --config configs/la_seg3d/problem2b.yaml

# train the 2D U-Net on axial slices
CUDA_VISIBLE_DEVICES=0 python train_segmentation2d.py --config configs/segmentation2d.yaml
# resume 2D training from the last checkpoint
CUDA_VISIBLE_DEVICES=0 python train_segmentation2d.py --config configs/segmentation2d.yaml --resume outputs/segmentation2d/last.pt
# test the 2D model on full volumes
CUDA_VISIBLE_DEVICES=0 python evaluate_segmentation2d.py --config configs/segmentation2d.yaml --split test
# write the 2D vs 3D tables and figures
python compare_segmentation2d3d.py --config configs/segmentation2d.yaml
```

Outputs: `outputs/baseline`, `outputs/weighted_ce`, `outputs/focal`, `outputs/segmentation3d`, `outputs/segmentation2d`.
