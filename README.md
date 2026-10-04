# ELEC5690

## Environment

Python 3.10 and an NVIDIA GPU. PyTorch must be a CUDA build; a CPU wheel will not train these models.

```bash
conda create -n elec5690 python=3.10 -y
conda activate elec5690
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install "numpy>=1.24" "pandas>=2.0" "scikit-learn>=1.3" "matplotlib>=3.7" "pillow>=10.0" "pyyaml>=6.0" "scipy>=1.11" "nibabel>=5.0"
cd /data/HKUST_elec5690_programming
```

`cu128` matches this machine (CUDA 12.8). On another GPU, pick the wheel from [pytorch.org](https://pytorch.org/get-started/locally/). Do not install `torch` from the default PyPI index after that; it can replace the CUDA build.

| Package | Used for |
|---|---|
| torch, torchvision | training, DINOv2, 2D/3D U-Net |
| numpy, pillow | images and volumes |
| pandas, scikit-learn | Problem 1 metrics and tables |
| matplotlib | figures |
| pyyaml | configs |
| scipy | surface distance (ASD, HD95) |
| nibabel | NIfTI export |

Problem 1 also needs the backbone file `weights/dinov2_vitb14_pretrain.pth`.

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
