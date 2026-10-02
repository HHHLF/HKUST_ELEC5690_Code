"""ISIC 2018 Task 3 seven-class baseline (ELEC5690 Problem 1 a–c)."""

CLASS_NAMES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]

# Official DINOv2 preprocessing uses ImageNet mean/std.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Counts stated in the assignment. Real files are checked against these and any gap is reported.
EXPECTED_SPLIT_SIZES = {"train": 10015, "val": 193, "test": 1512}
