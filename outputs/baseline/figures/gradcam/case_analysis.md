# Grad-CAM case notes

These test images were chosen after the checkpoint was selected by validation accuracy.
Heatmap layer: `backbone.blocks[-1].norm1`. The CLS token is removed and the patch tokens are reshaped onto the patch grid.
High-response locations below are computed from the heatmap. They are not a medical diagnosis. Check the original image before writing an interpretation.

## Case 1 (correct)
- image ID: ISIC_0034962
- true label: NV
- predicted label: NV
- predicted probability: 1.0000
- second-highest class: MEL (0.0000), margin 1.0000
- observed high-response region: peak at about 22% of the image height and 9% of the width; hottest quadrant is bottom_left (45%); the center 50% holds 32% of the activation mass and the border holds 68%.
- Fill in after looking at the image. Do not write a pathological conclusion before that: does the high response fall on the lesion interior, the lesion border, or a non-lesion region such as a ruler, hair, or black edge?
- Possible misclassification reason: this prediction is correct, so none is listed. Record only whether the visible pattern matches the high-response location.

## Case 2 (correct)
- image ID: ISIC_0034706
- true label: BKL
- predicted label: BKL
- predicted probability: 1.0000
- second-highest class: AKIEC (0.0000), margin 1.0000
- observed high-response region: peak at about 60% of the image height and 16% of the width; hottest quadrant is bottom_left (34%); the center 50% holds 44% of the activation mass and the border holds 56%.
- Fill in after looking at the image. Do not write a pathological conclusion before that: does the high response fall on the lesion interior, the lesion border, or a non-lesion region such as a ruler, hair, or black edge?
- Possible misclassification reason: this prediction is correct, so none is listed. Record only whether the visible pattern matches the high-response location.

## Case 3 (incorrect)
- image ID: ISIC_0034584
- true label: MEL
- predicted label: NV
- predicted probability: 1.0000
- second-highest class: MEL (0.0000), margin 0.9999
- observed high-response region: peak at about 22% of the image height and 22% of the width; hottest quadrant is top_left (41%); the center 50% holds 24% of the activation mass and the border holds 76%.
- Fill in after looking at the image. Do not write a pathological conclusion before that: does the high response fall on the lesion interior, the lesion border, or a non-lesion region such as a ruler, hair, or black edge?
- Possible misclassification reason (check the numbers, then the image; do not invent a histological cause): predicted NV but the true label is MEL. The true class is the second-highest probability, so this is a confusion between nearby classes. The heatmap mass is toward the border, so check whether the model responds to the frame or an artifact.
- Manual note (write this after viewing the original, heatmap, and overlay): 

## Case 4 (incorrect)
- image ID: ISIC_0035436
- true label: BKL
- predicted label: MEL
- predicted probability: 0.4971
- second-highest class: BKL (0.4670), margin 0.0301
- observed high-response region: peak at about 53% of the image height and 34% of the width; hottest quadrant is bottom_right (31%); the center 50% holds 45% of the activation mass and the border holds 55%.
- Fill in after looking at the image. Do not write a pathological conclusion before that: does the high response fall on the lesion interior, the lesion border, or a non-lesion region such as a ruler, hair, or black edge?
- Possible misclassification reason (check the numbers, then the image; do not invent a histological cause): predicted MEL but the true label is BKL. The true class is the second-highest probability, so this is a confusion between nearby classes. The heatmap mass is toward the center, so check whether the center color and border look more like the predicted class.
- Manual note (write this after viewing the original, heatmap, and overlay): 
