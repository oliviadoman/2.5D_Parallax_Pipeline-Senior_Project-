import cv2
import numpy as np

mask_0 = cv2.imread('masks/DepthAnythingV2_layer_0.png', cv2.IMREAD_GRAYSCALE)
mask_1 = cv2.imread('masks/DepthAnythingV2_layer_1.png', cv2.IMREAD_GRAYSCALE)
mask_2 = cv2.imread('masks/DepthAnythingV2_layer_2.png', cv2.IMREAD_GRAYSCALE)

# Cumulative masks for inpainting
inpaint_mask_layer0 = cv2.bitwise_or(mask_1, mask_2)  # MG + FG
inpaint_mask_layer1 = mask_2                           # FG only
# layer 2 needs no inpainting mask