from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2

sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
mask_generator = SamAutomaticMaskGenerator(sam)
image = cv2.imread("original.png")
masks = mask_generator.generate(image)
# masks is a list of dicts, each with 'segmentation', 'area', 'bbox' etc.