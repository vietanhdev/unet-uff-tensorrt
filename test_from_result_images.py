"""
Evaluation code for binary segmentation
"""
import os
import cv2
from imutils import paths
from metrics import *

IMAGE_FOLDER = "/mnt/DATA/GRADUATION_RESEARCH/Workstation_Sources/LaneDetection/UNet/UNet-train/data/test/images"
GROUND_TRUTH_MASK_FOLDER = "/mnt/DATA/GRADUATION_RESEARCH/Workstation_Sources/LaneDetection/UNet/UNet-train/data/test/masks"
PREDICT_MASK_FOLDER = "/mnt/DATA/GRADUATION_RESEARCH/Workstation_Sources/LaneDetection/UNet/UNet-train/data/test/output_448x448_fp32"

images = list(paths.list_images(IMAGE_FOLDER))
images.sort()

image_names = [os.path.basename(i) for i in images]
gt_masks = [os.path.join(GROUND_TRUTH_MASK_FOLDER, i) for i in image_names]
pr_masks = [os.path.join(PREDICT_MASK_FOLDER, i) for i in image_names]

IOUs = []
for i in range(len(gt_masks)):
    gt = cv2.imread(gt_masks[i], cv2.IMREAD_GRAYSCALE)
    pr = cv2.imread(pr_masks[i], cv2.IMREAD_GRAYSCALE)
    IOUs.append(mean_IU(pr, gt))
mean_IOU = sum(IOUs) / len(IOUs)

print("Mean IoU: {}".format(mean_IOU))