import sys
import os
import cv2
import shutil
import numpy as np

input_folder = "/home/anhnv/Workspace/CARD/data_gen/data/raw"
output_folder = "/home/anhnv/Workspace/CARD/data_gen/data/white"

img_paths = os.listdir(input_folder)
for img_path in img_paths:
    input_path = os.path.join(input_folder, img_path)
    output_path = os.path.join(output_folder, img_path)
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    _, img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
    non_0 = cv2.countNonZero(img)
    
    if non_0 < 50:
        shutil.move(input_path, output_path)