import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import cv2
import numpy as np
import tensorflow as tf 
import segmentation_models as sm
import importlib
import glob
from tensorflow.keras.metrics import TrueNegatives, TruePositives, FalseNegatives, FalsePositives
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping, LearningRateScheduler, TensorBoard
from tensorflow.graph_util import convert_variables_to_constants, remove_training_nodes
from segmentation_models import Unet
from segmentation_models.metrics import IOUScore, FScore, Precision, Recall
import time

from model.datasets import Dataset, Dataloader, get_training_augmentation, get_validation_augmentation, visualize

from list_configs import configs
from metrics import *

ALL_MODEL_OUTPUT_PATH = "test_results"

for c in configs:

    CONFIG_NAME = c["name"]
    MODEL_PATH = c["model_path"]
    IMAGE_FOLDER = "/mnt/DATA/GRADUATION_RESEARCH/Workstation_Sources/LaneDetection/UNet/UNet-train/data/test/images"
    GROUND_TRUTH_FOLDER = "/mnt/DATA/GRADUATION_RESEARCH/Workstation_Sources/LaneDetection/UNet/UNet-train/data/test/masks"
    OUTPUT_FOLDER = os.path.join(ALL_MODEL_OUTPUT_PATH, CONFIG_NAME)

    cfg = importlib.import_module('model.list_config.' + CONFIG_NAME)
    tf.keras.backend.set_learning_phase(False)
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    import pathlib
    pathlib.Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

    img_paths = os.listdir(IMAGE_FOLDER)

    print("=== {} ===".format(CONFIG_NAME))
    running_mean_IoU = 0
    running_mean_acc = 0
    total_time = 0
    for img_path in img_paths:

        if not (img_path.endswith("jpg") or img_path.endswith("png")):
            continue

        input_path = os.path.join(IMAGE_FOLDER, img_path)
        mask_path = os.path.join(GROUND_TRUTH_FOLDER, img_path) 
        output_path = os.path.join(OUTPUT_FOLDER, img_path)
        img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (cfg.image_size, cfg.image_size), interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask, (cfg.image_size, cfg.image_size), interpolation=cv2.INTER_NEAREST)

        begin_time = time.time()
        net_input = np.expand_dims(img, axis=0)
        pr_mask = model.predict(net_input)[0]
        pr_mask[pr_mask > 0.5] = 1
        pr_mask[pr_mask < 1] = 0
        pr_mask *= 255
        pr_mask = pr_mask.astype(np.uint8)
        pr_mask = pr_mask.reshape(pr_mask.shape[:2])
        total_time += time.time() - begin_time

        running_mean_IoU += mean_IU(pr_mask, mask)
        running_mean_acc += mean_accuracy(pr_mask, mask)

        # result = np.concatenate((img, pr_mask), axis=1)
        # cv2.imwrite(output_path, pr_mask)

    print("Mean Acc: {}".format(running_mean_acc / len(img_paths)))
    print("Mean IoU: {}".format(running_mean_IoU / len(img_paths)))
    print("Avg. Time: {}".format(total_time / len(img_paths)))

