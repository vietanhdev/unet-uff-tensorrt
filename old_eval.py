import os
# os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import cv2
import numpy as np
import tensorflow as tf 
import segmentation_models as sm
import importlib
import albumentations as A
from model.datasets import Dataset, Dataloader, get_training_augmentation, get_validation_augmentation, visualize

CONFIG_NAME = "config05"
MODEL_PATH = "/mnt/DATA/GRADUATION_RESEARCH/Workstation_Sources/LaneDetection/UNet/UNet-train/saved_models/config05/model_.116-0.024944.h5"
TEST_IMAGES = ["./data/test/images"]
TEST_MASKS = ["./data/test/masks"]

cfg = importlib.import_module('model.list_config.' + CONFIG_NAME)

tf.keras.backend.set_learning_phase(False)
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
model.compile(
    'Adam',
    loss=sm.losses.bce_jaccard_loss,
    metrics=metrics,
)

def test_preprocessing(image, mask):
    sample = {}
    image = cv2.resize(image, (cfg.image_size, cfg.image_size), interpolation=cv2.INTER_NEAREST)
    mask = cv2.resize(mask, (cfg.image_size, cfg.image_size), interpolation=cv2.INTER_NEAREST)
    sample["image"] = image
    sample["mask"] = mask
    # cv2.imshow("image", image)
    # cv2.imshow("mask", mask)
    # cv2.waitKey(0)
    return sample

# Dataset for testation images
test_dataset = Dataset(
    TEST_IMAGES, 
    TEST_MASKS, 
    classes=cfg.CLASSES,
    preprocessing=test_preprocessing
)
test_dataloader = Dataloader(test_dataset, batch_size=1, shuffle=False)

scores = model.evaluate_generator(test_dataloader, verbose=1)
print("Loss: {:.5}".format(scores[0]))
for metric, value in zip(metrics, scores[1:]):
    print("mean {}: {:.5}".format(metric.__name__, value))