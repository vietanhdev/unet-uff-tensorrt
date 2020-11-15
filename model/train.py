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
from segmentation_models.metrics import IOUScore, FScore, Precision, Recall
from models import get_model
from segmentation_models.utils import set_trainable
from datasets import Dataset, Dataloader, get_training_augmentation, get_validation_augmentation, visualize
from utils import resize_with_padding

print("tensorflow version: {}".format(tf.__version__))
sm.set_framework('tf.keras')

list_config = [x.split('/')[-1].replace('.py', '') for x in glob.glob('model/list_config/[!__init__]*.py')]
list_config.sort()


def scheduler(epoch):
    if epoch < 50:
        return 0.01
    elif epoch < 100:
        return 0.001
    elif epoch < 150:
        return 0.0001
    else:
        return 0.00001


for x in list_config:

    cfg = importlib.import_module('list_config.'+x)
    print('>>>>>>>>>> Prepare to train with file config {} is {}'.format(x, cfg.TRAIN))
    if cfg.TRAIN==False:
        continue
    print(cfg.description)
    n_classes = 1 if len(cfg.CLASSES) == 1 else (len(cfg.CLASSES) + 1)

    def train_preprocessing(image, mask):
        sample = {}
        # image = resize_with_padding(image, cfg.image_size, interpolation=cv2.INTER_LINEAR)
        # mask = resize_with_padding(mask, cfg.image_size, interpolation=cv2.INTER_NEAREST)
        # mask.reshape((cfg.image_size, cfg.image_size, n_classes))
        sample["image"] = image
        sample["mask"] = mask
        return sample


    # Dataset for train images
    train_dataset = Dataset(
        cfg.list_x_train_dirs, 
        cfg.list_y_train_dirs,
        classes=cfg.CLASSES, 
        augmentation=get_training_augmentation(cfg.image_size),
        preprocessing=train_preprocessing
    )

    # Dataset for validation images
    valid_dataset = Dataset(
        cfg.list_x_valid_dirs, 
        cfg.list_y_valid_dirs, 
        classes=cfg.CLASSES, 
        augmentation=get_validation_augmentation(cfg.image_size),
        preprocessing=train_preprocessing
    )

    train_dataloader = Dataloader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    valid_dataloader = Dataloader(valid_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)

    # check shapes for errors
    assert train_dataloader[0][0].shape == (cfg.BATCH_SIZE, cfg.image_size, cfg.image_size, 3)
    assert train_dataloader[0][1].shape == (cfg.BATCH_SIZE, cfg.image_size, cfg.image_size, n_classes)

    loss = cfg.loss
    optim = tf.keras.optimizers.SGD(lr=cfg.LR, momentum=0.9, nesterov=True)
    metrics = [
        IOUScore(threshold=cfg.metric_threshold, per_image=cfg.metric_per_image), 
        FScore(threshold=cfg.metric_threshold, per_image=cfg.metric_per_image), 
        Precision(threshold=cfg.metric_threshold, per_image=cfg.metric_per_image), 
        Recall(threshold=cfg.metric_threshold, per_image=cfg.metric_per_image),
        TruePositives(thresholds=cfg.metric_threshold), 
        TrueNegatives(thresholds=cfg.metric_threshold), 
        FalsePositives(thresholds=cfg.metric_threshold),
        FalseNegatives(thresholds=cfg.metric_threshold)
    ]

    model = get_model(cfg, training=True)
    model.compile(optim, loss, metrics)


    callbacks = [
        ModelCheckpoint(
            cfg.base_model_path + ".{epoch:03d}-{val_loss:.6f}.h5", 
            monitor='val_loss', 
            # verbose=1, 
            save_best_only=(not cfg.SAVE_ALL_MODELS),
            save_weights_only=False,
            mode='min'),
        ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.1,
            patience=10, 
            verbose=1,
            min_delta=0.0001,
            cooldown=0,
            min_lr=0),
        CSVLogger(cfg.log_file),
        EarlyStopping(
            monitor='val_loss', 
            min_delta=0.0001, 
            patience=15, 
            verbose=1,
            mode='min',
            baseline=None
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=cfg.tensorboard_log_dir
        )
    ]

    model.fit_generator(
        train_dataloader, 
        steps_per_epoch=len(train_dataloader),
        epochs=cfg.EPOCHS,
        shuffle=True,
        callbacks=callbacks, 
        validation_data=valid_dataloader, 
        validation_steps=len(valid_dataloader),
        max_queue_size=48,
        workers=6
    )
