import cv2
import tensorflow as tf
import numpy as np
import albumentations as A
import os
import matplotlib.pyplot as plt


def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(64, 20))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    # plt.show()
    plt.savefig('plot.png')


# classes for data loading and preprocessing
class Dataset:
    """Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        image_dir (str): path to images folder
        mask_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    def __init__(
            self, 
            list_image_dirs, 
            list_mask_dirs,
            mask_format = "png",
            classes=None,
            augmentation=None, 
            preprocessing=None,
    ):
        self.images_fps = []
        self.masks_fps = []
        for image_dir, mask_dir in zip(list_image_dirs, list_mask_dirs):
            image_ids = os.listdir(image_dir)
            image_ids.sort()

            mask_ids = [ "{}.{}".format(os.path.splitext(image_id)[0], mask_format) for image_id in image_ids]
            self.images_fps.extend([os.path.join(image_dir, image_id) for image_id in image_ids])
            self.masks_fps.extend([os.path.join(mask_dir, mask_id) for mask_id in image_ids])
        
        # convert str names to class values on masks
        self.class_values = [idx for cls, idx in classes.items()]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images_fps[i])
        assert image is not None, self.images_fps[i] + " image is NONE"
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i])
        assert mask is not None, self.masks_fps[i] + " mask is NONE"
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        
        # extract certain classes from mask (e.g. cars)
        masks = [(np.all(mask == v, axis=-1)) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        
        # add background if mask is not binary
        if mask.shape[-1] != 1:
            background = 1 - mask.sum(axis=-1, keepdims=True)
            mask = np.concatenate((mask, background), axis=-1)
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # cv2.imshow("Debug", image)
        # cv2.waitKey(0)
            
        return image, mask
        
    def __len__(self):
        return len(self.images_fps)
    
    
class Dataloader(tf.keras.utils.Sequence):
    """Load data from dataset and form batches
    
    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
    """
    
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i):
        
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])
        
        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        
        return batch[0], batch[1]
    
    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size


def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)


# define heavy augmentations
def get_training_augmentation(size):
    train_transform = [
        A.LongestMaxSize(max_size=size, always_apply=True),
        A.PadIfNeeded(min_height=size, min_width=size, always_apply=True, border_mode=0),
        # A.RandomCrop(height=size, width=size, always_apply=True),

        # A.VerticalFlip(p=0.5),
        # A.HorizontalFlip(p=0.5),            
        # A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(scale_limit=0.2, rotate_limit=0, shift_limit=0.2, p=0.1, border_mode=0),
        A.IAAPerspective(p=0.1),

        A.CoarseDropout(p=0.1),
        A.ChannelDropout(p=0.1),
        A.RGBShift(p=0.1),

        A.OneOf(
            [
                A.OpticalDistortion(p=0.5),
                A.GridDistortion(p=0.5)
            ],
            p=0.1,
        ), 

        A.OneOf(
            [
                A.CLAHE(p=0.5),
                A.RandomBrightness(p=0.5),
                A.RandomGamma(p=0.5),
            ],
            p=0.5,
        ),

        A.OneOf(
            [
                A.GaussianBlur(p=0.1),
                A.IAASharpen(p=0.5),
                A.Blur(blur_limit=5, p=0.5),
                A.MotionBlur(blur_limit=5, p=0.5),
            ],
            p=0.5,
        ),

        A.OneOf(
            [
                A.RandomContrast(p=0.5),
                A.HueSaturationValue(p=0.5),
            ],
            p=0.1,
        ),

        A.Lambda(mask=round_clip_0_1),
        A.Cutout(num_holes=8, max_h_size=20, max_w_size=20, fill_value=0, always_apply=False, p=0.2),
        A.CoarseDropout(max_holes=8, max_height=20, max_width=20, min_holes=None, min_height=None, min_width=None, fill_value=0, always_apply=False, p=0.2),
        # A.GlassBlur(sigma=0.7, max_delta=4, iterations=2, always_apply=False, mode='fast', p=0.2)


    ]
    return A.Compose(train_transform)


def get_validation_augmentation(size):
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        A.LongestMaxSize(max_size=size, always_apply=True),
        A.PadIfNeeded(min_height=size, min_width=size, always_apply=True, border_mode=0)
    ]
    return A.Compose(test_transform)