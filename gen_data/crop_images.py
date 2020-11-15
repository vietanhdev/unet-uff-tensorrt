import os
import random
import re
from glob import glob
from multiprocessing import Pool

import cv2
import numpy as np
from tqdm import tqdm


class data_aug(object):
    @classmethod
    def data_aug(cls, t):
        cls.augment(*t)

    @classmethod
    def augment(cls, image_path, image_output_dir, mask_output_dir):
        image_name = os.path.basename(image_path)
        pattern = re.compile(r"/images/")
        mask_path = pattern.sub(r"/mask/", image_path)
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(mask_path, 0)
        result = np.where(mask > 0)
        h, w = image.shape[:2]
        add = random.choice([5,10,15,20])
        x_min = max([min(result[1]) - add, 0])  # min in width
        x_max = min([max(result[1]) + add, w])  # max in width
        y_min = max([min(result[0]) - add, 0])  # min in height
        y_max = min([max(result[0]) + add, h])  # max in height
        c_h = y_max - y_min
        c_w = x_max - x_min
        top = 0
        left = 0
        bottom = h
        right = w
        crop_top = random.choice([True, False])
        if crop_top:
            ratio_top = random.uniform(0, 0.15)
            top = y_min + int(c_h * ratio_top)
        crop_left = random.choice([True, False])
        if crop_left:
            ratio_left = random.uniform(0, 0.15)
            left = x_min + int(c_w * ratio_left)
        crop_bottom = random.choice([True, False])
        if crop_bottom:
            ratio_bottom = random.uniform(0, 0.15)
            bottom = y_max - int(c_h * ratio_bottom)
        crop_right = random.choice([True, False])
        if crop_right:
            ratio_right = random.uniform(0, 0.15)
            right = x_max - int(c_w * ratio_right)

        image_crop = image[top:bottom, left:right]
        mask_crop = mask[top:bottom, left:right]
        cv2.imwrite(os.path.join(image_output_dir, image_name), image_crop)
        cv2.imwrite(os.path.join(mask_output_dir, image_name), mask_crop)


def main():
    base_input = '/media/ai-team/HDD/anhdhv/POC/softbank/data_train/images'
    list_image_dir = glob(os.path.join(base_input, '*'))
    extra = len(list_image_dir)
    for images_dir in list_image_dir[:3]:
        p = Pool(8)
        name_dir = int(os.path.basename(images_dir)) + extra
        image_output_dir = os.path.join(base_input, str(name_dir))
        os.makedirs(image_output_dir, exist_ok=True)
        pattern = re.compile(r"/images/")
        mask_output_path = pattern.sub(r"/mask/", image_output_dir)
        os.makedirs(mask_output_path, exist_ok=True)
        images_list_path = glob(os.path.join(images_dir, '*'))
        no = len(images_list_path)

        for _ in tqdm(p.imap_unordered(data_aug.data_aug,
                                       zip(
                                           images_list_path,
                                           [image_output_dir] * no,
                                           [mask_output_path] * no,
                                       )), total=no):
            pass
        p.terminate()


if __name__ == '__main__':
    main()