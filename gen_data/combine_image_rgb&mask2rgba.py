import cv2
import os
import numpy as np
from glob import glob
from multiprocessing import Pool
from tqdm import tqdm
from PIL import Image


def imageRGBAndMask2RGBA(path):
    image_path, images_dir, masks_dir, output_dir_save_rgba_image = path
    image_name = os.path.basename(image_path)[:-4]
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    mask_path = image_path.replace(images_dir, masks_dir)
    mask = cv2.imread(mask_path, 0)
    _, mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)
    b,g,r = cv2.split(image)
    b = np.where(mask > 100, b, 0)
    g = np.where(mask > 100, g, 0)
    r = np.where(mask > 100, r, 0)
    rgba_image = cv2.merge((b, g, r, mask))
    image_rgba_output_path = os.path.join(output_dir_save_rgba_image, '{}.png'.format(image_name))
    cv2.imwrite(image_rgba_output_path, rgba_image)


def removeBorderTranparency(path):
    rgba_image_input_path, rgba_crop_image_output_dir = path
    img = Image.open(rgba_image_input_path)
    img.load()

    result = np.where(np.array(img) != 0)
    left = min(result[1])
    right = max(result[1])
    upper = min(result[0])
    lower = max(result[0])
    crop_img = img.crop((left, upper, right, lower))
    width, height = crop_img.size
    if width > height:
        resized_img_width = 1280
        resized_img_height = int(height * (1280 / width))
    else:
        resized_img_height = 1280
        resized_img_width = int(width * (1280 / height))
    resized_img = crop_img.resize((resized_img_width, resized_img_height), Image.ANTIALIAS)
    image_name = os.path.basename(rgba_image_input_path)
    resized_img.save(os.path.join(rgba_crop_image_output_dir, image_name))


if __name__ == '__main__':
    images_dir = '/media/anhdvh/WD/data/poc/Receipt/data_train_05_02_2020/images'
    masks_dir = '/media/anhdvh/WD/data/poc/Receipt/data_train_05_02_2020/masks'
    output_dir_save_rgba_image = '/media/anhdvh/WD/data/poc/Receipt/data_train_05_02_2020/rgba'
    os.makedirs(output_dir_save_rgba_image, exist_ok=True)
    output_dir_save_rgba_crop_image = '/media/anhdvh/WD/data/poc/Receipt/data_train_05_02_2020/rgba_crop'
    os.makedirs(output_dir_save_rgba_crop_image, exist_ok=True)

    list_image_path = glob(os.path.join(images_dir, '*'))
    list_image_path.sort(key=lambda x: int(os.path.basename(x)[:-4]))
    number_of_images = len(list_image_path)
    p = Pool(3)

    for _ in tqdm(p.imap_unordered(imageRGBAndMask2RGBA,
                                   zip(
                                       list_image_path,
                                       [images_dir] * number_of_images,
                                       [masks_dir] * number_of_images,
                                       [output_dir_save_rgba_image] * number_of_images,
                                   )), total=number_of_images):
        pass


    list_rgba_images_path = glob(os.path.join(output_dir_save_rgba_image, '*'))
    list_rgba_images_path.sort(key=lambda x: int(os.path.basename(x)[:-4]))

    for _ in tqdm(p.imap_unordered(removeBorderTranparency,
                                   zip(
                                       list_rgba_images_path,
                                       [output_dir_save_rgba_crop_image]*number_of_images,
                                   )), total=number_of_images):
        pass
    p.terminate()