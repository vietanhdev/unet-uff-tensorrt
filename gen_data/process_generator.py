import os
import numpy as np
import random
from PIL import Image
from gen_data.background_generator import BackgroundGenerator
import cv2
import random

from PIL import Image, ImageDraw, ImageEnhance
import numpy
import imutils

def find_coeffs(pa, pb):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

    A = numpy.matrix(matrix, dtype=numpy.float)
    B = numpy.array(pb).reshape(8)

    res = numpy.dot(numpy.linalg.inv(A.T * A) * A.T, B)
    return numpy.array(res).reshape(8)

def random_perspective_transform(img):
    width_input = img.size[0]
    height_input = img.size[1]

    padding_height = int(height_input * 0.1)
    padding_width = int(width_input * 0.1)
    a = random.randint(0, padding_height)
    b = random.randint(0, padding_width)
    c = random.randint(0, padding_height)
    d = random.randint(0, padding_width)

    coeffs = find_coeffs(
        [(a, 0), (height_input-a, b), (height_input, width_input-b), (0, width_input)],
        [(0, 0), (height_input, 0), (height_input, width_input), (0, width_input)]
    )

    return img.transform((width_input, height_input), Image.PERSPECTIVE, coeffs, Image.BICUBIC)

def add_corners(im, rad):
    circle = Image.new('L', (rad * 2, rad * 2), 0)
    draw = ImageDraw.Draw(circle)
    draw.ellipse((0, 0, rad * 2, rad * 2), fill=255)
    alpha = Image.new('L', im.size, 255)
    w, h = im.size
    alpha.paste(circle.crop((0, 0, rad, rad)), (0, 0))
    alpha.paste(circle.crop((0, rad, rad, rad * 2)), (0, h - rad))
    alpha.paste(circle.crop((rad, 0, rad * 2, rad)), (w - rad, 0))
    alpha.paste(circle.crop((rad, rad, rad * 2, rad * 2)), (w - rad, h - rad))
    im.putalpha(alpha)
    return im


import colorsys
rgb_to_hsv = np.vectorize(colorsys.rgb_to_hsv)
hsv_to_rgb = np.vectorize(colorsys.hsv_to_rgb)

def shift_hue(arr, hout):
    r, g, b, a = np.rollaxis(arr, axis=-1)
    h, s, v = rgb_to_hsv(r, g, b)
    h = hout
    r, g, b = hsv_to_rgb(h, s, v)
    arr = np.dstack((r, g, b, a))
    return arr

def colorize(image, hue):
    """
    Colorize PIL image `original` with the given
    `hue` (hue within 0-360); returns another PIL image.
    """
    img = image.convert('RGBA')
    arr = np.array(np.asarray(img).astype('float'))
    new_img = Image.fromarray(shift_hue(arr, hue/360.).astype('uint8'), 'RGBA')

    return new_img


class ProcessGenerator(object):

    @classmethod
    def generate_from_tuple(cls, t):
        cls.gen_image(*t)


    @classmethod
    def gen_image(cls, index, image_input_path, base_output, base_mask_output, shadow_mask_path, background_dir):

        no_fd = (index // 1000)

        png = Image.open(image_input_path)
        png.load()

        # Use background as card
        if random.random() < 0.05:
            png = BackgroundGenerator.picture(random.randint(300, 400), random.randint(200, 250), background_dir)
            png = png.convert('RGBA')

        png = png.convert('RGBA')

        if random.random() < 0.5:
            png = add_corners(png, random.randint(0, 30))

        if random.random() < 0.5:
            png = random_perspective_transform(png)

        if random.random() < 0.5:
            factor = 0.2 + random.random() * 2
            enhancer = ImageEnhance.Brightness(png)
            png = enhancer.enhance(factor)

        if random.random() < 0.8:
            png = colorize(png, random.randint(0, 360))

        angle = random.choice(np.arange(0, 360, 30))
        png = png.rotate(angle, expand=True)
        width_input = png.size[0]
        height_input = png.size[1]

        png_mask = ProcessGenerator.image_rgba_2_mask(png)
        ratio = random.choice([1, 1, 0, 0, 0])
        if ratio == 1:
            ratio_width = random.randrange(100, 140) / 100
            ratio_height = random.randrange(100, 140) / 100
        else:
            ratio_width = random.randrange(160, 200) / 100
            ratio_height = random.randrange(160, 200) / 100

        width_bg = int(width_input * ratio_width)
        height_bg = int(height_input * ratio_height)
        mask_bg = Image.fromarray(np.zeros((height_bg, width_bg)))
        bg = BackgroundGenerator.picture(width_bg, height_bg, background_dir)
        bg = bg.convert('RGB')

        x = random.randint(0, width_bg - width_input)
        y = random.randint(0, height_bg - height_input)
        bg.paste(png, (x, y), png)
        mask_bg.paste(png_mask, (x, y), png_mask)
        image_output_name = '{}_{}_{}_{}_{}.png'.format(index, x, y, x + width_input, y + height_input)
        threshold = 100
        mask_bg = mask_bg.convert('L')
        mask_bg = mask_bg.point(lambda p: p > threshold and 255)
        blur = random.choice([0, 1])
        if blur == 1:
            bg = ProcessGenerator.blur_image(bg)
        shadow = random.choice([0, 0, 0, 0, 0, 1])
        if shadow == 1:
            bg = ProcessGenerator.add_shadow(bg, shadow_mask_path)

        mask_bg = ProcessGenerator.morphology_image(mask_bg)

        bg = bg.convert('RGB')
        bg = np.array(bg) 
        bg = imutils.resize(bg, width=512, inter=cv2.INTER_NEAREST)
        bg = cv2.cvtColor(bg, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(base_output, image_output_name), bg)

        mask_bg = mask_bg.convert('RGB')
        mask_bg = np.array(mask_bg) 
        mask_bg = imutils.resize(mask_bg, width=512, inter=cv2.INTER_NEAREST)
        mask_bg = cv2.cvtColor(mask_bg, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(base_mask_output, image_output_name), mask_bg)

        # cv2.imshow("Image", bg)
        # cv2.waitKey(0)

    @staticmethod
    def image_rgba_2_mask(rgba_image):
        rgba_image = np.array(rgba_image)
        image_channel_a = rgba_image[:, :, 3]
        mark = np.full_like(image_channel_a, 255)
        mask = np.where(image_channel_a > 0, mark, mark * 0)
        mask = Image.fromarray(mask)
        return mask


    @staticmethod
    def add_shadow(image, mask_path):
        ## Conversion to HLS
        mask = cv2.imread(mask_path, 0)
        image = np.array(image)
        image_HSV = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(image_HSV)
        height, width, _ = image.shape
        mask = cv2.resize(mask, (width, height))
        ratio = random.uniform(0.5, 1.5)
        points = np.argwhere(mask > 0)
        for point in points:
            temp = v[point[0], point[1]] * ratio
            if temp > 255:
                v[point[0], point[1]] = 255
            else:
                v[point[0], point[1]] = temp
        v = v.astype('uint8')
        ## Conversion to RGB
        final_hsv = cv2.merge((h, s, v))
        final_hsv = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
        final_hsv = Image.fromarray(final_hsv)
        return final_hsv


    @staticmethod
    def blur_image(src):
        src = np.array(src)
        dst = cv2.GaussianBlur(src, (3, 3), cv2.BORDER_DEFAULT)
        blur_image = Image.fromarray(dst)
        return blur_image


    @staticmethod
    def morphology_image(src):
        src = np.array(src)
        dst = cv2.dilate(src, kernel=np.ones((3, 3)), iterations=2)
        dst = cv2.erode(dst, kernel=np.ones((3, 3)), iterations=2)
        image_morphology = Image.fromarray(dst)
        return image_morphology