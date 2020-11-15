import os
import random
from tqdm import tqdm
from multiprocessing import Pool
from process_generator import ProcessGenerator
import re
import pathlib
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--n_images', type=int, required=True)
args = parser.parse_args()

images_dir = 'data/generator/raw'
shadow_dir = 'data/generator/mask_brightness'
background_dir = 'data/generator/background'
output_dir = args.output_dir
n_images = args.n_images


os.makedirs(output_dir, exist_ok=True)
base_output = os.path.join(output_dir, 'images')
base_mask_output = os.path.join(output_dir, 'masks')

def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower()
            for text in _nsre.split(s)]


def create_image_input_path(images_dir, no_image):
    list_image_paths = []
    image_paths = [os.path.join(images_dir, f) for f in sorted(os.listdir(images_dir), key = natural_sort_key)]
    print(image_paths)
    for _ in tqdm(range(no_image)):
        index = random.randint(0, len(image_paths) - 1)
        image_path = image_paths[index]
        list_image_paths.append(image_path)

    return list_image_paths

def main(output_dir, n_images):

    no = n_images

    p = Pool(10)
    process_generator = ProcessGenerator()
    list_image_paths = create_image_input_path(images_dir, no)
    list_shadow_mask = create_image_input_path(shadow_dir, no)

    pathlib.Path(base_output).mkdir(parents=True, exist_ok=True)
    pathlib.Path(base_mask_output).mkdir(parents=True, exist_ok=True)

    for _ in tqdm(p.imap_unordered(process_generator.generate_from_tuple,
                                   zip(
                                       [i for i in range(no)],
                                       list_image_paths,
                                       [base_output] * no,
                                       [base_mask_output] * no,
                                       list_shadow_mask,
                                       [background_dir] * no
                                   )), total=no):
        pass
    p.terminate()


if __name__ == '__main__':
    main(output_dir, n_images)