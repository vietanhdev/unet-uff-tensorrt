export PYTHONPATH=.
# python gen_data/main_generator.py --output_dir data/gen_test --n_images 100
python gen_data/main_generator.py --output_dir data/train_2 --n_images 5000
# python gen_data/main_generator.py --output_dir data/val --n_images 500
# python gen_data/main_generator.py --output_dir data/test --n_images 100