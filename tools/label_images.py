import cv2
import imutils
import numpy as np
import os
import time
import shutil

# Global
n_total_images = 0
n_labeled_images = 0
images = []
seg_output_format = "png"
current_img_id = 0
current_img = None
label_image = None
list_points = []

last_left_click_time = 0

image_dir = '/home/anhnv/Downloads/test/images'
log_dir = '/home/anhnv/Downloads/test/masks'

label_image_dir = '/home/anhnv/Downloads/test/masks'

def mark_point(event, x, y, flags, _):
    global list_points
    global current_img_id
    global n_labeled_images
    global last_left_click_time

    if len(list_points) == 0:
        list_points.append([])

    if event == cv2.EVENT_LBUTTONDOWN:
        list_points[-1].append([x, y])
        last_left_click_time = time.time()
    elif event == cv2.EVENT_MBUTTONDOWN:
        if len(list_points[-1]) > 0:
            list_points[-1].pop()

def draw_img():
    global images
    global list_points
    global current_img_id
    global current_img
    if current_img is not None:
        draw = current_img.copy()
        if len(images) > 0:
            # Using cv2.putText() method 
            draw = cv2.putText(draw, "{}/{} = {}".format(n_labeled_images, n_total_images, images[current_img_id]), (50, 50) , cv2.FONT_HERSHEY_SIMPLEX , 1.5, (0,0,255), 4, cv2.LINE_AA)

            if len(list_points) != 0:
                for i in range(len(list_points)):
                    for p in list_points[i]:
                        cv2.circle(draw, tuple(p), 5, (0,0,255), -1)
                    if len(list_points[i]) > 0:
                        draw_points = np.array([list_points[i]], np.int32)
                        cv2.polylines(draw, [draw_points], True, (0, 255, 0), thickness=2)

        cv2.imshow('Image', draw)

def get_new_image():
    global images
    global list_points
    global current_img_id
    global current_img
    global label_image
    global image_dir
    if len(images) == 0:
        print("Empty image list!")
        exit(1)
    if current_img_id < 0:
        current_img_id = 0
    elif current_img_id >= len(images):
        current_img_id = len(images) - 1
    image_path = os.path.join(image_dir, images[current_img_id])
    print(image_path)
    current_img = cv2.imread(image_path)
    label_image = current_img.copy()
    label_image[:,:] = [0, 0, 0]
    list_points = []


def save_img():
    global log_dir
    global label_image
    draw_segment()
    save_label()


def draw_segment():
    global list_points
    global label_image

    if len(list_points) == 0:
        return

    if len(list_points[-1]) > 0:
        label_image = cv2.fillPoly(label_image, np.array([list_points[-1]]), (255, 255, 255))

def save_label():
    label_image_name = '{}.png'.format(os.path.splitext(images[current_img_id])[0])
    label_path_file = os.path.join(log_dir, label_image_name)
    cv2.imwrite(label_path_file, label_image)


if __name__ == '__main__':

    os.makedirs(log_dir, exist_ok=True)
    images = os.listdir(image_dir)
    images.sort(key=lambda x: x[:-4])
    n_total_images = len(images)

    marked_images = os.listdir(log_dir)
    n_labeled_images = len(marked_images)

    # for i in marked_images:
    #     if not i.endswith(".png"):
    #         continue
    #     shutil.copyfile(os.path.join(image_dir, i[:-3] + "jpg"), os.path.join(label_image_dir, i[:-3] + "jpg"))
    # exit(1)

    images = [image_name for image_name in images if image_name.replace(image_name[-3:], seg_output_format) not in marked_images and image_name.endswith(".jpg")]
    

    print(images)

    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Image', mark_point)

    current_img_id = 0
    get_new_image()

    while True:

        draw_img()

        key = cv2.waitKey(1)
        if not key:
            continue

        if key == ord("q"):
            break
        elif key == ord(" "):
            save_img()
            current_img_id += 1
            get_new_image()
            draw_img()
            n_labeled_images += 1
        elif key == ord("z"):
            current_img_id -= 1
            if current_img_id >= 0:
                n_labeled_images -= 1
            get_new_image()
            draw_img()
        elif key == ord("n"):
            draw_segment()
            list_points.append([])