import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import ast


class BoundingBoxA:
    def __init__(self, x_min, y_min, x_max, y_max, width, height):
        self.width = width
        self.height = height
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max


class CaptchaField:
    def __init__(self, x_min, y_min, x_max, y_max):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.width = x_max - x_min
        self.height = y_max - y_min


def get_intersections_for_all_images(pred_matrix_dir, img_dir, test_dir, show_result=True, img_out_dir=None, matrices_out_dir=None):
    for pred_matrix_txt_file in [file for file in os.listdir(test_dir) if file.endswith('.txt')]:

        with open(f'{os.path.join(pred_matrix_dir, pred_matrix_txt_file)}') as txt_file:
            pred_matrix = txt_file.read()
            pred_matrix = ast.literal_eval(pred_matrix)

            print(os.path.join(img_dir, f'{pred_matrix_txt_file[:-4]}.png'))
            image = cv2.imread(os.path.join(img_dir, f'{pred_matrix_txt_file[:-4]}.png'))
            image = cv2.resize(image, (600, 600))
            print((image.shape[0]/4))

            for ind_row, row in enumerate(pred_matrix):
                for ind_col, col in enumerate(row):
                    if pred_matrix[ind_row][ind_col] is True:
                        x, y, w, h = ind_col * int((image.shape[0]/4)), ind_row * int((image.shape[0]/4)), int((image.shape[0]/4) - 1), int((image.shape[0]/4) - 1)
                        mask = np.zeros_like(image)
                        cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 255, 255), -1)

                        white_mask = np.ones_like(image) * 255
                        opacity = 0.25

                        # Blend the border mask with the image
                        image_with_border = cv2.addWeighted(image, 1 - opacity, mask, opacity, 0)

                        # Blend the mask with the image
                        image_masked = cv2.addWeighted(image_with_border, 1 - opacity, white_mask, opacity, 0)
                        image_masked[mask == 0] = image[mask == 0]
                        image = image_masked
            for number in [int((image.shape[0]/4)), int(2*(image.shape[0]/4)), int(3*(image.shape[0]/4))]:
                cv2.line(image, (number, 0), (number, image.shape[0]), (255, 255, 255), 2)
                cv2.line(image, (number - 1, 0), (number - 1, image.shape[0]), (255, 255, 255), 2)
                cv2.line(image, (0, number), (image.shape[0], number), (255, 255, 255), 2)
                cv2.line(image, (0, number - 1), (image.shape[0], number - 1), (255, 255, 255), 2)

            if show_result:
                plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                plt.title(pred_matrix_txt_file)
                plt.show()
            if img_out_dir is not None:
                cv2.imwrite(os.path.join(img_out_dir, f'{pred_matrix_txt_file[:-4]}.png'), image)
                # plt.imsave(os.path.join(img_out_dir, f'{pred_matrix_txt_file[:-4]}.png'), image)
            if matrices_out_dir is not None:
                with open(os.path.join(matrices_out_dir, f'{pred_matrix_txt_file[:-4]}.txt'), 'w') as f:
                    f.write(f'{pred_matrix}')


def do_boxes_intersect(a, b):
    return ((abs((a.x_min + a.width / 2) - (b.x_min + b.width / 2)) * 2 < (a.width + b.width)) and
            (abs((a.y_min + a.height / 2) - (b.y_min + b.height / 2)) * 2 < (a.height + b.height)))


# xml_directory = os.path.join('neural_networks', 'Tensorflow', 'workspace', 'models', 'efficientdet_d0_coco17_tpu-32', 'v1', 'predictions', 'annotations')
pred_matrix_directory = r'C:\Users\janku\PycharmProjects\Maturitaetsarbeit\Python_Code\Solve_Manually\person_matrices\matrix_output_person1'
# pred_matrix_directory = r'C:\Users\janku\PycharmProjects\Maturitaetsarbeit\Python_Code\train_val_test_split\captcha_matrices'
test_directory = r'C:\Users\janku\PycharmProjects\Maturitaetsarbeit\Python_Code\Solve_Manually\person_matrices\matrix_output_person1'
images_directory = os.path.join('..', 'dataset', 'Crosswalk')
# images_output_directory = os.path.join('neural_networks', 'Tensorflow', 'workspace', 'models', 'efficientdet_d0_coco17_tpu-32', 'v2', 'predictions', 'imgs_captcha_style')
images_output_directory = r'C:\Users\janku\Downloads'
# matrices_output_directory = os.path.join('neural_networks', 'Tensorflow', 'workspace', 'models', 'efficientdet_d0_coco17_tpu-32', 'v1', 'predictions', 'anns_captcha_style')
matrices_output_directory = None

if matrices_output_directory is not None:
    os.makedirs(f'{matrices_output_directory}', exist_ok=True)

if images_output_directory is not None:
    os.makedirs(f'{images_output_directory}', exist_ok=True)

get_intersections_for_all_images(pred_matrix_directory, images_directory, test_directory, True, images_output_directory,
                                 matrices_output_directory)

