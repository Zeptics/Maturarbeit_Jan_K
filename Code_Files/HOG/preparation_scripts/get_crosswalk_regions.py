import os
import xml.etree.ElementTree as ET

import cv2
from tqdm import tqdm


def read_voc_xml(xmlfile: str) -> dict:
    root = ET.parse(xmlfile).getroot()
    boxes = {'filename': root.find('filename').text,
             'objects': []}
    for box in root.iter('object'):
        bb = box.find('bndbox')
        obj = {
            'name': box.find('name').text,
            'xmin': int(bb.find('xmin').text),
            'ymin': int(bb.find('ymin').text),
            "xmax": int(bb.find("xmax").text),
            "ymax": int(bb.find("ymax").text),
        }
        boxes['objects'].append(obj)
    return boxes


def square_image(image, final_size):
    # Get the dimensions
    height, width = image.shape[:2]

    # Determine the size of the square
    square_size = min(height, width)

    # Calculate top left corner
    start_x = (width - square_size) // 2
    start_y = (height - square_size) // 2

    # Crop the square
    square_img = image[start_y:start_y + square_size, start_x:start_x + square_size]
    # Resize to right size
    resized_img = cv2.resize(square_img, final_size)

    return resized_img


def extract_crosswalk_regions(xml_files_dir, img_files_dir, square_img_output_dir, final_size):
    # iterate trough every xml file in root directory
    for xml_file in tqdm([file for file in os.listdir(xml_files_dir) if file.endswith('.xml')]):
        bbox_counter = 0
        xml_file_dict = read_voc_xml(f'{xml_files_dir}\\{xml_file}')
        # get all bounding boxes of img
        img = cv2.imread(os.path.join(img_files_dir, f'{xml_file[:-4]}.png'))  # f'{img_files_dir}\\{xml_file}.png'
        # print(f'{img_files_dir}\\{xml_file[:-4]}.png')
        for bbox in xml_file_dict['objects']:
            bbox_counter += 1
            # print(bbox)
            xmin, ymin, xmax, ymax = bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']
            # print(xmin, ymin, xmax, ymax)
            portion_in_bbox = img[ymin:ymax, xmin:xmax]
            square_img = square_image(portion_in_bbox, final_size)
            output_path = os.path.join(square_img_output_dir, f'{xml_file[:-4]}{bbox_counter}.png')
            # print(output_path)
            cv2.imwrite(output_path, square_img)


xml_files_dir = os.path.join(os.getcwd(), '..', 'dataset', 'train_val_test_split', 'annotations', 'train_bb')
img_files_dir = os.path.join(os.getcwd(), '..', 'train_val_test_split', 'images', 'train')
square_img_output_dir = os.path.join(os.getcwd(), 'data', 'images', 'square_crosswalk_regions')
os.makedirs(square_img_output_dir, exist_ok=True)

final_size = (32, 32)

extract_crosswalk_regions(xml_files_dir, img_files_dir, square_img_output_dir, final_size)
