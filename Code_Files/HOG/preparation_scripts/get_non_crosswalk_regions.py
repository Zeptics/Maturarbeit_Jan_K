import os
import xml.etree.ElementTree as ET

import cv2
import numpy as np
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


def find_non_crosswalk_region(image, bboxes, size, final_size):
    attempts = 0
    max_attempts = 100
    while attempts < max_attempts:
        # Generate random coordinates for the top-left corner of the region
        x = np.random.randint(0, image.shape[1] - size)
        y = np.random.randint(0, image.shape[0] - size)

        # Check if the region intersects with any of the bounding boxes
        intersects = False
        for bbox in bboxes:
            xmin, ymin, xmax, ymax = bbox
            bbox_x, bbox_y = xmin, ymin
            bbox_w, bbox_h = xmax - xmin, ymax - ymin
            if (x < bbox_x + bbox_w and x + size > bbox_x and
                y < bbox_y + bbox_h and y + size > bbox_y):
                intersects = True
                break

        # If the region doesn't intersect with any bounding box, return it
        if not intersects:
            if size != final_size:
                # Scale image to 64x64 if its size isn't already 64x64
                scaled_image = cv2.resize(image[y:y+size, x:x+size], (final_size, final_size))
                return scaled_image
            return image[y:y+size, x:x+size]

        attempts += 1

    # No suitable region found within the specified size and attempts
    return None


def extract_non_crosswalk_regions(image_dir, labels_dir, output_dir, final_size):
    # Iterate over each file in the image directory
    for file in tqdm([file for file in os.listdir(image_dir) if file.endswith('.png')]):
        # define bboxes list
        bboxes = []
        # Read the image
        image_path = os.path.join(image_dir, file)
        image = cv2.imread(image_path)

        # Get path for corresponding label file
        label_path = os.path.join(labels_dir, f'{file[:-4]}.xml')

        # Get bounding boxes for opened image
        # xml_file_dict = read_voc_xml(f'{labels_dir}\\{file[:-4]}.xml')
        xml_file_dict = read_voc_xml(os.path.join(labels_dir, f'{file[:-4]}.xml'))
        for bbox in xml_file_dict['objects']:
            # xmin, ymin, xmax, ymax = bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']
            bbox_coord_list = [bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']]
            bboxes.append(bbox_coord_list)

        # Extract 3 non-crosswalk regions (per image)
        for i in range(3):
            # Attempt to find a non-crosswalk region
            region = None
            size_list = []
            for pot_size in [64, 32, 16]:
                if pot_size <= final_size:
                    size_list.append(pot_size)
            for size in size_list:
                region = find_non_crosswalk_region(image, bboxes, size, final_size)
                if region is not None:
                    break

            # If no non-crosswalk region is found, skip the image
            if region is None:
                print(f"No suitable non-crosswalk region found for {file}")
                break

            # Save the region to the output directory
            output_path = os.path.join(output_dir, f"{file[:-4]}_non_crosswalk_{i}.png")
            cv2.imwrite(output_path, region)


image_dir = os.path.join(os.getcwd(), '..', 'dataset', 'train_val_test_split', 'images', 'train')
output_dir = os.path.join(os.getcwd(), 'data', 'images', 'square_non_crosswalk_regions')
# output_dir = r'C:\Users\janku\Documents\AI\Projects\GoogleReCaptcha\archive\here test_bb'
labels_dir = os.path.join(os.getcwd(), '..', 'dataset', 'train_val_test_split', 'annotations', 'train_bb')

final_size = 32

os.makedirs(output_dir, exist_ok=True)

extract_non_crosswalk_regions(image_dir, labels_dir, output_dir, final_size)