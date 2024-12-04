import json
import numpy as np
from PIL import Image
import os
import cv2
from tqdm import tqdm

def create_trimap_from_labelme(labelme_json_path, image_shape):
    # Load the LabelMe JSON file
    with open(labelme_json_path, 'r') as f:
        data = json.load(f)

    # Initialize trimap
    trimap = np.zeros(image_shape, dtype=np.uint8)  # Assume the shape is (height, width)

    # Create the trimap based on annotations
    for shape in data['shapes']:
        points = np.array(shape['points'], dtype=np.int32)
        if shape['label'] == "Crosswalk":  # Adjust according to your labeling
            cv2.fillPoly(trimap, [points], 1)  # Foreground
        # else:
        #     cv2.fillPoly(trimap, [points], 0)  # Background

    # Set unknown area
    unknown_mask = (trimap == 0)
    trimap[unknown_mask] = 0  # Unknown

    return trimap


def save_trimap(trimap, save_path):
    # Scale the trimap values to 0-255
    scaled_trimap = (trimap * 1).astype(np.uint8)  # Scale from 0-3 to 0-255
    img = Image.fromarray(scaled_trimap)
    img.save(save_path)


# Define your paths and image shape
labelme_folder = r'input_dir'  # Folder containing JSON files
output_folder = r'output_dir'  # Folder to save trimaps

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for json_file in tqdm(os.listdir(labelme_folder)):
    if json_file.endswith('.json'):
        json_path = os.path.join(labelme_folder, json_file)
        image_shape = (120, 120)  # Set the correct image dimensions here
        trimap = create_trimap_from_labelme(json_path, image_shape)
        output_path = os.path.join(output_folder, json_file.replace('.json', '_trimap.png'))
        save_trimap(trimap, output_path)
