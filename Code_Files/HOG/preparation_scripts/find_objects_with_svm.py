import cv2
from tqdm import tqdm
from sklearn import svm
import joblib
import numpy as np
import os
from PIL import Image, ImageEnhance
import xml.etree.ElementTree as ET


def increase_contrast(image_path, enhancement_factor):
    # Open the image
    image = Image.open(image_path)

    image = image.convert('L')
    # Enhance the contrast
    enhancer = ImageEnhance.Contrast(image)
    enhanced_image = enhancer.enhance(enhancement_factor)
    # enhanced_image = enhanced_image.convert('L')
    return enhanced_image


def compute_hog_features(image):
    # Define HOG parameters
    win_size = (32, 32)
    block_size = (16, 16)
    block_stride = (4, 4)
    cell_size = (16, 16)
    nbins = 9
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    hog_features = hog.compute(image)
    hog_features = hog_features.flatten()
    return hog_features


def sliding_window(image, clf, win_size, stride, threshold):
    detections = []
    height, width = image.shape[:2]

    for y in range(0, height - win_size[1], stride):
        for x in range(0, width - win_size[0], stride):
            window = image[y:y + win_size[1], x:x + win_size[0]]
            hog_features = compute_hog_features(window)  # Compute HOG features for the window
            decision = clf.decision_function(hog_features.reshape(1, -1))[0]  # Decision function value
            # print(decision)
            if decision > threshold:
                detections.append((x, y, x + win_size[0], y + win_size[1]))  # Store the bounding box coordinates

    return detections


def multi_scale_sliding_window(image, clf, win_size, stride, threshold, scale_range):
    detections = []
    height, width = image.shape[:2]

    for scale in scale_range:
        # print(scale)
        resized_image = cv2.resize(image, (int(width * scale), int(height * scale)))
        scaled_detections = sliding_window(resized_image, clf, win_size, stride, threshold)
        # Scale detections back to the original image size
        for (x1, y1, x2, y2) in scaled_detections:
            x1 = int(x1 / scale)
            y1 = int(y1 / scale)
            x2 = int(x2 / scale)
            y2 = int(y2 / scale)
            detections.append((x1, y1, x2, y2))
    return detections


def save_bounding_boxes_to_xml(image_name, detections, output_dir):
    # Create root element
    root = ET.Element("annotation")
    filename = ET.SubElement(root, "filename")
    filename.text = image_name

    # Create object elements for each bounding box
    for idx, (x1, y1, x2, y2) in enumerate(detections, start=1):
        obj = ET.SubElement(root, "object")
        obj_name = ET.SubElement(obj, "name")
        obj_name.text = f"object_{idx}"
        bbox = ET.SubElement(obj, "bndbox")
        xmin = ET.SubElement(bbox, "xmin")
        xmin.text = str(x1)
        ymin = ET.SubElement(bbox, "ymin")
        ymin.text = str(y1)
        xmax = ET.SubElement(bbox, "xmax")
        xmax.text = str(x2)
        ymax = ET.SubElement(bbox, "ymax")
        ymax.text = str(y2)

    # Create XML tree and write to file
    tree = ET.ElementTree(root)
    xml_filename = os.path.splitext(image_name)[0] + ".xml"
    xml_path = os.path.join(output_dir, xml_filename)
    tree.write(xml_path)


# Define parameters
win_size = (32, 32)
stride = 16
threshold = 2.35
scale_range = [2, 1.5]
clf = joblib.load('svm_model.pkl')

image_dir = os.path.join(os.getcwd(), '..', 'train_val_test_split', 'Crosswalk')
output_dir_imgs = os.path.join(os.getcwd(), 'images', 'predicted_crosswalk_regions')
output_dir_xmls = os.path.join(os.getcwd(), 'Labels', 'predicted_crosswalk_regions')

for image in tqdm([file for file in os.listdir(image_dir) if file.endswith('.png')]):
    image_path = os.path.join(image_dir, image)
    image_open = cv2.imread(image_path)
    enhanced_image = np.array(increase_contrast(image_path, 1000))
    # print(enhanced_image.shape)
    # if enhanced_image.shape[2] == 4:
    #     enhanced_image = enhanced_image[:, :, :3]
    # cv2.imshow('fortnite', np.array(enhanced_image))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    detections = multi_scale_sliding_window(np.array(enhanced_image), clf, win_size, stride, threshold, scale_range)

    # save detections to .xml file
    save_bounding_boxes_to_xml(image, detections, output_dir_xmls)

    # # Draw bounding boxes on the image
    # for (x1, y1, x2, y2) in detections:
    #     cv2.rectangle(image_open, (x1, y1), (x2, y2), (0, 255, 0), 1)
    #
    # resized_image = cv2.resize(image_open, (480, 480))
    # # Display the result
    # # cv2.imshow('Object Detection', resized_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #
    # # cv2.imwrite(os.path.join(output_dir_imgs, f'{image}_pred.png'), resized_image)
