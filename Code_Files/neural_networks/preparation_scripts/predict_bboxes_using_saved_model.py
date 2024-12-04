import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import xml.etree.ElementTree as ET

model_dir = (r'path_to_directory_with_saved_model.pb')
detect_fn = tf.saved_model.load(model_dir)


def load_image_into_numpy_array(path):
    img_data = cv2.imread(path)
    return np.array(img_data)


def visualize_results(image_np, boxes, classes, scores, category_index, image_name, image_out_dir, threshold):
    image_np_with_detections = image_np.copy()
    for i in range(min(boxes.shape[0], 1000)):  # Adjust the number of boxes to display
        print(scores[i])
        if scores[i] > threshold:
            box = tuple(boxes[i].tolist())
            class_name = category_index[classes[i]]['name']
            score = scores[i]
            # Draw bounding box and label on the image
            cv2.rectangle(image_np_with_detections,
                          (int(box[1] * image_np.shape[1]), int(box[0] * image_np.shape[0])),
                          (int(box[3] * image_np.shape[1]), int(box[2] * image_np.shape[0])),
                          (0, 255, 0), 1)
            # label = f'{class_name}: {score:.2f}'
            # cv2.putText(image_np_with_detections, label,
            #             (int(box[1] * image_np.shape[1]), int(box[0] * image_np.shape[0]) - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    fig = plt.figure(figsize=(12, 8))
    fig.canvas.manager.set_window_title(image_name)
    if image_out_dir is not None:
        plt.imsave(os.path.join(image_out_dir, image_name), cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
        plt.show()


def create_xml_files_for_detected_bboxes(image_name, xml_file_dir, detections, scores, width, height, threshold):
    root = ET.Element('annotation')
    filename = ET.SubElement(root, 'filename')
    filename.text = image_name

    # Create object elements for each bounding box
    print('Niery')
    for idx, (y1, x1, y2, x2) in enumerate(detections, start=1):
        if scores[idx - 1] >= threshold:
            obj = ET.SubElement(root, 'object')
            obj_name = ET.SubElement(obj, 'name')
            obj_name.text = f'object_{idx}'
            bbox = ET.SubElement(obj, 'bndbox')
            xmin = ET.SubElement(bbox, 'xmin')
            xmin.text = str(int(width*x1))
            ymin = ET.SubElement(bbox, 'ymin')
            ymin.text = str(int(height*y1))
            xmax = ET.SubElement(bbox, 'xmax')
            xmax.text = str(int(width*x2))
            ymax = ET.SubElement(bbox, 'ymax')
            ymax.text = str(int(height*y2))

    # Create XML tree and write to file
    tree = ET.ElementTree(root)
    xml_filename = os.path.splitext(image_name)[0] + ".xml"
    xml_path = os.path.join(xml_file_dir, xml_filename)
    tree.write(xml_path)


def get_predictions_for_all_images(img_dir, category_index, img_out_dir, xml_out_dir, threshold):
    for image in tqdm([file for file in os.listdir(images_directory) if file.endswith('.png')]):
        image_path = os.path.join(images_directory, image)
        image_np = load_image_into_numpy_array(image_path)
        input_tensor = tf.convert_to_tensor(image_np)
        input_tensor = input_tensor[tf.newaxis, ...]

        img_width, img_height = image_np.shape[:2]

        detections = detect_fn(input_tensor)
        # 'detections' is a dictionary containing detection results
        # For example, 'detections['detection_boxes']' contains bounding box coordinates
        boxes = detections['detection_boxes'][0].numpy()
        classes = detections['detection_classes'][0].numpy().astype(np.int64)
        scores = detections['detection_scores'][0].numpy()

        visualize_results(image_np, boxes, classes, scores, category_index, image, img_out_dir, threshold)
        if xml_out_dir is not None:
            create_xml_files_for_detected_bboxes(image, xml_out_dir, boxes, scores, img_width, img_height, threshold)


# print(boxes)
# print(classes)
# print(scores)
# Assuming you have a category_index mapping class IDs to names
category_index = {1: {'id': 1, 'name': 'Crosswalk'}}  # Adjust as per your classes
threshold = -0.10

images_directory = os.path.join('../../dataset/train_val_test_split', 'images', 'test_bb')
images_output_directory = os.path.join('../Tensorflow', 'workspace', 'models', 'efficientdet_d4_coco17_tpu-32', 'test_d4', 'predictions', 'images')  # set to None if you don't want to save images
xml_output_directory = os.path.join('../Tensorflow', 'workspace', 'models', 'efficientdet_d4_coco17_tpu-32', 'test_d4', 'predictions', 'annotations')
# xml_output_directory = None  # set to None if you don't want to save bounding boxes to xml files

for directory in [images_output_directory, xml_output_directory]:
    os.makedirs(directory, exist_ok=True)

get_predictions_for_all_images(images_directory, category_index, None, None,
                               threshold)





