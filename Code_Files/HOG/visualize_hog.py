import os
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import xml.etree.ElementTree as ET


# Function to parse Pascal VOC XML and extract bounding box coordinates
def parse_bboxes(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    bboxes = []
    for obj in root.findall('object'):
        # Extract bounding box coordinates
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        bboxes.append((xmin, ymin, xmax, ymax))
        print(f"Bounding box: {(xmin, ymin, xmax, ymax)}")  # Debug print
    return bboxes


# Function to draw bounding boxes on an image
# Function to draw bounding boxes on an image
def draw_bboxes(image, bboxes, color=(0, 255, 0), thickness=1):  # Thickness changed to 1
    image_copy = image.copy()
    for (xmin, ymin, xmax, ymax) in bboxes:
        cv2.rectangle(image_copy, (xmin, ymin), (xmax, ymax), color, thickness)
    return image_copy



# Function to display images with bounding boxes
def plot_bboxes(true_image, predicted_image, show_image=True, save_image=False, save_dir=None, image_filename=None):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # Plot true bounding boxes
    axs[0].imshow(true_image)
    axs[0].axis('off')
    axs[0].set_title("True Bounding Boxes")

    # Plot predicted bounding boxes
    axs[1].imshow(predicted_image)
    axs[1].axis('off')
    axs[1].set_title("Predicted Bounding Boxes")

    # Save the figure if save_image is True
    if save_image and save_dir and image_filename:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
        save_filename = f"{image_filename}.png"  # Use the original image file name for saving
        plt.savefig(save_path / save_filename, bbox_inches='tight')

    # Show the images if show_image is True
    if show_image:
        plt.show()


# Main script
def display_bounding_boxes(show_image=True, save_image=False):
    # Get directories from user
    original_dir = r"../dataset/train_val_test_split/images/test"
    true_bboxes_dir = r"../dataset/train_val_test_split/annotations/test_bb"
    predicted_bboxes_dir = r"Optimizing/Predicted_Labels_rbf/svm_model_32_8_8_8_9_c1_b0.1/threshold_0.75"
    save_dir = r"saved_images"  # Directory where images will be saved

    # List all image files in the original directory (now looking for PNG files)
    image_files = sorted([f for f in os.listdir(original_dir) if f.endswith('.png')])
    print(f"Image files: {image_files}")  # Debug print

    # Display each image with bounding boxes
    for image_file in image_files:
        # Define paths to corresponding XML files
        true_bbox_file = os.path.join(true_bboxes_dir, f'{image_file[:-4]}.xml')
        predicted_bbox_file = os.path.join(predicted_bboxes_dir, f'{image_file[:-4]}.xml')

        # Check if the corresponding XML files exist
        if os.path.exists(true_bbox_file) and os.path.exists(predicted_bbox_file):
            print(f"Loading files for {image_file}...")
            # Load the image (PNG files)
            original_image = cv2.cvtColor(cv2.imread(os.path.join(original_dir, image_file)), cv2.COLOR_BGR2RGB)
            print(f"Original image shape: {original_image.shape}")  # Debug print

            # Parse bounding boxes
            true_bboxes = parse_bboxes(true_bbox_file)
            predicted_bboxes = parse_bboxes(predicted_bbox_file)

            # Draw bounding boxes
            true_image_with_bboxes = draw_bboxes(original_image, true_bboxes)
            predicted_image_with_bboxes = draw_bboxes(original_image, predicted_bboxes)

            # Display and save images
            plot_bboxes(
                true_image_with_bboxes,
                predicted_image_with_bboxes,
                show_image=show_image,
                save_image=save_image,
                save_dir=save_dir,
                image_filename=image_file[:-4]  # Save with the original image filename
            )
        else:
            print(f"Skipping {image_file} as the corresponding bbox files were not found.")


# Run the script with parameters
if __name__ == "__main__":
    # Set show_image and save_image based on your needs
    show_image = False  # Set to False if you do not want to display the images
    save_image = True  # Set to False if you do not want to save the images

    # Call the function with the parameters
    display_bounding_boxes(show_image=show_image, save_image=save_image)
