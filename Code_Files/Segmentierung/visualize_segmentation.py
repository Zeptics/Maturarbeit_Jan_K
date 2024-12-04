import os
from pathlib import Path
import matplotlib.pyplot as plt
import cv2


# Function to display images with titles
def plot_masks(images, titles, show_image=True, save_image=False, save_dir=None, image_filename=None):
    fig, axs = plt.subplots(1, len(images), figsize=(15, 5))
    for i, img in enumerate(images):
        axs[i].imshow(img)
        axs[i].axis('off')
        axs[i].set_title(titles[i])  # Add title to each image

    # Save the figure if save_image is True
    if save_image and save_dir and image_filename:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
        save_filename = f"{image_filename}.png"  # Use the original image file name for saving
        plt.savefig(save_path / save_filename, bbox_inches='tight')

    # Show the image if show_image is True
    if show_image:
        plt.show()


# Main script
def display_image_triplets(show_image=True, save_image=False):
    # Get directories from user
    original_dir = r"data/images/jpgs/test"
    true_mask_dir = r"data/Trimaps/test_trimap"
    predicted_mask_dir = r"C:\Users\janku\PycharmProjects\Maturitaetsarbeit\Python_Code\Segmentation\models\DeepLabV3_MobilenetV2\v7\checkpoints_steps\step20000\predictions"
    save_dir = r"saved_images"  # Directory where images will be saved

    # List all image files in the original directory
    image_files = sorted([f for f in os.listdir(original_dir)])

    # Display each triplet of images
    for image_file in image_files:
        # Define paths to corresponding mask files
        true_mask_file = f'{image_file[:-4]}_trimap.png'
        predicted_mask_file = os.path.join(predicted_mask_dir, f'{image_file[:-4]}.png')

        # Check if the corresponding files exist
        if true_mask_file and predicted_mask_file:
            print(f"Loading files for {image_file}...")
            # Load images
            original_image = cv2.cvtColor(cv2.imread(os.path.join(original_dir, image_file)), cv2.COLOR_BGR2RGB)
            true_mask = cv2.imread(os.path.join(true_mask_dir, true_mask_file), cv2.IMREAD_GRAYSCALE)
            predicted_mask = cv2.imread(os.path.join(predicted_mask_dir, predicted_mask_file), cv2.IMREAD_GRAYSCALE)

            # Display and save images
            plot_masks(
                [original_image, true_mask, predicted_mask],
                ["Original Image", "True Mask", "Predicted Mask"],
                show_image=show_image,
                save_image=save_image,
                save_dir=save_dir,
                image_filename=image_file[:-4]  # Save with the original image filename
            )
        else:
            print(f"Skipping {image_file} as the corresponding mask files were not found.")


# Run the script with parameters
if __name__ == "__main__":
    # Set show_image and save_image based on your needs
    show_image = False  # Set to False if you do not want to display the images
    save_image = True  # Set to False if you do not want to save the images

    # Call the function with the parameters
    display_image_triplets(show_image=show_image, save_image=save_image)
