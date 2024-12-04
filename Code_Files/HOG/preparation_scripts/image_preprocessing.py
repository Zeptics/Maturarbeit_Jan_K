import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import cv2
from PIL import Image, ImageEnhance


def increase_contrast(image_path, contrast_enhancement_factor, brightness_enhancement_factor):
    # Open the image
    image = Image.open(image_path)

    image = image.convert('L')
    # Enhance the contrast
    enhancer = ImageEnhance.Contrast(image)
    enhanced_image = enhancer.enhance(contrast_enhancement_factor)

    enhancer = ImageEnhance.Brightness(enhanced_image)
    enhanced_image = enhancer.enhance(brightness_enhancement_factor)

    # enhanced_image = enhanced_image.convert('L')
    return enhanced_image


def adjust_image_folder(contrast_enhancement_fact, brightness_enhancement_fact, image_dir, show_imgs, save_imgs):
    for img in [file for file in os.listdir(image_dir) if file.endswith('.png')]:
        image_path = os.path.join(image_dir, img)
        adjusted_image = increase_contrast(image_path, contrast_enhancement_fact, brightness_enhancement_fact)

        if show_imgs:
            # Convert the PIL image to a NumPy array
            adjusted_image_np = np.array(adjusted_image)

            # Display the image using Matplotlib
            plt.imshow(adjusted_image_np, cmap='gray')
            plt.title(img)
            plt.axis('off')  # Hide axes
            plt.show()


adjust_image_folder(10, 0.005, r"path", True, False)
