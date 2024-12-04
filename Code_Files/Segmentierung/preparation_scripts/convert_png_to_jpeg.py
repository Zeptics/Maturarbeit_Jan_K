import os
from PIL import Image


def convert_png_to_jpg(input_dir, output_dir):
    # Check if output directory exists, if not create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Loop over all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".png"):
            # Open the image
            img_path = os.path.join(input_dir, filename)
            img = Image.open(img_path)

            # Convert the image to RGB (necessary for JPG format)
            rgb_img = img.convert('RGB')

            # Define the output path with .jpg extension
            output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.jpg")

            rgb_img = rgb_img.resize((120, 120))

            # Save the image as JPG
            rgb_img.save(output_path, "JPEG")

            print(f"Converted {filename} to JPG")


if __name__ == "__main__":
    input_dir = r"input_dir"  # Specify the input directory containing PNGs
    output_dir = r"output_dir"  # Specify the output directory where JPGs will be saved

    convert_png_to_jpg(input_dir, output_dir)
