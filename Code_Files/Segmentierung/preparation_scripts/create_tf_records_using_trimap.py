import os
import hashlib
import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm


def create_tf_example(image_path, trimap_path):
    """Converts image and trimap to a tf.Example proto."""
    with tf.io.gfile.GFile(image_path, 'rb') as fid:
        encoded_jpg = fid.read()

    image = Image.open(image_path)
    image_height, image_width = image.size

    # Read the trimap
    with tf.io.gfile.GFile(trimap_path, 'rb') as fid:
        encoded_trimap = fid.read()

    # Create the feature dictionary
    feature_dict = {
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[image_height])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[image_width])),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_jpg])),
        'image/segmentation/class/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_trimap])),
    }

    # Create and return the tf.train_bb.Example
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example


def create_tf_record(image_dir, output_path, set):
    """Loads images and trimaps from a directory and converts them to tf.Record format."""
    with tf.io.TFRecordWriter(output_path) as writer:
        for filename in tqdm(os.listdir(image_dir)):
            if filename.endswith('.jpg') or filename.endswith('.png'):  # Include image formats as needed
                image_path = os.path.join(image_dir, filename)
                trimap_path = os.path.join('..', 'data', 'Trimaps', set, f'{filename[:-4]}_trimap.png')  # Adjust if necessary
                # print(trimap_path)

                if not os.path.exists(trimap_path):
                    print(f"Trimap not found for {filename}, skipping.")
                    continue

                tf_example = create_tf_example(image_path, trimap_path)
                writer.write(tf_example.SerializeToString())
                # print(f"Processed {filename} and {trimap_path}")


def main():
    image_dir = r'/Python_Code/Segmentation/data\images\jpgs\val'  # Update this path to your image directory
    output_path = '../data/tfrecords/val.tfrecord'  # Define the output TFRecord file name
    set = 'val_trimap'

    create_tf_record(image_dir, output_path, set)
    print('Finished creating TFRecord.')


if __name__ == '__main__':
    main()
