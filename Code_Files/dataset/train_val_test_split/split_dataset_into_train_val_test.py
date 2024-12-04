import os
import shutil
import random
from tqdm import tqdm


def split_dataset(input_dir, images_output_dir, annotations_output_dir, train_split=0.8, val_split=0.1):
    images = [file for file in os.listdir(input_dir) if file.endswith('.png')]
    random.shuffle(images)
    train_size = int(len(images) * train_split)
    val_size = int(len(images) * val_split)

    train_images = images[:train_size]
    val_images = images[train_size:train_size + val_size]
    test_images = images[train_size + val_size:]

    def copy_files(file_list, img_out_dir, ann_out_dir, type):
        for file in tqdm(file_list):
            xml_file = file.replace('.png', '.xml')
            # shutil.copy(os.path.join(input_dir, file), os.path.join(target_dir, 'Images', file))
            # shutil.copy(os.path.join(input_dir, xml_file), os.path.join(target_dir, 'Images', xml_file))

            shutil.copy(os.path.join(input_dir, file), os.path.join(img_out_dir, type, file))
            shutil.copy(os.path.join(input_dir, xml_file), os.path.join(ann_out_dir, type, xml_file))

    copy_files(train_images, images_output_dir, annotations_output_dir, 'train_bb')
    copy_files(val_images, images_output_dir, annotations_output_dir, 'val_bb')
    copy_files(test_images, images_output_dir, annotations_output_dir, 'test_bb')


input_directory = os.path.join(os.getcwd(), '..', 'Crosswalk')
images_output_directory = os.path.join(os.getcwd(), 'images1')
annotations_output_directory = os.path.join(os.getcwd(), 'annotations1')

os.makedirs(os.path.join(images_output_directory, 'train_bb'), exist_ok=True)
os.makedirs(os.path.join(images_output_directory, 'val_bb'), exist_ok=True)
os.makedirs(os.path.join(images_output_directory, 'test_bb'), exist_ok=True)
os.makedirs(os.path.join(annotations_output_directory, 'train_bb'), exist_ok=True)
os.makedirs(os.path.join(annotations_output_directory, 'val_bb'), exist_ok=True)
os.makedirs(os.path.join(annotations_output_directory, 'test_bb'), exist_ok=True)

split_dataset(input_directory, images_output_directory, annotations_output_directory)
