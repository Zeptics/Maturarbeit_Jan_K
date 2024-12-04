import os
import shutil


def copy_train_valid_test_files_to_dest_folder(sorted_parent_dir, unsorted_parent_dir):
    for folder in ['train', 'val', 'test']:
        for file in os.listdir(os.path.join(sorted_parent_dir, folder)):
            print(os.path.join(unsorted_parent_dir, f'{file[:-4]}.json'))
            dst_json = os.path.join(unsorted_parent_dir, folder, f'{file[:-4]}.json')
            dst_png = os.path.join(unsorted_parent_dir, folder, file)
            os.makedirs(os.path.join(unsorted_parent_dir, folder), exist_ok=True)
            src_json = os.path.join(unsorted_parent_dir, 'all_data', f'{file[:-4]}.json')
            src_png = os.path.join(unsorted_parent_dir, 'all_data', file)

            shutil.copy(src_png, dst_png)
            shutil.copy(src_json, dst_json)


sorted_parent_directory = r'../../dataset/train_val_test_split/images'
unsorted_parent_directory = r'../../dataset/Crosswalk'


copy_train_valid_test_files_to_dest_folder(sorted_parent_directory, unsorted_parent_directory)

