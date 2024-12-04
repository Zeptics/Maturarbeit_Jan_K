import shutil
import glob
import time
import os

model_name = 'DeepLabV3_MobilenetV2'
model_version = 'v8'

source_directory = os.path.join(f'models/{model_name}', model_version, 'trained_model')
# destination_root_directory = r'E:\faster_rcnn_resnet152_v1_640x640_coco17_tpu-8\v1\checkpoints_steps'
destination_root_directory = os.path.join(f'models/{model_name}', model_version, 'checkpoints_steps')
total_steps = 100000
last_ckpt_savings_dir = 'step0'

# Check for current checkpoint files to avoid duplication
copied_checkpoints = set()


def copy_files(index_file, data_file, destination_directory):
    # Copy the new checkpoint file to the destination directory
    shutil.copy(index_file, destination_directory)
    copied_checkpoints.add(index_file)
    shutil.copy(data_file, destination_directory)
    copied_checkpoints.add(data_file)

    print(f'copied {index_file} to {destination_directory}')
    print(f'copied {data_file} to {destination_directory}')


while True:
    # Find all current ckpt files
    checkpoint_files = glob.glob(f"{source_directory}/ckpt-*")

    # Filter out already copied checkpoints
    new_checkpoints = [file for file in checkpoint_files if file not in copied_checkpoints]

    for file in new_checkpoints:
        # Dynamically add checkpoint files to new directories
        if file.endswith('.index'):
            num_steps = file.split('-')[-1][:-6]
            # '''This checks for the last created checkpoint file --> as it's just a model saving everything and no
            # training is happening the checkpoint file should be saved to the last directory'''
            # print(num_steps, (total_steps) + 1)
            # if int(num_steps) == (total_steps) + 1:
            #     destination_excact_dir = last_ckpt_savings_dir
            #     data_file_name = f'{file.split(".")[0]}.data-00000-of-00001'
            #     copy_files(file, data_file_name, destination_excact_dir)
            # for the first 1000 files, save a model for every 100 steps
            if int(num_steps) <= 1000:
                destination_excact_dir = os.path.join(destination_root_directory, f'step{int(num_steps)}')
                os.makedirs(destination_excact_dir, exist_ok=True)
                data_file_name = f'{file.split(".")[0]}.data-00000-of-00001'
                copy_files(file, data_file_name, destination_excact_dir)
            # Until 5000 steps, save a model for every 500 steps
            elif int(num_steps) <= 5000:
                left_over = int(num_steps) % 500
                if left_over != 0:
                    exactly_wanted_ckpt_file = False
                    step_folder_num = int(num_steps) + (500 - left_over)
                else:
                    exactly_wanted_ckpt_file = True
                    step_folder_num = int(num_steps)
                destination_excact_dir = os.path.join(destination_root_directory, f'step{int(step_folder_num)}')
                os.makedirs(destination_excact_dir, exist_ok=True)
                data_file_name = f'{file.split(".")[0]}.data-00000-of-00001'
                if exactly_wanted_ckpt_file:
                    copy_files(file, data_file_name, destination_excact_dir)
                # Until 20000 steps save a model for every 1000 steps
            elif int(num_steps) <= 20000:
                left_over = int(num_steps) % 1000
                if left_over != 0:
                    exactly_wanted_ckpt_file = False
                    step_folder_num = int(num_steps) + (1000 - left_over)
                else:
                    exactly_wanted_ckpt_file = True
                    step_folder_num = int(num_steps)
                destination_excact_dir = os.path.join(destination_root_directory, f'step{int(step_folder_num)}')
                os.makedirs(destination_excact_dir, exist_ok=True)
                data_file_name = f'{file.split(".")[0]}.data-00000-of-00001'
                if exactly_wanted_ckpt_file:
                    copy_files(file, data_file_name, destination_excact_dir)
                # Past 5000 training steps save a model every 5000 training steps
            elif int(num_steps) > 20000:
                left_over = int(num_steps) % 5000
                if left_over != 0:
                    exactly_wanted_ckpt_file = False
                    step_folder_num = int(num_steps) + (5000 - left_over)
                else:
                    exactly_wanted_ckpt_file = True
                    step_folder_num = int(num_steps)
                destination_excact_dir = os.path.join(destination_root_directory, f'step{int(step_folder_num)}')
                os.makedirs(destination_excact_dir, exist_ok=True)
                data_file_name = f'{file.split(".")[0]}.data-00000-of-00001'
                if exactly_wanted_ckpt_file:
                    copy_files(file, data_file_name, destination_excact_dir)

            # set last_ckpt_savings_dir to old destination_exact_dir
            last_ckpt_savings_dir = destination_excact_dir

    # Wait before checking again
    time.sleep(30)
