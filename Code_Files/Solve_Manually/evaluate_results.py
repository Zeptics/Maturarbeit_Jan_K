import os
import ast
from tqdm import tqdm
import openpyxl
from openpyxl import styles


def evaluate_person(actual_matrix_dir, person_pred_matrix_dir, specific_correct_list, specific_correct_dict):
    in_image_accuracy_list = []
    perfect_accuracy_list = []
    perfect_accuracy_name_list = []
    counter = 0
    for pred_matrix_txt_file in [file for file in os.listdir(person_pred_matrix_dir) if file.endswith('.txt') and file != 'statistics.txt']:
        pred_accuracy = 0
        with open(f'{os.path.join(actual_matrix_dir, pred_matrix_txt_file)}') as actual_matrix_file:
            actual_matrix = actual_matrix_file.read()
            actual_matrix = ast.literal_eval(actual_matrix)

            with open(f'{os.path.join(person_pred_matrix_dir, pred_matrix_txt_file)}') as pred_matrix_file:
                pred_matrix = pred_matrix_file.read()
                pred_matrix = ast.literal_eval(pred_matrix)

                for x in range(4):
                    for y in range(4):
                        if pred_matrix[x][y] == actual_matrix[x][y]:
                            pred_accuracy += 1

                for num in specific_correct_list:
                    if isinstance(num, str):
                        if num.startswith(':'):
                            specific_number = num.split(':')[1]
                            if pred_accuracy <= int(specific_number):
                                specific_correct_dict[f'number_{num}'] += 1
                    elif isinstance(num, int):
                        if pred_accuracy == num:
                            specific_correct_dict[f'number_{num}'] += 1

        accuracy_for_img = pred_accuracy / 16

        in_image_accuracy_list.append(accuracy_for_img)
        if accuracy_for_img == 1:
            perfect_accuracy_list.append(1)
            perfect_accuracy_name_list.append(pred_matrix_txt_file[:-4])
            image_accuracy_dict[f'{pred_matrix_txt_file[:-4]}.png'].append(1)
        else:
            perfect_accuracy_list.append(0)
            image_accuracy_dict[f'{pred_matrix_txt_file[:-4]}.png'].append(0)

    # print(in_image_accuracy_list)
    # print(len(in_image_accuracy_list))
    # print(perfect_accuracy_list)

    in_image_accuracy = round((sum(in_image_accuracy_list) / len(in_image_accuracy_list) * 100), 2)
    perfect_accuracy = round((sum(perfect_accuracy_list) / len(perfect_accuracy_list) * 100), 2)

    return in_image_accuracy, perfect_accuracy


specific_correct_list = [':4', 13, 14, 15]
specific_correct_dictionary = {}
for number in specific_correct_list:
    specific_correct_dictionary[f'number_{number}'] = 0

actual_matrix_directory = os.path.join(os.getcwd(), '..', 'train_val_test_split', 'captcha_matrices')
# actual_matrix_directory = os.path.join(os.getcwd(), 'matrix_output_person6')
pred_matrix_directory = os.path.join(os.getcwd(), 'matrix_output_person9')
# pred_matrix_directory = r'C:\Users\janku\Desktop\Maturaarbeit Jan\matrix_output'

person_matrix_directory = os.path.join('person_matrices')
iia_list = []
pa_list = []

image_accuracy_dict = {}
test_images_path = os.path.join('../dataset/train_val_test_split/images/test')
for image in os.listdir(test_images_path):
    image_accuracy_dict[image] = []

# print(image_accuracy_dict)


for person in tqdm(os.listdir(person_matrix_directory)):
    for sub_person in os.listdir(person_matrix_directory):
        if sub_person != person:
            iia, pa = evaluate_person(os.path.join(person_matrix_directory, person), os.path.join(person_matrix_directory, sub_person), specific_correct_list, specific_correct_dictionary)
            iia_list.append(iia)
            pa_list.append(pa)

average_iia = sum(iia_list) / len(iia_list)
average_pa = sum(pa_list) / len(pa_list)

for i, image in enumerate(image_accuracy_dict):
    accuracy_list = image_accuracy_dict.get(image)
    avg_perfect_accuracy = 100 * (sum(accuracy_list) / len(accuracy_list))
    print(f'({i+1}, {avg_perfect_accuracy})')
    if i+1 == 87:
        print(image)

print('')
print(average_iia, '%')
print(average_pa, '%')

# evaluate_person(actual_matrix_directory, pred_matrix_directory, specific_correct_list, specific_correct_dictionary)

