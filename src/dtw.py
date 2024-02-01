import numpy as np
from fastdtw import fastdtw 
from scipy.spatial.distance import euclidean
import csv
import numba as nb
from tqdm.auto import tqdm
import time 
import os 

#2  1060~ 1122
#5  1195~ 1267
#18 2347~ 2423

np.set_printoptions(suppress = True)


def read_file_to_array(file_path):
    values = []
    with open(file_path, "r") as f:
        for row in f:
            row = row.strip()  
            if row:  
                values.append(float(row))
    return np.array(values)


def dis(dic,grad_number1:str,grad_number2:str):
    distance, _ = fastdtw(dic[grad_number1], 
    dic[grad_number2],dist=euclidean)
    return distance


# def save_array_to_dict(folder,grad_ids:list):
#     grad_dict = {}
#     for i  in grad_ids:
#         file_path = rf'{folder}\{i}_clear.csv'
#         grad_array = read_file_to_array(file_path)
#         grad_dict[f'grad_{i}'] = grad_array.reshape(-1, 1)
#     return  grad_dict


def save_array_to_dict(folder, grad_ids):
    grad_dict = {}
    for i in grad_ids:
        grad_files = []
        for entry in os.scandir(folder):
            if entry.is_file() and entry.name.endswith('.csv') and str(i) in entry.name:
                file_path = entry.path
                grad_array = read_file_to_array(file_path)
                grad_dict[f'grad_{i}'] = grad_array.reshape(-1, 1)
    
    return grad_dict

def create_dict_name(test_id, sub_id, run_id):
    return f"Test{test_id}_{sub_id}_{run_id}"


def calculate_distance(grad_dict,grad_ids):
    num_grads = len(grad_ids)
    test_array = np.zeros((num_grads,num_grads))
    for i in range(num_grads):
        for j in range(num_grads):
            grad_id_i = grad_ids[i]  
            grad_id_j = grad_ids[j]
            distance = dis(grad_dict, f"grad_{grad_id_i}", f"grad_{grad_id_j}")
            test_array[i, j] = distance
            test_array[j, i] = distance

    return test_array.round(0)





# def run():
#     all_test_dict = {}
#     test_id = "6"
#     folder_path = [r"C:\Users\Lab_205\Desktop\image_overlapping_project\src\contours\test\two_five\noslid\pointup",r"C:\Users\Lab_205\Desktop\image_overlapping_project\src\contours\test\two_five\noslid\rotated_45",r"C:\Users\Lab_205\Desktop\image_overlapping_project\src\contours\test\two_five\noslid\rotated_90",r"C:\Users\Lab_205\Desktop\image_overlapping_project\src\contours\test\two_five\noslid\rotated_135",r"C:\Users\Lab_205\Desktop\image_overlapping_project\src\contours\test\two_five\noslid\rotated_150",r"C:\Users\Lab_205\Desktop\image_overlapping_project\src\contours\test\two_five\noslid\rotated_180"]
    
#     grad_ids = [1060,1061,1062,1195,1196,1197]

#     for folder, sub_id in tqdm(zip(folder_path, range(1, 7))):
#         run_id = "3"
#         dict_name = create_dict_name(test_id, sub_id, run_id)
#         array_dict = save_array_to_dict(folder,grad_ids)
#         all_test_dict[dict_name] = calculate_distance(array_dict,grad_ids)
        
#     return all_test_dict

def run():
    all_test_dict = {}
    test_id = "6"
    folder_path = [r"C:\Users\Lab_205\Desktop\image_overlapping_project\src\contours\test\grad\correctgrad\slid20_1",r"C:\Users\Lab_205\Desktop\image_overlapping_project\src\contours\test\grad\grad_rotated90\slid20_1",r"C:\Users\Lab_205\Desktop\image_overlapping_project\src\contours\test\grad\grad_rotated135\slid20_1",r"C:\Users\Lab_205\Desktop\image_overlapping_project\src\contours\test\grad\grad_rotated150\slid20_1",r"C:\Users\Lab_205\Desktop\image_overlapping_project\src\contours\test\grad\grad_rotated180\slid20_1"]
    
    grad_ids = [i for i in range(1060,1123)]

    for folder, sub_id in tqdm(zip(folder_path, range(1,6))):
        run_id = "1"
        dict_name = create_dict_name(test_id, sub_id, run_id)
        array_dict = save_array_to_dict(folder,grad_ids)
        all_test_dict[dict_name] = calculate_distance(array_dict,grad_ids)
        
    return all_test_dict



for _ in tqdm(range(1)):
    all_test_array = run()

all_test_array['test6_1_1']
all_test_array
test_y =[]
for i in all_test_array["Testtemp_0_3"]:
    for j in i:
        test_y.append(j)
print(test_y)

# point up grad_20  slid20_1

# grad_rotated45 grad_20 slid20_1

# grad_rotated90 grad_20 slid20_1

# grad_rotated135 grad_20 slid20_1

# grad_rotated150 grad_20 slid20_1

# grad_rotated180 grad_20 slid20_1



import seaborn as sns
sns.displot(test_y)