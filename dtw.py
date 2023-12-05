import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import csv


def read_file_to_array(file_path):
    values = []
    with open(file_path, "r") as f:
        for row in f:
            row = row.strip()  
            if row:  
                values.append(float(row))
    return np.array(values)




####################### test no equal grads numbers ##############
####################### all grads is 20,1 and through sliding 20 ,1
#2  1060~ 1122
#5  1195~ 1267
#18 2347~ 2423

Test1 = {}
for i in range(1060,1123):
    file_path = rf'C:\Users\Lab_205\Desktop\image_overlapping_project\dataset_output\find_pattern\2_Chinese horse chestnut\contourfiles01\plot_20_rotated\slid\{i}_clear_slid.csv'
    grad_array = read_file_to_array(file_path)
    Test1[f'grad_{i}'] = grad_array.reshape(-1, 1)



def dis(grad_number1:str,grad_number2:str):
    distance, _ = fastdtw(Test1[f"grad_{grad_number1}"], 
    Test1[f"grad_{grad_number2}"],dist=euclidean)
    return distance



M1_1 = np.zeros((5,5))
for i in range(5):
    for j in range(5):
        distance = dis(str(i+1060), str(j+1060))
        M1_1[i, j] = distance
        M1_1[j, i] = distance
M1_1


M1_2 = np.zeros((5,5))
for i in range(5):
    for j in range(5):
        distance = dis(str(i+1060), str(j+1195))
        M1_2[i, j] = distance
        M1_2[j, i] = distance
M1_2

M1_3 = np.zeros((5,5))
for i in range(5):
    for j in range(5):
        distance = dis(str(i+1060), str(j+2347))
        M1_3[i, j] = distance
        M1_3[j, i] = distance
M1_3

############## TEST 2  about 200 grad numbers   #####################
#2  1060~ 1122
#5  1195~ 1267
#18 2347~ 2423

Test2 = {}
len(Test1)

for i in range(1195,1268):
    file_path = rf'C:\Users\Lab_205\Desktop\image_overlapping_project\dataset_output\find_pattern\5_true indigo\contourfiles01\plot_20_rotated\slid\size_20_3\{i}_clear_slid.csv'
    grad_array = read_file_to_array(file_path)
    Test2[f'grad_{i}'] = grad_array.reshape(-1, 1)

M1_1_2 = np.zeros((5,5))
for i in range(5):
    for j in range(5):
        distance = dis(str(i+1060), str(j+1060))
        M1_1_2[i, j] = distance
        M1_1_2[j, i] = distance
M1_1_2

M1_2_2 = np.zeros((5,5))
for i in range(5):
    for j in range(5):
        distance = dis(str(i+1060), str(j+1195))
        M1_2_2[i, j] = distance
        M1_2_2[j, i] = distance
M1_2_2

M1_3_2 = np.zeros((5,5))
for i in range(5):
    for j in range(5):
        distance = dis(str(i+1060), str(j+2347))
        M1_3_2[i, j] = distance
        M1_3_2[j, i] = distance

M1_3_2


M1_1 == M1_1_2

M1_2 == M1_2_2

M1_3 == M1_3_2