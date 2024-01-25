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


def dis(dic,grad_number1:str,grad_number2:str):
    distance, _ = fastdtw(dic[grad_number1], 
    dic[grad_number2],dist=euclidean)
    return distance


####################### test no equal grads numbers ##############
####################### all grads is 20,1 and through sliding 20 ,1
#2  1060~ 1122
#5  1195~ 1267
#18 2347~ 2423

Test1 = {}
for i in range(1060,1123):
    file_path = rf'C:\Users\Lab_205\Desktop\image_overlapping_project\src\contours\test\grad20\slid20_1\{i}_clear_slid.csv'
    grad_array = read_file_to_array(file_path)
    Test1[f'grad_{i}'] = grad_array.reshape(-1, 1)

Test1
Test1_list = list(Test1.keys())

### TEST FUNCTION ###
d , l =fastdtw(Test1["grad_1060"],Test1["grad_1062"],radius=100,dist = euclidean)
print(d)
#####################

M1_1 = np.zeros((5,5))
for i in range(5): 
    for j in range(5):
        distance = dis(Test1,Test1_list[i], Test1_list[j])
        M1_1[i, j] = distance
        M1_1[j, i] = distance
M1_1


M1_2 = np.zeros((5,5))
for i in range(5):
    for j in range(5):
        distance = dis(Test1,str(i+1060), str(j+1195))
        M1_2[i, j] = distance
        M1_2[j, i] = distance
M1_2

M1_3 = np.zeros((5,5))
for i in range(5):
    for j in range(5):
        distance = dis(Test1,str(i+1060), str(j+2347))
        M1_3[i, j] = distance
        M1_3[j, i] = distance
M1_3

############## TEST 2  about 200 grad numbers   #####################
#2  1060~ 1122
#5  1195~ 1267
#18 2347~ 2423

Test2 = {}


for i in range(1060,1123):
    file_path = rf'C:\Users\Lab_205\Desktop\image_overlapping_project\dataset_output\find_pattern\2_Chinese horse chestnut\contourfiles01\plot_20_rotated\slid\size_20_5\{i}_clear_slid.csv'
    grad_array = read_file_to_array(file_path)
    Test2[f'grad_{i}'] = grad_array.reshape(-1, 1)

M1_1_2 = np.zeros((5,5))
for i in range(5):
    for j in range(5):
        distance = dis(Test2,str(i+1060), str(j+1060))
        M1_1_2[i, j] = distance
        M1_1_2[j, i] = distance
M1_1_2

M1_2_2 = np.zeros((5,5))
for i in range(5):
    for j in range(5):
        distance = dis(Test2,str(i+1060), str(j+1195))
        M1_2_2[i, j] = distance
        M1_2_2[j, i] = distance
M1_2_2

M1_3_2 = np.zeros((5,5))
for i in range(5):
    for j in range(5):
        distance = dis(Test2,str(i+1060), str(j+2347))
        M1_3_2[i, j] = distance
        M1_3_2[j, i] = distance

M1_3_2


M1_1 == M1_1_2

M1_2 == M1_2_2

M1_3 == M1_3_2


### Test3  try to find different betewwn leaf tip point up and point down ###

Test3 = {}

for i in range(1060,1123):
    file_path = rf'C:\Users\Lab_205\Desktop\image_overlapping_project\src\contours\test\grad20\correctgrad20\slid20_1\{i}_clear_slid.csv'
    grad_array = read_file_to_array(file_path)
    Test3[f'grad_{i}'] = grad_array.reshape(-1, 1)


T3_1 = np.zeros((5,5))
for i in range(5):
    for j in range(5):
        distance = dis(Test3,str(i+1060), str(j+1060))
        T3_1[i, j] = distance
        T3_1[j, i] = distance
T3_1





# 反面 葉尖朝下
Test3_2 = {}

temp1 = [1065, 1067, 1069, 1070, 1071]
for i in temp1:
    file_path = rf'C:\Users\Lab_205\Desktop\image_overlapping_project\src\contours\test\grad20\slid20_1\{i}_clear_slid.csv'
    grad_array = read_file_to_array(file_path)
    Test3_2[f'grad_{i}'] = grad_array.reshape(-1, 1)

Test3_2_list = list(Test3_2.keys())

T3_2 = np.zeros((5,5))
for i in range(5):
    for j in range(5):
        distance = dis(Test3_2,Test3_2_list[i], Test3_2_list[j])
        T3_2[i, j] = distance
        T3_2[j, i] = distance

T3_2


# 正面  葉尖朝上
Test3_3 = {}

temp2 = [1065, 1067, 1069, 1070, 1071]
for i in temp2:
    file_path = rf'C:\Users\Lab_205\Desktop\image_overlapping_project\src\contours\test\grad20\correctgrad20\slid20_1\{i}_clear_slid.csv'
    grad_array = read_file_to_array(file_path)
    Test3_3[f'grad_{i}'] = grad_array.reshape(-1, 1)

Test3_3_list = list(Test3_3.keys())

T3_3 = np.zeros((5,5))
for i in range(5):
    for j in range(5):
        distance = dis(Test3_3,Test3_3_list[i], Test3_3_list[j])
        T3_3[i, j] = distance
        T3_3[j, i] = distance

T3_3


# 3朝上3朝下   
Test3_4 = {}

temp3 = [1060, 1061, 1062, 1069, 1070, 1071]
for i in temp3:
    file_path = rf'C:\Users\Lab_205\Desktop\image_overlapping_project\src\contours\test\grad20\slid20_1\{i}_clear_slid.csv'
    grad_array = read_file_to_array(file_path)
    Test3_4[f'grad_{i}'] = grad_array.reshape(-1, 1)

Test3_4_list = list(Test3_4.keys())

T3_4 = np.zeros((6,6))
for i in range(6):
    for j in range(6):
        distance = dis(Test3_4,Test3_4_list[i], Test3_4_list[j])
        T3_4[i, j] = distance
        T3_4[j, i] = distance

T3_4


# 6朝上   
Test3_5 = {}

temp4 = [1060, 1061, 1062, 1069, 1070, 1071]
for i in temp4:
    file_path = rf'C:\Users\Lab_205\Desktop\image_overlapping_project\src\contours\test\grad20\correctgrad20\slid20_1\{i}_clear_slid.csv'
    grad_array = read_file_to_array(file_path)
    Test3_5[f'grad_{i}'] = grad_array.reshape(-1, 1)

Test3_5_list = list(Test3_5.keys())

T3_5 = np.zeros((6,6))
for i in range(6):
    for j in range(6):
        distance = dis(Test3_5,Test3_5_list[i], Test3_5_list[j])
        T3_5[i, j] = distance
        T3_5[j, i] = distance

T3_5

T3_2
T3_3
T3_4
T3_5