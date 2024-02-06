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


def save_array_to_dict(folder, grad_ids):
    grad_dict = {}
    for i in grad_ids:
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

#######################################

def run():
    all_test_dict = {}
    test_id = "6"
    folder_path = [r"C:\Users\Lab_205\Desktop\image_overlapping_project\src\contours\test_4\upand45",
    r"C:\Users\Lab_205\Desktop\image_overlapping_project\src\contours\test_4\upand90"]
    
    ### need change ###
    grad_ids = [i for i in range(1061,1069)]

    for folder, sub_id in tqdm(zip(folder_path, range(1,3))):
        run_id = "3"
        dict_name = create_dict_name(test_id, sub_id, run_id)
        array_dict = save_array_to_dict(folder,grad_ids)
        all_test_dict[dict_name] = calculate_distance(array_dict,grad_ids)
        
    return all_test_dict



for _ in tqdm(range(1)):
    all_test_array = run()


all_test_array_1  #pointup
all_test_array_2  #45
all_test_array_3 # up 45 and up 90

########## plot  ##########
keys = ["Test6_1_2", 'Test6_2_2', "Test6_3_2", "Test6_4_2", "Test6_5_2","Test6_6_2"]
labels = ["Test6_1_2", 'Test6_2_2_45', "Test6_3_2_90", "Test6_4_2_135", "Test6_5_2_150","Test6_6_2_180"]

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
for key , label in zip(keys,labels):
    test_y =[]
    for i in all_test_array_3[key]:
        for j in i:
            test_y.append(j)    

    kde = KernelDensity(bandwidth=1.0, kernel='gaussian')
    kde.fit(np.array(test_y).reshape(-1, 1))
 
    x = np.linspace(min(test_y), max(test_y), 1000).reshape(-1, 1)
    pdf = np.exp(kde.score_samples(x))
    cdf = np.cumsum(pdf) * (x[1] - x[0])

    confidence_level = 0.95
    lower_bound = x[np.argmax(cdf >= (1-confidence_level)/2)][0]
    upper_bound = x[np.argmax(cdf >= 1-(1-confidence_level)/2)][0]


    p= sns.histplot(test_y, kde=True)
    plt.title(label)
    plt.axvline(lower_bound, color="r", linestyle="--", label=f"{confidence_level*100}% CI")
    plt.axvline(upper_bound, color="r", linestyle="--")
    p.set_xlabel("distance")
    p.set_ylabel("freq")
    plt.legend()
    plt.show()
    print(f"95% Confidence Interval: ({lower_bound}, {upper_bound})")
