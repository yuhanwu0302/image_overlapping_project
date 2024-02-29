import glob
import os
from itertools import product

import numpy as np
import pandas as pd
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from tqdm.auto import tqdm
from collections import defaultdict
#2  1060~ 1122
#5  1195~ 1267
#18 2347~ 2423

np.set_printoptions(suppress = True)

def run():
    # dir_1 = r"C:\Users\Lab_205\Desktop\image_overlapping_project\src\contours\test\grad\grad_rotated180"
    # dir_2 = r"C:\Users\Lab_205\Desktop\image_overlapping_project\src\contours\test\grad\grad_rotated150"


    ### TODO: need change dir name  ###

    "../dataset_output/find_pattern/{leaf_name}/contourfiles/grad/{rotate_???}/"
    leaf_names = ["2_Chinese horse chestnut"]
    grad_rotates = ["rotate_90"]
    all_gradient_dict = {}
    for leaf_name in leaf_names:
        for grad_rotate in grad_rotates:
            image_path = fr"../dataset_output/find_pattern/{leaf_name}/contourfiles/grad/{grad_rotate}/" 
            # image_path = fr"C:\Users\Lab_205\Desktop\image_overlapping_project\src\contours\test\{leaf_name}\{grad_rotate}"
            file_list = get_all_file_path(image_path)
            gradient_dict = get_all_gradients(file_list)
            all_gradient_dict[f"{leaf_name}_{grad_rotate}"] = gradient_dict
    print(all_gradient_dict["2_Chinese horse chestnut_rotate_90"])
    selected_leaves_grads = ["2_Chinese horse chestnut_rotate_90"]
    selected_ids = [str(id) + "_clear" for id in range(1060, 1065)]
    pair_dist = calculate_pairwise_dist(all_gradient_dict, selected_leaves_grads, selected_ids)

    return pair_dist

def get_all_file_path(image_path):    
    return glob.glob(os.path.join(image_path, "*.csv"))

# def create_dict_name(leaf_id, angle, grad):
#     return f"Test{leaf_id}_{angle}_{grad}"


def get_all_gradients(file_list):
    gradient_dict = {}
    for filepath in file_list:
        grad = "20"
        # TODO: rename the images so that we can know all the information from their basenames.
        data_id = os.path.basename(filepath).strip(".csv")
        gradient_dict[data_id] = load_gradients(filepath)
    return gradient_dict

def load_gradients(file_path):
    with open(file_path, "r") as f:
        gradients = np.array([str_to_float(row) for row in f.readlines()])

    return gradients.reshape(-1, 1)

def str_to_float(r):
    # If error is raised, there should be problems in gradient calculations.
    # So don't handle any exceptions here. If raised, fix the gradient calculation code and raise exceptions there. 
    return float(r.strip())

def calculate_pairwise_dist(all_gradient_dict, leaves_grads, ids):
    combined = list(product(leaves_grads, ids))
    matrix_size = len(combined)
    # A 2D-dict
    pairwise_dist = defaultdict(dict)
    
    for i, (leaf_grad_1, id_1) in enumerate(combined):
        name_1 = f"{leaf_grad_1}_{id_1}"
        # The distance should be 0
        diagonal_dis = dis(all_gradient_dict[leaf_grad_1][id_1], all_gradient_dict[leaf_grad_1][id_1])
        # TODO: Check why not 0 in some cases
        # assert np.isclose(diagonal_dis, 0), f"The distance to itself should be 0, but get {diagonal_dis}"
        pairwise_dist[name_1][name_1] = diagonal_dis

        for j, (leaf_grad_2, id_2) in enumerate(combined):
            if j >= i:
                continue
            name_2 = f"{leaf_grad_2}_{id_2}"
            print(name_1, name_2)
            pairwise_dist[name_1][name_2] = dis(all_gradient_dict[leaf_grad_1][id_1], all_gradient_dict[leaf_grad_2][id_2])
            pairwise_dist[name_2][name_1] = pairwise_dist[name_1][name_2]
    
    return pairwise_dist
            
def dis(grad_1, grad_2):
    distance, _ = fastdtw(grad_1, grad_2,dist=euclidean)
    return distance

#     all_test_dict = {}    
#     # Calculate the pairwise dist matrix
#     all_test_dict[dict_name] = calculate_distance(array_dict,selected_ids) #FIXME        
#     return all_test_dict

# file_list = get_all_file_path(image_path) # All file (name/path) you want to work with

#     # for pytest
#     # file_list = ["f1.csv", "f2.csv"]
# gradient_dict = get_all_gradients(file_list) # get all gradients from these files {"image_id": gradient[]}


#     # for pytest
#     # gradient_dict = {
#     #            "image_id_1": [1,2,3], 
#     #            "image_id_2": [2,4,6,7,9]
#     #}
# selected_ids = [""]
# pair_dist = calculate_pairwise_dist(gradient_dict) #  


result = run()
df = pd.DataFrame(result)
df

result["grad_grad_rotated180_1060_clear"]


for _ in tqdm(range(1)):
    all_test_array = run()


all_test_array_1  #pointup
all_test_array_2  #45
all_test_array_3 # up 45 and up 90

########## plot  ##########
keys = ["Test6_1_2", 'Test6_2_2', "Test6_3_2", "Test6_4_2", "Test6_5_2","Test6_6_2"]
labels = ["Test6_1_2", 'Test6_2_2_45', "Test6_3_2_90", "Test6_4_2_135", "Test6_5_2_150","Test6_6_2_180"]

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
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
