import glob
import os
from itertools import product

import numpy as np
import pandas as pd
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from tqdm.auto import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KernelDensity
#2  1060~ 1122
#5  1195~ 1267
#18 2347~ 2423



def run(leaf_name, grad_rotate, dir,selected_ids):


    ### TODO: need change dir name  ###
    np.set_printoptions(suppress = True)
    all_gradient_dict = {}

    file_list = get_all_file_path(dir)
    gradient_dict = get_all_gradients(file_list)
    all_gradient_dict[rf"{leaf_name}_{grad_rotate}"] = gradient_dict
    selected_leaves_grads = ["detect_diff_rotate_90"]
    pair_dist = calculate_pairwise_dist(all_gradient_dict, selected_leaves_grads, selected_ids)

    return pair_dist

def get_all_file_path(image_path):    
    return glob.glob(os.path.join(image_path, "*.csv"))

def get_all_gradients(file_list):
    gradient_dict = {}
    for filepath in file_list:
        data_id = os.path.basename(filepath).strip(".csv")
        gradient_dict[data_id] = load_gradients(filepath)
    return gradient_dict

def load_gradients(file_path):
    with open(file_path, "r") as f:
        gradients = np.array([str_to_float(row) for row in f.readlines()])

    return gradients.reshape(-1, 1)

def str_to_float(r):
    return float(r.strip())

def calculate_pairwise_dist(all_gradient_dict, leaves_grads, ids):
    combined = list(product(leaves_grads, ids))
    # A 2D-dict
    pairwise_dist = defaultdict(dict)
    
    for _, (leaf_grad_1, id_1) in enumerate(combined):
        name_1 = f"{leaf_grad_1}_{id_1}"
        # The distance should be 0
        diagonal_dis = dis(all_gradient_dict[leaf_grad_1][id_1], all_gradient_dict[leaf_grad_1][id_1])
        pairwise_dist[name_1][name_1] = diagonal_dis

        for _, (leaf_grad_2, id_2) in enumerate(combined):
            name_2 = f"{leaf_grad_2}_{id_2}"
            print(name_1, name_2)
            pairwise_dist[name_1][name_2] = dis(all_gradient_dict[leaf_grad_1][id_1], all_gradient_dict[leaf_grad_2][id_2])
            pairwise_dist[name_2][name_1] = pairwise_dist[name_1][name_2]
    
    return pairwise_dist
            
def dis(grad_1, grad_2):
    distance, _ = fastdtw(grad_1, grad_2,dist=euclidean)
    return distance


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


########## plot  ##########


def plot_kde_and_CI(values,CI:int, compare_n_values=10 ,compare=False):
    value = np.array(values).reshape(-1,1)

    kde = KernelDensity(bandwidth=1.0, kernel='gaussian')
    kde.fit(value)

    x = np.linspace(min(value), max(value), 1000).reshape(-1, 1)
    pdf = np.exp(kde.score_samples(x))
    cdf = np.cumsum(pdf) * (x[1] - x[0])

    confidence_level = CI
    lower_bound = x[np.argmax(cdf >= (1-confidence_level)/2)][0]
    upper_bound = x[np.argmax(cdf >= 1-(1-confidence_level)/2)][0]

    value_1d = value.flatten()
    p = sns.histplot(value_1d, kde=True,label='All Data', color='blue')
    if compare:
        compare_value = value_1d[-compare_n_values:]
        sns.histplot(compare_value, kde=False, color='red', label='Last 10 Data', bins=30, alpha=0.7)

    plt.title('test_detect_diff')
    plt.axvline(lower_bound, color="r", linestyle="--", label=f"{confidence_level*100}% CI")
    plt.axvline(upper_bound, color="r", linestyle="--")
    p.set_xlabel("distance")
    p.set_ylabel("freq")
    plt.xlim(0,1800)
    plt.legend()
    plt.show()
    print(f"95% Confidence Interval: ({lower_bound}, {upper_bound})")





for _ in tqdm(range(1)):
    leaf_names=["detect_diff"]
    grad_rotates =["rotate_90"]
    selected_ids = [str(id) + "_clear" for id in list(range(1195, 1220)) + list(range(2114, 2119))]
    for leaf_name in leaf_names:
        for grad_rotate in grad_rotates:
                image_path = fr"../dataset_output/find_pattern/{leaf_name}/contourfiles/grad/{grad_rotate}/" 
                result1 = run(leaf_name,grad_rotate,image_path,selected_ids)
df1 = pd.DataFrame(result1)

values =[]
for i in df1.values:
    for j in i:
        values.append(j)

plot_kde_and_CI(values,0.95,6,True)