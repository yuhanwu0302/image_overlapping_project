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

    file_list = get_all_csv_path(dir)
    gradient_dict = get_all_gradients(file_list)
    all_gradient_dict[rf"{leaf_name}_{grad_rotate}"] = gradient_dict
    selected_leaves_grads = [rf"{leaf_name}_{grad_rotate}"]
    pair_dist = calculate_pairwise_dist(all_gradient_dict, selected_leaves_grads, selected_ids)

    return pair_dist

def get_all_csv_path(image_path):    
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
    distance,_ = fastdtw(grad_1, grad_2,dist=euclidean)
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
def plot_kde_and_CI(title:str,values,CI:int, compare_n_values=0 ,compare=False):
    """
    Plot KDE and confidence interval for given data.

    Parameters:
    - title (str): The title of the plot.
    - values (gradient values[float])
    - CI (float): The confidence interval level, between 0 and 1 (e.g., 0.95 for 95% CI).
    - compare_n_values (int, optional): Number of values to compare. Defaults to 0.
    - compare (bool, optional): Whether to compare the last n values. Defaults to False.
    """
    if not values:
        raise ValueError("The 'values' parameter cannot be empty.")
    if not (0 < CI < 1):
        raise ValueError("The 'CI' parameter must be between 0   1.")
    
    if compare:
        if compare_n_values <= 0:
            raise ValueError("The 'compare_n_values' must be greater than 0 and less than the length of 'values'.")
        value = values[:-compare_n_values]
        detect_value = values[-compare_n_values:]
    else:
        value = values
    value = np.array(value).reshape(-1,1)
    kde = KernelDensity(bandwidth=1.0, kernel='gaussian')
    kde.fit(value)

    x = np.linspace(min(value), max(value), 1000).reshape(-1, 1)
    pdf = np.exp(kde.score_samples(x))
    cdf = np.cumsum(pdf) * (x[1] - x[0])

    confidence_level = CI
    lower_bound = x[np.argmax(cdf >= (1-confidence_level)/2)][0]
    upper_bound = x[np.argmax(cdf >= 1-(1-confidence_level)/2)][0]

    value_1d = value.flatten()
    p = sns.histplot(value_1d, kde=True,label='All Data', color='blue',bins =30)
    if compare:
        detect_value = np.array(detect_value).reshape(-1, 1)
        detect_value1d = detect_value.flatten()
        sns.histplot(detect_value1d, kde=False, color='red', label=f'{title}_Data', bins=30, alpha=0.7)

    plt.title(title)
    plt.axvline(lower_bound, color="r", linestyle="--", label=f"{confidence_level*100}% CI")
    plt.axvline(upper_bound, color="r", linestyle="--")
    p.set_xlabel("distance")
    p.set_ylabel("freq")
    plt.legend()
    plt.show()
    print(f"{CI*100}% Confidence Interval: ({lower_bound}, {upper_bound})")

#1195 1268   #1324  #1160
def temp_run():
    for _ in tqdm(range(1)):
        leaf_names=["overlapping"]
        grad_rotates =["correctgrad"]
        selected_ids1 = [str(id) + "_clear" for id in list(range(1195, 1200)) + list(range(6000, 6003))]
        # selected_ids2 = [str(id) + "_clear" for id in list(range(1324, 1330))]
        # select_id_lists = [selected_ids1,selected_ids2]
        select_id_lists=[selected_ids1]
        all_results = {}
        for leaf_name,grad_rotate,selected_ids in zip(leaf_names,grad_rotates,select_id_lists):
                        image_path = fr"../../dataset_output/find_pattern/{leaf_name}/contourfiles/grad/{grad_rotate}/" 
                        result = run(leaf_name,grad_rotate,image_path,selected_ids)
                        
                        key = f"{leaf_name}_{grad_rotate}"
                        all_results[key] = result

    df  = pd.DataFrame(all_results['detect_diff_correctgrad'])
    df1 = pd.DataFrame(all_results['overlapping_correctgrad'])
    df1.shape


    temp = np.zeros((34, 34))
    index = 0
    for i in range(34):
        for j in range(34):
            temp[i,j] = df1.iloc[i,j]
            index += 1


    # leaf1_self_dis
    leaf1_dis = []  
    for i in range(0,31):
        for j in range(i+1,31):
            leaf1_dis.append(temp[i,j])

    #leaf2_self_dis
    leaf2_dis = []
    for i in range(31,34):
        for j in range(i+1,34):
            leaf2_dis.append(temp[i,j])
    # difference leaf dis 
    diff_dis = []
    for i in range(0,31):
        for j in range(31,34):
            diff_dis.append(temp[i,j])

    len(diff_dis)
    dis_list=[]
    dis_list = leaf1_dis + diff_dis
    plot_kde_and_CI('diff_93',dis_list,0.55,compare_n_values=93,compare=True)


