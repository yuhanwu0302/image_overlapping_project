import math
import scipy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os 

original_path = r"C:\Users\Lab_205\Desktop\image_overlapping_project\dataset_output\find_pattern\rotated_overlapping\grad20_1"
original_files= os.listdir(r"C:\Users\Lab_205\Desktop\image_overlapping_project\dataset_output\find_pattern\rotated_overlapping\grad20_1")
original_merged_df = pd.DataFrame()
for file_name in os.listdir(original_path):
    if file_name.endswith('.csv'):
        original_file_path = os.path.join(original_path, file_name)
        single_column_series = pd.read_csv(original_file_path, header=None, squeeze=True)
        column_name = os.path.splitext(file_name)[0]
        original_merged_df[column_name] = single_column_series

original_merged_df




down_path = r"C:\Users\Lab_205\Desktop\image_overlapping_project\dataset_output\find_pattern\rotated_overlapping\down"
down_files = os.listdir(r"C:\Users\Lab_205\Desktop\image_overlapping_project\dataset_output\find_pattern\rotated_overlapping\down")
down_merged_df = pd.DataFrame()

for file_name in os.listdir(down_path):
    if file_name.endswith('.csv'):  
        down_file_path = os.path.join(down_path, file_name)
        single_column_series = pd.read_csv(down_file_path, header=None, squeeze=True)
        column_name = os.path.splitext(file_name)[0] 
        down_merged_df[column_name] = single_column_series

data_mean = down_merged_df.mean(axis=0)
data_var = down_merged_df.var(axis=0)

down_merged_df


