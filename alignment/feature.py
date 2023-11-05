import cv2 as cv 
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import procrustes
import pandas as pd 
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
# 獲取專案根目錄的路徑
project_dir = os.path.dirname(current_dir)
# 添加專案根目錄到 sys.path
if project_dir not in sys.path:
    sys.path.append(project_dir)

from contours.draw_contours import read_contours

num_of_points = 100
resampled_contours = []

contour_files = glob.glob(r"C:\Users\Lab_205\Desktop\image_overlapping_project\dataset_output\find_pattern\2_Chinese horse chestnut\contourfiles01\*.csv")

for file in contour_files:
    df = pd.read_csv(file,header=None)
    points =

 
root_path = r"C:\Users\Lab_205\Desktop\image_overlapping_project"

# 遍歷目錄樹
for dirpath, dirnames, filenames in os.walk(root_path):
    # 打印目錄路徑
    print(f"Found directory: {dirpath}")
    # 打印所有子目錄
    for dirname in dirnames:
        print(f"Subdirectory: {dirname}")
    # 打印所有文件
    for filename in filenames:
        print(f"File: {filename}")


import os

def list_files(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print(f'{subindent}{f}')

# 使用這個函數並提供您想要列出的資料夾路徑
list_files(r"C:\Users\Lab_205\Desktop\image_overlapping_project")