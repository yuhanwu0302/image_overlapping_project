import numpy as np
import os
import matplotlib.pyplot as plt
import re
import sys
import csv
import pandas as pd
sys.path.append(r'C:\Users\Lab_205\Desktop\image_overlapping_project')
sys.path.append(r'C:\Users\Lab_205\Desktop\image_overlapping_project\src')
sys.path.append(r'C:\Users\Lab_205\Desktop\image_overlapping_project\src\contours')
os.sys.path
from src.contours.draw_contours import *
del globals()['main']

# Need to modify import contours method !!!!!!

def slidingwindow(k,arr,w=1):
    start = 0
    total = 0.0
    add = k-1 
    result = []
    for end in range((len(arr))):
        total += arr[end]
        if end >= add:
            result.append(total/k)
            add += w
            if w != 1:
                total -= sum(arr[start:start+w])
            else:
                total -= arr[start]
            start += w
            if start >= len(arr):  
                break
    return list(map(int,result)) , len(list(map(int,result)))

def main():
    points =read_contours(r"C:\Users\Lab_205\Desktop\image_overlapping_project\dataset_output\find_pattern\2_Chinese horse chestnut\contourfiles\1060_clear.csv")
    gradients = calculate_gradients(points,interval = 15 , move=1)
    gradient_values = [gradient.value for gradient in gradients]
    sliding_values,_ = slidingwindow(5,gradient_values,1)
    plt.plot(sliding_values)
    plt.show()

if __name__ == "__main__":
    main()


def slid(input, output,step):
    ### if you want to calculate a lot of grad modify you target dir 
    targetdir = input
    #r'C:\Users\Lab_205\Desktop\image_overlapping_project\src\contours\test\grad\correctgrad'
    all_files = get_all_file_paths(targetdir)
    all_files = [grad_files for grad_files in all_files if not grad_files.endswith('.jpg')]

    ### if you want to creat new dir please check here!!!!
    output_dir = output
    #r'C:\Users\Lab_205\Desktop\image_overlapping_project\src\contours\test\grad\correctgrad\slid20_2'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir) 


    for csvfile in all_files:
        base_name = os.path.basename(csvfile)
        plotname, _ = os.path.splitext(base_name)
        output_csv_name = os.path.join(output_dir, f"{plotname}_slid.csv")
        output_image_name = os.path.join(output_dir, f"{plotname}_slid.jpg")

    
        values = []
        with open(csvfile, "r") as f:
            reader = f.readlines()
            for row in reader:
                row = float(row)
                values.append(row)
        #set slidingwinsow parameter
        sliding_grad ,_= slidingwindow(20,values,step)
        ###  create csv file ###


        output_gradvalue(sliding_grad, output_image_name, output_csv_name)





paths = [r"C:\Users\Lab_205\Desktop\image_overlapping_project\src\contours\test_2\grad\rotated_45",r"C:\Users\Lab_205\Desktop\image_overlapping_project\src\contours\test_2\grad\rotated_90",r"C:\Users\Lab_205\Desktop\image_overlapping_project\src\contours\test_2\grad\rotated_135",
r"C:\Users\Lab_205\Desktop\image_overlapping_project\src\contours\test_2\grad\rotated_150",r"C:\Users\Lab_205\Desktop\image_overlapping_project\src\contours\test_2\grad\rotated_180"
]      
rotatedegree = [r"slid20_1",r"slid20_2",r"slid20_5",r"slid20_10"]
slidmove = [1,2,5,10]
for i in paths:
    for j ,k in zip(rotatedegree,slidmove):
        full_path = os.path.join(i, j)
        print(i,"\n",full_path,"\n", k)
        slid(i,full_path,k)