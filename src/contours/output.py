import cv2 as cv
import numpy as np
import os 
import glob
import re
import matplotlib.pyplot as plt
from typing import List
from point import Point , Gradient
import pandas as pd 

def get_all_file_paths(targetdir: str):
    file_paths = []
    for dirpath, _, filenames in os.walk(targetdir):
        for file in filenames:
            full_path = os.path.join(dirpath, file)
            file_paths.append(full_path)

    return file_paths

def file_name(file_path: str, output_dir: str, note:str =""):
    base_name = os.path.basename(file_path)
    filename, extension = os.path.splitext(base_name)
    if extension.lower() == '.csv':
        output_csv_name = os.path.join(output_dir, f"{filename}{note}.csv")
        return output_csv_name
    
    elif extension.lower() == '.jpg':
        output_image_name = os.path.join(output_dir, f"{filename}{note}.jpg")
        return output_image_name
    
    else:
        return None



def contour_to_csv(contour_points, output_name:str, output_dir:str):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)    
    file_name = os.path.join(output_dir, f"{output_name}.csv")
    with open(file_name, "w") as f:
        for layer_1 in contour_points:
            for layer_2 in layer_1:
                f.writelines([f"{pixel}," for pixel in layer_2])
                f.writelines("\n")

def draw_contour_to_img(contour_points:np.array, output_name:str,output_dir:str,size=(800, 600)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    canvas = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    contour_int = np.round(contour_points).astype(np.int32)
    canvas=cv.polylines(canvas, [contour_int], isClosed=True, color=(255, 255, 255), thickness=2)
    cv.imwrite(f"{output_name}",canvas)


def gradients_plot(gradients: List[Gradient]):
    gradient_values = [gradient.value for gradient in gradients]
    x_values = list(range(len(gradient_values)))
    plt.plot(x_values, gradient_values)
    plt.xlabel('Point Index')
    plt.ylabel('Gradient Value')
    plt.show()

def grad_to_csv(gradlist: List[float],output_csv_name: str):
    df = pd.DataFrame(gradlist)
    df.to_csv(output_csv_name, index=False,encoding="utf-8",header=False)
 



