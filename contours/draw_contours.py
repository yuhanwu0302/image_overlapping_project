import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from point import Point , Gradient
from typing import List

def read_contours(file):
    all_points = []
    with open(file, 'r') as f:
        for row in f.readlines()    :
            result = re.search(r"(\d+)\s+(\d+)", row)
            if result:
                x, y = map(int, result.groups())
                p = Point(x, y)
                all_points.append(p)
    return all_points

def calculate_gradients(points: List[Point], interval: int = 1, move: int = 1) -> List[Gradient]:
    gradients = [Gradient(points[i], points[i - interval]) for i in range(interval, len(points), move)]
    return gradients    

def plot_gradients(gradients: List[Gradient]):
    gradient_values = [gradient.value for gradient in gradients]
    x_values = list(range(len(gradient_values)))
    plt.plot(x_values, gradient_values)
    plt.xlabel('Point Index')
    plt.ylabel('Gradient Value')
    plt.show()

def get_all_file_paths(directory):
    file_paths = []
    for filename in os.listdir(directory):
        full_path = os.path.join(directory, filename)
        if os.path.isfile(full_path):
            file_paths.append(full_path)
    return file_paths



def output_gradvalue(gradlist: List[float],output_image_name: str,output_csv_name: str):
    df = pd.DataFrame(gradlist, columns=['Gradient Value'])
    df.to_csv(output_csv_name, index=False)
    plt.plot(gradlist)
    plt.xlabel('Point Index')
    plt.ylabel('Gradient Value')
    plt.savefig(output_image_name)
    plt.close()


def main():
    targetdir = r'C:\Users\Lab_205\Desktop\image_overlapping_project\dataset_output\find_pattern\2_Chinese horse chestnut\contourfiles'
    output_dir = r'C:\Users\Lab_205\Desktop\testoutput'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_files = get_all_file_paths(targetdir)
    
    for csvfile in all_files:
        points = read_contours(csvfile)
        gradients = calculate_gradients(points, interval=1, move=1)

        base_name = os.path.basename(csvfile)
        plotname, _ = os.path.splitext(base_name)

        output_csv_name = os.path.join(output_dir, f"{plotname}.csv")
        output_image_name = os.path.join(output_dir, f"{plotname}.jpg")

        gradient_values = [gradient.value for gradient in gradients]
        output_gradvalue(gradient_values, output_image_name, output_csv_name)


if __name__ == "__main__":
    main()