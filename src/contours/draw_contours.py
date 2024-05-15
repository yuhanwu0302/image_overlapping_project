import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from contours.point import Point, Gradient
from typing import List


def read_contours(file):
    all_points = []
    with open(file, 'r') as f:
        for row in f.readlines():
            result = re.search(r"(\d+)\s+(\d+)", row)
            if result:
                x, y = map(int, result.groups())
                p = Point(x, y)
                all_points.append(p)
    return all_points


def calculate_gradients(points: List[Point], interval: int = 1, move: int = 1) -> List[Gradient]:
    n = len(points)
    gradients = [Gradient(points[i], points[(i + interval) % n]) for i in range(0, n, move)]
    return gradients


def calculate_second_derivatives(gradients: List[Gradient], interval: int = 1) -> List[float]:
    n = len(gradients)
    second_derivatives = []
    for i in range(0, n, interval):
        first_gradient = gradients[i].value
        second_gradient = gradients[(i + interval) % n].value
        second_derivative = (second_gradient - first_gradient) / interval
        second_derivatives.append(second_derivative)
    return second_derivatives


def plot_gradients(gradients: List[Gradient]):
    gradient_values = [gradient.value for gradient in gradients]
    x_values = list(range(len(gradient_values)))
    plt.plot(x_values, gradient_values)
    plt.xlabel('Point Index')
    plt.ylabel('Gradient Value')
    plt.show()


def plot_second_derivatives(second_derivatives: List[float]):
    x_values = list(range(len(second_derivatives)))
    plt.plot(x_values, second_derivatives)
    plt.xlabel('Point Index')
    plt.ylabel('Second Derivative Value')
    plt.show()


def get_all_file_paths(directory):
    file_paths = []
    for filename in os.listdir(directory):
        full_path = os.path.join(directory, filename)
        if os.path.isfile(full_path):
            file_paths.append(full_path)
    return file_paths


def output_gradvalue(gradlist: List[float], output_image_name: str, output_csv_name: str):
    df = pd.DataFrame(gradlist)
    df.to_csv(output_csv_name, index=False, encoding="utf-8")
    plt.plot(gradlist)
    plt.xlabel('Point Index')
    plt.ylabel('Gradient Value')
    plt.savefig(output_image_name)
    plt.close()


def output_second_derivative(second_derivatives: List[float], output_image_name: str, output_csv_name: str):
    df = pd.DataFrame(second_derivatives)
    df.to_csv(output_csv_name, index=False, encoding="utf-8")
    plt.plot(second_derivatives)
    plt.xlabel('Point Index')
    plt.ylabel('Second Derivative Value')
    plt.savefig(output_image_name)
    plt.close()


def main():
    targetdir = r'C:\Users\Lab_205\Desktop\overlapping_1\clear\contourfiles'
    output_dir = r'C:\Users\Lab_205\Desktop\overlapping_1\clear\contourfiles\grad_20'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_files = get_all_file_paths(targetdir)

    for csvfile in all_files:
        points = read_contours(csvfile)
        gradients = calculate_gradients(points, interval=20, move=1)
        second_derivatives = calculate_second_derivatives(gradients, interval=1)

        base_name = os.path.basename(csvfile)
        plotname, _ = os.path.splitext(base_name)

        grad_output_csv_name = os.path.join(output_dir, f"{plotname}_gradient.csv")
        grad_output_image_name = os.path.join(output_dir, f"{plotname}_gradient.jpg")
        output_gradvalue([gradient.value for gradient in gradients], grad_output_image_name, grad_output_csv_name)

        second_derivative_output_csv_name = os.path.join(output_dir, f"{plotname}_second_derivative.csv")
        second_derivative_output_image_name = os.path.join(output_dir, f"{plotname}_second_derivative.jpg")
        output_second_derivative(second_derivatives, second_derivative_output_image_name, second_derivative_output_csv_name)
        

if __name__ == "__main__":
    main()
