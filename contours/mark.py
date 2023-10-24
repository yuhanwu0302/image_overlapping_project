import os
import re
import matplotlib.pyplot as plt
from point import Point, Gradient
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
    gradients = [Gradient(points[i], points[i - interval]) for i in range(interval, len(points), move)]
    return gradients

def plot_contours_and_gradients(points: List[Point], gradients: List[Gradient], interval: int, move: int, start_percent: float = 0, end_percent: float = 25, image_size: int = 512):
    x = [point.x for point in points]
    y = [point.y for point in points]
    plt.figure()
    plt.plot(x, y, label='Contour')

    start_index = int(len(points) * (start_percent / 100))
    end_index = int(len(points) * (end_percent / 100))
    
    for i in range(start_index, end_index, move):
        if i + interval < len(points):
            plt.plot([points[i].x, points[i + interval].x], [points[i].y, points[i + interval].y], 'r-')

    plt.xlim(0, image_size)
    plt.ylim(0, image_size)
    plt.legend()
    plt.show()


def plot_gradients(gradients: List[Gradient], interval: int, move: int, start_percent: float = 0, end_percent: float = 25):
    gradient_values = [gradient.value for gradient in gradients]
    x_values = list(range(len(gradient_values)))

    start_index = int(len(gradient_values) * (start_percent / 100))
    end_index = int(len(gradient_values) * (end_percent / 100))
    
    plt.figure()
    plt.plot(x_values, gradient_values, label='Gradient')

    for i in range(start_index, end_index, move):
        if i + 1 < len(gradient_values):
            plt.plot([x_values[i], x_values[i + 1]], [gradient_values[i], gradient_values[i + 1]], 'r-')

    plt.xlabel('Point Index')
    plt.ylabel('Gradient Value')
    plt.legend()
    plt.show()


# setting filepath intervel move and which part to which part  
def main():
    filepath = r'C:\Users\Lab_205\Desktop\image_overlapping_project\dataset_output\find_pattern\2_Chinese horse chestnut\contourfiles\1069_clear.csv'
    interval = 10
    move = 5
    points = read_contours(filepath)
    gradients = calculate_gradients(points, interval, move)
    plot_contours_and_gradients(points, gradients, interval, move,0,3)
    plot_gradients(gradients, interval, move,0,3)

if __name__ == "__main__":
    main()
