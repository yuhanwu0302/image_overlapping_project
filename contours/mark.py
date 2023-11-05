import os
import re
import matplotlib.pyplot as plt
from point import Point, Gradient
from typing import List
from draw_contours import calculate_gradients

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

def plot_mark_contours(points: List[Point], gradients: List[Gradient], interval: int, move: int, start_percent: float = 0, end_percent: float = 25, image_size: int = 512):
    x = [point.x for point in points]
    y = [point.y for point in points]
    plt.figure()
    plt.plot(x, y, label='Contour')

    num_gradients = len(gradients)
    start_gradient_index = int(num_gradients * (start_percent / 100))
    end_gradient_index = int(num_gradients * (end_percent / 100))
    
    start_point_index = start_gradient_index * move
    end_point_index = (end_gradient_index * move + interval) % len(points)

    for i in range(start_point_index, end_point_index, move):
        plt.plot([points[i].x, points[(i + interval) % len(points)].x], [points[i].y, points[(i + interval) % len(points)].y], 'r-')

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
    move = 1
    points = read_contours(filepath)
    gradients = calculate_gradients(points, interval, move)
    plot_mark_contours(points, gradients, interval, move,40,80)
    plot_gradients(gradients, interval, move,40,80)

if __name__ == "__main__":
    main()
