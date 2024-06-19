import os
import re
import matplotlib.pyplot as plt
from contours.point import Point, Gradient
from typing import List
from contours.draw_contours import calculate_gradients ,calculate_second_derivatives
from contours.output import grad_to_csv
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

def plot_mark_contours(points: List[Point], gradients: List[Gradient], start_percent: float = 0, end_percent: float = 25):
    x = [point.x for point in points]
    y = [point.y for point in points]
    
    plt.figure()
    plt.plot(x, y, label='Contour')

    num_gradients = len(gradients)
    start_gradient_index = int(num_gradients * (start_percent / 100))
    end_gradient_index = int(num_gradients * (end_percent / 100))
    
    start_point_index = start_gradient_index
    end_point_index = end_gradient_index

    for i in range(start_point_index, end_point_index):
        plt.plot([points[i].x, points[(i + 1) % len(points)].x], [points[i].y, points[(i + 1) % len(points)].y], 'r-')

    plt.xlim(0, 800)
    plt.ylim(600, 0)
    plt.legend()
    plt.plot(points[start_point_index].x, points[start_point_index].y, 'go', label='Starting Point')  
    plt.plot(points[end_point_index].x, points[end_point_index].y, 'ro', label='Ending Point')  

    plt.xlim(0, 800)
    plt.ylim(600, 0)
    plt.legend()
    plt.title(f"Gradient Marking from {start_percent}% to {end_percent}%")
    plt.show()






def plot_gradients(gradients: List[Gradient], start_percent: float = 0, end_percent: float = 25, savevalue=False):
    gradient_values = [gradient.value for gradient in gradients]
    x_values = list(range(len(gradient_values)))

    start_index = int(len(gradient_values) * (start_percent / 100))
    end_index = int(len(gradient_values) * (end_percent / 100))
    
    plt.figure()
    plt.plot(x_values, gradient_values, label='Gradient')

    for i in range(start_index, end_index):
        if i + 1 < len(gradient_values):
            plt.plot([x_values[i], x_values[i + 1]], [gradient_values[i], gradient_values[i + 1]], 'r-')
               
    plt.xlabel('Point Index')
    plt.ylabel('Gradient Value')
    plt.legend()    
    plt.show()
    
    if savevalue:
        if start_index < 0:
            overlapping_part =gradient_values[start_index:]+gradient_values[:end_index]
        else:
            overlapping_part = gradient_values[start_index:end_index]
        return overlapping_part
    else:
        return None
    

def plot_second_derivatives(second_derivatives: List[float], start_percent: float = 0, end_percent: float = 25, savevalue: bool = False):
    x_values = list(range(len(second_derivatives)))

    start_index = int(len(second_derivatives) * (start_percent / 100))
    end_index = int(len(second_derivatives) * (end_percent / 100))

    plt.figure()
    plt.plot(x_values, second_derivatives, label='Second Derivative')

    for i in range(start_index, end_index):
        if i + 1 < len(second_derivatives):
            plt.plot([x_values[i], x_values[i + 1]], [second_derivatives[i], second_derivatives[i + 1]], 'r-')

    plt.xlabel('Point Index')
    plt.ylabel('Second Derivative Value')
    plt.legend()
    plt.title(f"Second Derivative from {start_percent}% to {end_percent}%")
    plt.show()

    if savevalue:
        if start_index < 0:
            overlapping_part=second_derivatives[start_index:]+second_derivatives[:end_index]
        else:
            overlapping_part = second_derivatives[start_index:end_index]
        return overlapping_part
    
    else:
        return None




def save_mark_values(markpart:list, output_name:str, output_dir:str):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)    
    file_name = os.path.join(output_dir, f"{output_name}.csv")
    with open(file_name, "w") as f:
        for value in markpart:
            f.writelines(value)

# setting filepath intervel move and which part to which part  
def main(filepath,start,end):
    filepath = filepath
    interval = 20
    move = 1
    start = start
    end = end
    points = read_contours(filepath)
    gradients = calculate_gradients(points, interval, move)
    plot_mark_contours(points, gradients,start,end)
    part1 = plot_gradients(gradients,start,end,savevalue=True)
    # plot_mark_contours(points, gradients, interval, move,63,90)
    # part2 = plot_gradients(gradients, interval, move,63,90,savevalue=True)
    #second_derivatives=calculate_second_derivatives(gradients, 1)
    #part1_second = plot_second_derivatives(second_derivatives,start,end,savevalue=True)
    
    
    
    
    return part1 #,down_second  
 
down = main(r'C:\Users\Lab_205\Desktop\overlapping\1-overlapping_image\clear\contourfiles\7015_overlapping_clear.csv',9,63)

grad_to_csv(down,"7015_down",r"C:\Users\Lab_205\Desktop\overlapping\1-overlapping_image\clear\contourfiles\down")


grad_to_csv(down_second,"70311_second_down",r"C:\Users\Lab_205\Desktop\image_overlapping_project\dataset_output\find_pattern\overlapping_1\contourfiles\grad\down")


