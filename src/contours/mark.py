import os
import cv2 as cv
import numpy as np
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
    marked_points = points[start_point_index:end_point_index + 1]
    closed_contour = marked_points + [points[start_point_index]]
    
    return marked_points, closed_contour





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


def calculate_area(closed_contour: List[Point]):
    contour_points = np.array([(point.x, point.y) for point in closed_contour], dtype=np.int32)
    area = cv.contourArea(contour_points)
    return area

def save_area(area, outputname="", output_dir="output"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_path = os.path.join(output_dir, f"{outputname}_area.csv")
    with open(file_path, "w") as f:
        f.write("area\n")
        f.write(f"{area}\n")
    print(f"Area saved to: {file_path}")

def save_contour_image(closed_contour: List[Point],outputname="", output_dir="output"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    contour_img = np.zeros((512, 512, 3), dtype=np.uint8)
    contour_points = np.array([(point.x, point.y) for point in closed_contour], dtype=np.int32)
    cv.polylines(contour_img, [contour_points], isClosed=True, color=(0, 255, 0), thickness=2)
    file_path = os.path.join(output_dir, f"{outputname}_closed_contour.png")
    cv.imwrite(file_path, contour_img)
    print(f"Contour image saved to: {file_path}")


def save_mark_values(markpart:list, output_name:str, output_dir:str):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)    
    file_name = os.path.join(output_dir, f"{output_name}.csv")
    with open(file_name, "w") as f:
        for value in markpart:
            f.writelines(value)

# setting filepath intervel move and which part to which part  
def main(filepath, start, end):
    points = read_contours(filepath)
    gradients = calculate_gradients(points, 20, 1)
    part1 = plot_gradients(gradients, start, end, savevalue=True)
    _, closed_contour = plot_mark_contours(points, gradients, start, end)

    # plot_mark_contours(points, gradients, interval, move,63,90)
    # part2 = plot_gradients(gradients, interval, move,63,90,savevalue=True)
    #second_derivatives=calculate_second_derivatives(gradients, 1)
    #part1_second = plot_second_derivatives(second_derivatives,start,end,savevalue=True)
    
    
    
    
    return part1,closed_contour #,down_second  
 
def main2(closed_contour, outputname="", output_dir=""):
    area = calculate_area(closed_contour)
    save_area(area, outputname=outputname, output_dir=output_dir)
    save_contour_image(closed_contour, outputname=outputname, output_dir=output_dir)
    grad_to_csv(down, f"{outputname}_down", output_dir)
    print(f"All files saved to {output_dir} with base name {outputname}")

down, closed_contour = main(r'C:\Users\Lab_205\Desktop\overlapping\1-overlapping_image\clear\contourfiles\7072_overlapping_clear.csv', 0, 100)

main2(closed_contour=closed_contour, outputname="7000", output_dir=r"C:\Users\Lab_205\Desktop")





