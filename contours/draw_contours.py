import os
import re
from typing import List
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"({self.x}, {self.y})"

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __ne__(self, other):
        return not self.__eq__(other)

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)

class GradientCalculator:
    GRAD_MAX = 20  # the maximum gradient value

    def __init__(self, points: List[Point]):
        self.points = points
        self.grad = []  # Added to store gradient values
        self.calc_gradient()

    def calc_gradient(self):
        for i, p in enumerate(self.points):
            if i > 0:  # Added to handle the first point
                diff_x = p.x - self.points[i - 1].x
                if diff_x == 0:
                    result = GradientCalculator.GRAD_MAX
                else:
                    diff_y = p.y - self.points[i - 1].y
                    result = diff_y / diff_x
                self.grad.append(result)

def read_contours(file):
    x_li = []
    y_li = []
    with open(file, 'r') as f:
        for row in f.readlines():
            result = re.search(r"(\d+)\s+(\d+)", row)
            x, y = result.group(1), result.group(2)
            x_li.append(x)
            y_li.append(y)
    return list(map(int, x_li)), list(map(int, y_li))

def gradient(x, y, interval, output_name):
    points = [Point(x_val, y_val) for x_val, y_val in zip(x, y)]
    grad_calculator = GradientCalculator(points)
    grad_calculator.calc_gradient()
    grad = grad_calculator.grad
    plt.plot(grad)
    plt.savefig(output_name)
    plt.close()
    return grad

def get_all_file_paths(targetdir):
    file_paths = []
    for dirpath, _, filenames in os.walk(targetdir):
        for file in filenames:
            full_path = os.path.join(dirpath, file)
            file_paths.append(full_path)
    return file_paths

def main():
    targetdir = r'C:\Users\Lab_205\Desktop\image_processing_opencv\plant leaf\Flavia dataset\csvfiles'
    all_files = get_all_file_paths(targetdir)
    output_dir = r'C:\Users\Lab_205\Desktop\image_processing_opencv\plant leaf\Flavia dataset\plot_20'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for csvfile in all_files:
        x_li, y_li = read_contours(csvfile)
        base_name = os.path.basename(csvfile)
        plotname, _ = os.path.splitext(base_name)
        output_image_name = os.path.join(output_dir, f"{plotname}.jpg")
        gradient(x_li, y_li, 20, output_image_name)

    # Display the image (You can uncomment this part if needed)
    # points = np.array([x_li, y_li], dtype=np.int32).T
    # draw = np.zeros([512, 512], dtype=np.uint8)
    # contours_site = cv.drawContours(draw, [points], -1, (255, 255, 255), thickness=1)
    # plt.imshow(draw, cmap='gray')
    # plt.show()

if __name__ == "__main__":
    main()
