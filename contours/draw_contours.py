import os
import re
from typing import List, Type





import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def readcontours(file):
    x_li=[]
    y_li=[]
    with open(file,'r') as f:
        for row in f.readlines():
            result = re.search(r"(\d+)\s+(\d+)",row)
            x,y = result.group(1) ,result.group(2)
            x_li.append(x)
            y_li.append(y)
    return list(map(int,x_li)),list(map(int,y_li))


#### calculate the gradient
def gradient(x,y,interval,output_name):
    grad_x =[]
    grad_y =[]
    assert  len(x) == len(y) , "The lengths of the lists x and y are not the same."
    for i in np.linspace(interval,len(x)-1,int(len(x)/interval),dtype='int'):
        diff_x = int(x[i]) - int(x[i-interval])
        diff_y = int(y[i]) - int(y[i-interval])
        grad_x.append(diff_x)
        grad_y.append(diff_y)

    grad = []
    for i in range(len(grad_x)):
        if int(grad_y[i]) == 0:
            result = 0
        else:
            result = int(grad_x[i])/int(grad_y[i])
        grad.append(result)

    plt.plot(grad)
    plt.savefig(output_name)
    plt.close()


class Point():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"({self.x},{self.y})"

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __ne__(self, other):
        return self.x != other.x or self.y != other.y

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)


class Gradient():
    GRAD_MAX = 20  # the maximum gradient value
    def __init__(self, points: List[Point]):
        self.points = points
        # self.calc_gradient()

    def calc_gradient(self):
        for i, p in enumerate(self.points):
            diff_x = None  # TODO: calculation
            if diff_x == 0:
                result = Gradient.GRAD_MAX
            else:
                diff_y = None  # TODO: calculation
                result = diff_y / diff_x
            self.grad.append(result)


def get_all_file_paths(targetdir):
    file_paths = []

    # os.walk()返回三個值：目錄的路徑、目錄中的子目錄名稱、目錄中的檔案名稱
    for dirpath, _, filenames in os.walk(targetdir):
        for file in filenames:
            full_path = os.path.join(dirpath, file)
            file_paths.append(full_path)

    return file_paths





targetdir = r'C:\Users\Lab_205\Desktop\image_processing_opencv\plant leaf\Flavia dataset\csvfiles'
all_files = get_all_file_paths(targetdir)
output_dir = r'C:\Users\Lab_205\Desktop\image_processing_opencv\plant leaf\Flavia dataset\plot_20'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


for csvfile in all_files:
    x_li , y_li = readcontours(csvfile)
    base_name = os.path.basename(csvfile)
    plotname, _ = os.path.splitext(base_name)
    output_image_name = os.path.join(output_dir, f"{plotname}.jpg")
    gradient(x_li,y_li,20,output_image_name)






# ####### Display the image
# points = np.array([x_li,y_li],dtype=np.int32).T
# draw = np.zeros([512, 512], dtype=np.uint8)
# contours_site = cv.drawContours(draw, [points], -1, (255, 255, 255), thickness=1)
# plt.imshow(draw, cmap='gray')
# plt.show()
