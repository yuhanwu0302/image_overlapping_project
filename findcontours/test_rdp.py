import cv2 as cv
import csv
import numpy as np
import re
import matplotlib.pyplot as plt
from rdp import rdp

###  rdp可以簡化邊緣位點的數量

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

x_li,y_li = readcontours(r'C:\Users\Lab_205\Desktop\image_processing_opencv\findcontours\contours_2_0627.csv')

num_values = len(x_li)
x_array = np.empty(num_values, dtype=int)
y_array = np.empty(num_values, dtype=int)
for i in range(num_values):
    x_array[i] = x_li[i]

for i in range(num_values):
    y_array[i] = y_li[i]
x_array
y_array
result_array = np.column_stack((x_array, y_array)).reshape(-1)

array = result_array.reshape(677,2)
len(array)

### epsilon 0~2 is ok
rdp_array =rdp(array,epsilon=3)


draw = np.zeros([512, 512], dtype=np.uint8)
contours_site = cv.drawContours(draw, [rdp_array], -1, (255, 255, 255), thickness=1)
plt.imshow(draw, cmap='gray')
plt.show()

rdp_array.shape






    
