import cv2 as cv
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import re

x_li=[]
y_li=[]
with open(r"C:\Users\Lab_205\Desktop\image_processing_opencv\contours\contours_1_0625.csv") as f:
    for row in f.readlines():
        result = re.search(r"(\d+)\s+(\d+)",row)
        x,y = result.group(1) ,result.group(2)
        x_li.append(x)
        y_li.append(y)
        print(x,y)
points = np.array([x_li,y_li],dtype=np.int32).T
####### save contours to csv 
points_1 = np.around(points,decimals=0)
# np.savetxt('points_0627.csv',points_1,delimiter=',')
draw = np.zeros([512, 512], dtype=np.uint8)
####### save contours_site to csv
contours_site = cv.drawContours(draw, [points], -1, (255, 255, 255), thickness=1)
contours_site_1 = np.around(contours_site,decimals=2)
# np.savetxt('contours_site_1_0627.csv',contours_site_1,delimiter=',',fmt='%d')

####### Display the image
plt.imshow(draw, cmap='gray')
plt.show()



#### calculate the gradient
grad_x =[]
grad_y =[]
for i in np.linspace(10,770,77,dtype='int'):
    diff_x = int(x_li[i]) - int(x_li[i-10])
    diff_y = int(y_li[i]) - int(y_li[i-10])
    print(diff_x,diff_y)
    grad_x.append(diff_x)
    grad_y.append(diff_y)
grad_x
grad_y

grad = []
for i in range(len(grad_x)):
    if int(grad_y[i]) == 0:
        result = 0
    else:   
        result = int(grad_x[i])/int(grad_y[i])
    grad.append(result)
    print(grad)

plt.plot(grad)
plt.show

# ###### 麻煩的方法
# contours = pd.read_csv("contours_1.csv", header=None, delimiter=",", usecols=[0], names=['X'])
# assert contours is not None, "Couldn't read the CSV"
# contours.insert(1, column="Y", value=0)
# contours["X"], contours["Y"] = contours['X'].str.split(" ", 1).str
# contours['X'] = contours['X'].str.replace(r'\[|\]', '', regex=True)
# contours['Y'] = contours['Y'].str.replace(r'\[|\]', '', regex=True)

# x = []
# y = []
# for index, row in contours.iterrows():
#     if row['X'] and row['Y']:
#         x.append(float(row['X']))
#         y.append(float(row['Y']))

# draw = np.zeros([512, 512], dtype=np.uint8)
# contour_points = np.array([x, y], dtype=np.int32).T
# cv.drawContours(draw, [contour_points], -1, (255, 255, 255), thickness=2)

# contour_points
# # Display the image
# plt.imshow(draw, cmap='gray')
# plt.show()


