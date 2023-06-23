import cv2 as cv
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import re

x_li=[]
y_li=[]
with open("contours_1.csv") as f:
    for row in f.readlines():
        result = re.search(r"(\d+)\s+(\d+)",row)
        x,y = result.group(1) ,result.group(2)
        x_li.append(x)
        y_li.append(y)
        print(x,y)
points = np.array([x_li,y_li],dtype=np.int32).T
draw = np.zeros([512, 512], dtype=np.uint8)
cv.drawContours(draw, [points], -1, (255, 255, 255), thickness=2)
points
# Display the image
plt.imshow(draw, cmap='gray')
plt.show()



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


