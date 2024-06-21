import cv2 as cv
import numpy as np
import os
import pandas as pd 
from src.contours.draw_contours import read_contours
idlist = [i for i in range(7001, 7019)]


con = pd.read_csv(r"C:\Users\Lab_205\Desktop\overlapping\1-overlapping_image\clear\contourfiles\7000_overlapping_clear.csv",header=None)

con.drop([1], axis=1, inplace=True)
con = con.T
conlist = [i for i in con.iloc[0]]
contour_points_list =[]
for point_str in conlist:
    point = list(map(int, point_str.strip('[]').split()))
    contour_points_list.append(point)
contour_points = np.array(contour_points_list, dtype=np.int32)
contour_points = contour_points.reshape((-1, 1, 2))
area = cv.contourArea(contour_points)
print(area)