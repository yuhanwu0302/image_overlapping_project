import cv2 as cv
import numpy as np
import os
import pandas as pd 
from contours.draw_contours import read_contours

#### whole area calculation
idlist = [i for i in range(7000, 7073)]
ans_whloe_dic = {}

for i in idlist:
    con = pd.read_csv(fr"C:\Users\Lab_205\Desktop\overlapping_1\clear\contourfiles\{i}_clear.csv",header=None)

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
    ans_whloe_dic[i] = area
ans_whloe_dic
df = pd.DataFrame(list(ans_whloe_dic.items()), columns=['id', 'whole_area'])
df.to_csv(r'C:\Users\Lab_205\Desktop\ans_area.csv', index=False)

###### down area calculation
