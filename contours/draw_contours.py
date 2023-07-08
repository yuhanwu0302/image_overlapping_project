import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
import re
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
def gradient(x,y,interval):
    grad_x =[]
    grad_y =[]
    assert  len(x) == len(y) , "The lengths of the lists x and y are not the same."
    for i in np.linspace(interval,len(x)-1,int(len(x)/interval),dtype='int'):
        diff_x = int(x[i]) - int(x[i-interval])
        diff_y = int(y[i]) - int(y[i-interval])
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


x_li , y_li = readcontours(r"C:/Users/baba/Desktop/image_processing_opencv/contours/contours_2_0627.csv")
gradient(x_li,y_li,10)


# ####### Display the image in excel or jupyter 
# points = np.array([x_li,y_li],dtype=np.int32).T
# ####### save contours to csv 
# points_1 = np.around(points,decimals=0)
# # np.savetxt('points_0627.csv',points_1,delimiter=',')
# ####### save contours_site to csv
# draw = np.zeros([512, 512], dtype=np.uint8)
# contours_site = cv.drawContours(draw, [points], -1, (255, 255, 255), thickness=1)
# contours_site_1 = np.around(contours_site,decimals=2)
# #np.savetxt('contours_site_1_0627.csv',contours_site_1,delimiter=',',fmt='%d')

# plt.imshow(draw, cmap='gray')
# plt.show()
