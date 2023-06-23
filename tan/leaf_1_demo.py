import cv2 as cv 
import numpy as np
import os 

img = cv.imread("leaf_1_demo.jpg")
img = cv.resize(img,(512,512))
cv.imshow("clear",img)
cv.waitKey(0)
cv.destroyAllWindows()

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow("gray",gray)
cv.waitKey(0)
cv.destroyAllWindows()

_ ,thresh = cv.threshold(gray,20,255,cv.THRESH_TOZERO)
contours,hierarchy=cv.findContours(thresh,cv.RETR_TREE,cv.CHAIN_APPROX_NONE)
cv.drawContours(img,contours,-1,(0,0,255),3)
cv.imshow("idk",img)
cv.waitKey(0)
cv.destroyAllWindows()

with open("./contours_1.csv", "w") as f:
    for layer_1 in contours:
        for layer_2 in layer_1:
            f.writelines([f"{pixel}," for pixel in layer_2])
            f.writelines("\n")