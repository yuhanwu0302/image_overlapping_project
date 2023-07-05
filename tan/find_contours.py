import cv2 as cv 
import numpy as np
import os 

img = cv.imread(r"C:\Users\baba\Desktop\image_processing_opencv\tan\leaf_2.jpg")
img = cv.resize(img,(512,512))
cv.imshow("clear",img)
cv.waitKey(0)
cv.destroyAllWindows()

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow("gray",gray)
cv.waitKey(0)
cv.destroyAllWindows()

_ ,thresh = cv.threshold(gray,20,255,cv.THRESH_TOZERO)
contours,hierarchy=cv.findContours(thresh,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
outer_contour = contours[0]  # 最外層的輪廓
cv.drawContours(img,[outer_contour],0,(0,0,255),1)
cv.imshow("idk",img)
cv.waitKey(0)
cv.destroyAllWindows()

with open("./contours_2_0627.csv", "w") as f:
    for layer_1 in contours:
        for layer_2 in layer_1:
            f.writelines([f"{pixel}," for pixel in layer_2])
            f.writelines("\n")