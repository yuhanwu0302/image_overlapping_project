import cv2 as cv 
import numpy as np

img = cv.imread("leaf_demo.png")
cv.imshow("clear",img)
cv.waitKey(0)
cv.destroyAllWindows()

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow("gray",gray)
cv.waitKey(0)
cv.destroyAllWindows()

# with open("./gray_pixel.csv", "w") as f:
#     for layer_1 in gray:
#         f.writelines([f"{pixel}," for pixel in layer_1])
#         f.writelines("\n")

_ ,thresh = cv.threshold(gray,254,255,cv.THRESH_TOZERO_INV)
contours,hierarchy=cv.findContours(thresh,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(img,contours,-1,(0,0,255),3)
cv.imshow("idk",img)
cv.waitKey(0)
cv.destroyAllWindows()






