import cv2 as cv 
import numpy as np
import os 


def findcontours(path):
## read image
    filename = os.path.basename(path)
    filename,_ = os.path.splitext(filename)

    img = cv.imread(path)
    img = cv.resize(img,(512,512))

# cv.imshow("clear",img)
# cv.waitKey(0)
# cv.destroyAllWindows()


# convert to gray
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# cv.imshow("gray",gray)
# cv.waitKey(0)
# cv.destroyAllWindows()

    _ ,thresh = cv.threshold(gray,0,255,cv.THRESH_TOZERO+cv.THRESH_OTSU)
    contours,hierarchy=cv.findContours(thresh,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)


    con= cv.drawContours(img,contours,-1,(0,0,255),1)
# cv.imshow("con",con)
# cv.waitKey(0)
# cv.destroyAllWindows()
    outputname = f"{filename}.csv"
    with open(outputname, "w") as f:
        for layer_1 in contours:
            for layer_2 in layer_1:
                f.writelines([f"{pixel}," for pixel in layer_2])
                f.writelines("\n")

findcontours(r"C:\Users\Lab_205\Desktop\image_processing_opencv\overlap\leaf1_edges.jpg")
