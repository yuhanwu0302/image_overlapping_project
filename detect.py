import cv2
from matplotlib import pyplot as pltd 

image = cv2.imread("image.png")
img_gray  = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
pltd.subplot(1,1,1)
pltd.imshow(img_rgb)
pltd.show()





# Import OpenCV module  
import cv2  
# Import pyplot from matplotlib as plt  
from matplotlib import pyplot as pltd  
# Opening the image from files  
imaging = cv2.imread("image.png")  
# Altering properties of image with cv2  
imaging_gray = cv2.cvtColor(imaging, cv2.COLOR_BGR2GRAY)  
imaging_rgb = cv2.cvtColor(imaging, cv2.COLOR_BGR2RGB)  
# Importing Haar cascade classifier xml data  
xml_data = cv2.CascadeClassifier("XML-data.xml")  
# Detecting object in the image with Haar cascade classifier   
detecting = xml_data.detectMultiScale(imaging_gray,   
                                   minSize = (30, 30))  
# Amount of object detected  
amountDetecting = len(detecting)  
# Using if condition to highlight the object detected  
if amountDetecting != 0:  
    for (a, b, width, height) in detecting:  
        cv2.rectangle(imaging_rgb, (a, b), # Highlighting detected object with rectangle  
                      (a + height, b + width),   
                      (0, 275, 0), 9)  
# Plotting image with subplot() from plt  
pltd.subplot(1, 1, 1)  
# Displaying image in the output  
pltd.imshow(imaging_rgb)  
pltd.show()  