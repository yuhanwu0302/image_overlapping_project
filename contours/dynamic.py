import matplotlib.pyplot as plt
import re 
import numpy as np

x_li=[]
y_li=[]
with open(r'C:\Users\baba\Desktop\image_processing_opencv\contours\contours_2_0627.csv') as f :
    for row in f.readlines():
        print(row)
        result = re.search(r"(\d*)\s(\d*)",row)
        x,y = result.group(1) ,result.group(2)
        x_li.append(x)
        y_li.append(y)
        print(x,y)


points=[]
plt.ion()
for i in range(0,len(x_li)):
    point=[int((x_li[i])),int(y_li[i])]
    points.append(point)
    plt.xlim(0,512)
    plt.ylim(0,512)
    plt.scatter(int(x_li[i]),int(y_li[i]))
    plt.pause(0.0000001)
    
plt.show()







