def slidingwindow(k,arr):
    start = 0
    sum = 0.0
    result = []
    for end in range((len(arr))):
        sum += arr[end]
        if end >= k-1:
            result.append(sum/k)
            sum -= arr[start]
            start += 1
    return list(map(int,result))


import numpy as np
import os
import matplotlib.pyplot as plt
import re


x_li=[]
y_li=[]
with open(r"C:\Users\baba\Desktop\image_processing_opencv\contours\contours_2_0627.csv") as f:
    for row in f.readlines():
        result = re.search(r"(\d+)\s+(\d+)",row)
        x,y = result.group(1) ,result.group(2)
        x_li.append(x)
        y_li.append(y)
        print(x,y)
x_li = list(map(int,x_li))
y_li = list(map(int,y_li))

sliding_x = slidingwindow(10,x_li)
sliding_y = slidingwindow(10,y_li)


#### calculate the gradient
grad_x =[]
grad_y =[]
for i in np.linspace(10,667,68,dtype='int'):
    diff_x = int(sliding_x[i]) - int(sliding_x[i-10])
    diff_y = int(sliding_y[i]) - int(sliding_y[i-10])
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


