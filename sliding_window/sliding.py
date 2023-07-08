import numpy as np
import os
import matplotlib.pyplot as plt
import re

def slidingwindow(k,arr,w=1):
    start = 0
    total = 0.0
    add = k-1 
    result = []
    for end in range((len(arr))):
        total += arr[end]
        if end >= add:
            result.append(total/k)
            add += w
            if w != 1:
                total -= sum(arr[start:start+w])
            else:
                total -= arr[start]
            start += w
            if start >= len(arr):  
                break
    return list(map(int,result)) , len(list(map(int,result)))


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
    
###### read contours 
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



x_li,y_li = readcontours(r'C:\Users\baba\Desktop\image_processing_opencv\contours\contours_2_0627.csv')
sliding_x , x_num = slidingwindow(10,x_li,1)
sliding_y , y_num = slidingwindow(10,y_li,1)
gradient(sliding_x,sliding_y,10)
