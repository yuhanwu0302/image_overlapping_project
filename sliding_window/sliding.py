import numpy as np
import os
import matplotlib.pyplot as plt
import re

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
def gradient(x,y,interval,output_name):
    grad_x =[]
    grad_y =[]
    assert  len(x) == len(y) , "The lengths of the lists x and y are not the same."
    for i in np.linspace(interval,len(x)-1,int(len(x)/interval),dtype='int'):
        diff_x = int(x[i]) - int(x[i-interval])
        diff_y = int(y[i]) - int(y[i-interval])
        grad_x.append(diff_x)
        grad_y.append(diff_y)
    
    grad = []
    for i in range(len(grad_x)):
        if int(grad_y[i]) == 0:
            result = 0
        else:   
            result = int(grad_x[i])/int(grad_y[i])
        grad.append(result)
    
    plt.plot(grad)
    plt.savefig(output_name)
    plt.close()


    


def get_all_file_paths(targetdir):
    file_paths = []

    # os.walk()返回三個值：目錄的路徑、目錄中的子目錄名稱、目錄中的檔案名稱
    for dirpath, _, filenames in os.walk(targetdir):
        for file in filenames:
            full_path = os.path.join(dirpath, file)
            file_paths.append(full_path)

    return file_paths






### if you want to calculate a lot of grad midify you target dir 
targetdir = r'C:\Users\Lab_205\Desktop\image_processing_opencv\plant leaf\Flavia dataset\csvfiles'
all_files = get_all_file_paths(targetdir)

### if you want to creat new dir please check here!!!!
output_dir = r'C:\Users\Lab_205\Desktop\image_processing_opencv\plant leaf\Flavia dataset\plot_sliding_5,1 grad=10'
if not os.path.exists(output_dir):
    os.makedirs(output_dir) 


for csvfile in all_files:
    x_li , y_li = readcontours(csvfile)
    base_name = os.path.basename(csvfile)
    plotname, _ = os.path.splitext(base_name)
    output_image_name = os.path.join(output_dir, f"{plotname}.jpg")
    sliding_x , x_num = slidingwindow(5,x_li,1)
    sliding_y , y_num = slidingwindow(5,y_li,1)
    gradient(sliding_x,sliding_y,interval=10,output_name=output_image_name)



