import cv2 as cv 
import numpy as np
import os 

def set_starting_point_to_leftmost(contour):
    x = contour[:, 0, 0]
    y = contour[:, 0, 1]
    
    min_index = np.argmin(x)
    
    #最左邊的點成為起點
    reordered_contour = np.roll(contour, -min_index, axis=0)
    
    return reordered_contour


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
    reordered_contours = [set_starting_point_to_leftmost(contour) for contour in contours]

# con= cv.drawContours(img,reordered_contours,-1,(0,0,255),1)
# cv.imshow("con",con)
# cv.waitKey(0)
# cv.destroyAllWindows()


    output_dir = r'C:\Users\Lab_205\Desktop\image_processing_opencv\plant leaf\Flavia dataset\csvfiles'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    
    outputname = os.path.join(output_dir, f"{filename}.csv")
    with open(outputname, "w") as f:
        for layer_1 in reordered_contours:
            for layer_2 in layer_1:
                f.writelines([f"{pixel}," for pixel in layer_2])
                f.writelines("\n")


def get_all_file_paths(targetdir):
    file_paths = []

    # os.walk()返回三個值：目錄的路徑、目錄中的子目錄名稱、目錄中的檔案名稱
    for dirpath, _, filenames in os.walk(targetdir):
        for file in filenames:
            full_path = os.path.join(dirpath, file)
            file_paths.append(full_path)

    return file_paths






targetdir = r'C:\Users\Lab_205\Desktop\image_processing_opencv\plant leaf\Flavia dataset\clear'
all_files = get_all_file_paths(targetdir)

for file_path in all_files:
    findcontours(file_path)
