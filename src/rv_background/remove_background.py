import sys
import os
import re
import glob
import cv2 as cv
import numpy as np

def create_savename(file, save_folder):
    basename = os.path.basename(file)
    basename = re.sub(".jpg$", "_clear.jpg", basename, flags=re.IGNORECASE)
    savename = os.path.join(save_folder, basename)
    return savename

def remove_background(file, save_folder):
    filename = file
    if file is None:
        print(f"Can not read the image: {filename}")
        return False
    
    img = cv.imread(file)
    if img is None:
        print(f"Can not read the image: {filename}")
        return False

    img = cv.resize(img, (512, 512))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (55, 55), 0)
    _, thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    mask = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
    
    # 檢查是否存在相同名稱的檔案
    savename = create_savename(filename, save_folder)
    if os.path.exists(savename):
        print(f"The file {os.path.basename(savename)} already exists.")
        return False  # 返回 False 表示操作失敗

    cv.imwrite(savename, mask)
    print(f"{filename} processed successfully.")
    return True  # 返回 True 表示操作成功

def process_images(input_path):
    if os.path.isdir(input_path):  
        files = [file for file in os.listdir(input_path) if file.lower().endswith(('.jpg', '.jpeg', '.png'))]
        save_folder = os.path.join(input_path, "clear")
        os.makedirs(save_folder, exist_ok=True)
        success_count = 0
        for file in files:
            file_path = os.path.join(input_path, file)
            if remove_background(file_path, save_folder):
                success_count += 1
            else:
                print(f"Failed to process {file}")
        return success_count == len(files)  
    elif os.path.isfile(input_path):  
        save_folder = os.path.join(os.path.dirname(input_path), "clear")
        os.makedirs(save_folder, exist_ok=True)
        return remove_background(input_path, save_folder)
    else:
        print("The input path does not exist!")
        return False

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python remove_background.py <path_to_image_or_directory>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    process_images(input_path)