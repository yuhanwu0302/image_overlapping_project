import cv2 as cv
import numpy as np
import os
import glob
import re
folder = r"C:\Users\Lab_205\Desktop\image_overlapping_project\dataset_output\find_pattern\overlapping\raw"

def create_savename(file, save_folder):
    basename = os.path.basename(file)
    basename = os.path.join(save_folder, basename)
    basename = re.sub(".jpg$", "_clear.jpg", basename, flags=re.IGNORECASE)
    savename = os.path.join(folder, basename)
    return savename

def remove_background (file, save_folder):
    filename = file
    if file is None:
        print(f"Failed to load image: {filename}")
        return
    file = cv.imread(file)
    file = cv.resize(file,(512,512))
    gray = cv.cvtColor(file,cv.COLOR_BGR2GRAY)
    blur =cv.GaussianBlur(gray,(55,55),0)
    _, thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    mask = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
    savename = create_savename(filename, save_folder)
    print(savename)
    cv.imwrite(savename,mask)

def grab_files(folder):
    files = glob.glob(os.path.join(folder, "*.jpg"))
    save_folder = "clear"
    os.makedirs(os.path.join(folder, save_folder), exist_ok=True)
    for file in files:
        remove_background(file, save_folder)

grab_files(folder)
