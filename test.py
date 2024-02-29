import numpy as np
def clean_up_rows(row):
    row = row.strip()
    if row:
        try:
            return float(row)
        except ValueError:
            return None



    
def read(path):
    values = []
    with open(path, "r") as f:
        for row in f:
            cleaned_value = clean_up_rows(row)
            if cleaned_value is not None:  
                values.append(cleaned_value)
    array = np.array(values)
    values_array = array.reshape(-1, 1)  
    return values_array
    
            


read(r"C:\Users\Lab_205\Desktop\image_overlapping_project\src\contours\test\grad\correctgrad\slid20_1\1060_clear_slid.csv")




import glob

glob.glob(r"C:\Users\Lab_205\Desktop\image_overlapping_project\src\dataset_output\find_pattern\2_Chinese horse chestnut\contourfiles")

def read_file_to_array(file_path):
    values = []
    with open(file_path, "r") as f:
        for row in f:
            row = row.strip()  
            if row:  
                values.append(float(row))
    return np.array(values)

read_file_to_array(r"C:\Users\Lab_205\Desktop\image_overlapping_project\src\contours\test\grad\correctgrad\slid20_1\1060_clear_slid.csv")