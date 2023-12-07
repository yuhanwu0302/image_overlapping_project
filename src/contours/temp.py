import cv2 as cv 
import numpy as np
import os 
from draw_contours import *
import pandas as pd
con = {}
for i in range(1060,1123):
    con[i] = True

con[1065,1067,1069,1070,1071,1072,1074,1076,1085,1086,1087,1088,1089,1090,1092,1093,1094,1096,1100,1102,1103,1106,1107,1111,1112,1113,1114,1117,1118]=False


get_all_file_paths()