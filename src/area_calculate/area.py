import pandas as pd
import numpy as np
import cv2 as cv
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor


estimateid = 7000

dict_name = [i for i in range(7000, 7073) if i != estimateid ]
results_all = {}
each_best_match = {}

for i in dict_name:
    results_df = pd.DataFrame(pd.read_csv(fr'C:\Users\Lab_205\Desktop\overlapping\7000\7000_{i}.csv'))
    results_all[i] = results_df

    distances = list(results_all[i]["distance"])
    min_distance_index = distances.index(min(distances))
    min_size = results_all[i]["size"][min_distance_index]
    min_start_index = results_all[i]["start_index"][min_distance_index]
    min_end_index = results_all[i]["end_index"][min_distance_index]
    min_distance = distances[min_distance_index]
    each_best_match[i] = {"size": min_size, "start_index": min_start_index, "end_index": min_end_index, "distance": min_distance}

each_best_match_df = pd.DataFrame(each_best_match).T
# each_best_match_df.to_csv(f'C:\\Users\\Lab_205\\Desktop\\image_overlapping_project\\dataset_output\\find_pattern\\overlapping_1\\area\\7001_best_match.csv', index=True)
for i in dict_name:
    reference_grad = pd.read_csv(fr'C:\\Users\\Lab_205\\Desktop\\image_overlapping_project\\dataset_output\\find_pattern\\overlapping_1\\contourfiles\\grad_20\\{i}_clear_gradient.csv').values.reshape(-1, 1)
    reference_start_index =int(each_best_match_df[each_best_match_df.index == i]["start_index"].values[0])
    reference_end_index = int(each_best_match_df[each_best_match_df.index == i]["end_index"].values[0])


    plt.plot(reference_grad)
    plt.plot(range(reference_start_index, reference_end_index), reference_grad[reference_start_index:reference_end_index], color='red', label='Subsequence')

    plt.xlabel('index')
    plt.ylabel('gradient value')
    plt.title(f'Reference {i} best match part')
    plt.savefig(fr'C:\Users\Lab_205\Desktop\overlapping\7000\each_best\7000_{i}_best_match.png')
    plt.clf()



###### step 4  refer to reference position 


# 初始化字典來存儲每個ID的DataFrame



area_dic ={"id":[],
            "area":[],
            "contour_area":[],
            "poly_area":[]
    }

gradient_position = {}
for each_id in range(7000, 7073):
    if each_id == estimateid:
        continue    
    contours_data = pd.read_csv(fr'C:\Users\Lab_205\Desktop\overlapping_1\clear\contourfiles\{each_id}_clear.csv', header=None)
    contours_data = contours_data.drop(1, axis=1)
    n = len(contours_data)
    interval = 20
 
    # 初始化字典來存儲每個ID的DataFrame

    # ID列表

    
    grad_index_list = []
    index1_list = []
    index2_list = []

    for grad_index in range(n):
        start_index = grad_index
        end_index = (grad_index + interval) % n
        grad_index_list.append(grad_index)  # 這裡的 +1 是因為你提到的index從1開始
        index1_list.append(start_index)
        index2_list.append(end_index)

    # 將列表轉換為DataFrame
    df = pd.DataFrame({
        "grad_index": grad_index_list,
        "index1": index1_list,
        "index2": index2_list
    })

    # 將DataFrame存儲到字典中
    gradient_position[each_id] = df



    start_grad =each_best_match[each_id]["start_index"]
    end_grad = each_best_match[each_id]["end_index"]
    start_grad_row = gradient_position[each_id].query(f"grad_index == {start_grad}")
    if end_grad > gradient_position[each_id]["grad_index"].max():
        end_grad =end_grad -1
        end_grad_row = gradient_position[each_id].query(f"grad_index == {end_grad}")
    else:
        end_grad_row = gradient_position[each_id].query(f"grad_index == {end_grad}")
    pointA_index =start_grad_row["index1"].values[0]
    pointB_index =end_grad_row["index2"].values[0]

    ####come back to contours data and mark the pointA pointB

    pointA =contours_data.iloc[pointA_index]
    pointB =contours_data.iloc[pointB_index]

    #use open cv to creat a canva import contours data and draw the contours mark the pointA pointB
    #first crea a 512*512 canva



    contours_data[0] = contours_data[0].apply(lambda x: np.fromstring(x.strip('[]'), sep=' ', dtype=int))
    points = np.array(contours_data[0].tolist())
    pointA = np.array(pointA[0].strip("[]").split(), dtype=int)
    pointB = np.array(pointB[0].strip("[]").split(), dtype=int)

    img = np.zeros((512, 512, 3), np.uint8)
    cv.drawContours(img, [points], 0, (255, 255, 255), 1)

    # 画出点A和点B
    cv.circle(img, (pointA[0], pointA[1]), 5, (255, 0, 0), -1)
    cv.circle(img, (pointB[0], pointB[1]), 5, (0, 0, 255), -1)
    cv.line(img, (pointA[0], pointA[1]), (pointB[0], pointB[1]), (255, 0, 0), 1)

    # 定义判断点是否在线段下方的函数
    def is_above_line(point, pointA, pointB):
        return (pointB[0] - pointA[0]) * (point[1] - pointA[1]) - (point[0] - pointA[0]) * (pointB[1] - pointA[1]) > 0

    # 找到位于线段下方的轮廓点
    upper_points = np.array([point for point in points if is_above_line(point, pointA, pointB)])

    # 确保点按顺序排列形成一个封闭的多边形
    if len(upper_points) > 0:
        polygon_points = np.vstack([pointA, upper_points, pointB])
    else:
        polygon_points = np.array([pointA, pointB])

    # 填充多边形区域
    cv.fillPoly(img, [polygon_points], (0, 255, 0))
    cv.imwrite(fr'C:\Users\Lab_205\Desktop\testoutput\{each_id}_polygon.png', cv.fillPoly(img, [polygon_points], (0, 255, 0)))
    cv.imwrite(fr'C:\Users\Lab_205\Desktop\testoutput\{each_id}_all.png', cv.fillPoly(img, [points], (0, 255, 0)))


    contour_area = cv.contourArea(points)
    target_area = cv.contourArea(points) -cv.contourArea(polygon_points)


    area_dic["id"].append({each_id})
    area_dic["area"].append(target_area)
    area_dic["contour_area"].append(contour_area)
    area_dic["poly_area"].append(cv.contourArea(polygon_points))
    area_df = pd.DataFrame(area_dic)
area_df.to_csv(rf'C:\Users\Lab_205\Desktop\overlapping\7000\{estimateid}_esatimate_area.csv', index=False)
# 显示图像
# cv.imshow("Contours", img)
# cv.waitKey(0)
# cv.destroyAllWindows()