import pandas as pd
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

#### create each  best match part and grad match part
estimateids = [i for i in range(7000,7073) if i!= 7001]

for estimateid in estimateids:
    dict_names = [i for i in range(7000, 7073) if i != estimateid and i!= 7001 ]
    results_all = {}
    each_best_match = {}

    for dict_name in dict_names:
        results_df = pd.DataFrame(pd.read_csv(fr'C:\Users\Lab_205\Desktop\overlapping\{estimateid}\{estimateid}_{dict_name}.csv'))
        results_all[dict_name] = results_df

        distances = list(results_all[dict_name]["distance"])
        min_distance_index = distances.index(min(distances))
        min_size = results_all[dict_name]["size"][min_distance_index]
        min_start_index = results_all[dict_name]["start_index"][min_distance_index]
        min_end_index = results_all[dict_name]["end_index"][min_distance_index]
        min_distance = distances[min_distance_index]
        each_best_match[dict_name] = {"size": min_size, "start_index": min_start_index, "end_index": min_end_index, "distance": min_distance}

    each_best_match_df = pd.DataFrame(each_best_match).T
    each_best_match_df.to_csv(fr'C:\Users\Lab_205\Desktop\overlapping\{estimateid}\\{estimateid}_best_match.csv', index=True)
    
    for i in dict_names:
        reference_grad = pd.read_csv(fr'C:\\Users\\Lab_205\\Desktop\\image_overlapping_project\\dataset_output\\find_pattern\\overlapping_1\\contourfiles\\grad_20\\{i}_clear_gradient.csv').values.reshape(-1, 1)
        reference_start_index =int(each_best_match_df[each_best_match_df.index == i]["start_index"].values[0])
        reference_end_index = int(each_best_match_df[each_best_match_df.index == i]["end_index"].values[0])


        plt.plot(reference_grad)
        plt.plot(range(reference_start_index, reference_end_index), reference_grad[reference_start_index:reference_end_index], color='red', label='Subsequence')

        plt.xlabel('index')
        plt.ylabel('gradient value')
        plt.title(f'Reference {i} best match part')
        plt.savefig(fr'C:\Users\Lab_205\Desktop\overlapping\{estimateid}\each_best\{estimateid}{i}_best_match.png')
        plt.clf()



    ###### step 4  refer to reference position  create eastimate area data


    # 初始化字典來存儲每個ID的DataFrame



    area_dic ={"id":[],
                "area":[],
                "contour_area":[],
                "poly_area":[]
        }

    gradient_position = {}
    for each_id in range(7000, 7073):
        if each_id ==  estimateid or each_id == 7001:
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
        cv.imwrite(fr'C:\Users\Lab_205\Desktop\overlapping\{estimateid}\area\{each_id}_polygon.png', cv.fillPoly(img, [polygon_points], (0, 255, 0)))
        cv.imwrite(fr'C:\Users\Lab_205\Desktop\overlapping\{estimateid}\area\{each_id}_all.png', cv.fillPoly(img, [points], (0, 255, 0)))


        contour_area = cv.contourArea(points)
        target_area = cv.contourArea(points) -cv.contourArea(polygon_points)


        area_dic["id"].append({each_id})
        area_dic["area"].append(target_area)
        area_dic["contour_area"].append(contour_area)
        area_dic["poly_area"].append(cv.contourArea(polygon_points))
        area_df = pd.DataFrame(area_dic)

    area_df.to_csv(rf'C:\Users\Lab_205\Desktop\overlapping\{estimateid}\{estimateid}_esatimate_area.csv', index=False)
# 显示图像
# cv.imshow("Contours", img)
# cv.waitKey(0)
# cv.destroyAllWindows()


#####3  combine all estimate area data into one file

all_of_estimate_area_data = {}
for estimateid in estimateids:
    all_of_estimate_area_data[estimateid] = pd.read_csv(fr'C:\Users\Lab_205\Desktop\overlapping\{estimateid}\{estimateid}_esatimate_area.csv')

all_of_estimate_area_data
combined_df = pd.DataFrame()

for key, df in all_of_estimate_area_data.items():
    df.insert(0, 'estimate_id', key)
    empty_row = pd.DataFrame([[''] * df.shape[1]], columns=df.columns)
    combined_df = pd.concat([combined_df, empty_row, df], ignore_index=True)

combined_df.to_csv(r'C:\Users\Lab_205\Desktop\overlapping\combined_estimate_area_data.csv', index=False)







############ ALL True Value ######################

true_value_dict = {}
true_values = pd.read_csv(r"C:\Users\Lab_205\Desktop\overlapping\ans_area.csv")
for id, true_value in zip(true_values["id"], true_values["true_value"]):
    true_value_dict[id] = true_value


# true_values = true_values.drop(true_values.index[-1])  # because no 7072 data
# true_value_dict.pop(7072)
############### C.I.   #################
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from sklearn.neighbors import KernelDensity



ld_list  = [i for i in range(7000,7011) if i!= 7001]
for id in ld_list:
    mean_area = all_of_estimate_area_data[id]["area"].mean()
    sem_area = stats.sem(all_of_estimate_area_data[id]["area"])
    medium = np.median(all_of_estimate_area_data[id]["area"])
    #  I want to create two values that is contain 90% of the data use medium dont use mean and sem
    # KDE Estimation
    kde = KernelDensity(kernel='gaussian', bandwidth=2000).fit(all_of_estimate_area_data[id]["area"].values.reshape(-1, 1))
    x_d = np.linspace(min(all_of_estimate_area_data[id]["area"]), max(all_of_estimate_area_data[id]["area"]), 1000)
    log_dens = kde.score_samples(x_d.reshape(-1, 1))

    # calculate 90C.I.
    confidence_interval = np.percentile(all_of_estimate_area_data[id]["area"], [5, 95])

    plt.figure(figsize=(10, 6))
    plt.hist(all_of_estimate_area_data[id]["area"], bins=60, color='blue', alpha=0.5,edgecolor='black', label='Data')

    plt.plot(x_d, np.exp(log_dens) * len(all_of_estimate_area_data[id]["area"]) * (x_d[1] - x_d[0]), color='k', linestyle='dashed')

    # mean and true value
    mean_line = plt.axvline(mean_area, color='k', linestyle='dashed', linewidth=1, label='Mean')
    median_line = plt.axvline(medium, color='b', linestyle='dashed', linewidth=1, label='Median')
    true_value_line = plt.axvline(true_value_dict[id], color='g', linestyle='dashed', linewidth=1, label='True value')

    lower_ci_line = plt.axvline(confidence_interval[0], color='red', linestyle="solid", linewidth=1, label='90% Data Lower')
    upper_ci_line = plt.axvline(confidence_interval[1], color='red', linestyle='solid', linewidth=1, label='90% Data  Upper')

    plt.text(mean_area, plt.ylim()[1]*0.9, f'Mean: {mean_area:.2f}', horizontalalignment='center', verticalalignment='center')
    plt.text(medium, plt.ylim()[1]*0.7, f'Median: {medium:.2f}',color ='blue', horizontalalignment='center', verticalalignment='center')
    plt.text(true_value_dict[id], plt.ylim()[1]*0.8, f'True Value: {true_value_dict[id]}', color='green', horizontalalignment='center', verticalalignment='center')
    plt.text(confidence_interval[0], plt.ylim()[1]*0.6, f'90% Data Lower: {confidence_interval[0]:.2f}', color='red', horizontalalignment='center', verticalalignment='center')
    plt.text(confidence_interval[1], plt.ylim()[1]*0.6, f'90% Data Upper: {confidence_interval[1]:.2f}', color='red', horizontalalignment='center', verticalalignment='center')

    plt.xlabel(f'Estimate {id} Area')
    plt.ylabel('Count')
    plt.legend()
    plt.title(f'Estimate {id} Area with KDE-based 90% Data')
    plt.show()
    plt.close()
    plt.savefig(rf"C:\Users\Lab_205\Desktop\thesis_img\result_img\{id}_area_kde.png")
    plt.close()
#######################################################################  log 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.neighbors import KernelDensity

# 處理NaN值
def preprocess_data(data):
    data = data.dropna()
    # 確保所有值都是正數，否則進行適當的處理
    data = data[data > 0]
    return data
# 例子使用
ld_list  = [i for i in range(7000,7073) if i!= 7001]
for id in ld_list:
    # 對數轉換面積數據
    data = preprocess_data(all_of_estimate_area_data[id]["area"])
    log_area_data = np.log(data)
    
    mean_log_area = log_area_data.mean()
    sem_log_area = stats.sem(log_area_data)
    median_log_area = np.median(log_area_data)

    # KDE Estimation
    kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(log_area_data.values.reshape(-1, 1))
    x_d = np.linspace(min(log_area_data), max(log_area_data), 1000)
    log_dens = kde.score_samples(x_d.reshape(-1, 1))

    # Calculate 90% C.I. in log space
    log_confidence_interval = np.percentile(log_area_data, [5, 95])

    plt.figure(figsize=(10, 6))
    plt.hist(log_area_data, bins=60, color='blue', alpha=0.5, edgecolor='black', label='Log Data')

    plt.plot(x_d, np.exp(log_dens) * len(log_area_data) * (x_d[1] - x_d[0]), color='k', linestyle='dashed')

    # Mean, median and true value (in log scale)
    mean_line = plt.axvline(mean_log_area, color='k', linestyle='dashed', linewidth=1, label='Mean')
    median_line = plt.axvline(median_log_area, color='b', linestyle='dashed', linewidth=1, label='Median')
    true_value_log = np.log(true_value_dict[id])
    true_value_line = plt.axvline(true_value_log, color='g', linestyle='dashed', linewidth=1, label='True value')

    lower_ci_line = plt.axvline(log_confidence_interval[0], color='red', linestyle="solid", linewidth=1, label='90% Data Lower')
    upper_ci_line = plt.axvline(log_confidence_interval[1], color='red', linestyle='solid', linewidth=1, label='90% Data Upper')

    plt.text(mean_log_area, plt.ylim()[1]*0.9, f'Mean: {mean_log_area:.2f}', horizontalalignment='center', verticalalignment='center')
    plt.text(median_log_area, plt.ylim()[1]*0.7, f'Median: {median_log_area:.2f}', color='blue', horizontalalignment='center', verticalalignment='center')
    plt.text(true_value_log, plt.ylim()[1]*0.8, f'True Value: {np.log(true_value_dict[id])}', color='green', horizontalalignment='center', verticalalignment='center')
    plt.text(log_confidence_interval[0], plt.ylim()[1]*0.6, f'90% Data Lower: {log_confidence_interval[0]:.2f}', color='red', horizontalalignment='center', verticalalignment='center')
    plt.text(log_confidence_interval[1], plt.ylim()[1]*0.6, f'90% Data Upper: {log_confidence_interval[1]:.2f}', color='red', horizontalalignment='center', verticalalignment='center')

    plt.xlabel(f'Estimate {id} Log Area')
    plt.ylabel('Count')
    plt.legend()
    plt.title(f'Estimate {id} Log Area with KDE-based 90% Data')
    plt.savefig(rf"C:\Users\Lab_205\Desktop\thesis_img\result_img\{id}_log_area_kde.png")
    plt.close()


##############  ECDF 
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import stats
# from sklearn.neighbors import KernelDensity

# # ECDF and DKW confidence band functions (using provided code)
# from statsmodels.distributions.empirical_distribution import ECDFDiscrete , _conf_set

# def dkw_confidence_band(data, alpha=0.1):
#     n = len(data)
#     epsilon = np.sqrt(np.log(2 / alpha) / (2 * n))
    
#     ecdf = ECDFDiscrete(data)
#     F_hat = ecdf.y[1:]  # ECDF returns values with an extra initial zero
    
#     lower_band = np.maximum(F_hat - epsilon, 0)
#     upper_band = np.minimum(F_hat + epsilon, 1)
    
#     return ecdf.x[1:], F_hat, lower_band, upper_band

# def check_data_within_bands(data, lower_band, upper_band, ecdf):
#     data_cdf_values = ecdf(data)
#     within_bands = (data_cdf_values >= lower_band.min()) & (data_cdf_values <= upper_band.max())
#     return within_bands

# # Example usage for KDE and PDF
# ld_list  = [7000]  # Example with 7000
# for id in ld_list:
#     data = all_of_estimate_area_data[id]["area"]
#     mean_area = data.mean()
#     sem_area = stats.sem(data)
#     medium = np.median(data)

#     # KDE Estimation
#     kde = KernelDensity(kernel='gaussian', bandwidth=2000).fit(data.values.reshape(-1, 1))
#     x_d = np.linspace(min(data), max(data), 1000)
#     log_dens = kde.score_samples(x_d.reshape(-1, 1))
#     pdf_values = np.exp(log_dens)

#     # Calculate 90% C.I.
#     confidence_interval = np.percentile(data, [5, 95])

#     plt.figure(figsize=(10, 6))
#     plt.plot(x_d, pdf_values, color='blue', label='PDF')
#     plt.hist(data, bins=60, color='blue', alpha=0.5,edgecolor='black', label='Data')
#     plt.axvline(mean_area, color='k', linestyle='dashed', linewidth=1)
#     plt.axvline(medium, color='b', linestyle='dashed', linewidth=1)
#     plt.axvline(confidence_interval[0], color='red', linestyle="solid", linewidth=1)
#     plt.axvline(confidence_interval[1], color='red', linestyle='solid', linewidth=1)

#     plt.text(mean_area, plt.ylim()[1]*0.9, f'Mean: {mean_area:.2f}', horizontalalignment='center', verticalalignment='center')
#     plt.text(medium, plt.ylim()[1]*0.7, f'Median: {medium:.2f}', horizontalalignment='center', verticalalignment='center')
#     plt.text(confidence_interval[0], plt.ylim()[1]*0.6, f'90% CI Lower: {confidence_interval[0]:.2f}', color='red', horizontalalignment='center', verticalalignment='center')
#     plt.text(confidence_interval[1], plt.ylim()[1]*0.6, f'90% CI Upper: {confidence_interval[1]:.2f}', color='red', horizontalalignment='center', verticalalignment='center')

#     plt.xlabel(f'Estimate {id} Area')
#     plt.ylabel('Density')
#     plt.legend(['PDF', 'Mean', 'Median', '90% CI Lower', '90% CI Upper'])
#     plt.title(f'Estimate {id} Area with KDE-based 90% CI')
#     plt.show()
#     plt.close()

######################################### SINGLE PROFORMANCE ANALYSIS #############

# 假设你已有 all_of_estimate_area_data 和 true_value_dict 这两个变量

# 对编号 7000 进行分析
id = 7000
data = all_of_estimate_area_data[id]["area"]
true_value = true_value_dict[id]

# 计算均值和中位数
mean_area = data.mean()
median_area = np.median(data)

# 计算 90% 置信区间
confidence_interval = np.percentile(data, [5, 95])
lower_band = confidence_interval[0]
upper_band = confidence_interval[1]

# 判断真值是否在 90% 置信区间内
true_within_band = 'O' if lower_band <= true_value <= upper_band else 'X'

# 计算误差指标
rmse = np.sqrt(((data - true_value) ** 2).mean())
mape = np.abs((data - true_value) / true_value * 100).mean()

# 创建 DataFrame
result_df = pd.DataFrame({
    'true_value': [true_value],
    'xbar': [mean_area],
    'median': [median_area],
    'Data lower': [lower_band],
    'Data upper': [upper_band],
    f'In 90% band': [true_within_band],
    'RMSE': [rmse],
    'MAPE': [mape]
}, index=[7000])
plt.figure(figsize=(6, 6))
plt.scatter(true_value, data, color='blue')
plt.plot([min(true_value, data), max(true_value, data)], 
            [min(true_value, data), max(true_value, data)], 'r--')
plt.xlabel('True Value(pixel)')
plt.ylabel('Mean Area')
plt.title(f'True Value vs Mean Area for ID {id}')
plt.show()
plt.close()
plt.savefig(fr"C:\Users\Lab_205\Desktop\thesis_img\single_result\true_vs_mean_area_{id}.png")
plt.close()
print(result_df)

#######################  all of result ##############
plt.hist(np.abs((data - true_value) / true_value * 100))
stat_result = {}
id=7000
ld_list = [i for i in range(7000, 7073) if i != 7001]

for id in ld_list:
    data = all_of_estimate_area_data[id]["area"]
    true_value = true_value_dict[id]
    
    mean_area = data.mean()
    median_area = np.median(data)
    
    confidence_interval = np.percentile(data, [5, 95])
    lower_band = confidence_interval[0]
    upper_band = confidence_interval[1]
    
    true_within_band = 'O' if lower_band <= true_value <= upper_band else 'X'
    
    rmse = np.sqrt(((data - true_value) ** 2).mean())
    mape = np.abs((data - true_value) / true_value * 100).mean()
    
    result_df = pd.DataFrame({
        'true_value': [true_value],
        'xbar': [mean_area],
        'median': [median_area],
        'Data lower': [lower_band],
        'Data upper': [upper_band],
        f'In 90% band': [true_within_band],
        'RMSE': [rmse],
        'RMSE/yi':[rmse/true_value],
        'MAPE': [mape]
    }, index=[id])
    
    stat_result[str(id)] = result_df


stat_result_df = pd.concat(stat_result.values(), axis=0)
stat_result_df.index.name = 'ID'
stat_result_df.to_csv(r"C:\Users\Lab_205\Desktop\stat_result.csv")
for key, df in stat_result.items():
    print(f"Result for ID {key}:")
    print(df)
    print("\n")
stat_result_df.T.to_csv(r"C:\Users\Lab_205\Desktop\stat_result.t.csv")
################################### LOG ########################

stat_log_result = {}

ld_list = [i for i in range(7000, 7073) if i != 7001]
def preprocess_data(data):
    data = data.dropna()
    # 確保所有值都是正數，否則進行適當的處理
    data = data[data > 0]
    return data

for id in ld_list:
    data = preprocess_data(all_of_estimate_area_data[id]["area"])
    data = np.log(data)
    true_value = np.log(true_value_dict[id])
    
    mean_area = data.mean()
    median_area = np.median(data)
    
    confidence_interval = np.percentile(data, [5, 95])
    lower_band = confidence_interval[0]
    upper_band = confidence_interval[1]
    
    true_within_band = 'O' if lower_band <= true_value <= upper_band else 'X'
    
    rmse = np.sqrt(((data - true_value) ** 2).mean())
    mape = np.abs((data - true_value) / true_value * 100).mean()
    
    result_df = pd.DataFrame({
        'true_value': [true_value],
        'xbar': [mean_area],
        'median': [median_area],
        'Data lower': [lower_band],
        'Data upper': [upper_band],
        f'In 90% band': [true_within_band],
        'RMSE': [rmse],
        'NORMSE':[rmse/true_value],
        'MAPE': [mape]
    }, index=[id])
    
    stat_result[str(id)] = result_df


stat_log_result_df = pd.concat(stat_result.values(), axis=0)
stat_log_result_df.index.name = 'ID'
stat_log_result_df.to_csv(r"C:\Users\Lab_205\Desktop\stat_log_result.csv")
for key, df in stat_log_result.items():
    print(f"Result for ID {key}:")
    print(df)
    print("\n")
stat_log_result_df.T.to_csv(r"C:\Users\Lab_205\Desktop\stat_log_result_t.csv")




########################### all of data to combine a final result
def preprocess_data(data):
    data = data.dropna()
    # 確保所有值都是正數，否則進行適當的處理
    data = data[data > 0]
    return data
est_list= [i for i in range(7000,7073) if i!= 7001]
est_mean_dict= {
    "est_id": [], 
    "mean_area": [],
    "medium_mean_area":[],
    "true_value":[],
    "log_mean_area":[],
    "log_medium_mean_area":[],
    "log_true_value":[],
}
for i in est_list:
    est_mean = all_of_estimate_area_data[i]["area"].mean() 
    est_mean_medium= np.median(all_of_estimate_area_data[i]["area"])
    est_mean_dict["est_id"].append(i)
    est_mean_dict["mean_area"].append(est_mean)
    est_mean_dict["medium_mean_area"].append(est_mean_medium)
    est_mean_dict["true_value"].append(true_value_dict[i])

    ### log 
    est_log_mean = np.log(preprocess_data(all_of_estimate_area_data[i]["area"])).mean()
    est_log_mean_medium= np.median(np.log(preprocess_data(all_of_estimate_area_data[i]["area"])))
    est_mean_dict["log_mean_area"].append(est_log_mean)
    est_mean_dict["log_medium_mean_area"].append(est_log_mean_medium)
    est_mean_dict["log_true_value"].append(np.log(true_value_dict[i]))

est_mean_dict_df = pd.DataFrame(est_mean_dict)





### MEAN
est_mean_dict_df['error'] = est_mean_dict_df['mean_area'] - est_mean_dict_df['true_value']
est_mean_dict_df['absolute_error'] = np.abs(est_mean_dict_df['error'])
est_mean_dict_df['squared_error'] = est_mean_dict_df['error'] ** 2
est_mean_dict_df['percentage_error'] = est_mean_dict_df['absolute_error'] / est_mean_dict_df['true_value'] * 100

### MEDIUM
est_mean_dict_df['medium_error'] =est_mean_dict_df['medium_mean_area'] - est_mean_dict_df['true_value']
est_mean_dict_df['medium_absolute_error'] = np.abs(est_mean_dict_df['medium_error'])
est_mean_dict_df['medium_squared_error'] = est_mean_dict_df['medium_error'] ** 2
est_mean_dict_df['medium_percentage_error'] = est_mean_dict_df['medium_absolute_error'] / est_mean_dict_df['true_value'] * 100

# MEAN指標
mae = est_mean_dict_df['absolute_error'].mean()  # Mean Absolute Error
mse = est_mean_dict_df['squared_error'].mean()  # Mean Squared Error
rmse = np.sqrt(mse)  # Root Mean Squared Error
normse = rmse / np.mean(est_mean_dict_df['true_value'])  # Normalized Root Mean Squared Error
mape = est_mean_dict_df['percentage_error'].mean()  # Mean Absolute Percentage Error

#MEDIUM指標
medium_mae = est_mean_dict_df['medium_absolute_error'].mean()  # Mean Absolute Error
medium_mse = est_mean_dict_df['medium_squared_error'].mean()  # Mean Squared Error
medium_rmse = np.sqrt(medium_mse)  # Root Mean Squared Error
medium_normse = medium_rmse / np.mean(est_mean_dict_df['true_value'])  # Normalized Root Mean Squared Error
medium_mape = est_mean_dict_df['medium_percentage_error'].mean()  # Mean Absolute Percentage Error



#### log mean
est_mean_dict_df['log_error'] = est_mean_dict_df['log_mean_area'] - est_mean_dict_df['log_true_value']
est_mean_dict_df['log_absolute_error'] = np.abs(est_mean_dict_df['log_error'])
est_mean_dict_df['log_squared_error'] = est_mean_dict_df['log_error'] ** 2
est_mean_dict_df['log_percentage_error'] = est_mean_dict_df['log_absolute_error'] / est_mean_dict_df['log_true_value'] * 100

####log medium
est_mean_dict_df['log_medium_error'] =est_mean_dict_df['log_medium_mean_area'] - est_mean_dict_df['log_true_value']
est_mean_dict_df['log_medium_absolute_error'] = np.abs(est_mean_dict_df['log_medium_error'])
est_mean_dict_df['log_medium_squared_error'] = est_mean_dict_df['log_medium_error'] ** 2
est_mean_dict_df['log_medium_percentage_error'] = est_mean_dict_df['log_medium_absolute_error'] / est_mean_dict_df['log_true_value'] * 100

# log mean指標
log_mae = est_mean_dict_df['log_absolute_error'].mean()  # Mean Absolute Error
log_mse = est_mean_dict_df['log_squared_error'].mean()  # Mean Squared Error
log_rmse = np.sqrt(log_mse)  # Root Mean Squared Error
log_normse = log_rmse / np.mean(est_mean_dict_df['log_true_value'])  # Normalized Root Mean Squared Error
log_mape = est_mean_dict_df['log_percentage_error'].mean()  # Mean Absolute Percentage Error

# log meduim指標
log_medium_mae = est_mean_dict_df['log_medium_absolute_error'].mean()  # Mean Absolute Error
log_medium_mse = est_mean_dict_df['log_medium_squared_error'].mean()  # Mean Squared Error
log_medium_rmse = np.sqrt(log_medium_mse)  # Root Mean Squared Error
log_medium_normse = log_medium_rmse / np.mean(est_mean_dict_df['log_true_value'])  # Normalized Root Mean Squared Error
log_medium_mape = est_mean_dict_df['log_medium_percentage_error'].mean()  # Mean Absolute Percentage Error

output_df = pd.DataFrame({
    "Mean_MAPE": [mape],
    "Mean_NRMSE": [normse],
    "Median_MAPE": [medium_mape],
    "Median_NRMSE": [medium_normse],
    "Log_Mean_MAPE": [log_mape],
    "Log_Mean_NRMSE": [log_normse],
    "Log_Median_MAPE": [log_medium_mape],
     "Log_Median_NRMSE": [log_medium_normse],
}, index=['Removed outliers'])


output_df

##　mean medium
print(f"Mean Absolute Percentage Error (MAPE): {mape}%")
print(f"Normalized Root Mean Squared Error (NRMSE): {normse}")

print(f"Normalized Root Mean Squared Error (NRMSE)for median: {medium_normse}%")
print(f"Mean Absolute Percentage Error (MAPE) for median: {medium_mape}%")


## log mean medium
print(f"Normalized Root Mean Squared Error (NRMSE)for log mean: {log_normse}%")
print(f"Mean Absolute Percentage Error (MAPE) for log mean: {log_mape}%")

print(f"Normalized Root Mean Squared Error (NRMSE)for log median: {log_medium_normse}%")
print(f"Mean Absolute Percentage Error (MAPE) for log median: {log_medium_mape}%")
est_mean_dict_df.to_csv(r"C:\Users\Lab_205\Desktop\est_mean_dict_df.csv")

# 視覺化分析 mean medium
plt.figure(figsize=(10, 5))

# 預估值與真實值的散點圖
plt.subplot(1, 2, 1)
plt.scatter(est_mean_dict_df['true_value'], est_mean_dict_df['mean_area'], color='blue')
plt.plot([est_mean_dict_df['true_value'].min(), est_mean_dict_df['true_value'].max()], [est_mean_dict_df['true_value'].min(), est_mean_dict_df['true_value'].max()], 'r--')

plt.xlabel('True Value(pixel)')
plt.ylabel('Mean Area')
plt.title('True Value vs Mean Area')


plt.subplot(1, 2, 2)
plt.scatter(est_mean_dict_df['true_value'], est_mean_dict_df['medium_mean_area'], color='blue')
plt.plot([est_mean_dict_df['true_value'].min(), est_mean_dict_df['true_value'].max()], [est_mean_dict_df['true_value'].min(), est_mean_dict_df['true_value'].max()], 'r--')

plt.xlabel('True Value(pixel)')
plt.ylabel('Median Area')
plt.title('True Value vs Median Area')

plt.show()

# 視覺化分析 log mean medium
plt.figure(figsize=(10, 5))

# 預估值與真實值的散點圖
plt.subplot(1, 2, 1)
plt.scatter(est_mean_dict_df['log_true_value'], est_mean_dict_df['log_mean_area'], color='blue')
plt.plot([est_mean_dict_df['log_true_value'].min(), est_mean_dict_df['log_true_value'].max()], [est_mean_dict_df['log_true_value'].min(), est_mean_dict_df['log_true_value'].max()], 'r--')


plt.xlabel('Log True Value(pixel)')
plt.ylabel('Log Mean Area')
plt.title('Log True Value vs Log Mean Area')


plt.subplot(1, 2, 2)
plt.scatter(est_mean_dict_df['log_true_value'], est_mean_dict_df['log_medium_mean_area'], color='blue')
plt.plot([est_mean_dict_df['log_true_value'].min(), est_mean_dict_df['log_true_value'].max()], [est_mean_dict_df['log_true_value'].min(), est_mean_dict_df['log_true_value'].max()], 'r--')

plt.xlabel('Log True Value(pixel)')
plt.ylabel('Log Median Area')
plt.title('Log True Value vs Log Median Area')

plt.show()

########################################## remove outliers  #

def preprocess_data(data):
    data = data.dropna()
    # 確保所有值都是正數，否則進行適當的處理
    data = data[data > 0]
    return data
est_mean_dict_df['absolute_error'][est_mean_dict_df['absolute_error']>10000].index
est_list= [i for i in range(7000,7073) if i not in [7001,7038,7067,7068,7070,7071,7072]] 

est_mean_dict= {
    "est_id": [], 
    "mean_area": [],
    "medium_mean_area":[],
    "true_value":[],
    "log_mean_area":[],
    "log_medium_mean_area":[],
    "log_true_value":[],
}
for i in est_list:
    est_mean = all_of_estimate_area_data[i]["area"].mean()
    est_mean_medium= np.median(all_of_estimate_area_data[i]["area"])
    est_mean_dict["est_id"].append(i)
    est_mean_dict["mean_area"].append(est_mean)
    est_mean_dict["medium_mean_area"].append(est_mean_medium)
    est_mean_dict["true_value"].append(true_value_dict[i])

    ### log 
    est_log_mean = np.log(preprocess_data(all_of_estimate_area_data[i]["area"])).mean()
    est_log_mean_medium= np.median(np.log(preprocess_data(all_of_estimate_area_data[i]["area"])))
    est_mean_dict["log_mean_area"].append(est_log_mean)
    est_mean_dict["log_medium_mean_area"].append(est_log_mean_medium)
    est_mean_dict["log_true_value"].append(np.log(true_value_dict[i]))

est_mean_dict_df = pd.DataFrame(est_mean_dict)





### MEAN
est_mean_dict_df['error'] = est_mean_dict_df['mean_area'] - est_mean_dict_df['true_value']
est_mean_dict_df['absolute_error'] = np.abs(est_mean_dict_df['error'])
est_mean_dict_df['squared_error'] = est_mean_dict_df['error'] ** 2
est_mean_dict_df['percentage_error'] = est_mean_dict_df['absolute_error'] / est_mean_dict_df['true_value'] * 100

### MEDIUM
est_mean_dict_df['medium_error'] =est_mean_dict_df['medium_mean_area'] - est_mean_dict_df['true_value']
est_mean_dict_df['medium_absolute_error'] = np.abs(est_mean_dict_df['medium_error'])
est_mean_dict_df['medium_squared_error'] = est_mean_dict_df['medium_error'] ** 2
est_mean_dict_df['medium_percentage_error'] = est_mean_dict_df['medium_absolute_error'] / est_mean_dict_df['true_value'] * 100

# MEAN指標
mae = est_mean_dict_df['absolute_error'].mean()  # Mean Absolute Error
mse = est_mean_dict_df['squared_error'].mean()  # Mean Squared Error
rmse = np.sqrt(mse)  # Root Mean Squared Error
normse = rmse / np.mean(est_mean_dict_df['true_value'])  # Normalized Root Mean Squared Error
mape = est_mean_dict_df['percentage_error'].mean()  # Mean Absolute Percentage Error

#MEDIUM指標
medium_mae = est_mean_dict_df['medium_absolute_error'].mean()  # Mean Absolute Error
medium_mse = est_mean_dict_df['medium_squared_error'].mean()  # Mean Squared Error
medium_rmse = np.sqrt(medium_mse)  # Root Mean Squared Error
medium_normse = medium_rmse / np.mean(est_mean_dict_df['true_value'])  # Normalized Root Mean Squared Error
medium_mape = est_mean_dict_df['medium_percentage_error'].mean()  # Mean Absolute Percentage Error



#### log mean
est_mean_dict_df['log_error'] = est_mean_dict_df['log_mean_area'] - est_mean_dict_df['log_true_value']
est_mean_dict_df['log_absolute_error'] = np.abs(est_mean_dict_df['log_error'])
est_mean_dict_df['log_squared_error'] = est_mean_dict_df['log_error'] ** 2
est_mean_dict_df['log_percentage_error'] = est_mean_dict_df['log_absolute_error'] / est_mean_dict_df['log_true_value'] * 100

####log medium
est_mean_dict_df['log_medium_error'] =est_mean_dict_df['log_medium_mean_area'] - est_mean_dict_df['log_true_value']
est_mean_dict_df['log_medium_absolute_error'] = np.abs(est_mean_dict_df['log_medium_error'])
est_mean_dict_df['log_medium_squared_error'] = est_mean_dict_df['log_medium_error'] ** 2
est_mean_dict_df['log_medium_percentage_error'] = est_mean_dict_df['log_medium_absolute_error'] / est_mean_dict_df['log_true_value'] * 100

# log mean指標
log_mae = est_mean_dict_df['log_absolute_error'].mean()  # Mean Absolute Error
log_mse = est_mean_dict_df['log_squared_error'].mean()  # Mean Squared Error
log_rmse = np.sqrt(log_mse)  # Root Mean Squared Error
log_normse = log_rmse / np.mean(est_mean_dict_df['log_true_value'])  # Normalized Root Mean Squared Error
log_mape = est_mean_dict_df['log_percentage_error'].mean()  # Mean Absolute Percentage Error

# log meduim指標
log_medium_mae = est_mean_dict_df['log_medium_absolute_error'].mean()  # Mean Absolute Error
log_medium_mse = est_mean_dict_df['log_medium_squared_error'].mean()  # Mean Squared Error
log_medium_rmse = np.sqrt(log_medium_mse)  # Root Mean Squared Error
log_medium_normse = log_medium_rmse / np.mean(est_mean_dict_df['log_true_value'])  # Normalized Root Mean Squared Error
log_medium_mape = est_mean_dict_df['log_medium_percentage_error'].mean()  # Mean Absolute Percentage Error



##　mean medium
print(f"Mean Absolute Percentage Error (MAPE): {mape}%")
print(f"Normalized Root Mean Squared Error (NRMSE): {normse}")

print(f"Normalized Root Mean Squared Error (NRMSE)for median: {medium_normse}%")
print(f"Mean Absolute Percentage Error (MAPE) for median: {medium_mape}%")


## log mean medium
print(f"Normalized Root Mean Squared Error (NRMSE)for log mean: {log_normse}%")
print(f"Mean Absolute Percentage Error (MAPE) for log mean: {log_mape}%")

print(f"Normalized Root Mean Squared Error (NRMSE)for log median: {log_medium_normse}%")
print(f"Mean Absolute Percentage Error (MAPE) for log median: {log_medium_mape}%")
est_mean_dict_df.to_csv(r"C:\Users\Lab_205\Desktop\est_mean_dict_df_rvoutliers.csv")

# 視覺化分析 mean medium
plt.figure(figsize=(10, 5))

# 預估值與真實值的散點圖
plt.subplot(1, 2, 1)
plt.scatter(est_mean_dict_df['true_value'], est_mean_dict_df['mean_area'], color='blue')
plt.plot([est_mean_dict_df['true_value'].min(), est_mean_dict_df['true_value'].max()], [est_mean_dict_df['true_value'].min(), est_mean_dict_df['true_value'].max()], 'r--')

plt.xlabel('True Value(pixel)')
plt.ylabel('Mean Area')
plt.title('True Value vs Mean Area')


plt.subplot(1, 2, 2)
plt.scatter(est_mean_dict_df['true_value'], est_mean_dict_df['medium_mean_area'], color='blue')
plt.plot([est_mean_dict_df['true_value'].min(), est_mean_dict_df['true_value'].max()], [est_mean_dict_df['true_value'].min(), est_mean_dict_df['true_value'].max()], 'r--')

plt.xlabel('True Value(pixel)')
plt.ylabel('Median Area')
plt.title('True Value vs Median Area')

plt.show()

# 視覺化分析 log mean medium
plt.figure(figsize=(10, 5))

# 預估值與真實值的散點圖
plt.subplot(1, 2, 1)
plt.scatter(est_mean_dict_df['log_true_value'], est_mean_dict_df['log_mean_area'], color='blue')
plt.plot([est_mean_dict_df['log_true_value'].min(), est_mean_dict_df['log_true_value'].max()], [est_mean_dict_df['log_true_value'].min(), est_mean_dict_df['log_true_value'].max()], 'r--')


plt.xlabel('Log True Value(pixel)')
plt.ylabel('Log Mean Area')
plt.title('Log True Value vs Log Mean Area')


plt.subplot(1, 2, 2)
plt.scatter(est_mean_dict_df['log_true_value'], est_mean_dict_df['log_medium_mean_area'], color='blue')
plt.plot([est_mean_dict_df['log_true_value'].min(), est_mean_dict_df['log_true_value'].max()], [est_mean_dict_df['log_true_value'].min(), est_mean_dict_df['log_true_value'].max()], 'r--')

plt.xlabel('Log True Value(pixel)')
plt.ylabel('Log Median Area')
plt.title('Log True Value vs Log Median Area')

plt.show()
output_df = pd.DataFrame({
    "Mean_MAPE": [mape],
    "Mean_NRMSE": [normse],
    "Median_MAPE": [medium_mape],
    "Median_NRMSE": [medium_normse],
    "Log_Mean_MAPE": [log_mape],
    "Log_Mean_NRMSE": [log_normse],
    "Log_Median_MAPE": [log_medium_mape],
     "Log_Median_NRMSE": [log_medium_normse],
}, index=['Removed outliers'])


output_df



























##############    combine 12 result data into one figure   #############

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.neighbors import KernelDensity

ld_list = [i for i in range(7000, 7073) if i != 7001]
total_plots = len(ld_list)
plots_per_figure = 12  # 每張圖像顯示12個子圖

# 總共需要的圖像數量
num_figures = (total_plots + plots_per_figure - 1) // plots_per_figure

for fig_num in range(num_figures):
    fig, axs = plt.subplots(4, 3, figsize=(20, 20))  # 4行3列的子圖網格
    start_idx = fig_num * plots_per_figure
    end_idx = min(start_idx + plots_per_figure, total_plots)

    for idx in range(start_idx, end_idx):
        id = ld_list[idx]
        mean_area = all_of_estimate_area_data[id]["area"].mean()
        sem_area = stats.sem(all_of_estimate_area_data[id]["area"])
        median_area = np.median(all_of_estimate_area_data[id]["area"])
        # KDE Estimation
        kde = KernelDensity(kernel='gaussian', bandwidth=2000).fit(all_of_estimate_area_data[id]["area"].values.reshape(-1, 1))
        x_d = np.linspace(min(all_of_estimate_area_data[id]["area"]), max(all_of_estimate_area_data[id]["area"]), 1000)
        log_dens = kde.score_samples(x_d.reshape(-1, 1))

        # calculate 90C.I.
        confidence_interval = np.percentile(all_of_estimate_area_data[id]["area"], [5, 95])

        # 獲取當前子圖的位置
        ax = axs[(idx - start_idx) // 3, (idx - start_idx) % 3]

        ax.hist(all_of_estimate_area_data[id]["area"], bins=60, color='blue', alpha=0.5, edgecolor='black', label='Marked Up')
        ax.plot(x_d, np.exp(log_dens) * len(all_of_estimate_area_data[id]["area"]) * (x_d[1] - x_d[0]), color='k', linestyle='dashed')

        # mean and true value
        ax.axvline(mean_area, color='k', linestyle='dashed', linewidth=1)
        ax.axvline(median_area, color='k', linestyle='dashed', linewidth=1)
        ax.axvline(true_value_dict[id], color='g', linestyle='dashed', linewidth=1)
        ax.axvline(confidence_interval[0], color='red', linestyle="solid", linewidth=1)
        ax.axvline(confidence_interval[1], color='red', linestyle='solid', linewidth=1)

        ax.text(mean_area, ax.get_ylim()[1] * 0.9, f'Mean: {mean_area:.2f}', horizontalalignment='center', verticalalignment='center')
        ax.text(median_area, ax.get_ylim()[1] * 0.8, f'Median: {median_area:.2f}', horizontalalignment='center', verticalalignment='center')
        ax.text(true_value_dict[id], ax.get_ylim()[1] * 0.7, f'True Value: {true_value_dict[id]}', color='green', horizontalalignment='center', verticalalignment='center')
        ax.text(confidence_interval[0], ax.get_ylim()[1] * 0.6, f'90% Data Lower: {confidence_interval[0]:.2f}', color='red', horizontalalignment='center', verticalalignment='center')
        ax.text(confidence_interval[1], ax.get_ylim()[1] * 0.6, f'90% Date Upper: {confidence_interval[1]:.2f}', color='red', horizontalalignment='center', verticalalignment='center')

        ax.set_xlabel(f'Estimate {id} Area')
        ax.set_ylabel('Count')
        ax.legend([f'Estimate {id} area', 'Mean', 'True value', '90% Data Lower', '90% Data Upper'])
        ax.set_title(f'Estimate {id} Area with KDE-based 90% Data')

    plt.tight_layout()
    plt.savefig(rf"C:\Users\Lab_205\Desktop\thesis_img\result_img\combined_area_kde_{fig_num + 1}.png")
    plt.close()

##############    combine 12 log result data into one figure   #############

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.neighbors import KernelDensity

def preprocess_data(data):
    data = data.dropna()
    # 確保所有值都是正數，否則進行適當的處理
    data = data[data > 0]
    return data

ld_list = [i for i in range(7000, 7073) if i != 7001]
total_plots = len(ld_list)
plots_per_figure = 12  # 每張圖像顯示12個子圖

# 總共需要的圖像數量
num_figures = (total_plots + plots_per_figure - 1) // plots_per_figure

for fig_num in range(num_figures):
    fig, axs = plt.subplots(4, 3, figsize=(20, 20))  # 4行3列的子圖網格
    start_idx = fig_num * plots_per_figure
    end_idx = min(start_idx + plots_per_figure, total_plots)

    for idx in range(start_idx, end_idx):
        id = ld_list[idx]
        np.set_printoptions(precision=3)
        log_mean_area = np.log(preprocess_data(all_of_estimate_area_data[id]["area"])).mean()
        log_sem_area = stats.sem(np.log(preprocess_data(all_of_estimate_area_data[id]["area"])))
        log_median_area = np.median(np.log(preprocess_data(all_of_estimate_area_data[id]["area"])))
        # KDE Estimation
        kde = KernelDensity(kernel='gaussian', bandwidth=2000).fit(np.log(preprocess_data(all_of_estimate_area_data[id]["area"])).values.reshape(-1, 1))
        x_d = np.linspace(min(np.log(preprocess_data(all_of_estimate_area_data[id]["area"]))), max(np.log(preprocess_data(all_of_estimate_area_data[id]["area"]))), 1000)
        log_dens = kde.score_samples(x_d.reshape(-1, 1))

        # calculate 90C.I.
        confidence_interval = np.percentile(np.log(preprocess_data(all_of_estimate_area_data[id]["area"])), [5, 95])

        # 獲取當前子圖的位置
        ax = axs[(idx - start_idx) // 3, (idx - start_idx) % 3]

        ax.hist(np.log(preprocess_data(all_of_estimate_area_data[id]["area"])), bins=60, color='blue', alpha=0.5, edgecolor='black', label='Marked Up')
        ax.plot(x_d, np.exp(log_dens) * len(np.log(preprocess_data(all_of_estimate_area_data[id]["area"]))) * (x_d[1] - x_d[0]), color='k', linestyle='dashed')

        # mean and true value
        ax.axvline(log_mean_area, color='k', linestyle='dashed', linewidth=1)
        ax.axvline(log_median_area, color='k', linestyle='dashed', linewidth=1)
        ax.axvline(np.log(true_value_dict[id]), color='g', linestyle='dashed', linewidth=1)
        ax.axvline(confidence_interval[0], color='red', linestyle="solid", linewidth=1)
        ax.axvline(confidence_interval[1], color='red', linestyle='solid', linewidth=1)

        ax.text(log_mean_area, ax.get_ylim()[1] * 0.9, f'Log Mean: {log_mean_area:.2f}', horizontalalignment='center', verticalalignment='center')
        ax.text(log_median_area, ax.get_ylim()[1] * 0.8, f'Log Median: {log_median_area:.2f}', horizontalalignment='center', verticalalignment='center')
        ax.text(np.log(true_value_dict[id]), ax.get_ylim()[1] * 0.7, f'Log True Value: {np.log(true_value_dict[id]):.2f}', color='green', horizontalalignment='center', verticalalignment='center')
        ax.text(confidence_interval[0], ax.get_ylim()[1] * 0.6, f'90% Data Lower: {confidence_interval[0]:.2f}', color='red', horizontalalignment='center', verticalalignment='center')
        ax.text(confidence_interval[1], ax.get_ylim()[1] * 0.6, f'90% Date Upper: {confidence_interval[1]:.2f}', color='red', horizontalalignment='center', verticalalignment='center')

        ax.set_xlabel(f'Estimate {id} Log Area')
        ax.set_ylabel('Count')
        ax.legend([f'Estimate {id} Log area', 'Log Mean', 'Log True value', '90% Data Lower', '90% Data Upper'])
        ax.set_title(f'Estimate {id} Log Area with KDE-based 90% Data')

    plt.tight_layout()
    plt.savefig(rf"C:\Users\Lab_205\Desktop\thesis_img\result_img\12log\combined_log_area_kde_{fig_num + 1}.png")
    plt.close()






#draw kde plot for all_of_estimate_area_data[7000].query(f"area <30000")

nice_data = all_of_estimate_area_data[7000].query(f"area <30000")
mean_area = nice_data["area"].mean()
sem_area = stats.sem(nice_data["area"])

# KDE Estimation
kde = KernelDensity(kernel='gaussian', bandwidth=2000).fit(nice_data["area"].values.reshape(-1, 1))
x_d = np.linspace(min(nice_data["area"]), max(nice_data["area"]), 1000)
log_dens = kde.score_samples(x_d.reshape(-1, 1))

# calculate 90C.I.
confidence_interval = np.percentile(nice_data["area"], [5, 95])

plt.figure(figsize=(10, 6))
plt.hist(nice_data["area"], bins=60, color='blue', alpha=0.5, edgecolor='black', label='Marked Up')
plt.plot(x_d, np.exp(log_dens) * len(nice_data["area"]) * (x_d[1] - x_d[0]), color='k', linestyle='dashed')
plt.axvline(mean_area, color='k', linestyle='dashed', linewidth=1)
plt.axvline(confidence_interval[0], color='red', linestyle="solid", linewidth=1)
plt.axvline(confidence_interval[1], color='red', linestyle='solid', linewidth=1)
plt.text(mean_area, plt.ylim()[1] * 0.9, f'Mean: {mean_area:.2f}', horizontalalignment='center', verticalalignment='center')
plt.xlabel('Estimate 7000 Area')
plt.ylabel('Count')
plt.legend([f'Estimate 7000 area', 'KDE'])
plt.title(f'Estimate 7000 Area with KDE-based 90% CI')  

#評估指標
mae = np.abs(nice_data["area"] - true_value_dict[7000]).mean()  # Mean Absolute Error
mse = ((nice_data["area"] - true_value_dict[7000]) ** 2).mean()  # Mean Squared Error
rmse = np.sqrt(mse)  # Root Mean Squared Error
mape = np.abs((nice_data["area"] - true_value_dict[7000]) / true_value_dict[7000] * 100).mean()  # Mean Absolute Percentage Error

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Percentage Error (MAPE): {mape}%")






######  fliterdf   ##############



fliterdf = {

}
for temp in range(7000, 7073):
    if temp == 7001:
        continue
    tempdf= all_of_estimate_area_data[temp]
    tempdf = tempdf[tempdf["id"] != "{7007}"]
    tempdf = tempdf[tempdf["id"] != "{7035}"]
    tempdf = tempdf[tempdf["id"] != "{7068}"]
    tempdf = tempdf[tempdf["id"] != "{7071}"]
    tempdf = tempdf[tempdf["id"] != "{7072}"]
    tempdf = tempdf[tempdf["id"] != "{7070}"]
    fliterdf[temp] = tempdf

fliterdf



def preprocess_data(data):
    data = data.dropna()
    # 確保所有值都是正數，否則進行適當的處理
    data = data[data > 0]
    return data
est_list= [i for i in range(7000,7073) if i!= 7001 and i!= 7007 and i!= 7035 and i!= 7068 and i!= 7071 and i!= 7072 and i!= 7070]
fliterdf_est_mean_dict= {
    "est_id": [], 
    "mean_area": [],
    "medium_mean_area":[],
    "true_value":[],
    "log_mean_area":[],
    "log_medium_mean_area":[],
    "log_true_value":[],
}
for i in est_list:
    est_mean = fliterdf[i]["area"].mean() 
    est_mean_medium= np.median(fliterdf[i]["area"]).mean()
    fliterdf_est_mean_dict["est_id"].append(i)
    fliterdf_est_mean_dict["mean_area"].append(est_mean)
    fliterdf_est_mean_dict["medium_mean_area"].append(est_mean_medium)
    fliterdf_est_mean_dict["true_value"].append(true_value_dict[i])

    ### log 
    est_log_mean = np.log(preprocess_data(fliterdf[i]["area"])).mean()
    est_log_mean_medium= np.median(np.log(preprocess_data(fliterdf[i]["area"]))).mean()
    fliterdf_est_mean_dict["log_mean_area"].append(est_log_mean)
    fliterdf_est_mean_dict["log_medium_mean_area"].append(est_log_mean_medium)
    fliterdf_est_mean_dict["log_true_value"].append(np.log(true_value_dict[i]))

fliterdf_est_mean_dict_df = pd.DataFrame(fliterdf_est_mean_dict)


fliterdf_est_mean_dict_df


### MEAN
fliterdf_est_mean_dict_df['error'] = fliterdf_est_mean_dict_df['mean_area'] - fliterdf_est_mean_dict_df['true_value']
fliterdf_est_mean_dict_df['absolute_error'] = np.abs(fliterdf_est_mean_dict_df['error'])
fliterdf_est_mean_dict_df['squared_error'] = fliterdf_est_mean_dict_df['error'] ** 2
fliterdf_est_mean_dict_df['percentage_error'] = fliterdf_est_mean_dict_df['absolute_error'] / fliterdf_est_mean_dict_df['true_value'] * 100

### MEDIUM
fliterdf_est_mean_dict_df['medium_error'] =fliterdf_est_mean_dict_df['medium_mean_area'] - fliterdf_est_mean_dict_df['true_value']
fliterdf_est_mean_dict_df['medium_absolute_error'] = np.abs(fliterdf_est_mean_dict_df['medium_error'])
fliterdf_est_mean_dict_df['medium_squared_error'] = fliterdf_est_mean_dict_df['medium_error'] ** 2
fliterdf_est_mean_dict_df['medium_percentage_error'] = fliterdf_est_mean_dict_df['medium_absolute_error'] / fliterdf_est_mean_dict_df['true_value'] * 100

# MEAN指標
mae = fliterdf_est_mean_dict_df['absolute_error'].mean()  # Mean Absolute Error
mse = fliterdf_est_mean_dict_df['squared_error'].mean()  # Mean Squared Error
rmse = np.sqrt(mse)  # Root Mean Squared Error
normse = rmse / np.mean(fliterdf_est_mean_dict_df['true_value'])  # Normalized Root Mean Squared Error
mape = fliterdf_est_mean_dict_df['percentage_error'].mean()  # Mean Absolute Percentage Error

#MEDIUM指標
medium_mae = fliterdf_est_mean_dict_df['medium_absolute_error'].mean()  # Mean Absolute Error
medium_mse = fliterdf_est_mean_dict_df['medium_squared_error'].mean()  # Mean Squared Error
medium_rmse = np.sqrt(medium_mse)  # Root Mean Squared Error
medium_normse = medium_rmse / np.mean(fliterdf_est_mean_dict_df['true_value'])  # Normalized Root Mean Squared Error
medium_mape = fliterdf_est_mean_dict_df['medium_percentage_error'].mean()  # Mean Absolute Percentage Error



#### log mean
fliterdf_est_mean_dict_df['log_error'] = fliterdf_est_mean_dict_df['log_mean_area'] - fliterdf_est_mean_dict_df['log_true_value']
fliterdf_est_mean_dict_df['log_absolute_error'] = np.abs(fliterdf_est_mean_dict_df['log_error'])
fliterdf_est_mean_dict_df['log_squared_error'] = fliterdf_est_mean_dict_df['log_error'] ** 2
fliterdf_est_mean_dict_df['log_percentage_error'] = fliterdf_est_mean_dict_df['log_absolute_error'] / fliterdf_est_mean_dict_df['log_true_value'] * 100

####log medium
fliterdf_est_mean_dict_df['log_medium_error'] =fliterdf_est_mean_dict_df['log_medium_mean_area'] - fliterdf_est_mean_dict_df['log_true_value']
fliterdf_est_mean_dict_df['log_medium_absolute_error'] = np.abs(fliterdf_est_mean_dict_df['log_medium_error'])
fliterdf_est_mean_dict_df['log_medium_squared_error'] = fliterdf_est_mean_dict_df['log_medium_error'] ** 2
fliterdf_est_mean_dict_df['log_medium_percentage_error'] = fliterdf_est_mean_dict_df['log_medium_absolute_error'] / fliterdf_est_mean_dict_df['log_true_value'] * 100

# log mean指標
log_mae = fliterdf_est_mean_dict_df['log_absolute_error'].mean()  # Mean Absolute Error
log_mse = fliterdf_est_mean_dict_df['log_squared_error'].mean()  # Mean Squared Error
log_rmse = np.sqrt(log_mse)  # Root Mean Squared Error
log_normse = log_rmse / np.mean(fliterdf_est_mean_dict_df['log_true_value'])  # Normalized Root Mean Squared Error
log_mape = fliterdf_est_mean_dict_df['log_percentage_error'].mean()  # Mean Absolute Percentage Error

# log meduim指標
log_medium_mae = fliterdf_est_mean_dict_df['log_medium_absolute_error'].mean()  # Mean Absolute Error
log_medium_mse = fliterdf_est_mean_dict_df['log_medium_squared_error'].mean()  # Mean Squared Error
log_medium_rmse = np.sqrt(log_medium_mse)  # Root Mean Squared Error
log_medium_normse = log_medium_rmse / np.mean(fliterdf_est_mean_dict_df['log_true_value'])  # Normalized Root Mean Squared Error
log_medium_mape = fliterdf_est_mean_dict_df['log_medium_percentage_error'].mean()  # Mean Absolute Percentage Error

fliterdf_output_df = pd.DataFrame({
    "Mean_MAPE": [mape],
    "Mean_NRMSE": [normse],
    "Median_MAPE": [medium_mape],
    "Median_NRMSE": [medium_normse],
    "Log_Mean_MAPE": [log_mape],
    "Log_Mean_NRMSE": [log_normse],
    "Log_Median_MAPE": [log_medium_mape],
     "Log_Median_NRMSE": [log_medium_normse],
}, index=['Removed outliers'])


fliterdf_output_df.to_csv(r"C:\Users\Lab_205\Desktop\fliterdf_output_df.csv")

##　mean medium
print(f"Mean Absolute Percentage Error (MAPE): {mape}%")
print(f"Normalized Root Mean Squared Error (NRMSE): {normse}")

print(f"Normalized Root Mean Squared Error (NRMSE)for median: {medium_normse}%")
print(f"Mean Absolute Percentage Error (MAPE) for median: {medium_mape}%")


## log mean medium
print(f"Normalized Root Mean Squared Error (NRMSE)for log mean: {log_normse}%")
print(f"Mean Absolute Percentage Error (MAPE) for log mean: {log_mape}%")

print(f"Normalized Root Mean Squared Error (NRMSE)for log median: {log_medium_normse}%")
print(f"Mean Absolute Percentage Error (MAPE) for log median: {log_medium_mape}%")
fliterdf_est_mean_dict_df.to_csv(r"C:\Users\Lab_205\Desktop\fliterdf_est_mean_dict_df.csv")

# 視覺化分析 mean medium
plt.figure(figsize=(10, 5))

# 預估值與真實值的散點圖
plt.subplot(1, 2, 1)
plt.scatter(fliterdf_est_mean_dict_df['true_value'], fliterdf_est_mean_dict_df['mean_area'], color='blue')
plt.plot([fliterdf_est_mean_dict_df['true_value'].min(), fliterdf_est_mean_dict_df['true_value'].max()], [fliterdf_est_mean_dict_df['true_value'].min(), fliterdf_est_mean_dict_df['true_value'].max()], 'r--')

plt.xlabel('True Value(pixel)')
plt.ylabel('Mean Area')
plt.title('True Value vs Mean Area')


plt.subplot(1, 2, 2)
plt.scatter(fliterdf_est_mean_dict_df['true_value'], fliterdf_est_mean_dict_df['medium_mean_area'], color='blue')
plt.plot([fliterdf_est_mean_dict_df['true_value'].min(), fliterdf_est_mean_dict_df['true_value'].max()], [fliterdf_est_mean_dict_df['true_value'].min(), fliterdf_est_mean_dict_df['true_value'].max()], 'r--')

plt.xlabel('True Value(pixel)')
plt.ylabel('Median Area')
plt.title('True Value vs Median Area')

plt.show()

# 視覺化分析 log mean medium
plt.figure(figsize=(10, 5))

# 預估值與真實值的散點圖
plt.subplot(1, 2, 1)
plt.scatter(fliterdf_est_mean_dict_df['log_true_value'], fliterdf_est_mean_dict_df['log_mean_area'], color='blue')
plt.plot([fliterdf_est_mean_dict_df['log_true_value'].min(), fliterdf_est_mean_dict_df['log_true_value'].max()], [fliterdf_est_mean_dict_df['log_true_value'].min(), fliterdf_est_mean_dict_df['log_true_value'].max()], 'r--')


plt.xlabel('Log True Value(pixel)')
plt.ylabel('Log Mean Area')
plt.title('Log True Value vs Log Mean Area')


plt.subplot(1, 2, 2)
plt.scatter(fliterdf_est_mean_dict_df['log_true_value'], fliterdf_est_mean_dict_df['log_medium_mean_area'], color='blue')
plt.plot([fliterdf_est_mean_dict_df['log_true_value'].min(), fliterdf_est_mean_dict_df['log_true_value'].max()], [fliterdf_est_mean_dict_df['log_true_value'].min(), fliterdf_est_mean_dict_df['log_true_value'].max()], 'r--')

plt.xlabel('Log True Value(pixel)')
plt.ylabel('Log Median Area')
plt.title('Log True Value vs Log Median Area')

plt.show()