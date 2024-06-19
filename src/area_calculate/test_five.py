import pandas as pd
import numpy as np
import cv2 as cv
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

def dtw_distance(segment, subsequence, radius):
    distance, _ = fastdtw(segment, subsequence, dist=euclidean, radius=radius)
    return distance

def process_subsequence(start_index, segment, full_sequence, length, radius):
    end_index = start_index + length
    if end_index > len(full_sequence):
        return float('inf'), -1, -1
    subsequence = full_sequence[start_index:end_index]
    distance = dtw_distance(segment, subsequence, radius)
    return distance, start_index, end_index

def find_best_match_variable_length(segment, full_sequence, range_percentage=0.1, radius=3,outputid="", n_jobs=16):  # Reduce n_jobs
        
    segment_length = len(segment)
    min_length = int(segment_length * (1 - range_percentage))
    max_length = int(segment_length * (1 + range_percentage))
    
    results = []
    distances_record = []
    full_length = len(full_sequence)
    
    for length in tqdm(range(min_length, max_length + 1), desc="Searching lengths"):
        best_distance = float('inf')
        best_start_index = -1
        best_end_index = -1
        
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [executor.submit(process_subsequence, start_index, segment, full_sequence, length, radius) for start_index in range(full_length)]
            results_list = [future.result() for future in futures]
        
        for distance, start_index, end_index in results_list:
            if distance < best_distance:
                best_distance = distance
                best_start_index = start_index
                best_end_index = end_index
            
            distances_record.append({
                "size": length,
                "start_index": start_index,
                "end_index": end_index,
                "distance": distance
            })
        
        results.append({
            "size": length,
            "start_index": best_start_index,
            "end_index": best_end_index,
            "distance": best_distance
        })

    results_df = pd.DataFrame(results)
    distances_record_df = pd.DataFrame(distances_record)

    return results_df, distances_record_df

if __name__ == '__main__':
    
 ###### step 3   
    for j in range(7011,7016):
        segment = pd.read_csv(fr'C:\Users\Lab_205\Desktop\overlapping\1-overlapping_image\clear\contourfiles\down\\{j}_down.csv').values.reshape(-1, 1)
                            #7000 ~7073 
        for i in tqdm(range(7000, 7073), desc="Processing files"):
            reference = pd.read_csv(rf'C:\\Users\\Lab_205\\Desktop\\image_overlapping_project\\dataset_output\\find_pattern\\overlapping_1\\contourfiles\\grad_20\\{i}_clear_gradient.csv').values.reshape(-1, 1)

            results_df, distances_record_df = find_best_match_variable_length(segment, reference, range_percentage=0.1, radius=3,outputid=f"{j}_{i}", n_jobs=16)

            # print(results_df)
            plt.plot(results_df['size'], results_df['distance'])
            plt.xlabel('Size')
            plt.ylabel('DTW distance')
            plt.title(f'DTW distance vs. Size {j} vs {i} search radius=3')
            plt.savefig(fr'C:\Users\Lab_205\Desktop\overlapping\{j}\{j}{i}_search_radius_3.png')
            # plt.show()
            results_df.to_csv(fr'C:\Users\Lab_205\Desktop\overlapping\{j}\{j}_{i}.csv', index=False)
            plt.clf()
            distances_record_df.to_csv(fr'C:\Users\Lab_205\Desktop\overlapping\{j}\{j}_{i}_all.csv', index=False)


