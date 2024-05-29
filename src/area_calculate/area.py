import pandas as pd
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from tqdm.auto import tqdm

def find_best_match_variable_length(segment, full_sequence, min_length:int=10, max_length:int=50):
    results = []

    full_length = len(full_sequence)
    for length in tqdm(range(min_length, max_length + 1), desc="Searching lengths"):
        best_distance = float('inf')
        best_start_index = -1
        best_end_index = -1
        
        for start_index in range(full_length):
            end_index = start_index + length
            if end_index > full_length:
                break
            subsequence = full_sequence[start_index:end_index]
            
            distance, _ = fastdtw(segment, subsequence, dist=euclidean)
            
            if distance < best_distance:
                best_distance = distance
                best_start_index = start_index
                best_end_index = end_index
        
        results.append({
            "size": length,
            "start_index": best_start_index,
            "end_index": best_end_index,
            "distance": best_distance
        })
        
        print(f"seiz {length}: Best match starts at index {best_start_index} and ends at index {best_end_index}")
        print(f"DTW distance: {best_distance}")

    results_df = pd.DataFrame(results)
    return results_df


def find_best_match_variable_length(segment, full_sequence, range_percentage=0.1,radius=3):
    segment_length = len(segment)
    min_length = int(segment_length * (1 - range_percentage))
    max_length = int(segment_length * (1 + range_percentage))
    
    results = []

    full_length = len(full_sequence)
    for length in tqdm(range(min_length, max_length + 1), desc="Searching lengths"):
        best_distance = float('inf')
        best_start_index = -1
        best_end_index = -1
        
        for start_index in range(full_length):
            end_index = start_index + length
            if end_index > full_length:
                break
            subsequence = full_sequence[start_index:end_index]
            
            distance, _ = fastdtw(segment, subsequence, dist=euclidean, radius=radius)
            
            if distance < best_distance:
                best_distance = distance
                best_start_index = start_index
                best_end_index = end_index
        
        results.append({
            "size": length,
            "start_index": best_start_index,
            "end_index": best_end_index,
            "distance": best_distance
        })
        
        print(f"size {length}: Best match starts at index {best_start_index} and ends at index {best_end_index}")
        print(f"DTW distance: {best_distance}")

    results_df = pd.DataFrame(results)
    return results_df

for i in tqdm(range(7003, 7073), desc="Processing files"):
    segment = pd.read_csv(r'C:\Users\Lab_205\Desktop\image_overlapping_project\dataset_output\find_pattern\overlapping_1\contourfiles\grad\down\7001_down.csv').values.reshape(-1, 1)
    grad7002 = pd.read_csv(rf'C:\Users\Lab_205\Desktop\image_overlapping_project\dataset_output\find_pattern\overlapping_1\contourfiles\grad_20\{i}_clear_gradient.csv').values.reshape(-1, 1)

    results_df = find_best_match_variable_length(segment, grad7002, range_percentage=0.1,radius=3)

    print(results_df)
    plt.plot(results_df['size'], results_df['distance'])
    plt.xlabel('Size')
    plt.ylabel('DTW distance')
    plt.title(f'DTW distance vs. Size 7001 vs {i} search =3')
    plt.show()  
    results_df.to_csv(f'C:\Users\Lab_205\Desktop\image_overlapping_project\dataset_output\find_pattern\overlapping_1\area\7001_{i}.csv', index=False)

