import pytest
import os 
import sys
import tqdm
import tempfile
sys.path.append(r"../../")
from src.dtw.dtw import *  

@pytest.fixture(scope="module")
def temp_dir():
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


def test_get_csv_path(temp_dir):
    currpath = os.getcwd()

    path1 = os.path.join(temp_dir, "123.csv")
    path2 = os.path.join(temp_dir, "456.csv")
    path3 = os.path.join(temp_dir, "789.txt")
    
    with open(path1,"w") as f1:
        f1.writelines("123")
    with open (path2,"w") as f2:
        f2.write("456")
    with open (path3,"w") as f3:
        f3.write("789")
    
    result_paths = get_all_csv_path(temp_dir)
    expected_paths = [os.path.abspath(path1), os.path.abspath(path2)]

    assert sorted(result_paths) == sorted(expected_paths), "The function should return paths of CSV files only."


def test_get_all_gradients_with_test_files():
    test_files_dir = os.path.join(os.path.dirname(__file__), "test_data")
    file1_path = os.path.join(test_files_dir, "1060_test.csv")
    file2_path = os.path.join(test_files_dir, "1061_test.csv")
    
    file_list = [file1_path, file2_path]
    gradients_dict = get_all_gradients(file_list)
    
 
    assert len(gradients_dict) == 2
    assert "1060_test" in gradients_dict, "The 1st gradient value does not input the dic"
    assert "1061_test" in gradients_dict, "The 2nd gradient value does not input the dic"
    assert isinstance(gradients_dict["1060_test"], np.ndarray),"The dic 1st value type not np.ndarray"
    assert isinstance(gradients_dict["1061_test"], np.ndarray),"The dic 2nd value type not np.ndarray" 


def test_dis():
    grad_1 = np.array([1, 2, 3]).reshape(-1, 1)
    grad_2 = np.array([1, 2, 3]).reshape(-1, 1)
    grad_3 = np.array([4, 5, 6]).reshape(-1, 1)

    distance_same = dis(grad_1, grad_1)
    assert distance_same == 0, "Distance between identical gradients should be 0"

    distance_diff = dis(grad_1, grad_3)
    assert distance_diff > 0, "Distance between different gradients should be greater than 0"
    expected_distance, _ = fastdtw(grad_1, grad_3, dist=euclidean)
    assert distance_diff == expected_distance, "The dis function should return the correct DTW distance"



def test_calculate_pairwise_dist():
    all_gradient_dict = {
        'leaf1': {'id1': np.array([1, 2, 3]).reshape(-1, 1), 'id2': np.array([4, 5, 6]).reshape(-1, 1)},
        'leaf2': {'id1': np.array([7, 8, 9]).reshape(-1, 1), 'id2': np.array([10, 11, 12]).reshape(-1, 1)}
    }
    leaves_grads = ['leaf1', 'leaf2']
    ids = ['id1', 'id2']

    pairwise_dist = calculate_pairwise_dist(all_gradient_dict, leaves_grads, ids)

    for leaf_grad, id_ in product(leaves_grads, ids):
        name = f"{leaf_grad}_{id_}"
        assert pairwise_dist[name][name] == 0, f"Distance between {name} and itself should be 0"

        for leaf_grad_2, id_2 in product(leaves_grads, ids):
            name_2 = f"{leaf_grad_2}_{id_2}"

            assert pairwise_dist[name][name_2] == pairwise_dist[name_2][name], f"Distance between {name} and {name_2} should be symmetric"
