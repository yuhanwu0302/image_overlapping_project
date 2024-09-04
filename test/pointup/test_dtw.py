import pytest
import os 
import sys
import tqdm
import tempfile
from dtw.dtw import *  

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

    expected_gradients_dict = {
        "1060_test": np.array([[0], [1.5], [1.5], [1.5], [1.5], [1.5], [1.5], [1.307692], [1.5], [1.384615]]),

        "1061_test": np.array([[0], [1.125], [1.125], [1.0625], [1.0625], [1.133333], [1.133333], [1.], [1.058823], [1.058823]])
    }

    assert set(gradients_dict.keys()) == set(expected_gradients_dict.keys()), "Keys do not match."

    for key in gradients_dict:
        assert np.array_equal(gradients_dict[key], expected_gradients_dict[key]), f"Arrays do not match for key: {key}"



def test_dis():
    grad_1 = np.array([1, 2, 3]).reshape(-1, 1)
    grad_2 = np.array([1, 2, 3]).reshape(-1, 1)
    grad_3 = np.array([4, 5, 6]).reshape(-1, 1)
    grad_4 = np.array([1,1,2,2,3,3]).reshape(-1, 1)
    grad_5 = np.array([4,4,4,5,5,5,6,6,6]).reshape(-1, 1)
    grad_6 = np.array([4,4,1,4,5,5,2,5,6,6,3,6]).reshape(-1, 1)

    dis_pair = ["grad1_3","grad1_4","grad1_5","grad1_6",
                "grad3_4","grad3_5","grad3_6",
                "grad4_5","grad4_6",
                "grad5_6"] 
    
    expected_dis = {"grad1_3": 9.0, "grad1_4":0.0, "grad1_5":21.0,"grad1_6":23.0,"grad3_4":15.0,"grad3_5":0.0,"grad3_6":9.0, "grad4_5":24.0,"grad4_6":25.0,
    "grad5_6":9.0
    }
    result = {"grad1_3": dis(grad_1,grad_3), "grad1_4":dis(grad_1,grad_4), "grad1_5":dis(grad_1,grad_5),"grad1_6":dis(grad_1,grad_6),"grad3_4":dis(grad_3,grad_4),"grad3_5":dis(grad_3,grad_5),"grad3_6":dis(grad_3,grad_6), "grad4_5":dis(grad_4,grad_5),"grad4_6":dis(grad_4,grad_6),
    "grad5_6":9.0
    }

    distance_self = dis(grad_1, grad_2)
    assert distance_self == 0.0, "Distance between identical gradients should be 0"
    
    for key in dis_pair:
        assert expected_dis[key]  == result[key]
        "The dis function should return the correct DTW distance"



def test_calculate_pairwise_dist():
    all_gradient_dict = {
        'leaf1': {'id1': np.array([1, 2, 3]).reshape(-1, 1), 'id2': np.array([4, 5, 6]).reshape(-1, 1)},
        'leaf2': {'id1': np.array([7, 8, 9]).reshape(-1, 1), 'id2': np.array([10, 11, 12]).reshape(-1, 1)}
    }
    leaves_grads = ['leaf1', 'leaf2']
    ids = ['id1', 'id2']

    pairwise_dist = calculate_pairwise_dist(all_gradient_dict, leaves_grads, ids)
    expect_dic = defaultdict(dict,
            {'leaf1_id1': {'leaf1_id1': 0.0,
              'leaf1_id2': 9.0,
              'leaf2_id1': 18.0,
              'leaf2_id2': 27.0},
             'leaf1_id2': {'leaf1_id1': 9.0,
              'leaf1_id2': 0.0,
              'leaf2_id1': 9.0,
              'leaf2_id2': 18.0},
             'leaf2_id1': {'leaf1_id1': 18.0,
              'leaf1_id2': 9.0,
              'leaf2_id1': 0.0,
              'leaf2_id2': 9.0},
             'leaf2_id2': {'leaf1_id1': 27.0,
              'leaf1_id2': 18.0,
              'leaf2_id1': 9.0,
              'leaf2_id2': 0.0}})


    for leaf_grad, id_ in product(leaves_grads, ids):
        name = f"{leaf_grad}_{id_}"
        assert pairwise_dist[name][name] == expect_dic[name][name] , f"Distance between {name} and itself should be 0"

        for leaf_grad_2, id_2 in product(leaves_grads, ids):
            name_2 = f"{leaf_grad_2}_{id_2}"

            assert expect_dic == pairwise_dist,"The dict key and value is incorrect"