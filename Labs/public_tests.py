import numpy as np
import math
import random
from tensorflow.keras.activations import relu, linear
from tensorflow.keras.layers import Dense
from test_utils import test

def compute_cost_test(target):
    # print("Using X with shape (4, 1)")
    # Case 1
    x = np.array([2, 4, 6, 8]).T
    y = np.array([7, 11, 15, 19]).T
    initial_w = 2
    initial_b = 3.0
    cost = target(x, y, initial_w, initial_b)
    assert cost == 0, f"Case 1: Cost must be 0 for a perfect prediction but got {cost}"
    
    # Case 2
    x = np.array([2, 4, 6, 8]).T
    y = np.array([7, 11, 15, 19]).T
    initial_w = np.array([2.])
    initial_b = 1.0
    cost = target(x, y, initial_w, initial_b)
    assert cost == 2, f"Case 2: Cost must be 2 but got {cost}"
    
    # print("Using X with shape (5, 1)")
    # Case 3
    x = np.array([1.5, 2.5, 3.5, 4.5, 1.5]).T
    y = np.array([4, 7, 10, 13, 5]).T
    initial_w = 1
    initial_b = 0.0
    cost = target(x, y, initial_w, initial_b)
    assert np.isclose(cost, 15.325), f"Case 3: Cost must be 15.325 for a perfect prediction but got {cost}"
    
    # Case 4
    initial_b = 1.0
    cost = target(x, y, initial_w, initial_b)
    assert np.isclose(cost, 10.725), f"Case 4: Cost must be 10.725 but got {cost}"
    
    # Case 5
    y = y - 2
    initial_b = 1.0
    cost = target(x, y, initial_w, initial_b)
    assert  np.isclose(cost, 4.525), f"Case 5: Cost must be 4.525 but got {cost}"
    
    print("\033[92mAll tests passed!")
    
def compute_gradient_test(target):
    print("Using X with shape (4, 1)")
    # Case 1
    x = np.array([2, 4, 6, 8]).T
    y = np.array([4.5, 8.5, 12.5, 16.5]).T
    initial_w = 2.
    initial_b = 0.5
    dj_dw, dj_db = target(x, y, initial_w, initial_b)
    #assert dj_dw.shape == initial_w.shape, f"Wrong shape for dj_dw. {dj_dw} != {initial_w.shape}"
    assert dj_db == 0.0, f"Case 1: dj_db is wrong: {dj_db} != 0.0"
    assert np.allclose(dj_dw, 0), f"Case 1: dj_dw is wrong: {dj_dw} != [[0.0]]"
    
    # Case 2 
    x = np.array([2, 4, 6, 8]).T
    y = np.array([4, 7, 10, 13]).T + 2
    initial_w = 1.5
    initial_b = 1
    dj_dw, dj_db = target(x, y, initial_w, initial_b)
    #assert dj_dw.shape == initial_w.shape, f"Wrong shape for dj_dw. {dj_dw} != {initial_w.shape}"
    assert dj_db == -2, f"Case 2: dj_db is wrong: {dj_db} != -2"
    assert np.allclose(dj_dw, -10.0), f"Case 1: dj_dw is wrong: {dj_dw} != -10.0"   
    
    print("\033[92mAll tests passed!")
    
def sigmoid_test(target):
    z = np.array([2.5, 0])
    g = target(z)
    assert g.shape == z.shape, f"Output should have the same shape as the input. Expected {z.shape}, got {g.shape}"

    g_scalar = target(3.0)
    assert g_scalar.shape == (), f"Scalar input should also output a scalar. Got shape {g_scalar.shape}"

    expected = 0.9525741268224334
    assert np.isclose(g_scalar, expected), f"Failed for scalar input. Expected {expected}, got {g_scalar}"

    z = np.array([2.5, 0])
    g = target(z)
    expected = np.array([0.92414182, 0.5])
    assert np.allclose(g, expected), f"Failed for 1D array. Expected {expected}, got {g}"

    z = np.array([[2.5, -2.5], [0, 1]])
    g = target(z)
    expected = np.array([[0.92414182, 0.07585818], [0.5, 0.73105858]])
    assert np.allclose(g, expected), f"Failed for 2D array.\nExpected:\n{expected}\nGot:\n{g}"
    
    print('\033[92mAll tests passed!')

def predict_test(target):
    np.random.seed(5)
    b = 0.5    
    w = np.random.randn(3)
    X = np.random.randn(8, 3)
    
    result = target(X, w, b)
    wrong_1 = [1., 1., 0., 0., 1., 0., 0., 1.]
    expected_1 = [1., 1., 1., 0., 1., 0., 0., 1.]
    if np.allclose(result, wrong_1):
        raise ValueError("Did you apply the sigmoid before applying the threshold?")
    assert result.shape == (len(X),), f"Wrong length. Expected : {(len(X),)} got: {result.shape}"
    assert np.allclose(result, expected_1), f"Wrong output: Expected : {expected_1} got: {result}"
    
    b = -1.7    
    w = np.random.randn(4) + 0.6
    X = np.random.randn(6, 4)
    
    result = target(X, w, b)
    expected_2 = [0., 0., 0., 1., 1., 0.]
    assert result.shape == (len(X),), f"Wrong length. Expected : {(len(X),)} got: {result.shape}"
    assert np.allclose(result,expected_2), f"Wrong output: Expected : {expected_2} got: {result}"

    print('\033[92mAll tests passed!')

def compute_cost_reg_test(target):
    np.random.seed(1)
    w = np.random.randn(3)
    b = 0.4
    X = np.random.randn(6, 3)
    y = np.array([0, 1, 1, 0, 1, 1])
    lambda_ = 0.1
    expected_output = target(X, y, w, b, lambda_)
    
    assert np.isclose(expected_output, 0.5469746792761936), f"Wrong output. Expected: {0.5469746792761936} got:{expected_output}"
    
    w = np.random.randn(5)
    b = -0.6
    X = np.random.randn(8, 5)
    y = np.array([1, 0, 1, 0, 0, 1, 0, 1])
    lambda_ = 0.01
    output = target(X, y, w, b, lambda_)
    assert np.isclose(output, 1.2608591964119995), f"Wrong output. Expected: {1.2608591964119995} got:{output}"
    
    w = np.array([2, 2, 2, 2, 2])
    b = 0
    X = np.zeros((8, 5))
    y = np.array([0.5] * 8)
    lambda_ = 3
    output = target(X, y, w, b, lambda_)
    expected = -np.log(0.5) + 3. / (2. * 8.) * 20.
    assert np.isclose(output, expected), f"Wrong output. Expected: {expected} got:{output}"
    
    print('\033[92mAll tests passed!')

def compute_gradient_reg_test(target):
    np.random.seed(1)
    w = np.random.randn(5)
    b = 0.2
    X = np.random.randn(7, 5)
    y = np.array([0, 1, 1, 0, 1, 1, 0])
    lambda_ = 0.1
    expected1 = (-0.1506447567869257, np.array([ 0.19530838, -0.00632206,  0.19687367,  0.15741161,  0.02791437]))
    dj_db, dj_dw = target(X, y, w, b, lambda_)
    
    assert np.isclose(dj_db, expected1[0]), f"Wrong dj_db. Expected: {expected1[0]} got: {dj_db}"
    assert np.allclose(dj_dw, expected1[1]), f"Wrong dj_dw. Expected: {expected1[1]} got: {dj_dw}"

    
    w = np.random.randn(7)
    b = 0
    X = np.random.randn(7, 7)
    y = np.array([1, 0, 0, 0, 1, 1, 0])
    lambda_ = 0
    expected2 = (0.02660329857573818, np.array([ 0.23567643, -0.06921029, -0.19705212, -0.0002884 ,  0.06490588,
        0.26948175,  0.10777992]))
    dj_db, dj_dw = target(X, y, w, b, lambda_)
    assert np.isclose(dj_db, expected2[0]), f"Wrong dj_db. Expected: {expected2[0]} got: {dj_db}"
    assert np.allclose(dj_dw, expected2[1]), f"Wrong dj_dw. Expected: {expected2[1]} got: {dj_dw}"
    
    print('\033[92mAll tests passed!') 

def compute_cost_test2(target):
    X = np.array([[0, 0, 0, 0]]).T
    y = np.array([0, 0, 0, 0])
    w = np.array([0])
    b = 1
    result = target(X, y, w, b)
    if math.isinf(result):
        raise ValueError("Did you get the sigmoid of z_wb?")
    
    np.random.seed(17)  
    X = np.random.randn(5, 2)
    y = np.array([1, 0, 0, 1, 1])
    w = np.random.randn(2)
    b = 0
    result = target(X, y, w, b)
    assert np.isclose(result, 2.15510667), f"Wrong output. Expected: {2.15510667} got: {result}"
    
    X = np.random.randn(4, 3)
    y = np.array([1, 1, 0, 0])
    w = np.random.randn(3)
    b = 0
    
    result = target(X, y, w, b)
    assert np.isclose(result, 0.80709376), f"Wrong output. Expected: {0.80709376} got: {result}"

    X = np.random.randn(4, 3)
    y = np.array([1, 0,1, 0])
    w = np.random.randn(3)
    b = 3
    result = target(X, y, w, b)
    assert np.isclose(result, 0.4529660647), f"Wrong output. Expected: {0.4529660647} got: {result}. Did you inizialized z_wb = b?"
    
    print('\033[92mAll tests passed!')

def compute_gradient_test2(target):
    np.random.seed(1)
    X = np.random.randn(7, 3)
    y = np.array([1, 0, 1, 0, 1, 1, 0])
    test_w = np.array([1, 0.5, -0.35])
    test_b = 1.7
    dj_db, dj_dw  = target(X, y, test_w, test_b)
    
    assert np.isclose(dj_db, 0.28936094), f"Wrong value for dj_db. Expected: {0.28936094} got: {dj_db}" 
    assert dj_dw.shape == test_w.shape, f"Wrong shape for dj_dw. Expected: {test_w.shape} got: {dj_dw.shape}" 
    assert np.allclose(dj_dw, [-0.11999166, 0.41498775, -0.71968405]), f"Wrong values for dj_dw. Got: {dj_dw}"

    print('\033[92mAll tests passed!')

def compute_entropy_test(target):
    y = np.array([1] * 10)
    result = target(y)
    
    assert result == 0, "Entropy must be 0 with array of ones"
    
    y = np.array([0] * 10)
    result = target(y)
    
    assert result == 0, "Entropy must be 0 with array of zeros"
    
    y = np.array([0] * 12 + [1] * 12)
    result = target(y)
    
    assert result == 1, "Entropy must be 1 with same ammount of ones and zeros"
    
    y = np.array([1, 0, 1, 0, 1, 1, 1, 0, 1])
    assert np.isclose(target(y), 0.918295, atol=1e-6), "Wrong value. Something between 0 and 1"
    assert np.isclose(target(-y + 1), target(y), atol=1e-6), "Wrong value"
    
    print("\033[92m All tests passed. ")

def split_dataset_test(target):

    # Case 1
    X = np.array([[1, 0], 
         [1, 0], 
         [1, 1], 
         [0, 0], 
         [0, 1]])
    X_t = np.array([[0, 1, 0, 1, 0]])
    X = np.concatenate((X, X_t.T), axis=1)

    left, right = target(X, list(range(5)), 2)
    expected = {'left': np.array([1, 3]),
                'right': np.array([0, 2, 4])}

    assert type(left) == list, f"Wrong type for left. Expected: list got: {type(left)}"
    assert type(right) == list, f"Wrong type for right. Expected: list got: {type(right)}"
    
    assert type(left[0]) == int, f"Wrong type for elements in the left list. Expected: int got: {type(left[0])}"
    assert type(right[0]) == int, f"Wrong type for elements in the right list. Expected: number got: {type(right[0])}"
    
    assert len(left) == 2, f"left must have 2 elements but got: {len(left)}"
    assert len(right) == 3, f"right must have 3 elements but got: {len(right)}"

    assert np.allclose(right, expected['right']), f"Wrong value for right. Expected: { expected['right']} \ngot: {right}"
    assert np.allclose(left, expected['left']), f"Wrong value for left. Expected: { expected['left']} \ngot: {left}"


    # Case 2
    X = np.array([[0, 1], 
         [1, 1], 
         [1, 1], 
         [0, 0], 
         [1, 0]])
    X_t = np.array([[0, 1, 0, 1, 0]])
    X = np.concatenate((X_t.T, X), axis=1)

    left, right = target(X, list(range(5)), 0)
    expected = {'left': np.array([1, 3]),
                'right': np.array([0, 2, 4])}


    assert len(left) == 2, f"left must have 2 elements but got: {len(left)}" 
    assert len(right) == 3, f"right must have 3 elements but got: {len(right)}"
    assert np.allclose(right, expected['right']) and np.allclose(left, expected['left']), f"Wrong value when target is at index 0."


    # Case 3
    X = (np.random.rand(11, 3) > 0.5) * 1 # Just random binary numbers
    X_t = np.array([[0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0]])
    X = np.concatenate((X, X_t.T), axis=1)

    left, right = target(X, [1, 2, 3, 6, 7, 9, 10], 3)
    expected = {'left': np.array([1, 3, 6]),
                'right': np.array([2, 7, 9, 10])}

    assert np.allclose(right, expected['right']) and np.allclose(left, expected['left']), f"Wrong value when target is at index 0. \nExpected: {expected} \ngot: \{left:{left}, 'right': {right}\}"
 
    
    print("\033[92m All tests passed.")

def compute_information_gain_test(target):
    X = np.array([[1, 0], 
         [1, 0], 
         [1, 0], 
         [0, 0], 
         [0, 1]])
    
    y = np.array([[0, 0, 0, 0, 0]]).T
    node_indexes = list(range(5))

    result1 = target(X, y, node_indexes, 0)
    result2 = target(X, y, node_indexes, 0)
    
    assert result1 == 0 and result2 == 0, f"Information gain must be 0 when target variable is pure. Got {result1} and {result2}"
    
    y = np.array([[0, 1, 0, 1, 0]]).T
    node_indexes = list(range(5))
    
    result = target(X, y, node_indexes, 0)
    assert np.isclose(result, 0.019973, atol=1e-6), f"Wrong information gain. Expected {0.019973} got: {result}"
    
    result = target(X, y, node_indexes, 1)
    assert np.isclose(result, 0.170951, atol=1e-6), f"Wrong information gain. Expected {0.170951} got: {result}"

    node_indexes = list(range(4))
    result = target(X, y, node_indexes, 0)
    assert np.isclose(result, 0.311278, atol=1e-6), f"Wrong information gain. Expected {0.311278} got: {result}"

    result = target(X, y, node_indexes, 1)
    assert np.isclose(result, 0, atol=1e-6), f"Wrong information gain. Expected {0.0} got: {result}"

    print("\033[92m All tests passed.")

def get_best_split_test(target):
    X = np.array([[1, 0], 
         [1, 0], 
         [1, 0], 
         [0, 0], 
         [0, 1]])

    y = np.array([[0, 0, 0, 0, 0]]).T
    node_indexes = list(range(5))

    result = target(X, y, node_indexes)
    
    assert result == -1, f"When the target variable is pure, there is no best split to do. Expected -1, got {result}"
    
    y = X[:,0]
    result = target(X, y, node_indexes)
    assert result == 0, f"If the target is fully correlated with other feature, that feature must be the best split. Expected 0, got {result}"
    y = X[:,1]
    result = target(X, y, node_indexes)
    assert result == 1, f"If the target is fully correlated with other feature, that feature must be the best split. Expected 1, got {result}"

    y = 1 - X[:,0]
    result = target(X, y, node_indexes)
    assert result == 0, f"If the target is fully correlated with other feature, that feature must be the best split. Expected 0, got {result}"

    y = np.array([[0, 1, 0, 1, 0]]).T
    result = target(X, y, node_indexes)
    assert result == 1, f"Wrong result. Expected 1, got {result}"

    y = np.array([[0, 1, 0, 1, 0]]).T    
    node_indexes = [2, 3, 4]
    result = target(X, y, node_indexes)
    assert result == 0, f"Wrong result. Expected 0, got {result}"

    n_samples = 100
    X0 = np.array([[1] * n_samples])
    X1 = np.array([[0] * n_samples])
    X2 = (np.random.rand(1, 100) > 0.5) * 1
    X3 = np.array([[1] * int(n_samples / 2) + [0] * int(n_samples / 2)])
    
    y = X2.T
    node_indexes = list(range(20, 80))
    X = np.array([X0, X1, X2, X3]).T.reshape(n_samples, 4)
    result = target(X, y, node_indexes)
    
    assert result == 2, f"Wrong result. Expected 2, got {result}"
    
    y = X0.T
    result = target(X, y, node_indexes)
    assert result == -1, f"When the target variable is pure, there is no best split to do. Expected -1, got {result}"
    print("\033[92m All tests passed.")

def compute_centroids_test(target):
    # With 3 centroids
    X = np.array([[-1, -1], [-1.5, -1.5], [-1.5, 1],
                  [-1, 1.5], [2.5, 1.5], [-1.1, -1.7], [-1.6, 1.2]])
    idx = np.array([1, 1, 1, 0, 0, 0, 2])
    K = 3
    centroids = target(X, idx, K)
    expected_centroids = np.array([[0.13333333,  0.43333333],
                                   [-1.33333333, -0.5      ],
                                   [-1.6,        1.2       ]])
    
    assert type(centroids) == np.ndarray, "Wrong type"
    assert centroids.shape == (K, X.shape[1]), f"Wrong shape. Expected: {centroids.shape} got: {(K, X.shape[1])}"
    assert np.allclose(centroids, expected_centroids), f"Wrong values. Expected: {expected_centroids}, got: {centroids}"
    
    X = np.array([[2, 2.5], [2.5, 2.5], [-1.5, -1.5],
                  [2, 2], [-1.5, -1], [-1, -1]])
    idx = np.array([0, 0, 1, 0, 1, 1])
    K = 2
    centroids = target(X, idx, K)
    expected_centroids = np.array([[[ 2.16666667,  2.33333333],
                                    [-1.33333333, -1.16666667]]])
    
    assert type(centroids) == np.ndarray, "Wrong type"
    assert centroids.shape == (K, X.shape[1]), f"Wrong shape. Expected: {(len(X),)} got: {idx.shape}"
    assert np.allclose(centroids, expected_centroids), f"Wrong values. Expected: {expected_centroids}, got: {centroids}"
    
    print("\033[92mAll tests passed!")
    
def find_closest_centroids_test(target):
    # With 2 centroids
    X = np.array([[-1, -1], [-1.5, -1.5], [-1.5, -1],
                  [2, 2],[2.5, 2.5],[2, 2.5]])
    initial_centroids = np.array([[-1, -1], [2, 2]])
    idx = target(X, initial_centroids)
    
    assert type(idx) == np.ndarray, "Wrong type"
    assert idx.shape == (len(X),), f"Wrong shape. Expected: {(len(X),)} got: {idx.shape}"
    assert np.allclose(idx, [0, 0, 0, 1, 1, 1]), "Wrong values"
    
    # With 3 centroids
    X = np.array([[-1, -1], [-1.5, -1.5], [-1.5, 1],
                  [-1, 1.5], [2.5, 1.5], [2, 2]])
    initial_centroids = np.array([[2.5, 2], [-1, -1], [-1.5, 1.]])
    idx = target(X, initial_centroids)
    
    assert type(idx) == np.ndarray, "Wrong type"
    assert idx.shape == (len(X),), f"Wrong shape. Expected: {(len(X),)} got: {idx.shape}"
    assert np.allclose(idx, [1, 1, 2, 2, 0, 0]), f"Wrong values. Expected {[2, 2, 0, 0, 1, 1]}, got: {idx}"
    
    # With 3 centroids
    X = np.array([[-1, -1], [-1.5, -1.5], [-1.5, 1],
                  [-1, 1.5], [2.5, 1.5], [-1.1, -1.7], [-1.6, 1.2]])
    initial_centroids = np.array([[2.5, 2], [-1, -1], [-1.5, 1.]])
    idx = target(X, initial_centroids)
    
    assert type(idx) == np.ndarray, "Wrong type"
    assert idx.shape == (len(X),), f"Wrong shape. Expected: {(len(X),)} got: {idx.shape}"
    assert np.allclose(idx, [1, 1, 2, 2, 0, 1, 2]), f"Wrong values. Expected {[2, 2, 0, 0, 1, 1]}, got: {idx}"
    
    print("\033[92mAll tests passed!")

def select_threshold_test(target):
    p_val = np.array([i / 100 for i in range(30)])
    y_val = np.array([1] * 5 + [0] * 25)
    
    best_epsilon, best_F1 = target(y_val, p_val)
    assert np.isclose(best_epsilon, 0.04, atol=0.3 / 1000), f"Wrong best_epsilon. Expected: {0.04} got: {best_epsilon}"
    assert best_F1 == 1, f"Wrong best_F1. Expected: 1 got: {best_F1}"
    
    y_val = np.array([1] * 5 + [0] * 25)
    y_val[2] = 0 # Introduce noise
    best_epsilon, best_F1 = target(y_val, p_val)
    assert np.isclose(best_epsilon, 0.04, atol=0.3 / 1000), f"Wrong best_epsilon. Expected: {0.04} got: {best_epsilon}"
    assert np.isclose(best_F1, 0.8888888), f"Wrong best_F1. Expected: 0.8888888 got: {best_F1}"
    
    p_val = np.array([i / 1000 for i in range(50)])
    y_val = np.array([1] * 8 + [0] * 42)
    y_val[5] = 0
    index = [*range(50)]
    random.shuffle(index)
    p_val = p_val[index]
    y_val = y_val[index]

    best_epsilon, best_F1 = target(y_val, p_val)
    assert np.isclose(best_epsilon, 0.007, atol=0.05 / 1000), f"Wrong best_epsilon. Expected: {0.0070070} got: {best_epsilon}"
    assert np.isclose(best_F1, 0.933333333), f"Wrong best_F1. Expected: 0.933333333 got: {best_F1}"
    print("\033[92mAll tests passed!")

def estimate_gaussian_test(target):
    np.random.seed(273)
    
    X = np.array([[1, 1, 1], 
                  [2, 2, 2], 
                  [3, 3, 3]]).T
    
    mu, var = target(X)
    
    assert type(mu) == np.ndarray, f"Wrong type for mu. Expected: {np.ndarray} got: {type(mu)}"
    assert type(var) == np.ndarray, f"Wrong type for var. Expected: {np.ndarray} got: {type(var)}"
    
    assert mu.shape == (X.shape[1],), f"Wrong shape for mu. Expected: {(X.shape[1],)} got: {mu.shape}"
    assert type(var) == np.ndarray, f"Wrong shape for var. Expected: {(X.shape[1],)} got: {var.shape}"
    
    assert np.allclose(mu, [1., 2., 3.]), f"Wrong value for mu. Expected: {[1, 2, 3]} got: {mu}"
    assert np.allclose(var, [0., 0., 0.]), f"Wrong value for var. Expected: {[0, 0, 0]} got: {var}"
    
    X = np.array([[1, 2, 3], 
                  [2, 4, 6], 
                  [3, 6, 9]]).T
    
    mu, var = target(X)
    
    assert type(mu) == np.ndarray, f"Wrong type for mu. Expected: {np.ndarray} got: {type(mu)}"
    assert type(var) == np.ndarray, f"Wrong type for var. Expected: {np.ndarray} got: {type(var)}"
    
    assert mu.shape == (X.shape[1],), f"Wrong shape for mu. Expected: {(X.shape[1],)} got: {mu.shape}"
    assert type(var) == np.ndarray, f"Wrong shape for var. Expected: {(X.shape[1],)} got: {var.shape}"
    
    assert np.allclose(mu, [2., 4., 6.]), f"Wrong value for mu. Expected: {[2., 4., 6.]} got: {mu}"
    assert np.allclose(var, [2. / 3, 8. / 3., 18. / 3.]), f"Wrong value for var. Expected: {[2. / 3, 8. / 3., 18. / 3.]} got: {var}"
    
    
    m = 500
    X = np.array([np.random.normal(0, 1, m), 
                  np.random.normal(1, 2, m), 
                  np.random.normal(3, 1.5, m)]).T
    
    mu, var = target(X)
    
    assert type(mu) == np.ndarray, f"Wrong type for mu. Expected: {np.ndarray} got: {type(mu)}"
    assert type(var) == np.ndarray, f"Wrong type for var. Expected: {np.ndarray} got: {type(var)}"
    
    assert mu.shape == (X.shape[1],), f"Wrong shape for mu. Expected: {(X.shape[1],)} got: {mu.shape}"
    assert type(var) == np.ndarray, f"Wrong shape for var. Expected: {(X.shape[1],)} got: {var.shape}"
    
    assert np.allclose(mu, [0., 1., 3.], atol=0.2), f"Wrong value for mu. Expected: {[0, 1, 3]} got: {mu}"
    assert np.allclose(var, np.square([1., 2., 1.5]), atol=0.2), f"Wrong value for var. Expected: {np.square([1., 2., 1.5])} got: {var}"
    
    print("\033[92mAll tests passed!")

def test_tower(target):
    num_outputs = 32
    i = 0
    assert len(target.layers) == 3, f"Wrong number of layers. Expected 3 but got {len(target.layers)}"
    expected = [[Dense, [None, 256], relu],
                [Dense, [None, 128], relu],
                [Dense, [None, num_outputs], linear]]

    for layer in target.layers:
        assert type(layer) == expected[i][0], \
            f"Wrong type in layer {i}. Expected {expected[i][0]} but got {type(layer)}"
        output_shape = layer.output.shape
        output_shape = output_shape.as_list() if hasattr(output_shape, "as_list") else list(output_shape)
        assert output_shape == expected[i][1], \
            f"Wrong number of units in layer {i}. Expected {expected[i][1]} but got {output_shape}"
        assert layer.activation == expected[i][2], \
            f"Wrong activation in layer {i}. Expected {expected[i][2]} but got {layer.activation}"
        i = i + 1

    print("\033[92mAll tests passed!")
    

def test_cofi_cost_func(target):
    num_users_r = 4
    num_movies_r = 5 
    num_features_r = 3

    X_r = np.ones((num_movies_r, num_features_r))
    W_r = np.ones((num_users_r, num_features_r))
    b_r = np.zeros((1, num_users_r))
    Y_r = np.zeros((num_movies_r, num_users_r))
    R_r = np.zeros((num_movies_r, num_users_r))
    
    J = target(X_r, W_r, b_r, Y_r, R_r, 2);
    assert not np.isclose(J, 13.5), f"Wrong value. Got {J}. Did you multiply the regularization term by lambda_?"
    assert np.isclose(J, 27), f"Wrong value. Expected {27}, got {J}. Check the regularization term"
    
    
    X_r = np.ones((num_movies_r, num_features_r))
    W_r = np.ones((num_users_r, num_features_r))
    b_r = np.ones((1, num_users_r))
    Y_r = np.ones((num_movies_r, num_users_r))
    R_r = np.ones((num_movies_r, num_users_r))

    # Evaluate cost function
    J = target(X_r, W_r, b_r, Y_r, R_r, 0);
    
    assert np.isclose(J, 90), f"Wrong value. Expected {90}, got {J}. Check the term without the regularization"
    
    
    X_r = np.ones((num_movies_r, num_features_r))
    W_r = np.ones((num_users_r, num_features_r))
    b_r = np.ones((1, num_users_r))
    Y_r = np.zeros((num_movies_r, num_users_r))
    R_r = np.ones((num_movies_r, num_users_r))

    # Evaluate cost function
    J = target(X_r, W_r, b_r, Y_r, R_r, 0);
    
    assert np.isclose(J, 160), f"Wrong value. Expected {160}, got {J}. Check the term without the regularization"
    
    X_r = np.ones((num_movies_r, num_features_r))
    W_r = np.ones((num_users_r, num_features_r))
    b_r = np.ones((1, num_users_r))
    Y_r = np.ones((num_movies_r, num_users_r))
    R_r = np.ones((num_movies_r, num_users_r))

    # Evaluate cost function
    J = target(X_r, W_r, b_r, Y_r, R_r, 1);
    
    assert np.isclose(J, 103.5), f"Wrong value. Expected {103.5}, got {J}. Check the term without the regularization"
    
    num_users_r = 3
    num_movies_r = 4 
    num_features_r = 4
    
    #np.random.seed(247)
    X_r = np.array([[0.36618032, 0.9075415,  0.8310605,  0.08590986],
                     [0.62634721, 0.38234325, 0.85624346, 0.55183039],
                     [0.77458727, 0.35704147, 0.31003294, 0.20100006],
                     [0.34420469, 0.46103436, 0.88638208, 0.36175401]])#np.random.rand(num_movies_r, num_features_r)
    W_r = np.array([[0.04786854, 0.61504665, 0.06633146, 0.38298908], 
                    [0.16515965, 0.22320207, 0.89826005, 0.14373251], 
                    [0.1274051 , 0.22757303, 0.96865613, 0.70741111]])#np.random.rand(num_users_r, num_features_r)
    b_r = np.array([[0.14246472, 0.30110933, 0.56141144]])#np.random.rand(1, num_users_r)
    Y_r = np.array([[0.20651685, 0.60767914, 0.86344527], 
                    [0.82665019, 0.00944765, 0.4376798 ], 
                    [0.81623732, 0.26776794, 0.03757507], 
                    [0.37232161, 0.19890823, 0.13026598]])#np.random.rand(num_movies_r, num_users_r)
    R_r = np.array([[1, 0, 1], [1, 0, 0], [1, 0, 0], [0, 1, 0]])#(np.random.rand(num_movies_r, num_users_r) > 0.4) * 1

    # Evaluate cost function
    J = target(X_r, W_r, b_r, Y_r, R_r, 3);
    
    assert np.isclose(J, 13.621929978531858, atol=1e-8), f"Wrong value. Expected {13.621929978531858}, got {J}."
    
    print('\033[92mAll tests passed!')
    
def test_sq_dist(target):
    a1 = np.array([1.0, 2.0, 3.0]); b1 = np.array([1.0, 2.0, 3.0])
    c1 = target(a1, b1)
    a2 = np.array([1.1, 2.1, 3.1]); b2 = np.array([1.0, 2.0, 3.0])
    c2 = target(a2, b2)
    a3 = np.array([0, 1]);          b3 = np.array([1, 0])
    c3 = target(a3, b3)
    a4 = np.array([1, 1, 1, 1, 1]); b4 = np.array([0, 0, 0, 0, 0])
    c4 = target(a4, b4)
    
    assert np.isclose(c1, 0), f"Wrong value. Expected {0}, got {c1}"
    assert np.isclose(c2, 0.03), f"Wrong value. Expected {0.03}, got {c2}" 
    assert np.isclose(c3, 2), f"Wrong value. Expected {2}, got {c3}" 
    assert np.isclose(c4, 5), f"Wrong value. Expected {5}, got {c4}" 
    
    print('\033[92mAll tests passed!')

### ex 2        
def basic_sigmoid_test(target):
    x = 1
    expected_output = 0.7310585786300049
    test_cases = [
        {
            "name": "datatype_check",
            "input": [x],
            "expected": float,
            "error": "Datatype mismatch."
        },
        {
            "name": "equation_output_check",
            "input": [x],
            "expected": expected_output,
            "error": "Wrong output."
        }
    ]
    
    print(f"\nTest Case 1 (basic_sigmoid(x={x})):")
    test(test_cases, target)
    
    x = 0.5
    expected_output = 0.6224593312018546
    test_cases = [
        {
            "name": "datatype_check",
            "input": [x],
            "expected": float,
            "error": "Datatype mismatch."
        },
        {
            "name": "equation_output_check",
            "input": [x],
            "expected": expected_output,
            "error": "Wrong output."
        }
    ]
    
    print(f"\nTest Case 2 (basic_sigmoid(x={x})):")
    test(test_cases, target)

### ex 3    
def sigmoid_test(target):
    x = np.array([1, 2, 3])
    expected_output = np.array([0.73105858,
                                0.88079708,
                                0.95257413])
    test_cases = [
        {
            "name":"datatype_check",
            "input": [x],
            "expected": np.ndarray,
            "error":"Datatype mismatch."
        },
        {
            "name": "shape_check",
            "input": [x],
            "expected": expected_output,
            "error": "Wrong shape."
        },
        {
            "name": "equation_output_check",
            "input": [x],
            "expected": expected_output,
            "error": "Wrong output."
        }
    ]
    
    test(test_cases, target)
    
            
### ex 4        
def sigmoid_derivative_test(target):
    x = np.array([1, 2, 3])
    expected_output = np.array([0.19661193,
                                0.10499359,
                                0.04517666])
    test_cases = [
        {
            "name":"datatype_check",
            "input": [x],
            "expected": np.ndarray,
            "error":"The function should return a numpy array."
        },
        {
            "name": "shape_check",
            "input": [x],
            "expected": expected_output,
            "error": "Wrong shape."
        },
        {
            "name": "equation_output_check",
            "input": [x],
            "expected": expected_output,
            "error": "Wrong output."
        }
    ]
    
    test(test_cases, target)

    
### ex 5    
def image2vector_test(target):
    image = np.array([[[ 0.67826139,  0.29380381],
                      [ 0.90714982,  0.52835647],
                      [ 0.4215251 ,  0.45017551]],

                     [[ 0.92814219,  0.96677647],
                      [ 0.85304703,  0.52351845],
                      [ 0.19981397,  0.27417313]],

                     [[ 0.60659855,  0.00533165],
                      [ 0.10820313,  0.49978937],
                      [ 0.34144279,  0.94630077]]])
    
    expected_output = np.array([[ 0.67826139],
                                [ 0.29380381],
                                [ 0.90714982],
                                [ 0.52835647],
                                [ 0.4215251 ],
                                [ 0.45017551],
                                [ 0.92814219],
                                [ 0.96677647],
                                [ 0.85304703],
                                [ 0.52351845],
                                [ 0.19981397],
                                [ 0.27417313],
                                [ 0.60659855],
                                [ 0.00533165],
                                [ 0.10820313],
                                [ 0.49978937],
                                [ 0.34144279],
                                [ 0.94630077]])
    test_cases = [
        {
            "name":"datatype_check",
            "input": [image],
            "expected": np.ndarray,
            "error":"The function should return a numpy array."
        },
        {
            "name": "shape_check",
            "input": [image],
            "expected": expected_output,
            "error": "Wrong shape"
        },
        {
            "name": "equation_output_check",
            "input": [image],
            "expected": expected_output,
            "error": "Wrong output"
        } 
    ]
    
    test(test_cases, target)

    
### ex 6    
def normalizeRows_test(target):
    x = np.array([[0., 3., 4.],
                  [1., 6., 4.]])
    expected_output = np.array([[ 0., 0.6, 0.8 ],
                                [ 0.13736056, 0.82416338, 0.54944226]])
    
    test_cases = [
        {
            "name":"datatype_check",
            "input": [x],
            "expected": np.ndarray,
            "error":"The function should return a numpy array."
        },
        {
            "name": "shape_check",
            "input": [x],
            "expected": expected_output,
            "error": "Wrong shape"
        },
        {
            "name": "equation_output_check",
            "input": [x],
            "expected": expected_output,
            "error": "Wrong output"
        } 
    ]
    
    test(test_cases, target)       

    
### ex 7    
def softmax_test(target):
    x = np.array([[9, 2, 5, 0, 0],
                  [7, 5, 0, 0 ,0]])
    expected_output = np.array([[ 9.80897665e-01, 8.94462891e-04,
                                 1.79657674e-02, 1.21052389e-04,
                                 1.21052389e-04],
                                
                                [ 8.78679856e-01, 1.18916387e-01,
                                 8.01252314e-04, 8.01252314e-04,
                                 8.01252314e-04]])
    test_cases = [
        {
            "name":"datatype_check",
            "input": [x],
            "expected": np.ndarray,
            "error":"The function should return a numpy array."
        },
        {
            "name": "shape_check",
            "input": [x],
            "expected": expected_output,
            "error": "Wrong shape"
        },
        {
            "name": "equation_output_check",
            "input": [x],
            "expected": expected_output,
            "error": "Wrong output"
        } 
    ]
    
    test(test_cases, target)

    
### ex 8    
def L1_test(target):
    yhat = np.array([.9, 0.2, 0.1, .4, .9])
    y = np.array([1, 0, 0, 1, 1])
    expected_output = 1.1
    test_cases = [
        {
            "name":"datatype_check",
            "input": [yhat, y],
            "expected": float,
            "error":"The function should return a float."
        },
        {
            "name": "equation_output_check",
            "input": [yhat, y],
            "expected": expected_output,
            "error": "Wrong output"
        } 
    ]
    
    test(test_cases, target)


### ex 9    
def L2_test(target):
    yhat = np.array([.9, 0.2, 0.1, .4, .9])
    y = np.array([1, 0, 0, 1, 1])
    expected_output = 0.43
    
    test_cases = [
        {
            "name":"datatype_check",
            "input": [yhat, y],
            "expected": float,
            "error":"The function should return a float."
        },
        {
            "name": "equation_output_check",
            "input": [yhat, y],
            "expected": expected_output,
            "error": "Wrong output"
        } 
    ]
    
    test(test_cases, target)

