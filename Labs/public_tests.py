import numpy as np
import math

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