import struct
import numpy as np
import gzip
try:
    from simple_ml_ext import *
except:
    pass


def add(x, y):
    """ A trivial 'add' function you should implement to get used to the
    autograder and submission system.  The solution to this problem is in the
    the homework notebook.

    Args:
        x (Python number or numpy array)
        y (Python number or numpy array)

    Return:
        Sum of x + y
    """
    ### BEGIN YOUR CODE
    return x + y
    ### END YOUR CODE


def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded 
                data.  The dimensionality of the data should be 
                (num_examples x input_dim) where 'input_dim' is the full 
                dimension of the data, e.g., since MNIST images are 28x28, it 
                will be 784.  Values should be of type np.float32, and the data 
                should be normalized to have a minimum value of 0.0 and a 
                maximum value of 1.0 (i.e., scale original values of 0 to 0.0 
                and 255 to 1.0).

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE
    # read image file
    with gzip.open(image_filename, 'rb') as fi:
        image_content = fi.read()

    image_magic = struct.unpack('>I', image_content[:4])[0]
    if image_magic != 2051 :
        print("read file format error!")

    image_num = struct.unpack('>I', image_content[4:8])[0]
    image_row = struct.unpack('>I', image_content[8:12])[0]
    image_col = struct.unpack('>I', image_content[12:16])[0]

    """ bad efficiency form
    X = np.zeros((image_num, image_row*image_col), dtype=np.float32)
    # get X from image_content
    for i in range(image_num):
        for j in range(image_row * image_col):
            grey_num = np.uint8(image_content[16 + image_row*image_col*i + j])
            X[i][j] =  np.float32(float(grey_num)/255)
    """

    ### vectorizaton form, very quick
    # Reshape image_content into a 3D array (image_num, image_row, image_col)
    image_content_reshaped = np.frombuffer(image_content, dtype=np.uint8, offset=16).reshape(image_num, image_row, image_col)
    # Convert uint8 values to float32 and normalize
    X = image_content_reshaped.astype(np.float32) / 255.0
    # Reshape X into a 2D array (image_num, image_row * image_col)
    X = X.reshape(image_num, -1)

    # --------------------------------------------------------------------------------------#

    # read label file
    with gzip.open(label_filename, 'rb') as fl:
        label_content = fl.read()

    label_magic = struct.unpack('>I', label_content[:4])[0]
    if label_magic != 2049 :
        print("read file format error!")

    label_num = struct.unpack('>I', label_content[4:8])[0]
    y = np.zeros(label_num, dtype=np.uint8)

    # get y from label_content
    for i in range (label_num):
        label = label_content[8 + i]
        y[i] = np.uint8(label)
    
    # --------------------------------------------------------------------------------------#

    return (X, y)
    ### END YOUR CODE


def softmax_loss(Z, y):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (np.ndarray[np.float32]): 2D numpy array of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (np.ndarray[np.uint8]): 1D numpy array of shape (batch_size, )
            containing the true label of each example.

    Returns:
        Average softmax loss over the sample.
    """
    ### BEGIN YOUR CODE
    Z_y = np.zeros(len(y), dtype=np.float32)
    for i in range(len(y)) :
        Z_y[i] = Z[i][y[i]]

    avg_le = 0.0
    for i in range(len(y)) :
        Z_exp_i = np.exp(Z[i]) 
        le_i = np.log(np.sum(Z_exp_i)) - Z_y[i]
        avg_le += le_i

    return avg_le / len(y)
    ### END YOUR CODE

## defined by mingxi
def print_shape(array, array_name):
    """ 
       print a 2D array's shape
    """ 
    rows, cols = array.shape
    print(f"array:{array_name}, {rows}x{cols}")

## defined by mingxi
def softmax(Matrix):
    Matrix_exp = np.exp(Matrix)
    Matrix_exp_sum = np.sum(Matrix_exp, axis=1)
    Matrix_exp_sum_expand = np.tile(Matrix_exp_sum[:, np.newaxis], (1, len(Matrix[0])))

    return Matrix_exp / Matrix_exp_sum_expand 

## defined by mingxi
def relu(Matrix):
    return np.maximum(0, Matrix)

## defined by mingxi
def relu_mask(Matrix):
    return (Matrix > 0).astype(float)

def softmax_regression_epoch(X, y, theta, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for softmax regression on the data, using
    the step size lr and specified batch size.  This function should modify the
    theta matrix in place, and you should iterate through batches in X _without_
    randomizing the order.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        theta (np.ndarrray[np.float32]): 2D array of softmax regression
            parameters, of shape (input_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    #print(f"X shape ({len(X)},{len(X[0])}); y shape {len(y)}; theta shape ({len(theta)},{len(theta[0])}); batch shape {batch}")
    iter_num = int(len(X) / batch)

    for i in range(iter_num) :
        X_batch = X[i*batch:(i+1)*batch, :]
        Z_batch = softmax(X_batch @ theta)

        I_y = np.zeros((batch, len(theta[0])), dtype=np.float32)
        for j in range(batch) :
            I_y[j][y[batch*i+j]] = 1.0

        #print_shape(X_batch.T, "X_batch_T")
        #print_shape(Z_batch, "Z_batch")
        #print_shape(I_y, "I_y")

        #theta = theta - lr / batch * X_batch.T @ (Z_batch - I_y) # error use, here left theta is a local variable 
        theta -= lr / batch * X_batch.T @ (Z_batch - I_y)
    #print(theta)
    ### END YOUR CODE


def nn_epoch(X, y, W1, W2, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).  It should modify the
    W1 and W2 matrices in place.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (np.ndarray[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (np.ndarray[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    iter_num = int(len(X) / batch)

    for i in range(iter_num):
        X_batch = X[i*batch:(i+1)*batch, :]
        X_batch_W1 = X_batch @ W1 
        Z1 = relu(X_batch_W1)
        #print_shape(X_batch, "X_batch")
        #print_shape(W1, "W1")
        #print_shape(W2, "W2")
        #print_shape(Z1, "Z1")

        I_y = np.zeros((batch, len(W2[0])), dtype=np.float32)
        for j in range(batch) :
            I_y[j][y[batch*i+j]] = 1.0

        #print_shape(I_y, "I_y")
        #print_shape(Z1@W2, "Z1_W2")
        G2 = softmax(Z1 @ W2) - I_y 
        G1 = relu_mask(Z1) * (G2 @ W2.T)

        W1 -= lr / batch * (X_batch.T @ G1) 
        W2 -= lr / batch * (Z1.T @ G2) 
    ### END YOUR CODE



### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h,y):
    """ Helper funciton to compute both loss and error"""
    return softmax_loss(h,y), np.mean(h.argmax(axis=1) != y)


def train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.5, batch=100,
                  cpp=False):
    """ Example function to fully train a softmax regression classifier """
    theta = np.zeros((X_tr.shape[1], y_tr.max()+1), dtype=np.float32)
    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        if not cpp:
            softmax_regression_epoch(X_tr, y_tr, theta, lr=lr, batch=batch)
        else:
            #print("cpp version")
            softmax_regression_epoch_cpp(X_tr, y_tr, theta, lr=lr, batch=batch)
        train_loss, train_err = loss_err(X_tr @ theta, y_tr)
        test_loss, test_err = loss_err(X_te @ theta, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))


def train_nn(X_tr, y_tr, X_te, y_te, hidden_dim = 500,
             epochs=10, lr=0.5, batch=100):
    """ Example function to train two layer neural network """
    n, k = X_tr.shape[1], y_tr.max() + 1
    np.random.seed(0)
    W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
    W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)

    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        nn_epoch(X_tr, y_tr, W1, W2, lr=lr, batch=batch)
        train_loss, train_err = loss_err(np.maximum(X_tr@W1,0)@W2, y_tr)
        test_loss, test_err = loss_err(np.maximum(X_te@W1,0)@W2, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))



if __name__ == "__main__":
    X_tr, y_tr = parse_mnist("data/train-images-idx3-ubyte.gz",
                             "data/train-labels-idx1-ubyte.gz")
    X_te, y_te = parse_mnist("data/t10k-images-idx3-ubyte.gz",
                             "data/t10k-labels-idx1-ubyte.gz")

    print("Training softmax regression")
    train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr = 0.1)

    print("\nTraining two layer neural network w/ 100 hidden units")
    train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=100, epochs=20, lr = 0.2)
