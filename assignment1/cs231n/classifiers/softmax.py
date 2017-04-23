import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """

    # reimplemented sort of for fun
    N, D = X.shape
    pred = X.dot(W)
    pred -= np.max(pred, axis=1, keepdims=True)
    a = np.exp(pred)
    b = a[np.arange(N),y] / a.sum(1)

    c = -np.log(b)
    loss = np.sum(c) / N

    loss += 0.5 * reg * np.sum(W * W)

    return loss, 0
    #return softmax_loss_vectorized(W, X, y, reg)



def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    N = X.shape[0]
    D,C  = W.shape
    loss = 0.0
    dW = np.zeros_like(W)

    if y is None:
        y = np.zeros(N).astype(int)

    y_pred = X.dot(W)
    y_pred -= np.max(y_pred, 1).reshape(N,1)
    fyi = y_pred[np.arange(N), y] # N array

    j = y_pred
    ej = np.exp(j)
    jej = np.sum(ej, 1)
    efyi = np.exp(fyi)
    invden = 1 / jej
    a = efyi * invden
    b = np.log(a)
    f = -1 * b

    dfdb = -1
    dbda = 1 / a
    dadefyi = invden
    dadinvden = efyi
    dinvdendjej = -1 / (jej * jej)
    djejdej = 1
    dejdj = ej
    djdW = X
    djdX = W
    defyidfyi = efyi

    dfdfyi = dfdb * dbda * dadefyi * defyidfyi
    dfyidW = dfdfyi.reshape(N,1) * X # This has the same shape as X: N,D
    # first get an N,C indexing matrix
    index = np.zeros((N, C))
    index[np.arange(N), y] = 1 # This is N,C
    dW += dfyidW.T.dot(index) # distributes Xs to the proper Ds of the proper Cs

    dfdj = (dfdb * dbda * dadinvden * dinvdendjej * djejdej).reshape(N,1) * dejdj # N,C matrix
    dfdW = djdW.T.dot(dfdj)
    dW +=  dfdW

    loss = np.sum(f) / N
    loss += 0.5 * reg * np.sum(W * W)
    dW /= N

    return loss, dW
