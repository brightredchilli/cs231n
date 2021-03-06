import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    # also can be obtained by multipling the dimensions because X = (N, d_1, ..., d_k)
    # ... like so D = np.prod(x.shape()[1:])
    D = w.shape[0] # but this is cleaner
    N = x.shape[0]
    x_new = x.reshape(N, D)
    out = x_new.dot(w) + b
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
        - x: Input data, of shape (N, d_1, ... d_k)
        - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    N, M = dout.shape
    D, _ = w.shape
    x_reshaped = x.reshape(N, D) # N, D

    # f = X*W + b (N,M)
    db = dout.sum(0) # in addition, the gradient just gets distributed.
    dw = x_reshaped.T.dot(dout)
    dx = dout.dot(w.T).reshape(*x.shape)

    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = x * (x>0)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache

    # only allow values which were positive in forward pass to be let through
    dx = dout * (x>0)

    #############################################################################
    # TODO: Implement the ReLU backward pass.                                   #
    #############################################################################
    pass
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the mean
    and variance of each feature, and these averages are used to normalize data
    at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7 implementation
    of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N = x.shape[0]
    D = gamma.size # gamma should just be one dimensional, so using size works
    x = x.reshape(N, D)

    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, {}
    mean, var = None, None
    if mode == 'train':
        #############################################################################
        # TODO: Implement the training-time forward pass for batch normalization.   #
        # Use minibatch statistics to compute the mean and variance, use these      #
        # statistics to normalize the incoming data, and scale and shift the        #
        # normalized data using gamma and beta.                                     #
        #                                                                           #
        # You should store the output in the variable out. Any intermediates that   #
        # you need for the backward pass should be stored in the cache variable.    #
        #                                                                           #
        # You should also use your computed sample mean and variance together with  #
        # the momentum variable to update the running mean and running variance,    #
        # storing your result in the running_mean and running_var variables.        #
        #############################################################################
        mean = x.mean(0)
        var = np.sum((x - mean)**2, 0) / N

        bn_param['running_mean'] = momentum * running_mean + (1 - momentum) * mean
        bn_param['running_var'] = momentum * running_var + (1 - momentum) * var

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
    elif mode == 'test':
        #############################################################################
        # TODO: Implement the test-time forward pass for batch normalization. Use   #
        # the running mean and variance to normalize the incoming data, then scale  #
        # and shift the normalized data using gamma and beta. Store the result in   #
        # the out variable.                                                         #
        #############################################################################
        mean = running_mean
        var = running_var
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    xmu = x - mean
    sqrtvar = np.sqrt(var + eps)
    x = xmu / sqrtvar
    out = x * gamma + beta

    if mode == 'train':
      cache['x'] = x # note that this x is different than before
      cache['xmu'] = xmu
      cache['gamma'] = gamma
      cache['beta'] = beta
      cache['sqrtvar'] = sqrtvar

    return out, cache

def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None

    #############################################################################
    # TODO: Implement the backward pass for batch normalization. Store the      #
    # results in the dx, dgamma, and dbeta variables.                           #
    #############################################################################
    # mean = x_o.mean(0)
    # var = np.sum((x_o - mean)**2, 0) / N
    # x1 = (x_o - mean) / np.sqrt(var + eps)
    # out = x1 * gamma + beta

    xmu = cache['xmu']
    x = cache['x'] # N x D
    N, D = x.shape
    gamma = cache['gamma']
    beta = cache['beta']
    sqrtvar = cache['sqrtvar']
    dx = np.zeros_like(x)

    dbeta = dout.sum(0)
    dgamma = (x * dout).sum(0)
    dx1 = dout*gamma

    isqrtvar = 1/sqrtvar
    x1 = xmu * isqrtvar

    dxmu = isqrtvar * dx1
    dsqrtvar = np.sum(dx1 * xmu, 0)
    dsqrtvar = dsqrtvar * -1/(sqrtvar**2)
    dvar = dsqrtvar * 1/(2 * sqrtvar)

    dvar /= N
    dvar = 2 * xmu * dvar
    dxmu += dvar

    dx += dxmu
    dx += -dxmu.sum(0) / N

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    #############################################################################
    # TODO: Implement the backward pass for batch normalization. Store the      #
    # results in the dx, dgamma, and dbeta variables.                           #
    #                                                                           #
    # After computing the gradient with respect to the centered inputs, you     #
    # should be able to compute gradients with respect to the inputs in a       #
    # single statement; our implementation fits on a single 80-character line.  #
    #############################################################################
    xmu = cache['xmu']
    x = cache['x'] # N x D
    N, D = x.shape
    gamma = cache['gamma']
    beta = cache['beta']
    sqrtvar = cache['sqrtvar']
    dx = np.zeros_like(x)

    dbeta = dout.sum(0)
    dgamma = (x * dout).sum(0)
    dx1 = dout*gamma

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
        - p: Dropout parameter. We drop each neuron output with probability p.
        - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
        - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not in
        real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
        mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = (np.random.rand(*x.shape) > p) / p

    if mode == 'train':
        ###########################################################################
        # TODO: Implement the training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                            #
        ###########################################################################
        out = x * mask
        ###########################################################################
        #                            END OF YOUR CODE                             #
        ###########################################################################
    elif mode == 'test':
        ###########################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.       #
        ###########################################################################
        out = x
        ###########################################################################
        #                            END OF YOUR CODE                             #
        ###########################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        ###########################################################################
        # TODO: Implement the training phase backward pass for inverted dropout.  #
        ###########################################################################
        dx = dout * mask
        ###########################################################################
        #                            END OF YOUR CODE                             #
        ###########################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and width
    W. We convolve each input with F different filters, where each filter spans
    all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
        - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
        - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
        H' = 1 + (H + 2 * pad - HH) / stride
        W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """

    out = None
    #############################################################################
    # TODO: Implement the convolutional forward pass.                           #
    # Hint: you can use the function np.pad for padding.                        #
    #############################################################################
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    S = conv_param["stride"]
    P = conv_param["pad"]
    H_new = (H - HH + 2 * P) / S + 1
    W_new = (W - WW + 2 * P) / S + 1
    # print("N: {}  C: {}  H: {}  W: {}".format(*x.shape))
    # print("F: {}  C: {}  HH: {}  WW: {}".format(*w.shape))
    # print("S: {}  P: {}".format(S, P))
    # print("H': {}  W': {}".format(H_new, W_new))

    # stride and pad settings should never result in non integer width and heights
    if W_new != int(W_new): raise AssertionError()
    if H_new != int(H_new): raise AssertionError()

    # We have a placeholder to store the new, padded x for backprop
    x_ = np.zeros((N, HH * WW * C, W_new * H_new))

    x = np.pad(x, [(0,0),(0,0),(P,P),(P,P)], 'constant', constant_values=0)
    w_ = w.reshape((F, -1)) # F(number of filters, or depth) x (HH * WW * C)filter area
    out = np.zeros([N, F, H_new, W_new])
    for i_n, x_cur in enumerate(x):
        # Build an input volume.
        x_new = np.zeros((HH * WW * C, W_new * H_new)) # receptive field x new area
        for (i, (row, col)) in enumerate(np.ndindex(H_new, W_new)):
            row_offset = row * S
            col_offset = col * S
            cur = x_cur[:, row_offset:row_offset + HH, col_offset:col_offset + WW]
            x_new[:, i] = cur.reshape(-1)
        result = w_.dot(x_new) + b[:, None] # b[:, None] force transposes 1d array
        x_[i_n] = x_new
        out[i_n] = result.reshape(F, H_new, W_new)


    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    cache = (x_, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    #############################################################################
    # TODO: Implement the convolutional backward pass.                          #
    #############################################################################
    # x_ is naming convention to indicate that it is x in the form with overlap
    # -ping receptive fields
    x_, w, b, conv_param = cache
    N, receptive_field, new_area = x_.shape
    F, C, HH, WW = w.shape
    N, _, H_n, W_n = dout.shape
    S = conv_param["stride"]
    P = conv_param["pad"]
    H = S * (H_n - 1) + HH - 2 * P # get height and width by reversing formula
    W = S * (W_n - 1) + WW - 2 * P

    # stride and pad settings should never result in non integer width and heights
    if H != int(H): raise AssertionError()
    if W != int(W): raise AssertionError()

    db = dout.sum(0).sum(1).sum(1) # sum over n, then over the ouput volume area

    # flatten weights and output so that each filter is a row
    dout = dout.reshape(N, F, H_n * W_n)
    w = w.reshape(F, C * HH * WW)

    # define all of the arrays that will hold data
    dx_ = np.zeros((N, HH * WW * C, W_n * H_n))
    dw = np.zeros((F, C * HH * WW))

    # for each filter
    for i, dout_cur in enumerate(dout):
        # dx_cur = w'(receptive field x F) x dout_cur(F x new area)
        dx_cur = w.T.dot(dout_cur)
        dx_[i] = dx_cur # this is dx_ because it is padded.

        # dw_cur = dout_cur(F x new area) x X'(new area x receptive field)
        dw_cur = dout_cur.dot(x_[i].T)
        dw += dw_cur

    #padded H and W
    HP = H + 2 * P
    WP = W + 2 * P

    # reshape dx_ back to a padded dx
    dx = np.zeros((N, C, HP, WP))

    # the index i here refers to the receptive field of 'pixel' i in the output
    # volume
    for (i, (row, col)) in enumerate(np.ndindex(H_n, W_n)):
        row_offset = row * S
        col_offset = col * S
        cur = dx_[:, :, i].reshape(N, C, HH, WW)
        dx[:, :, row_offset:row_offset + HH, col_offset:col_offset + WW] += cur

    dx = dx[:, :, P:-P, P:-P] # unpad the array
    dw = dw.reshape(F, C, HH, WW)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
        - 'pool_height': The height of each pooling region
        - 'pool_width': The width of each pooling region
        - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None

    #############################################################################
    # TODO: Implement the max pooling forward pass                              #
    #############################################################################
    N, C, H, W = x.shape
    F_h = pool_param["pool_height"]
    F_w = pool_param["pool_width"]
    S = pool_param["stride"]

    H_n = (H - F_h) / S + 1
    W_n = (W - F_w) / S + 1

    out = np.zeros((N, C, H_n, W_n))

    # mask = np.zeros(x.shape)
    # print("x = {}".format(x.shape))
    # print("N = {}, C = {}, H = {}, W = {}".format(N, C, H, W))
    # print("F_h = {}, F_w = {}, S = {}".format(F_h, F_w, S))
    # print("H_n = {}, W_n = {}".format(H_n, W_n))
    for (i, (row, col)) in enumerate(np.ndindex(H_n, W_n)):
        y_offset = row * S
        x_offset = col * S
        # get the maximum in the filter area
        x_cur = x[:, :, y_offset:y_offset + F_h, x_offset:x_offset + F_w]
        # mask[:, :, y_offset:y_offset + F_h, x_offset:x_offset + F_w] = x_cur == x_cur.max(axis=(2,3), keepdims=True)
        out[:, :, row, col] = x_cur.max(axis=(2,3))
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    #############################################################################
    # TODO: Implement the max pooling backward pass                             #
    #############################################################################
    x, pool_param = cache
    dx = np.zeros(x.shape)
    N, C, H_n, W_n = dout.shape
    F_h = pool_param["pool_height"]
    F_w = pool_param["pool_width"]
    S = pool_param["stride"]
    for (i, (row, col)) in enumerate(np.ndindex(H_n, W_n)):
        y_offset = row * S
        x_offset = col * S
        # get the maximum in the filter area
        x_cur = x[:, :, y_offset:y_offset + F_h, x_offset:x_offset + F_w]

        # get a mask - this is to get an boolean index that we multiply with
        # the gradient from below to get the gradient in each of the
        # 'max' positions
        mask = x_cur == x_cur.max(axis=(2,3), keepdims=True)

        # This dx_ has the gradient in the proper position in the filter area
        # we are currently looking at. dout[:, :, row, col] produces a (N, C)
        # array. We have to reshape so that in can be broadcast with the mask.
        dx_ = mask * dout[:, :, row, col].reshape(N, C, 1, 1)

        dx[:, :, y_offset:y_offset + F_h, x_offset:x_offset + F_w] += dx_

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return dx

def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
        - mode: 'train' or 'test'; required
        - eps: Constant for numeric stability
        - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
        - running_mean: Array of shape (D,) giving running mean of features
        - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    #############################################################################
    # Implement the forward pass for spatial batch normalization.               #
    #                                                                           #
    # HINT: You can implement spatial batch normalization using the vanilla     #
    # version of batch normalization defined above. Your implementation should  #
    # be very short; ours is less than five lines.                              #
    #############################################################################
    N, C, H, W = x.shape
    tmp = x.transpose(0,2,3,1).reshape(-1, C)
    out, cache = batchnorm_forward(tmp, gamma, beta, bn_param)
    out = out.reshape(N, H, W, C).transpose(0,3,1,2)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None


    #############################################################################
    # TODO: Implement the backward pass for spatial batch normalization.        #
    #                                                                           #
    # HINT: You can implement spatial batch normalization using the vanilla     #
    # version of batch normalization defined above. Your implementation should  #
    # be very short; ours is less than five lines.                              #
    #############################################################################

    N, C, H, W = dout.shape
    dout = dout.transpose(0,2,3,1).reshape(-1, C)
    dx, dgamma, dbeta = batchnorm_backward(dout, cache)
    dx = dx.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return dx, dgamma, dbeta

def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    N = x.shape[0]
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
