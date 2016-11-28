import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *

class ParamHolder:
  """
  A simple class to make writing weight variables less ugly.
  """

  def __init__(self, param_name, params):
    """
    Inputs:
    - param_name: The prefix for the param. For example, 'W' causes params
                  to be written as W1, W2, etc.
    - params: The actual param dictionary we will be writing to.
    """
    self.param_name = param_name
    self.params = params

  def get_key(self, index):
    return "{}{}".format(self.param_name, index + 1)

  def __getitem__(self, index):
    return self.params[self.get_key(index)]

  def __setitem__(self, index, value):
    self.params[self.get_key(index)] = value


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:

  conv - relu - 2x2 max pool - affine - relu - affine - softmax

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """

  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.

    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype

    stride = 1
    pad = (filter_size - 1) / 2
    pool_size = 2
    pool_stride = 2

    # pass conv_param to the forward pass for the convolutional layer
    self.conv_param = {'stride': stride, 'pad': pad}

    # pass pool_param to the forward pass for the max-pooling layer
    self.pool_param = { 'pool_height': pool_size,
                        'pool_width': pool_size,
                        'stride': pool_stride }
    W1 = (input_dim[1] - filter_size + 2 * pad) / stride + 1
    Wp = (W1 - pool_size) / pool_stride + 1 # Wp = size after pooling layer

    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    self.W = ParamHolder('W', self.params)
    self.b = ParamHolder('b', self.params)

    self.W[0] = np.random.normal(scale=weight_scale,
                                 size=(num_filters,
                                       input_dim[0],
                                       filter_size,
                                       filter_size))
    self.b[0] = np.zeros(num_filters)

    self.W[1] = np.random.normal(scale=weight_scale,
                                 size=(num_filters * Wp * Wp, hidden_dim))
    self.b[1] = np.zeros(hidden_dim)

    self.W[2] = np.random.normal(scale=weight_scale,
                                 size=(hidden_dim, num_classes))
    self.b[2] = np.zeros(num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.

    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################

    out, cache1 = conv_relu_pool_forward(X, W1, b1, self.conv_param, self.pool_param)
    out, cache2 = affine_relu_forward(out, W2, b2)
    scores, cache3 = affine_forward(out, W3, b3)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    if y is None:
      return scores

    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    dW, db = ParamHolder("W", grads), ParamHolder("b", grads)

    loss, dout3 = softmax_loss(scores, y)

    w1_reg = 0.5 * self.reg * np.sum(W1 * W1)
    w2_reg = 0.5 * self.reg * np.sum(W2 * W2)
    w3_reg = 0.5 * self.reg * np.sum(W3 * W3)

    loss += w1_reg + w2_reg + w3_reg

    dout2, dw3, db3 = affine_backward(dout3, cache3)
    dout1, dw2, db2 = affine_relu_backward(dout2, cache2)
    dx, dw1, db1 = conv_relu_pool_backward(dout1, cache1)


    dw1_reg = 0.5 * self.reg * (2 * W1)
    dw2_reg = 0.5 * self.reg * (2 * W2)
    dw3_reg = 0.5 * self.reg * (2 * W3)

    dW[0] = dw1 + dw1_reg
    dW[1] = dw2 + dw2_reg
    dW[2] = dw3 + dw3_reg
    db[0] = db1
    db[1] = db2
    db[2] = db3

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


pass
