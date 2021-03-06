import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network with ReLU nonlinearity and
  softmax loss that uses a modular layer design. We assume an input dimension
  of D, a hidden dimension of H, and perform classification over C classes.

  The architecure should be affine - relu - affine - softmax.

  Note that this class does not implement gradient descent; instead, it
  will interact with a separate Solver object that is responsible for running
  optimization.

  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to numpy arrays.
  """

  def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
               weight_scale=1e-3, reg=0.0):
    """
    Initialize a new network.

    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dim: An integer giving the size of the hidden layer
    - num_classes: An integer giving the number of classes to classify
    - dropout: Scalar between 0 and 1 giving dropout strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
    self.params = {}
    self.reg = reg

    self.params['W1'] = np.random.normal(scale=weight_scale, size=(input_dim, hidden_dim))
    self.params['b1'] = np.zeros(hidden_dim)
    self.params['W2'] = np.random.normal(scale=weight_scale, size=(hidden_dim, num_classes))
    self.params['b2'] = np.zeros(num_classes)


  def loss(self, X, y=None):
    """
    Compute loss and gradient for a minibatch of data.

    Inputs:
    - X: Array of input data of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N, C) giving classification scores, where
      scores[i, c] is the classification score for X[i] and class c.

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """
    scores = None
    N = X.shape[0]
    W1 = self.params['W1']
    b1 = self.params['b1']
    W2 = self.params['W2']
    b2 = self.params['b2']

    ############################################################################
    # Implement the forward pass for the two-layer net, computing the    #
    # class scores for X and storing them in the scores variable.              #
    ############################################################################
    X1, cache1 = affine_relu_forward(X, W1, b1)
    scores, cache2 = affine_forward(X1, W2, b2)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If y is None then we are in test mode so just return scores
    if y is None:
      return scores

    loss, grads = 0, {}
    ############################################################################
    # Implement the backward pass for the two-layer net. Store the loss  #
    # in the loss variable and gradients in the grads dictionary. Compute data #
    # loss using softmax, and make sure that grads[k] holds the gradients for  #
    # self.params[k]. Don't forget to add L2 regularization!                   #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    loss, dx = softmax_loss(scores, y)
    w1_reg = 0.5 * self.reg * np.sum(W1* W1)
    w2_reg = 0.5 * self.reg * np.sum(W2 * W2)
    loss += (w1_reg + w2_reg)

    dx1, dw2, db2 = affine_backward(dx, cache2)
    dx, dw1, db1 = affine_relu_backward(dx1, cache1)

    dw1_reg = 0.5 * self.reg * (2 * W1)
    dw2_reg = 0.5 * self.reg * (2 * W2)

    grads['b2'] = db2
    grads['W2'] = dw2 + dw2_reg
    grads['b1'] = db1
    grads['W1'] = dw1 + dw1_reg
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    return loss, grads


class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be

  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.

  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None):
    """
    Initialize a new FullyConnectedNet.

    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}

    ############################################################################
    # TODO: Initialize the parameters of the network, storing all values in    #
    # the self.params dictionary. Store weights and biases for the first layer #
    # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
    # initialized from a normal distribution with standard deviation equal to  #
    # weight_scale and biases should be initialized to zero.                   #
    #                                                                          #
    # When using batch normalization, store scale and shift parameters for the #
    # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
    # beta2, etc. Scale parameters should be initialized to one and shift      #
    # parameters should be initialized to zero.                                #
    ############################################################################
    self.dims = [input_dim]
    self.dims += hidden_dims
    self.dims.append(num_classes)
    for i in xrange(len(self.dims) - 1):

      # I don't really understand why, but this initialization gives better convergence.
      # In particular, if I do not multiply by weight_scale, I get some infinite losses.
      w = (np.random.randn(self.dims[i], self.dims[i+1]) * weight_scale) / np.sqrt(2.0/(self.dims[i] + self.dims[i+1]))
      b = (np.random.randn(self.dims[i+1]) * weight_scale) / np.sqrt(2.0/(self.dims[i] + self.dims[i+1]))

      # w = np.random.normal(scale=weight_scale, size=(self.d ims[i], self.dims[i+1]))
      # b = np.zeros(self.dims[i+1])

      self.set_w_at(i, w)
      self.set_b_at(i, b)
      if use_batchnorm and i < self.num_layers - 1:
        self.set_param_at('gamma', i, np.ones(self.dims[i]))
        self.set_param_at('beta', i, np.zeros(self.dims[i]))

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed

    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)]

    # Cast all parameters to the correct datatype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)

    print("Batchnorm: {} dropout: {}".format(self.use_batchnorm, self.use_dropout))


  def w_name_at(self, i):
    return "W{}".format(i+1)

  def w_at(self, i):
    return self.params[self.w_name_at(i)]

  def set_w_at(self, i, val):
    self.params[self.w_name_at(i)] = val

  def b_name_at(self, i):
    return "b{}".format(i+1)

  def b_at(self, i):
    return self.params[self.b_name_at(i)]

  def param_name_at(self, name, i):
    return "{}{}".format(name, i+1)

  def param_at(self, name, i):
    return self.params[self.param_name_at(name, i)]

  def set_param_at(self, name, i, val):
    self.params[self.param_name_at(name, i)] = val

  def set_b_at(self, i, val):
    self.params[self.b_name_at(i)] = val

  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    N = X.shape[0]
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the fully-connected net, computing  #
    # the class scores for X and storing them in the scores variable.          #
    #                                                                          #
    # When using dropout, you'll need to pass self.dropout_param to each       #
    # dropout forward pass.                                                    #
    #                                                                          #
    # When using batch normalization, you'll need to pass self.bn_params[0] to #
    # the forward pass for the first batch normalization layer, pass           #
    # self.bn_params[1] to the forward pass for the second batch normalization #
    # layer, etc.                                                              #
    ############################################################################

    cache = None
    w_reg = 0
    scores = X
    caches = []
    dropout_caches = []
    for i in xrange(self.num_layers):
      W = self.w_at(i)
      b = self.b_at(i)
      # print("i:{} x.shape = {}, w.shape = {}, b.shape = {}".format(i, scores.shape, W.shape, b.shape))
      if i == self.num_layers - 1:
        # print("[{}] affine forward ".format(i))
        scores, cache = affine_forward(scores, W, b)
        caches.append(cache)
      else:
        if self.use_batchnorm:
          gamma = self.param_at("gamma", i)
          beta = self.param_at("beta", i)
          bn_params = self.bn_params[i]
          scores, cache = batchnorm_affine_relu_forward(scores, W, b, gamma, beta, bn_params)
        else:
          # print("[{}] affine relu forward".format(i))
          scores, cache = affine_relu_forward(scores, W, b)
        caches.append(cache)
        if self.use_dropout:
          # print("forward shape[{}] {}".format(i, scores.shape))
          scores, dropout_cache = dropout_forward(scores, self.dropout_param)
          dropout_caches.append(dropout_cache)

      w_reg += 0.5 * self.reg * np.sum(W * W)


    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If test mode return early
    if mode == 'test':
      return scores

    ############################################################################
    # TODO: Implement the backward pass for the fully-connected net. Store the #
    # loss in the loss variable and gradients in the grads dictionary. Compute #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    #                                                                          #
    # When using batch normalization, you don't need to regularize the scale   #
    # and shift parameters.                                                    #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################

    loss, grads = 0.0, {}

    loss, dx = softmax_loss(scores, y)
    loss += w_reg

    for i in reversed(xrange(self.num_layers)):
      cache = caches[i]
      dw, db = None, None
      if i == self.num_layers - 1:
        # print("[{}] affine backward".format(i))
        dx, dw, db = affine_backward(dx, cache)
      else:
        if self.use_dropout:
          dropout_param, mask = dropout_caches[i]
          # print("backward shape[{}] {}".format(i, mask.shape))
          dx = dropout_backward(dx, dropout_caches[i])
        if self.use_batchnorm:
          dx, dgamma, dbeta, dw, db = batchnorm_affine_relu_backward(dx, cache)
          grads[self.param_name_at("gamma", i)] = dgamma
          grads[self.param_name_at("beta", i)] = dbeta
        else:
          # print("[{}] affine relu backward".format(i))
          dx, dw, db = affine_relu_backward(dx, cache)

      grads[self.w_name_at(i)] = (dw) + (0.5 * self.reg * 2 * self.w_at(i))
      grads[self.b_name_at(i)] = (db) #+ (0.5 * self.reg * 2 * self.b_at(i))

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads

def batchnorm_affine_relu_forward(x, w, b, gamma, beta, bn_params):
  """
  Convenience layer that performs a batch normalization, followed by
  an affine transform, followed by a ReLU

  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer
  - gamma, beta: Scale and shift parameters for the batchnorm layer

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """

  bn_out, bn_cache = batchnorm_forward(x, gamma, beta, bn_params)
  out, cache = affine_relu_forward(bn_out, w, b)
  return out, (bn_cache, cache)

def batchnorm_affine_relu_backward(dout, cache):
  """
  Backward pass for the batchnorm-affine-relu convenience layer
  """
  bn_cache, ar_cache = cache
  dx, dw, db = affine_relu_backward(dout, ar_cache)
  dx, dgamma, dbeta = batchnorm_backward(dx, bn_cache)
  return dx, dgamma, dbeta, dw, db
