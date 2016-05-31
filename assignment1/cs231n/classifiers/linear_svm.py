import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]

    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:,j] += X[i]
        dW[:,y[i]] -= X[i]


  loss /= num_train
  dW /= num_train
  dW += 0.5 * W # gradient due to regularization

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  return loss, dW

def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
    Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  """
  num_train = X.shape[0] # N, number of sampless
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  scores = X.dot(W) # scores (N, C)
  correct_scores = scores[np.arange(scores.shape[0]), y] # N array
  correct_scores = correct_scores.reshape(correct_scores.size, 1) # (N,1) array of correct scores, so they can be broadcasted
  margins = scores - correct_scores + 1 # still the same size as scores, (N, C)
  margins[np.arange(scores.shape[0]), y] = 0 # zero out correct classes
  loss = margins[margins>0].sum()

  num_scores_over_margin = (margins>0).sum(1) # sum over the columns, N array
  gradients = np.ones(margins.shape)*(margins>0)
  gradients[np.arange(scores.shape[0]), y] = -num_scores_over_margin
  dW = X.T.dot(gradients) # (D, C)

  loss /= num_train
  dW /= num_train
  dW += 0.5 * W # gradient due to regularization
  loss += 0.5 * reg * np.sum(W * W)

  return loss, dW
