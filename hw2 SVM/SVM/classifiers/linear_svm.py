import numpy as np
from random import shuffle


def linear_svm_loss_vectorized(W, X, y, reg):
  """
  Linear SVM loss function, vectorized implementation.
  Inputs have dimension D and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, ) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the linear SVM         #
  # loss, storing the result in dW.                                            #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
 
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
