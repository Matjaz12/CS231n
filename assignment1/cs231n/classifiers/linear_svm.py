from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


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
    # note that grad.W (dW) should be of same shape as the W.
    dW = np.zeros(W.shape)  # initialize the gradient as zero
    
    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0

    # Iterate over all training samples
    for i in range(num_train):
        # Compute the scores for the i-th training sampe
        scores = X[i].dot(W) # (C x 1)
        correct_class_score = scores[y[i]]

        # Iterate over all classes
        num_times_margin_over_zero = 0
        for j in range(num_classes):
            if j == y[i]:
                continue
            
            # Compute the margin
            margin = (scores[j] - correct_class_score + 1)  # note delta = 1
            if margin > 0:
                loss += margin
                dW[:, j] += X[i].T
                num_times_margin_over_zero += 1

        # correct class gradient part
        dW[:, y[i]] += (-1) * num_times_margin_over_zero * X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    dW /= num_train
    dW = dW + 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    scores = X.dot(W)
    correct_class_scores = np.expand_dims(scores[np.arange(X.shape[0]), y], axis=1)
    margin = scores - correct_class_scores + 1
    margin[np.arange(X.shape[0]), y] = 0
    loss = np.sum(np.maximum(np.zeros_like(margin), margin), axis=1)
    loss = np.mean(loss, axis=0)
    loss += reg * np.sum(W * W)
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dloss = 1.0
    dW = 2 * reg * W
    temp = np.zeros_like(margin)
    temp[np.where(margin > 0)] = 1.0 / X.shape[0]
    dmargin = temp * dloss
    dcorrect_class_scores = np.sum(-1.0 * dmargin, axis=1, keepdims=True)
    temp1 = np.zeros_like(scores)
    temp1[np.arange(X.shape[0]), y] = 1.0
    dscores = 1.0 * dmargin + temp1 * dcorrect_class_scores
    dW += X.T @ dscores

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
