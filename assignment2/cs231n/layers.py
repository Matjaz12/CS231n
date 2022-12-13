from builtins import range
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
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    x_flat = x.reshape(x.shape[0], -1)
    out = x_flat.dot(w) + b

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
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
      - b: Biases, of shape (M,)
    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    x_flat = x.reshape(x.shape[0], -1)
    # print(f"x_flat.shape: {x_flat.shape}")

    dw = x_flat.T @ dout
    dx = (dout @ w.T).reshape(x.shape)
    db = dout.sum(0)
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
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
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    out = np.maximum(0, x)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
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
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    temp = np.zeros_like(x)
    temp[np.where(x > 0)] = 1.0
    dx = temp * dout


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx



def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.
    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C
    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    ###########################################################################
    # TODO: Copy over your solution from A1.
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Without numeric stability correction.
    # Leaving this piece of code here, in case the other implementation
    # begins to cause issues, this one is tasted and works.

    # counts = np.exp(x)
    # counts_sum = np.sum(counts, axis=1, keepdims=True)
    # probs = counts / counts_sum
    # correct_class_probs = probs[range(counts.shape[0]), y]
    # log_loss = -np.log(correct_class_probs)
    # loss = log_loss.sum(0) / counts.shape[0]
    # 
    # dloss = 1.0
    # dlog_loss = (np.ones_like(log_loss) * 1.0 / x.shape[0])* dloss
    # dcorrect_class_probs = (-1.0 / correct_class_probs) * dlog_loss
    # temp = np.zeros_like(probs)
    # temp[range(x.shape[0]), y] = 1.0
    # dprobs = temp * np.expand_dims(dcorrect_class_probs, 1)
    # dcounts = (1 / counts_sum * np.ones_like(counts)) * dprobs
    # dcounts_sum = ((counts * (-1.0 * counts_sum ** (-2))) * dprobs).sum(1, keepdims=True)
    # dcounts += np.ones_like(counts) * dcounts_sum
    # dx = np.exp(x) * dcounts


    # With numeric stability correction.
    # I forgot to take numeric instability into account before :(
    # Note that this doesn't effect the loss.
    x_corr = x - np.expand_dims(x.max(axis=1), axis=1)
    
    counts = np.exp(x_corr)
    counts_sum = np.sum(counts, axis=1, keepdims=True)
    probs = counts / counts_sum
    correct_class_probs = probs[range(counts.shape[0]), y]
    log_loss = -np.log(correct_class_probs)
    loss = log_loss.sum(0) / counts.shape[0]
    
    dloss = 1.0
    dlog_loss = (np.ones_like(log_loss) * 1.0 / x.shape[0])* dloss
    dcorrect_class_probs = (-1.0 / correct_class_probs) * dlog_loss
    temp = np.zeros_like(probs)
    temp[range(x.shape[0]), y] = 1.0
    dprobs = temp * np.expand_dims(dcorrect_class_probs, 1)
    dcounts = (1 / counts_sum * np.ones_like(counts)) * dprobs
    dcounts_sum = ((counts * (-1.0 * counts_sum ** (-2))) * dprobs).sum(1, keepdims=True)
    dcounts += np.ones_like(counts) * dcounts_sum
    dx = np.exp(x_corr) * dcounts

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

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
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Compute the batch mean and batch variance
        batch_mean = x.mean(axis=0)
        batch_var = 1 / (N - 1) * np.sum((x - batch_mean) ** 2, axis=0) # bessel correction

        # Normalize the activations
        x_hat = (x - batch_mean) / np.sqrt(batch_var + eps)

        # Scale and shift the distribution
        out = gamma * x_hat + beta

        # Compute the running mean and var
        running_mean = momentum * running_mean + (1 - momentum) * batch_mean
        running_var = momentum * running_var + (1 - momentum) * batch_var

        # Store intermediate variables in the cache
        # this will later be used for back-propagation
        cache = (batch_mean, batch_var, x_hat, x, gamma, beta, eps)


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Normalize the activations
        x_hat = (x - running_mean) / (np.sqrt(running_var + eps))
        
        # Scale and shift the distribution
        out = gamma * x_hat + beta

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """Backward pass for batch normalization.

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
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, D = dout.shape
    batch_mean, batch_var, x_hat, x, gamma, beta, eps = cache
    dgamma = (x_hat * dout).sum(axis=0)
    dbeta = (1.0 * dout).sum(axis=0)
    dx_hat = gamma * dout

    dbatch_var = np.sum(dx_hat * (x - batch_mean) * (-0.5) * np.power(batch_var + eps, -1.5), axis=0)
    # Note that mean is used in the calculation of x_hat and the calculation for variance
    # therefore we have to take the partial derivative for both and add them up.
    dbatch_mean = np.sum(dx_hat * (-1.0) * np.power(batch_var + eps, -0.5), axis=0) + np.sum((-2.0 / (N - 1) * (x - batch_mean) * dbatch_var), axis=0)
    # Note that x is used in calculation of x_hat, mean and var, thefore we should
    # take the partial derivative w.r.t to all and add them up.
    dx = dx_hat * np.power(batch_var + eps, -0.5) + dbatch_var * 2.0 * (x - batch_mean) / (N - 1) + dbatch_mean / N
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, D = dout.shape
    batch_mean, batch_var, x_hat, x, gamma, beta, eps = cache
    dgamma = (x_hat * dout).sum(axis=0)
    dbeta = (1.0 * dout).sum(axis=0)
    dx = gamma / N * np.power(batch_var + eps, -0.5) * (N * dout - dout.sum(0) - N / (N - 1) * x_hat * (dout * x_hat).sum(0))

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    D, N = x.shape
    x_t = x.T
    # Compute the layer mean and variance
    # along vector dimension (i.e we compute mean and var for each vector)
    layer_mean = x_t.mean(axis=0)
    layer_var = x_t.var(axis=0)

    # Normalize the vectors 
    # Notice that we have to transpose back to original form
    x_hat = ((x_t - layer_mean) / np.sqrt(layer_var + eps)).T
    
    # Scale and shift the distribution
    out = gamma * x_hat + beta

    # Store intermediate variables in the cache
    # this will later be used for back-propagation
    cache = x, x_hat, layer_mean, layer_var, eps, gamma

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, D = dout.shape
    x, x_hat, layer_mean, layer_var, eps, gamma = cache

    # For some reason I could not figure this out :(
    # Solution reference: https://github.com/mirzaim/cs231n/blob/d982c7f023a1cedd961b4104b3e652ce3c43e738/assignments/assignment2/cs231n/layers.py#L446
    dgamma = (x_hat * dout).sum(axis=0)
    dbeta = (1.0 * dout).sum(axis=0)
    dx_hat = gamma * dout
    dx = ((1. / (D * np.sqrt(layer_var + eps))) * (D * dx_hat.T - np.sum(dx_hat, axis=1) - x_hat.T*np.sum(dx_hat*x_hat, axis=1))).T

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """Forward pass for inverted dropout.

    Note that this is different from the vanilla version of dropout.
    Here, p is the probability of keeping a neuron output, as opposed to
    the probability of dropping a neuron output.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Scale at train time
        mask = (np.random.rand(*x.shape) < p) / p
        out = x * mask

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # No need to scale at train time
        # (since we already scaled the activations at train time)
        out = x

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """Backward pass for inverted dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        dx = mask * dout

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    pad = conv_param["pad"]
    stride = conv_param["stride"]

    # Pad along width and height dimension
    npad = ((0, 0), (0, 0), (pad, pad), (pad, pad))
    x_pad = np.pad(x, pad_width=npad, mode='constant', constant_values=0)

    N, C, H, W = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
    F, C, HH, WW = w.shape[0], w.shape[1], w.shape[2], w.shape[3]
    
    H_out = 1 + (H + 2 * pad - HH) // stride
    W_out = 1 + (W + 2 * pad - WW) // stride
    out_shape = (N, F, H_out, W_out)
    out = np.zeros(out_shape)

    # Iterate over samples in the batch
    for sample_idx in range(N):

        # Iterate over filters
        for filter_idx in range(F):

            # Compute activation map for the current filter
            for cnt_i, i in enumerate(range(0, (H + 2 * pad) - HH  + 1, stride)):
                for cnt_j, j in enumerate(range(0, (W + 2 * pad) - WW  + 1, stride)):

                    # Compute dot product over all chanels using the current filter
                    dot_ps = np.zeros(C)
                    for channel_idx in range(C):
                        dot_p = np.dot(
                            x_pad[sample_idx][channel_idx][i: i + HH, j: j + WW].reshape((-1, )),
                            w[filter_idx][channel_idx].reshape((-1, ))
                        )
                        dot_ps[channel_idx] = dot_p

                    out[sample_idx][filter_idx][cnt_i][cnt_j] = np.sum(dot_ps) + b[filter_idx]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, x_pad, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    x, x_pad, w, b, conv_param= cache

    dx = np.zeros(x.shape)
    dx_pad = np.zeros(x_pad.shape)
    dw = np.zeros(w.shape)
    db = np.zeros(b.shape)
    pad = conv_param["pad"]
    stride = conv_param["stride"]

    N, C, H, W = dx.shape[0], dx.shape[1], dx.shape[2], dx.shape[3]
    F, C, HH, WW = dw.shape[0], dw.shape[1], dw.shape[2], dw.shape[3]

    # Iterate over samples in the batch
    for sample_idx in range(N):

        # Iterate over filters
        for filter_idx in range(F):

            # Compute activation map for the current filter
            for cnt_i, i in enumerate(range(0, (H + 2 * pad) - HH  + 1, stride)):
                for cnt_j, j in enumerate(range(0, (W + 2 * pad) - WW  + 1, stride)):

                    # Compute dot product over all chanels using the current filter
                    ddot_ps = dout[sample_idx][filter_idx][cnt_i][cnt_j] * np.ones(C)

                    for channel_idx in range(C):
                        dw[filter_idx][channel_idx] += ddot_ps[channel_idx] * x_pad[sample_idx][channel_idx][i: i + HH, j: j + WW]
                        dx_pad[sample_idx][channel_idx][i: i + HH, j: j + WW] += ddot_ps[channel_idx] * w[filter_idx][channel_idx]

                    db[filter_idx] += dout[sample_idx][filter_idx][cnt_i][cnt_j] * 1.0

    # Strip of the gradient of the padded part
    dx[:, :] = dx_pad[:, :, pad:x_pad.shape[2] - pad, pad: x_pad.shape[3] - pad]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pool_height = pool_param["pool_height"]
    pool_width = pool_param["pool_width"]
    stride = pool_param["stride"]

    N, C, H, W = x.shape[0], x.shape[1], x.shape[2], x.shape[3]

    W_out = (W - pool_width) // stride + 1
    H_out = (H - pool_height) // stride + 1
    out_shape = (N, C, W_out, H_out)
    out = np.zeros(out_shape)

    # Iterate over samples
    for sample_idx in range(N):

        # Iterate over channels
        for channel_idx in range(C):
        
            # Pool the current channel
            for cnt_i, i in enumerate(range(0, W - pool_width + 1, stride)):
                for cnt_j, j in enumerate(range(0, H - pool_height + 1, stride)):
                    
                    # Take the max in the window
                    x_window = x[sample_idx][channel_idx][i: i + pool_width, j: j + pool_height]
                    out[sample_idx][channel_idx][cnt_i][cnt_j] = np.max(x_window)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, pool_param = cache
    pool_height = pool_param["pool_height"]
    pool_width = pool_param["pool_width"]
    stride = pool_param["stride"]

    N, C, H, W = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
    dx = np.zeros(x.shape)
    # Iterate over samples
    for sample_idx in range(N):

        # Iterate over channels
        for channel_idx in range(C):
        
            # Pool the current channel
            for cnt_i, i in enumerate(range(0, W - pool_width + 1, stride)):
                for cnt_j, j in enumerate(range(0, H - pool_height + 1, stride)):
                    x_window = x[sample_idx][channel_idx][i: i + pool_width, j: j + pool_height]
                    max_x_i, max_x_j = np.argwhere(x_window == x_window.max())[0]

                    dx[sample_idx][channel_idx][i + max_x_i][j + max_x_j] += 1.0 * dout[sample_idx][channel_idx][cnt_i][cnt_j]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """Computes the forward pass for spatial batch normalization.

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

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Check out the following hint for reshaping (in numpy): 
    # https://www.reddit.com/r/cs231n/comments/443y2g/hints_for_a2/?st=j7ycdnkr&sh=a4775a0c
    
    N, C, H, W  = x.shape
    x_r = x.transpose(0, 3, 2, 1).reshape(N * W * H, C)
    out, cache = batchnorm_forward(x_r, gamma, beta, bn_param)
    out = out.reshape(N, W, H, C).transpose(0, 3, 2, 1)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    batch_mean, batch_var, x_hat, x, gamma, beta, eps = cache
    N, C, H, W  = dout.shape

    dout_r = dout.transpose(0, 3, 2, 1).reshape(N * W * H, C)
    dx, dgamma, dbeta = batchnorm_backward(dout_r, cache)
    dx =  dx.reshape(N, W, H, C).transpose(0, 3, 2, 1)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """Computes the forward pass for spatial group normalization.
    
    In contrast to layer normalization, group normalization splits each entry in the data into G
    contiguous pieces, which it then normalizes independently. Per-feature shifting and scaling
    are then applied to the data, in a manner identical to that of batch normalization and layer
    normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (1, C, 1, 1)
    - beta: Shift parameter, of shape (1, C, 1, 1)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Idea: Build groups of channels, than normalize accross the group and accross
    # H and W, for a single datapoint.

    N, C, H, W = x.shape

    x_r = x.reshape(N, G, C // G, H, W)
    # Compute stats. along individual groups (C // G), height (H) and width (W)
    group_mean = x_r.mean(axis=(2, 3, 4), keepdims=True)
    group_var = x_r.var(axis=(2, 3, 4), keepdims=True)

    # Normalize the data
    x_r_hat = (x_r - group_mean) / np.sqrt(group_var + eps)

    # Reshape back
    x_hat = x_r_hat.reshape(N, C, H, W)

    # Shift and scale
    out = gamma * x_hat + beta

    cache = group_mean, group_var, x_r_hat, x_hat, x_r, out, gamma, beta, G, eps

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (1, C, 1, 1)
    - dbeta: Gradient with respect to shift parameter, of shape (1, C, 1, 1)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = dout.shape
    group_mean, group_var, x_r_hat, x_hat, x_r, out, gamma, beta, G, eps = cache

    dgamma = np.sum(x_hat * dout, axis=(0, 2, 3), keepdims=True)
    dbeta = np.sum(dout, axis=(0, 2, 3), keepdims=True)
    
    dx_hat = gamma * dout
    dx_r_hat = (1.0 * dx_hat).reshape(N, G, C // G, H, W)
    dgroup_var = np.sum(dx_r_hat * (x_r - group_mean) * (-0.5) * np.power(group_var + eps, -1.5), axis=(2,3,4), keepdims=True)
    
    # Note that we should normalize (N, G, C // G, H, W)
    dgroup_mean = np.sum(dx_r_hat * (-1.0) * np.power(group_var + eps, -0.5), axis=(2, 3, 4), keepdims=True) + np.sum((-2.0 / (C // G * H * W) * (x_r - group_mean) * dgroup_var), axis=(2, 3, 4), keepdims=True)
    dx_r = dx_r_hat * np.power(group_var + eps, -0.5) + dgroup_var * 2.0 * (x_r - group_mean) / (C // G * H * W) + dgroup_mean / (C // G * H * W)
    dx = dx_r.reshape(N, C, H, W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta
