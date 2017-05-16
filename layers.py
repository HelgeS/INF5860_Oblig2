from __future__ import print_function, division
import tensorflow as tf
import numpy as np


def conv2d(x, number_of_features, k_size=3, stride=1):
  """
  
  :param x: Input of 4-dimensional tensorflow tensor
  :param number_of_features: Number of output channels
  :param k_size: The spatial size of the filter in both weight and width
  :param stride: Stride in spatial dimensions
  :return: x filtered by the kernal W and added bias b
  """
  params = {'W': None, 'b': None}
  out = None
  #------------------------------------------------------------#
  # TODO: Implement a 2D convolutional layer
  # Use the tf.nn.conv2d function from tensorflow and add a bias, but use no activation function.
  # store the weight and bias variables in the params dictionary.

  #YOUR CODE
  filters_in = x.get_shape().as_list()[-1]
  kernel = tf.Variable(tf.random_normal([k_size, k_size, filters_in,  number_of_features], stddev=np.sqrt(2.0 / (filters_in*k_size**2*number_of_features))), trainable=True)
  bias = tf.Variable(tf.zeros([number_of_features]), trainable=True)
  
  out = tf.nn.conv2d(
            input=x,
            filter=kernel,
            strides=(1, stride, stride, 1),
            padding='SAME'
            ) + bias

  params['W'] = kernel
  params['b'] = bias

  #END OF YOUR CODE
  return out, params


def fully_connected_layer(x, out_size):
  params = {'W': None, 'b': None}
  out = None
  #------------------------------------------------------------#
  # TODO: Implement a fully connected layer
  # Use tf.matmul. The dimension of x is either 2 or 4,
  # but the output should have 2 dimensions (batch_size, out_size)

  #YOUR CODE
  inputs = np.product(x.get_shape().as_list()[1:])
  W = tf.Variable(tf.truncated_normal([inputs, out_size], stddev=np.sqrt(3.0 / (inputs+out_size))))
  b = tf.Variable(tf.zeros([out_size]))
  out = tf.matmul(tf.reshape(x, (-1, inputs)), W) + b
  
  params['W'] = W
  params['b'] = b
  
  #END OF YOUR CODE
  return out, params


def batch_norm(x, is_training=True):
  params = {'gamma': None, 'beta': None, 'average_mean': None, 'average_var': None}
  update_op = None
  out = None
  is_training_tensor = tf.convert_to_tensor(is_training)
  assign_mean = None
  assign_var = None
  # TODO: Implement batch norm
  # Average over batches and spatial size, but not over channels.
  # Use exponential smoothing (https://en.wikipedia.org/wiki/Exponential_smoothing)
  # to keep the average mean and average variation.
  # Gamma should be initialized to 1 and beta 2, but remember that they should be trainable (Variable)

  #YOUR CODE
  inputs_shape = x.get_shape()
  axis = list(range(len(inputs_shape) - 1))
  params_shape = inputs_shape #[1,1,1,inputs_shape[-1]]
  epsilon = 1e-7

  gamma = tf.Variable(tf.ones(params_shape), trainable=True)
  beta = tf.Variable(tf.zeros(params_shape), trainable=True)
  
  average_mean = tf.Variable(tf.zeros(params_shape), trainable=False)
  average_var = tf.Variable(tf.ones(params_shape), trainable=False)

  batch_mean, batch_var = tf.nn.moments(x, axes=axis, keep_dims=True)

  assign_mean = average_mean.assign(average_mean*0.85 + batch_mean*0.15)
  assign_var = average_var.assign(average_var*0.85 + batch_var*0.15)

  mean, var = tf.cond(is_training_tensor, lambda: (batch_mean, batch_var), lambda: (average_mean, average_var))
  
  dividend = (x - mean)
  divisor = tf.sqrt(var) # + epsilon)
  normed = tf.div(dividend, divisor)
  out = gamma * normed + beta

  params['gamma'] = gamma
  params['beta'] = beta
  params['average_mean'] = average_mean
  params['average_var'] = average_var
  
  #END OF YOUR CODE
  update_op = tf.group(assign_mean, assign_var)
  return out, params, update_op




def resnet_block(x, filters=(32, 64), k_size=(3, 3), stride=1, is_training=True):
  params = {}
  out = None
  update_ops = []
  params = {'A/W': None, 'A/b': None, 'A/gamma': None, 'A/beta': None,
            'B/W': None, 'B/b': None, 'B/gamma': None, 'B/beta': None,
            'shortcut/W': None, 'shortcut/b': None}
  # TODO: implament a ResNet-block
  # In this exercise you should implement a "preactivated" resnet-block with 3 convolutions
  # and batch norm. The params should contain all parameters, and and the names of each conv and batch norm
  # should start with either 'A/', 'B/' or 'shortcut/', depending on which convolution it
  # belongs to.
  bn_out1, bn_params1, update_op1 = batch_norm(x, is_training)
  relu_out1 = tf.nn.relu(bn_out1, name='relu1')
  conv1, conv_params1 = conv2d(relu_out1, number_of_features=filters[0], stride=stride, k_size=k_size[0])

  bn_out2, bn_params2, update_op2 = batch_norm(conv1, is_training)
  relu_out2 = tf.nn.relu(bn_out2, name='relu2')
  conv2, conv_params2 = conv2d(relu_out2, number_of_features=filters[1], stride=1, k_size=k_size[1])

  shortcut_out, shortcut_params = conv2d(x, number_of_features=filters[-1], stride=stride, k_size=1)
  
  out = conv2 + shortcut_out

  params['A/gamma'] = bn_params1['gamma']
  params['A/beta'] = bn_params1['beta']
  params['A/W'] = conv_params1['W']
  params['A/b'] = conv_params1['b']

  params['B/gamma'] = bn_params2['gamma']
  params['B/beta'] = bn_params2['beta']
  params['B/W'] = conv_params2['W']
  params['B/b'] = conv_params2['b']
  update_ops = [update_op1, update_op2]

  params['shortcut/W'] = shortcut_params['W']
  params['shortcut/b'] = shortcut_params['b']
  #END OF YOUR CODE
  update_op = tf.group(*tuple(update_ops))
  return out, params, update_op
