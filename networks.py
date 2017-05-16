import tensorflow as tf
import numpy as np

from layers import conv2d, fully_connected_layer, batch_norm, resnet_block


def deep_network(x, y=None, number_of_classes=2, filters=(16, 32, 64, 128), strides=(2, 1, 2, 1)):
  # TODO: Use the conv2d, tf.nn.relu and fully_connected_layer to create a (n+1)-layer network,
  # where n corresponds to the length of filters and strides.
  # your n first layers should be convoultional and the last layer fully connected.
  logits = None
  params = {}
  assert len(filters)==len(strides), 'The parameters filter and stride should have the same length, had length %d and %d' \
  %((len(filters), len(strides)))

  ###### YOUR CODE #######
  # Build your network and output logits
  out = x
  
  for i, (filter, stride) in enumerate(zip(filters, strides), start=1):
    conv, conv_params = conv2d(out, number_of_features=filter, stride=stride, k_size=3) # k_size given by assignment
    out = tf.nn.relu(conv)

    for key, value in conv_params.items():
      params['conv%d/%s' % (i, key)] = value

  logits, dense_params = fully_connected_layer(out, number_of_classes)

  for key, value in dense_params.items():
    params['fc/%s' % key] = value

  # END OF YOUR CODE

  if y is None:
    return logits, params

  # TODO: Calculate softmax cross-entropy
  #  without using any of the softmax or cross-entropy functions from Tensorflow
  loss = None

  ###### YOUR CODE #######

  # Calculate loss
  h = tf.exp(logits - tf.reduce_max(logits, axis=1, keep_dims=True))
  h /= tf.reduce_sum(h, axis=1, keep_dims=True)

  loss = -tf.reduce_sum(y * tf.log(h), axis=1, keep_dims=True)
  loss = tf.reduce_mean(loss)
  #loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y) # For comparison and debug
  # END OF YOUR CODE

  return logits, loss, params


def deep_network_with_batchnorm(x, y=None,
                                number_of_classes=2,
                                filters=(16, 32, 64, 128),
                                strides=(2, 1, 2, 1),
                                is_training=True):
  # TODO: Do the same as with deep_network, but this time add batchnorm before each convoulution.

  logits = None
  params = {}
  assert len(filters)==len(strides), 'The parameters filter and stride should have the same length, had length %d and %d' \
  %((len(filters), len(strides)))

  update_ops = [] #Fill this with update_ops from batch_norm

  ###### YOUR CODE #######
  # Build your network and output logits
  out = x
  
  for i, (filter, stride) in enumerate(zip(filters, strides), start=1):
    bn_out, bn_params, update_op = batch_norm(out, is_training)
    conv, conv_params = conv2d(bn_out, number_of_features=filter, stride=stride, k_size=3) # k_size given by assignment
    out = tf.nn.relu(conv)

    for key, value in conv_params.items() + bn_params.items():
      params['conv%d/%s' % (i, key)] = value

    update_ops.append(update_op)

  logits, dense_params = fully_connected_layer(out, number_of_classes)

  for key, value in dense_params.items():
    params['fc/%s' % key] = value

  # END OF YOUR CODE

  if y is None:
    return logits, params, update_ops

  # TODO: Calculate softmax cross-entropy
  #  without using any of the softmax or cross-entropy functions from Tensorflow
  loss = None

  ###### YOUR CODE #######
  # Calculate loss
  h = tf.exp(logits - tf.reduce_max(logits, axis=1, keep_dims=True))
  h /= tf.reduce_sum(h, axis=1, keep_dims=True)

  loss = -tf.reduce_sum(y * tf.log(h), axis=1, keep_dims=True)
  loss = tf.reduce_mean(loss)
  #loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y) # For comparison and debug
  # END OF YOUR CODE

  update_op = tf.group(*tuple(update_ops))
  return logits, loss, params, update_op


def deep_residual_network(x, y=None,
                                number_of_classes=2,
                                filters=(16, 32, 64, 128),
                                strides=(2, 1, 2, 1),
                                is_training=True):
  # TODO: Do the same as with deep_network_with_batchnorm*, but this time use the residual_blocks

  logits = None
  params = {}
  assert len(filters)==len(strides), 'The parameters filter and stride should have the same length, had length %d and %d' \
  %((len(filters), len(strides)))

  update_ops = [] #Fill this with update_ops from batch_norm

  ###### YOUR CODE #######
  # Build your network and output logits
  out = x
  
  for i, (filter, stride) in enumerate(zip(filters, strides), start=1):
    out, resnet_params, update_op = resnet_block(out, filters=(filter, filter), k_size=(3,3), stride=stride, is_training=is_training)

    for key, value in resnet_params.items():
      params['resnet%d/%s' % (i, key)] = value

    update_ops.append(update_op)
  
  logits, dense_params = fully_connected_layer(out, number_of_classes)

  for key, value in dense_params.items():
    params['fc/%s' % key] = value
  # END OF YOUR CODE

  if y is None:
    return logits, params, update_ops

  # TODO: Calculate softmax cross-entropy loss
  #  without using any of the softmax or cross-entropy functions from Tensorflow
  loss = None

  ###### YOUR CODE #######
  # Calculate loss
  h = tf.exp(logits)
  h /= tf.reduce_sum(h) #, axis=1, keep_dims=True)

  loss = -tf.reduce_mean(tf.to_float(y) * tf.log(h))
  #loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y) # For comparison and debug
  # END OF YOUR CODE

  update_op = tf.group(*tuple(update_ops))
  return logits, loss, params, update_op

