from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import re
my_weight_decay = 0.001
def _variable(name, shape, initializer): # 获取参数（并保存）的函数
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the Variable
    shape: list of ints
    initializer: initializer of Variable

  Returns:
    Variable Tensor
  """
  var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
  return var

def _variable_with_weight_decay(name, shape, stddev, wd): # 获取
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with truncated normal distribution
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable 
    shape: list of ints
    stddev: standard devision of a truncated Gaussian / 截断正态分�?    wd: add L2Loss weight decay multiplied by this float. If None, weight   / 衰减系数 
    decay is not added for this Variable.

 Returns:
    Variable Tensor 
  """
  var = _variable(name, shape,
    tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

def conv2d(scope, input, kernel_size, stride=1, dilation=1, relu=True, wd=my_weight_decay):  # 加入 scope �?weight decay
  """convolutional layer

  Args:
    input: 4-D tensor [batch_size, height, width, depth]
    scope: variable_scope name  / 
    kernel_size: [k_height, k_width, in_channel, out_channel] / 滤波器参�?    stride: int32 / 跨度
  Return:
    output: 4-D tensor [batch_size, height * stride, width * stride, out_channel]
  """
  name = scope
  with tf.variable_scope(scope) as scope:
    kernel = _variable_with_weight_decay('weights',       # 定义filter参数
                                    shape=kernel_size,
                                    stddev=5e-2,
                                    wd=wd)
    if dilation == 1:        # 正常卷积
      conv = tf.nn.conv2d(input, kernel, [1, stride, stride, 1], padding='SAME')  # tf自带2维卷积功能，保持图片大小
    else:                    # 空洞卷积
      conv = tf.nn.atrous_conv2d(input, kernel, dilation, padding='SAME') 
    biases = _variable('biases', kernel_size[3:], tf.constant_initializer(0.0))   # 提取自己定义的biases
    bias = tf.nn.bias_add(conv, biases) # 加入bias到conv layer
    if relu:  # 选用ReLU
      conv1 = tf.nn.relu(bias)
    else:    # 不选用ReLU
      conv1 = bias 
  return conv1   # 返回输出的feature map

def deconv2d(scope, input, kernel_size, stride=1, wd=my_weight_decay):   # 同样加入 scope �?weight decay
  """de-convolutional layer / 反卷�?
  Args:
    input: 4-D tensor [batch_size, height, width, depth]
    scope: variable_scope name 
    kernel_size: [k_height, k_width, in_channel, out_channel]   / 滤波器参�?    stride: int32 / 跨度
  Return:
    output: 4-D tensor [batch_size, height * stride, width * stride, out_channel]
  """
  pad_size = int((kernel_size[0] - 1)/2)  # 
  #input = tf.pad(input, [[0,0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]], "CONSTANT")
  batch_size, height, width, in_channel = [int(i) for i in input.get_shape()]
  out_channel = kernel_size[3] 
  kernel_size = [kernel_size[0], kernel_size[1], kernel_size[3], kernel_size[2]]
  output_shape = [batch_size, height * stride, width * stride, out_channel]   # 确定反卷积后的尺�?  with tf.variable_scope(scope) as scope:
    kernel = _variable_with_weight_decay('weights', 
                                    shape=kernel_size,
                                    stddev=5e-2,
                                    wd=wd)
    deconv = tf.nn.conv2d_transpose(input, kernel, output_shape, [1, stride, stride, 1], padding='SAME') # tf自带反卷积功�?
    biases = _variable('biases', (out_channel), tf.constant_initializer(0.0)) # 加入偏置
    bias = tf.nn.bias_add(deconv, biases)
    deconv1 = tf.nn.relu(bias)

  return deconv1 

def batch_norm(scope, x, train=True, reuse=False):  # batch normalization
  return tf.contrib.layers.batch_norm(x, center=True, scale=True, updates_collections=None, is_training=train, trainable=True, scope=scope)

