# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 16:10:13 2018

@author: LHF
"""

from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np

XAVIER_INIT = True
USE_FP16 = False
TOTAL_LOSS_COLLECTION = 'my_total_losses'
WEIGHT_DECAY = 0.0

def variable_initializer(prev_units, curr_units, kernel_size, stddev_factor = 1.0):
    """Initialization for CONV2D in the style of Xavier Glorot et al.(2010).
    stddev_factor should be 1.0 for linear activations, and 2.0 for ReLUs.
    ArgS:
        prev_units: The number of channels in the previous layer.
        curr_units: The number of channels in the current layer.
        stddev_factor: 
    Returns:
        Initial value of the weights of the current conv/transpose conv layer.
    """
    
    if XAVIER_INIT:
        stddev = np.sqrt(stddev_factor/(np.sqrt(prev_units*curr_units)*kernel_size*kernel_size))
    else:
        stddev = 0.01
    
    return tf.truncated_normal_initializer(mean = 0.0, stddev = stddev)


def variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.

    Args:
        name: name of the variable
        shape: list of ints
        initializer: initializer for Variable

    Returns:
        Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float16 if USE_FP16 else tf.float32
        var = tf.get_variable(name, shape, initializer = initializer, dtype = dtype)
    return var


def variable_with_weight_decay(name, shape, weight_decay = None, is_conv = True):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
        name: name of the variable
        shape: list of ints
        stddev: standard deviation of a truncated Gaussian
        wd: add L2Loss weight decay multiplied by this float. If None, weight
            decay is not added for this Variable.
        is_conv: Initialization mode for variables: 'conv' and 'convT'

    Returns:
        Variable Tensor
    """
    if is_conv == True:
        initializer = variable_initializer(shape[2], shape[3], shape[0], stddev_factor = 1.0)
    else:
        initializer = variable_initializer(shape[3], shape[2], shape[0], stddev_factor = 1.0)
    
    var = variable_on_cpu(name, shape, initializer)
    if weight_decay is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), weight_decay, name = 'weight_loss')
        tf.add_to_collection(TOTAL_LOSS_COLLECTION, weight_decay)
    
    return var


def pooling(inputs, kernel_size, stride, mode = 'max', name = None):
    """Excute max or average pooling on inputs.
    
    Args:
        inputs: input tensor
        kernel_size: kernel size for pooling
        stride: stride used in pooling
        mode: 'max' | 'avg'
        name: name for this operation
    Returns:
        Tensor that is pooled, with size changed.
    """
    
    strides = [1, stride, stride, 1]
    ksize = [1, kernel_size, kernel_size, 1]
    if mode == 'max':
        inputs = tf.nn.max_pool(inputs, ksize, strides, padding = 'SAME', name = name)
    elif mode == 'avg':
        inputs = tf.nn.avg_pool(inputs, ksize, strides, padding = 'SAME', name = name)
    else:
        raise ValueError("Unknown pooling %s!" % mode)
        
    return inputs


def conv2d_layer(inputs, kernel_shape, stride = 1, weight_decay = 0.0, name = None):
    """A common convolutional layer that excutes 'SAME' convolution and bias addition.
    
    Args:
        inputs: 4D input tensor.
        kernel_shape: shape of convolution kernels, [batch_size, batch_size, in_maps, out_maps].
        weight_decay: weight decay factor.
        name: name for this operation.
    Returns:
        result tensor of conv & add operations.
    """
    with tf.variable_scope(name):
        W = variable_with_weight_decay('weights', kernel_shape, weight_decay, is_conv = True)
        b = variable_on_cpu('biases', [kernel_shape[3]], tf.constant_initializer())
        strides = [1, stride, stride, 1]
        padding = 'SAME'
        conv_name = name + 'conv_op'
        inputs = tf.nn.conv2d(inputs, W, strides, padding, name = conv_name)
    
    add_name = name + 'add_op'
    inputs = tf.nn.bias_add(inputs, b, name = add_name)
    
    return tf.nn.relu(inputs, name + 'relu')


def conv2d_layer_no_act(inputs, kernel_shape, stride = 1, weight_decay = 0.0, name = None):
    """A common convolutional layer that excutes 'SAME' convolution and bias addition.
    
    Args:
        inputs: 4D input tensor.
        kernel_shape: shape of convolution kernels, [batch_size, batch_size, in_maps, out_maps].
        weight_decay: weight decay factor.
        name: name for this operation.
    Returns:
        result tensor of conv & add operations.
    """
    with tf.variable_scope(name):
        W = variable_with_weight_decay('weights', kernel_shape, weight_decay, is_conv = True)
        b = variable_on_cpu('biases', [kernel_shape[3]], tf.constant_initializer())
        
        strides = [1, stride, stride, 1]
        padding = 'SAME'
        conv_name = name + 'conv_op'
        inputs = tf.nn.conv2d(inputs, W, strides, padding, name = conv_name)
    
    add_name = name + 'add_op'
    inputs = tf.nn.bias_add(inputs, b, name = add_name)
    
    return inputs


def batch_norm(inputs, is_training, name):
    """Functional interface for the batch normalization layer.
    
    Args:
        inputs: Tensor input.
        is_training: for training or testing.
        name: name for this operation.
    """
    
    BATCH_NORM_DECAY = 0.997
    BATCH_NORM_EPSILON = 1e-5
    inputs = tf.layers.batch_normalization(inputs = inputs,
                                           axis = 3,
                                           momentum = BATCH_NORM_DECAY,
                                           epsilon = BATCH_NORM_EPSILON,
                                           center = True,
                                           scale = True,
                                           training = is_training,
                                           fused = True)
    
    return inputs


def activation(inputs, types = 'relu', name = None):
    """ Common activation functions.
    
    Args:
        inputs: inputs 4D tensor.
        types: string that describes which kind of activation function will be used.
        name: name for this operation.
    Returns:
        result tensor.
    """
    if types == 'elu':
        inputs = tf.nn.elu(inputs, name + 'elu')
    elif types == 'relu':
        inputs = tf.nn.relu(inputs, name + 'relu')
    elif types == 'relu6':
        inputs = tf.nn.relu6(inputs, name + 'relu6')
    elif types == 'selu':
        inputs = tf.nn.selu(inputs, name + 'selu')
    elif types == 'softmax':
        inputs = tf.nn.softmax(inputs, 0, name + 'softmax')
    elif types == 'softplus':
        inputs = tf.nn.softplus(inputs, name + 'softplus')
    elif types == 'leaky_relu':
        alpha = tf.get_variable(name + "alpha", shape = (), dtype = tf.float32,
                            initializer = tf.constant_initializer(0.20))
        tf.summary.scalar(name + "leaky_relu_alpha", alpha)
        inputs = tf.nn.leaky_relu(inputs, alpha, name + 'leaky_relu')
    elif types == 'softsign':
        inputs = tf.nn.softsign(inputs, name + 'softplus')
    elif types == 'relu01':
        with tf.name_scope(name + 'relu01'):
            inputs = tf.minimum(tf.maximum(inputs, 0.0), 1.0)
    elif types == 'relu-1':
        with tf.name_scope(name + 'relu-1'):
            inputs = tf.minimum(tf.maximum(inputs, -0.5), 0.5)
    else:
        print("Invalid name for activation function!")
        return
    
    return inputs


def adaptive_res_block_bn_pre_acti(inputs, nlayer, wdecay = 0.0, name = None):
    maps = int(inputs.get_shape()[3])
    ksize_m = 2
    stride_m = 2
    
    ksize = 3
    stride = 1
    pool_mode = 'max'
    
    Kshape = [ksize, ksize, maps, 2*maps]
    inputs = conv2d_layer(inputs, Kshape, stride,  wdecay, (name + 'conv2d_layer_upsmple'))
    # nlayer >= 2
    long_skip = inputs
    for i in range(nlayer):
        short_skip = inputs
        Kshape = [ksize, ksize, 2*maps, 2*maps]
        inputs = batch_norm(inputs, is_training = True, name = (name + ('BN_%d' % i)))
        inputs = activation(inputs, types = 'relu', name = (name + ('relu_%d' % i)))
        inputs = conv2d_layer_no_act(inputs, Kshape, stride,  wdecay, name = (name + ('conv2d_layer_%d' % i)))
        inputs = inputs + short_skip  
        
    inputs = inputs + long_skip
    inputs = pooling(inputs, ksize_m, stride_m, pool_mode, (name + 'maxpool'))
    return inputs


def adaptive_res_block_BN(inputs, nlayer, wdecay = 0.0, name = None):
    maps = int(inputs.get_shape()[3])
    ksize_m = 2
    stride_m = 2
    
    ksize = 3
    stride = 1
    pool_mode = 'max'
    
    Kshape = [ksize, ksize, maps, 2*maps]
    inputs = conv2d_layer(inputs, Kshape, stride,  wdecay, (name + 'conv2d_layer_upsmple'))
    # nlayer >= 2
    long_skip = inputs
    for i in range(nlayer):
        short_skip = inputs
        Kshape = [ksize, ksize, 2*maps, 2*maps]
        inputs = conv2d_layer_no_act(inputs, Kshape, stride,  wdecay, (name + ('conv2d_layer_%d' % i)))
        inputs = batch_norm(inputs, is_training = True, name = (name + ('BN_%d' % i)))
        inputs = activation(inputs, types = 'relu', name = (name + ('relu_%d' % i)))
        inputs = inputs + short_skip  
    inputs = inputs + long_skip
    inputs = pooling(inputs, ksize_m, stride_m, pool_mode, (name + 'maxpool'))
    return inputs


def adaptive_res_block(inputs, nlayer, wdecay = 0.0, name = None):
    maps = int(inputs.get_shape()[3])
    ksize_m = 2
    stride_m = 2
    
    ksize = 3
    stride = 1
    pool_mode = 'max'
    
    Kshape = [ksize, ksize, maps, 2*maps]
    inputs = conv2d_layer(inputs, Kshape, stride,  wdecay, (name + 'conv2d_layer_upsmple'))
    # nlayer >= 2
    long_skip = inputs
    for i in range(nlayer):
        short_skip = inputs
        Kshape = [ksize, ksize, 2*maps, 2*maps]
        inputs = conv2d_layer(inputs, Kshape, stride,  wdecay, (name + ('conv2d_layer_%d' % i)))
        inputs = inputs + short_skip  
        
    inputs = inputs + long_skip
    inputs = pooling(inputs, ksize_m, stride_m, pool_mode, (name + 'maxpool'))
    return inputs


def adaptive_block(inputs, nlayer, wdecay = 0.0, name = None):
    maps = int(inputs.get_shape()[3])
    ksize_m = 2
    stride_m = 2
    
    ksize = 3
    stride = 1
    pool_mode = 'max'
    
    Kshape = [ksize, ksize, maps, 2*maps]
    inputs = conv2d_layer(inputs, Kshape, stride,  wdecay, (name + 'conv2d_layer_upsample'))
    
    for i in range(nlayer):
        Kshape = [ksize, ksize, 2*maps, 2*maps]
        inputs = conv2d_layer(inputs, Kshape, stride,  wdecay, (name + ('conv2d_layer_%d' % i)))
    inputs = pooling(inputs, ksize_m, stride_m, pool_mode, 'maxpool')
    return inputs


def fcn(inputs, maps_s1, maps_s2, maps_s3, isTraining):
    std = 0.01
    mean = 0.0
    dtype = tf.float32
    dropprob = 0.5 if isTraining else 1
    c, w, d = inputs.get_shape().as_list()[1:4]
    length = c * w * d
    inputs = tf.reshape(inputs, [-1, length], name="reshape")
    with tf.variable_scope("fc1"):
        fc1_weight = tf.Variable(tf.truncated_normal([length, maps_s1], mean, std, dtype, name="fc1_Weight"))
        fc1_bias = tf.Variable(tf.constant(0.0, dtype, shape=[maps_s1], name="fc1_bias"))
        fc1 = tf.matmul(inputs, fc1_weight)
        fc1 = tf.nn.bias_add(fc1, fc1_bias)
        fc1 = tf.nn.relu(fc1)
    fc1_drop = tf.nn.dropout(fc1, dropprob, name = "fc1_drop")  
    
    with tf.variable_scope("fc2"):
        fc2_weight = tf.Variable(tf.truncated_normal([maps_s1, maps_s2], mean, std, dtype, name="fc2_Weight"))
        fc2_bias = tf.Variable(tf.constant(0.0, dtype, shape=[maps_s2], name="fc2_bias"))
        fc2 = tf.matmul(fc1_drop, fc2_weight)
        fc2 = tf.nn.bias_add(fc2, fc2_bias)
        fc2 = tf.nn.relu(fc2)
    fc2_drop = tf.nn.dropout(fc2, dropprob, name = "fc2_drop")
    
    with tf.variable_scope("fc3"):
        fc3_weight = tf.Variable(tf.truncated_normal([maps_s2, maps_s3], mean, std, dtype, name = "fc3_Weight"))
        fc3_bias = tf.Variable(tf.constant(0.0, dtype, shape=[maps_s3], name = "fc3_bias"))
        fc3 = tf.matmul(fc2_drop, fc3_weight)
        fc3 = tf.nn.bias_add(fc3, fc3_bias)
        fc3 = tf.nn.relu(fc3)
    return fc3


def my_VGG(inputs):
    assert len(inputs.shape) == 4, "The dimension of inputs should be 4!"
    kernel_size_m = 2
    conv_stride_m = 2
    
    kernel_size = 3
    conv_stride = 1
    pool_mode = 'max'
    inputs = tf.space_to_depth(inputs, 2)
    in_maps = int(inputs.get_shape()[3])
    
    with tf.variable_scope("conv1_1"):
        inputs = conv2d_layer(inputs, [kernel_size, kernel_size, in_maps, 32], conv_stride,  WEIGHT_DECAY, 'conv2d_layer1_1')      

    with tf.variable_scope("conv1_2"):
        inputs = conv2d_layer(inputs, [kernel_size, kernel_size, 32, 32], conv_stride,  WEIGHT_DECAY, 'conv2d_layer1_2')
    inputs = pooling(inputs, kernel_size_m, conv_stride_m, pool_mode, "maxpool1")

    with tf.variable_scope("conv2_1"):
         inputs = conv2d_layer(inputs, [kernel_size, kernel_size, 32, 64], conv_stride,  WEIGHT_DECAY, 'conv2d_layer2_1')

    with tf.variable_scope("conv2_2"):
        inputs = conv2d_layer(inputs, [kernel_size, kernel_size, 64, 64], conv_stride,  WEIGHT_DECAY, 'conv2d_layer2_2')

    inputs = pooling(inputs, kernel_size_m, conv_stride_m, pool_mode, "maxpool2")
    
    with tf.variable_scope("conv3_1"):
        inputs = conv2d_layer(inputs, [kernel_size, kernel_size, 64, 128], conv_stride,  WEIGHT_DECAY, 'conv2d_layer3_1')

    with tf.variable_scope("conv3_2"):
        inputs = conv2d_layer(inputs, [kernel_size, kernel_size, 128, 128], conv_stride,  WEIGHT_DECAY, 'conv2d_layer3_2')

    with tf.variable_scope("conv3_3"):
        inputs = conv2d_layer(inputs, [kernel_size, kernel_size, 128, 128], conv_stride,  WEIGHT_DECAY, 'conv2d_layer3_3')

    inputs = pooling(inputs, kernel_size_m, conv_stride_m, pool_mode, "maxpool3")

    with tf.variable_scope("conv4_1"):
        inputs = conv2d_layer(inputs, [kernel_size, kernel_size, 128, 256], conv_stride,  WEIGHT_DECAY, 'conv2d_layer4_1')

    with tf.variable_scope("conv4_2"):
        inputs = conv2d_layer(inputs, [kernel_size, kernel_size, 256, 256], conv_stride,  WEIGHT_DECAY, 'conv2d_layer4_2')

    with tf.variable_scope("conv4_3"):
        inputs = conv2d_layer(inputs, [kernel_size, kernel_size, 256, 256], conv_stride,  WEIGHT_DECAY, 'conv2d_layer4_3')

    inputs = pooling(inputs, kernel_size_m, conv_stride_m, pool_mode, "maxpool4")

    inputs = pooling(inputs, kernel_size_m, conv_stride_m, pool_mode, "maxpool5")
    with tf.variable_scope("conv5_1"):
        inputs = conv2d_layer(inputs, [kernel_size, kernel_size, 256, 512], conv_stride,  WEIGHT_DECAY, 'conv2d_layer5_1')
        
    with tf.variable_scope("conv5_2"):
       inputs = conv2d_layer(inputs, [kernel_size, kernel_size, 512, 512], conv_stride,  WEIGHT_DECAY, 'conv2d_layer5_2')

    with tf.variable_scope("conv5_3"):
        inputs = conv2d_layer(inputs, [kernel_size, kernel_size, 512, 512], conv_stride,  WEIGHT_DECAY, 'conv2d_layer5_3')
    inputs = fcn(inputs, 1024, 1024, 5)
    
    return inputs


def Res_Block_BN(inputs, nlayer, wdecay = 0.0, name = None):
    
    maps = int(inputs.get_shape()[3])
    ksize = 3
    stride = 1
    # nlayer >= 2
    short_skip = inputs
    for i in range(nlayer):
        Kshape = [ksize, ksize, maps, maps]
        inputs = conv2d_layer_no_act(inputs, Kshape, stride,  wdecay, name = (name + ('conv2d_layer_%d' % i))) 
        inputs = batch_norm(inputs, is_training = True, name = (name + ('BN_%d' % i)))
        inputs = activation(inputs, types = 'relu', name = (name + ('relu_%d' % i)))
    inputs = inputs + short_skip
    return inputs


def Transition_Layer_BN(inputs,wdecay = 0.0, name = 'TLB'):
    maps = int(inputs.get_shape()[3])
    ksize = 3
    
    Kshape = [1, 1, maps, 2*maps]
    inputs = activation(inputs, types = 'relu', name = (name + ('relu_1')))
    short_skip = conv2d_layer_no_act(inputs, Kshape, 2,  wdecay, name = (name + ('conv2d_layer_1'))) 
    
    Kshape = [ksize, ksize, maps, 2*maps]
    inputs = conv2d_layer_no_act(inputs, Kshape, 2,  wdecay, name = (name + ('conv2d_layer_2'))) 
    inputs = batch_norm(inputs, is_training = True, name = (name + ('BN_1')))
    inputs = activation(inputs, types = 'relu', name = (name + ('relu_2')))
    
    Kshape = [ksize, ksize, 2*maps, 2*maps]
    inputs = conv2d_layer_no_act(inputs, Kshape, 1,  wdecay, name = (name + ('conv2d_layer_3')))
    inputs = batch_norm(inputs, is_training = True, name = (name + ('BN_2')))
    inputs = activation(inputs, types = 'relu', name = (name + ('relu_3')))
    
    inputs = inputs + short_skip
    return inputs


def adaptive_VGG_model(inputs):
    inputs = tf.space_to_depth(inputs, 4)
    inp_maps = int(inputs.get_shape()[3])
    num_block = 3
    init_maps = 32
    init_ksize = 3
    nlayer = 2
    inp_mean = tf.reduce_mean(inputs)
    inputs = inputs - inp_mean
    stride = 1
    wdecay = 0.0
    Kshape = [init_ksize, init_ksize, inp_maps, init_maps]
    inputs = conv2d_layer_no_act(inputs, Kshape, stride,  wdecay, 'conv2d_layer_init')
    
    for i in range(num_block):
        inputs = adaptive_block(inputs, nlayer, wdecay = 0.0, name = ('resbk%d/' % i))
        
    inputs = fcn(inputs, 1024, 256, 2)
    
    inputs = inputs + inp_mean
    return inputs
    


def Res_Block_Bn_Pre_Acti(inputs, nlayer, wdecay = 0.0, name = None):
    maps = int(inputs.get_shape()[3])
    ksize = 3
    stride = 1
    # nlayer >= 2
    short_skip = inputs
    for i in range(nlayer):
        Kshape = [ksize, ksize, maps, maps]
        inputs = batch_norm(inputs, is_training = True, name = (name + ('BN_%d' % i)))
        inputs = activation(inputs, types = 'relu', name = (name + ('relu_%d' % i)))
        inputs = conv2d_layer_no_act(inputs, Kshape, stride,  wdecay, name = (name + ('conv2d_layer_%d' % i))) 
        
    inputs = inputs + short_skip
    return inputs


def fcn_2(inputs, drop, maps_s1, maps_s2, isTraining):
    std = 0.01
    mean = 0.0
    dtype = tf.float32
    c, w, d = inputs.get_shape().as_list()[1:4]
    length = c * w * d
    inputs = tf.reshape(inputs, [-1, length], name="reshape")
    dropprob = drop if isTraining else 1
    with tf.variable_scope("fc1"):
        fc1_weight = tf.Variable(tf.truncated_normal([length, maps_s1], mean, std, dtype, name="fc1_Weight"))
        fc1_bias = tf.Variable(tf.constant(0.0, dtype, shape=[maps_s1], name="fc1_bias"))
        fc1 = tf.matmul(inputs, fc1_weight)
        fc1 = tf.nn.bias_add(fc1, fc1_bias)
        fc1 = tf.nn.relu(fc1)
    fc1_drop = tf.nn.dropout(fc1, dropprob, name = "fc1_drop")  
    
    with tf.variable_scope("fc2"):
        fc2_weight = tf.Variable(tf.truncated_normal([maps_s1, maps_s2], mean, std, dtype, name="fc2_Weight"))
        fc2_bias = tf.Variable(tf.constant(0.0, dtype, shape=[maps_s2], name="fc2_bias"))
        fc2 = tf.matmul(fc1_drop, fc2_weight)
        fc2 = tf.nn.bias_add(fc2, fc2_bias)
        fc2 = tf.nn.relu(fc2)
    return fc2


def Res_Net(inputs, isTraining):
    assert len(inputs.shape) == 4, "The dimension of inputs should be 4!"
    inputs = tf.space_to_depth(inputs, 4)
    inp_maps = int(inputs.get_shape()[3])
    ksize_m = 2
    stride_m = 2
    pool_mode = 'max'
    init_maps = 64
    init_ksize = 3
    nlayer = 2
    #inp_mean = tf.reduce_mean(inputs)
    #inputs = inputs - inp_mean
    stride = 1
    wdecay = 0.0
    Kshape = [init_ksize, init_ksize, inp_maps, init_maps]
    inputs = conv2d_layer(inputs, Kshape, stride,  wdecay, 'conv2d_layer_init')
    inputs = pooling(inputs, ksize_m, stride_m, pool_mode, 'maxpool')
    for i in range(3):
        inputs = Res_Block_BN(inputs, nlayer, wdecay = 0.0, name = ('resbk1_%d/' % i))
    
    inputs = Transition_Layer_BN(inputs, wdecay = 0.0, name = 'TLB_1')
    
    for i in range(3):
        inputs = Res_Block_BN(inputs, nlayer, wdecay = 0.0, name = ('resbk2_%d/' % i))
        
    inputs = Transition_Layer_BN(inputs, wdecay = 0.0, name = 'TLB_2')
    
    for i in range(6):
        inputs = Res_Block_BN(inputs, nlayer, wdecay = 0.0, name = ('resbk3_%d/' % i))
    
    inputs = Transition_Layer_BN(inputs, wdecay = 0.0, name = 'TLB_3')
    
    for i in range(2):
        inputs = Res_Block_BN(inputs, nlayer, wdecay = 0.0, name = ('resbk4_%d/' % i))
    inputs = fcn(inputs, 1024, 1024, 5, isTraining)
    
    #inputs = inputs + inp_mean
    return inputs    
    
    
def My_Model(inputs, isTraining):
    assert len(inputs.shape) == 4, "The dimension of inputs should be 4!"
    
    with tf.variable_scope("CONV_BLOCK_1"):
        in_maps = int(inputs.get_shape()[3])
        kshape = [7, 7, in_maps, 64]
        inputs = conv2d_layer(inputs, kshape, 2, WEIGHT_DECAY, 'conv2d_layer_1')
        inputs = pooling(inputs, 3, 2, 'max', 'maxpooling_layer')
    
    with tf.variable_scope("CONV_BLOCK_2"):
        in_maps = int(inputs.get_shape()[3])
        kshape = [1, 1, in_maps, 64]
        inputs = conv2d_layer(inputs, kshape, 1, WEIGHT_DECAY, 'conv2d_layer_1')
        kshape = [3, 3, 64, 192]
        inputs = conv2d_layer(inputs, kshape, 2, WEIGHT_DECAY, 'conv2d_layer_2')
        
    with tf.variable_scope("CONV_BLOCK_3a"):
        in_maps = int(inputs.get_shape()[3])
        inputs_3a = pooling(inputs, 3, 2, 'max', 'maxpooling_layer')
        with tf.variable_scope("Inception_Block"):
            kshape = [1, 1, in_maps, 64]
            inputs_3a_1 = conv2d_layer(inputs_3a, kshape, 1, WEIGHT_DECAY, 'conv2d_layer_1')
            
            kshape = [1, 1, in_maps, 96]
            inputs_3a_2 = conv2d_layer(inputs_3a, kshape, 1, WEIGHT_DECAY, 'conv2d_layer_2')
            kshape = [3, 3, 96, 128]
            inputs_3a_2 = conv2d_layer(inputs_3a_2, kshape, 1, WEIGHT_DECAY, 'conv2d_layer_3')
            
            kshape = [1, 1, in_maps, 16]
            inputs_3a_3 = conv2d_layer(inputs_3a, kshape, 1, WEIGHT_DECAY, 'conv2d_layer_4')
            kshape = [5, 5, 16, 32]
            inputs_3a_3 = conv2d_layer(inputs_3a_3, kshape, 1, WEIGHT_DECAY, 'conv2d_layer_5')
            
            inputs_3a_4 = pooling(inputs_3a, 3, 1, 'max', 'maxpooling_layer')
            kshape = [1, 1, in_maps, 32]
            inputs_3a_4 = conv2d_layer(inputs_3a_4, kshape, 1, WEIGHT_DECAY, 'conv2d_layer_6')
            
            inputs = tf.concat([inputs_3a_1, inputs_3a_2, inputs_3a_3, inputs_3a_4], 3)
    
    with tf.variable_scope("CONV_BLOCK_3b"):
        in_maps = int(inputs.get_shape()[3])
        inputs_3b = inputs
        with tf.variable_scope("Inception_Block"):
            kshape = [1, 1, in_maps, 128]
            inputs_3b_1 = conv2d_layer(inputs_3b, kshape, 1, WEIGHT_DECAY, 'conv2d_layer_1')
            
            kshape = [1, 1, in_maps, 128]
            inputs_3b_2 = conv2d_layer(inputs_3b, kshape, 1, WEIGHT_DECAY, 'conv2d_layer_2')
            kshape = [3, 3, 128, 192]
            inputs_3b_2 = conv2d_layer(inputs_3b_2, kshape, 1, WEIGHT_DECAY, 'conv2d_layer_3')
            
            kshape = [1, 1, in_maps, 32]
            inputs_3b_3 = conv2d_layer(inputs_3b, kshape, 1, WEIGHT_DECAY, 'conv2d_layer_4')
            kshape = [5, 5, 32, 96]
            inputs_3b_3 = conv2d_layer(inputs_3b_3, kshape, 1, WEIGHT_DECAY, 'conv2d_layer_5')
            
            inputs_3b_4 = pooling(inputs_3b, 3, 1, 'max', 'maxpooling_layer')
            kshape = [1, 1, in_maps, 64]
            inputs_3b_4 = conv2d_layer(inputs_3b_4, kshape, 1, WEIGHT_DECAY, 'conv2d_layer_6')
            
            inputs = tf.concat([inputs_3b_1, inputs_3b_2, inputs_3b_3, inputs_3b_4], 3)
    
    with tf.variable_scope("CONV_BLOCK_4a"):
        in_maps = int(inputs.get_shape()[3])
        inputs_4a = pooling(inputs, 3, 2, 'max', 'maxpooling_layer')
        with tf.variable_scope("Inception_Block"):
            kshape = [1, 1, in_maps, 192]
            inputs_4a_1 = conv2d_layer(inputs_4a, kshape, 1, WEIGHT_DECAY, 'conv2d_layer_1')
            
            kshape = [1, 1, in_maps, 96]
            inputs_4a_2 = conv2d_layer(inputs_4a, kshape, 1, WEIGHT_DECAY, 'conv2d_layer_2')
            kshape = [3, 3, 96, 208]
            inputs_4a_2 = conv2d_layer(inputs_4a_2, kshape, 1, WEIGHT_DECAY, 'conv2d_layer_3')
            
            kshape = [1, 1, in_maps, 16]
            inputs_4a_3 = conv2d_layer(inputs_4a, kshape, 1, WEIGHT_DECAY, 'conv2d_layer_4')
            kshape = [5, 5, 16, 48]
            inputs_4a_3 = conv2d_layer(inputs_4a_3, kshape, 1, WEIGHT_DECAY, 'conv2d_layer_5')
            
            inputs_4a_4 = pooling(inputs_4a, 3, 1, 'max', 'maxpooling_layer')
            kshape = [1, 1, in_maps, 64]
            inputs_4a_4 = conv2d_layer(inputs_4a_4, kshape, 1, WEIGHT_DECAY, 'conv2d_layer_6')
            
            inputs = tf.concat([inputs_4a_1, inputs_4a_2, inputs_4a_3, inputs_4a_4], 3)
    
    with tf.variable_scope("CONV_BLOCK_4b"):
        in_maps = int(inputs.get_shape()[3])
        inputs_4b = inputs
        with tf.variable_scope("Inception_Block"):
            kshape = [1, 1, in_maps, 160]
            inputs_4b_1 = conv2d_layer(inputs_4b, kshape, 1, WEIGHT_DECAY, 'conv2d_layer_1')
            
            kshape = [1, 1, in_maps, 112]
            inputs_4b_2 = conv2d_layer(inputs_4b, kshape, 1, WEIGHT_DECAY, 'conv2d_layer_2')
            kshape = [3, 3, 112, 224]
            inputs_4b_2 = conv2d_layer(inputs_4b_2, kshape, 1, WEIGHT_DECAY, 'conv2d_layer_3')
            
            kshape = [1, 1, in_maps, 24]
            inputs_4b_3 = conv2d_layer(inputs_4b, kshape, 1, WEIGHT_DECAY, 'conv2d_layer_4')
            kshape = [5, 5, 24, 64]
            inputs_4b_3 = conv2d_layer(inputs_4b_3, kshape, 1, WEIGHT_DECAY, 'conv2d_layer_5')
            
            inputs_4b_4 = pooling(inputs_4b, 3, 1, 'max', 'maxpooling_layer')
            kshape = [1, 1, in_maps, 64]
            inputs_4b_4 = conv2d_layer(inputs_4b_4, kshape, 1, WEIGHT_DECAY, 'conv2d_layer_6')
            
            inputs = tf.concat([inputs_4b_1, inputs_4b_2, inputs_4b_3, inputs_4b_4], 3)
            
            inputs_1 = pooling(inputs_4b, 5, 4, 'avg', 'avgpooling_layer')
            in_maps = int(inputs_1.get_shape()[3])
            kshape = [1, 1, in_maps, 128]
            inputs_1 = conv2d_layer(inputs_1, kshape, 1, WEIGHT_DECAY, 'conv2d_layer_7')
            inputs_1 = fcn_2(inputs_1, 0.7, 1024, 5, isTraining)
    
    with tf.variable_scope("CONV_BLOCK_4c"):
        in_maps = int(inputs.get_shape()[3])
        inputs_4c = inputs
        with tf.variable_scope("Inception_Block"):
            kshape = [1, 1, in_maps, 128]
            inputs_4c_1 = conv2d_layer(inputs_4c, kshape, 1, WEIGHT_DECAY, 'conv2d_layer_1')
            
            kshape = [1, 1, in_maps, 128]
            inputs_4c_2 = conv2d_layer(inputs_4c, kshape, 1, WEIGHT_DECAY, 'conv2d_layer_2')
            kshape = [3, 3, 128, 256]
            inputs_4c_2 = conv2d_layer(inputs_4c_2, kshape, 1, WEIGHT_DECAY, 'conv2d_layer_3')
            
            kshape = [1, 1, in_maps, 24]
            inputs_4c_3 = conv2d_layer(inputs_4c, kshape, 1, WEIGHT_DECAY, 'conv2d_layer_4')
            kshape = [5, 5, 24, 64]
            inputs_4c_3 = conv2d_layer(inputs_4c_3, kshape, 1, WEIGHT_DECAY, 'conv2d_layer_5')
            
            inputs_4c_4 = pooling(inputs_4c, 3, 1, 'max', 'maxpooling_layer')
            kshape = [1, 1, in_maps, 64]
            inputs_4c_4 = conv2d_layer(inputs_4c_4, kshape, 1, WEIGHT_DECAY, 'conv2d_layer_6')
            
            inputs = tf.concat([inputs_4c_1, inputs_4c_2, inputs_4c_3, inputs_4c_4], 3)
    with tf.variable_scope("CONV_BLOCK_4d"):
        in_maps = int(inputs.get_shape()[3])
        inputs_4d = inputs
        with tf.variable_scope("Inception_Block"):
            kshape = [1, 1, in_maps, 112]
            inputs_4d_1 = conv2d_layer(inputs_4d, kshape, 1, WEIGHT_DECAY, 'conv2d_layer_1')
            
            kshape = [1, 1, in_maps, 144]
            inputs_4d_2 = conv2d_layer(inputs_4d, kshape, 1, WEIGHT_DECAY, 'conv2d_layer_2')
            kshape = [3, 3, 144, 288]
            inputs_4d_2 = conv2d_layer(inputs_4d_2, kshape, 1, WEIGHT_DECAY, 'conv2d_layer_3')
            
            kshape = [1, 1, in_maps, 32]
            inputs_4d_3 = conv2d_layer(inputs_4d, kshape, 1, WEIGHT_DECAY, 'conv2d_layer_4')
            kshape = [5, 5, 32, 64]
            inputs_4d_3 = conv2d_layer(inputs_4d_3, kshape, 1, WEIGHT_DECAY, 'conv2d_layer_5')
            
            inputs_4d_4 = pooling(inputs_4d, 3, 1, 'max', 'maxpooling_layer')
            kshape = [1, 1, in_maps, 64]
            inputs_4d_4 = conv2d_layer(inputs_4d_4, kshape, 1, WEIGHT_DECAY, 'conv2d_layer_6')
            
            inputs = tf.concat([inputs_4d_1, inputs_4d_2, inputs_4d_3, inputs_4d_4], 3)
            
    with tf.variable_scope("CONV_BLOCK_4e"):
        in_maps = int(inputs.get_shape()[3])
        inputs_4e = inputs
        with tf.variable_scope("Inception_Block"):
            kshape = [1, 1, in_maps, 112]
            inputs_4e_1 = conv2d_layer(inputs_4e, kshape, 1, WEIGHT_DECAY, 'conv2d_layer_1')
            
            kshape = [1, 1, in_maps, 144]
            inputs_4e_2 = conv2d_layer(inputs_4e, kshape, 1, WEIGHT_DECAY, 'conv2d_layer_2')
            kshape = [3, 3, 144, 288]
            inputs_4e_2 = conv2d_layer(inputs_4e_2, kshape, 1, WEIGHT_DECAY, 'conv2d_layer_3')
            
            kshape = [1, 1, in_maps, 32]
            inputs_4e_3 = conv2d_layer(inputs_4e, kshape, 1, WEIGHT_DECAY, 'conv2d_layer_4')
            kshape = [5, 5, 32, 64]
            inputs_4e_3 = conv2d_layer(inputs_4e_3, kshape, 1, WEIGHT_DECAY, 'conv2d_layer_5')
            
            inputs_4e_4 = pooling(inputs_4e, 3, 1, 'max', 'maxpooling_layer')
            kshape = [1, 1, in_maps, 64]
            inputs_4e_4 = conv2d_layer(inputs_4e_4, kshape, 1, WEIGHT_DECAY, 'conv2d_layer_6')
            
            inputs = tf.concat([inputs_4e_1, inputs_4e_2, inputs_4e_3, inputs_4e_4], 3)
    
            inputs_2 = pooling(inputs_4e, 5, 4, 'avg', 'avgpooling_layer')
            in_maps = int(inputs_2.get_shape()[3])
            kshape = [1, 1, in_maps, 128]
            inputs_2 = conv2d_layer(inputs_2, kshape, 1, WEIGHT_DECAY, 'conv2d_layer_7')

            inputs_2 = fcn_2(inputs_2, 0.7, 1024, 5, isTraining)
    
    with tf.variable_scope("CONV_BLOCK_5a"):
        in_maps = int(inputs.get_shape()[3])
        inputs_5a = pooling(inputs, 3, 2, 'max', 'maxpooling_layer')
        with tf.variable_scope("Inception_Block"):
            kshape = [1, 1, in_maps, 64]
            inputs_5a_1 = conv2d_layer(inputs_5a, kshape, 1, WEIGHT_DECAY, 'conv2d_layer_1')
            
            kshape = [1, 1, in_maps, 96]
            inputs_5a_2 = conv2d_layer(inputs_5a, kshape, 1, WEIGHT_DECAY, 'conv2d_layer_2')
            kshape = [3, 3, 96, 128]
            inputs_5a_2 = conv2d_layer(inputs_5a_2, kshape, 1, WEIGHT_DECAY, 'conv2d_layer_3')
            
            kshape = [1, 1, in_maps, 16]
            inputs_5a_3 = conv2d_layer(inputs_5a, kshape, 1, WEIGHT_DECAY, 'conv2d_layer_4')
            kshape = [5, 5, 16, 32]
            inputs_5a_3 = conv2d_layer(inputs_5a_3, kshape, 1, WEIGHT_DECAY, 'conv2d_layer_5')
            
            inputs_5a_4 = pooling(inputs_5a, 3, 1, 'max', 'maxpooling_layer')
            kshape = [1, 1, in_maps, 32]
            inputs_5a_4 = conv2d_layer(inputs_5a_4, kshape, 1, WEIGHT_DECAY, 'conv2d_layer_6')
            
            inputs = tf.concat([inputs_5a_1, inputs_5a_2, inputs_5a_3, inputs_5a_4], 3)
    
    with tf.variable_scope("CONV_BLOCK_5b"):
        in_maps = int(inputs.get_shape()[3])
        inputs_5b = inputs
        with tf.variable_scope("Inception_Block"):
            kshape = [1, 1, in_maps, 128]
            inputs_5b_1 = conv2d_layer(inputs_5b, kshape, 1, WEIGHT_DECAY, 'conv2d_layer_1')
            
            kshape = [1, 1, in_maps, 128]
            inputs_5b_2 = conv2d_layer(inputs_5b, kshape, 1, WEIGHT_DECAY, 'conv2d_layer_2')
            kshape = [3, 3, 128, 192]
            inputs_5b_2 = conv2d_layer(inputs_5b_2, kshape, 1, WEIGHT_DECAY, 'conv2d_layer_3')
            
            kshape = [1, 1, in_maps, 32]
            inputs_5b_3 = conv2d_layer(inputs_5b, kshape, 1, WEIGHT_DECAY, 'conv2d_layer_4')
            kshape = [5, 5, 32, 96]
            inputs_5b_3 = conv2d_layer(inputs_5b_3, kshape, 1, WEIGHT_DECAY, 'conv2d_layer_5')
            
            inputs_5b_4 = pooling(inputs_5b, 3, 1, 'max', 'maxpooling_layer')
            kshape = [1, 1, in_maps, 64]
            inputs_5b_4 = conv2d_layer(inputs_5b_4, kshape, 1, WEIGHT_DECAY, 'conv2d_layer_6')
            
            inputs = tf.concat([inputs_5b_1, inputs_5b_2, inputs_5b_3, inputs_5b_4], 3)
            
            c = inputs.get_shape().as_list()[1]
            inputs_3= pooling(inputs, c, 1, 'avg', 'avgpooling_layer')
            inputs_3 = fcn_2(inputs_3, 0.7, 512, 5, isTraining)
    inputs = inputs_1 * 0.3 + inputs_2 * 0.3 + inputs_3 * 0.4
    return inputs
    
    
    
    
    
    
    
    