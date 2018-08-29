from __future__ import print_function
import tensorflow as tf
import numpy as np
import math
import scipy.ndimage

from scipy import misc
from PIL import Image
from tensorflow.python.framework import ops

import pdb

import utils_binary as utils

SHOW_MODEL = False

#========================#
#    Model Components    #
#========================#
#-----------#
#    CNN    #
#-----------#
def CReLU(
    net,
    initializer,
    scope = "CReLU"
    ):
    
    input_channel = net.get_shape().as_list()[-1]
    output_channel = 2 * input_channel
    
    with tf.variable_scope(scope):
        net_negtive = tf.negative(net)
        net = tf.concat([net, net_negtive], axis = 3)
        
        scales = tf.get_variable("scales", [1, 1, output_channel, 1], tf.float32, initializer = initializer)
        shifts = tf.get_variable("shifts", [output_channel], tf.float32, initializer = initializer)
        
        net = tf.nn.depthwise_conv2d( input   = net, 
                                      filter  = scales, 
                                      strides = [1, 1, 1, 1], 
                                      padding = "SAME", 
                                      rate    = [1, 1])
        
        net = tf.nn.bias_add(net, shifts)
        
    return net
          
"""
def batch_norm(
    net, 
    is_training
    ): 
    
	return tf.contrib.layers.batch_norm( 
            inputs                  = net, 
            decay                   = 0.999,
            center                  = True,
            scale					= True,
            epsilon                 = 0.001,
            activation_fn           = None,
            param_initializers      = None,
            param_regularizers      = None,
            updates_collections     = tf.GraphKeys.UPDATE_OPS,
            is_training             = is_training,
            reuse                   = None,
            variables_collections   = None,
            outputs_collections     = None,
            trainable               = True,
            batch_weights           = None,
            fused                   = None,
            data_format             = "NHWC",
            zero_debias_moving_mean = False,
            scope                   = "Batch_Norm",
            renorm                  = False,
            renorm_clipping         = None,
            renorm_decay            = 0.99)
"""

def batch_norm(
    net, 
    is_training,
    data_format,
    trainable = True
    ):            

    net = tf.layers.batch_normalization(
            inputs    = net, 
            axis      = 1 if data_format == 'NCHW' else 3,
            momentum  = 0.997, 
            epsilon   = 1e-5, 
            center    = True,
            scale     = True, 
            training  = is_training,
            trainable = trainable,            
            fused     = True)
    
    return net

"""
def batch_norm(
    net,
    is_training,
    ):
    bn_train = tf.contrib.layers.batch_norm(
        net, 
        decay               = 0.999, 
        center              = True, 
        scale               = True,
        updates_collections = None,
        is_training         = True,
        reuse               = None, # is this right?
        trainable           = True,
        scope               = "Batch_Norm")
    
    bn_inference = tf.contrib.layers.batch_norm(
        net, 
        decay               = 0.999, 
        center              = True, 
        scale               = True,
        updates_collections = None,
        is_training         = False,
        reuse               = True, # is this right?
        trainable           = True,
        scope               = "Batch_Norm")
            
    z = tf.cond(is_training, lambda: bn_train, lambda: bn_inference)
    return z                
"""
               
def ternarize_weights(
    float32_weights, 
    ternary_weights_bd
    ):
    ternary_weights = tf.multiply(tf.cast(tf.less_equal(float32_weights, ternary_weights_bd[0]), tf.float32), tf.constant(-1, tf.float32))
    ternary_weights = tf.add(ternary_weights, tf.multiply(tf.cast(tf.greater(float32_weights, ternary_weights_bd[1]), tf.float32), tf.constant( 1, tf.float32)))
    
    return ternary_weights
	
def ternarize_biases(
    float32_biases, 
    ternary_biases_bd
    ):
    
    ternary_biases = tf.multiply(tf.cast(tf.less_equal(float32_biases, ternary_biases_bd[0]), tf.float32), tf.constant(-1, tf.float32))
    ternary_biases = tf.add(ternary_biases, tf.multiply(tf.cast(tf.greater(float32_biases, ternary_biases_bd[1]), tf.float32), tf.constant( 1, tf.float32)))
    
    return ternary_biases

def binarize_weights(
    float32_weights, 
    binary_weights_bd
    ):
    binary_weights = tf.multiply(tf.cast(tf.less_equal(float32_weights, binary_weights_bd), tf.float32), tf.constant(-1, tf.float32))
    binary_weights = tf.add(binary_weights, tf.multiply(tf.cast(tf.greater(float32_weights, binary_weights_bd), tf.float32), tf.constant(1, tf.float32)))
    
    return binary_weights
	
def binarize_biases(
    float32_biases, 
    binary_biases_bd
    ):
    
    binary_biases = tf.multiply(tf.cast(tf.less_equal(float32_biases, binary_biases_bd[0]), tf.float32), tf.constant(-1, tf.float32))
    binary_biases = tf.add(binary_biases, tf.multiply(tf.cast(tf.greater(float32_biases, binary_biases_bd[1]), tf.float32), tf.constant( 1, tf.float32)))
    
    return binary_biases

def quantize_activation(
    float32_net
    ):
    
    tf.add_to_collection("float32_net", float32_net)
    
    m = tf.get_variable("manssstissa", dtype=tf.float32, initializer=tf.constant([7], tf.float32))
    f = tf.get_variable("fraction"   , dtype=tf.float32, initializer=tf.constant([0], tf.float32))
    
    tf.add_to_collection("mantissa"  , m)
    tf.add_to_collection("fraction"  , f)
    
    upper_bd  =  tf.pow(tf.constant([2], tf.float32),  m)
    lower_bd  = -tf.pow(tf.constant([2], tf.float32),  m)
    step_size =  tf.pow(tf.constant([2], tf.float32), -f)
    
    step = tf.cast(tf.cast(tf.divide(float32_net, step_size), tf.int32), tf.float32)
    quantized_net = tf.multiply(step, step_size)
    quantized_net = tf.maximum(lower_bd, tf.minimum(upper_bd, quantized_net))
    
    return quantized_net

def quantize_Module(
    net, 
    is_quantized_activation,
    data_format
    ):
    
   
    
    #quantized_net = quantize_activation(net)
    if data_format == "NHWC":
        [batch, height, width, channel] = net.get_shape().as_list()
        boundary = tf.reduce_mean(net, axis = [0, 1, 2])
        boundary = tf.expand_dims(boundary, axis = 0)
        boundary = tf.expand_dims(boundary, axis = 0)
        boundary = tf.expand_dims(boundary, axis = 0)
        boundary = tf.tile(boundary, [1, height, width, 1])
        quantized_net = tf.subtract(net, boundary)
        quantized_net = tf.sign(quantized_net)
    elif data_format == "NCHW":
        [batch, channel, height, width] = net.get_shape().as_list()
        boundary = tf.reduce_mean(net, axis = [0, 2, 3])
        boundary = tf.expand_dims(boundary, axis = 0)
        boundary = tf.expand_dims(boundary, axis = 2)
        boundary = tf.expand_dims(boundary, axis = 3)
        boundary = tf.tile(boundary, [1, 1, height, width])
        quantized_net = tf.subtract(net, boundary)
        quantized_net = tf.sign(quantized_net)
        
    net = tf.cond(is_quantized_activation, lambda: quantized_net, lambda: net)
    
    tf.add_to_collection("is_quantized_activation", is_quantized_activation)
    tf.add_to_collection("quantized_net", net)
    
    return net

def shuffle_net(
    net,
    group,
    data_format
    ):
    
    if data_format == "NHWC":
        input_channel = net.get_shape().as_list()[-1]
    elif data_format == "NCHW":
        input_channel = net.get_shape().as_list()[1]
        
    group_size = input_channel / group
    
    if group_size > group:
        group_group_size = group_size / group
        group_group = pow(group, 2)
    else:
        group_group_size = 1
        group_group = input_channel
    
    for group_group_out in range(group_group):
        which_group = group_group_out % group
        which_group_in_group = group_group_out / group
        start_point = which_group * group_size + which_group_in_group * group_group_size
        #print("{}, {}, {}" .format(which_group, which_group_in_group, start_point))
        
        if data_format == "NHWC":
            net_in = net[:, :, :, start_point : start_point + group_group_size]
        elif data_format == "NCHW":
            net_in = net[:, start_point : start_point + group_group_size, :, :]
        #print(net_in.get_shape().as_list())
        if group_group_out == 0:
            net_tmp = net_in
        else:
            if data_format == "NHWC":
                net_tmp = tf.concat([net_tmp, net_in], axis = 3)
            elif data_format == "NCHW":
                net_tmp = tf.concat([net_tmp, net_in], axis = 1)
    #pdb.set_trace()
    return net_tmp

def combine_add(
    net,
    is_training,
    group,
    combine_number,
    is_batch_norm,
    Activation,
    initializer,
    data_format,
    scope
    ):
    with tf.variable_scope(scope):
        if data_format == "NHWC":
            input_channel = net.get_shape().as_list()[-1]
            net = tf.concat([net, net], axis = 3)
        elif data_format == "NCHW":
            input_channel = net.get_shape().as_list()[1]
            net = tf.concat([net, net], axis = 1)
        
        group_input_channel = int(input_channel / group)
        
        # Define each group in list
        net_tmp_list = []
        for g in range(group):
            if data_format == "NHWC":
                net_tmp_list.append(net[:, :, :, g*group_input_channel:(g+1)*group_input_channel])
            elif data_format == "NCHW":
                net_tmp_list.append(net[:, g*group_input_channel:(g+1)*group_input_channel, :, :])
        ## Define combine variables
        #combine_weights = tf.get_variable("combine_weights", [combine_number], tf.float32, initializer = initializer)
        # Combine
        for g in range(group):
            with tf.variable_scope('group%d'%(g)):
                # Define combine variables
                combine_weights = tf.get_variable("combine_weights", [combine_number], tf.float32, initializer = initializer)
                #combine_weights = tf.ones([combine_number], tf.float32)
                # add all up
                for gg in range(combine_number):
                    if gg == 0:
                        net_tmp = net_tmp_list[g] 
                        net_tmp = tf.multiply(net_tmp, combine_weights[0])
                    else:
                        net_tmp = tf.add(net_tmp, tf.multiply(net_tmp_list[(g+gg)%group], combine_weights[gg]))
                # Output
                # merge every group net together
                if g==0:
                    net_ = net_tmp
                else:
                    if data_format == "NHWC":
                        net_ = tf.concat([net_, net_tmp], axis=3)
                    elif data_format == "NCHW":
                        net_ = tf.concat([net_, net_tmp], axis=1)
        
        net = net_
        
        # Batch Normalization
        if is_batch_norm == True:
            ## Show the model
            if SHOW_MODEL:
                print("-> Batch_Norm")
            net = batch_norm(net, is_training, data_format, trainable)
        
        # Activation
        if Activation == 'ReLU':
            ## Show the model
            if SHOW_MODEL:
                print("-> ReLU")
            net = tf.nn.relu(net)
        elif Activation == 'Sigmoid':
            ## Show the model
            if SHOW_MODEL:
                print("-> Sigmoid")
            net = tf.nn.sigmoid(net)
        elif Activation == 'CReLU':
            ## Show the model
            if SHOW_MODEL:
                print("-> CReLU")
            net = CReLU(net, initializer = initializer)
        else:
            net = net

    return net
    
def combine_conv(
    net,
    is_training,
    group,
    combine_number,
    is_batch_norm,
    Activation,
    initializer,
    data_format,
    scope
    ):
    with tf.variable_scope(scope):
        if data_format == "NHWC":
            input_channel = net.get_shape().as_list()[-1]
            net = tf.concat([net, net], axis = 3)
        elif data_format == "NCHW":
            input_channel = net.get_shape().as_list()[1]
            net = tf.concat([net, net], axis = 1)
        
        group_input_channel = int(input_channel / group)
        group_output_channel = int(input_channel / group)
        
        # Define combine variables
        combine_weights_input_channel = group_input_channel * combine_number
        combine_weights_output_channel = group_output_channel
        combine_weights = tf.get_variable(
            name        = "combine_weights", 
            shape       = [1, 1, combine_weights_input_channel, combine_weights_output_channel], 
            dtype       = tf.float32, 
            initializer = initializer)
        # Combine
        for g in range(group):
            with tf.variable_scope('group%d'%(g)):
                # input
                start = g * group_input_channel
                end = start + combine_weights_input_channel
                if data_format == "NHWC":
                    net_tmp = net[:, :, :, start:end]
                elif data_format == "NCHW":
                    net_tmp = net[:, start:end, :, :]
                # 1x1 conv
                net_tmp = conv2d_fixed_padding(
                    inputs      = net_tmp, 
                    filters     = combine_weights, 
                    kernel_size = 1, 
                    stride      = 1, 
                    data_format = data_format)
                # Output
                # merge every group net together
                if g==0:
                    net_ = net_tmp
                else:
                    if data_format == "NHWC":
                        net_ = tf.concat([net_, net_tmp], axis=3)
                    elif data_format == "NCHW":
                        net_ = tf.concat([net_, net_tmp], axis=1)
    
        net = net_
        
        # Batch Normalization
        if is_batch_norm == True:
            ## Show the model
            if SHOW_MODEL:
                print("-> Batch_Norm")
            net = batch_norm(net, is_training, data_format, trainable)
        
        # Activation
        if Activation == 'ReLU':
            ## Show the model
            if SHOW_MODEL:
                print("-> ReLU")
            net = tf.nn.relu(net)
        elif Activation == 'Sigmoid':
            ## Show the model
            if SHOW_MODEL:
                print("-> Sigmoid")
            net = tf.nn.sigmoid(net)
        elif Activation == 'CReLU':
            ## Show the model
            if SHOW_MODEL:
                print("-> CReLU")
            net = CReLU(net, initializer = initializer)
        else:
            net = net
    
    return net
    
def conv2D_Variable(
    kernel_size,
    input_channel,
    output_channel,
    trainable,
    initializer,
    is_training,
    is_add_biases,
    is_ternary,
    IS_TERNARY,
    is_binary,
    IS_BINARY,
    IS_DIVERSITY,
    fast_mode,
    complexity_mode,
    rebuilding_now,
    is_depthwise,
    data_format
    ):
    
    ## Weights & Biases
    if is_depthwise:
        float32_weights = tf.get_variable(
            name        = "float32_weights", 
            shape       = [kernel_size, kernel_size, input_channel, 1], 
            dtype       = tf.float32,
            trainable   = trainable and not IS_DIVERSITY,
            initializer = initializer)
        if is_add_biases:
            float32_biases = tf.get_variable(
                name        = "float32_biases", 
                shape       = [input_channel], 
                dtype       = tf.float32, 
                trainable   = trainable and not IS_DIVERSITY,
                initializer = initializer)
        else:
            float32_biases = None
    else:
        if data_format == "NHWC":
            float32_weights = tf.get_variable(
                name        = "float32_weights", 
                shape       = [kernel_size, kernel_size, input_channel, output_channel], 
                dtype       = tf.float32, 
                trainable   = trainable and not IS_DIVERSITY,
                initializer = initializer)
        elif data_format == "NCHW":
            float32_weights = tf.get_variable(
                name        = "float32_weights", 
                shape       = [kernel_size, kernel_size, input_channel, output_channel], 
                dtype       = tf.float32, 
                trainable   = trainable and not IS_DIVERSITY,
                initializer = initializer)
        if is_add_biases:
            float32_biases = tf.get_variable(
                name        = "float32_biases", 
                shape       = [output_channel], 
                dtype       = tf.float32, 
                trainable   = trainable and not IS_DIVERSITY,
                initializer = initializer)
        else:
            float32_biases = None

    tf.add_to_collection("float32_weights", float32_weights)
    if is_add_biases:
        tf.add_to_collection("float32_biases" , float32_biases)
    
    tf.add_to_collection("float32_params" , float32_weights)
    if is_add_biases:
        tf.add_to_collection("float32_params" , float32_biases)
    
    ## Pruning Mask (1 means exist; 0 means being pruned)
    float32_weights_mask = tf.Variable(
        initial_value = tf.ones_like(tensor = float32_weights, dtype = tf.float32),
        trainable     = False,
        name          = "float32_weights_mask",
        dtype         = tf.float32)
    tf.add_to_collection("float32_weights_mask", float32_weights_mask)

    if is_add_biases:    
        float32_biases_mask = tf.Variable(
            initial_value = tf.ones_like(tensor = float32_biases, dtype = tf.float32),
            trainable     = False,
            name          = "float32_biases_mask",
            dtype         = tf.float32)
        tf.add_to_collection("float32_biases_mask", float32_biases_mask)
        
    ## is_train Mask (1 means trainable; 0 means using Constant Weights)
    is_train_float32_weights_mask = tf.Variable(
        initial_value = tf.ones_like(tensor = float32_weights, dtype = tf.float32),
        trainable     = False,
        name          = "is_train_float32_weights_mask",
        dtype         = tf.float32)
    tf.add_to_collection("is_train_float32_weights_mask", is_train_float32_weights_mask)
    
    if is_add_biases:    
        is_train_float32_biases_mask = tf.Variable(
            initial_value = tf.ones_like(tensor = float32_biases, dtype = tf.float32),
            trainable     = False,
            name          = "is_train_float32_biases_mask",
            dtype         = tf.float32)
        tf.add_to_collection("is_train_float32_biases_mask", is_train_float32_biases_mask)
    
    ## Constant Weights
    constant_float32_weights = tf.Variable(
        initial_value = tf.zeros_like(tensor = float32_weights, dtype = tf.float32),
        trainable     = False,
        name          = "constant_float32_weights",
        dtype         = tf.float32)
    tf.add_to_collection("constant_float32_weights", constant_float32_weights)
    
    if is_add_biases:    
        constant_float32_biases = tf.Variable(
            initial_value = tf.zeros_like(tensor = float32_biases, dtype = tf.float32),
            trainable     = False,
            name          = "constant_float32_biases",
            dtype         = tf.float32)
        tf.add_to_collection("constant_float32_biases", constant_float32_biases)
    
    #--------------------------------#
    #    DIVERSITY Rebuilding conv   #
    #--------------------------------#
    ## Complexity Model
    cond0 = fast_mode == '4'
    cond1 = fast_mode == '5'
    cond2 = fast_mode == '6'
    cond3 = fast_mode == '7'
    cond4 = fast_mode == '8'
    cond5 = fast_mode == '9'
    if cond0 or cond1 or cond2 or cond3 or cond4 or cond5:
        complexity_mode_mask = tf.Variable(
            initial_value = tf.zeros_like(tensor = float32_weights, dtype = tf.float32),
            trainable     = False,
            name          = "complexity_mode_mask",
            dtype         = tf.float32)
        tf.add_to_collection("complexity_mode_mask", complexity_mode_mask)
        
        if cond3 or cond4 or cond5:
            # float32_weights_mask
            float32_weights_mask_value = tf.cast(
                tf.greater_equal(
                    x = complexity_mode_mask,
                    y = tf.subtract(complexity_mode, tf.constant(1, tf.float32))),
                dtype = tf.float32)
            assign_float32_weights_mask = tf.assign(
                float32_weights_mask, 
                float32_weights_mask_value)
            
            # is_train_float32_weights
            is_train_float32_weights_mask_value = tf.cast(
                tf.less(
                    x = complexity_mode_mask,
                    y = complexity_mode),
                dtype = tf.float32)
            assign_is_train_float32_weights_mask = tf.assign(
                is_train_float32_weights_mask,
                is_train_float32_weights_mask_value)
                
            tf.add_to_collection("assign_var_list", assign_float32_weights_mask)
            tf.add_to_collection("assign_var_list", assign_is_train_float32_weights_mask)
            
            if rebuilding_now:
                trainable_weights = tf.multiply(tf.multiply(float32_weights, float32_weights_mask), is_train_float32_weights_mask)
                constant_weights = tf.multiply(constant_float32_weights, tf.subtract(1., is_train_float32_weights_mask))
                weights = tf.add(trainable_weights, constant_weights)
                
                if is_add_biases:
                    trainable_biases = tf.multiply(tf.multiply(float32_biases, float32_biases_mask), is_train_float32_biases_mask)
                    constant_biases = tf.multiply(constant_float32_biases, tf.subtract(1., is_train_float32_biases_mask))
                    biases = tf.add(trainable_biases, constant_biases)
                
                if is_add_biases:
                    return weights, biases, None, None
                else:
                    return weights, None, None, None
            else:
                constant_weights = tf.multiply(constant_float32_weights, tf.subtract(1., is_train_float32_weights_mask))
                if is_add_biases:
                    constant_biases = tf.multiply(constant_float32_biases, tf.subtract(1., is_train_float32_biases_mask))
                
                if is_add_biases:
                    return constant_weights, constant_biases, None, None
                else:
                    return constant_weights, None, None, None
            
    #-------------------------#
    #    DIVERSITY 2x2 conv   #
    #-------------------------#
    if IS_DIVERSITY and fast_mode == '0':
        kernel_size_fast = 2
        # float32 Variable
        if is_depthwise:
            float32_weights_fast = tf.get_variable(
                name        = "float32_weights_fast", 
                shape       = [kernel_size_fast, kernel_size_fast, input_channel, 1], 
                dtype       = tf.float32,
                trainable   = trainable,
                initializer = initializer)
            if is_add_biases:
                float32_biases_fast = tf.get_variable(
                    name        = "float32_biases_fast", 
                    shape       = [input_channel], 
                    dtype       = tf.float32, 
                    trainable   = trainable,
                    initializer = initializer)
            else:
                float32_biases_fast = None
        else:
            if data_format == "NHWC":
                float32_weights_fast = tf.get_variable(
                    name        = "float32_weights_fast", 
                    shape       = [kernel_size_fast, kernel_size_fast, input_channel, output_channel], 
                    dtype       = tf.float32, 
                    trainable   = trainable,
                    initializer = initializer)
            elif data_format == "NCHW":
                float32_weights_fast = tf.get_variable(
                    name        = "float32_weights_fast", 
                    shape       = [kernel_size_fast, kernel_size_fast, input_channel, output_channel], 
                    dtype       = tf.float32, 
                    trainable   = trainable,
                    initializer = initializer)
            if is_add_biases:
                float32_biases_fast = tf.get_variable(
                    name        = "float32_biases_fast", 
                    shape       = [output_channel], 
                    dtype       = tf.float32, 
                    trainable   = trainable,
                    initializer = initializer)
            else:
                float32_biases_fast = None
    
        tf.add_to_collection("float32_weights_fast", float32_weights_fast)
        if is_add_biases:
            tf.add_to_collection("float32_biases_fast" , float32_biases_fast)
        
        tf.add_to_collection("float32_params" , float32_weights_fast)
        if is_add_biases:
            tf.add_to_collection("float32_params" , float32_biases_fast)
        
        # Pruning Mask
        float32_weights_mask_fast = tf.Variable(
            initial_value = tf.ones_like(tensor = float32_weights_fast, dtype = tf.float32),
            trainable     = False,
            name          = "float32_weights_mask_fast",
            dtype         = tf.float32)
        tf.add_to_collection("float32_weights_mask_fast", float32_weights_mask_fast)
    
        if is_add_biases:    
            float32_biases_mask_fast = tf.Variable(
                initial_value = tf.ones_like(tensor = float32_biases_fast, dtype = tf.float32),
                trainable     = False,
                name          = "float32_biases_mask_fast",
                dtype         = tf.float32)
            tf.add_to_collection("float32_biases_mask_fast", float32_biases_mask_fast)
            
        # is_train Mask
        is_train_float32_weights_mask_fast = tf.Variable(
            initial_value = tf.ones_like(tensor = float32_weights_fast, dtype = tf.float32),
            trainable     = False,
            name          = "is_train_float32_weights_mask_fast",
            dtype         = tf.float32)
        tf.add_to_collection("is_train_float32_weights_mask_fast", is_train_float32_weights_mask_fast)
        
        if is_add_biases:    
            is_train_float32_biases_mask_fast = tf.Variable(
                initial_value = tf.ones_like(tensor = float32_biases_fast, dtype = tf.float32),
                trainable     = False,
                name          = "is_train_float32_biases_mask_fast",
                dtype         = tf.float32)
            tf.add_to_collection("is_train_float32_biases_mask_fast", is_train_float32_biases_mask_fast)
        
        # Constant Weights
        constant_float32_weights_fast = tf.Variable(
            initial_value = tf.zeros_like(tensor = float32_weights_fast, dtype = tf.float32),
            trainable     = False,
            name          = "constant_float32_weights_fast",
            dtype         = tf.float32)
        tf.add_to_collection("constant_float32_weights_fast", constant_float32_weights_fast)
        
        if is_add_biases:    
            constant_float32_biases_fast = tf.Variable(
                initial_value = tf.zeros_like(tensor = float32_biases_fast, dtype = tf.float32),
                trainable     = False,
                name          = "constant_float32_biases_fast",
                dtype         = tf.float32)
            tf.add_to_collection("constant_float32_biases_fast", constant_float32_biases_fast)
        
        weights_ = tf.add(tf.multiply(tf.multiply(float32_weights, float32_weights_mask), is_train_float32_weights_mask), constant_float32_weights)
        if is_add_biases:
            biases_ = tf.add(tf.multiply(tf.multiply(float32_biases, float32_biases_mask), is_train_float32_biases_mask), constant_float32_biases)
        
        weights_fast = tf.add(tf.multiply(tf.multiply(float32_weights_fast, float32_weights_mask_fast), is_train_float32_weights_mask_fast), constant_float32_weights_fast)
        if is_add_biases:
            biases_fast = tf.add(tf.multiply(tf.multiply(float32_biases_fast, float32_biases_mask_fast), is_train_float32_biases_mask_fast), constant_float32_biases_fast)
        
        if is_add_biases:
            return weights_, biases_, weights_fast, biases_fast
        else:
            return weights_, None, weights_fast, None
    #-------------------------#
    #    Ternary Variables    #
    #-------------------------#
    elif IS_TERNARY:
        # Ternary boundary of weights and biases 
        ternary_weights_bd = tf.get_variable(
            name        = "ternary_weights_bd", 
            shape       = [2], 
            dtype       = tf.float32, 
            trainable   = False,
            initializer = initializer)
           
        if is_add_biases:
            ternary_biases_bd  = tf.get_variable(
                name        = "ternary_biases_bd" , 
                shape       = [2], 
                dtype       = tf.float32, 
                trainable   = False,
                initializer = initializer)
        
        """
        ternary_weights_bd = tf.Variable(
            initial_value = [-0.7 * tf.reduce_mean(float32_weights), 0.7 * tf.reduce_mean(float32_weights)],
            trainable     = False,
            name          = "ternary_weights_bd",
            dtype         = tf.float32)
            
        if is_add_biases:
            ternary_biases_bd = tf.Variable(
                initial_value = [-0.7 * tf.reduce_mean(float32_biases), 0.7 * tf.reduce_mean(float32_biases)],
                trainable     = False,
                name          = "ternary_biases_bd",
                dtype         = tf.float32)
        """        
        tf.add_to_collection("ternary_weights_bd", ternary_weights_bd)
        if is_add_biases:
            tf.add_to_collection("ternary_biases_bd" , ternary_biases_bd)
        
        # Choose Precision
        weights_tmp = tf.cond(is_ternary, 
                                lambda: ternarize_weights(float32_weights, ternary_weights_bd), 
                                lambda: float32_weights)
        if is_add_biases:
            biases_tmp = tf.cond(is_ternary, 
                                lambda: ternarize_biases(float32_biases , ternary_biases_bd),
                                lambda: float32_biases)
    
        if is_depthwise:
            final_weights = tf.get_variable(
                name        = "final_weights", 
                shape       = [kernel_size, kernel_size, input_channel, 1], 
                dtype       = tf.float32, 
                trainable   = True, 
                initializer = initializer)
            if is_add_biases:
                final_biases = tf.get_variable(
                    name        = "final_biases", 
                    shape       = [input_channel], 
                    dtype       = tf.float32, 
                    trainable   = True,
                    initializer = initializer)
            else:
                final_biases = None
        else:
            final_weights = tf.get_variable(
                name        = "final_weights", 
                shape       = [kernel_size, kernel_size, input_channel, output_channel], 
                dtype       = tf.float32,
                trainable   = True,
                initializer = initializer)
            if is_add_biases:
                final_biases = tf.get_variable("final_biases" , [output_channel], tf.float32, initializer = initializer)
            else:
                final_biases = None
        
        # Assign ternarized weights to final weights
        assign_final_weights = tf.assign(final_weights, weights_tmp)
        if is_add_biases:
            assign_final_biases = tf.assign(final_biases , biases_tmp)
        tf.add_to_collection("assign_var_list", assign_final_weights)
        if is_add_biases:
            tf.add_to_collection("assign_var_list", assign_final_biases)
        
        # Multiply with mask
        weights_ = tf.add(tf.multiply(tf.multiply(final_weights, float32_weights_mask), is_train_float32_weights_mask), constant_float32_weights)
        if is_add_biases:
            biases_ = tf.add(tf.multiply(tf.multiply(final_biases, float32_biases_mask), is_train_float32_biases_mask), constant_float32_biases)

        if is_add_biases:
            return weights_, biases_, None, None
        else:
            return weights_, None, None, None
    #------------------------#
    #    Binary Variables    #
    #------------------------#
    elif IS_BINARY:
        # Assign clip weights to float32_weights
        min = tf.constant(-1, tf.float32)
        max = tf.constant(1, tf.float32)
        clip_weights = tf.clip_by_value(float32_weights, min, max)
        assign_clip_weights = tf.assign(float32_weights, clip_weights)
        tf.add_to_collection('clip_weights', clip_weights)
        tf.add_to_collection("assign_var_list", assign_clip_weights)

        # Binary Weights
        graph = tf.get_default_graph()
        with graph.gradient_override_map({"Sign": "Identity"}):
            weights_ = float32_weights
            weights_ = tf.sign(weights_)
        
        tf.add_to_collection('binary_weights', weights_)
        
        # Binary Scale
        ## Show the model
        if SHOW_MODEL:
            print("-> Binary")
        with tf.variable_scope('Binary_Scale'):
            binary_scale = tf.get_variable("binary_scale", [1], tf.float32, initializer=initializer)
            tf.add_to_collection("binary_scale", binary_scale)
            weights_ = tf.multiply(weights_, binary_scale)
        
        weights_ = tf.add(tf.multiply(tf.multiply(weights_, float32_weights_mask), is_train_float32_weights_mask), constant_float32_weights)
        if is_add_biases:
            biases_ = tf.add(tf.multiply(tf.multiply(float32_biases, float32_biases_mask), is_train_float32_biases_mask), constant_float32_biases)
        
        if is_add_biases:
            return weights_, biases_, None, None
        else:
            return weights_, None, None, None
    else:
        trainable_weights = tf.multiply(tf.multiply(float32_weights, float32_weights_mask), is_train_float32_weights_mask)
        constant_weights = tf.multiply(constant_float32_weights, tf.subtract(1., is_train_float32_weights_mask))
        weights_ = tf.add(trainable_weights, constant_weights)
        
        if is_add_biases:
            trainable_biases = tf.multiply(tf.multiply(float32_biases, float32_biases_mask), is_train_float32_biases_mask)
            constant_biases = tf.multiply(constant_float32_biases, tf.subtract(1., is_train_float32_biases_mask))
            biases_ = tf.add(trainable_biases, constant_biases)
            
        if is_add_biases:
            return weights_, biases_, None, None
        else:
            return weights_, None, None, None

def fixed_padding(
    inputs, 
    kernel_size,
    data_format
    ):

    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    
    if data_format == "NHWC":
        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
    elif data_format == "NCHW":
        padded_inputs = tf.pad(inputs, [[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]])
        
    return padded_inputs

def conv2d_fixed_padding(
    inputs, 
    filters, 
    kernel_size, 
    stride,
    data_format
    ):
    
    if stride > 1:
        inputs = fixed_padding(inputs, kernel_size, data_format)

    outputs = tf.nn.conv2d( 
                input       = inputs, 
                filter      = filters, 
                strides     = [1, stride, stride, 1] if data_format == "NHWC" else [1, 1, stride, stride], 
                padding     = ('SAME' if stride == 1 else 'VALID'),
                data_format = data_format)

    #outputs = tf.layers.conv2d(
    #  inputs             = inputs, 
    #  filters            = filters, 
    #  kernel_size        = kernel_size, 
    #  strides            = stride,
    #  padding            = ('SAME' if stride == 1 else 'VALID'), 
    #  use_bias           = False,
    #  kernel_initializer = tf.variance_scaling_initializer())            

    return outputs
        
def conv2D_Module( 
	net, kernel_size, stride, output_channel, rate, group,
    trainable               ,
	initializer             ,
    is_training             ,
    is_add_biases           ,
	is_batch_norm           ,
	is_dilated              ,
	is_depthwise            ,
	is_ternary              ,
    is_binary               ,
	is_quantized_activation ,
	IS_TERNARY              ,
    IS_BINARY               ,
	IS_QUANTIZED_ACTIVATION ,
    IS_DIVERSITY            ,
    is_fast_mode            ,
    fast_mode               ,
    complexity_mode         ,
    rebuilding_now          ,
    complexity_mode_now     ,
	Activation              ,
	padding                 ,
    data_format             ,
	Analysis                ,
	scope                   
	):
    with tf.variable_scope(scope):
        if data_format == "NHWC":
            input_channel = net.get_shape().as_list()[-1]
        elif data_format == "NCHW":
            input_channel = net.get_shape().as_list()[1]
        
        #if not is_depthwise:
        #    group=1
        
        # -- Analyzer --
        utils.Analyzer( Analysis, 
                        net, 
                        type                    = 'CONV', 
                        kernel_shape            = [kernel_size,kernel_size, input_channel, output_channel], 
                        stride                  = stride, 
                        group                   = group, 
                        is_depthwise            = is_depthwise,
                        IS_TERNARY              = IS_TERNARY,
                        IS_BINARY               = IS_BINARY,
                        IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
                        padding                 = padding, 
                        name                    = 'Conv')
        
        group_input_channel = int(input_channel / group)
        group_output_channel = int(output_channel / group)
        
        net_tmp_list = []
        computation = 0
        #-------------------#
        #    Convolution    #
        #-------------------#
        for g in range(group):
            ## Show the model
            if SHOW_MODEL:
                print("\033[0;36mgroup%d\033[0m"%(g))
            with tf.variable_scope('group%d'%(g)):
                # Inputs
                if data_format == "NHWC":
                    net_tmp = net[:, :, :, g*group_input_channel : (g+1)*group_input_channel]
                elif data_format == "NCHW":
                    net_tmp = net[:, g*group_input_channel : (g+1)*group_input_channel, :, :]
                
                # Variable define
                weights, biases, weights_fast, biases_fast = conv2D_Variable(
                    kernel_size     = kernel_size,
                    input_channel   = group_input_channel,
                    output_channel  = group_output_channel,
                    trainable       = trainable,
                    initializer     = initializer,
                    is_training     = is_training,
                    is_add_biases   = is_add_biases,
                    is_ternary      = is_ternary,
                    IS_TERNARY      = IS_TERNARY,
                    is_binary       = is_binary,
                    IS_BINARY       = IS_BINARY,
                    IS_DIVERSITY    = IS_DIVERSITY,
                    fast_mode       = fast_mode,
                    complexity_mode = complexity_mode,
                    rebuilding_now  = rebuilding_now,
                    is_depthwise    = is_depthwise,
                    data_format     = data_format)
                
                # Convolution
                if is_dilated:
                    if is_depthwise:
                        ## Show the model
                        if SHOW_MODEL:
                            print("-> Dilated-Depthwise Conv, Weights:{}, stride:{}" .format(weights.get_shape().as_list(), stride))
                        if IS_DIVERSITY and fast_mode != '3':
                            net_fast_tmp = tf.nn.depthwise_conv2d( 
                                input       = net_tmp, 
                                filter      = weights_fast, 
                                strides     = [1, stride, stride, 1] if data_format == "NHWC" else [1, 1, stride, stride], 
                                padding     = padding, 
                                rate        = [rate, rate],
                                data_format = data_format)
                        net_tmp = tf.nn.depthwise_conv2d( 
                            input       = net_tmp, 
                            filter      = weights, 
                            strides     = [1, stride, stride, 1] if data_format == "NHWC" else [1, 1, stride, stride], 
                            padding     = padding, 
                            rate        = [rate, rate],
                            data_format = data_format)
                    else:
                        ## Show the model
                        if SHOW_MODEL:
                            print("-> Dilated Conv, Weights:{}, stride:{}" .format(weights.get_shape().as_list(), stride))
                        if IS_DIVERSITY and fast_mode != '3':
                            net_fast_tmp = tf.nn.atrous_conv2d( 
                                value       = net_tmp, 
                                filters     = weights_fast, 
                                rate        = rate, 
                                padding     = padding)
                        net_tmp = tf.nn.atrous_conv2d( 
                            value       = net_tmp, 
                            filters     = weights, 
                            rate        = rate, 
                            padding     = padding)
                else:
                    if is_depthwise:
                        ## Show the model
                        if SHOW_MODEL:
                            print("-> Depthwise Conv, Weights:{}, stride:{}" .format(weights.get_shape().as_list(), stride))
                        if IS_DIVERSITY and fast_mode != '3':
                            net_fast_tmp = tf.nn.depthwise_conv2d( 
                                input       = net_tmp, 
                                filter      = weights_fast, 
                                strides     = [1, stride, stride, 1] if data_format == "NHWC" else [1, 1, stride, stride], 
                                padding     = padding,
                                data_format = data_format)	
                        net_tmp = tf.nn.depthwise_conv2d( 
                            input       = net_tmp, 
                            filter      = weights, 
                            strides     = [1, stride, stride, 1] if data_format == "NHWC" else [1, 1, stride, stride], 
                            padding     = padding,
                            data_format = data_format)	
                    else:
                        ## Show the model
                        if SHOW_MODEL:
                            print("-> Conv, Weights:{}, stride:{}" .format(weights.get_shape().as_list(), stride))
                        if IS_DIVERSITY and fast_mode != '3':
                            net_fast_tmp = conv2d_fixed_padding(
                                inputs      = net_tmp, 
                                filters     = weights_fast,  
                                kernel_size = 2, 
                                stride      = stride, 
                                data_format = data_format)
                        net_tmp = conv2d_fixed_padding(
                            inputs      = net_tmp, 
                            filters     = weights,  
                            kernel_size = kernel_size, 
                            stride      = stride, 
                            data_format = data_format)
                       
                tf.add_to_collection("conv_outputs", net_tmp)
                
                # Add bias
                if is_add_biases:
                    ## Show the model
                    if SHOW_MODEL:
                        print("-> Add Biases, Biases:{}".format(biases.get_shape().as_list()))
                    if IS_DIVERSITY and fast_mode != '3':
                        net_fast_tmp = tf.nn.bias_add(net_fast_tmp, biases_fast, data_format)
                    net_tmp = tf.nn.bias_add(net_tmp, biases, data_format)
                
                ## For checking the final weights/biases is ternary/binary or not
                tf.add_to_collection("weights", weights)
                tf.add_to_collection("weights_fast", weights_fast)
                if is_add_biases:
                    tf.add_to_collection("biases", biases)
                    tf.add_to_collection("biases_fast", biases_fast)
                # List the net_tmp
                """
                net_tmp_list.append(net_tmp)
                """
                # Output
                # merge every group net together
                if g==0:
                    if IS_DIVERSITY and fast_mode != '3':
                        net_fast_ = net_fast_tmp
                    net_ = net_tmp
                else:
                    if data_format == "NHWC":
                        if IS_DIVERSITY and fast_mode != '3':
                            net_fast_ = tf.concat([net_fast_, net_fast_tmp], axis=3)
                        net_ = tf.concat([net_, net_tmp], axis=3)
                    elif data_format == "NCHW":
                        if IS_DIVERSITY and fast_mode != '3':
                            net_fast_ = tf.concat([net_fast_, net_fast_tmp], axis=1)
                        net_ = tf.concat([net_, net_tmp], axis=1)

        #-----------------#
        #    Operation    #
        #-----------------#
        if IS_DIVERSITY and fast_mode != '3':
            net_fast = net_fast_
        net = net_
        
        # Ternary Scale
        if IS_TERNARY:
            ## Show the model
            if SHOW_MODEL:
                print("-> Ternary")
            if IS_DIVERSITY and fast_mode != '3':
                with tf.variable_scope('Ternary_Scalse_fast'):
                    ternary_scale_fast = tf.get_variable("ternary_scale_fast", [1], tf.float32, initializer=initializer)
                    tf.add_to_collection("ternary_scale_fast", ternary_scale_fast)
                    net_fast = tf.multiply(net_fast, ternary_scale_fast)
            with tf.variable_scope('Ternary_Scalse'):
                ternary_scale = tf.get_variable("ternary_scale", [1], tf.float32, initializer=initializer)
                tf.add_to_collection("ternary_scale", ternary_scale)
                net = tf.multiply(net, ternary_scale)
        
        # (Diversity) Choose mode
        if IS_DIVERSITY and fast_mode != '3':
            net = tf.cond(is_fast_mode,
                            lambda: net_fast,
                            lambda: net)
        
        # Batch Normalization
        if is_batch_norm == True:
            ## Show the model
            if SHOW_MODEL:
                print("-> Batch_Norm")
            """
            if IS_DIVERSITY and fast_mode != '3':
                net = batch_norm(net, tf.logical_and(is_training, tf.constant(False, tf.bool)), data_format, False)
                net_fast = batch_norm(net_fast, is_training, data_format, trainable)
            else:
                net = batch_norm(net, is_training, data_format, trainable)
            """
            
            cond0 = fast_mode == '4'
            cond1 = fast_mode == '5'
            cond2 = fast_mode == '6'
            cond3 = fast_mode == '7'
            cond4 = fast_mode == '8'
            cond5 = fast_mode == '9'
            
            # Normal
            if cond0 or cond1 or cond2:
                is_training_ = is_training #tf.logical_and(is_training, tf.equal(complexity_mode, tf.constant(0, tf.float32)))
                trainable_ = fast_mode!='0' and trainable
            elif (cond3 or cond4 or cond5) and rebuilding_now:
                is_training_ = is_training
                trainable_ = trainable and complexity_mode_now == 0
            else:
                is_training_ = is_training
                trainable_ = fast_mode!='0' and trainable
            
            net_ = batch_norm(net, is_training_, data_format, trainable_)
            """
            # Pruning / Rebuilding
            if cond0 or cond1 or cond2 or cond3 or cond4 or cond5:
                with tf.variable_scope('complexity_1'):
                    # is_training
                    if cond0 or cond1 or cond2:
                        is_training_ = tf.logical_and(is_training, tf.equal(complexity_mode, tf.constant(0, tf.float32)))
                        trainable_ = trainable and complexity_mode_now == 1
                    elif (cond3 or cond4 or cond5) and rebuilding_now:
                        is_training_ = tf.logical_and(is_training, tf.equal(complexity_mode, tf.constant(2, tf.float32)))
                        trainable_ = trainable and complexity_mode_now == 1
                    else:
                        is_training_ = is_training
                        trainable_ = trainable
                    net_1 = batch_norm(net, is_training_, data_format, trainable_)
                with tf.variable_scope('complexity_2'):
                    # is_training
                    if cond0 or cond1 or cond2:
                        is_training_ = tf.logical_and(is_training, tf.equal(complexity_mode, tf.constant(1, tf.float32)))
                        trainable_ = trainable and complexity_mode_now == 2
                    elif (cond3 or cond4 or cond5) and rebuilding_now:
                        is_training_ = tf.logical_and(is_training, tf.equal(complexity_mode, tf.constant(3, tf.float32)))
                        trainable_ = trainable and complexity_mode_now == 2
                    else:
                        is_training_ = is_training
                        trainable_ = trainable
                    net_2 = batch_norm(net, is_training_, data_format, trainable_)
                with tf.variable_scope('complexity_3'):
                    # is_training
                    if cond0 or cond1 or cond2:
                        is_training_ = tf.logical_and(is_training, tf.equal(complexity_mode, tf.constant(2, tf.float32)))
                        trainable_ = trainable and complexity_mode_now == 3
                    elif (cond3 or cond4 or cond5) and rebuilding_now:
                        is_training_ = tf.logical_and(is_training, tf.equal(complexity_mode, tf.constant(4, tf.float32)))
                        trainable_ = trainable and complexity_mode_now == 3
                    else:
                        is_training_ = is_training
                        trainable_ = trainable
                    net_3 = batch_norm(net, is_training_, data_format, trainable_)
                with tf.variable_scope('complexity_4'):
                    # is_training
                    if cond0 or cond1 or cond2:
                        is_training_ = tf.logical_and(is_training, tf.equal(complexity_mode, tf.constant(3, tf.float32)))
                        trainable_ = trainable and complexity_mode_now == 4
                    elif (cond3 or cond4 or cond5) and rebuilding_now:
                        is_training_ = tf.logical_and(is_training, tf.equal(complexity_mode, tf.constant(5, tf.float32)))
                        trainable_ = trainable and complexity_mode_now == 4
                    else:
                        is_training_ = is_training
                        trainable_ = trainable
                    net_4 = batch_norm(net, is_training_, data_format, trainable_)
                with tf.variable_scope('complexity_5'):
                    # is_training
                    if cond0 or cond1 or cond2:
                        is_training_ = tf.logical_and(is_training, tf.equal(complexity_mode, tf.constant(4, tf.float32)))
                        trainable_ = trainable and complexity_mode_now == 5
                    elif (cond3 or cond4 or cond5) and rebuilding_now:
                        is_training_ = tf.logical_and(is_training, tf.equal(complexity_mode, tf.constant(6, tf.float32)))
                        trainable_ = trainable and complexity_mode_now == 5
                    else:
                        is_training_ = is_training
                        trainable_ = trainable
                    net_5 = batch_norm(net, is_training_, data_format, trainable_)
                with tf.variable_scope('complexity_6'):
                    # is_training
                    if cond0 or cond1 or cond2:
                        is_training_ = tf.logical_and(is_training, tf.equal(complexity_mode, tf.constant(5, tf.float32)))
                        trainable_ = trainable and complexity_mode_now == 6
                    elif (cond3 or cond4 or cond5) and rebuilding_now:
                        is_training_ = tf.logical_and(is_training, tf.equal(complexity_mode, tf.constant(7, tf.float32)))
                        trainable_ = trainable and complexity_mode_now == 6
                    else:
                        is_training_ = is_training
                        trainable_ = trainable
                    net_6 = batch_norm(net, is_training_, data_format, trainable_)
                
                if cond0 or cond1 or cond2:
                    net = tf.case(
                        pred_fn_pairs = {
                            tf.equal(complexity_mode, tf.constant(0, tf.float32)): lambda: net_1,
                            tf.equal(complexity_mode, tf.constant(1, tf.float32)): lambda: net_2,
                            tf.equal(complexity_mode, tf.constant(2, tf.float32)): lambda: net_3,
                            tf.equal(complexity_mode, tf.constant(3, tf.float32)): lambda: net_4,
                            tf.equal(complexity_mode, tf.constant(4, tf.float32)): lambda: net_5,
                            tf.equal(complexity_mode, tf.constant(5, tf.float32)): lambda: net_6},
                        default = None, exclusive = True)
                                   
                if cond3 or cond4 or cond5:
                    if rebuilding_now:
                        net = tf.case(
                            pred_fn_pairs = {
                                tf.equal(complexity_mode, tf.constant(1, tf.float32)): lambda: net_,
                                tf.equal(complexity_mode, tf.constant(2, tf.float32)): lambda: net_1,
                                tf.equal(complexity_mode, tf.constant(3, tf.float32)): lambda: net_2,
                                tf.equal(complexity_mode, tf.constant(4, tf.float32)): lambda: net_3,
                                tf.equal(complexity_mode, tf.constant(5, tf.float32)): lambda: net_4,
                                tf.equal(complexity_mode, tf.constant(6, tf.float32)): lambda: net_5},
                            default = None, exclusive = True)
                    else:
                        net = tf.case(
                            pred_fn_pairs = {
                                tf.equal(complexity_mode, tf.constant(0, tf.float32)): lambda: net_,
                                tf.equal(complexity_mode, tf.constant(1, tf.float32)): lambda: net_1,
                                tf.equal(complexity_mode, tf.constant(2, tf.float32)): lambda: net_2,
                                tf.equal(complexity_mode, tf.constant(3, tf.float32)): lambda: net_3,
                                tf.equal(complexity_mode, tf.constant(4, tf.float32)): lambda: net_4,
                                tf.equal(complexity_mode, tf.constant(5, tf.float32)): lambda: net_5,
                                tf.equal(complexity_mode, tf.constant(6, tf.float32)): lambda: net_6},
                            default = None, exclusive = True)
            else:
                net = net_
            """
            net = net_
        # Activation
        if Activation == 'ReLU':
            ## Show the model
            if SHOW_MODEL:
                print("-> ReLU")
            net = tf.nn.relu(net)
        elif Activation == 'Sigmoid':
            ## Show the model
            if SHOW_MODEL:
                print("-> Sigmoid")
            net = tf.nn.sigmoid(net)
        elif Activation == 'CReLU':
            ## Show the model
            if SHOW_MODEL:
                print("-> CReLU")
            net = CReLU(net, initializer = initializer)
        elif Activation == 'HardTanh':
            ## Show the model
            if SHOW_MODEL:
                print("-> HardTanh")
            net = tf.clip_by_value(net, -1, 1)
        elif Activation == 'PReLU':
            ## Show the model
            if SHOW_MODEL:
                print("-> PReLU")
            net = tf.keras.layers.PReLU()(net)
        else:
            net = net
            
        # Activation Quantization
        if IS_QUANTIZED_ACTIVATION:
            ## Show the model
            if SHOW_MODEL:
                print("-> Quantized Activation ")
            net = quantize_Module(net, is_quantized_activation, data_format)

    return net

def shortcut_Module( 
    net,
    group                   ,
    destination             ,
    initializer             ,
    is_training             ,
    is_add_biases           ,
    is_projection_shortcut  ,
    shortcut_type           ,
    shortcut_connection     ,
    is_batch_norm           ,
    is_ternary              ,
    is_binary               ,
    is_quantized_activation ,
    IS_TERNARY              ,
    IS_BINARY               ,
    IS_QUANTIZED_ACTIVATION ,
    padding                 ,	
    data_format             ,
    Analysis                ,
    trainable               = True,
    IS_DIVERSITY            = False,
    is_fast_mode            = False,
    fast_mode               = None,
    complexity_mode         = None,
    rebuilding_now          = False,
    complexity_mode_now     = None
    ):
    
    if data_format == "NHWC":
        [batch,  input_height,  input_width,  input_channel] = net.get_shape().as_list()
        [batch, output_height, output_width, output_channel] = destination.get_shape().as_list()
    elif data_format == "NCHW":
        [batch,  input_channel,  input_height,  input_width] = net.get_shape().as_list()
        [batch, output_channel, output_height, output_width] = destination.get_shape().as_list()
        
    with tf.variable_scope("shortcut"):
        # Height & Width & Depth
        size_not_equal = input_height!=output_height or input_width!=output_width
        depth_not_equal = input_channel!=output_channel and shortcut_connection == "ADD"
        
        if size_not_equal or depth_not_equal or is_projection_shortcut:
            stride_height = int(input_height / output_height)
            stride_width  = int(input_width  / output_width )
            
            if shortcut_type == "CONV":
                shortcut = conv2D_Module( 
                    net, kernel_size=1, stride=stride_height, output_channel=output_channel, rate=1, group=group,
                    trainable               = trainable                 ,
                    initializer             = initializer               ,
                    is_training             = is_training               ,
                    is_add_biases           = is_add_biases             ,
                    is_batch_norm           = False                     ,
                    is_dilated              = False                     ,
                    is_depthwise            = False                     ,
                    is_ternary              = is_ternary                ,
                    is_binary               = is_binary                 ,
                    is_quantized_activation = is_quantized_activation   ,
                    IS_TERNARY              = IS_TERNARY                ,
                    IS_BINARY               = IS_BINARY                 ,
                    IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION   ,
                    IS_DIVERSITY            = IS_DIVERSITY              ,
                    is_fast_mode            = is_fast_mode              ,
                    fast_mode               = fast_mode                 ,
                    complexity_mode         = complexity_mode           ,
                    rebuilding_now          = rebuilding_now            ,
                    complexity_mode_now     = complexity_mode_now       ,
                    Activation              = None                      ,
                    padding                 = padding                   ,
                    data_format             = data_format               ,
                    Analysis                = Analysis                  ,
                    scope                   = "conv_1x1"                )
            elif shortcut_type == "AVG_POOL":
                if data_format == "NHWC":
                    ksize   = [1, stride_height, stride_height, 1]
                    strides = [1, stride_height, stride_height, 1]
                elif data_format == "NCHW":
                    ksize   = [1, 1, stride_height, stride_height]
                    strides = [1, 1, stride_height, stride_height]
                shortcut = tf.nn.avg_pool(
                    value       = net,
                    ksize       = ksize,
                    strides     = strides,
                    padding     = 'SAME',
                    data_format = data_format)
        else:
            shortcut = net
            
    return shortcut

def conv2D(	
    net, kernel_size=3, stride=1, internal_channel=64, output_channel=64, rate=1, group=1,
    trainable               = True,
    initializer             = tf.contrib.layers.variance_scaling_initializer(),
    is_training             = False,
    is_add_biases           = True,
    is_batch_norm           = True,       # For Batch Normalization
    is_dilated              = False,      # For Dilated Convoution
    is_depthwise            = False,      # For Depthwise Convolution
    is_ternary              = False,      # (tensor) For weight ternarization
    is_binary               = False,      # (tensor) For weight binarization
    is_quantized_activation = False,      # (tensor) For activation quantization
    IS_TERNARY              = False,   
    IS_BINARY               = False,
    IS_QUANTIZED_ACTIVATION = False,
    IS_DIVERSITY            = False,
    is_fast_mode            = False,
    fast_mode               = None,
    complexity_mode         = None,
    rebuilding_now          = False,
    complexity_mode_now     = None,
    Activation              = 'ReLU',
    padding                 = "SAME",
    data_format             = "NHWC",
    Analysis                = None,
    scope                   = "conv"
    ):
		
    with tf.variable_scope(scope):
        net = conv2D_Module( 
            net, 
            kernel_size             = kernel_size               , 
            stride                  = stride                    , 
            output_channel          = output_channel            , 
            rate                    = rate                      , 
            group                   = group                     ,
            trainable               = trainable                 ,
            initializer             = initializer               ,
            is_training             = is_training               ,
            is_add_biases           = is_add_biases             ,
            is_batch_norm           = is_batch_norm             ,
            is_dilated              = is_dilated                ,
            is_depthwise            = is_depthwise              ,
            is_ternary              = is_ternary                ,
            is_binary               = is_binary                 ,
            is_quantized_activation = is_quantized_activation   ,
            IS_TERNARY              = IS_TERNARY                ,
            IS_BINARY               = IS_BINARY                 ,
            IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION   ,
            IS_DIVERSITY            = IS_DIVERSITY              ,
            is_fast_mode            = is_fast_mode              ,
            fast_mode               = fast_mode                 ,
            complexity_mode         = complexity_mode           ,
            rebuilding_now          = rebuilding_now            ,
            complexity_mode_now     = complexity_mode_now       ,
            Activation              = Activation                ,
            padding                 = padding                   ,
            data_format             = data_format               ,
            Analysis                = Analysis                  ,
            scope                   = "conv"                    )
    return net

def FC_Variable(
    kernel_size_h,
    kernel_size_w,
    input_channel,
    output_channel,
    trainable,
    initializer,
    is_training,
    is_add_biases,
    is_ternary,
    IS_TERNARY,
    is_binary,
    IS_BINARY,
    IS_DIVERSITY,
    fast_mode,
    is_depthwise,
    data_format
    ):
    
    # float32 Variable
    if data_format == "NHWC":
        float32_weights = tf.get_variable(
            name        = "float32_weights", 
            shape       = [kernel_size_h, kernel_size_w, input_channel, output_channel], 
            dtype       = tf.float32,
            trainable   = trainable and not IS_DIVERSITY,
            initializer = initializer)
    elif data_format == "NCHW":
        float32_weights = tf.get_variable(
            name        = "float32_weights", 
            shape       = [kernel_size_h, kernel_size_w, input_channel, output_channel], 
            dtype       = tf.float32, 
            trainable   = trainable and not IS_DIVERSITY,
            initializer = initializer)
    if is_add_biases:
        float32_biases = tf.get_variable(
            name        = "float32_biases" , 
            shape       = [output_channel], 
            dtype       = tf.float32, 
            trainable   = trainable and not IS_DIVERSITY,
            initializer = initializer)
    else:
        float32_biases = None

    tf.add_to_collection("float32_weights", float32_weights)
    if is_add_biases:
        tf.add_to_collection("float32_biases" , float32_biases)
    
    tf.add_to_collection("float32_params" , float32_weights)
    if is_add_biases:
        tf.add_to_collection("float32_params" , float32_biases)
    
    # Pruning Mask
    float32_weights_mask = tf.Variable(
        initial_value = tf.ones_like(tensor = float32_weights, dtype = tf.float32),
        trainable     = False,
        name          = "float32_weights_mask",
        dtype         = tf.float32)
    tf.add_to_collection("float32_weights_mask", float32_weights_mask)
    
    if is_add_biases:    
        float32_biases_mask = tf.Variable(
            initial_value = tf.ones_like(tensor = float32_biases, dtype = tf.float32),
            trainable     = False,
            name          = "float32_biases_mask",
            dtype         = tf.float32)
        tf.add_to_collection("float32_biases_mask", float32_biases_mask)
    #--------------------#
    #    IS_DIVERSITY    #
    #--------------------#
    elif IS_DIVERSITY:
        if fast_mode == 0:
            kernel_size_fast = 2
        # float32 Variable
        if is_depthwise:
            float32_weights_fast = tf.get_variable(
                name        = "float32_weights_fast", 
                shape       = [kernel_size_fast, kernel_size_fast, input_channel, 1], 
                dtype       = tf.float32,
                trainable   = trainable,
                initializer = initializer)
            if is_add_biases:
                float32_biases_fast = tf.get_variable(
                    name        = "float32_biases_fast", 
                    shape       = [input_channel], 
                    dtype       = tf.float32, 
                    trainable   = trainable,
                    initializer = initializer)
            else:
                float32_biases_fast = None
        else:
            if data_format == "NHWC":
                float32_weights_fast = tf.get_variable(
                    name        = "float32_weights_fast", 
                    shape       = [kernel_size_fast, kernel_size_fast, input_channel, output_channel], 
                    dtype       = tf.float32, 
                    trainable   = trainable,
                    initializer = initializer)
            elif data_format == "NCHW":
                float32_weights_fast = tf.get_variable(
                    name        = "float32_weights_fast", 
                    shape       = [kernel_size_fast, kernel_size_fast, input_channel, output_channel], 
                    dtype       = tf.float32, 
                    trainable   = trainable,
                    initializer = initializer)
            if is_add_biases:
                float32_biases_fast = tf.get_variable(
                    name        = "float32_biases_fast", 
                    shape       = [output_channel], 
                    dtype       = tf.float32, 
                    trainable   = trainable,
                    initializer = initializer)
            else:
                float32_biases_fast = None
    
        tf.add_to_collection("float32_weights_fast", float32_weights_fast)
        if is_add_biases:
            tf.add_to_collection("float32_biases_fast" , float32_biases_fast)
        
        tf.add_to_collection("float32_params" , float32_weights_fast)
        if is_add_biases:
            tf.add_to_collection("float32_params" , float32_biases_fast)
        
        # Pruning Mask
        float32_weights_mask_fast = tf.Variable(
            initial_value = tf.ones_like(tensor = float32_weights_fast, dtype = tf.float32),
            trainable     = False,
            name          = "float32_weights_mask_fast",
            dtype         = tf.float32)
        tf.add_to_collection("float32_weights_mask_fast", float32_weights_mask_fast)
    
        if is_add_biases:    
            float32_biases_mask_fast = tf.Variable(
                initial_value = tf.ones_like(tensor = float32_biases_fast, dtype = tf.float32),
                trainable     = False,
                name          = "float32_biases_mask_fast",
                dtype         = tf.float32)
            tf.add_to_collection("float32_biases_mask_fast", float32_biases_mask_fast)
            
        # is_train Mask
        is_train_float32_weights_mask_fast = tf.Variable(
            initial_value = tf.ones_like(tensor = float32_weights_fast, dtype = tf.float32),
            trainable     = False,
            name          = "is_train_float32_weights_mask_fast",
            dtype         = tf.float32)
        tf.add_to_collection("is_train_float32_weights_mask_fast", is_train_float32_weights_mask_fast)
        
        if is_add_biases:    
            is_train_float32_biases_mask_fast = tf.Variable(
                initial_value = tf.ones_like(tensor = float32_biases_fast, dtype = tf.float32),
                trainable     = False,
                name          = "is_train_float32_biases_mask_fast",
                dtype         = tf.float32)
            tf.add_to_collection("is_train_float32_biases_mask_fast", is_train_float32_biases_mask_fast)
        
        # Constant Weights
        constant_float32_weights_fast = tf.Variable(
            initial_value = tf.zeros_like(tensor = float32_weights_fast, dtype = tf.float32),
            trainable     = False,
            name          = "constant_float32_weights_fast",
            dtype         = tf.float32)
        tf.add_to_collection("constant_float32_weights_fast", constant_float32_weights_fast)
        
        if is_add_biases:    
            constant_float32_biases_fast = tf.Variable(
                initial_value = tf.zeros_like(tensor = float32_biases_fast, dtype = tf.float32),
                trainable     = False,
                name          = "constant_float32_biases_fast",
                dtype         = tf.float32)
            tf.add_to_collection("constant_float32_biases_fast", constant_float32_biases_fast)
        
        weights_ = tf.add(tf.multiply(tf.multiply(float32_weights, float32_weights_mask), is_train_float32_weights_mask), constant_float32_weights)
        if is_add_biases:
            biases_ = tf.add(tf.multiply(tf.multiply(float32_biases, float32_biases_mask), is_train_float32_biases_mask), constant_float32_biases)
        
        weights_fast = tf.add(tf.multiply(tf.multiply(float32_weights_fast, float32_weights_mask_fast), is_train_float32_weights_mask_fast), constant_float32_weights_fast)
        if is_add_biases:
            biases_fast = tf.add(tf.multiply(tf.multiply(float32_biases_fast, float32_biases_mask_fast), is_train_float32_biases_mask_fast), constant_float32_biases_fast)
        
        if is_add_biases:
            return weights_, biases_, weights_fast, biases_fast
        else:
            return weights_, None, weights_fast, None
    #-------------------------#
    #    Ternary Variables    #
    #-------------------------#
    if IS_TERNARY:
        # Ternary boundary of weights and biases 
        ternary_weights_bd = tf.get_variable("ternary_weights_bd", [2], tf.float32, initializer = initializer)
        if is_add_biases:
            ternary_biases_bd  = tf.get_variable("ternary_biases_bd" , [2], tf.float32, initializer = initializer)
        tf.add_to_collection("ternary_weights_bd", ternary_weights_bd)
        if is_add_biases:
            tf.add_to_collection("ternary_biases_bd" , ternary_biases_bd)
        
        # Choose Precision
        weights_tmp = tf.cond(is_ternary, lambda: ternarize_weights(float32_weights, ternary_weights_bd), lambda: float32_weights)
        if is_add_biases:
            biases_tmp = tf.cond(is_ternary, lambda: ternarize_biases (float32_biases , ternary_biases_bd) , lambda: float32_biases )
    
        final_weights = tf.get_variable("final_weights", [kernel_size_h, kernel_size_w, input_channel, output_channel], tf.float32, initializer=initializer)
        if is_add_biases:
            final_biases = tf.get_variable("final_biases" , [output_channel], tf.float32, initializer = initializer)
        else:
            final_biases = None
            
        assign_final_weights = tf.assign(final_weights, weights_tmp)
        if is_add_biases:
            assign_final_biases = tf.assign(final_biases , biases_tmp)
        tf.add_to_collection("assign_var_list", assign_final_weights)
        
        if is_add_biases:
            tf.add_to_collection("assign_var_list", assign_final_biases)
        
        return final_weights, final_biases, None, None
    #------------------------#
    #    Binary Variables    #
    #------------------------#
    if IS_BINARY:
        # Assign clip weights to float32_weights
        min = tf.constant(-1, tf.float32)
        max = tf.constant(1, tf.float32)
        clip_weights = tf.clip_by_value(float32_weights, min, max)
        assign_clip_weights = tf.assign(float32_weights, clip_weights)
        tf.add_to_collection('clip_weights', clip_weights)
        tf.add_to_collection("assign_var_list", assign_clip_weights)
  
        # Binary Weights
        graph = tf.get_default_graph()
        with ops.name_scope("Binarized") as name:
            with graph.gradient_override_map({"Sign": "Identity"}):
                weights_ = float32_weights #tf.clip_by_value(float32_weights, -1, 1)
                weights_ = tf.sign(weights_)
                
        tf.add_to_collection('binary_weights', weights_)
        
        # Binary Scale
        #if IS_BINARY:
        ## Show the model
        if SHOW_MODEL:
            print("-> Binary")
        with tf.variable_scope('Binary_Scalse'):
            binary_scale = tf.get_variable("binary_scale", [1], tf.float32, initializer=initializer)
            tf.add_to_collection("binary_scale", binary_scale)
            weights_ = tf.multiply(weights_, binary_scale)
        
        if is_add_biases:
            biases_ = tf.multiply(float32_biases, float32_biases_mask)
            
        if is_add_biases:
            return weights_, biases_, None, None
        else:
            return weights_, None, None, None
    else:
        if is_add_biases:
            return tf.multiply(float32_weights, float32_weights_mask), tf.multiply(float32_biases, float32_biases_mask), None, None
        else:
            return tf.multiply(float32_weights, float32_weights_mask), None, None, None

def FC(
    net                     , 
    output_channel          ,
	initializer             ,
    is_training             ,
    is_add_biases           ,
	is_batch_norm           ,
	is_dilated              ,
	is_ternary              ,
    is_binary               ,
	is_quantized_activation ,
	IS_TERNARY              ,
    IS_BINARY               ,
	IS_QUANTIZED_ACTIVATION ,
	Activation              ,
    data_format             ,
	Analysis                ,
	scope                   ,
    trainable               = True,
    IS_DIVERSITY            = False,
    is_fast_mode            = False,
    fast_mode               = None
	):
    with tf.variable_scope(scope):
        stride = 1
        group = 1
        rate = 1
        is_depthwise = False
        padding = 'VALID'
        
        if data_format == "NHWC":
            input_channel = net.get_shape().as_list()[-1]
            kernel_size_h = net.get_shape().as_list()[1]
            kernel_size_w = net.get_shape().as_list()[2]
        elif data_format == "NCHW":
            input_channel = net.get_shape().as_list()[1]
            kernel_size_h = net.get_shape().as_list()[2]
            kernel_size_w = net.get_shape().as_list()[3]
        
        # -- Analyzer --
        utils.Analyzer( Analysis, 
                        net, 
                        type                    = 'CONV', 
                        kernel_shape            = [kernel_size_h, kernel_size_w, input_channel, output_channel], 
                        stride                  = stride, 
                        group                   = group,
                        is_depthwise            = is_depthwise,
                        IS_TERNARY              = IS_TERNARY,
                        IS_BINARY               = IS_BINARY,
                        IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
                        padding                 = padding, 
                        name                    = 'Conv')

        #-------------------#
        #    Convolution    #
        #-------------------#
        # Variable define
        weights, biases, weights_fast, biases_fast = FC_Variable(
            kernel_size_h  = kernel_size_h,
            kernel_size_w  = kernel_size_w,
            input_channel  = input_channel,
            output_channel = output_channel,
            trainable      = trainable,
            initializer    = initializer,
            is_add_biases  = is_add_biases,
            is_training    = is_training,
            is_ternary     = is_ternary,
            IS_TERNARY     = IS_TERNARY,
            is_binary      = is_binary,
            IS_BINARY      = IS_BINARY,
            IS_DIVERSITY   = IS_DIVERSITY,
            fast_mode      = fast_mode,
            is_depthwise   = is_depthwise,
            data_format    = data_format)
        
        # Convolution
        if is_dilated:
            ## Show the model
            if SHOW_MODEL:
                print("-> Dilated Conv, Weights:{}, stride:{}" .format(weights.get_shape().as_list(), stride))
            if IS_DIVERSITY:
                net_fast = tf.nn.atrous_conv2d( 
                    value       = net, 
                    filters     = weights, 
                    rate        = rate, 
                    padding     = padding)
            net = tf.nn.atrous_conv2d( 
                value       = net, 
                filters     = weights, 
                rate        = rate, 
                padding     = padding)
        else:
            ## Show the model
            if SHOW_MODEL:
                print("-> Conv, Weights:{}, stride:{}" .format(weights.get_shape().as_list(), stride))
            if IS_DIVERSITY:
                net_fast = tf.nn.conv2d( 
                    input       = net, 
                    filter      = weights, 
                    strides     = [1, stride, stride, 1] if data_format == "NHWC" else [1, 1, stride, stride], 
                    padding     = padding,
                    data_format = data_format)
            net = tf.nn.conv2d( 
                input       = net, 
                filter      = weights, 
                strides     = [1, stride, stride, 1] if data_format == "NHWC" else [1, 1, stride, stride], 
                padding     = padding,
                data_format = data_format)
        
        tf.add_to_collection('conv_outputs', net)
        
        if is_add_biases:
            ## Show the model
            if SHOW_MODEL:
                print("-> Add Biases, Biases:{}".format(biases.get_shape().as_list()))
            if IS_DIVERSITY:
                net_fast = tf.nn.bias_add(net_fast, biases, data_format)
            net = tf.nn.bias_add(net, biases, data_format)
        
        ## For checking the final weights/biases is ternary/binary or not
        tf.add_to_collection("weights", weights)
        if is_add_biases:
            tf.add_to_collection("biases", biases)

        #-----------------#
        #    Operation    #
        #-----------------#
        # Ternary Scale
        if IS_TERNARY:
            ## Show the model
            if SHOW_MODEL:
                print("-> Ternary")
            if IS_DIVERSITY:
                with tf.variable_scope('Ternary_Scalse_fast'):
                    ternary_scale_fast = tf.get_variable("ternary_scale_fast", [1], tf.float32, initializer=initializer)
                    tf.add_to_collection("ternary_scale_fast", ternary_scale_fast)
                    net_fast = tf.multiply(net_fast, ternary_scale_fast)
            with tf.variable_scope('Ternary_Scalse'):
                ternary_scale = tf.get_variable("ternary_scale", [1], tf.float32, initializer=initializer)
                tf.add_to_collection("ternary_scale", ternary_scale)
                net = tf.multiply(net, ternary_scale)
        
        # (Diversity) Choose mode
        if IS_DIVERSITY:
            net = tf.cond(is_fast_mode,
                            lambda: net_fast,
                            lambda: net)
        
        # Batch Normalization
        if is_batch_norm == True:
            ## Show the model
            if SHOW_MODEL:
                print("-> Batch_Norm")
            """
            if IS_DIVERSITY:
                net_fast = batch_norm(net_fast, is_training, data_format)
            net = batch_norm(net, is_training, data_format)
            """
            net = batch_norm(net, is_training, data_format, fast_mode!='0' and trainable) #trainable and not IS_DIVERSITY)
        
        # Activation
        if Activation == 'ReLU':
            ## Show the model
            if SHOW_MODEL:
                print("-> ReLU")
            net = tf.nn.relu(net)
        elif Activation == 'Sigmoid':
            ## Show the model
            if SHOW_MODEL:
                print("-> Sigmoid")
            net = tf.nn.sigmoid(net)
        elif Activation == 'CReLU':
            ## Show the model
            if SHOW_MODEL:
                print("-> CReLU")
            net = CReLU(net, initializer = initializer)
        elif Activation == 'HardTanh':
            ## Show the model
            if SHOW_MODEL:
                print("-> HardTanh")
            net = tf.clip_by_value(net, -1, 1)
        else:
            net = net
            
        # Activation Quantization
        if IS_QUANTIZED_ACTIVATION:
            ## Show the model
            if SHOW_MODEL:
                print("-> Quantized Activation")
            net = quantize_Module(net, is_quantized_activation, data_format)
            
    return net
#------------#
#    POOL    #
#------------#
def Pyramid_Pooling(
    net, 
    strides, 
    output_channel,
    is_training,
    is_add_biases
    ):
    
    with tf.variable_scope("Pyramid_Pooling"):
        input_shape = net.get_shape().as_list()
        
        for level, stride in enumerate(strides):
            with tf.variable_scope('pool%d' %(level)):
                net_tmp = tf.nn.avg_pool(net, ksize=[1, stride, stride, 1], strides=[1, stride, stride, 1], padding='SAME')
                
                net_tmp = conv2D_Module( net, kernel_size=1, stride=1, output_channel=output_channel, rate=rate, group=1,
                                         initializer             = initializer              ,
                                         is_training             = is_training              ,
                                         is_add_biases           = is_add_biases            ,
                                         is_batch_norm           = True                     ,
                                         is_dilated              = False                    ,
                                         is_depthwise            = False                    ,
                                         is_ternary              = is_ternary               ,
                                         is_quantized_activation = is_quantized_activation  ,
                                         IS_TERNARY              = IS_TERNARY               ,
                                         IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION  ,
                                         Activation              = Activation               ,
                                         padding                 = padding                  ,
                                         Analysis                = Analysis                 ,
                                         scope                   = "conv_1x1"               )

                net_tmp = tf.image.resize_images(net_tmp, [input_shape[1], input_shape[2]])
                
            net = tf.concat([net, net_tmp], axis=3)
    return net

def indice_pool(
    net, 
    kernel_size, 
    stride,  
    IS_QUANTIZED_ACTIVATION, # For Analyzer
    Analysis,
    scope,
    ):

    with tf.variable_scope(scope):    
        output_shape = net.get_shape().as_list()
        
        # -- Analyzer --
        utils.Analyzer( Analysis, 
                        net, 
                        type                    = 'POOL', 
                        kernel_shape            = [stride, stride, output_shape[3], output_shape[3]], 
                        stride                  = stride, 
                        IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
                        name                    = 'Pool')
        
        net, indices = tf.nn.max_pool_with_argmax( 
            input   = net, 
            ksize   = [1, kernel_size, kernel_size, 1],
            strides = [1, stride, stride, 1],
            padding = "SAME",
            Targmax = None,
            name    = None)
    return net, indices, output_shape
	
def indice_unpool(
    net,
    indices,
    output_shape,
    scope
    ):
    """
    Max unpolling by indices and output_shape.
    
    Args:
    (1) net          : Input. An 4D tensor. Shape=[Batch_Size, Image_Height, Image_Width, Image_Depth]
    (2) output_shape : The shape to be restored to.
    (3) indices      : An indices of the max pooling. More detail in "max_pooling"
    
    Return:
    (1) net : An output tensor after max unpooling.
    """
    
    with tf.variable_scope(scope):
        input_shape = net.get_shape().as_list()
        
        # Calculate indices for batch, height, width and channel
        meshgrid = tf.meshgrid(tf.range(input_shape[1]), tf.range(input_shape[0]), tf.range(input_shape[2]), tf.range(input_shape[3]))
        b = tf.cast(meshgrid[1], tf.int64)
        h = indices // (output_shape[2] * output_shape[3])
        w = indices // output_shape[3] - h * output_shape[2]
        c = indices - (h * output_shape[2] + w) * output_shape[3]
        
        # transpose indices & reshape update values to one dimension
        updates_size = tf.size(net)
        indices = tf.transpose(tf.reshape(tf.stack([b, h, w, c]), [4, updates_size]))
        values = tf.reshape(net, [updates_size])
        net = tf.scatter_nd(indices, values, output_shape)
        
    return net

def ID_0550225(
    net,
    initializer,
    is_training,
    is_ternary,
    is_quantized_activation,
    data_format
    ):
    
    Analysis = {}
    
    if data_format == "channels_first":
        net = tf.transpose(net, [0, 3, 1, 2])
        
    net = tf.layers.conv2d(
        inputs             = net, 
        filters            = 16, 
        kernel_size        = 3, 
        strides            = 1,
        padding            = 'SAME', 
        use_bias           = False,
        kernel_initializer = tf.variance_scaling_initializer(),
        data_format        = data_format)
    
    net = batch_norm(net, is_training, data_format)
    
    net = tf.nn.relu(net)
    
    #net = conv2D_Module( 
    #    net, kernel_size = 3, stride = 1, output_channel = 16, rate = 1, group = 1,
    #    initializer             = initializer,
    #    is_training             = is_training,
    #    is_add_biases           = False,
    #    is_batch_norm           = True,
    #    is_dilated              = False,
    #    is_depthwise            = False,
    #    is_ternary              = is_ternary,
    #    is_quantized_activation = is_quantized_activation,
    #    IS_TERNARY              = False,
    #    IS_QUANTIZED_ACTIVATION = False,
    #    Activation              = "ReLU",
    #    padding                 = "SAME",
    #    Analysis                = Analysis,
    #    scope                   = "layer0")
    
    shortcut = tf.layers.conv2d(
        inputs             = net, 
        filters            = 16, 
        kernel_size        = 1, 
        strides            = 1,
        padding            = 'SAME', 
        use_bias           = False,
        kernel_initializer = tf.variance_scaling_initializer(),
        data_format        = data_format)
    
    #shortcut = conv2D_Module( 
    #    net, kernel_size = 1, stride = 1, output_channel = 16, rate = 1, group = 1,
    #    initializer             = initializer,
    #    is_training             = is_training,
    #    is_add_biases           = False,
    #    is_batch_norm           = False,
    #    is_dilated              = False,
    #    is_depthwise            = False,
    #    is_ternary              = is_ternary,
    #    is_quantized_activation = is_quantized_activation,
    #    IS_TERNARY              = False,
    #    IS_QUANTIZED_ACTIVATION = False,
    #    Activation              = None,
    #    padding                 = "SAME",
    #    Analysis                = Analysis,
    #    scope                   = "shortcut0")
    
    net = tf.layers.conv2d(
        inputs             = net, 
        filters            = 16, 
        kernel_size        = 3, 
        strides            = 1,
        padding            = 'SAME', 
        use_bias           = False,
        kernel_initializer = tf.variance_scaling_initializer(),
        data_format        = data_format)
    
    net = batch_norm(net, is_training, data_format)
    
    net = tf.nn.relu(net)
    
    #net = conv2D_Module( 
    #    net, kernel_size = 3, stride = 1, output_channel = 16, rate = 1, group = 1,
    #    initializer             = initializer,
    #    is_training             = is_training,
    #    is_add_biases           = False,
    #    is_batch_norm           = True,
    #    is_dilated              = False,
    #    is_depthwise            = False,
    #    is_ternary              = is_ternary,
    #    is_quantized_activation = is_quantized_activation,
    #    IS_TERNARY              = False,
    #    IS_QUANTIZED_ACTIVATION = False,
    #    Activation              = "ReLU",
    #    padding                 = "SAME",
    #    Analysis                = Analysis,
    #    scope                   = "layer1")
    
    net = tf.layers.conv2d(
        inputs             = net, 
        filters            = 16, 
        kernel_size        = 3, 
        strides            = 1,
        padding            = 'SAME', 
        use_bias           = False,
        kernel_initializer = tf.variance_scaling_initializer(),
        data_format        = data_format)
    
    #net = conv2D_Module( 
    #    net, kernel_size = 3, stride = 1, output_channel = 16, rate = 1, group = 1,
    #    initializer             = initializer,
    #    is_training             = is_training,
    #    is_add_biases           = False,
    #    is_batch_norm           = False,
    #    is_dilated              = False,
    #    is_depthwise            = False,
    #    is_ternary              = is_ternary,
    #    is_quantized_activation = is_quantized_activation,
    #    IS_TERNARY              = False,
    #    IS_QUANTIZED_ACTIVATION = False,
    #    Activation              = None,
    #    padding                 = "SAME",
    #    Analysis                = Analysis,
    #    scope                   = "layer2")
    
    net = net + shortcut
    
    shortcut = net # layer3
    
    net = batch_norm(net, is_training, data_format)
    
    net = tf.nn.relu(net)
    
    net = tf.layers.conv2d(
        inputs             = net, 
        filters            = 16, 
        kernel_size        = 3, 
        strides            = 1,
        padding            = 'SAME', 
        use_bias           = False,
        kernel_initializer = tf.variance_scaling_initializer(),
        data_format        = data_format)
    
    net = batch_norm(net, is_training, data_format)
    
    net = tf.nn.relu(net)
    
    #net = conv2D_Module( 
    #    net, kernel_size = 3, stride = 1, output_channel = 16, rate = 1, group = 1,
    #    initializer             = initializer,
    #    is_training             = is_training,
    #    is_add_biases           = False,
    #    is_batch_norm           = True,
    #    is_dilated              = False,
    #    is_depthwise            = False,
    #    is_ternary              = is_ternary,
    #    is_quantized_activation = is_quantized_activation,
    #    IS_TERNARY              = False,
    #    IS_QUANTIZED_ACTIVATION = False,
    #    Activation              = "ReLU",
    #    padding                 = "SAME",
    #    Analysis                = Analysis,
    #    scope                   = "layer4")
    
    net = tf.layers.conv2d(
        inputs             = net, 
        filters            = 16, 
        kernel_size        = 3, 
        strides            = 1,
        padding            = 'SAME', 
        use_bias           = False,
        kernel_initializer = tf.variance_scaling_initializer(),
        data_format        = data_format)
    
    #net = conv2D_Module( 
    #    net, kernel_size = 3, stride = 1, output_channel = 16, rate = 1, group = 1,
    #    initializer             = initializer,
    #    is_training             = is_training,
    #    is_add_biases           = False,
    #    is_batch_norm           = False,
    #    is_dilated              = False,
    #    is_depthwise            = False,
    #    is_ternary              = is_ternary,
    #    is_quantized_activation = is_quantized_activation,
    #    IS_TERNARY              = False,
    #    IS_QUANTIZED_ACTIVATION = False,
    #    Activation              = None,
    #    padding                 = "SAME",
    #    Analysis                = Analysis,
    #    scope                   = "layer5")
    
    net = net + shortcut
    
    shortcut = net # layer6
    
    net = batch_norm(net, is_training, data_format)
    
    net = tf.nn.relu(net)
    
    net = tf.layers.conv2d(
        inputs             = net, 
        filters            = 16, 
        kernel_size        = 3, 
        strides            = 1,
        padding            = 'SAME', 
        use_bias           = False,
        kernel_initializer = tf.variance_scaling_initializer(),
        data_format        = data_format)
    
    net = batch_norm(net, is_training, data_format)
    
    net = tf.nn.relu(net)
    
    #net = conv2D_Module( 
    #    net, kernel_size = 3, stride = 1, output_channel = 16, rate = 1, group = 1,
    #    initializer             = initializer,
    #    is_training             = is_training,
    #    is_add_biases           = False,
    #    is_batch_norm           = True,
    #    is_dilated              = False,
    #    is_depthwise            = False,
    #    is_ternary              = is_ternary,
    #    is_quantized_activation = is_quantized_activation,
    #    IS_TERNARY              = False,
    #    IS_QUANTIZED_ACTIVATION = False,
    #    Activation              = "ReLU",
    #    padding                 = "SAME",
    #    Analysis                = Analysis,
    #    scope                   = "layer7")
    
    net = tf.layers.conv2d(
        inputs             = net, 
        filters            = 16, 
        kernel_size        = 3, 
        strides            = 1,
        padding            = 'SAME', 
        use_bias           = False,
        kernel_initializer = tf.variance_scaling_initializer(),
        data_format        = data_format)
    
    #net = conv2D_Module( 
    #    net, kernel_size = 3, stride = 1, output_channel = 16, rate = 1, group = 1,
    #    initializer             = initializer,
    #    is_training             = is_training,
    #    is_add_biases           = False,
    #    is_batch_norm           = False,
    #    is_dilated              = False,
    #    is_depthwise            = False,
    #    is_ternary              = is_ternary,
    #    is_quantized_activation = is_quantized_activation,
    #    IS_TERNARY              = False,
    #    IS_QUANTIZED_ACTIVATION = False,
    #    Activation              = None,
    #    padding                 = "SAME",
    #    Analysis                = Analysis,
    #    scope                   = "layer8")
    
    net = net + shortcut
    
    net = batch_norm(net, is_training, data_format)
    
    net = tf.nn.relu(net) # layer9
    
    shortcut = tf.layers.conv2d(
        inputs             = net, 
        filters            = 32, 
        kernel_size        = 1, 
        strides            = 2,
        padding            = 'SAME', 
        use_bias           = False,
        kernel_initializer = tf.variance_scaling_initializer(),
        data_format        = data_format)
    
    #shortcut = conv2D_Module( 
    #    net, kernel_size = 1, stride = 2, output_channel = 32, rate = 1, group = 1,
    #    initializer             = initializer,
    #    is_training             = is_training,
    #    is_add_biases           = False,
    #    is_batch_norm           = False,
    #    is_dilated              = False,
    #    is_depthwise            = False,
    #    is_ternary              = is_ternary,
    #    is_quantized_activation = is_quantized_activation,
    #    IS_TERNARY              = False,
    #    IS_QUANTIZED_ACTIVATION = False,
    #    Activation              = None,
    #    padding                 = "SAME",
    #    Analysis                = Analysis,
    #    scope                   = "shortcut1")
    
    net = tf.layers.conv2d(
        inputs             = net, 
        filters            = 32, 
        kernel_size        = 3, 
        strides            = 2,
        padding            = 'SAME', 
        use_bias           = False,
        kernel_initializer = tf.variance_scaling_initializer(),
        data_format        = data_format)
    
    net = batch_norm(net, is_training, data_format)
    
    net = tf.nn.relu(net)
    
    #net = conv2D_Module( 
    #    net, kernel_size = 3, stride = 2, output_channel = 32, rate = 1, group = 1,
    #    initializer             = initializer,
    #    is_training             = is_training,
    #    is_add_biases           = False,
    #    is_batch_norm           = True,
    #    is_dilated              = False,
    #    is_depthwise            = False,
    #    is_ternary              = is_ternary,
    #    is_quantized_activation = is_quantized_activation,
    #    IS_TERNARY              = False,
    #    IS_QUANTIZED_ACTIVATION = False,
    #    Activation              = "ReLU",
    #    padding                 = "SAME",
    #    Analysis                = Analysis,
    #    scope                   = "layer10")
    
    net = tf.layers.conv2d(
        inputs             = net, 
        filters            = 32, 
        kernel_size        = 3, 
        strides            = 1,
        padding            = 'SAME', 
        use_bias           = False,
        kernel_initializer = tf.variance_scaling_initializer(),
        data_format        = data_format)
    
    #net = conv2D_Module( 
    #    net, kernel_size = 3, stride = 1, output_channel = 32, rate = 1, group = 1,
    #    initializer             = initializer,
    #    is_training             = is_training,
    #    is_add_biases           = False,
    #    is_batch_norm           = False,
    #    is_dilated              = False,
    #    is_depthwise            = False,
    #    is_ternary              = is_ternary,
    #    is_quantized_activation = is_quantized_activation,
    #    IS_TERNARY              = False,
    #    IS_QUANTIZED_ACTIVATION = False,
    #    Activation              = None,
    #    padding                 = "SAME",
    #    Analysis                = Analysis,
    #    scope                   = "layer11")
    
    net = net + shortcut
    
    shortcut = net # layer12
    
    net = batch_norm(net, is_training, data_format)
    
    net = tf.nn.relu(net)
    
    net = tf.layers.conv2d(
        inputs             = net, 
        filters            = 32, 
        kernel_size        = 3, 
        strides            = 1,
        padding            = 'SAME', 
        use_bias           = False,
        kernel_initializer = tf.variance_scaling_initializer(),
        data_format        = data_format)
    
    net = batch_norm(net, is_training, data_format)
    
    net = tf.nn.relu(net)
    
    #net = conv2D_Module( 
    #    net, kernel_size = 3, stride = 1, output_channel = 32, rate = 1, group = 1,
    #    initializer             = initializer,
    #    is_training             = is_training,
    #    is_add_biases           = False,
    #    is_batch_norm           = True,
    #    is_dilated              = False,
    #    is_depthwise            = False,
    #    is_ternary              = is_ternary,
    #    is_quantized_activation = is_quantized_activation,
    #    IS_TERNARY              = False,
    #    IS_QUANTIZED_ACTIVATION = False,
    #    Activation              = "ReLU",
    #    padding                 = "SAME",
    #    Analysis                = Analysis,
    #    scope                   = "layer13")
    
    net = tf.layers.conv2d(
        inputs             = net, 
        filters            = 32, 
        kernel_size        = 3, 
        strides            = 1,
        padding            = 'SAME', 
        use_bias           = False,
        kernel_initializer = tf.variance_scaling_initializer(),
        data_format        = data_format)
    
    #net = conv2D_Module( 
    #    net, kernel_size = 3, stride = 1, output_channel = 32, rate = 1, group = 1,
    #    initializer             = initializer,
    #    is_training             = is_training,
    #    is_add_biases           = False,
    #    is_batch_norm           = False,
    #    is_dilated              = False,
    #    is_depthwise            = False,
    #    is_ternary              = is_ternary,
    #    is_quantized_activation = is_quantized_activation,
    #    IS_TERNARY              = False,
    #    IS_QUANTIZED_ACTIVATION = False,
    #    Activation              = None,
    #    padding                 = "SAME",
    #    Analysis                = Analysis,
    #    scope                   = "layer14")
    
    net = net + shortcut
    
    shortcut = net # layer15
    
    net = batch_norm(net, is_training, data_format)
    
    net = tf.nn.relu(net)
    
    net = tf.layers.conv2d(
        inputs             = net, 
        filters            = 32, 
        kernel_size        = 3, 
        strides            = 1,
        padding            = 'SAME', 
        use_bias           = False,
        kernel_initializer = tf.variance_scaling_initializer(),
        data_format        = data_format)
    
    net = batch_norm(net, is_training, data_format)
    
    net = tf.nn.relu(net)
    
    #net = conv2D_Module( 
    #    net, kernel_size = 3, stride = 1, output_channel = 32, rate = 1, group = 1,
    #    initializer             = initializer,
    #    is_training             = is_training,
    #    is_add_biases           = False,
    #    is_batch_norm           = True,
    #    is_dilated              = False,
    #    is_depthwise            = False,
    #    is_ternary              = is_ternary,
    #    is_quantized_activation = is_quantized_activation,
    #    IS_TERNARY              = False,
    #    IS_QUANTIZED_ACTIVATION = False,
    #    Activation              = "ReLU",
    #    padding                 = "SAME",
    #    Analysis                = Analysis,
    #    scope                   = "layer16")
    
    net = tf.layers.conv2d(
        inputs             = net, 
        filters            = 32, 
        kernel_size        = 3, 
        strides            = 1,
        padding            = 'SAME', 
        use_bias           = False,
        kernel_initializer = tf.variance_scaling_initializer(),
        data_format        = data_format)
    
    #net = conv2D_Module( 
    #    net, kernel_size = 3, stride = 1, output_channel = 32, rate = 1, group = 1,
    #    initializer             = initializer,
    #    is_training             = is_training,
    #    is_add_biases           = False,
    #    is_batch_norm           = False,
    #    is_dilated              = False,
    #    is_depthwise            = False,
    #    is_ternary              = is_ternary,
    #    is_quantized_activation = is_quantized_activation,
    #    IS_TERNARY              = False,
    #    IS_QUANTIZED_ACTIVATION = False,
    #    Activation              = None,
    #    padding                 = "SAME",
    #    Analysis                = Analysis,
    #    scope                   = "layer17")
    
    net = net + shortcut
    
    net = batch_norm(net, is_training, data_format)
    
    net = tf.nn.relu(net) # layer18
    
    shortcut = tf.layers.conv2d(
        inputs             = net, 
        filters            = 64, 
        kernel_size        = 1, 
        strides            = 2,
        padding            = 'SAME', 
        use_bias           = False,
        kernel_initializer = tf.variance_scaling_initializer(),
        data_format        = data_format)
    
    #shortcut = conv2D_Module( 
    #    net, kernel_size = 1, stride = 2, output_channel = 64, rate = 1, group = 1,
    #    initializer             = initializer,
    #    is_training             = is_training,
    #    is_add_biases           = False,
    #    is_batch_norm           = False,
    #    is_dilated              = False,
    #    is_depthwise            = False,
    #    is_ternary              = is_ternary,
    #    is_quantized_activation = is_quantized_activation,
    #    IS_TERNARY              = False,
    #    IS_QUANTIZED_ACTIVATION = False,
    #    Activation              = None,
    #    padding                 = "SAME",
    #    Analysis                = Analysis,
    #    scope                   = "shortcut2")
    
    net = tf.layers.conv2d(
        inputs             = net, 
        filters            = 64, 
        kernel_size        = 3, 
        strides            = 2,
        padding            = 'SAME', 
        use_bias           = False,
        kernel_initializer = tf.variance_scaling_initializer(),
        data_format        = data_format)
    
    net = batch_norm(net, is_training, data_format)
    
    net = tf.nn.relu(net)
    
    #net = conv2D_Module( 
    #    net, kernel_size = 3, stride = 2, output_channel = 64, rate = 1, group = 1,
    #    initializer             = initializer,
    #    is_training             = is_training,
    #    is_add_biases           = False,
    #    is_batch_norm           = True,
    #    is_dilated              = False,
    #    is_depthwise            = False,
    #    is_ternary              = is_ternary,
    #    is_quantized_activation = is_quantized_activation,
    #    IS_TERNARY              = False,
    #    IS_QUANTIZED_ACTIVATION = False,
    #    Activation              = "ReLU",
    #    padding                 = "SAME",
    #    Analysis                = Analysis,
    #    scope                   = "layer19")
    
    net = tf.layers.conv2d(
        inputs             = net, 
        filters            = 64, 
        kernel_size        = 3, 
        strides            = 1,
        padding            = 'SAME', 
        use_bias           = False,
        kernel_initializer = tf.variance_scaling_initializer(),
        data_format        = data_format)
    
    #net = conv2D_Module( 
    #    net, kernel_size = 3, stride = 1, output_channel = 64, rate = 1, group = 1,
    #    initializer             = initializer,
    #    is_training             = is_training,
    #    is_add_biases           = False,
    #    is_batch_norm           = False,
    #    is_dilated              = False,
    #    is_depthwise            = False,
    #    is_ternary              = is_ternary,
    #    is_quantized_activation = is_quantized_activation,
    #    IS_TERNARY              = False,
    #    IS_QUANTIZED_ACTIVATION = False,
    #    Activation              = None,
    #    padding                 = "SAME",
    #    Analysis                = Analysis,
    #    scope                   = "layer20")
    
    net = net + shortcut
    
    shortcut = net # layer21
    
    net = batch_norm(net, is_training, data_format)
    
    net = tf.nn.relu(net)
    
    net = tf.layers.conv2d(
        inputs             = net, 
        filters            = 64, 
        kernel_size        = 3, 
        strides            = 1,
        padding            = 'SAME', 
        use_bias           = False,
        kernel_initializer = tf.variance_scaling_initializer(),
        data_format        = data_format)
    
    net = batch_norm(net, is_training, data_format)
    
    net = tf.nn.relu(net)
    
    #net = conv2D_Module( 
    #    net, kernel_size = 3, stride = 1, output_channel = 64, rate = 1, group = 1,
    #    initializer             = initializer,
    #    is_training             = is_training,
    #    is_add_biases           = False,
    #    is_batch_norm           = True,
    #    is_dilated              = False,
    #    is_depthwise            = False,
    #    is_ternary              = is_ternary,
    #    is_quantized_activation = is_quantized_activation,
    #    IS_TERNARY              = False,
    #    IS_QUANTIZED_ACTIVATION = False,
    #    Activation              = "ReLU",
    #    padding                 = "SAME",
    #    Analysis                = Analysis,
    #    scope                   = "layer22")
    
    net = tf.layers.conv2d(
        inputs             = net, 
        filters            = 64, 
        kernel_size        = 3, 
        strides            = 1,
        padding            = 'SAME', 
        use_bias           = False,
        kernel_initializer = tf.variance_scaling_initializer(),
        data_format        = data_format)
    
    #net = conv2D_Module( 
    #    net, kernel_size = 3, stride = 1, output_channel = 64, rate = 1, group = 1,
    #    initializer             = initializer,
    #    is_training             = is_training,
    #    is_add_biases           = False,
    #    is_batch_norm           = False,
    #    is_dilated              = False,
    #    is_depthwise            = False,
    #    is_ternary              = is_ternary,
    #    is_quantized_activation = is_quantized_activation,
    #    IS_TERNARY              = False,
    #    IS_QUANTIZED_ACTIVATION = False,
    #    Activation              = None,
    #    padding                 = "SAME",
    #    Analysis                = Analysis,
    #    scope                   = "layer23")
    
    net = net + shortcut
    
    shortcut = net # layer24
    
    net = batch_norm(net, is_training, data_format)
    
    net = tf.nn.relu(net)
    
    net = tf.layers.conv2d(
        inputs             = net, 
        filters            = 64, 
        kernel_size        = 3, 
        strides            = 1,
        padding            = 'SAME', 
        use_bias           = False,
        kernel_initializer = tf.variance_scaling_initializer(),
        data_format        = data_format)
    
    net = batch_norm(net, is_training, data_format)
    
    net = tf.nn.relu(net)
    
    #net = conv2D_Module( 
    #    net, kernel_size = 3, stride = 1, output_channel = 64, rate = 1, group = 1,
    #    initializer             = initializer,
    #    is_training             = is_training,
    #    is_add_biases           = False,
    #    is_batch_norm           = True,
    #    is_dilated              = False,
    #    is_depthwise            = False,
    #    is_ternary              = is_ternary,
    #    is_quantized_activation = is_quantized_activation,
    #    IS_TERNARY              = False,
    #    IS_QUANTIZED_ACTIVATION = False,
    #    Activation              = "ReLU",
    #    padding                 = "SAME",
    #    Analysis                = Analysis,
    #    scope                   = "layer25")
    
    net = tf.layers.conv2d(
        inputs             = net, 
        filters            = 64, 
        kernel_size        = 3, 
        strides            = 1,
        padding            = 'SAME', 
        use_bias           = False,
        kernel_initializer = tf.variance_scaling_initializer(),
        data_format        = data_format)
    
    #net = conv2D_Module( 
    #    net, kernel_size = 3, stride = 1, output_channel = 64, rate = 1, group = 1,
    #    initializer             = initializer,
    #    is_training             = is_training,
    #    is_add_biases           = False,
    #    is_batch_norm           = False,
    #    is_dilated              = False,
    #    is_depthwise            = False,
    #    is_ternary              = is_ternary,
    #    is_quantized_activation = is_quantized_activation,
    #    IS_TERNARY              = False,
    #    IS_QUANTIZED_ACTIVATION = False,
    #    Activation              = None,
    #    padding                 = "SAME",
    #    Analysis                = Analysis,
    #    scope                   = "layer26")
    
    net = net + shortcut
    
    net = batch_norm(net, is_training, data_format)
    
    net = tf.nn.relu(net) # layer27
    
    net = tf.layers.average_pooling2d(
        inputs      = net, 
        pool_size   = 8, 
        strides     = 1, 
        padding     = 'VALID',
        data_format = data_format)
    
    net = tf.reshape(net, [-1, 64])
    
    net = tf.layers.dense(
        inputs = net, 
        units  = 10)
    
    #net = tf.nn.avg_pool( # layer28
    #    value   = net,
    #    ksize   = [1, 8, 8, 1],
    #    strides = [1, 8, 8, 1],
    #    padding = 'SAME')
    #
    #net = conv2D_Module( 
    #    net, kernel_size = 1, stride = 1, output_channel = 10, rate = 1, group = 1,
    #    initializer             = initializer,
    #    is_training             = is_training,
    #    is_add_biases           = True,
    #    is_batch_norm           = False,
    #    is_dilated              = False,
    #    is_depthwise            = False,
    #    is_ternary              = is_ternary,
    #    is_quantized_activation = is_quantized_activation,
    #    IS_TERNARY              = False,
    #    IS_QUANTIZED_ACTIVATION = False,
    #    Activation              = None,
    #    padding                 = "SAME",
    #    Analysis                = Analysis,
    #    scope                   = "layer29")
    
    return net, Analysis
    
    
    