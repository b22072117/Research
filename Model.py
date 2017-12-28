from __future__ import print_function
import tensorflow as tf
import numpy as np
import math
import scipy.ndimage

from scipy import misc
from PIL import Image

import pdb

import utils

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
    is_training
    ):            

    net = tf.layers.batch_normalization(
            inputs   = net, 
            axis     = 3,
            momentum = 0.997, 
            epsilon  = 1e-5, 
            center   = True,
            scale    = True, 
            training = is_training, 
            fused    = True)
            
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
    is_quantized_activation
    ):
    
    quantized_net = quantize_activation(net)
    net = tf.cond(is_quantized_activation, lambda: quantized_net, lambda: net)
    
    tf.add_to_collection("is_quantized_activation", is_quantized_activation)
    tf.add_to_collection("quantized_net", net)

def shuffle_net(
    net,
    group
    ):
    
    input_channel = net.get_shape().as_list()[-1]
    group_size = input_channel / group
    
    for channel_out in range(input_channel):
        which_group = channel_out % group
        which_channel_in_group = channel_out / group
        channel_in = which_group * group_size + which_channel_in_group
        #print("{}, {}, {}" .format(which_group, which_channel_in_group, channel_in))
        net_in = tf.expand_dims(net[:, :, :, channel_in], axis=3)
        
        if channel_out == 0:
            net_tmp = net_in
        else:
            net_tmp = tf.concat([net_tmp, net_in], axis = 3)
    
    return net_tmp
    
def conv2D_Variable(
    kernel_size,
    input_channel,
    output_channel, 
    initializer,
    is_add_biases,
    is_ternary,
    IS_TERNARY,
    is_depthwise
    ):
    
    # float32 Variable
    if is_depthwise:
        float32_weights = tf.get_variable("float32_weights", [kernel_size, kernel_size, input_channel, 1], tf.float32, initializer = initializer)
        if is_add_biases:
            float32_biases = tf.get_variable("float32_biases" , [input_channel], tf.float32, initializer = initializer)
        else:
            float32_biases = None
    else:
        float32_weights = tf.get_variable("float32_weights", [kernel_size, kernel_size, input_channel, output_channel], tf.float32, initializer = initializer)
        if is_add_biases:
            float32_biases = tf.get_variable("float32_biases" , [output_channel], tf.float32, initializer = initializer)
        else:
            float32_biases = None
        
    tf.add_to_collection("float32_weights", float32_weights)
    if is_add_biases:
        tf.add_to_collection("float32_biases" , float32_biases)
    
    tf.add_to_collection("float32_params" , float32_weights)
    if is_add_biases:
        tf.add_to_collection("float32_params" , float32_biases)
    
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
    
        if is_depthwise:
            final_weights = tf.get_variable("final_weights", [kernel_size, kernel_size, input_channel, 1], tf.float32, initializer=initializer)
            if is_add_biases:
                final_biases = tf.get_variable("final_biases" , [input_channel], tf.float32, initializer = initializer)
            else:
                final_biases = None
        else:
            final_weights = tf.get_variable("final_weights", [kernel_size, kernel_size, input_channel, output_channel], tf.float32, initializer=initializer)
            if is_add_biases:
                final_biases = tf.get_variable("final_biases" , [output_channel], tf.float32, initializer = initializer)
            else:
                final_biases = None
            
        # var_list : Record the variables which will be computed gradients
        tf.add_to_collection("var_list", final_weights)
        if is_add_biases:
            tf.add_to_collection("var_list", final_biases)
        
        assign_final_weights = tf.assign(final_weights, weights_tmp)
        if is_add_biases:
            assign_final_biases = tf.assign(final_biases , biases_tmp)
        tf.add_to_collection("assign_var_list", assign_final_weights)
        if is_add_biases:
            tf.add_to_collection("assign_var_list", assign_final_biases)
        
        return final_weights, final_biases
    else:
        tf.add_to_collection("var_list", float32_weights)
        if is_add_biases:
            tf.add_to_collection("var_list", float32_biases)
            
        return float32_weights, float32_biases

def fixed_padding(
    inputs, 
    kernel_size
    ):

    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])

    return padded_inputs

def conv2d_fixed_padding(
    inputs, 
    filters, 
    kernel_size, 
    stride
    ):
    
    if stride > 1:
        inputs = fixed_padding(inputs, kernel_size)

    outputs = tf.nn.conv2d( 
                input   = inputs, 
                filter  = filters, 
                strides = [1, stride, stride, 1], 
                padding = ('SAME' if stride == 1 else 'VALID'))
    #outputs = tf.layers.conv2d(
    #  inputs             = inputs, 
    #  filters            = filters, 
    #  kernel_size        = kernel_size, 
    #  strides            = stride,
    #  padding            = ('SAME' if stride == 1 else 'VALID'), use_bias=False,
    #  kernel_initializer = tf.variance_scaling_initializer())            
    return outputs
        
def conv2D_Module( 
	net, kernel_size, stride, output_channel, rate, group,
	initializer             ,
    is_training             ,
    is_add_biases           ,
	is_batch_norm           ,
	is_dilated              ,
	is_depthwise            ,
	is_ternary              ,
	is_quantized_activation ,
	IS_TERNARY              ,
	IS_QUANTIZED_ACTIVATION ,
	Activation              ,
	padding                 ,
	Analysis                ,
	scope
	):
    with tf.variable_scope(scope):
        test = {}
        input_channel = net.get_shape().as_list()[-1]
        
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
                        IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
                        padding                 = padding, 
                        name                    = 'Conv')
        
        group_input_channel = int(input_channel / group)
        group_output_channel = int(output_channel / group)
        
        net_tmp_list = []
        for g in range(group):
            print("\033[0;36mgroup%d\033[0m"%(g))
            with tf.variable_scope('group%d'%(g)):
                # Inputs
                net_tmp = net[:, :, :, g*group_input_channel : (g+1)*group_input_channel]
                
                # Variable define
                weights, biases = conv2D_Variable( kernel_size    = kernel_size,
                                                   input_channel  = group_input_channel,		
                                                   output_channel = group_output_channel, 
                                                   initializer    = initializer,
                                                   is_add_biases  = is_add_biases,
                                                   is_ternary     = is_ternary,
                                                   IS_TERNARY     = IS_TERNARY,
                                                   is_depthwise   = is_depthwise)			                                                   
                # Convolution
                if is_dilated:
                    if is_depthwise:
                        ## Show the model
                        print("-> Dilated-Depthwise Conv, Weights:{}, stride:{}" .format(weights.get_shape().as_list(), stride))
                        net_tmp = tf.nn.depthwise_conv2d( input   = net_tmp, 
                                                          filter  = weights, 
                                                          strides = [1, stride, stride, 1], 
                                                          padding = padding, 
                                                          rate    = [rate, rate])	
                    else:
                        ## Show the model
                        print("-> Dilated Conv, Weights:{}, stride:{}" .format(weights.get_shape().as_list(), stride))
                        net_tmp = tf.nn.atrous_conv2d( value   = net_tmp, 
                                                       filters = weights, 
                                                       rate    = rate, 
                                                       padding = padding)
                else:
                    if is_depthwise:
                        ## Show the model
                        print("-> Depthwise Conv, Weights:{}, stride:{}" .format(weights.get_shape().as_list(), stride))
                        net_tmp = tf.nn.depthwise_conv2d( input   = net_tmp, 
                                                          filter  = weights, 
                                                          strides = [1, stride, stride, 1], 
                                                          padding = padding)	
                    else:
                        ## Show the model
                        print("-> Conv, Weights:{}, stride:{}" .format(weights.get_shape().as_list(), stride))
                        net_tmp = conv2d_fixed_padding(net_tmp, weights, kernel_size, stride)
                        
                # Add bias
                if is_add_biases:
                    ## Show the model
                    print("-> Add Biases, Biases:{}".format(biases.get_shape().as_list()))
                    net_tmp = tf.nn.bias_add(net_tmp, biases)
                
                ## For checking the final weights/biases is ternary or not
                tf.add_to_collection("final_weights", weights)
                if is_add_biases:
                    tf.add_to_collection("final_biases", biases)
                
                # List the net_tmp
                net_tmp_list.append(net_tmp)
            
        
                    
                # Ternary Scale
                if IS_TERNARY:
                    ## Show the model
                    print("-> Ternary")
                    with tf.variable_scope('Ternary_Scalse'):
                        ternary_scale = tf.get_variable("ternary_scale", [1], tf.float32, initializer=initializer)
                        tf.add_to_collection("ternary_scale", ternary_scale)
                        net_tmp = tf.multiply(net_tmp, ternary_scale)
    
                # Batch Normalization
                if is_batch_norm == True:
                    ## Show the model
                    print("-> Batch_Norm")
                    net_tmp = batch_norm(net_tmp, is_training)
                
                # Activation
                if Activation == 'ReLU':
                    ## Show the model
                    print("-> ReLU")
                    net_tmp = tf.nn.relu(net_tmp)
                elif Activation == 'Sigmoid':
                    ## Show the model
                    print("-> Sigmoid")
                    net_tmp = tf.nn.sigmoid(net_tmp)
                elif Activation == 'CReLU':
                    ## Show the model
                    print("-> CReLU")
                    net_tmp = CReLU(net_tmp, initializer = initializer)
                else:
                    net_tmp = net_tmp
                    
                # Activation Quantization
                if IS_QUANTIZED_ACTIVATION:
                    ## Show the model
                    print("-> Activation Quantization")
                    quantize_Module(net_tmp, is_quantized_activation)

                # merge every group net together
                if g==0:
                    net_ = net_tmp
                else:
                    net_ = tf.concat([net_, net_tmp], axis=3)
        net = net_
        
    return net

def shortcut_Module( 
    net, 
    destination             ,
    initializer             ,
    is_training             ,
    is_add_biases           ,
    is_projection_shortcut  ,
    is_batch_norm           ,
    is_ternary              ,
    is_quantized_activation ,
    IS_TERNARY              , 	
    IS_QUANTIZED_ACTIVATION ,
    padding                 ,			     
    Analysis                
    ):
    
    [batch,  input_height,  input_width,  input_channel] = net.get_shape().as_list()
    [batch, output_height, output_width, output_channel] = destination.get_shape().as_list()
    
    with tf.variable_scope("shortcut"):
        # Height & Width & Depth
        if input_height!=output_height or input_width!=output_width or input_channel!=output_channel or is_projection_shortcut:
            stride_height = int(input_height / output_height)
            stride_width  = int(input_width  / output_width )
            
            shortcut = conv2D_Module( net, kernel_size=1, stride=stride_height, output_channel=output_channel, rate=1, group=1,
                                      initializer              = initializer              ,
                                      is_training              = is_training              ,
                                      is_add_biases            = is_add_biases            ,
                                      is_batch_norm            = False                    ,
                                      is_dilated               = False                    ,
                                      is_depthwise             = False                    ,
                                      is_ternary               = is_ternary               ,
                                      is_quantized_activation  = is_quantized_activation  ,
                                      IS_TERNARY               = IS_TERNARY               ,
                                      IS_QUANTIZED_ACTIVATION  = IS_QUANTIZED_ACTIVATION  ,
                                      Activation               = None                     ,
                                      padding                  = padding                  ,
                                      Analysis                 = Analysis                 ,
                                      scope                    = "conv_1x1"               )
        else:
            shortcut = net
        
        # Height & Width
        """
        if input_height!=output_height or input_width!=output_width:
            stride_height = input_height / output_height
            stride_width  = input_width  / output_width
            shortcut = tf.nn.avg_pool( value   = shortcut,
                                       ksize   = [1, 3, 3, 1],
                                       strides = [1, stride_height, stride_width, 1],
                                       padding = padding)
                                       
            #shortcut = tf.image.resize_images( 
            #            images = shortcut, 
            #            size   = [tf.constant(output_height), tf.constant(output_width)])
        """
    return shortcut

def inception_Module( 
    net, kernel_size, stride, output_channel, rate, group,
    initializer              ,
    is_training              ,
    is_add_biases            ,
    is_batch_norm            ,
    is_dilated               ,
    is_depthwise             ,
    is_ternary               ,
    is_quantized_activation  ,
    IS_TERNARY               ,
    IS_QUANTIZED_ACTIVATION  ,
    Activation               ,
    padding                  ,
    Analysis                 
    ):
    
    with tf.variable_scope('inception'):
        net_1x1 = conv2D_Module( net, kernel_size=1, stride=stride, output_channel=output_channel, rate=1, group=group,
                                 initializer              = initializer              ,
                                 is_training              = is_training              ,
                                 is_add_biases            = is_add_biases            ,
                                 is_batch_norm            = is_batch_norm            ,
                                 is_dilated               = False                    ,
                                 is_depthwise             = False                    ,
                                 is_ternary               = is_ternary               ,
                                 is_quantized_activation  = is_quantized_activation  ,
                                 IS_TERNARY               = IS_TERNARY               ,
                                 IS_QUANTIZED_ACTIVATION  = IS_QUANTIZED_ACTIVATION  ,
                                 Activation               = Activation               ,
                                 padding                  = padding                  ,
                                 Analysis                 = Analysis                 ,
                                 scope                    = "conv_1x1"               )
        
        net = conv2D_Module( net, kernel_size=kernel_size, stride=stride, output_channel=output_channel, rate=rate, group=group,
                             initializer              = initializer              ,
                             is_training              = is_training              ,
                             is_add_biases            = is_add_biases            ,
                             is_batch_norm            = is_batch_norm            ,
                             is_dilated               = is_dilated               ,
                             is_depthwise             = is_depthwise             ,
                             is_ternary               = is_ternary               ,
                             is_quantized_activation  = is_quantized_activation  ,
                             IS_TERNARY               = IS_TERNARY               ,
                             IS_QUANTIZED_ACTIVATION  = IS_QUANTIZED_ACTIVATION  ,
                             Activation               = Activation               ,
                             padding                  = padding                  ,
                             Analysis                 = Analysis                 ,
                             scope                    = "conv"                   )
        if is_depthwise:
            net = conv2D_Module( net, kernel_size=1, stride=1, output_channel=output_channel, rate=rate, group=group,
                                 initializer              = initializer              ,
                                 is_training              = is_training              ,
                                 is_add_biases            = is_add_biases            ,
                                 is_batch_norm            = is_batch_norm            ,
                                 is_dilated               = False                    ,
                                 is_depthwise             = False                    ,
                                 is_ternary               = is_ternary               ,
                                 is_quantized_activation  = is_quantized_activation  ,
                                 IS_TERNARY               = IS_TERNARY               ,
                                 IS_QUANTIZED_ACTIVATION  = IS_QUANTIZED_ACTIVATION  ,
                                 Activation               = Activation               ,
                                 padding                  = padding                  ,
                                 Analysis                 = Analysis                 ,
                                 scope                    = "depthwise_1x1"          )
                                
        net = tf.add(net, net_1x1)
        
        # -- Analyzer --
        utils.Analyzer( Analysis, 
                        net, 
                        type                    = 'ADD', 
                        IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION, 
                        name                    = 'inception_ADD')
        
        # Activation Quantization
        if IS_QUANTIZED_ACTIVATION:
            quantize_Module(net, is_quantized_activation)
            
    return net
   
def bottleneck_Module( 
    net, kernel_size, stride, internal_channel, output_channel, rate, group,
    initializer             ,
    is_training             ,
    is_add_biases           ,
    is_batch_norm           ,  
    is_dilated              , 
    is_depthwise            ,
    is_inception            ,
    is_ternary              , 
    is_quantized_activation , 
    IS_TERNARY              ,  	
    IS_QUANTIZED_ACTIVATION ,  
    Activation              ,
    padding                 ,
    Analysis                
    ): 
    
    with tf.variable_scope("bottle_neck"):
        net = conv2D_Module( net, kernel_size=1, stride=1, output_channel=internal_channel, rate=rate, group=group,
                             initializer              = initializer              ,
                             is_training              = is_training              ,
                             is_add_biases            = is_add_biases            ,
                             is_batch_norm            = is_batch_norm            ,
                             is_dilated               = is_dilated               ,
                             is_depthwise             = False                    ,
                             is_ternary               = is_ternary               ,
                             is_quantized_activation  = is_quantized_activation  ,
                             IS_TERNARY               = IS_TERNARY               ,
                             IS_QUANTIZED_ACTIVATION  = IS_QUANTIZED_ACTIVATION  ,
                             Activation               = Activation               ,
                             padding                  = padding                  ,
                             Analysis                 = Analysis                 ,
                             scope                    = "Reduction_1x1"          )
        if is_inception:
            net = inception_Module( net, kernel_size=kernel_size, stride=stride, output_channel=internal_channel, rate=rate, group=group,
                                    initializer               = initializer              ,
                                    is_training               = is_training              ,
                                    is_batch_norm             = is_batch_norm            ,
                                    is_dilated                = is_dilated               ,
                                    is_depthwise              = is_depthwise             ,
                                    is_ternary                = is_ternary               ,
                                    is_quantized_activation   = is_quantized_activation  ,
                                    IS_TERNARY                = IS_TERNARY               ,
                                    IS_QUANTIZED_ACTIVATION   = IS_QUANTIZED_ACTIVATION  ,
                                    Activation                = Activation               ,
                                    padding                   = padding                  ,
                                    Analysis                  = Analysis                 )
        else :
            net = conv2D_Module( net, kernel_size=kernel_size, stride=stride, output_channel=internal_channel, rate=rate, group=group,
                                 initializer              = initializer              ,
                                 is_training              = is_training              ,
                                 is_add_biases            = is_add_biases            ,
                                 is_batch_norm            = is_batch_norm            ,
                                 is_dilated               = is_dilated               ,
                                 is_depthwise             = is_depthwise             ,
                                 is_ternary               = is_ternary               ,
                                 is_quantized_activation  = is_quantized_activation  ,
                                 IS_TERNARY               = IS_TERNARY               ,
                                 IS_QUANTIZED_ACTIVATION  = IS_QUANTIZED_ACTIVATION  ,
                                 Activation               = Activation               ,
                                 padding                  = padding                  ,
                                 Analysis                 = Analysis                 ,
                                 scope                    = "conv"                   )
        
        net = conv2D_Module( net, kernel_size=1, stride=1, output_channel=output_channel, rate=rate, group=group,
                             initializer              = initializer              ,
                             is_training              = is_training              ,
                             is_add_biases            = is_add_biases            ,
                             is_batch_norm            = False                    ,
                             is_dilated               = is_dilated               ,
                             is_depthwise             = False                    ,
                             is_ternary               = is_ternary               ,
                             is_quantized_activation  = is_quantized_activation  ,
                             IS_TERNARY               = IS_TERNARY               ,
                             IS_QUANTIZED_ACTIVATION  = IS_QUANTIZED_ACTIVATION  ,
                             Activation               = None                     ,
                             padding                  = padding                  ,
                             Analysis                 = Analysis                 ,
                             scope                    = "Recovery_1x1"           )
    return net
        
def shuffle_Module(
    net, kernel_size, stride, internal_channel, output_channel, rate, group,
    initializer             ,
    is_training             ,
    is_add_biases           ,
    is_batch_norm           ,  
    is_dilated              , 
    is_depthwise            ,
    is_inception            ,
    is_ternary              , 
    is_quantized_activation , 
    IS_TERNARY              ,  	
    IS_QUANTIZED_ACTIVATION ,  
    Activation              ,
    padding                 ,
    Analysis                
    ): 
    
    with tf.variable_scope("shuffle_module"):
        net = conv2D_Module( net, kernel_size=1, stride=1, output_channel=internal_channel, rate=rate, group=group,
                             initializer              = initializer              ,
                             is_training              = is_training              ,
                             is_add_biases            = is_add_biases            ,
                             is_batch_norm            = is_batch_norm            ,
                             is_dilated               = is_dilated               ,
                             is_depthwise             = False                    ,
                             is_ternary               = is_ternary               ,
                             is_quantized_activation  = is_quantized_activation  ,
                             IS_TERNARY               = IS_TERNARY               ,
                             IS_QUANTIZED_ACTIVATION  = IS_QUANTIZED_ACTIVATION  ,
                             Activation               = Activation               ,
                             padding                  = padding                  ,
                             Analysis                 = Analysis                 ,
                             scope                    = "Reduction_1x1"          )

        net = shuffle_net(net, group = group)

        if is_inception:
            net = inception_Module( net, kernel_size=kernel_size, stride=stride, output_channel=internal_channel, rate=rate, group=group,
                                    initializer               = initializer              ,
                                    is_training              = is_training              ,
                                    is_batch_norm             = is_batch_norm            ,
                                    is_dilated                = is_dilated               ,
                                    is_depthwise              = is_depthwise             ,
                                    is_ternary                = is_ternary               ,
                                    is_quantized_activation   = is_quantized_activation  ,
                                    IS_TERNARY                = IS_TERNARY               ,
                                    IS_QUANTIZED_ACTIVATION   = IS_QUANTIZED_ACTIVATION  ,
                                    Activation                = Activation               ,
                                    padding                   = padding                  ,
                                    Analysis                  = Analysis                 )
        else :
            net = conv2D_Module( net, kernel_size=kernel_size, stride=stride, output_channel=internal_channel, rate=rate, group=group,
                                 initializer              = initializer              ,
                                 is_training              = is_training              ,
                                 is_add_biases            = is_add_biases            ,
                                 is_batch_norm            = is_batch_norm            ,
                                 is_dilated               = is_dilated               ,
                                 is_depthwise             = is_depthwise             ,
                                 is_ternary               = is_ternary               ,
                                 is_quantized_activation  = is_quantized_activation  ,
                                 IS_TERNARY               = IS_TERNARY               ,
                                 IS_QUANTIZED_ACTIVATION  = IS_QUANTIZED_ACTIVATION  ,
                                 Activation               = Activation               ,
                                 padding                  = padding                  ,
                                 Analysis                 = Analysis                 ,
                                 scope                    = "conv"                   )
        
        net = conv2D_Module( net, kernel_size=1, stride=1, output_channel=output_channel, rate=rate, group=group,
                             initializer              = initializer              ,
                             is_training              = is_training              ,
                             is_add_biases            = is_add_biases            ,
                             is_batch_norm            = is_batch_norm            ,
                             is_dilated               = is_dilated               ,
                             is_depthwise             = False                    ,
                             is_ternary               = is_ternary               ,
                             is_quantized_activation  = is_quantized_activation  ,
                             IS_TERNARY               = IS_TERNARY               ,
                             IS_QUANTIZED_ACTIVATION  = IS_QUANTIZED_ACTIVATION  ,
                             Activation               = Activation               ,
                             padding                  = padding                  ,
                             Analysis                 = Analysis                 ,
                             scope                    = "Recovery_1x1"           )
    return net
    
def conv2D(	
    net, kernel_size=3, stride=1, internal_channel=64, output_channel=64, rate=1, group=1,
    initializer             = tf.contrib.layers.variance_scaling_initializer(),
    is_training             = False,
    is_add_biases           = True,
    is_bottleneck           = False,      # For Residual
    is_batch_norm           = True,       # For Batch Normalization
    is_dilated              = False,      # For Dilated Convoution
    is_depthwise            = False,      # For Depthwise Convolution
    is_inception            = False,      # For Inception Convolution
    is_shuffle              = False,      # For shuffle unit
    is_ternary              = False,      # (tensor) For weight ternarization
    is_quantized_activation = False,      # (tensor) For activation quantization
    IS_TERNARY              = False,      
    IS_QUANTIZED_ACTIVATION = False,      
    Activation              = 'ReLU',
    padding                 = "SAME",
    Analysis                = None,
    scope                   = "conv"
    ):
		
    with tf.variable_scope(scope):
        #=====================#
        #    shuffle units    #
        #=====================#
        if is_shuffle:
            net = shuffle_Module( net, 
                                  kernel_size             = kernel_size             , 
                                  stride                  = stride                  , 
                                  internal_channel        = internal_channel        , 
                                  output_channel          = output_channel          , 
                                  rate                    = rate                    , 
                                  group                   = group                   ,
                                  initializer             = initializer             ,
                                  is_training             = is_training             ,
                                  is_add_biases           = is_add_biases           ,
                                  is_batch_norm           = is_batch_norm           ,  
                                  is_dilated              = is_dilated              , 
                                  is_depthwise            = is_depthwise            ,
                                  is_inception            = is_inception            ,
                                  is_ternary              = is_ternary              , 
                                  is_quantized_activation = is_quantized_activation , 
                                  IS_TERNARY              = IS_TERNARY              ,  	
                                  IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION ,  
                                  Activation              = Activation              ,
                                  padding                 = padding                 ,
                                  Analysis                = Analysis                )
        else:                       
            #==============================#
            #    Bottleneck Convolution    #
            #==============================#
            if is_bottleneck:
                net = bottleneck_Module( net, 
                                         kernel_size             = kernel_size             , 
                                         stride                  = stride                  , 
                                         internal_channel        = internal_channel        , 
                                         output_channel          = output_channel          , 
                                         rate                    = rate                    , 
                                         group                   = group                   ,
                                         initializer             = initializer             ,
                                         is_training             = is_training             ,
                                         is_add_biases           = is_add_biases           ,
                                         is_batch_norm           = is_batch_norm           , 
                                         is_dilated              = is_dilated              , 
                                         is_depthwise            = is_depthwise            , 
                                         is_inception            = is_inception            ,
                                         is_ternary              = is_ternary              , 
                                         is_quantized_activation = is_quantized_activation , 
                                         IS_TERNARY              = IS_TERNARY              ,  	
                                         IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION ,  
                                         Activation              = Activation              ,
                                         padding                 = padding                 ,
                                         Analysis                = Analysis                )
            else:  
            #=============================#
            #    Inception Convolution    #
            #=============================#
                if is_inception:
                    net = inception_Module( net, 
                                            kernel_size              = kernel_size              , 
                                            stride                   = stride                   , 
                                            output_channel           = output_channel           , 
                                            rate                     = rate                     , 
                                            group                    = group                    ,
                                            initializer              = initializer              ,
                                            is_training              = is_training              ,
                                            is_add_biases            = is_add_biases           ,
                                            is_batch_norm            = is_batch_norm            ,
                                            is_dilated               = is_dilated               ,
                                            is_depthwise             = is_depthwise             ,
                                            is_ternary               = is_ternary               ,
                                            is_quantized_activation  = is_quantized_activation  ,
                                            IS_TERNARY               = IS_TERNARY               ,
                                            IS_QUANTIZED_ACTIVATION  = IS_QUANTIZED_ACTIVATION  ,
                                            Activation               = Activation               ,
                                            padding                  = padding                  ,
                                            Analysis                 = Analysis                 )
            #==========================#
            #    Normal Convolution    #
            #==========================#                                        
                else:
                    net = conv2D_Module( net, 
                                         kernel_size             = kernel_size              , 
                                         stride                  = stride                   , 
                                         output_channel          = output_channel           , 
                                         rate                    = rate                     , 
                                         group                   = group                    ,
                                         initializer             = initializer              ,
                                         is_training             = is_training              ,
                                         is_add_biases           = is_add_biases            ,
                                         is_batch_norm           = is_batch_norm            ,
                                         is_dilated              = is_dilated               ,
                                         is_depthwise            = is_depthwise             ,
                                         is_ternary              = is_ternary               ,
                                         is_quantized_activation = is_quantized_activation  ,
                                         IS_TERNARY              = IS_TERNARY               ,
                                         IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION  ,
                                         Activation              = Activation               ,
                                         padding                 = padding                  ,
                                         Analysis                = Analysis                 ,
                                         scope                   = "conv"                   )
                    if is_depthwise:
                        net = conv2D_Module( net, 
                                             kernel_size             = 1                        , 
                                             stride                  = 1                        , 
                                             output_channel          = output_channel           , 
                                             rate                    = rate                     , 
                                             group                   = 1                        ,
                                             initializer             = initializer              ,
                                             is_training             = is_training              ,
                                             is_add_biases           = is_add_biases            ,
                                             is_batch_norm           = is_batch_norm            ,
                                             is_dilated              = False                    ,
                                             is_depthwise            = False                    ,
                                             is_ternary              = is_ternary               ,
                                             is_quantized_activation = is_quantized_activation  ,
                                             IS_TERNARY              = IS_TERNARY               ,
                                             IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION  ,
                                             Activation              = Activation               ,
                                             padding                 = padding                  ,
                                             Analysis                = Analysis                 ,
                                             scope                   = "depthwise_1x1"          )
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
        
        net, indices = tf.nn.max_pool_with_argmax( input   = net, 
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

#           # Combine the net
#        for g in range(group):
#            with tf.variable_scope('group%d'%(g)):
#                if group >= 4:
#                    net_tmp = tf.add(net_tmp_list[g], net_tmp_list[(g+1)%group])
#                else:
#                    net_tmp = net_tmp_list[g]
