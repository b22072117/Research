import tensorflow as tf
import numpy as np
import scipy.ndimage
import pdb
import math
from PIL import Image
from scipy import misc
import utils


def residual_50(net, class_num, is_training, is_testing, reuse=None, scope="Residual_50"):
	with tf.variable_scope(scope, reuse=reuse):
		with tf.variable_scope("conv1_x"):
			net = utils.conv2D(net, kernel_size=7, stride=2, output_channel=64, rate=1,
							is_shortcut		= False, 
							is_bottleneck	= False, 
							is_batch_norm	= True, 
							is_training		= is_training, 
							is_testing		= is_testing, 
							is_dilated		= False, 
							scope			= "conv1")
		with tf.variable_scope("conv2_x"), tf.device("/gpu:0"):
			net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
			net = utils.conv2D(net, kernel_size=3, stride=1, internal_channel=64, output_channel=256, rate=1,
							is_shortcut		= True, 
							is_bottleneck	= True, 
							is_batch_norm	= True, 
							is_training		= is_training, 
							is_testing		= is_testing, 
							is_dilated		= False, 
							scope			= "conv1")
							
			net = utils.conv2D(net, kernel_size=3, stride=1, internal_channel=64, output_channel=256, rate=1,
							is_shortcut		= True, 
							is_bottleneck	= True, 
							is_batch_norm	= True, 
							is_training		= is_training, 
							is_testing		= is_testing, 
							is_dilated		= False, 
							scope			= "conv2")
							
			net = utils.conv2D(net, kernel_size=3, stride=1, internal_channel=64, output_channel=256, rate=1,
							is_shortcut		= True, 
							is_bottleneck	= True, 
							is_batch_norm	= True, 
							is_training		= is_training, 
							is_testing		= is_testing, 
							is_dilated		= False, 
							scope			= "conv3")
							
		with tf.variable_scope("conv3_x"), tf.device("/gpu:1"):
			net = utils.conv2D(net, kernel_size=3, stride=2, internal_channel=128, output_channel=512, rate=1,
							is_shortcut		= True, 
							is_bottleneck	= True, 
							is_batch_norm	= True, 
							is_training		= is_training, 
							is_testing		= is_testing, 
							is_dilated		= False, 
							scope			= "conv1")
							
			net = utils.conv2D(net, kernel_size=3, stride=1, internal_channel=128, output_channel=512, rate=1,
							is_shortcut		= True, 
							is_bottleneck	= True, 
							is_batch_norm	= True, 
							is_training		= is_training, 
							is_testing		= is_testing, 
							is_dilated		= False, 
							scope			= "conv2")
							
			net = utils.conv2D(net, kernel_size=3, stride=1, internal_channel=128, output_channel=512, rate=1,
							is_shortcut		= True, 
							is_bottleneck	= True, 
							is_batch_norm	= True, 
							is_training		= is_training, 
							is_testing		= is_testing, 
							is_dilated		= False, 
							scope			= "conv3")
							
		with tf.variable_scope("conv4_x"), tf.device("/gpu:2"):
			net = utils.conv2D(net, kernel_size=3, stride=2, internal_channel=256, output_channel=1024, rate=1,
							is_shortcut		= True, 
							is_bottleneck	= True, 
							is_batch_norm	= True, 
							is_training		= is_training, 
							is_testing		= is_testing, 
							is_dilated		= False, 
							scope			= "conv1")
							
			net = utils.conv2D(net, kernel_size=3, stride=1, internal_channel=256, output_channel=1024, rate=1,
							is_shortcut		= True, 
							is_bottleneck	= True, 
							is_batch_norm	= True, 
							is_training		= is_training, 
							is_testing		= is_testing, 
							is_dilated		= False, 
							scope			= "conv2")
							
			net = utils.conv2D(net, kernel_size=3, stride=1, internal_channel=256, output_channel=1024, rate=1,
							is_shortcut		= True, 
							is_bottleneck	= True, 
							is_batch_norm	= True, 
							is_training		= is_training, 
							is_testing		= is_testing, 
							is_dilated		= False, 
							scope			= "conv3")
														
		with tf.variable_scope("conv5_x"), tf.device("/gpu:3"):
			net = utils.conv2D(net, kernel_size=3, stride=2, internal_channel=512, output_channel=2048, rate=1,
							is_shortcut		= True, 
							is_bottleneck	= True, 
							is_batch_norm	= True, 
							is_training		= is_training, 
							is_testing		= is_testing, 
							is_dilated		= False, 
							scope			= "conv1")
							
			net = utils.conv2D(net, kernel_size=3, stride=1, internal_channel=512, output_channel=2048, rate=1,
							is_shortcut		= True, 
							is_bottleneck	= True, 
							is_batch_norm	= True, 
							is_training		= is_training, 
							is_testing		= is_testing, 
							is_dilated		= False, 
							scope			= "conv2")
							
			net = utils.conv2D(net, kernel_size=3, stride=1, internal_channel=512, output_channel=2048, rate=1,
							is_shortcut		= True, 
							is_bottleneck	= True, 
							is_batch_norm	= True, 
							is_training		= is_training, 
							is_testing		= is_testing, 
							is_dilated		= False, 
							scope			= "conv3")
							
		with tf.variable_scope("Average_Pooling"):
			net = tf.reduce_mean(net, [1, 2], keep_dims=True)
			
		with tf.variable_scope("fc1"):
			net = utils.conv2D(net, kernel_size=1, stride=1, output_channel=class_num, rate=1,
							is_shortcut		= False, 
							is_bottleneck	= False, 
							is_batch_norm	= False, 
							is_training		= is_training, 
							is_testing		= is_testing, 
							is_dilated		= False, 
							scope			= "conv1")
	return net
	
def PSPNet(net, class_num, is_training, is_testing, reuse=None, scope="PSPNet"):
	input_shape = net.get_shape().as_list()
	with tf.variable_scope(scope, reuse=reuse):
		with tf.variable_scope("ResNet50"):
			with tf.variable_scope("root_block"):
				net = utils.conv2D(net, kernel_size=3, stride=2, output_channel=64, rate=1,
								is_shortcut		= False, 
								is_bottleneck	= False, 
								is_batch_norm	= True, 
								is_training		= is_training, 
								is_testing		= is_testing, 
								is_dilated		= False, 
								scope			= "conv1_1")

				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
								is_shortcut		= False, 
								is_bottleneck	= False, 
								is_batch_norm	= True, 
								is_training		= is_training, 
								is_testing		= is_testing, 
								is_dilated		= False, 
								scope			= "conv1_2")

				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=128, rate=1,
								is_shortcut		= False, 
								is_bottleneck	= False, 
								is_batch_norm	= True, 
								is_training		= is_training, 
								is_testing		= is_testing, 
								is_dilated		= False, 
								scope			= "conv1_3")
				
				net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

			with tf.variable_scope("Block1"), tf.device("/gpu:0"):
				net = utils.conv2D(net, kernel_size=3, stride=2, internal_channel=64, output_channel=256, rate=1,
								is_shortcut		= True, 
								is_bottleneck	= True, 
								is_batch_norm	= True, 
								is_training		= is_training, 
								is_testing		= is_testing, 
								is_dilated		= False, 
								scope			= "conv2_r1")
								
				net = utils.conv2D(net, kernel_size=3, stride=1, internal_channel=64, output_channel=256, rate=1,
								is_shortcut		= True, 
								is_bottleneck	= True, 
								is_batch_norm	= True, 
								is_training		= is_training, 
								is_testing		= is_testing, 
								is_dilated		= False, 
								scope			= "conv2_r2")
								
				net = utils.conv2D(net, kernel_size=3, stride=1, internal_channel=64, output_channel=256, rate=1,
								is_shortcut		= True, 
								is_bottleneck	= True, 
								is_batch_norm	= True, 
								is_training		= is_training, 
								is_testing		= is_testing, 
								is_dilated		= False, 
								scope			= "conv2_r3")
								
			with tf.variable_scope("Block2"), tf.device("/gpu:1"):
				net = utils.conv2D(net, kernel_size=3, stride=2, internal_channel=128, output_channel=512, rate=1,
								is_shortcut		= True, 
								is_bottleneck	= True, 
								is_batch_norm	= True, 
								is_training		= is_training, 
								is_testing		= is_testing, 
								is_dilated		= False, 
								scope			= "conv3_r1")
								
				net = utils.conv2D(net, kernel_size=3, stride=1, internal_channel=128, output_channel=512, rate=1,
								is_shortcut		= True, 
								is_bottleneck	= True, 
								is_batch_norm	= True, 
								is_training		= is_training, 
								is_testing		= is_testing, 
								is_dilated		= False, 
								scope			= "conv3_r2")
								
				net = utils.conv2D(net, kernel_size=3, stride=1, internal_channel=128, output_channel=512, rate=1,
								is_shortcut		= True, 
								is_bottleneck	= True, 
								is_batch_norm	= True, 
								is_training		= is_training, 
								is_testing		= is_testing, 
								is_dilated		= False, 
								scope			= "conv3_r3")
								
				net = utils.conv2D(net, kernel_size=3, stride=1, internal_channel=128, output_channel=512, rate=1,
								is_shortcut		= True, 
								is_bottleneck	= True, 
								is_batch_norm	= True, 
								is_training		= is_training, 
								is_testing		= is_testing, 
								is_dilated		= False, 
								scope			= "conv3_r4")

			with tf.variable_scope("Block3"), tf.device("/gpu:2"):
				net = utils.conv2D(net, kernel_size=3, stride=1, internal_channel=256, output_channel=1024, rate=2,
								is_shortcut		= True, 
								is_bottleneck	= True, 
								is_batch_norm	= True, 
								is_training		= is_training, 
								is_testing		= is_testing, 
								is_dilated		= True, 
								scope			= "conv4_r1")
								
				net = utils.conv2D(net, kernel_size=3, stride=1, internal_channel=256, output_channel=1024, rate=2,
								is_shortcut		= True, 
								is_bottleneck	= True, 
								is_batch_norm	= True, 
								is_training		= is_training, 
								is_testing		= is_testing, 
								is_dilated		= True, 
								scope			= "conv4_r2")
								
				net = utils.conv2D(net, kernel_size=3, stride=1, internal_channel=256, output_channel=1024, rate=2,
								is_shortcut		= True, 
								is_bottleneck	= True, 
								is_batch_norm	= True, 
								is_training		= is_training, 
								is_testing		= is_testing, 
								is_dilated		= True, 
								scope			= "conv4_r3")
															
				net = utils.conv2D(net, kernel_size=3, stride=1, internal_channel=256, output_channel=1024, rate=2,
								is_shortcut		= True, 
								is_bottleneck	= True, 
								is_batch_norm	= True, 
								is_training		= is_training, 
								is_testing		= is_testing, 
								is_dilated		= True, 
								scope			= "conv4_r4")
								
				net = utils.conv2D(net, kernel_size=3, stride=1, internal_channel=256, output_channel=1024, rate=2,
								is_shortcut		= True, 
								is_bottleneck	= True, 
								is_batch_norm	= True, 
								is_training		= is_training, 
								is_testing		= is_testing, 
								is_dilated		= True, 
								scope			= "conv4_r5")
								
				net = utils.conv2D(net, kernel_size=3, stride=1, internal_channel=256, output_channel=1024, rate=2,
								is_shortcut		= True, 
								is_bottleneck	= True, 
								is_batch_norm	= True, 
								is_training		= is_training, 
								is_testing		= is_testing, 
								is_dilated		= True, 
								scope			= "conv4_r6")

			with tf.variable_scope("Block4"), tf.device("/gpu:2"):
				net = utils.conv2D(net, kernel_size=3, stride=1, internal_channel=512, output_channel=2048, rate=4,
								is_shortcut		= True, 
								is_bottleneck	= True, 
								is_batch_norm	= True, 
								is_training		= is_training, 
								is_testing		= is_testing, 
								is_dilated		= True, 
								scope			= "conv5_r1")
								
				net = utils.conv2D(net, kernel_size=3, stride=1, internal_channel=512, output_channel=2048, rate=4,
								is_shortcut		= True, 
								is_bottleneck	= True, 
								is_batch_norm	= True, 
								is_training		= is_training, 
								is_testing		= is_testing, 
								is_dilated		= True, 
								scope			= "conv5_r2")
								
				net = utils.conv2D(net, kernel_size=3, stride=1, internal_channel=512, output_channel=2048, rate=4,
								is_shortcut		= True, 
								is_bottleneck	= True, 
								is_batch_norm	= True, 
								is_training		= is_training, 
								is_testing		= is_testing, 
								is_dilated		= True, 
								scope			= "conv5_r3")
								
		with tf.variable_scope("Pyramid_Average_Pooling"):
			net = utils.Pyramid_Pooling(net, strides=[60, 30, 20 ,10], output_channel=512,
								is_training		= is_training,
								is_testing		= is_testing)

			net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=512, rate=1,
								is_shortcut		= False, 
								is_bottleneck	= False, 
								is_batch_norm	= True, 
								is_training		= is_training, 
								is_testing		= is_testing, 
								is_dilated		= False, 
								scope			= "conv6_1")
			#net = tf.cond(is_training, lambda: tf.nn.dropout(net, keep_prob=0.9), lambda: net)
			
			net = utils.conv2D(net, kernel_size=1, stride=1, output_channel=class_num, rate=1,
								is_shortcut		= False, 
								is_bottleneck	= False, 
								is_batch_norm	= False, 
								is_training		= is_training, 
								is_testing		= is_testing, 
								is_dilated		= False, 
								scope			= "conv6_2")

			net = tf.image.resize_images(net, [input_shape[1], input_shape[2]])

	return net
	
def SegNet_VGG_16(net, class_num, is_training, is_testing, reuse=None, scope="SegNet_VGG_16"):
	with tf.variable_scope(scope, reuse=reuse):
		with tf.variable_scope("encoder"):
			with tf.variable_scope("224X224"): # 1/1
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut		= False, 
							is_bottleneck	= False, 
							is_batch_norm	= True, 
							is_training		= is_training, 
							is_testing		= is_testing, 
							is_dilated		= False, 
							scope			= "conv1")
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut		= False, 
							is_bottleneck	= False, 
							is_batch_norm	= True, 
							is_training		= is_training, 
							is_testing		= is_testing, 
							is_dilated		= False, 
							scope			= "conv2")	
							
				net, indices1, output_shape1 = utils.indice_pool(net, stride=2, scope="Pool1")
				
			with tf.variable_scope("112X112"): # 1/2
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=128, rate=1,
							is_shortcut		= False, 
							is_bottleneck	= False, 
							is_batch_norm	= True, 
							is_training		= is_training, 
							is_testing		= is_testing, 
							is_dilated		= False, 
							scope			= "conv1")
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=128, rate=1,
							is_shortcut		= False, 
							is_bottleneck	= False, 
							is_batch_norm	= True, 
							is_training		= is_training, 
							is_testing		= is_testing, 
							is_dilated		= False, 
							scope			= "conv2")	
							
				net, indices2, output_shape2 = utils.indice_pool(net, stride=2, scope="Pool2")
				
			with tf.variable_scope("56X56"): # 1/4
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=256, rate=1, 
							is_shortcut		= False, 
							is_bottleneck	= False, 
							is_batch_norm	= True, 
							is_training		= is_training, 
							is_testing		= is_testing, 
							is_dilated		= False, 
							scope			= "conv1")
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=256, rate=1,
							is_shortcut		= False, 
							is_bottleneck	= False, 
							is_batch_norm	= True, 
							is_training		= is_training, 
							is_testing		= is_testing, 
							is_dilated		= False, 
							scope			= "conv2")	
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=256, rate=1,
							is_shortcut		= False, 
							is_bottleneck	= False, 
							is_batch_norm	= True, 
							is_training		= is_training, 
							is_testing		= is_testing, 
							is_dilated		= False, 
							scope			= "conv3")	
							
				net, indices3, output_shape3 = utils.indice_pool(net, stride=2, scope="Pool3")
				
			with tf.variable_scope("28X28"): # 1/8
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=512, rate=1,
							is_shortcut		= False, 
							is_bottleneck	= False, 
							is_batch_norm	= True, 
							is_training		= is_training, 
							is_testing		= is_testing, 
							is_dilated		= False, 
							scope			= "conv1")
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=512, rate=1,
							is_shortcut		= False, 
							is_bottleneck	= False, 
							is_batch_norm	= True, 
							is_training		= is_training, 
							is_testing		= is_testing, 
							is_dilated		= False, 
							scope			= "conv2")	
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=512, rate=1,
							is_shortcut		= False, 
							is_bottleneck	= False, 
							is_batch_norm	= True, 
							is_training		= is_training, 
							is_testing		= is_testing, 
							is_dilated		= False, 
							scope			= "conv3")		
							
				net, indices4, output_shape4 = utils.indice_pool(net, stride=2, scope="Pool4")
				
			with tf.variable_scope("14X14"): # 1/16
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=512, rate=1,
							is_shortcut		= False, 
							is_bottleneck	= False, 
							is_batch_norm	= True, 
							is_training		= is_training, 
							is_testing		= is_testing, 
							is_dilated		= False, 
							scope			= "conv1")
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=512, rate=1,
							is_shortcut		= False, 
							is_bottleneck	= False, 
							is_batch_norm	= True, 
							is_training		= is_training, 
							is_testing		= is_testing, 
							is_dilated		= False, 
							scope			= "conv2")	
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=512, rate=1,
							is_shortcut		= False, 
							is_bottleneck	= False, 
							is_batch_norm	= True, 
							is_training		= is_training, 
							is_testing		= is_testing, 
							is_dilated		= False, 
							scope			= "conv3")		
							
				net, indices5, output_shape5 = utils.indice_pool(net, stride=2, scope="Pool5")
						
		with tf.variable_scope("decoder"):
			with tf.variable_scope("14X14_D"): # 1/ # conv5_D
				net = utils.indice_unpool(net, stride=2, output_shape=output_shape5, indices=indices5, scope="unPool5")
				
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=512, rate=1,
							is_shortcut		= False, 
							is_bottleneck	= False, 
							is_batch_norm	= True, 
							is_training		= is_training, 
							is_testing		= is_testing, 
							is_dilated		= False, 
							scope			= "conv1")
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=512, rate=1,
							is_shortcut		= False, 
							is_bottleneck	= False, 
							is_batch_norm	= True, 
							is_training		= is_training, 
							is_testing		= is_testing, 
							is_dilated		= False, 
							scope			= "conv2")	
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=512, rate=1,
							is_shortcut		= False, 
							is_bottleneck	= False, 
							is_batch_norm	= True, 
							is_training		= is_training, 
							is_testing		= is_testing, 
							is_dilated		= False, 
							scope			= "conv3")							
				
			with tf.variable_scope("28X28_D"): # 1/8 # conv4_D
				net = utils.indice_unpool(net, stride=2, output_shape=output_shape4, indices=indices4, scope="unPool4")
				
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=512, rate=1,
							is_shortcut		= False, 
							is_bottleneck	= False, 
							is_batch_norm	= True, 
							is_training		= is_training, 
							is_testing		= is_testing, 
							is_dilated		= False, 
							scope			= "conv1")
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=512, rate=1,
							is_shortcut		= False, 
							is_bottleneck	= False, 
							is_batch_norm	= True, 
							is_training		= is_training, 
							is_testing		= is_testing, 
							is_dilated		= False, 
							scope			= "conv2")	
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=256, rate=1,
							is_shortcut		= False, 
							is_bottleneck	= False, 
							is_batch_norm	= True, 
							is_training		= is_training, 
							is_testing		= is_testing, 
							is_dilated		= False, 
							scope			= "conv3")
				
			with tf.variable_scope("56X56_D"): # 1/4 # conv3_D
				net = utils.indice_unpool(net, stride=2, output_shape=output_shape3, indices=indices3, scope="unPool3")
				
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=256, rate=1,
							is_shortcut		= False, 
							is_bottleneck	= False, 
							is_batch_norm	= True, 
							is_training		= is_training, 
							is_testing		= is_testing, 
							is_dilated		= False, 
							scope			= "conv1")
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=256, rate=1,
							is_shortcut		= False, 
							is_bottleneck	= False, 
							is_batch_norm	= True, 
							is_training		= is_training, 
							is_testing		= is_testing, 
							is_dilated		= False, 
							scope			= "conv2")
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=128, rate=1,
							is_shortcut		= False, 
							is_bottleneck	= False, 
							is_batch_norm	= True, 
							is_training		= is_training, 
							is_testing		= is_testing, 
							is_dilated		= False, 
							scope			= "conv3")
							
			with tf.variable_scope("112X112_D"): # 1/2 # conv2_D
				net = utils.indice_unpool(net, stride=2, output_shape=output_shape2, indices=indices2, scope="unPool2")
				
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=128, rate=1,
							is_shortcut		= False, 
							is_bottleneck	= False, 
							is_batch_norm	= True, 
							is_training		= is_training, 
							is_testing		= is_testing, 
							is_dilated		= False, 
							scope			= "conv1")
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut		= False, 
							is_bottleneck	= False, 
							is_batch_norm	= True, 
							is_training		= is_training, 
							is_testing		= is_testing, 
							is_dilated		= False, 
							scope			= "conv2")	
				
			with tf.variable_scope("224X224_D"): # 1/1 # conv1_D
				net = utils.indice_unpool(net, stride=2, output_shape=output_shape1, indices=indices1, scope="unPool1")
				
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut		= False, 
							is_bottleneck	= False, 
							is_batch_norm	= True, 
							is_training		= is_training, 
							is_testing		= is_testing, 
							is_dilated		= False, 
							scope			= "conv1")
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=class_num, rate=1,
							is_shortcut		= False, 
							is_bottleneck	= False, 
							is_batch_norm	= False, 
							is_training		= is_training, 
							is_testing		= is_testing, 
							is_dilated		= False, 
							scope			= "conv2")	
	return net
	
	
	
	
	
	
	
	
	
	
