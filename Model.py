import tensorflow as tf
import numpy as np
import scipy.ndimage
import pdb
import math
from PIL import Image
from scipy import misc
import utils


def Discriminator(net, is_training, is_testing, reuse=None, scope="Discriminator"):
	with tf.variable_scope(scope, reuse=reuse):
		net = residual_50(net, 
						class_num		= 1, 
						is_training		= is_training, 
						is_testing		= is_testing)
	return net

def VGG_16(net, class_num, is_training, is_testing, is_ternary, is_quantized_activation, IS_TERNARY, IS_QUANTIZED_ACTIVATION, FILE, reuse=None, scope="VGG_16"):
	with tf.variable_scope(scope, reuse=reuse):
		Analysis = utils.Analyzer({}, net, type='DATA', name='Input')
		with tf.variable_scope("224X224"): # 1/1
			net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
						is_shortcut				= False, 
						is_bottleneck			= False, 
						is_batch_norm			= True, 
						is_training				= is_training, 
						is_testing				= is_testing, 
						is_dilated				= False, 
						is_ternary      		= is_ternary,
						is_quantized_activation = is_quantized_activation,
						IS_TERNARY				= IS_TERNARY,
						IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
						Analysis				= Analysis,
						scope					= "conv1")
						
			net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
						is_shortcut				= False, 
						is_bottleneck			= False, 
						is_batch_norm			= True, 
						is_training				= is_training, 
						is_testing				= is_testing, 
						is_dilated				= False, 
						is_ternary      		= is_ternary,
						is_quantized_activation = is_quantized_activation,
						IS_TERNARY				= IS_TERNARY,
						IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
						Analysis				= Analysis,
						scope					= "conv2")	
						
			net, indices1, output_shape1, Analysis = utils.indice_pool(net, stride=2, Analysis=Analysis, scope="Pool1")
			
		with tf.variable_scope("112X112"): # 1/2
			net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=128, rate=1,
						is_shortcut				= False, 
						is_bottleneck			= False, 
						is_batch_norm			= True, 
						is_training				= is_training, 
						is_testing				= is_testing, 
						is_dilated				= False, 
						is_ternary      		= is_ternary,
						is_quantized_activation = is_quantized_activation,
						IS_TERNARY				= IS_TERNARY,
						IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
						Analysis				= Analysis,
						scope					= "conv1")
						
			net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=128, rate=1,
						is_shortcut				= False, 
						is_bottleneck			= False, 
						is_batch_norm			= True, 
						is_training				= is_training, 
						is_testing				= is_testing, 
						is_dilated				= False, 
						is_ternary      		= is_ternary,
						is_quantized_activation = is_quantized_activation,
						IS_TERNARY				= IS_TERNARY,
						IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
						Analysis				= Analysis,
						scope					= "conv2")	
						
			net, indices2, output_shape2, Analysis = utils.indice_pool(net, stride=2, Analysis=Analysis, scope="Pool2")
			
		with tf.variable_scope("56X56"): # 1/4
			net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=256, rate=1, 
						is_shortcut				= False, 
						is_bottleneck			= False, 
						is_batch_norm			= True, 
						is_training				= is_training, 
						is_testing				= is_testing, 
						is_dilated				= False, 
						is_ternary      		= is_ternary,
						is_quantized_activation = is_quantized_activation,
						IS_TERNARY				= IS_TERNARY,
						IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
						Analysis				= Analysis,
						scope					= "conv1")
						
			net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=256, rate=1,
						is_shortcut				= False, 
						is_bottleneck			= False, 
						is_batch_norm			= True, 
						is_training				= is_training, 
						is_testing				= is_testing, 
						is_dilated				= False, 
						is_ternary      		= is_ternary,
						is_quantized_activation = is_quantized_activation,
						IS_TERNARY				= IS_TERNARY,
						IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
						Analysis				= Analysis,
						scope					= "conv2")	
						
			net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=256, rate=1,
						is_shortcut				= False, 
						is_bottleneck			= False, 
						is_batch_norm			= True, 
						is_training				= is_training, 
						is_testing				= is_testing, 
						is_dilated				= False, 
						is_ternary      		= is_ternary,
						is_quantized_activation = is_quantized_activation,
						IS_TERNARY				= IS_TERNARY,
						IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
						Analysis				= Analysis,
						scope					= "conv3")	
						
			net, indices3, output_shape3, Analysis = utils.indice_pool(net, stride=2, Analysis=Analysis, scope="Pool3")
			
		with tf.variable_scope("28X28"): # 1/8
			net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=512, rate=1,
						is_shortcut				= False, 
						is_bottleneck			= False, 
						is_batch_norm			= True, 
						is_training				= is_training, 
						is_testing				= is_testing, 
						is_dilated				= False, 
						is_ternary      		= is_ternary,
						is_quantized_activation = is_quantized_activation,
						IS_TERNARY				= IS_TERNARY,
						IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
						Analysis				= Analysis,
						scope					= "conv1")
						
			net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=512, rate=1,
						is_shortcut				= False, 
						is_bottleneck			= False, 
						is_batch_norm			= True, 
						is_training				= is_training, 
						is_testing				= is_testing, 
						is_dilated				= False, 
						is_ternary      		= is_ternary,
						is_quantized_activation = is_quantized_activation,
						IS_TERNARY				= IS_TERNARY,
						IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
						Analysis				= Analysis,
						scope					= "conv2")	
						
			net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=512, rate=1,
						is_shortcut				= False, 
						is_bottleneck			= False, 
						is_batch_norm			= True, 
						is_training				= is_training, 
						is_testing				= is_testing, 
						is_dilated				= False, 
						is_ternary      		= is_ternary,
						is_quantized_activation = is_quantized_activation,
						IS_TERNARY				= IS_TERNARY,
						IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
						Analysis				= Analysis,
						scope					= "conv3")		
						
			net, indices4, output_shape4, Analysis = utils.indice_pool(net, stride=2, Analysis=Analysis, scope="Pool4")
			
		with tf.variable_scope("14X14"): # 1/16
			net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=512, rate=1,
						is_shortcut				= False, 
						is_bottleneck			= False, 
						is_batch_norm			= True, 
						is_training				= is_training, 
						is_testing				= is_testing, 
						is_dilated				= False, 
						is_ternary      		= is_ternary,
						is_quantized_activation = is_quantized_activation,
						IS_TERNARY				= IS_TERNARY,
						IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
						Analysis				= Analysis,
						scope					= "conv1")
						
			net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=512, rate=1,
						is_shortcut				= False, 
						is_bottleneck			= False, 
						is_batch_norm			= True, 
						is_training				= is_training, 
						is_testing				= is_testing, 
						is_dilated				= False, 
						is_ternary      		= is_ternary,
						is_quantized_activation = is_quantized_activation,
						IS_TERNARY				= IS_TERNARY,
						IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
						Analysis				= Analysis,
						scope					= "conv2")	
						
			net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=512, rate=1,
						is_shortcut				= False, 
						is_bottleneck			= False, 
						is_batch_norm			= True, 
						is_training				= is_training, 
						is_testing				= is_testing, 
						is_dilated				= False, 
						is_ternary      		= is_ternary,
						is_quantized_activation = is_quantized_activation,
						IS_TERNARY				= IS_TERNARY,
						IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
						Analysis				= Analysis,
						scope					= "conv3")		
						
			net, indices5, output_shape5, Analysis = utils.indice_pool(net, stride=2, Analysis=Analysis, scope="Pool5")
		with tf.variable_scope("7X7"): # 1/32 Fully Connected Layers
			net = utils.conv2D(net, kernel_size=7, stride=1, output_channel=4096, rate=1,
						is_shortcut				= False, 
						is_bottleneck			= False, 
						is_batch_norm			= True, 
						is_training				= is_training, 
						is_testing				= is_testing, 
						is_dilated				= False, 
						is_ternary     			= is_ternary,
						is_quantized_activation = is_quantized_activation,
						IS_TERNARY				= IS_TERNARY,
						IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
						Analysis				= Analysis,
						padding					= "VALID",
						scope					= "fc1")		

			net = utils.conv2D(net, kernel_size=1, stride=1, output_channel=4096, rate=1,
						is_shortcut				= False, 
						is_bottleneck			= False, 
						is_batch_norm			= True, 
						is_training				= is_training, 
						is_testing				= is_testing, 
						is_dilated				= False, 
						is_ternary      		= is_ternary,
						is_quantized_activation = is_quantized_activation,
						IS_TERNARY				= IS_TERNARY,
						IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
						Analysis				= Analysis,
						scope					= "fc2")		

			net = utils.conv2D(net, kernel_size=1, stride=1, output_channel=class_num, rate=1,
						is_shortcut				= False, 
						is_bottleneck			= False, 
						is_batch_norm			= False, 
						is_training				= is_training, 
						is_testing				= is_testing, 
						is_dilated				= False, 
						is_ternary      		= is_ternary,
						is_quantized_activation = is_quantized_activation,
						IS_TERNARY				= IS_TERNARY,
						IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
						Analysis				= Analysis,
						scope					= "fc3")		
						
		utils.Save_Analyzsis_as_csv(Analysis, FILE)
	return net

def residual_50(net, class_num, is_training, is_testing, is_ternary, is_quantized_activation, IS_TERNARY, IS_QUANTIZED_ACTIVATION, FILE, reuse=None, scope="Residual_50"):
	with tf.variable_scope(scope, reuse=reuse):
		Analysis = utils.Analyzer({}, net, type='DATA', name='Input')
		with tf.variable_scope("conv1_x"):
			net = utils.conv2D(net, kernel_size=7, stride=2, output_channel=64, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_residual				= False,
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
		with tf.variable_scope("conv2_x"), tf.device("/gpu:0"):
			net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
			net = utils.conv2D(net, kernel_size=3, stride=1, internal_channel=64, output_channel=256, rate=1,
							is_shortcut				= True, 
							is_bottleneck			= True, 
							is_residual				= True,
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
			net = utils.conv2D(net, kernel_size=3, stride=1, internal_channel=64, output_channel=256, rate=1,
							is_shortcut				= True, 
							is_bottleneck			= True, 
							is_residual				= True,
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")
							
			net = utils.conv2D(net, kernel_size=3, stride=1, internal_channel=64, output_channel=256, rate=1,
							is_shortcut				= True, 
							is_bottleneck			= True, 
							is_residual				= True,
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv3")
							
		with tf.variable_scope("conv3_x"), tf.device("/gpu:1"):
			net = utils.conv2D(net, kernel_size=3, stride=2, internal_channel=128, output_channel=512, rate=1,
							is_shortcut				= True, 
							is_bottleneck			= True, 
							is_residual				= True,
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
			net = utils.conv2D(net, kernel_size=3, stride=1, internal_channel=128, output_channel=512, rate=1,
							is_shortcut				= True, 
							is_bottleneck			= True, 
							is_residual				= True,
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")
							
			net = utils.conv2D(net, kernel_size=3, stride=1, internal_channel=128, output_channel=512, rate=1,
							is_shortcut				= True, 
							is_bottleneck			= True, 
							is_residual				= True,
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv3")
							
		with tf.variable_scope("conv4_x"), tf.device("/gpu:2"):
			net = utils.conv2D(net, kernel_size=3, stride=2, internal_channel=256, output_channel=1024, rate=1,
							is_shortcut				= True, 
							is_bottleneck			= True, 
							is_residual				= True,
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
			net = utils.conv2D(net, kernel_size=3, stride=1, internal_channel=256, output_channel=1024, rate=1,
							is_shortcut				= True, 
							is_bottleneck			= True, 
							is_residual				= True,
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")
							
			net = utils.conv2D(net, kernel_size=3, stride=1, internal_channel=256, output_channel=1024, rate=1,
							is_shortcut				= True, 
							is_bottleneck			= True, 
							is_residual				= True,
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv3")
														
		with tf.variable_scope("conv5_x"), tf.device("/gpu:3"):
			net = utils.conv2D(net, kernel_size=3, stride=2, internal_channel=512, output_channel=2048, rate=1,
							is_shortcut				= True, 
							is_bottleneck			= True, 
							is_residual				= True,
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
			net = utils.conv2D(net, kernel_size=3, stride=1, internal_channel=512, output_channel=2048, rate=1,
							is_shortcut				= True, 
							is_bottleneck			= True, 
							is_residual				= True,
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")
							
			net = utils.conv2D(net, kernel_size=3, stride=1, internal_channel=512, output_channel=2048, rate=1,
							is_shortcut				= True, 
							is_bottleneck			= True, 
							is_residual				= True,
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv3")
							
		with tf.variable_scope("Average_Pooling"):
			net = tf.reduce_mean(net, [1, 2], keep_dims=True)
			
		with tf.variable_scope("fc1"):
			net = utils.conv2D(net, kernel_size=1, stride=1, output_channel=class_num, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_residual				= True,
							is_batch_norm			= False, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
		utils.Save_Analyzsis_as_csv(Analysis, FILE)
	return net
	
def PSPNet(net, class_num, is_training, is_testing, is_ternary, is_quantized_activation, IS_TERNARY, IS_QUANTIZED_ACTIVATION, FILE, reuse=None, scope="PSPNet"):
	input_shape = net.get_shape().as_list()
	with tf.variable_scope(scope, reuse=reuse):
		Analysis = utils.Analyzer({}, net, type='DATA', name='Input')
		with tf.variable_scope("ResNet50"):
			with tf.variable_scope("root_block"):
				net = utils.conv2D(net, kernel_size=3, stride=2, output_channel=64, rate=1,
								is_shortcut				= False, 
								is_bottleneck			= False, 
								is_batch_norm			= True, 
								is_training				= is_training, 
								is_testing				= is_testing, 
								is_dilated				= False, 
								is_ternary      		= is_ternary,
								is_quantized_activation = is_quantized_activation,
								IS_TERNARY				= IS_TERNARY,
								IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
								Analysis				= Analysis,
								scope					= "conv1_1")

				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
								is_shortcut				= False, 
								is_bottleneck			= False, 
								is_batch_norm			= True, 
								is_training				= is_training, 
								is_testing				= is_testing, 
								is_dilated				= False, 
								is_ternary      		= is_ternary,
								is_quantized_activation = is_quantized_activation,
								IS_TERNARY				= IS_TERNARY,
								IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
								Analysis				= Analysis,
								scope					= "conv1_2")

				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=128, rate=1,
								is_shortcut				= False, 
								is_bottleneck			= False, 
								is_batch_norm			= True, 
								is_training				= is_training, 
								is_testing				= is_testing, 
								is_dilated				= False, 
								is_ternary      		= is_ternary,
								is_quantized_activation = is_quantized_activation,
								IS_TERNARY				= IS_TERNARY,
								IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
								Analysis				= Analysis,
								scope					= "conv1_3")
				
				net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

			with tf.variable_scope("Block1"), tf.device("/gpu:0"):
				net = utils.conv2D(net, kernel_size=3, stride=2, internal_channel=64, output_channel=256, rate=1,
								is_shortcut				= True, 
								is_bottleneck			= True, 
								is_residual				= True,
								is_batch_norm			= True, 
								is_training				= is_training, 
								is_testing				= is_testing, 
								is_dilated				= False, 
								is_ternary      		= is_ternary,
								is_quantized_activation = is_quantized_activation,
								IS_TERNARY				= IS_TERNARY,
								IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
								Analysis				= Analysis,
								scope					= "conv2_r1")
								
				net = utils.conv2D(net, kernel_size=3, stride=1, internal_channel=64, output_channel=256, rate=1,
								is_shortcut				= True, 
								is_bottleneck			= True, 
								is_residual				= True,
								is_batch_norm			= True, 
								is_training				= is_training, 
								is_testing				= is_testing, 
								is_dilated				= False, 
								is_ternary      		= is_ternary,
								is_quantized_activation = is_quantized_activation,
								IS_TERNARY				= IS_TERNARY,
								IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
								Analysis				= Analysis,
								scope					= "conv2_r2")
								
				net = utils.conv2D(net, kernel_size=3, stride=1, internal_channel=64, output_channel=256, rate=1,
								is_shortcut				= True, 
								is_bottleneck			= True, 
								is_residual				= True,
								is_batch_norm			= True, 
								is_training				= is_training, 
								is_testing				= is_testing, 
								is_dilated				= False, 
								is_ternary      		= is_ternary,
								is_quantized_activation = is_quantized_activation,
								IS_TERNARY				= IS_TERNARY,
								IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
								Analysis				= Analysis,
								scope					= "conv2_r3")
								
			with tf.variable_scope("Block2"), tf.device("/gpu:1"):
				net = utils.conv2D(net, kernel_size=3, stride=2, internal_channel=128, output_channel=512, rate=1,
								is_shortcut				= True, 
								is_bottleneck			= True, 
								is_residual				= True,
								is_batch_norm			= True, 
								is_training				= is_training, 
								is_testing				= is_testing, 
								is_dilated				= False, 
								is_ternary     			= is_ternary,
								is_quantized_activation = is_quantized_activation,
								IS_TERNARY				= IS_TERNARY,
								IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
								Analysis				= Analysis,
								scope					= "conv3_r1")
								
				net = utils.conv2D(net, kernel_size=3, stride=1, internal_channel=128, output_channel=512, rate=1,
								is_shortcut				= True, 
								is_bottleneck			= True, 
								is_residual				= True,
								is_batch_norm			= True, 
								is_training				= is_training, 
								is_testing				= is_testing, 
								is_dilated				= False, 
								is_ternary     			= is_ternary,
								is_quantized_activation = is_quantized_activation,
								IS_TERNARY				= IS_TERNARY,
								IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
								Analysis				= Analysis,
								scope					= "conv3_r2")
								
				net = utils.conv2D(net, kernel_size=3, stride=1, internal_channel=128, output_channel=512, rate=1,
								is_shortcut				= True, 
								is_bottleneck			= True, 
								is_residual				= True,
								is_batch_norm			= True, 
								is_training				= is_training, 
								is_testing				= is_testing, 
								is_dilated				= False, 
								is_ternary      		= is_ternary,
								is_quantized_activation = is_quantized_activation,
								IS_TERNARY				= IS_TERNARY,
								IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
								Analysis				= Analysis,
								scope					= "conv3_r3")
								
				net = utils.conv2D(net, kernel_size=3, stride=1, internal_channel=128, output_channel=512, rate=1,
								is_shortcut				= True, 
								is_bottleneck			= True, 
								is_residual				= True,
								is_batch_norm			= True, 
								is_training				= is_training, 
								is_testing				= is_testing, 
								is_dilated				= False, 
								is_ternary      		= is_ternary,
								is_quantized_activation = is_quantized_activation,
								IS_TERNARY				= IS_TERNARY,
								IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
								Analysis				= Analysis,
								scope					= "conv3_r4")

			with tf.variable_scope("Block3"), tf.device("/gpu:2"):
				net = utils.conv2D(net, kernel_size=3, stride=1, internal_channel=256, output_channel=1024, rate=2,
								is_shortcut				= True, 
								is_bottleneck			= True, 
								is_residual				= True,
								is_batch_norm			= True, 
								is_training				= is_training, 
								is_testing				= is_testing, 
								is_dilated				= True, 
								is_ternary      		= is_ternary,
								is_quantized_activation = is_quantized_activation,
								IS_TERNARY				= IS_TERNARY,
								IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
								Analysis				= Analysis,
								scope					= "conv4_r1")
								
				net = utils.conv2D(net, kernel_size=3, stride=1, internal_channel=256, output_channel=1024, rate=2,
								is_shortcut				= True, 
								is_bottleneck			= True, 
								is_residual				= True,
								is_batch_norm			= True, 
								is_training				= is_training, 
								is_testing				= is_testing, 
								is_dilated				= True, 
								is_ternary      		= is_ternary,
								is_quantized_activation = is_quantized_activation,
								IS_TERNARY				= IS_TERNARY,
								IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
								Analysis				= Analysis,
								scope					= "conv4_r2")
								
				net = utils.conv2D(net, kernel_size=3, stride=1, internal_channel=256, output_channel=1024, rate=2,
								is_shortcut				= True, 
								is_bottleneck			= True, 
								is_residual				= True,
								is_batch_norm			= True, 
								is_training				= is_training, 
								is_testing				= is_testing, 
								is_dilated				= True, 
								is_ternary      		= is_ternary,
								is_quantized_activation = is_quantized_activation,
								IS_TERNARY				= IS_TERNARY,
								IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
								Analysis				= Analysis,
								scope					= "conv4_r3")
															
				net = utils.conv2D(net, kernel_size=3, stride=1, internal_channel=256, output_channel=1024, rate=2,
								is_shortcut				= True, 
								is_bottleneck			= True, 
								is_residual				= True,
								is_batch_norm			= True, 
								is_training				= is_training, 
								is_testing				= is_testing, 
								is_dilated				= True, 
								is_ternary      		= is_ternary,
								is_quantized_activation = is_quantized_activation,
								IS_TERNARY				= IS_TERNARY,
								IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
								Analysis				= Analysis,
								scope					= "conv4_r4")
								
				net = utils.conv2D(net, kernel_size=3, stride=1, internal_channel=256, output_channel=1024, rate=2,
								is_shortcut				= True, 
								is_bottleneck			= True, 
								is_residual				= True,
								is_batch_norm			= True, 
								is_training				= is_training, 
								is_testing				= is_testing, 
								is_dilated				= True, 
								is_ternary     			= is_ternary,
								is_quantized_activation = is_quantized_activation,
								IS_TERNARY				= IS_TERNARY,
								IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
								Analysis				= Analysis,
								scope					= "conv4_r5")
								
				net = utils.conv2D(net, kernel_size=3, stride=1, internal_channel=256, output_channel=1024, rate=2,
								is_shortcut				= True, 
								is_bottleneck			= True, 
								is_residual				= True,
								is_batch_norm			= True, 
								is_training				= is_training, 
								is_testing				= is_testing, 
								is_dilated				= True, 
								is_ternary      		= is_ternary,
								is_quantized_activation = is_quantized_activation,
								IS_TERNARY				= IS_TERNARY,
								IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
								Analysis				= Analysis,
								scope					= "conv4_r6")

			with tf.variable_scope("Block4"), tf.device("/gpu:2"):
				net = utils.conv2D(net, kernel_size=3, stride=1, internal_channel=512, output_channel=2048, rate=4,
								is_shortcut				= True, 
								is_bottleneck			= True, 
								is_residual				= True,
								is_batch_norm			= True, 
								is_training				= is_training, 
								is_testing				= is_testing, 
								is_dilated				= True, 
								is_ternary     			= is_ternary,
								is_quantized_activation = is_quantized_activation,
								IS_TERNARY				= IS_TERNARY,
								IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
								Analysis				= Analysis,
								scope					= "conv5_r1")
								
				net = utils.conv2D(net, kernel_size=3, stride=1, internal_channel=512, output_channel=2048, rate=4,
								is_shortcut				= True, 
								is_bottleneck			= True, 
								is_residual				= True,
								is_batch_norm			= True, 
								is_training				= is_training, 
								is_testing				= is_testing, 
								is_dilated				= True, 
								is_ternary     			= is_ternary,
								is_quantized_activation = is_quantized_activation,
								IS_TERNARY				= IS_TERNARY,
								IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
								Analysis				= Analysis,
								scope					= "conv5_r2")
								
				net = utils.conv2D(net, kernel_size=3, stride=1, internal_channel=512, output_channel=2048, rate=4,
								is_shortcut				= True, 
								is_bottleneck			= True, 
								is_residual				= True,
								is_batch_norm			= True, 
								is_training				= is_training, 
								is_testing				= is_testing, 
								is_dilated				= True, 
								is_ternary      		= is_ternary,
								is_quantized_activation = is_quantized_activation,
								IS_TERNARY				= IS_TERNARY,
								IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
								Analysis				= Analysis,
								scope					= "conv5_r3")
								
		with tf.variable_scope("Pyramid_Average_Pooling"):
			net = utils.Pyramid_Pooling(net, strides=[60, 30, 20 ,10], output_channel=512,
								is_training				= is_training,
								is_testing				= is_testing)

			net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=512, rate=1,
								is_shortcut				= False, 
								is_bottleneck			= False, 
								is_batch_norm			= True, 
								is_training				= is_training, 
								is_testing				= is_testing, 
								is_dilated				= False, 
								is_ternary      		= is_ternary,
								is_quantized_activation = is_quantized_activation,
								IS_TERNARY				= IS_TERNARY,
								IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
								Analysis				= Analysis,
								scope					= "conv6_1")
			#net = tf.cond(is_training, lambda: tf.nn.dropout(net, keep_prob=0.9), lambda: net)
			
			net = utils.conv2D(net, kernel_size=1, stride=1, output_channel=class_num, rate=1,
								is_shortcut		 		= False, 
								is_bottleneck	 		= False, 
								is_batch_norm	 		= False, 
								is_training		 		= is_training, 
								is_testing		 		= is_testing, 
								is_dilated		 		= False, 
								is_ternary       		= is_ternary,
								is_quantized_activation = is_quantized_activation,
								IS_TERNARY				= IS_TERNARY,
								IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
								Analysis				= Analysis,
								scope					= "conv6_2")

			net = tf.image.resize_images(net, [input_shape[1], input_shape[2]])
		utils.Save_Analyzsis_as_csv(Analysis, FILE)
	return net
	
def SegNet_VGG_16(
		net                     ,
		class_num               ,
		# Placerholder          
		is_training             ,
		is_testing              ,
		is_ternary              ,
		# Hyperparameter        
		is_quantized_activation ,
		IS_TERNARY              ,
		IS_QUANTIZED_ACTIVATION ,
		IS_CONV_BIAS            , #(Not Use)
		Activation              , #(Not Use)
		IS_DROPOUT              ,
		DROPOUT_RATE            ,
		IS_BN                   ,
		# Analysis File Path    
		FILE                    ,
		reuse = None            , 
		scope = "SegNet_VGG_16" 
	):
	with tf.variable_scope(scope, reuse=reuse):
		Analysis = utils.Analyzer({}, net, type='DATA', name='Input')
		with tf.variable_scope("encoder"):
			with tf.variable_scope("224X224"): # 1/1
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= IS_BN, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")

				tf.add_to_collection("partial_output", net)
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= IS_BN, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary     			= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")	
							
				tf.add_to_collection("partial_output", net)

				net, indices1, output_shape1, Analysis = utils.indice_pool(net, stride=2, Analysis=Analysis, scope="Pool1")
				
			with tf.variable_scope("112X112"): # 1/2
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=128, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= IS_BN, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				tf.add_to_collection("partial_output", net)

				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=128, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= IS_BN, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")	
							
				tf.add_to_collection("partial_output", net)

				net, indices2, output_shape2, Analysis = utils.indice_pool(net, stride=2, Analysis=Analysis, scope="Pool2")
				
			with tf.variable_scope("56X56"): # 1/4
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=256, rate=1, 
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= IS_BN, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary     			= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				tf.add_to_collection("partial_output", net)

				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=256, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= IS_BN, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")	
							
				tf.add_to_collection("partial_output", net)

				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=256, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= IS_BN, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary     			= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv3")	
							
				tf.add_to_collection("partial_output", net)
				
				net, indices3, output_shape3, Analysis = utils.indice_pool(net, stride=2, Analysis=Analysis, scope="Pool3")
				
			with tf.variable_scope("28X28"): # 1/8
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=512, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= IS_BN, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				tf.add_to_collection("partial_output", net)

				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=512, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= IS_BN, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")	
							
				tf.add_to_collection("partial_output", net)

				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=512, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= IS_BN, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary     			= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv3")		
							
				tf.add_to_collection("partial_output", net)

				net, indices4, output_shape4, Analysis = utils.indice_pool(net, stride=2, Analysis=Analysis, scope="Pool4")
				
			with tf.variable_scope("14X14"): # 1/16
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=512, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= IS_BN, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				tf.add_to_collection("partial_output", net)

				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=512, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= IS_BN, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")	
							
				tf.add_to_collection("partial_output", net)

				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=512, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= IS_BN, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv3")		
							
				tf.add_to_collection("partial_output", net)

				net, indices5, output_shape5, Analysis = utils.indice_pool(net, stride=2, Analysis=Analysis, scope="Pool5")
						
		with tf.variable_scope("decoder"):
			with tf.variable_scope("14X14_D"): # 1/ # conv5_D
				net = utils.indice_unpool(net, stride=2, output_shape=output_shape5, indices=indices5, scope="unPool5")
				
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=512, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= IS_BN, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")

				tf.add_to_collection("partial_output", net)
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=512, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= IS_BN, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")	
							
				tf.add_to_collection("partial_output", net)
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=512, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= IS_BN, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv3")							
				
				tf.add_to_collection("partial_output", net)

			with tf.variable_scope("28X28_D"): # 1/8 # conv4_D
				net = utils.indice_unpool(net, stride=2, output_shape=output_shape4, indices=indices4, scope="unPool4")
				
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=512, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= IS_BN, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				tf.add_to_collection("partial_output", net)

				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=512, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= IS_BN, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")	
							
				tf.add_to_collection("partial_output", net)

				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=256, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= IS_BN, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv3")
				
				tf.add_to_collection("partial_output", net)

			with tf.variable_scope("56X56_D"): # 1/4 # conv3_D
				net = utils.indice_unpool(net, stride=2, output_shape=output_shape3, indices=indices3, scope="unPool3")
				
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=256, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= IS_BN, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary     			= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				tf.add_to_collection("partial_output", net)

				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=256, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= IS_BN, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary     			= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")
							
				tf.add_to_collection("partial_output", net)

				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=128, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= IS_BN, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary     			= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv3")
							
				tf.add_to_collection("partial_output", net)

			with tf.variable_scope("112X112_D"): # 1/2 # conv2_D
				net = utils.indice_unpool(net, stride=2, output_shape=output_shape2, indices=indices2, scope="unPool2")
				
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=128, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= IS_BN, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				tf.add_to_collection("partial_output", net)

				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= IS_BN, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")	
				
				tf.add_to_collection("partial_output", net)

			with tf.variable_scope("224X224_D"): # 1/1 # conv1_D
				net = utils.indice_unpool(net, stride=2, output_shape=output_shape1, indices=indices1, scope="unPool1")
				
				tf.add_to_collection("partial_output", net)

				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= IS_BN, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				tf.add_to_collection("partial_output", net)
				
				if IS_DROPOUT:
					net = tf.cond(is_testing, lambda: net, lambda: tf.layers.dropout(net, DROPOUT_RATE))
					
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=class_num, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= False, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")	

				tf.add_to_collection("partial_output", net)
		utils.Save_Analyzsis_as_csv(Analysis, FILE)
	return net
	
	
	
def SegNet_VGG_10(
		net                     ,
		class_num               ,
		# Placerholder          
		is_training             ,
		is_testing              ,
		is_ternary              ,
		# Hyperparameter        
		is_quantized_activation ,
		IS_TERNARY              ,
		IS_QUANTIZED_ACTIVATION ,
		IS_CONV_BIAS            , #(Not Use)
		Activation              , #(Not Use)
		IS_DROPOUT              ,
		DROPOUT_RATE            ,
		IS_BN                   ,
		# Analysis File Path    
		FILE                    ,
		reuse = None            , 
		scope = "SegNet_VGG_10" ):
		
	with tf.variable_scope(scope, reuse=reuse):
		Analysis = utils.Analyzer({}, net, type='DATA', name='Input')
		with tf.variable_scope("encoder"):
			with tf.variable_scope("224X224"): # 1/1
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= IS_BN, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= IS_BN, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")	
							
				net, indices1, output_shape1, Analysis = utils.indice_pool(net, stride=2, Analysis=Analysis, scope="Pool1")
				
			with tf.variable_scope("112X112"): # 1/2
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=128, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= IS_BN, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=128, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= IS_BN, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")	
							
				net, indices2, output_shape2, Analysis = utils.indice_pool(net, stride=2, Analysis=Analysis, scope="Pool2")
				
			with tf.variable_scope("56X56"): # 1/4
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=256, rate=1, 
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= IS_BN, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=256, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= IS_BN, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")	
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=256, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= IS_BN, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary     			= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv3")	
							
				net, indices3, output_shape3, Analysis = utils.indice_pool(net, stride=2, Analysis=Analysis, scope="Pool3")
						
		with tf.variable_scope("decoder"):
			with tf.variable_scope("56X56_D"): # 1/4 # conv3_D
				net = utils.indice_unpool(net, stride=2, output_shape=output_shape3, indices=indices3, scope="unPool3")
				
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=256, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= IS_BN, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=256, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= IS_BN, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=128, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= IS_BN, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary     			= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv3")
							
			with tf.variable_scope("112X112_D"): # 1/2 # conv2_D
				net = utils.indice_unpool(net, stride=2, output_shape=output_shape2, indices=indices2, scope="unPool2")
				
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=128, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= IS_BN, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= IS_BN, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")	
				
			with tf.variable_scope("224X224_D"): # 1/1 # conv1_D
				net = utils.indice_unpool(net, stride=2, output_shape=output_shape1, indices=indices1, scope="unPool1")
				
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= IS_BN, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
				
				if IS_DROPOUT:
					net = tf.cond(is_testing, lambda: net, lambda: tf.layers.dropout(net, DROPOUT_RATE))
				
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=class_num, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= False, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")	
		utils.Save_Analyzsis_as_csv(Analysis, FILE)
	return net
	
def SegNet_VGG_10_dilated(net, class_num, is_training, is_testing, is_ternary, is_quantized_activation, IS_QUANTIZED_ACTIVATION, IS_TERNARY, FILE, reuse=None, scope="SegNet_VGG_10_dilated"):
	with tf.variable_scope(scope, reuse=reuse):
		Analysis = utils.Analyzer({}, net, type='DATA', name='Input')
		with tf.variable_scope("encoder"):
			with tf.variable_scope("224X224"), tf.device("/gpu:0"): # 1/1
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")	
				
			with tf.variable_scope("112X112"), tf.device("/gpu:1"): # 1/2
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=2,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= True, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=2,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= True, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")	
							
			with tf.variable_scope("56X56"), tf.device("/gpu:2"): # 1/4
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=4, 
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= True, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=4,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= True, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")	
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=4,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= True, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv3")	
							
		with tf.variable_scope("decoder"): # 1/1
			with tf.variable_scope("56X56_D"), tf.device("/gpu:3"): # 1/4 # conv3_D
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=4,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= True, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=4,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= True, 
							is_ternary     		 	= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=4,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= True, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv3")
							
			with tf.variable_scope("112X112_D"), tf.device("/gpu:0"): # 1/2 # conv2_D
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=2,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= True, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=2,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= True, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")	
				
			with tf.variable_scope("224X224_D"), tf.device("/gpu:1"): # 1/1 # conv1_D
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=class_num, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= False, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary     			= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")	
		utils.Save_Analyzsis_as_csv(Analysis, FILE)
	return net
	
	
def SegNet_VGG_10_v2(net, class_num, is_training, is_testing, is_ternary, is_quantized_activation, IS_TERNARY, IS_QUANTIZED_ACTIVATION, FILE, reuse=None, scope="SegNet_VGG_10"):
	with tf.variable_scope(scope, reuse=reuse):
		Analysis = utils.Analyzer({}, net, type='DATA', name='Input')
		with tf.variable_scope("encoder"):
			with tf.variable_scope("224X224"): # 1/1
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")	
							
				net, indices1, output_shape1, Analysis = utils.indice_pool(net, stride=2, Analysis=Analysis, scope="Pool1")
				
			with tf.variable_scope("112X112"): # 1/2
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")	
							
				net, indices2, output_shape2, Analysis = utils.indice_pool(net, stride=2, Analysis=Analysis, scope="Pool2")
				
			with tf.variable_scope("56X56"): # 1/4
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1, 
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")	
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary     			= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv3")	
							
				net, indices3, output_shape3, Analysis = utils.indice_pool(net, stride=2, Analysis=Analysis, scope="Pool3")
						
		with tf.variable_scope("decoder"):
			with tf.variable_scope("56X56_D"): # 1/4 # conv3_D
				net = utils.indice_unpool(net, stride=2, output_shape=output_shape3, indices=indices3, scope="unPool3")
				
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary     			= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv3")
							
			with tf.variable_scope("112X112_D"): # 1/2 # conv2_D
				net = utils.indice_unpool(net, stride=2, output_shape=output_shape2, indices=indices2, scope="unPool2")
				
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")	
				
			with tf.variable_scope("224X224_D"): # 1/1 # conv1_D
				net = utils.indice_unpool(net, stride=2, output_shape=output_shape1, indices=indices1, scope="unPool1")
				
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=class_num, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= False, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")	
		utils.Save_Analyzsis_as_csv(Analysis, FILE)
	return net
	
	

def SegNet_VGG_10_depthwise(net, class_num, is_training, is_testing, is_ternary, is_quantized_activation, IS_TERNARY, IS_QUANTIZED_ACTIVATION, FILE, reuse=None, scope="SegNet_VGG_10"):
	with tf.variable_scope(scope, reuse=reuse):
		Analysis = utils.Analyzer({}, net, type='DATA', name='Input')
		with tf.variable_scope("encoder"):
			with tf.variable_scope("224X224"): # 1/1
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= True,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= True,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")	
							
				net, indices1, output_shape1, Analysis = utils.indice_pool(net, stride=2, Analysis=Analysis, scope="Pool1")
				
			with tf.variable_scope("112X112"): # 1/2
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=128, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= True,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=128, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= True,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")	
							
				net, indices2, output_shape2, Analysis = utils.indice_pool(net, stride=2, Analysis=Analysis, scope="Pool2")
				
			with tf.variable_scope("56X56"): # 1/4
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=256, rate=1, 
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= True,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=256, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= True,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")	
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=256, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= True,
							is_ternary     			= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv3")	
							
				net, indices3, output_shape3, Analysis = utils.indice_pool(net, stride=2, Analysis=Analysis, scope="Pool3")
						
		with tf.variable_scope("decoder"):
			with tf.variable_scope("56X56_D"): # 1/4 # conv3_D
				net = utils.indice_unpool(net, stride=2, output_shape=output_shape3, indices=indices3, scope="unPool3")
				
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=256, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= True,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=256, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= True,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=128, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= True,
							is_ternary     			= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv3")
							
			with tf.variable_scope("112X112_D"): # 1/2 # conv2_D
				net = utils.indice_unpool(net, stride=2, output_shape=output_shape2, indices=indices2, scope="unPool2")
				
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=128, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= True,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= True,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")	
				
			with tf.variable_scope("224X224_D"): # 1/1 # conv1_D
				net = utils.indice_unpool(net, stride=2, output_shape=output_shape1, indices=indices1, scope="unPool1")
				
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= True,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=class_num, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= False, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= True,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")	
		utils.Save_Analyzsis_as_csv(Analysis, FILE)
	return net
	

def SegNet_VGG_10_depthwise_v2(net, class_num, is_training, is_testing, is_ternary, is_quantized_activation, IS_TERNARY, IS_QUANTIZED_ACTIVATION, reuse=None, scope="SegNet_VGG_10"):
	with tf.variable_scope(scope, reuse=reuse):
		Analysis = utils.Analyzer({}, net, type='DATA', name='Input')
		with tf.variable_scope("encoder"):
			with tf.variable_scope("224X224"): # 1/1
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= True,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= True,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")	
							
				net, indices1, output_shape1, Analysis = utils.indice_pool(net, stride=2, Analysis=Analysis, scope="Pool1")
				
			with tf.variable_scope("112X112"): # 1/2
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= True,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= True,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")	
							
				net, indices2, output_shape2, Analysis = utils.indice_pool(net, stride=2, Analysis=Analysis, scope="Pool2")
				
			with tf.variable_scope("56X56"): # 1/4
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1, 
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= True,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= True,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")	
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= True,
							is_ternary     			= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv3")	
							
				net, indices3, output_shape3, Analysis = utils.indice_pool(net, stride=2, Analysis=Analysis, scope="Pool3")
						
		with tf.variable_scope("decoder"):
			with tf.variable_scope("56X56_D"): # 1/4 # conv3_D
				net = utils.indice_unpool(net, stride=2, output_shape=output_shape3, indices=indices3, scope="unPool3")
				
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= True,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= True,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= True,
							is_ternary     			= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv3")
							
			with tf.variable_scope("112X112_D"): # 1/2 # conv2_D
				net = utils.indice_unpool(net, stride=2, output_shape=output_shape2, indices=indices2, scope="unPool2")
				
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= True,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= True,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")	
				
			with tf.variable_scope("224X224_D"): # 1/1 # conv1_D
				net = utils.indice_unpool(net, stride=2, output_shape=output_shape1, indices=indices1, scope="unPool1")
				
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= True,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=class_num, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= False, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= True,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")	
		utils.Save_Analyzsis_as_csv(Analysis, FILE)
	return net
	
	


def SegNet_VGG_10_residual(net, class_num, is_training, is_testing, is_ternary, is_quantized_activation, IS_TERNARY, IS_QUANTIZED_ACTIVATION, FILE, reuse=None, scope="SegNet_VGG_10"):
	with tf.variable_scope(scope, reuse=reuse):
		Analysis = utils.Analyzer({}, net, type='DATA', name='Input')
		with tf.variable_scope("encoder"):
			with tf.variable_scope("224X224"): # 1/1
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= False,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= False,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")	
							
				net, indices1, output_shape1, Analysis = utils.indice_pool(net, stride=2, Analysis=Analysis, scope="Pool1")
				
			with tf.variable_scope("112X112"): # 1/2
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=128, rate=1,
							is_shortcut				= True, 
							is_bottleneck			= True, 
							is_residual				= True,
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= False,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=128, rate=1,
							is_shortcut				= True, 
							is_bottleneck			= True, 
							is_residual				= True,
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= False,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")	
							
				net, indices2, output_shape2, Analysis = utils.indice_pool(net, stride=2, Analysis=Analysis, scope="Pool2")
				
			with tf.variable_scope("56X56"): # 1/4
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=256, rate=1, 
							is_shortcut				= True, 
							is_bottleneck			= True, 
							is_residual				= True,
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= False,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=256, rate=1,
							is_shortcut				= True, 
							is_bottleneck			= True, 
							is_residual				= True,
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= False,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")	
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=256, rate=1,
							is_shortcut				= True, 
							is_bottleneck			= True, 
							is_residual				= True,
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= False,
							is_ternary     			= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv3")	
							
				net, indices3, output_shape3, Analysis = utils.indice_pool(net, stride=2, Analysis=Analysis, scope="Pool3")
						
		with tf.variable_scope("decoder"):
			with tf.variable_scope("56X56_D"): # 1/4 # conv3_D
				net = utils.indice_unpool(net, stride=2, output_shape=output_shape3, indices=indices3, scope="unPool3")
				
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=256, rate=1,
							is_shortcut				= True, 
							is_bottleneck			= True, 
							is_residual				= True,
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= False,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=256, rate=1,
							is_shortcut				= True, 
							is_bottleneck			= True, 
							is_residual				= True,
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= False,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=128, rate=1,
							is_shortcut				= True, 
							is_bottleneck			= True, 
							is_residual				= True,
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= False,
							is_ternary     			= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv3")
							
			with tf.variable_scope("112X112_D"): # 1/2 # conv2_D
				net = utils.indice_unpool(net, stride=2, output_shape=output_shape2, indices=indices2, scope="unPool2")
				
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=128, rate=1,
							is_shortcut				= True, 
							is_bottleneck			= True, 
							is_residual				= True,
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= False,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut				= True, 
							is_bottleneck			= True, 
							is_residual				= True,
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= False,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")	
				
			with tf.variable_scope("224X224_D"): # 1/1 # conv1_D
				net = utils.indice_unpool(net, stride=2, output_shape=output_shape1, indices=indices1, scope="unPool1")
				
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= False,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=class_num, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= False, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= False,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")	
		utils.Save_Analyzsis_as_csv(Analysis, FILE)
	return net
	



def SegNet_VGG_10_residual_v2(net, class_num, is_training, is_testing, is_ternary, is_quantized_activation, IS_TERNARY, IS_QUANTIZED_ACTIVATION, FILE, reuse=None, scope="SegNet_VGG_10"):
	with tf.variable_scope(scope, reuse=reuse):
		Analysis = utils.Analyzer({}, net, type='DATA', name='Input')
		with tf.variable_scope("encoder"):
			with tf.variable_scope("224X224"): # 1/1
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= False,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= False,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")	
							
				net, indices1, output_shape1, Analysis = utils.indice_pool(net, stride=2, Analysis=Analysis, scope="Pool1")
				
			with tf.variable_scope("112X112"): # 1/2
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut				= True, 
							is_bottleneck			= True, 
							is_residual				= True,
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= False,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut				= True, 
							is_bottleneck			= True, 
							is_residual				= True,
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= False,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")	
							
				net, indices2, output_shape2, Analysis = utils.indice_pool(net, stride=2, Analysis=Analysis, scope="Pool2")
				
			with tf.variable_scope("56X56"): # 1/4
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1, 
							is_shortcut				= True, 
							is_bottleneck			= True, 
							is_residual				= True,
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= False,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut				= True, 
							is_bottleneck			= True, 
							is_residual				= True,
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= False,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")	
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut				= True, 
							is_bottleneck			= True, 
							is_residual				= True,
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= False,
							is_ternary     			= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv3")	
							
				net, indices3, output_shape3, Analysis = utils.indice_pool(net, stride=2, Analysis=Analysis, scope="Pool3")
						
		with tf.variable_scope("decoder"):
			with tf.variable_scope("56X56_D"): # 1/4 # conv3_D
				net = utils.indice_unpool(net, stride=2, output_shape=output_shape3, indices=indices3, scope="unPool3")
				
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut				= True, 
							is_bottleneck			= True, 
							is_residual				= True,
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= False,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut				= True, 
							is_bottleneck			= True, 
							is_residual				= True,
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= False,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut				= True, 
							is_bottleneck			= True, 
							is_residual				= True,
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= False,
							is_ternary     			= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv3")
							
			with tf.variable_scope("112X112_D"): # 1/2 # conv2_D
				net = utils.indice_unpool(net, stride=2, output_shape=output_shape2, indices=indices2, scope="unPool2")
				
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut				= True, 
							is_bottleneck			= True, 
							is_residual				= True,
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= False,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut				= True, 
							is_bottleneck			= True, 
							is_residual				= True,
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= False,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")	
				
			with tf.variable_scope("224X224_D"): # 1/1 # conv1_D
				net = utils.indice_unpool(net, stride=2, output_shape=output_shape1, indices=indices1, scope="unPool1")
				
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= False,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=class_num, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= False, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= False,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")	
		utils.Save_Analyzsis_as_csv(Analysis, FILE)
	return net
	



def SegNet_VGG_10_SEP(net, class_num, is_training, is_testing, is_ternary, is_quantized_activation, IS_TERNARY, IS_QUANTIZED_ACTIVATION, FILE, reuse=None, scope="SegNet_VGG_10"):
	with tf.variable_scope(scope, reuse=reuse):
		Analysis = utils.Analyzer({}, net, type='DATA', name='Input')
		with tf.variable_scope("encoder"):
			with tf.variable_scope("224X224"): # 1/1
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")	
							
				net, indices1, output_shape1, Analysis = utils.indice_pool(net, stride=2, Analysis=Analysis, scope="Pool1")
				
			with tf.variable_scope("112X112"): # 1/2
				net = utils.conv2D(net, kernel_size=3, stride=1, internal_channel=64, output_channel=128, rate=1,
							is_shortcut				= True, 
							is_bottleneck			= False, 
							is_SEP					= True,
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				net, indices2, output_shape2, Analysis = utils.indice_pool(net, stride=2, Analysis=Analysis, scope="Pool2")
				
			with tf.variable_scope("56X56"): # 1/4
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=256, rate=1, 
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				net = utils.conv2D(net, kernel_size=3, stride=1, internal_channel=128, output_channel=256, rate=1,
							is_shortcut				= True, 
							is_bottleneck			= False, 
							is_SEP					= True,
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")	
							
				net, indices3, output_shape3, Analysis = utils.indice_pool(net, stride=2, Analysis=Analysis, scope="Pool3")
						
		with tf.variable_scope("decoder"):
			with tf.variable_scope("56X56_D"): # 1/4 # conv3_D
				net = utils.indice_unpool(net, stride=2, output_shape=output_shape3, indices=indices3, scope="unPool3")
				
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=256, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				net = utils.conv2D(net, kernel_size=3, stride=1, internal_channel=64, output_channel=128, rate=1,
							is_shortcut				= True, 
							is_bottleneck			= False, 
							is_SEP					= True,
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary     			= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")
							
			with tf.variable_scope("112X112_D"): # 1/2 # conv2_D
				net = utils.indice_unpool(net, stride=2, output_shape=output_shape2, indices=indices2, scope="unPool2")
				
				net = utils.conv2D(net, kernel_size=3, stride=1, internal_channel=32, output_channel=64, rate=1,
							is_shortcut				= True, 
							is_bottleneck			= False, 
							is_SEP					= True,
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")	
				
			with tf.variable_scope("224X224_D"): # 1/1 # conv1_D
				net = utils.indice_unpool(net, stride=2, output_shape=output_shape1, indices=indices1, scope="unPool1")
				
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=class_num, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= False, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")	
		utils.Save_Analyzsis_as_csv(Analysis, FILE)
	return net
	

def SegNet_VGG_16_5x5(net, class_num, is_training, is_testing, is_ternary, is_quantized_activation, IS_TERNARY, IS_QUANTIZED_ACTIVATION, FILE, reuse=None, scope="SegNet_VGG_16"):
	with tf.variable_scope(scope, reuse=reuse):
		Analysis = utils.Analyzer({}, net, type='DATA', name='Input')
		with tf.variable_scope("encoder"):
			with tf.variable_scope("224X224"): # 1/1
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")

				tf.add_to_collection("partial_output", net)
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary     			= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")	
							
				tf.add_to_collection("partial_output", net)

				net, indices1, output_shape1, Analysis = utils.indice_pool(net, stride=2, Analysis=Analysis, scope="Pool1")
				
			with tf.variable_scope("112X112"): # 1/2
				tf.add_to_collection("partial_output", net)

				net = utils.conv2D(net, kernel_size=5, stride=1, output_channel=128, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				tf.add_to_collection("partial_output", net)
							
				net, indices2, output_shape2, Analysis = utils.indice_pool(net, stride=2, Analysis=Analysis, scope="Pool2")
				
			with tf.variable_scope("56X56"): # 1/4
				tf.add_to_collection("partial_output", net)
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=256, rate=1, 
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary     			= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				tf.add_to_collection("partial_output", net)

				net = utils.conv2D(net, kernel_size=5, stride=1, output_channel=256, rate=1, 
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary     			= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")
							
				tf.add_to_collection("partial_output", net)

				net, indices3, output_shape3, Analysis = utils.indice_pool(net, stride=2, Analysis=Analysis, scope="Pool3")
				
			with tf.variable_scope("28X28"): # 1/8
				tf.add_to_collection("partial_output", net)

				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=512, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				tf.add_to_collection("partial_output", net)
							
				net = utils.conv2D(net, kernel_size=5, stride=1, output_channel=512, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")
							
				tf.add_to_collection("partial_output", net)

				net, indices4, output_shape4, Analysis = utils.indice_pool(net, stride=2, Analysis=Analysis, scope="Pool4")
				
			with tf.variable_scope("14X14"): # 1/16
				tf.add_to_collection("partial_output", net)
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=512, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				tf.add_to_collection("partial_output", net)

				net = utils.conv2D(net, kernel_size=5, stride=1, output_channel=512, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")
							
				tf.add_to_collection("partial_output", net)

				net, indices5, output_shape5, Analysis = utils.indice_pool(net, stride=2, Analysis=Analysis, scope="Pool5")
						
		with tf.variable_scope("decoder"):
			with tf.variable_scope("14X14_D"): # 1/ # conv5_D
				net = utils.indice_unpool(net, stride=2, output_shape=output_shape5, indices=indices5, scope="unPool5")
				
				tf.add_to_collection("partial_output", net)

							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=512, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")

				tf.add_to_collection("partial_output", net)
							
				net = utils.conv2D(net, kernel_size=5, stride=1, output_channel=512, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")

				tf.add_to_collection("partial_output", net)

			with tf.variable_scope("28X28_D"): # 1/8 # conv4_D
				net = utils.indice_unpool(net, stride=2, output_shape=output_shape4, indices=indices4, scope="unPool4")
				

				tf.add_to_collection("partial_output", net)

				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=512, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				tf.add_to_collection("partial_output", net)

				net = utils.conv2D(net, kernel_size=5, stride=1, output_channel=256, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")
							
				tf.add_to_collection("partial_output", net)

			with tf.variable_scope("56X56_D"): # 1/4 # conv3_D
				net = utils.indice_unpool(net, stride=2, output_shape=output_shape3, indices=indices3, scope="unPool3")
				
				tf.add_to_collection("partial_output", net)


				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=256, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary     			= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				tf.add_to_collection("partial_output", net)

				net = utils.conv2D(net, kernel_size=5, stride=1, output_channel=128, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary     			= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")
							
				tf.add_to_collection("partial_output", net)

			with tf.variable_scope("112X112_D"): # 1/2 # conv2_D
				net = utils.indice_unpool(net, stride=2, output_shape=output_shape2, indices=indices2, scope="unPool2")
				
				tf.add_to_collection("partial_output", net)

				net = utils.conv2D(net, kernel_size=5, stride=1, output_channel=64, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")

				tf.add_to_collection("partial_output", net)

			with tf.variable_scope("224X224_D"): # 1/1 # conv1_D
				net = utils.indice_unpool(net, stride=2, output_shape=output_shape1, indices=indices1, scope="unPool1")
				
				tf.add_to_collection("partial_output", net)

				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				tf.add_to_collection("partial_output", net)

				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=class_num, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= False, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")	

				tf.add_to_collection("partial_output", net)
		utils.Save_Analyzsis_as_csv(Analysis, FILE)
	return net
	

def SegNet_VGG_10_depthwise_residual(net, class_num, is_training, is_testing, is_ternary, is_quantized_activation, IS_TERNARY, IS_QUANTIZED_ACTIVATION, FILE, reuse=None, scope="SegNet_VGG_10"):
	with tf.variable_scope(scope, reuse=reuse):
		Analysis = utils.Analyzer({}, net, type='DATA', name='Input')
		with tf.variable_scope("encoder"):
			with tf.variable_scope("224X224"): # 1/1
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= True,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= True,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")	
							
				net, indices1, output_shape1, Analysis = utils.indice_pool(net, stride=2, Analysis=Analysis, scope="Pool1")
				
			with tf.variable_scope("112X112"): # 1/2
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=128, rate=1,
							is_shortcut				= True, 
							is_bottleneck			= True, 
							is_residual				= True,
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= True,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=128, rate=1,
							is_shortcut				= True, 
							is_bottleneck			= True, 
							is_residual				= True,
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= True,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")	
							
				net, indices2, output_shape2, Analysis = utils.indice_pool(net, stride=2, Analysis=Analysis, scope="Pool2")
				
			with tf.variable_scope("56X56"): # 1/4
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=256, rate=1, 
							is_shortcut				= True, 
							is_bottleneck			= True, 
							is_residual				= True,
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= True,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=256, rate=1,
							is_shortcut				= True, 
							is_bottleneck			= True, 
							is_residual				= True,
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= True,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")	
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=256, rate=1,
							is_shortcut				= True, 
							is_bottleneck			= True, 
							is_residual				= True,
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= True,
							is_ternary     			= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv3")	
							
				net, indices3, output_shape3, Analysis = utils.indice_pool(net, stride=2, Analysis=Analysis, scope="Pool3")
						
		with tf.variable_scope("decoder"):
			with tf.variable_scope("56X56_D"): # 1/4 # conv3_D
				net = utils.indice_unpool(net, stride=2, output_shape=output_shape3, indices=indices3, scope="unPool3")
				
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=256, rate=1,
							is_shortcut				= True, 
							is_bottleneck			= True, 
							is_residual				= True,
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= True,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=256, rate=1,
							is_shortcut				= True, 
							is_bottleneck			= True, 
							is_residual				= True,
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= True,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=128, rate=1,
							is_shortcut				= True, 
							is_bottleneck			= True, 
							is_residual				= True,
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= True,
							is_ternary     			= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv3")
							
			with tf.variable_scope("112X112_D"): # 1/2 # conv2_D
				net = utils.indice_unpool(net, stride=2, output_shape=output_shape2, indices=indices2, scope="unPool2")
				
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=128, rate=1,
							is_shortcut				= True, 
							is_bottleneck			= True, 
							is_residual				= True,
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= True,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut				= True, 
							is_bottleneck			= True, 
							is_residual				= True,
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= True,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")	
				
			with tf.variable_scope("224X224_D"): # 1/1 # conv1_D
				net = utils.indice_unpool(net, stride=2, output_shape=output_shape1, indices=indices1, scope="unPool1")
				
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= True,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=class_num, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= False, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= True,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")	
		utils.Save_Analyzsis_as_csv(Analysis, FILE)
	return net
	


def SegNet_VGG_10_depthwise_5x5(net, class_num, is_training, is_testing, is_ternary, is_quantized_activation, IS_TERNARY, IS_QUANTIZED_ACTIVATION, FILE, reuse=None, scope="SegNet_VGG_10"):
	with tf.variable_scope(scope, reuse=reuse):
		Analysis = utils.Analyzer({}, net, type='DATA', name='Input')
		with tf.variable_scope("encoder"):
			with tf.variable_scope("224X224"): # 1/1
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= True,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= True,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")	
							
				net, indices1, output_shape1, Analysis = utils.indice_pool(net, stride=2, Analysis=Analysis, scope="Pool1")
				
			with tf.variable_scope("112X112"): # 1/2
				net = utils.conv2D(net, kernel_size=5, stride=1, output_channel=128, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= True,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				net, indices2, output_shape2, Analysis = utils.indice_pool(net, stride=2, Analysis=Analysis, scope="Pool2")
				
			with tf.variable_scope("56X56"): # 1/4
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=256, rate=1, 
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= True,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				net = utils.conv2D(net, kernel_size=5, stride=1, output_channel=256, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= True,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")	
							
				net, indices3, output_shape3, Analysis = utils.indice_pool(net, stride=2, Analysis=Analysis, scope="Pool3")
						
		with tf.variable_scope("decoder"):
			with tf.variable_scope("56X56_D"): # 1/4 # conv3_D
				net = utils.indice_unpool(net, stride=2, output_shape=output_shape3, indices=indices3, scope="unPool3")
				
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=256, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= True,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				net = utils.conv2D(net, kernel_size=5, stride=1, output_channel=128, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= True,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")
							
			with tf.variable_scope("112X112_D"): # 1/2 # conv2_D
				net = utils.indice_unpool(net, stride=2, output_shape=output_shape2, indices=indices2, scope="unPool2")
				
				net = utils.conv2D(net, kernel_size=5, stride=1, output_channel=64, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= True,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
			with tf.variable_scope("224X224_D"): # 1/1 # conv1_D
				net = utils.indice_unpool(net, stride=2, output_shape=output_shape1, indices=indices1, scope="unPool1")
				
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= True,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
				
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=class_num, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= False, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= True,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")	
		utils.Save_Analyzsis_as_csv(Analysis, FILE)
	return net
	

def SegNet_VGG_10_depthwise_group10(net, class_num, is_training, is_testing, is_ternary, is_quantized_activation, IS_TERNARY, IS_QUANTIZED_ACTIVATION, FILE, reuse=None, scope="SegNet_VGG_10"):
	with tf.variable_scope(scope, reuse=reuse):
		g = 10
		Analysis = utils.Analyzer({}, net, type='DATA', name='Input')
		with tf.variable_scope("encoder"):
			with tf.variable_scope("224X224"): # 1/1
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= True,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= True,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")	
							
				net, indices1, output_shape1, Analysis = utils.indice_pool(net, stride=2, Analysis=Analysis, scope="Pool1")
				
			with tf.variable_scope("112X112"): # 1/2
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=128, rate=1, group=g,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= True,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=128, rate=1, group=g,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= True,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")	
							
				net, indices2, output_shape2, Analysis = utils.indice_pool(net, stride=2, Analysis=Analysis, scope="Pool2")
				
			with tf.variable_scope("56X56"): # 1/4
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=256, rate=1, group=g, 
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= True,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=256, rate=1, group=g,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= True,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")	
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=256, rate=1, group=g,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= True,
							is_ternary     			= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv3")	
							
				net, indices3, output_shape3, Analysis = utils.indice_pool(net, stride=2, Analysis=Analysis, scope="Pool3")
						
		with tf.variable_scope("decoder"):
			with tf.variable_scope("56X56_D"): # 1/4 # conv3_D
				net = utils.indice_unpool(net, stride=2, output_shape=output_shape3, indices=indices3, scope="unPool3")
				
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=256, rate=1, group=g,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= True,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=256, rate=1, group=g,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= True,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=128, rate=1, group=g,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= True,
							is_ternary     			= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv3")
							
			with tf.variable_scope("112X112_D"): # 1/2 # conv2_D
				net = utils.indice_unpool(net, stride=2, output_shape=output_shape2, indices=indices2, scope="unPool2")
				
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=128, rate=1, group=g,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= True,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1, group=g,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= True,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")	
				
			with tf.variable_scope("224X224_D"): # 1/1 # conv1_D
				net = utils.indice_unpool(net, stride=2, output_shape=output_shape1, indices=indices1, scope="unPool1")
				
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= True,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=class_num, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= False, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= True,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")	
		utils.Save_Analyzsis_as_csv(Analysis, FILE)
	return net
	
def SegNet_VGG_10_5x5(net, class_num, is_training, is_testing, is_ternary, is_quantized_activation, IS_TERNARY, IS_QUANTIZED_ACTIVATION, FILE, reuse=None, scope="SegNet_VGG_16"):
	with tf.variable_scope(scope, reuse=reuse):
		Analysis = utils.Analyzer({}, net, type='DATA', name='Input')
		with tf.variable_scope("encoder"):
			with tf.variable_scope("224X224"): # 1/1
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")

				tf.add_to_collection("partial_output", net)
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary     			= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")	
							
				tf.add_to_collection("partial_output", net)

				net, indices1, output_shape1, Analysis = utils.indice_pool(net, stride=2, Analysis=Analysis, scope="Pool1")
				
			with tf.variable_scope("112X112"): # 1/2
				tf.add_to_collection("partial_output", net)

				net = utils.conv2D(net, kernel_size=5, stride=1, output_channel=128, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				tf.add_to_collection("partial_output", net)
							
				net, indices2, output_shape2, Analysis = utils.indice_pool(net, stride=2, Analysis=Analysis, scope="Pool2")
				
			with tf.variable_scope("56X56"): # 1/4
				tf.add_to_collection("partial_output", net)
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=256, rate=1, 
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary     			= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				tf.add_to_collection("partial_output", net)

				net = utils.conv2D(net, kernel_size=5, stride=1, output_channel=256, rate=1, 
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary     			= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")
							
				tf.add_to_collection("partial_output", net)

				net, indices3, output_shape3, Analysis = utils.indice_pool(net, stride=2, Analysis=Analysis, scope="Pool3")

		with tf.variable_scope("decoder"):
			with tf.variable_scope("56X56_D"): # 1/4 # conv3_D
				net = utils.indice_unpool(net, stride=2, output_shape=output_shape3, indices=indices3, scope="unPool3")
				
				tf.add_to_collection("partial_output", net)


				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=256, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary     			= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				tf.add_to_collection("partial_output", net)

				net = utils.conv2D(net, kernel_size=5, stride=1, output_channel=128, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary     			= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")
							
				tf.add_to_collection("partial_output", net)

			with tf.variable_scope("112X112_D"): # 1/2 # conv2_D
				net = utils.indice_unpool(net, stride=2, output_shape=output_shape2, indices=indices2, scope="unPool2")
				
				tf.add_to_collection("partial_output", net)

				net = utils.conv2D(net, kernel_size=5, stride=1, output_channel=64, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")

				tf.add_to_collection("partial_output", net)

			with tf.variable_scope("224X224_D"): # 1/1 # conv1_D
				net = utils.indice_unpool(net, stride=2, output_shape=output_shape1, indices=indices1, scope="unPool1")
				
				tf.add_to_collection("partial_output", net)

				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				tf.add_to_collection("partial_output", net)

				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=class_num, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= False, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")	

				tf.add_to_collection("partial_output", net)
		utils.Save_Analyzsis_as_csv(Analysis, FILE)
	return net
	

def SegNet_VGG_10_depthwise_5x5_group10(net, class_num, is_training, is_testing, is_ternary, is_quantized_activation, IS_TERNARY, IS_QUANTIZED_ACTIVATION, FILE, reuse=None, scope="SegNet_VGG_10"):
	with tf.variable_scope(scope, reuse=reuse):
		g = 10
		Analysis = utils.Analyzer({}, net, type='DATA', name='Input')
		with tf.variable_scope("encoder"):
			with tf.variable_scope("224X224"): # 1/1
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= True,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= True,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")	
							
				net, indices1, output_shape1, Analysis = utils.indice_pool(net, stride=2, Analysis=Analysis, scope="Pool1")
				
			with tf.variable_scope("112X112"): # 1/2
				net = utils.conv2D(net, kernel_size=5, stride=1, output_channel=128, rate=1, group=g,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= True,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				net, indices2, output_shape2, Analysis = utils.indice_pool(net, stride=2, Analysis=Analysis, scope="Pool2")
				
			with tf.variable_scope("56X56"): # 1/4
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=256, rate=1, group=g, 
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= True,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				net = utils.conv2D(net, kernel_size=5, stride=1, output_channel=256, rate=1, group=g,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= True,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")	
							
				net, indices3, output_shape3, Analysis = utils.indice_pool(net, stride=2, Analysis=Analysis, scope="Pool3")
						
		with tf.variable_scope("decoder"):
			with tf.variable_scope("56X56_D"): # 1/4 # conv3_D
				net = utils.indice_unpool(net, stride=2, output_shape=output_shape3, indices=indices3, scope="unPool3")
				
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=256, rate=1, group=g,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= True,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				net = utils.conv2D(net, kernel_size=5, stride=1, output_channel=128, rate=1, group=g,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= True,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")
							
			with tf.variable_scope("112X112_D"): # 1/2 # conv2_D
				net = utils.indice_unpool(net, stride=2, output_shape=output_shape2, indices=indices2, scope="unPool2")
				
				net = utils.conv2D(net, kernel_size=5, stride=1, output_channel=64, rate=1, group=g,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= True,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")	
				
			with tf.variable_scope("224X224_D"): # 1/1 # conv1_D
				net = utils.indice_unpool(net, stride=2, output_shape=output_shape1, indices=indices1, scope="unPool1")
				
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= True, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= True,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv1")
							
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=class_num, rate=1,
							is_shortcut				= False, 
							is_bottleneck			= False, 
							is_batch_norm			= False, 
							is_training				= is_training, 
							is_testing				= is_testing, 
							is_dilated				= False, 
							is_depthwise			= True,
							is_ternary      		= is_ternary,
							is_quantized_activation = is_quantized_activation,
							IS_TERNARY				= IS_TERNARY,
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							Analysis				= Analysis,
							scope					= "conv2")	
		utils.Save_Analyzsis_as_csv(Analysis, FILE)
	return net
	
