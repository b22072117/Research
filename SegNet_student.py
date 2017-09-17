import tensorflow as tf
import numpy as np
import scipy.ndimage
import pdb
import math
from PIL import Image
from scipy import misc
import utils

#=======================#
#	Global Parameter	#
#=======================#
BATCH_SIZE = 3
EPOCH_TIME = 20
IS_TRAINING = False
IS_TESTING = not IS_TRAINING
if IS_TESTING:	
	BATCH_SIZE = 1

LEARNING_RATE = 1e-3
EPOCH_DECADE = 200
LR_DECADE = 10
LAMBDA = 0.

#=======================#
#	Training File Name	#
#=======================#
if LEARNING_RATE <= 1e-5:
	TRAINING_WEIGHT_FILE = 'SegNet_Model/SegNet_student_0' + str(LEARNING_RATE).split('.')[0]
else:
	TRAINING_WEIGHT_FILE = 'SegNet_Model/SegNet_student_0' + str(LEARNING_RATE).split('.')[1]
TRAINING_WEIGHT_FILE = TRAINING_WEIGHT_FILE + '_epoch' + str(EPOCH_DECADE)
TRAINING_WEIGHT_FILE = TRAINING_WEIGHT_FILE + '_divide' + str(LR_DECADE)
TRAINING_WEIGHT_FILE = TRAINING_WEIGHT_FILE + '_L20' + str(LAMBDA).split('.')[1]	

#=======================#
#	Testing File Name	#
#=======================#
if LEARNING_RATE <= 1e-5:
	TESTING_WEIGHT_FILE = 'SegNet_Model/SegNet_student_0' + str(LEARNING_RATE).split('.')[0]
else:
	TESTING_WEIGHT_FILE = 'SegNet_Model/SegNet_student_0' + str(LEARNING_RATE).split('.')[1]
TESTING_WEIGHT_FILE = TESTING_WEIGHT_FILE + '_epoch' + str(EPOCH_DECADE)
TESTING_WEIGHT_FILE = TESTING_WEIGHT_FILE + '_divide' + str(LR_DECADE)
TESTING_WEIGHT_FILE = TESTING_WEIGHT_FILE + '_L20' + str(LAMBDA).split('.')[1]
TESTING_WEIGHT_FILE = TESTING_WEIGHT_FILE + '_' + str(EPOCH_TIME)
TESTING_WEIGHT_FILE = TESTING_WEIGHT_FILE + '.npz'


#===========# 
#	Define	#
#===========#
def SegNet_VGG_16_student(net, class_num, is_training, is_testing, reuse=None, scope="SegNet_VGG_16_student"):
	with tf.variable_scope(scope, reuse=reuse):
		with tf.variable_scope("encoder"):
			with tf.variable_scope("224X224"): # 1/1
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut=False, 
							is_bottleneck=False, 
							is_batch_norm=True, 
							is_training=is_training, 
							is_testing=is_testing, 
							is_dilated=False, 
							scope="conv1")
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut=False, 
							is_bottleneck=False, 
							is_batch_norm=True, 
							is_training=is_training, 
							is_testing=is_testing, 
							is_dilated=True, 
							scope="conv2")		
				#net, indices1, output_shape1 = indice_pool(net, stride=2, scope="Pool1")
				
			with tf.variable_scope("112X112"): # 1/2
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=2,
							is_shortcut=False, 
							is_bottleneck=False, 
							is_batch_norm=True, 
							is_training=is_training, 
							is_testing=is_testing, 
							is_dilated=True, 
							scope="conv1")
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=2,
							is_shortcut=False, 
							is_bottleneck=False, 
							is_batch_norm=True, 
							is_training=is_training, 
							is_testing=is_testing, 
							is_dilated=True, 
							scope="conv2")	
				#net, indices2, output_shape2 = indice_pool(net, stride=2, scope="Pool2")
				
			with tf.variable_scope("56X56"): # 1/4
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=4, 
							is_shortcut=False, 
							is_bottleneck=False, 
							is_batch_norm=True, 
							is_training=is_training, 
							is_testing=is_testing, 
							is_dilated=True, 
							scope="conv1")
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=4,
							is_shortcut=False, 
							is_bottleneck=False, 
							is_batch_norm=True, 
							is_training=is_training, 
							is_testing=is_testing, 
							is_dilated=True, 
							scope="conv2")	
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=4,
							is_shortcut=False, 
							is_bottleneck=False, 
							is_batch_norm=True, 
							is_training=is_training, 
							is_testing=is_testing, 
							is_dilated=True, 
							scope="conv3")			
				#net, indices3, output_shape3 = indice_pool(net, stride=2, scope="Pool3")
				
#			with tf.variable_scope("28X28"): # 1/8
#				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=8,
#							is_shortcut=False, 
#							is_bottleneck=False, 
#							is_batch_norm=True, 
#							is_training=is_training, 
#							is_testing=is_testing, 
#							is_dilated=False, 
#							scope="conv1")
#				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=8,
#							is_shortcut=False, 
#							is_bottleneck=False, 
#							is_batch_norm=True, 
#							is_training=is_training, 
#							is_testing=is_testing, 
#							is_dilated=False, 
#							scope="conv2")	
#				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=8,
#							is_shortcut=False, 
#							is_bottleneck=False, 
#							is_batch_norm=True, 
#							is_training=is_training, 
#							is_testing=is_testing, 
#							is_dilated=True, 
#							scope="conv3")			
#				#net, indices4, output_shape4 = indice_pool(net, stride=2, scope="Pool4")
#				
#			with tf.variable_scope("14X14"): # 1/16
#				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=16,
#							is_shortcut=False, 
#							is_bottleneck=False, 
#							is_batch_norm=True, 
#							is_training=is_training, 
#							is_testing=is_testing, 
#							is_dilated=False, 
#							scope="conv1")
#				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=16,
#							is_shortcut=False, 
#							is_bottleneck=False, 
#							is_batch_norm=True, 
#							is_training=is_training, 
#							is_testing=is_testing, 
#							is_dilated=False, 
#							scope="conv2")	
#				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=16,
#							is_shortcut=False, 
#							is_bottleneck=False, 
#							is_batch_norm=True, 
#							is_training=is_training, 
#							is_testing=is_testing, 
#							is_dilated=True, 
#							scope="conv3")			
#				#net, indices5, output_shape5 = indice_pool(net, stride=2, scope="Pool5")
#						
		with tf.variable_scope("decoder"):
#			with tf.variable_scope("14X14_D"): # 1/ # conv5_D
#				#net = indice_unpool(net, stride=2, output_shape=output_shape5, indices=indices5, scope="unPool5")
#				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=16,
#							is_shortcut=False, 
#							is_bottleneck=False, 
#							is_batch_norm=True, 
#							is_training=is_training, 
#							is_testing=is_testing, 
#							is_dilated=False, 
#							scope="conv1")
#				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=16,
#							is_shortcut=False, 
#							is_bottleneck=False, 
#							is_batch_norm=True, 
#							is_training=is_training, 
#							is_testing=is_testing, 
#							is_dilated=False, 
#							scope="conv2")	
#				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=16,
#							is_shortcut=False, 
#							is_bottleneck=False, 
#							is_batch_norm=True, 
#							is_training=is_training, 
#							is_testing=is_testing, 
#							is_dilated=False, 
#							scope="conv3")							
#				
#			with tf.variable_scope("28X28_D"): # 1/8 # conv4_D
#				#net = indice_unpool(net, stride=2, output_shape=output_shape4, indices=indices4, scope="unPool4")
#				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=8,
#							is_shortcut=False, 
#							is_bottleneck=False, 
#							is_batch_norm=True, 
#							is_training=is_training, 
#							is_testing=is_testing, 
#							is_dilated=False, 
#							scope="conv1")
#				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=8,
#							is_shortcut=False, 
#							is_bottleneck=False, 
#							is_batch_norm=True, 
#							is_training=is_training, 
#							is_testing=is_testing, 
#							is_dilated=False, 
#							scope="conv2")	
#				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=8,
#							is_shortcut=False, 
#							is_bottleneck=False, 
#							is_batch_norm=True, 
#							is_training=is_training, 
#							is_testing=is_testing, 
#							is_dilated=False, 
#							scope="conv3")
				
				
			with tf.variable_scope("56X56_D"): # 1/4 # conv3_D
#				#net = indice_unpool(net, stride=2, output_shape=output_shape3, indices=indices3, scope="unPool3")
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=4,
							is_shortcut=False, 
							is_bottleneck=False, 
							is_batch_norm=True, 
							is_training=is_training, 
							is_testing=is_testing, 
							is_dilated=True, 
							scope="conv1")
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=4,
							is_shortcut=False, 
							is_bottleneck=False, 
							is_batch_norm=True, 
							is_training=is_training, 
							is_testing=is_testing, 
							is_dilated=True, 
							scope="conv2")
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=4,
							is_shortcut=False, 
							is_bottleneck=False, 
							is_batch_norm=True, 
							is_training=is_training, 
							is_testing=is_testing, 
							is_dilated=True, 
							scope="conv3")
							
			with tf.variable_scope("112X112_D"): # 1/2 # conv2_D
				#net = indice_unpool(net, stride=2, output_shape=output_shape2, indices=indices2, scope="unPool2")
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=2,
							is_shortcut=False, 
							is_bottleneck=False, 
							is_batch_norm=True, 
							is_training=is_training, 
							is_testing=is_testing, 
							is_dilated=True, 
							scope="conv1")
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=2,
							is_shortcut=False, 
							is_bottleneck=False, 
							is_batch_norm=True, 
							is_training=is_training, 
							is_testing=is_testing, 
							is_dilated=True, 
							scope="conv2")	
				
			with tf.variable_scope("224X224_D"): # 1/1 # conv1_D
				#net = indice_unpool(net, stride=2, output_shape=output_shape1, indices=indices1, scope="unPool1")
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut=False, 
							is_bottleneck=False, 
							is_batch_norm=True, 
							is_training=is_training, 
							is_testing=is_testing, 
							is_dilated=False, 
							scope="conv1")
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=class_num, rate=1,
							is_shortcut=False, 
							is_bottleneck=False, 
							is_batch_norm=False, 
							is_training=is_training, 
							is_testing=is_testing, 
							is_dilated=False, 
							scope="conv2")	
	return net
	
def main(argv=None):
#===============#
#	File Read	#
#===============#
	CamVid_Path = '/home/2016/b22072117/ObjectSegmentation/codes/dataset/CamVid'
	SegNet_Y_pre_Path = '/home/2016/b22072117/ObjectSegmentation/codes/nets/SegNet_Y_pre' 

# Training file
	print("")
	print("Loading Training Data ...")
	# CamVid
	CamVid_train_data, CamVid_train_data_index = utils.read_dataset_file(CamVid_Path, '/train.txt')
	CamVid_train_target, _ = utils.read_dataset_file(CamVid_Path, '/trainannot.txt')
	
	if IS_TRAINING:	
		# SegNet Prediction Output
		SegNet_train_data = CamVid_train_data
		SegNet_train_target, SegNet_train_data_index = utils.read_Y_pre_file(SegNet_Y_pre_Path, '/train.txt')

	class_num = np.max(CamVid_train_target)+1
	CamVid_train_target = utils.one_of_k(CamVid_train_target, class_num)
	print("Shape of train data	: {Shape}" .format(Shape = np.shape(CamVid_train_data)))
	print("Shape of train target	: {Shape}" .format(Shape = np.shape(CamVid_train_target)))
	
# Validation file
	print("")
	print("Loading Validation Data ...")
	# CamVid
	CamVid_valid_data, CamVid_valid_data_index = utils.read_dataset_file(CamVid_Path, '/val.txt')
	CamVid_valid_target, _ = utils.read_dataset_file(CamVid_Path, '/valannot.txt')
	
	if IS_TRAINING:	
		# SegNet Prediction output
		SegNet_valid_data = CamVid_valid_data
		SegNet_valid_target, SegNet_valid_data_index = utils.read_Y_pre_file(SegNet_Y_pre_Path, '/val.txt')

	CamVid_valid_target = utils.one_of_k(CamVid_valid_target, class_num)
	print("Shape of valid data	: {Shape}" .format(Shape = np.shape(CamVid_valid_data)))
	print("Shape of valid target	: {Shape}" .format(Shape = np.shape(CamVid_valid_target)))

# Testing file
	print("")
	print("Loading Testing Data ...")
	# CamVid 
	CamVid_test_data, CamVid_test_data_index = utils.read_dataset_file(CamVid_Path, '/test.txt')
	CamVid_test_target, _ = utils.read_dataset_file(CamVid_Path, '/testannot.txt')
	
	if IS_TRAINING:	
		# SegNet Preidict output
		SegNet_test_data = CamVid_test_data
		SegNet_test_target, SegNet_test_data_index = utils.read_Y_pre_file(SegNet_Y_pre_Path, '/test.txt')
	
	CamVid_test_target = utils.one_of_k(CamVid_test_target, class_num)
	print("Shape of test data	: {Shape}" .format(Shape = np.shape(CamVid_test_data)))
	print("Shape of test target	: {Shape}" .format(Shape = np.shape(CamVid_test_target)))
	print("")



#===========#
#	Data	#
#===========#
	data_shape = np.shape(CamVid_train_data)
# Placeholder
	xs 			  = tf.placeholder(tf.float32, [BATCH_SIZE, data_shape[1], data_shape[2], data_shape[3]]) 
	ys 			  = tf.placeholder(tf.float32, [BATCH_SIZE, data_shape[1], data_shape[2], class_num])
	learning_rate = tf.placeholder(tf.float32)
	is_training   = tf.placeholder(tf.bool)
	is_testing 	  = tf.placeholder(tf.bool)
	
# Data Preprocessing
	xImage = xs
	
#===========#
#	Graph	#
#===========#
	net = xImage
	prediction = SegNet_VGG_16_student(net, class_num, is_training=is_training, is_testing=is_testing, reuse=None, scope="SegNet_VGG_16_student")
	#prediction = tf.nn.softmax(net)
	

#===============#
#	Collection	#
#===============#	
	weights_collection = tf.get_collection("weights", scope=None)
	bias_collection = tf.get_collection("bais", scope=None)
	mean_collection = tf.get_collection("batch_mean", scope=None)
	var_collection = tf.get_collection("batch_var", scope=None)
	scale_collection = tf.get_collection("batch_scale", scope=None)
	shift_collection = tf.get_collection("batch_shift", scope=None)
	trained_mean_collection = tf.get_collection("trained_mean", scope=None)
	trained_var_collection = tf.get_collection("trained_var", scope=None)
	params = tf.get_collection("params", scope=None) 
	
#=======================#
#	Training Strategy	#
#=======================#	
	#regression_lambda     = tf.constant(2*BATCH_SIZE, dtype=tf.float32)
	#regression_square_sum = tf.reduce_sum(tf.square(tf.subtract(prediction, ys)), 0)
	#regression_loss       = tf.multiply(regression_lambda, regression_square_sum)
	KL = tf.nn.softmax_cross_entropy_with_logits(labels = ys, logits = prediction) 
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(ys, -1), logits=prediction)

	l2_norm   = tf.reduce_mean(tf.stack([tf.nn.l2_loss(x) for x in weights_collection]))
	l2_lambda = tf.constant(LAMBDA)
	
	loss = cross_entropy
	#loss = tf.add(loss, tf.multiply(l2_lambda, l2_norm))
	train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
	
	init = tf.global_variables_initializer()
	
#=======================#
#	Weight Parameters	#
#=======================#	
	keys = ['conv1_1', 'conv1_1_b', 'conv1_1_mean', 'conv1_1_var', 'conv1_1_scale', 'conv1_1_shift',
			'conv1_2', 'conv1_2_b', 'conv1_2_mean', 'conv1_2_var', 'conv1_2_scale', 'conv1_2_shift', 
			'conv2_1', 'conv2_1_b', 'conv2_1_mean', 'conv2_1_var', 'conv2_1_scale', 'conv2_1_shift',
			'conv2_2', 'conv2_2_b', 'conv2_2_mean', 'conv2_2_var', 'conv2_2_scale', 'conv2_2_shift',
			'conv3_1', 'conv3_1_b', 'conv3_1_mean', 'conv3_1_var', 'conv3_1_scale', 'conv3_1_shift',
			'conv3_2', 'conv3_2_b', 'conv3_2_mean', 'conv3_2_var', 'conv3_2_scale', 'conv3_2_shift',
			'conv3_3', 'conv3_3_b', 'conv3_3_mean', 'conv3_3_var', 'conv3_3_scale', 'conv3_3_shift',
			#'conv4_1', 'conv4_1_b', 'conv4_1_mean', 'conv4_1_var', 'conv4_1_scale', 'conv4_1_shift',
			#'conv4_2', 'conv4_2_b', 'conv4_2_mean', 'conv4_2_var', 'conv4_2_scale', 'conv4_2_shift',
			#'conv4_3', 'conv4_3_b', 'conv4_3_mean', 'conv4_3_var', 'conv4_3_scale', 'conv4_3_shift',
			#'conv5_1', 'conv5_1_b', 'conv5_1_mean', 'conv5_1_var', 'conv5_1_scale', 'conv5_1_shift',
			#'conv5_2', 'conv5_2_b', 'conv5_2_mean', 'conv5_2_var', 'conv5_2_scale', 'conv5_2_shift',
			#'conv5_3', 'conv5_3_b', 'conv5_3_mean', 'conv5_3_var', 'conv5_3_scale', 'conv5_3_shift',
			#'conv5_3_D', 'conv5_3_D_b', 'conv5_3_D_mean', 'conv5_3_D_var', 'conv5_3_D_scale', 'conv5_3_D_shift',
			#'conv5_2_D', 'conv5_2_D_b', 'conv5_2_D_mean', 'conv5_2_D_var', 'conv5_2_D_scale', 'conv5_2_D_shift',
			#'conv5_1_D', 'conv5_1_D_b', 'conv5_1_D_mean', 'conv5_1_D_var', 'conv5_1_D_scale', 'conv5_1_D_shift',
			#'conv4_3_D', 'conv4_3_D_b', 'conv4_3_D_mean', 'conv4_3_D_var', 'conv4_3_D_scale', 'conv4_3_D_shift',
			#'conv4_2_D', 'conv4_2_D_b', 'conv4_2_D_mean', 'conv4_2_D_var', 'conv4_2_D_scale', 'conv4_2_D_shift',
			#'conv4_1_D', 'conv4_1_D_b', 'conv4_1_D_mean', 'conv4_1_D_var', 'conv4_1_D_scale', 'conv4_1_D_shift',
			'conv3_3_D', 'conv3_3_D_b', 'conv3_3_D_mean', 'conv3_3_D_var', 'conv3_3_D_scale', 'conv3_3_D_shift',
			'conv3_2_D', 'conv3_2_D_b', 'conv3_2_D_mean', 'conv3_2_D_var', 'conv3_2_D_scale', 'conv3_2_D_shift',
			'conv3_1_D', 'conv3_1_D_b', 'conv3_1_D_mean', 'conv3_1_D_var', 'conv3_1_D_scale', 'conv3_1_D_shift',
			'conv2_2_D', 'conv2_2_D_b', 'conv2_2_D_mean', 'conv2_2_D_var', 'conv2_2_D_scale', 'conv2_2_D_shift',
			'conv2_1_D', 'conv2_1_D_b', 'conv2_1_D_mean', 'conv2_1_D_var', 'conv2_1_D_scale', 'conv2_1_D_shift',
			'conv1_2_D', 'conv1_2_D_b', 'conv1_2_D_mean', 'conv1_2_D_var', 'conv1_2_D_scale', 'conv1_2_D_shift',
			'conv1_1_D', 'conv1_1_D_b']

	if np.shape(params)[0] != np.shape(keys)[0]:
		print("Number of Parameters is not equal to the number of Keys!")
		exit()

	parameters = {keys[x]: params[x] for x in range(len(params))}


#===============#
#	Session Run	#
#===============#
	with tf.Session() as sess:
	# Initialize
		sess.run(init)
		
	# Learning Rate	
		lr = LEARNING_RATE
	
		if IS_TRAINING == True:

			utils.Training_and_Validation( 
				# Training & Validation Data
				 train_data					= CamVid_train_data			, 
				 train_target				= CamVid_train_target		,
				 valid_data					= CamVid_valid_data			,
				 valid_target				= CamVid_valid_target		,
				# Parameter					
				 EPOCH_TIME					= EPOCH_TIME				,
				 BATCH_SIZE					= BATCH_SIZE				,
				 LR_DECADE					= LR_DECADE					,
				 EPOCH_DECADE				= EPOCH_DECADE				,
				 lr							= lr						,
				# Tensor					
				 train_step					= train_step				,
				 loss						= loss						,
				 prediction					= prediction				,
				# Placeholder					
				 xs							= xs						, 
				 ys							= ys						,
				 learning_rate				= learning_rate				,
				 is_training				= is_training				,
				 is_testing					= is_testing				,
				# Collection (For Saving Trained Weight)
				 mean_collection			= mean_collection			,
				 var_collection				= var_collection			,
				 trained_mean_collection	= trained_mean_collection	,
				 trained_var_collection		= trained_var_collection	,
				 params						= params					,
				# File Path (For Saving Trained Weight)
				 TRAINING_WEIGHT_FILE		= TRAINING_WEIGHT_FILE		,
				# Trained Weight Parameters (For Saving Trained Weight)
				 parameters					= parameters				,
				# Session
				 sess						= sess						)
#		# Loading Pre-trained Model
#			#print ""
#			#print("Loading Pre-trained weights ...")
#			#utils.load_pre_trained_weights(parameters, pre_trained_weight_file=TESTING_WEIGHT_FILE, sess=sess) 
#			
#		# Training & Validation
#			for epoch in range(EPOCH_TIME):
#				SegNet_train_data, SegNet_train_target = utils.shuffle_image(SegNet_train_data, SegNet_train_target)
#				CamVid_train_data, CamVid_train_target = utils.shuffle_image(CamVid_train_data, CamVid_train_target)
#
#				Train_acc = 0
#				Train_loss = 0
#				
#				print("")
#				print("Training ... ")
#				for i in range(int(data_shape[0]/BATCH_SIZE)):
#					# Train data in BATCH SIZR
#					#batch_xs    = SegNet_train_data   [i*BATCH_SIZE : (i+1)*BATCH_SIZE]
#					#batch_ys    = SegNet_train_target [i*BATCH_SIZE : (i+1)*BATCH_SIZE]
#					batch_xs    = CamVid_train_data   [i*BATCH_SIZE : (i+1)*BATCH_SIZE]
#					batch_ys    = CamVid_train_target [i*BATCH_SIZE : (i+1)*BATCH_SIZE]
#					
#					# Run Training Step
#					_, Loss, Prediction = sess.run([train_step, loss, prediction], 
#															feed_dict={xs: batch_xs, ys: batch_ys, learning_rate: lr, is_training: True, is_testing: False})
#					
#					# Result
#					y_pre = np.argmax(Prediction, -1)
#					batch_accuracy = np.mean(np.equal(np.argmax(batch_ys, -1), y_pre))
#					
#					Train_acc  = Train_acc  + batch_accuracy
#					Train_loss = Train_loss + np.mean(Loss)
#
#					# Print Training Process
#					print("Epoch : {ep}" .format(ep = epoch))
#					print("Batch Iteration : {Iter}" .format(Iter = i))
#					print("  Batch Accuracy : {acc}".format(acc = batch_accuracy))
#					print("  Loss : {Loss}".format(Loss = np.mean(Loss)))
#					print("  Learning Rate : {LearningRate}".format(LearningRate = lr))
#					
#					# Per Class Accuracy
#					utils.per_class_accuracy(Prediction, batch_ys)
#
#				# Per Epoch Training Result
#				if epoch==0:
#					Train_acc_per_epoch  = np.array([Train_acc ])
#					Train_loss_per_epoch = np.array([Train_loss])
#				else:
#					Train_acc_per_epoch  = np.concatenate([Train_acc_per_epoch , np.array([Train_acc ])], axis=0)
#					Train_loss_per_epoch = np.concatenate([Train_loss_per_epoch, np.array([Train_loss])], axis=0)
#
#				
#
#			# Validation
#				print("")
#				print("Epoch : {ep}".format(ep = epoch))
#				print("Validation ... ")
#				is_validation = True
#				#_, valid_accuracy, _, _, _ = utils.compute_accuracy(xs, ys, is_training, is_testing, is_validation, prediction, SegNet_valid_data, SegNet_valid_target, BATCH_SIZE, sess)
#				_, valid_accuracy, _, _, _ = utils.compute_accuracy(xs, ys, is_training, is_testing, is_validation, prediction, CamVid_valid_data, CamVid_valid_target, BATCH_SIZE, sess)
#				is_validation = False
#				print("")
#				print("Validation Accuracy = {Valid_Accuracy}".format(Valid_Accuracy=valid_accuracy))
#				
#				if ((epoch%EPOCH_DECADE==0) and (epoch!=0)):
#					lr = lr / LR_DECADE
#					
#			# Save trained weights
#				if (((epoch+1)%10==0) and ((epoch+1)>=10)):
#					print("Saving Trained Weights ... ")
#					#batch_xs = SegNet_train_data[0 : BATCH_SIZE]
#					batch_xs = CamVid_train_data[0 : BATCH_SIZE]
#					utils.assign_trained_mean_and_var(mean_collection, var_collection, trained_mean_collection, trained_var_collection, params, 
#												xs, ys, is_training, is_testing, batch_xs, sess)
#					utils.save_pre_trained_weights( (TRAINING_WEIGHT_FILE+'_'+str(epoch+1)), parameters, xs, batch_xs, sess)
#			
#			# Save Train info as csv file
#			Train_acc_per_epoch  = np.expand_dims(Train_acc_per_epoch , axis=1)
#			Train_loss_per_epoch = np.expand_dims(Train_loss_per_epoch, axis=1)
#			pdb.set_trace()
#			Train_info = np.concatenate([Train_acc_per_epoch, Train_loss_per_epoch], axis=1)
#			utils.Save_file_as_csv(TRAINING_WEIGHT_FILE+'_train_info' , Train_info)
	
		if IS_TESTING == True:
			#is_validation = False

			train_target_path = '/home/2016/b22072117/ObjectSegmentation/codes/result/CamVid/SegNet/'
			valid_target_path = '/home/2016/b22072117/ObjectSegmentation/codes/result/CamVid/SegNet/'
			test_target_path  = '/home/2016/b22072117/ObjectSegmentation/codes/result/CamVid/SegNet/'
			
			train_Y_pre_path  = '/home/2016/b22072117/ObjectSegmentation/codes/nets/SegNet_Y_pre/student'
			valid_Y_pre_path  = '/home/2016/b22072117/ObjectSegmentation/codes/nets/SegNet_Y_pre/student'
			test_Y_pre_path   = '/home/2016/b22072117/ObjectSegmentation/codes/nets/SegNet_Y_pre/student'
			
			
			utils.Testing(
				# Training & Validation & Testing Data
					train_data					= CamVid_train_data			, 
					train_target				= CamVid_train_target		,
					train_data_index			= CamVid_train_data_index	,
					valid_data					= CamVid_valid_data			,
					valid_target				= CamVid_valid_target		,
					valid_data_index			= CamVid_valid_data_index	,
					test_data					= CamVid_test_data			, 
					test_target					= CamVid_test_target		,
					test_data_index				= CamVid_test_data_index	,
				# Parameter	
					BATCH_SIZE					= BATCH_SIZE				,
					IS_SAVING_RESULT_AS_IMAGE	= False						,		
					IS_SAVING_RESULT_AS_NPZ		= False						,
				# Tensor	
					prediction					= prediction				,
				# Placeholder
					xs							= xs						, 
					ys							= ys						,
					is_training					= is_training				,
					is_testing					= is_testing				,
				# File Path (For Loading Trained Weight)
					TESTING_WEIGHT_FILE			= TESTING_WEIGHT_FILE		,
				# Trained Weight Parameters (For Loading Trained Weight)
					parameters					= parameters				,
				# File Path (For Saving Result)
					train_target_path 			= train_target_path 		,
					train_Y_pre_path  			= train_Y_pre_path  		,
					valid_target_path 			= valid_target_path 		,
					valid_Y_pre_path  			= valid_Y_pre_path  		,
					test_target_path 			= test_target_path 			,
					test_Y_pre_path  			= test_Y_pre_path  			,
				# Session        
					sess						= sess						)

		print("")
		print("Works are All Done !")
if __name__ == "__main__":
	tf.app.run()

