import tensorflow as tf
import numpy as np
import scipy.ndimage
import pdb
import math
from PIL import Image
from scipy import misc
import utils
import Model

#=======================#
#	Global Parameter	#
#=======================#
BATCH_SIZE = 4
EPOCH_TIME = 100
IS_TRAINING = True
IS_TESTING  = True 
IS_STUDENT  = False
#if IS_TESTING:	
#	BATCH_SIZE = 1

LEARNING_RATE = 1e-3
EPOCH_DECADE = 100
LR_DECADE = 10
LAMBDA = 0.

#=======================#
#	Training File Name	#
#=======================#
if LEARNING_RATE <= 1e-5:
	TRAINING_WEIGHT_FILE = 'PSPNet_Model/PSPNet_0' + str(LEARNING_RATE).split('.')[0]
else:
	TRAINING_WEIGHT_FILE = 'PSPNet_Model/PSPNet_0' + str(LEARNING_RATE).split('.')[1]
TRAINING_WEIGHT_FILE = TRAINING_WEIGHT_FILE + '_epoch' + str(EPOCH_DECADE)
TRAINING_WEIGHT_FILE = TRAINING_WEIGHT_FILE + '_divide' + str(LR_DECADE)
TRAINING_WEIGHT_FILE = TRAINING_WEIGHT_FILE + '_L20' + str(LAMBDA).split('.')[1]	

#=======================#
#	Testing File Name	#
#=======================#
if LEARNING_RATE <= 1e-5:
	TESTING_WEIGHT_FILE = 'PSPNet_Model/PSPNet_0' + str(LEARNING_RATE).split('.')[0]
else:
	TESTING_WEIGHT_FILE = 'PSPNet_Model/PSPNet_0' + str(LEARNING_RATE).split('.')[1]
TESTING_WEIGHT_FILE = TESTING_WEIGHT_FILE + '_epoch' + str(EPOCH_DECADE)
TESTING_WEIGHT_FILE = TESTING_WEIGHT_FILE + '_divide' + str(LR_DECADE)
TESTING_WEIGHT_FILE = TESTING_WEIGHT_FILE + '_L20' + str(LAMBDA).split('.')[1]
TESTING_WEIGHT_FILE = TESTING_WEIGHT_FILE + '_' + str(EPOCH_TIME)
TESTING_WEIGHT_FILE = TESTING_WEIGHT_FILE + '.npz'


#===========# 
#	Define	#
#===========#

def main(argv=None):
#===============#
#	File Read	#
#===============#
	[CamVid_train_data, CamVid_train_data_index, CamVid_train_target, 
	 CamVid_valid_data, CamVid_valid_data_index, CamVid_valid_target, 
	 CamVid_test_data , CamVid_test_data_index , CamVid_test_target , 
	 Y_pre_train_data , Y_pre_train_data_index , Y_pre_train_target , 
	 Y_pre_valid_data , Y_pre_valid_data_index , Y_pre_valid_target , 
	 Y_pre_test_data  , Y_pre_test_data_index  , Y_pre_test_target  , 
	 class_num] = utils.CamVid_data_parser(
				  # Path
				  CamVid_Path = '/home/2016/b22072117/ObjectSegmentation/codes/dataset/CamVid',
				  Y_pre_Path  = '/home/2016/b22072117/ObjectSegmentation/codes/nets/PSPNet_Y_pre',
				  # Parameter
				  IS_STUDENT  = True	   ,
				  IS_TRAINING = IS_TRAINING)

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
	prediction = Model.PSPNet(net, class_num, is_training=is_training, is_testing=is_testing, reuse=None, scope="PSPNet")
	#prediction = tf.nn.softmax(net)
	

#===============#
#	Collection	#
#===============#	
	weights_collection 		= tf.get_collection("weights"		, scope=None)
	bias_collection 		= tf.get_collection("bais"			, scope=None)
	mean_collection 		= tf.get_collection("batch_mean"	, scope=None)
	var_collection 			= tf.get_collection("batch_var"		, scope=None)
	scale_collection 		= tf.get_collection("batch_scale"	, scope=None)
	shift_collection 		= tf.get_collection("batch_shift"	, scope=None)
	trained_mean_collection = tf.get_collection("trained_mean"	, scope=None)
	trained_var_collection 	= tf.get_collection("trained_var"	, scope=None)
	params 					= tf.get_collection("params"		, scope=None) 
	
#=======================#
#	Training Strategy	#
#=======================#	
	regression_loss = tf.nn.softmax_cross_entropy_with_logits(labels = ys, logits = prediction) 

	l2_norm   = tf.reduce_mean(tf.stack([tf.nn.l2_loss(x) for x in weights_collection]))
	l2_lambda = tf.constant(LAMBDA)
	
	loss = regression_loss
	#loss = tf.add(loss, tf.multiply(l2_lambda, l2_norm))
	train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
	
	init = tf.global_variables_initializer()
	
#=======================#
#	Weight Parameters	#
#=======================#	
	keys = [
			# Root Block
			'conv1_1', 'conv1_1_b', 'conv1_1_mean', 'conv1_1_var', 'conv1_1_scale', 'conv1_1_shift',
			'conv1_2', 'conv1_2_b', 'conv1_2_mean', 'conv1_2_var', 'conv1_2_scale', 'conv1_2_shift', 
			'conv1_3', 'conv1_3_b', 'conv1_3_mean', 'conv1_3_var', 'conv1_3_scale', 'conv1_3_shift', 

			# Block1
			'conv2_r1_shortcut', 'conv2_r1_shortcut_b', 'conv2_r1_shortcut_mean', 'conv2_r1_shortcut_var', 'conv2_r1_shortcut_scale', 'conv2_r1_shortcut_shift',
			'conv2_r1_1', 'conv2_r1_1_b', 'conv2_r1_1_mean', 'conv2_r1_1_var', 'conv2_r1_1_scale', 'conv2_r1_1_shift',
			'conv2_r1_2', 'conv2_r1_2_b', 'conv2_r1_2_mean', 'conv2_r1_2_var', 'conv2_r1_2_scale', 'conv2_r1_2_shift',
			'conv2_r1_3', 'conv2_r1_3_b', 'conv2_r1_3_mean', 'conv2_r1_3_var', 'conv2_r1_3_scale', 'conv2_r1_3_shift',

			'conv2_r2_1', 'conv2_r2_1_b', 'conv2_r2_1_mean', 'conv2_r2_1_var', 'conv2_r2_1_scale', 'conv2_r2_1_shift',
			'conv2_r2_2', 'conv2_r2_2_b', 'conv2_r2_2_mean', 'conv2_r2_2_var', 'conv2_r2_2_scale', 'conv2_r2_2_shift',
			'conv2_r2_3', 'conv2_r2_3_b', 'conv2_r2_3_mean', 'conv2_r2_3_var', 'conv2_r2_3_scale', 'conv2_r2_3_shift',

			'conv2_r3_1', 'conv2_r3_1_b', 'conv2_r3_1_mean', 'conv2_r3_1_var', 'conv2_r3_1_scale', 'conv2_r3_1_shift',
			'conv2_r3_2', 'conv2_r3_2_b', 'conv2_r3_2_mean', 'conv2_r3_2_var', 'conv2_r3_2_scale', 'conv2_r3_2_shift',
			'conv2_r3_3', 'conv2_r3_3_b', 'conv2_r3_3_mean', 'conv2_r3_3_var', 'conv2_r3_3_scale', 'conv2_r3_3_shift',

			# Block2
			'conv3_r1_shortcut', 'conv2_r1_shortcut_b', 'conv2_r1_shortcut_mean', 'conv2_r1_shortcut_var', 'conv2_r1_shortcut_scale', 'conv2_r1_shortcut_shift',
			'conv3_r1_1', 'conv3_r1_1_b', 'conv3_r1_1_mean', 'conv3_r1_1_var', 'conv3_r1_1_scale', 'conv3_r1_1_shift',
			'conv3_r1_2', 'conv3_r1_2_b', 'conv3_r1_2_mean', 'conv3_r1_2_var', 'conv3_r1_2_scale', 'conv3_r1_2_shift',
			'conv3_r1_3', 'conv3_r1_3_b', 'conv3_r1_3_mean', 'conv3_r1_3_var', 'conv3_r1_3_scale', 'conv3_r1_3_shift',

			'conv3_r2_1', 'conv3_r2_1_b', 'conv3_r2_1_mean', 'conv3_r2_1_var', 'conv3_r2_1_scale', 'conv3_r2_1_shift',
			'conv3_r2_2', 'conv3_r2_2_b', 'conv3_r2_2_mean', 'conv3_r2_2_var', 'conv3_r2_2_scale', 'conv3_r2_2_shift',
			'conv3_r2_3', 'conv3_r2_3_b', 'conv3_r2_3_mean', 'conv3_r2_3_var', 'conv3_r2_3_scale', 'conv3_r2_3_shift',

			'conv3_r3_1', 'conv3_r3_1_b', 'conv3_r3_1_mean', 'conv3_r3_1_var', 'conv3_r3_1_scale', 'conv3_r3_1_shift',
			'conv3_r3_2', 'conv3_r3_2_b', 'conv3_r3_2_mean', 'conv3_r3_2_var', 'conv3_r3_2_scale', 'conv3_r3_2_shift',
			'conv3_r3_3', 'conv3_r3_3_b', 'conv3_r3_3_mean', 'conv3_r3_3_var', 'conv3_r3_3_scale', 'conv3_r3_3_shift',

			'conv3_r4_1', 'conv3_r4_1_b', 'conv3_r4_1_mean', 'conv3_r4_1_var', 'conv3_r4_1_scale', 'conv3_r4_1_shift',
			'conv3_r4_2', 'conv3_r4_2_b', 'conv3_r4_2_mean', 'conv3_r4_2_var', 'conv3_r4_2_scale', 'conv3_r4_2_shift',
			'conv3_r4_3', 'conv3_r4_3_b', 'conv3_r4_3_mean', 'conv3_r4_3_var', 'conv3_r4_3_scale', 'conv3_r4_3_shift',

			# Block3
			'conv4_r1_shortcut', 'conv4_r1_shortcut_b', 'conv4_r1_shortcut_mean', 'conv4_r1_shortcut_var', 'conv4_r1_shortcut_scale', 'conv4_r1_shortcut_shift',
			'conv4_r1_1', 'conv4_r1_1_b', 'conv4_r1_1_mean', 'conv4_r1_1_var', 'conv4_r1_1_scale', 'conv4_r1_1_shift',
			'conv4_r1_2', 'conv4_r1_2_b', 'conv4_r1_2_mean', 'conv4_r1_2_var', 'conv4_r1_2_scale', 'conv4_r1_2_shift',
			'conv4_r1_3', 'conv4_r1_3_b', 'conv4_r1_3_mean', 'conv4_r1_3_var', 'conv4_r1_3_scale', 'conv4_r1_3_shift',

			'conv4_r2_1', 'conv4_r2_1_b', 'conv4_r2_1_mean', 'conv4_r2_1_var', 'conv4_r2_1_scale', 'conv4_r2_1_shift',
			'conv4_r2_2', 'conv4_r2_2_b', 'conv4_r2_2_mean', 'conv4_r2_2_var', 'conv4_r2_2_scale', 'conv4_r2_2_shift',
			'conv4_r2_3', 'conv4_r2_3_b', 'conv4_r2_3_mean', 'conv4_r2_3_var', 'conv4_r2_3_scale', 'conv4_r2_3_shift',

			'conv4_r3_1', 'conv4_r3_1_b', 'conv4_r3_1_mean', 'conv4_r3_1_var', 'conv4_r3_1_scale', 'conv4_r3_1_shift',
			'conv4_r3_2', 'conv4_r3_2_b', 'conv4_r3_2_mean', 'conv4_r3_2_var', 'conv4_r3_2_scale', 'conv4_r3_2_shift',
			'conv4_r3_3', 'conv4_r3_3_b', 'conv4_r3_3_mean', 'conv4_r3_3_var', 'conv4_r3_3_scale', 'conv4_r3_3_shift',

			'conv4_r4_1', 'conv4_r4_1_b', 'conv4_r4_1_mean', 'conv4_r4_1_var', 'conv4_r4_1_scale', 'conv4_r4_1_shift',
			'conv4_r4_2', 'conv4_r4_2_b', 'conv4_r4_2_mean', 'conv4_r4_2_var', 'conv4_r4_2_scale', 'conv4_r4_2_shift',
			'conv4_r4_3', 'conv4_r4_3_b', 'conv4_r4_3_mean', 'conv4_r4_3_var', 'conv4_r4_3_scale', 'conv4_r4_3_shift',

			'conv4_r5_1', 'conv4_r5_1_b', 'conv4_r5_1_mean', 'conv4_r5_1_var', 'conv4_r5_1_scale', 'conv4_r5_1_shift',
			'conv4_r5_2', 'conv4_r5_2_b', 'conv4_r5_2_mean', 'conv4_r5_2_var', 'conv4_r5_2_scale', 'conv4_r5_2_shift',
			'conv4_r5_3', 'conv4_r5_3_b', 'conv4_r5_3_mean', 'conv4_r5_3_var', 'conv4_r5_3_scale', 'conv4_r5_3_shift',

			'conv4_r6_1', 'conv4_r6_1_b', 'conv4_r6_1_mean', 'conv4_r6_1_var', 'conv4_r6_1_scale', 'conv4_r6_1_shift',
			'conv4_r6_2', 'conv4_r6_2_b', 'conv4_r6_2_mean', 'conv4_r6_2_var', 'conv4_r6_2_scale', 'conv4_r6_2_shift',
			'conv4_r6_3', 'conv4_r6_3_b', 'conv4_r6_3_mean', 'conv4_r6_3_var', 'conv4_r6_3_scale', 'conv4_r6_3_shift',

			# Block4
			'conv5_r1_shortcut', 'conv5_r1_shortcut_b', 'conv5_r1_shortcut_mean', 'conv5_r1_shortcut_var', 'conv5_r1_shortcut_scale', 'conv5_r1_shortcut_shift',
			'conv5_r1_1', 'conv5_r1_1_b', 'conv5_r1_1_mean', 'conv5_r1_1_var', 'conv5_r1_1_scale', 'conv5_r1_1_shift',
			'conv5_r1_2', 'conv5_r1_2_b', 'conv5_r1_2_mean', 'conv5_r1_2_var', 'conv5_r1_2_scale', 'conv5_r1_2_shift',
			'conv5_r1_3', 'conv5_r1_3_b', 'conv5_r1_3_mean', 'conv5_r1_3_var', 'conv5_r1_3_scale', 'conv5_r1_3_shift',

			'conv5_r2_1', 'conv5_r2_1_b', 'conv5_r2_1_mean', 'conv5_r2_1_var', 'conv5_r2_1_scale', 'conv5_r2_1_shift',
			'conv5_r2_2', 'conv5_r2_2_b', 'conv5_r2_2_mean', 'conv5_r2_2_var', 'conv5_r2_2_scale', 'conv5_r2_2_shift',
			'conv5_r2_3', 'conv5_r2_3_b', 'conv5_r2_3_mean', 'conv5_r2_3_var', 'conv5_r2_3_scale', 'conv5_r2_3_shift',

			'conv5_r3_1', 'conv5_r3_1_b', 'conv5_r3_1_mean', 'conv5_r3_1_var', 'conv5_r3_1_scale', 'conv5_r3_1_shift',
			'conv5_r3_2', 'conv5_r3_2_b', 'conv5_r3_2_mean', 'conv5_r3_2_var', 'conv5_r3_2_scale', 'conv5_r3_2_shift',
			'conv5_r3_3', 'conv5_r3_3_b', 'conv5_r3_3_mean', 'conv5_r3_3_var', 'conv5_r3_3_scale', 'conv5_r3_3_shift',

			# Pyramid Pooling
			'conv6_1_1', 'conv6_1_1_b', 'conv6_1_1_mean', 'conv6_1_1_var', 'conv6_1_1_scale', 'conv6_1_1_shift',
			'conv6_1_2', 'conv6_1_2_b', 'conv6_1_2_mean', 'conv6_1_2_var', 'conv6_1_2_scale', 'conv6_1_2_shift',
			'conv6_1_3', 'conv6_1_3_b', 'conv6_1_3_mean', 'conv6_1_3_var', 'conv6_1_3_scale', 'conv6_1_3_shift',
			'conv6_1_4', 'conv6_1_4_b', 'conv6_1_4_mean', 'conv6_1_4_var', 'conv6_1_4_scale', 'conv6_1_4_shift',
			
			'conv6_2', 'conv6_2_b', 'conv6_2_mean', 'conv6_2_var', 'conv6_2_scale', 'conv6_2_shift',
			'conv6_3', 'conv6_3_b']

	if np.shape(params)[0] != np.shape(keys)[0]:
		print("Number of Parameters is not equal to the number of Keys!")
		print("Real : {R} V.S. Keys : {K}" .format(R=np.shape(params)[0], K=np.shape(keys)[0]))
		exit()

	parameters = {keys[x]: params[x] for x in range(len(params))}


#=================#
#	Session Run	  #
#=================#
	with tf.Session() as sess:
	# Initialize
		sess.run(init)
		
	# Learning Rate	
		lr = LEARNING_RATE
	
		if IS_TRAINING == True:
		# Loading Pre-trained Model
			#print ""
			#print("Loading Pre-trained weights ...")
			#utils.load_pre_trained_weights(parameters, pre_trained_weight_file=TESTING_WEIGHT_FILE, sess=sess) 
			
		# Training & Validation
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
				 TRAINED_WEIGHT_FILE		= TESTING_WEIGHT_FILE		,
				 TRAINING_WEIGHT_FILE		= TRAINING_WEIGHT_FILE		,
				# Trained Weight Parameters (For Saving Trained Weight)
				 parameters					= parameters				,
				# Session
				 sess						= sess						)
	
		if IS_TESTING == True:
			train_target_path = '/home/2016/b22072117/ObjectSegmentation/codes/result/CamVid/PSPNet/'
			valid_target_path = '/home/2016/b22072117/ObjectSegmentation/codes/result/CamVid/PSPNet/'
			test_target_path  = '/home/2016/b22072117/ObjectSegmentation/codes/result/CamVid/PSPNet/'
			
			train_Y_pre_path  = '/home/2016/b22072117/ObjectSegmentation/codes/nets/PSPNet_Y_pre/student'
			valid_Y_pre_path  = '/home/2016/b22072117/ObjectSegmentation/codes/nets/PSPNet_Y_pre/student'
			test_Y_pre_path   = '/home/2016/b22072117/ObjectSegmentation/codes/nets/PSPNet_Y_pre/student'
			
			
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
					BATCH_SIZE					= BATCH_SIZE							,
					IS_SAVING_RESULT_AS_IMAGE	= False						,		
					IS_SAVING_RESULT_AS_NPZ		= False						,
					IS_TRAINING					= IS_TRAINING				,
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

