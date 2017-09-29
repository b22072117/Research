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
Model_Name = 'SegNet_VGG_10'
BATCH_SIZE = 4
EPOCH_TIME = 200

IS_TRAINING 			= True
IS_TESTING  			= True
IS_STUDENT  			= True
IS_GAN	    			= False 
IS_TERNARY  			= False
IS_QUANTIZED_ACTIVATION = False

LEARNING_RATE = 1e-3
EPOCH_DECADE  = 200
LR_DECADE     = 10
LAMBDA        = 0.

DISCRIMINATOR_STEP 	       = 1
TERNARY_EPOCH              = 50
QUANTIZED_ACTIVATION_EPOCH = 150

if ((not IS_TRAINING) and IS_TESTING):
	BATCH_SIZE=1

print("BATCH_SIZE = {BS}" .format(BS=BATCH_SIZE))

#==========#
#   Path   #
#==========#
# For Saving Result Picture of Testing
train_target_path = '/home/2016/b22072117/ObjectSegmentation/codes/result/CamVid/SegNet/'
valid_target_path = '/home/2016/b22072117/ObjectSegmentation/codes/result/CamVid/SegNet/'
test_target_path  = '/home/2016/b22072117/ObjectSegmentation/codes/result/CamVid/SegNet/'

# For Saving Result in .npz file
train_Y_pre_path  = '/home/2016/b22072117/ObjectSegmentation/codes/nets/SegNet_Y_pre/student'
valid_Y_pre_path  = '/home/2016/b22072117/ObjectSegmentation/codes/nets/SegNet_Y_pre/student'
test_Y_pre_path   = '/home/2016/b22072117/ObjectSegmentation/codes/nets/SegNet_Y_pre/student'

#=======================#
#	Training File Name	#
#=======================#
if LEARNING_RATE <= 1e-5:
	TRAINING_WEIGHT_FILE = 'SegNet_Model/' + Model_Name + str(LEARNING_RATE).split('.')[0]
else:
	TRAINING_WEIGHT_FILE = 'SegNet_Model/' + Model_Name + str(LEARNING_RATE).split('.')[1]
TRAINING_WEIGHT_FILE = TRAINING_WEIGHT_FILE + '_epoch' + str(EPOCH_DECADE)
TRAINING_WEIGHT_FILE = TRAINING_WEIGHT_FILE + '_divide' + str(LR_DECADE)
TRAINING_WEIGHT_FILE = TRAINING_WEIGHT_FILE + '_L20' + str(LAMBDA).split('.')[1]	

#=======================#
#	Testing File Name	#
#=======================#
if LEARNING_RATE <= 1e-5:
	TESTING_WEIGHT_FILE = 'SegNet_Model/SegNet_student_ternary_0' + str(LEARNING_RATE).split('.')[0]
else:
	TESTING_WEIGHT_FILE = 'SegNet_Model/SegNet_student_ternary_0' + str(LEARNING_RATE).split('.')[1]
TESTING_WEIGHT_FILE = TESTING_WEIGHT_FILE + '_epoch' + str(EPOCH_DECADE)
TESTING_WEIGHT_FILE = TESTING_WEIGHT_FILE + '_divide' + str(LR_DECADE)
TESTING_WEIGHT_FILE = TESTING_WEIGHT_FILE + '_L20' + str(LAMBDA).split('.')[1]
TESTING_WEIGHT_FILE = TESTING_WEIGHT_FILE + '_' + str(EPOCH_TIME)


#============# 
#	Define   #
#============#
	
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
				  Y_pre_Path  = '/home/2016/b22072117/ObjectSegmentation/codes/nets/SegNet_Y_pre',
				  # Parameter
				  IS_STUDENT  = IS_STUDENT ,
				  IS_TRAINING = IS_TRAINING)
	
	if IS_STUDENT:
		train_data       = Y_pre_train_data
		train_target     = Y_pre_train_target
		train_data_index = Y_pre_train_data_index
	else:
		train_data       = CamVid_train_data
		train_target     = CamVid_train_target
		train_data_index = CamVid_train_data_index

	#===========#
	#	Data	#
	#===========#
	data_shape = np.shape(CamVid_train_data)
	# Placeholder
	xs 			            = tf.placeholder(tf.float32, [BATCH_SIZE, data_shape[1], data_shape[2], data_shape[3]]) 
	ys 			            = tf.placeholder(tf.float32, [BATCH_SIZE, data_shape[1], data_shape[2], class_num])
	learning_rate           = tf.placeholder(tf.float32)
	is_training             = tf.placeholder(tf.bool)
	is_testing 	            = tf.placeholder(tf.bool)
	is_ternary              = tf.placeholder(tf.bool)
	is_quantized_activation = tf.placeholder(tf.bool)

	# Data Preprocessing
	xImage = xs
	
	#===========#
	#	Graph	#
	#===========#
	net = xImage
	#prediction = SegNet_VGG_10_dilated(net, class_num, is_training, is_testing)
	prediction = Model.SegNet_VGG_10(net, class_num, is_training, is_testing, is_ternary, is_quantized_activation, IS_TERNARY, IS_QUANTIZED_ACTIVATION)
	#prediction = tf.nn.softmax(net)
	
	params = tf.get_collection("params", scope=None) 
	
	if IS_GAN:
		prediction = tf.nn.softmax(prediction) # (if no softmax)
		prediction_Gen	 = prediction
		prediction_Dis_0 = Model.Discriminator(prediction_Gen, is_training, is_testing)
		params_Dis       = tf.get_collection("params", scope=None)[np.shape(params)[0]:] 
		prediction_Dis_1 = Model.Discriminator(ys, is_training, is_testing, reuse=True)

	#===============#
	#	Collection	#
	#===============#	
	weights_collection	 		  = tf.get_collection("weights"           , scope=None)
	biases_collection  	    	  = tf.get_collection("biases"            , scope=None)
	mean_collection  	    	  = tf.get_collection("batch_mean"        , scope=None)
	var_collection 	 	    	  = tf.get_collection("batch_var"         , scope=None)
	scale_collection 			  = tf.get_collection("batch_scale"       , scope=None)
	shift_collection 			  = tf.get_collection("batch_shift"       , scope=None)
	trained_mean_collection 	  = tf.get_collection("trained_mean"      , scope=None)
	trained_var_collection 		  = tf.get_collection("trained_var"       , scope=None)
	ternary_weights_bd_collection = tf.get_collection("ternary_weights_bd", scope=None)
	ternary_biases_bd_collection  = tf.get_collection("ternary_biases_bd" , scope=None)
	final_weights_collection      = tf.get_collection("final_weights"     , scope=None)
	final_biases_collection       = tf.get_collection("final_biases"      , scope=None)
	var_list_collection		      = tf.get_collection("var_list"          , scope=None)
	assign_var_list_collection    = tf.get_collection("assign_var_list"   , scope=None)
	activation_collection	      = tf.get_collection("activation"        , scope=None)
	mantissa_collection		      = tf.get_collection("mantissa"          , scope=None)
	fraction_collection           = tf.get_collection("fraction"          , scope=None)
	final_net_collection	      = tf.get_collection("final_net"	      , scope=None)

	#=======================#
	#	Training Strategy	#
	#=======================#	
	# KL divergence
	KL = tf.nn.softmax_cross_entropy_with_logits(labels = ys, logits = prediction) 
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(ys, -1), logits=prediction)
	
	# L2 Regularization
	l2_norm  	= tf.reduce_mean(tf.stack([tf.nn.l2_loss(x) for x in weights_collection]))
	l2_lambda 	= tf.constant(LAMBDA)
	l2_norm   	= tf.multiply(l2_lambda, l2_norm)

	# Loss
	if IS_GAN:
		loss_Dis_0		= tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = prediction_Dis_0, labels = tf.ones_like(prediciton_Dis_0)))
		loss_Dis_1		= tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = prediction_Dis_1, labels = tf.zeros_like(prediction_Dis_1)))
		
		loss_Gen		= KL
		loss_Dis		= KL

		train_step_Gen	= tf.train.AdamOptimizer(learning_rate).minimize(loss_Gen, var_list=params)
		train_step_Dis	= tf.train.AdamOptimizer(learning_rate).minimize(loss_Dis, var_list=params_Dis)
	else:
		loss = KL
		opt = tf.train.AdamOptimizer(learning_rate)
		train_step_compute_gradients = opt.compute_gradients(loss, var_list=var_list_collection)
		gra_and_var = [(train_step_compute_gradients[i][0], params[i]) for i in range(np.shape(train_step_compute_gradients)[0])]
		train_step = opt.apply_gradients(gra_and_var)

	#===========#
	#   Saver   #
	#===========#	
	saver = tf.train.Saver()

	#=================#
	#	Session Run   #
	#=================#
	with tf.Session() as sess:
	# Initialize
		init = tf.global_variables_initializer()
		sess.run(init)
		
	# Learning Rate	
		lr = LEARNING_RATE
	
		if IS_TRAINING == True:

			utils.Training_and_Validation( 
				# Training & Validation Data
				train_data					  = train_data						, 
				train_target				  = train_target					,
				valid_data					  = CamVid_valid_data				,
				valid_target				  = CamVid_valid_target				,
				# Parameter					  		
				EPOCH_TIME					  = EPOCH_TIME						,
				BATCH_SIZE					  = BATCH_SIZE						,
				LR_DECADE					  = LR_DECADE						,
				EPOCH_DECADE				  = EPOCH_DECADE					,
				lr							  = lr								,
				# Tensor					  		
				train_step					  = train_step						,
				loss						  = loss							,
				prediction					  = prediction						,
				# Placeholder				  			
				xs							  = xs								, 
				ys							  = ys								,
				learning_rate				  = learning_rate					,
				is_training					  = is_training						,
				is_testing					  = is_testing						,
				# Collection (For Saving Trained Weight)		
				mean_collection				  = mean_collection					,
				var_collection				  = var_collection					,
				trained_mean_collection		  = trained_mean_collection			,
				trained_var_collection		  = trained_var_collection			,
				params						  = params							,
				# File Path (For Saving Trained Weight)		
				TRAINED_WEIGHT_FILE			  = None,
				TRAINING_WEIGHT_FILE		  = TRAINING_WEIGHT_FILE			,
				# Trained Weight Parameters (For Saving Trained Weight)		
				parameters					  = None							,
				saver						  = saver							,
				# (GAN) Parameter		
				IS_GAN						  = IS_GAN							,
				DISCRIMINATOR_STEP			  = DISCRIMINATOR_STEP				,
				# (GAN) tensor		
				train_step_Gen				  = None							,			
				train_step_Dis				  = None							,
				loss_Gen					  = None							,
				loss_Dis					  = None							,
				prediction_Gen				  = None							,
				prediction_Dis_0			  = None							,
				prediction_Dis_1			  = None							,
				# (ternary) Parameter			
				IS_TERNARY				      = IS_TERNARY						,
				TERNARY_EPOCH			      = TERNARY_EPOCH					,
				# (ternary) Placeholder		
				is_ternary					  = is_ternary						,
				# (ternary) Collection		
				weights_collection		      = weights_collection				,
				biases_collection			  = biases_collection				,
				ternary_weights_bd_collection = ternary_weights_bd_collection	,
				ternary_biases_bd_collection  = ternary_biases_bd_collection	,
				# (assign final weights)
				assign_var_list_collection	  = assign_var_list_collection		,
				# (quantize actvation) parameter
				IS_QUANTIZED_ACTIVATION		  = IS_QUANTIZED_ACTIVATION			,
				QUANTIZED_ACTIVATION_EPOCH	  = QUANTIZED_ACTIVATION_EPOCH		,
				# (quantize actvation) parameter
				is_quantized_activation		  = is_quantized_activation			,
				# (quantize actvation) collection
				activation_collection		  = activation_collection			, 
				mantissa_collection			  = mantissa_collection 			,
				fraction_collection     	  = fraction_collection				,
				# (debug)
				final_weights_collection	  = final_weights_collection		,
				final_net_collection		  = final_net_collection			,
				# Session
				 sess						  = sess							)
	
		if IS_TESTING == True:
			#is_validation = False

			
			
			utils.Testing(
				# Training & Validation & Testing Data
					train_data					= CamVid_train_data 			  	, 
					train_target				= CamVid_train_target				,
					train_data_index			= CamVid_train_data_index			,
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
					IS_TRAINING					= IS_TRAINING				,
					IS_TERNARY					= IS_TERNARY					,
					IS_QUANTIZED_ACTIVATION		= IS_QUANTIZED_ACTIVATION		,
				# Tensor	
					prediction					= prediction				,
				# Placeholder
					xs							= xs						, 
					ys							= ys						,
					is_training					= is_training				,
					is_testing					= is_testing				,
					is_quantized_activation		= is_quantized_activation	,
				# File Path (For Loading Trained Weight)
					TESTING_WEIGHT_FILE			= TESTING_WEIGHT_FILE		,
				# Trained Weight Parameters (For Loading Trained Weight)
					parameters					= None						,
					saver 						= saver						,
				# File Path (For Saving Result)
					train_target_path 			= train_target_path 		,
					train_Y_pre_path  			= train_Y_pre_path  		,
					valid_target_path 			= valid_target_path 		,
					valid_Y_pre_path  			= valid_Y_pre_path  		,
					test_target_path 			= test_target_path 			,
					test_Y_pre_path  			= test_Y_pre_path  			,
				# Session        
					sess						= sess						)
		pdb.set_trace()
		print("")
		print("Works are All Done !")

if __name__ == "__main__":
	tf.app.run()

