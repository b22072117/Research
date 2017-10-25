import tensorflow as tf
import numpy as np
import scipy.ndimage
import pdb
import math
from PIL import Image
from scipy import misc
try:
	import xml.etree.cElementTree as ET
except ImportError:
	import xml.etree.ElementTree as ET
from os import listdir
from os.path import isfile, join
import time

def Hyperparameter_Decoder(Hyperparameter):
"""
Hyperparameter Optimization: A Spectral Approach
|===============================================================================================|
| Num|              Type                 |             0            |              1            |
|===============================================================================================|
| 00 | Weight Initailization             | standard initializations | other initializations     |
| 01 | Weight Initialization             | ...                                                  |
| 02 | Optimization Method               | SGD                      | ADAM?                     |
| 03 | Initial Learning Rate             | < 0.01                   | >= 0.01                   |
| 04 | Initial Learning Rate             | < 0.001; < 0.1           | >=0.001; >= 0.1           |
| 05 | Initial Learning Rate             | 0.0001; 0.001; 0.01; 0.1 | 0.0003; 0.003; 0.03; 0.3  |
| 06 | Learning Rate Drop                | No                       | Yes                       |
| 07 | Learning Rate First Drop Time     | Drop by 1/10 at Epoch 40 | Drop by 1/10 at Epoch 60  |
| 08 | Learning Rate Second Drop Time    | Drop by 1/10 at Epoch 80 | Drop by 1/10 at Epoch 100 |
| 09 | Use Momentum                      | No                       | Yes                       |
| 10 | Momentum Rate                     | 0.9                      | 0.99                      |
| 11 | Initial Residual Link Weight      | (No Use)                                             | 
| 12 | Tune Residual Link Weight         | (No Use)                                             | 
| 13 | Tune Time of Residual Link Weight | (No Use)                                             | 
| 14 | Resblock First Activation         | (No Use)                                             | 
| 15 | Resblock Second Activation        | (No Use)                                             | 
| 16 | Resblock Third Activation         | (No Use)                                             | 
| 17 | Convolution bias                  | No                       | Yes                       |
| 18 | Activation                        | ReLU                     | Others                    |
| 19 | Activation                        | ReLU; Sigmoid            | ReLU; Tanh                |
| 20 | Use Dropout                       | No                       | Yes                       |
| 21 | Dropout Rate                      | Low                      | High                      |
| 22 | Dropout Rate                      | 0.05; 0.2                | 0.1; 0.3                  |
| 23 | Batch Normalization               | No                       | Yes                       |
| 24 | Batch Normalization Tuning        | (No Use)                                             |  
| 25 | Resnet Shortcut Type              | (No Use)                                             | 
| 26 | Resnet shortcut Type              | (No Use)                                             |
| 27 | Weight Decay                      | No                       | Yes                       | 
| 28 | Weight Decay Parameter            | 1e-4                     | 1e-4                      |
| 29 | Batch Size                        | Small                    | Big                       |
| 30 | Batch Size                        | 32; 128                  | 64; 256                   |
| 31 | Optnet                            | (No Use)                                             |
| 32 | Share gradInput                   | (No Use)                                             |
|===============================================================================================|
"""
	# Weight Initialization
	if   Hyperparameter[0:1]==[0, 0]:
	elif Hyperparameter[0:1]==[0, 1]:
	
	if   Hyperparameter[2]==0:
		if   Hyperparameter[3]==0:
			if   Hyperparameter[4]==0:
			elif Hyperparameter[4]==1:
		elif Hyperparameter[3]==1:
			if   Hyperparameter[4]==0:
			elif Hyperparameter[4]==1:
	elif Hyperparameter[2]==1:
		if   Hyperparameter[1]==0:
		elif Hyperparameter[1]==1:

	return Hyper_dict
	
def dataset_parser(
	# Path
	Dataset,	      # e.g. 'CamVid'
	Path, 		      # e.g. '/Path_to_Dataset/CamVid'
	Y_pre_Path,       # e.g. '/Path_to_Y_pre/CamVid'
	data_index,
	target_index,

	# Variable
	class_num,
	layer=None, # Choose the needed layer 
	H_resize=224,
	W_resize=224,

	# Parameter
	IS_STUDENT=False,
	IS_TRAINING=False
	):
	
	# Dataset
	if IS_TRAINING and IS_STUDENT:	 
		data  , data_shape = read_dataset_file(data_index, Path, Dataset, H_resize, W_resize)
		target, _          = read_Y_pre_file(data_index, Y_pre_Path, layer)
	else:
		data  , data_shape = read_dataset_file(data_index, Path, Dataset, H_resize, W_resize)
		target, _          = read_dataset_file(target_index, Path, Dataset, H_resize, W_resize, True)
	
	#print("Class Num : {CN}" .format(CN=np.max(target)))

	target = one_of_k(target, class_num)
	#print("Shape of data   : {Shape}" .format(Shape = np.shape(data)))
	#print("Shape of target : {Shape}" .format(Shape = np.shape(target)))
	
	return data, target

def file_name_parser(filepath):
	filename = [f for f in listdir(filepath) if isfile(join(filepath, f))]

	return filename


def Pyramid_Pooling(net, strides, output_channel,
	is_training, 
	is_testing ):
	input_shape = net.get_shape().as_list()
	for level, stride in enumerate(strides):
		with tf.variable_scope('pool%d' %(level)):
			net_tmp = tf.nn.avg_pool(net, ksize=[1, stride, stride, 1], strides=[1, stride, stride, 1], padding='SAME')
			net_tmp = conv2D(net_tmp, kernel_size=1, stride=1, output_channel=output_channel, rate=1,
							is_shortcut		= False, 
							is_bottleneck	= False, 
							is_batch_norm	= True, 
							is_training		= is_training, 
							is_testing		= is_testing, 
							is_dilated		= False, 
							scope			= "1x1")
			net_tmp = tf.image.resize_images(net_tmp, [input_shape[1], input_shape[2]])
		net = tf.concat([net, net_tmp], axis=3)
	return net
	

def Testing(
	# Training & Validation & Testing Data
	train_data					, 
	train_target				,
	train_data_index			,
	valid_data					,
	valid_target				,
	valid_data_index			,
	test_data					, 
	test_target					,
	test_data_index				,
	
	# Parameter	
	BATCH_SIZE					,
	IS_SAVING_RESULT_AS_IMAGE	,
	IS_SAVING_RESULT_AS_NPZ		,
	IS_SAVING_PARTIAL_RESULT	,
	IS_TRAINING					,
	IS_TERNARY					,
	IS_QUANTIZED_ACTIVATION		,
	
	
	# Tensor	
	prediction					,
	
	# Placeholder	
	xs							, 
	ys							,
	is_training					,
	is_testing					,
	is_quantized_activation		,
		
	# File Path (For Loading Trained Weight)
	TESTING_WEIGHT_FILE			= None,
		
	# Trained Weight Parameters (For Loading Trained Weight)
	parameters					= None,
	saver						= None,

	# File Path (For Saving Result)
	train_target_path 			= None,
	train_Y_pre_path  			= None,
	valid_target_path 			= None,
	valid_Y_pre_path  			= None,
	test_target_path 			= None,
	test_Y_pre_path  			= None,

	# Collection 
	partial_output_collection	= None,

	# Session
	sess						= None):
	
	is_validation = False

	# Load trained weights
	print("")
	print("Loading the trained weights ... ")
	print("\n\033[1;32;40mWeights File\033[0m : {WF}" .format(WF = TESTING_WEIGHT_FILE))
	if parameters==None:
		save_path = saver.restore(sess, TESTING_WEIGHT_FILE + ".ckpt")
	else:
		load_pre_trained_weights(parameters, pre_trained_weight_file=TESTING_WEIGHT_FILE, sess=sess) 
	
	#***********#
	#	TRAIN	#
	#***********#
	print("train result ... ")
	train_result, train_accuracy, train_accuracy_top2, train_accuracy_top3, Y_pre_train = compute_accuracy(
				xs                      = xs, 
				ys                      = ys, 
				is_training             = is_training, 
				is_testing              = is_testing, 
				is_validation           = is_validation, 
				is_quantized_activation = is_quantized_activation, 
				QUANTIZED_NOW           = IS_QUANTIZED_ACTIVATION, 
				prediction              = prediction, 
				v_xs                    = train_data, 
				v_ys                    = train_target, 
				BATCH_SIZE              = BATCH_SIZE, 
				sess                    = sess)
	print("Train Data Accuracy = {Train_Accuracy}, top2 = {top2}, top3 = {top3}"
	.format(Train_Accuracy=train_accuracy, top2=train_accuracy_top2, top3=train_accuracy_top3))

	
	#***********#
	#	VALID	#
	#***********#
	print(" valid result ... ")
	valid_result, valid_accuracy, valid_accuracy_top2, valid_accuracy_top3, Y_pre_valid = compute_accuracy(
				xs                      = xs, 
				ys                      = ys, 
				is_training             = is_training, 
				is_testing              = is_testing, 
				is_validation           = is_validation, 
				is_quantized_activation = is_quantized_activation, 
				QUANTIZED_NOW           = IS_QUANTIZED_ACTIVATION, 
				prediction              = prediction, 
				v_xs                    = valid_data, 
				v_ys                    = valid_target, 
				BATCH_SIZE              = BATCH_SIZE, 
				sess                    = sess)
	print("Valid Data Accuracy = {Valid_Accuracy}, top2 = {top2}, top3 = {top3}"
	.format(Valid_Accuracy=valid_accuracy, top2=valid_accuracy_top2, top3=valid_accuracy_top3))
	

	#***********#
	#	TEST	#
	#***********#
	print(" test result ... ")
	test_result, test_accuracy, test_accuracy_top2, test_accuracy_top3, Y_pre_test = compute_accuracy(
				xs                      = xs, 
				ys                      = ys, 
				is_training             = is_training, 
				is_testing              = is_testing, 
				is_validation           = is_validation, 
				is_quantized_activation = is_quantized_activation, 
				QUANTIZED_NOW           = IS_QUANTIZED_ACTIVATION, 
				prediction              = prediction, 
				v_xs                    = test_data, 
				v_ys                    = test_target, 
				BATCH_SIZE              = BATCH_SIZE, 
				sess                    = sess)
	print("Test Data Accuracy = {Test_Accuracy}, top2 = {top2}, top3 = {top3}"
	.format(Test_Accuracy=test_accuracy, top2=test_accuracy_top2, top3=test_accuracy_top3))
	
	# Print The Total Reulst Again
	print("")
	print("Train Data Accuracy = {Train_Accuracy}, top2 = {top2}, top3 = {top3}"
	.format(Train_Accuracy=train_accuracy, top2=train_accuracy_top2, top3=train_accuracy_top3))
	print("Valid Data Accuracy = {Valid_Accuracy}, top2 = {top2}, top3 = {top3}"
	.format(Valid_Accuracy=valid_accuracy, top2=valid_accuracy_top2, top3=valid_accuracy_top3))
	print("Test Data Accuracy = {Test_Accuracy}, top2 = {top2}, top3 = {top3}"
	.format(Test_Accuracy=test_accuracy, top2=test_accuracy_top2, top3=test_accuracy_top3))
	
	# Saving Result as Image
	if IS_SAVING_RESULT_AS_IMAGE:
		print("Coloring train result ... ")
		train_result = color_result(train_result)
		print("Saving the train data result as image ... ")
		Save_result_as_image(train_target_path, train_result, train_data_index)
		
		print("Coloring valid result ... ")
		valid_result = color_result(valid_result)
		print("Saving the validation data result as image ... ")
		Save_result_as_image(valid_target_path, valid_result, valid_data_index)
		
		print("Coloring test result ... ")
		test_result = color_result(test_result)
		print("Saving the test data result as image ... ")
		Save_result_as_image(test_target_path, test_result, test_data_index)

	# Saving Result as NPZ File
	if IS_SAVING_RESULT_AS_NPZ:
		print("Saving the train Y_pre result as npz ... ")
		Save_result_as_npz(	train_Y_pre_path, Y_pre_train, train_data_index,
							IS_SAVING_PARTIAL_RESULT,
							partial_output_collection, 
							# tensor
							xs						= xs,
							is_training				= is_training, 
							is_testing				= is_testing,              
							is_quantized_activation	= is_quantized_activation,
							# data
							v_xs					= train_data,
							# Parameter
							QUANTIZED_NOW 			= IS_QUANTIZED_ACTIVATION,
							sess					= sess)
		#Save_result_as_npz(train_Y_pre_path, Y_pre_train, CamVid_train_data_index, IS_SAVING_PARTIAL_RESULT)
		print("Saving the valid Y_pre result as npz ... ")
		Save_result_as_npz(	valid_Y_pre_path, Y_pre_valid, valid_data_index,
							IS_SAVING_PARTIAL_RESULT, 
							partial_output_collection, 
							# tensor
							xs						= xs,
							is_training				= is_training, 
							is_testing				= is_testing,              
							is_quantized_activation	= is_quantized_activation,
							# data
							v_xs					= valid_data,
							# Parameter
							QUANTIZED_NOW 			= IS_QUANTIZED_ACTIVATION,
							sess					= sess)
		#Save_result_as_npz(valid_Y_pre_path, Y_pre_valid, CamVid_valid_data_index, IS_SAVING_PARTIAL_RESULT)
		print("Saving the test Y_pre result as npz ... ")
		Save_result_as_npz(	test_Y_pre_path, Y_pre_test, test_data_index,
							IS_SAVING_PARTIAL_RESULT, 
							partial_output_collection, 
							# tensor
							xs						= xs,
							is_training				= is_training, 
							is_testing				= is_testing,              
							is_quantized_activation	= is_quantized_activation,
							# data
							v_xs					= test_data,
							# Parameter
							QUANTIZED_NOW 			= IS_QUANTIZED_ACTIVATION,
							sess					= sess)
		#Save_result_as_npz(test_Y_pre_path, Y_pre_test, CamVid_test_data_index, IS_SAVING_PARTIAL_RESULT)

	# Save Result as .csv
	accuracy_train = np.concatenate([[[train_accuracy]], [[train_accuracy_top2]], [[train_accuracy_top3]]], axis=1)
	accuracy_valid = np.concatenate([[[valid_accuracy]], [[valid_accuracy_top2]], [[valid_accuracy_top3]]], axis=1)
	accuracy_test  = np.concatenate([[[ test_accuracy]], [[ test_accuracy_top2]], [[ test_accuracy_top3]]], axis=1)
	accuracy       = np.concatenate([accuracy_train, accuracy_valid, accuracy_test], axis=0)
	Save_file_as_csv(TESTING_WEIGHT_FILE+'_accuracy' , accuracy)


def Training_and_Validation( 
		# (File Read) Path
		Dataset	                , 
		Dataset_Path            ,
		Y_pre_Path              ,
		
		# (File Read) Variable
		class_num               , 
		layer                   , 
		H_resize                ,
		W_resize                ,
		
		# (File Read) Parameter
		IS_STUDENT				,

		# Parameter	
		EPOCH_TIME				,
		BATCH_SIZE				,
		LR_DECADE				,
		EPOCH_DECADE			,
		lr						,
		
		# Tensor	
		train_step				,
		loss					,
		prediction				,
		
		# Placeholder	
		xs						, 
		ys						,
		learning_rate			,
		is_training				,
		is_testing				,
		
		# (Saving Trained Weight) Collection
		mean_collection					= None,
		var_collection					= None,
		trained_mean_collection			= None,
		trained_var_collection			= None,
		params							= None,
		
		# (Saving Trained Weight) File Path
		TRAINED_WEIGHT_FILE				= None,
		TRAINING_WEIGHT_FILE			= None,
		IS_SAVER						= True, 
		
		# (Saving Trained Weight) Trained Weight Parameters 
		parameters						= None,
		saver							= None,
				
		# (GAN) Parameter		
		IS_GAN							= False,
		DISCRIMINATOR_STEP				= 1,

		# (GAN) tensor
		train_step_Gen					= None,			
		train_step_Dis					= None,
		loss_Gen						= None,
		loss_Dis						= None,
		prediction_Gen					= None,
		prediction_Dis_0				= None,
		prediction_Dis_1				= None,
		
		# (ternary) Parameter			
		IS_TERNARY						= None,
		TERNARY_EPOCH					= None,
				
		# (ternary) Placeholder		
		is_ternary						= None,
				
		# (ternary) Collection		
		weights_collection				= None,
		biases_collection				= None,
		ternary_weights_bd_collection 	= None,
		ternary_biases_bd_collection	= None,
		
		# (ternary) variable
		weights_bd_ratio				= 50,
		biases_bd_ratio					= 50,
		
		# (assign final weights)
		assign_var_list_collection 		= None,
		
		# (quantize actvation) parameter
		IS_QUANTIZED_ACTIVATION			= None,
		QUANTIZED_ACTIVATION_EPOCH		= None,

		# (quantize actvation) tensor
		is_quantized_activation			= None,

		# (quantize actvation) collection
		activation_collection		    = None, 
		mantissa_collection			    = None, 
		fraction_collection     	    = None, 

		# (debug)
		final_weights_collection		= None,
		final_net_collection			= None,

		# Session
		sess							= None):

	#-------------------------------#
	#	Loading Pre-trained Model	#
	#-------------------------------#
	if TRAINED_WEIGHT_FILE!=None:
		print ""
		print("Loading Pre-trained weights ...")
		if parameters==None:
			save_path = saver.save(sess, TRAINED_WEIGHT_FILE + ".ckpt")
			print(save_path)
		else:
			load_pre_trained_weights(parameters, pre_trained_weight_file=TRAINED_WEIGHT_FILE, sess=sess)

	#-----------------------------#
	#   Some Controk Parameters   #
	#-----------------------------#
	QUANTIZED_NOW = False
	TERNARY_NOW = False

	#---------------#
	#   File Read   #
	#---------------#
	train_data_index   = np.array(open(Dataset_Path + '/train.txt', 'r').read().splitlines())
	train_target_index = np.array(open(Dataset_Path + '/trainannot.txt', 'r').read().splitlines())
	train_data_num = len(train_data_index)
	print("Training Data Number = {DS}" .format(DS = train_data_num))
	
	shuffle_index = np.arange(train_data_num)
	np.random.shuffle(shuffle_index)

	train_data_index   = train_data_index  [shuffle_index]
	train_target_index = train_target_index[shuffle_index]

	val_data_index   = open(Dataset_Path + '/val.txt', 'r').read().splitlines()
	val_target_index = open(Dataset_Path + '/valannot.txt', 'r').read().splitlines()
	val_data_num = len(val_data_index)
	print("Validation Data Number = {DS}" .format(DS = val_data_num))
	
	Data_Size_Per_Iter = 15

	#---------------#
	#	Per Epoch	#
	#---------------#
	for epoch in range(EPOCH_TIME):
		Train_acc = 0
		Train_loss = 0

		###
		tStart = time.time()
		###

		for data_iter in range((train_data_num/Data_Size_Per_Iter)+1):
			#-------------------#
			#   Training file   #
			#-------------------#
			# data_index_part
			if train_data_num<Data_Size_Per_Iter:
				train_data_index_part   = train_data_index
				train_target_index_part = train_target_index
			else:
				train_data_index_part   = train_data_index[data_iter*Data_Size_Per_Iter:(data_iter+1)*Data_Size_Per_Iter]
				train_target_index_part = train_target_index[data_iter*Data_Size_Per_Iter:(data_iter+1)*Data_Size_Per_Iter]

			print("Loading Training Data ...")
			train_data, train_target = dataset_parser(
				# Path
				Dataset	         = Dataset,
				Path             = Dataset_Path, 
				Y_pre_Path       = Y_pre_Path,
				data_index       = train_data_index_part,
				target_index     = train_target_index_part,		
				
				# Variable
				class_num        = class_num,
				layer            = layer,
				H_resize         = H_resize,
				W_resize         = W_resize,

				# Parameter
				IS_STUDENT       = IS_STUDENT ,
				IS_TRAINING      = True)
			
			#-------------------#
			#  Start Training   #
			#-------------------#
			data_shape = np.shape(train_data) 
			train_data, train_target = shuffle_image(train_data, train_target)

			
			#--------------------------#
			#   Quantizad Activation   #
			#--------------------------#
			if IS_QUANTIZED_ACTIVATION and (epoch+1)==QUANTIZED_ACTIVATION_EPOCH:
				batch_xs = train_data[0 : BATCH_SIZE]
				# Calculate Each Activation's appropriate mantissa and fractional bit
				m, f = quantized_m_and_f(activation_collection, xs, is_training, is_testing, is_quantized_activation, batch_xs, sess)	
				# Assign mantissa and fractional bit to the tensor
				assign_quantized_m_and_f(mantissa_collection, fraction_collection, m, f, sess)
				
				# Start Quantize Activation
				QUANTIZED_NOW = True
			
			#-------------#
			#   Ternary   #
			#-------------#
			if IS_TERNARY and (epoch+1)==TERNARY_EPOCH:
				# Calculate the ternary boundary of each layer's weights
				weights_bd, biases_bd, weights_table, biases_table = tenarized_bd(weights_collection,  biases_collection, weights_bd_ratio, biases_bd_ratio, sess)

				# assign ternary boundary to tensor
				assign_ternary_boundary(ternary_weights_bd_collection, ternary_biases_bd_collection, weights_bd, biases_bd, sess)

				# Start Quantize Activation
				TERNARY_NOW = True

			print("Training ... ")
			for i in range(int(data_shape[0]/BATCH_SIZE)):
				# Train data in BATCH SIZE
				batch_xs    = train_data   [i*BATCH_SIZE : (i+1)*BATCH_SIZE]
				batch_ys    = train_target [i*BATCH_SIZE : (i+1)*BATCH_SIZE]
				
				#-----------------------#
				#	Run Training Step	#
				#-----------------------#	
				if IS_GAN:
				# (GAN) Generator
					_, Loss, Prediction, Prediction_Dis_0 = sess.run([train_step_Gen, loss_Gen, prediction_Gen, prediction_Dis_0], feed_dict={xs: batch_xs, ys: batch_ys, learning_rate: lr, is_training: True, is_testing: False})
				# (GAN) Discriminator
					if i % DISCRIMINATOR_STEP==0:
						_, Loss_Dis, Prediction_Dis_1 = sess.run([train_step_Dis, loss_Dis, prediction_Dis_1], feed_dict={xs: batch_xs, ys: batch_ys, learning_rate: lr, is_training: True, is_testing: False})
				else:
				# (Normal)
					_, Loss, Prediction = sess.run([train_step, loss, prediction], feed_dict={xs: batch_xs, ys: batch_ys, learning_rate: lr, is_training: True, is_testing: False, is_ternary: TERNARY_NOW, is_quantized_activation: QUANTIZED_NOW})

					for assign_var_list_iter, assign_var_list in enumerate(assign_var_list_collection):
						sess.run(assign_var_list, feed_dict={is_ternary: TERNARY_NOW})
				
				#-----------#
				#	Result	#
				#-----------#
				y_pre = np.argmax(Prediction, -1)
				batch_accuracy = np.mean(np.equal(np.argmax(batch_ys, -1), y_pre))
				
				Train_acc  = Train_acc  + batch_accuracy
				Train_loss = Train_loss + np.mean(Loss)

				# Print Training Result Per Batch Size
				print("\033[1;34;40mEpoch\033[0m : {ep}" .format(ep = epoch))
				print("\033[1;34;40mData Iteration\033[0m : {Iter}" .format(Iter = i*BATCH_SIZE+data_iter*Data_Size_Per_Iter))
				print("\033[1;32;40m  Batch Accuracy\033[0m : {acc}".format(acc = batch_accuracy))
				print("\033[1;32;40m  Loss          \033[0m : {loss}".format(loss = np.mean(Loss)))
				print("\033[1;32;40m  Learning Rate \033[0m : {LearningRate}".format(LearningRate = lr))

				# GAN Discriminator Result
				if IS_GAN:
					batch_accuracy_Dis = np.mean(np.concatenate([Prediction_Dis_0, Prediction_Dis_1], axis=0))
					print("  Discriminator Output : {Dis_Out}" ,format(Dis_Out=batch_accuracy_Dis))

				# Per Class Accuracy
				"""
				per_class_accuracy(Prediction, batch_ys)
				"""
			#----------------------------------------------------------------------#
			#    Show Difference between quanitzed and non-quantized activation    #
			#----------------------------------------------------------------------#
			print("")
			if QUANTIZED_NOW:
				batch_xs = train_data[0 : BATCH_SIZE]
				for i, final_net in enumerate(final_net_collection):
					TRUE   = sess.run(final_net, feed_dict={xs: batch_xs, is_quantized_activation: True })
					FALSE  = sess.run(final_net, feed_dict={xs: batch_xs, is_quantized_activation: False})
					print("{iter}: {TRUE_MEAN}	{FALSE_MEAN}	{DIFF_MEAN}" .format(iter=i, TRUE_MEAN=np.mean(TRUE), FALSE_MEAN=np.mean(FALSE), DIFF_MEAN=np.mean(np.absolute(TRUE-FALSE))))
		###
		tEnd = time.time()
		print("It cost {TIME} sec" .format(TIME=tEnd - tStart))
		###
		
		# Record Per Epoch Training Result (Finally this will save as the .csv file)
		Train_acc  = Train_acc  / float(int(data_shape[0]/BATCH_SIZE)) / float(int(train_data_num/Data_Size_Per_Iter)) 
		Train_loss = Train_loss / float(int(data_shape[0]/BATCH_SIZE)) / float(int(train_data_num/Data_Size_Per_Iter)) 

		if epoch==0:
			Train_acc_per_epoch  = np.array([Train_acc ])
			Train_loss_per_epoch = np.array([Train_loss])
		else:
			Train_acc_per_epoch  = np.concatenate([Train_acc_per_epoch , np.array([Train_acc ])], axis=0)
			Train_loss_per_epoch = np.concatenate([Train_loss_per_epoch, np.array([Train_loss])], axis=0)
		

		
		#---------------#
		#	Validation	#
		#---------------#
		total_valid_accuracy = 0
		for iter in range(val_data_num/Data_Size_Per_Iter):
			#---------------------#
			#   Validation file   #
			#---------------------#
			# data_index_part
			if val_data_num<Data_Size_Per_Iter:
				val_data_index_part   = val_data_index
				val_target_index_part = val_target_index
			else:
				val_data_index_part   = val_data_index[iter*Data_Size_Per_Iter:(iter+1)*Data_Size_Per_Iter]
				val_target_index_part = val_target_index[iter*Data_Size_Per_Iter:(iter+1)*Data_Size_Per_Iter]

			print("")
			print("Loading Validation Data ...")
			val_data , val_target = dataset_parser(
				# Path
				Dataset	         = Dataset,
				Path             = Dataset_Path, 
				Y_pre_Path       = Y_pre_Path,
				data_index       = val_data_index_part,
				target_index     = val_target_index_part,		
				
				# Variable
				class_num        = class_num,
				layer            = layer,
				H_resize         = H_resize,
				W_resize         = W_resize,
				# Parameter
				IS_STUDENT       = IS_STUDENT ,
				IS_TRAINING      = True)

			print("")
			print("Epoch : {ep}".format(ep = epoch))
			print("Validation ... ")
			is_validation = True 
			_, valid_accuracy, _, _, _ = compute_accuracy(
						xs                      = xs, 
						ys                      = ys, 
						is_training             = is_training, 
						is_testing              = is_testing, 
						is_validation           = is_validation, 
						is_quantized_activation = is_quantized_activation, 
						QUANTIZED_NOW           = QUANTIZED_NOW, 
						prediction              = prediction, 
						v_xs                    = valid_data, 
						v_ys                    = valid_target, 
						BATCH_SIZE              = BATCH_SIZE, 
						sess                    = sess)
			is_validation = False
			total_valid_accuracy = total_valid_accuracy + valid_accuracy

		if val_data_num<Data_Size_Per_Iter:
			total_valid_accuracy = total_valid_accuracy
		else:
			total_valid_accuracy = total_valid_accuracy / float(int(val_data_num / Data_Size_Per_Iter))
			
		print("")
		print("Validation Accuracy = {Valid_Accuracy}".format(Valid_Accuracy=total_valid_accuracy))
		
		# Learning Rate Decay
		if (((epoch+1) % EPOCH_DECADE==0) and (epoch != 0)):
			lr = lr / LR_DECADE
			
		#---------------------------#
		#	Saving trained weights	#
		#---------------------------#
		print("Saving Trained Weights ... ")
	#	batch_xs = train_data[0 : BATCH_SIZE]
	#	assign_trained_mean_and_var(mean_collection, var_collection, trained_mean_collection, trained_var_collection, params, xs, ys, is_training, is_testing, batch_xs, sess)
		
		# Every 10 epoch 
		if (((epoch+1)%10==0) and ((epoch+1)>=10)):
			if IS_SAVER:
				save_path = saver.save(sess, TRAINING_WEIGHT_FILE + '_' + str(epoch+1) + ".ckpt")
				print(save_path)
			else:
				save_pre_trained_weights( (TRAINING_WEIGHT_FILE+'_'+str(epoch+1)), parameters, xs, batch_xs, sess)

	#-----------------------------------#
	#	Saving Train info as csv file	#
	#-----------------------------------#
	Train_acc_per_epoch  = np.expand_dims(Train_acc_per_epoch , axis=1)
	Train_loss_per_epoch = np.expand_dims(Train_loss_per_epoch, axis=1)
	Train_info = np.concatenate([Train_acc_per_epoch, Train_loss_per_epoch], axis=1)
	Save_file_as_csv(TRAINING_WEIGHT_FILE+'_train_info' , Train_info)

def calculate_class_weight(y):
	[NUM, H, W, DIM] = np.shape(y)
	class_num = np.zeros([DIM])
	for i in range(DIM):
		class_num[i] = np.count_nonzero(np.equal(np.argmax(y, -1), i))
	class_num = 1 / class_num
	total = np.sum(class_num)
	return class_num / total
		
		
		
def shuffle_image(x, y):
	[NUM, HEIGHT, WIDTH, DIM] = np.shape(x)
	
	shuffle_index = np.arange(NUM)
	np.random.shuffle(shuffle_index)
	
	#index_shuffle = index[shuffle_index]
	x_shuffle = x[shuffle_index, :, :, :]
	y_shuffle = y[shuffle_index, :, :, :]
	
	return x_shuffle, y_shuffle

def color_result(result):
	#***************************************#
	#	class0 : (	128 	128 	128	)	#
	#	class1 : (	128 	0 		0	)	#
	#	class2 : (	192 	192 	128	)	#
	#	class3 : (	128 	64 		128	)	#
	#	class4 : (	0 		0 		192	)	#
	#	class5 : (	128 	128 	0	)	#
	#	class6 : (	192 	128 	128	)	#
	#	class7 : (	64 		64 		128	)	#
	#	class8 : (	64 		0 		128	)	#
	#	class9 : (	64 		64 		0	)	#
	#	class10 : (	0		128 	192	)	#
	#	class11 : (	0		0		0	)	#
	#***************************************#
	shape = np.shape(result)
	
	RGB = np.zeros([shape[0], shape[1], shape[2], 3], np.uint8)
	
	for i in range(shape[0]):
		for x in range(shape[1]):
			for y in range(shape[2]):
				if result[i][x][y] == 0:
					RGB[i][x][y][0] = np.uint8(128)
					RGB[i][x][y][1] = np.uint8(128)
					RGB[i][x][y][2] = np.uint8(128)
				elif result[i][x][y] == 1:
					RGB[i][x][y][0] = np.uint8(128) 
					RGB[i][x][y][1] = np.uint8(0)
					RGB[i][x][y][2] = np.uint8(0) 
				elif result[i][x][y] == 2:
					RGB[i][x][y][0] = np.uint8(192)
					RGB[i][x][y][1] = np.uint8(192)
					RGB[i][x][y][2] = np.uint8(128)
				elif result[i][x][y] == 3:
					RGB[i][x][y][0] = np.uint8(128)
					RGB[i][x][y][1] = np.uint8(64)
					RGB[i][x][y][2] = np.uint8(128)
				elif result[i][x][y] == 4:
					RGB[i][x][y][0] = np.uint8(0)
					RGB[i][x][y][1] = np.uint8(0)
					RGB[i][x][y][2] = np.uint8(192)
				elif result[i][x][y] == 5:
					RGB[i][x][y][0] = np.uint8(128)
					RGB[i][x][y][1] = np.uint8(128)
					RGB[i][x][y][2] = np.uint8(0)
				elif result[i][x][y] == 6:
					RGB[i][x][y][0] = np.uint8(192)
					RGB[i][x][y][1] = np.uint8(128)
					RGB[i][x][y][2] = np.uint8(128)
				elif result[i][x][y] == 7:
					RGB[i][x][y][0] = np.uint8(64)
					RGB[i][x][y][1] = np.uint8(64)
					RGB[i][x][y][2] = np.uint8(128)
				elif result[i][x][y] == 8:
					RGB[i][x][y][0] = np.uint8(64)
					RGB[i][x][y][1] = np.uint8(0)
					RGB[i][x][y][2] = np.uint8(128)
				elif result[i][x][y] == 9:
					RGB[i][x][y][0] = np.uint8(64)
					RGB[i][x][y][1] = np.uint8(64)
					RGB[i][x][y][2] = np.uint8(0)
				elif result[i][x][y] == 10:
					RGB[i][x][y][0] = np.uint8(0)
					RGB[i][x][y][1] = np.uint8(128)
					RGB[i][x][y][2] = np.uint8(192)
				elif result[i][x][y] == 11:
					RGB[i][x][y][0] = np.uint8(0)
					RGB[i][x][y][1] = np.uint8(0)
					RGB[i][x][y][2] = np.uint8(0)
	return RGB

def read_dataset_file(data_index, Path, Dataset, H_resize, W_resize, IS_TARGET=False):
	for i, file_name in enumerate(data_index):
		# Read Data
		data_tmp  = misc.imread(Path + file_name)
		shape_tmp = np.shape(data_tmp)[0:2]

		# Data Preprocessing
		if Dataset=='ade20k':
			data_tmp = scipy.misc.imresize(data_tmp, (H_resize, W_resize))
			if len(np.shape(data_tmp))==2 and (not IS_TARGET):
				data_tmp = np.repeat(data_tmp[:, :, np.newaxis], 3, axis=2)


		# Concatenate the Data
		if i==0:
			data  = np.expand_dims(data_tmp , axis=0)
			shape = np.expand_dims(shape_tmp, axis=0)
		else:
			data  = np.concatenate([data , np.expand_dims(data_tmp , axis=0)], axis=0)
			shape = np.concatenate([shape, np.expand_dims(shape_tmp, axis=0)], axis=0)
	return data, shape

def Save_result_as_image(Path, result, file_index):
	for i, target in enumerate(result):
		scipy.misc.imsave(Path + file_index[i], target)
	
def Save_result_as_npz(	Path, result, file_index,
						#--------------------------#
						# Saving Each Layer Result #
						#--------------------------#
						IS_SAVING_PARTIAL_RESULT, 
						partial_output_collection, 
						# tensor
						xs,
						is_training,
						is_testing,              
						is_quantized_activation,
						# data
						v_xs,
						# Parameter
						QUANTIZED_NOW,
						sess,
						last_x_layer = 5
						):

	if IS_SAVING_PARTIAL_RESULT:
		for last_layer in range(last_x_layer):
			layer_now = np.shape(partial_output_collection)[0]-1-last_layer
			partial_output = partial_output_collection[layer_now]
			for i in range(np.shape(v_xs)[0]):
				file_index[i] = file_index[i].split('.')[0]
				v_xs_part = np.expand_dims(v_xs[i], axis=0)
	   			target = sess.run(partial_output, feed_dict={xs: v_xs_part, is_training: False, is_testing: True, is_quantized_activation: QUANTIZED_NOW})
	   			np.savez(Path + file_index[i] + '_' + str(layer_now), target)
	else:
		for i, target in enumerate(result):
			file_index[i] = file_index[i].split('.')[0]
			np.savez(Path + file_index[i], target)

def Save_file_as_csv(Path, file):
	np.savetxt(Path + '.csv', file, delimiter=",")
	
def one_of_k(target, class_num):
	target.astype('int64')
	one_of_k_target = np.zeros([np.shape(target)[0], np.shape(target)[1], np.shape(target)[2], class_num])
	
	meshgrid_target = np.meshgrid(np.arange(np.shape(target)[1]), np.arange(np.shape(target)[0]), np.arange(np.shape(target)[2]))
	
	one_of_k_target[meshgrid_target[1], meshgrid_target[0], meshgrid_target[2], target] = 1
	
	return one_of_k_target

def per_class_accuracy(prediction, batch_ys):
	print("  Per Class Accuracy	: ")
	[BATCH, HEIGHT, WIDTH, CLASS_NUM] = np.shape(batch_ys)
	correct_num = np.zeros([CLASS_NUM, 1])
	total_num = np.zeros([CLASS_NUM, 1])
	
	print_per_row = 10
	cn = np.zeros([print_per_row], np.int32)
	tn = np.zeros([print_per_row], np.int32)

	for i in range(CLASS_NUM):
		y_tmp = np.equal(np.argmax(batch_ys, -1), i)
		p_tmp = np.equal(np.argmax(prediction, -1), i)
		total_num = np.count_nonzero(y_tmp)
		zeros_num = np.count_nonzero( (p_tmp+y_tmp) == 0)
		correct_num = np.count_nonzero(np.equal(y_tmp, p_tmp)) - zeros_num
		if total_num == 0:
			accuracy = -1
		else:
			accuracy = float(correct_num) / float(total_num)
		

		if CLASS_NUM <= 15:
			print("    Class{Iter}	: {predict} / {target}".format(Iter = i, predict=correct_num, target=total_num))
		else:
			iter = i%print_per_row
			cn[iter] = correct_num
			tn[iter] = total_num
			if i%print_per_row==0:
				print("    Class{Iter}	: {predict} / {target}".format(Iter = i, predict=np.sum(cn), target=np.sum(tn)))
	
		
def compute_accuracy(xs, ys, is_training, is_testing, is_validation, is_quantized_activation, QUANTIZED_NOW, prediction, v_xs, v_ys, BATCH_SIZE, sess):
	test_batch_size = BATCH_SIZE
	batch_num = len(v_xs) / test_batch_size
	result_accuracy_top1 = 0
	result_accuracy_top2 = 0
	result_accuracy_top3 = 0
	for i in range(int(batch_num)):	
		v_xs_part = v_xs[i*test_batch_size : (i+1)*test_batch_size, :]
		v_ys_part = v_ys[i*test_batch_size : (i+1)*test_batch_size, :]
			
		Y_pre = sess.run(prediction, feed_dict={xs: v_xs_part, is_training: False, is_testing: (not is_validation), is_quantized_activation: QUANTIZED_NOW})
				
		# for post-processing
		if i==0:
			SegNet_Y_pre = Y_pre;
			#SegNet_Y_pre_annot = v_ys_part
		else:
			SegNet_Y_pre = np.concatenate([SegNet_Y_pre, Y_pre], axis=0)
	    	#SegNet_Y_pre_annot = np.concatenate([SegNet_Y_pre_annot, v_ys_part], axis=0)
		#--------------------

		top1 = np.argsort(-Y_pre, axis=-1)[:, :, :, 0] 
		top2 = np.argsort(-Y_pre, axis=-1)[:, :, :, 1] 
		top3 = np.argsort(-Y_pre, axis=-1)[:, :, :, 2] 
		
		correct_prediction_top1 = np.equal(top1, np.argmax(v_ys_part, -1))
		correct_prediction_top2 = np.equal(top2, np.argmax(v_ys_part, -1)) | correct_prediction_top1
		correct_prediction_top3 = np.equal(top3, np.argmax(v_ys_part, -1)) | correct_prediction_top2
		
		accuracy_top1 = np.mean(correct_prediction_top1.astype(float))
		accuracy_top2 = np.mean(correct_prediction_top2.astype(float))
		accuracy_top3 = np.mean(correct_prediction_top3.astype(float))
		
		print("Part : {iter}, top1 Accuracy = {Accuracy_top1}, top2 Accuracy = {Accuracy_top2}, top3 Accuracy = {Accuracy_top3}"
			.format(iter = i, Accuracy_top1 = accuracy_top1, Accuracy_top2 = accuracy_top2, Accuracy_top3 = accuracy_top3))
		
		result_accuracy_top1 = result_accuracy_top1 + accuracy_top1
		result_accuracy_top2 = result_accuracy_top2 + accuracy_top2
		result_accuracy_top3 = result_accuracy_top3 + accuracy_top3
		
		## if you just want to see the result of top1 accuarcy, then using follwing codes
		#if i==0:
		#	result = np.argmax(Y_pre, -1)
		#else:
		#	result = np.concatenate([result, np.argmax(Y_pre, -1)], axis=0)
		
		## if you want to see the top2 result, then using the following codes
		#if i==0:
		#	result = np.multiply(~correct_prediction_top2, top1) + np.multiply(correct_prediction_top2, np.argmax(v_ys_part, -1))
		#else:
		#	result = np.concatenate([result, np.multiply(~correct_prediction_top2, top1) + np.multiply(correct_prediction_top2, np.argmax(v_ys_part, -1))], axis=0)
		
		# if you want to see the top3 result, then using the following codes
		if i==0:
			result = np.multiply(~correct_prediction_top3, top1) + np.multiply(correct_prediction_top3, np.argmax(v_ys_part, -1))
		else:
			result = np.concatenate([result, np.multiply(~correct_prediction_top3, top1) + np.multiply(correct_prediction_top3, np.argmax(v_ys_part, -1))], axis=0)
		
	result_accuracy_top1 = result_accuracy_top1 / batch_num
	result_accuracy_top2 = result_accuracy_top2 / batch_num
	result_accuracy_top3 = result_accuracy_top3 / batch_num
	
	return result, result_accuracy_top1, result_accuracy_top2, result_accuracy_top3, SegNet_Y_pre
	
def load_pre_trained_weights(parameters, pre_trained_weight_file=None,  keys=None, sess=None):
	if pre_trained_weight_file != None and sess != None:
		weights = np.load(pre_trained_weight_file + '.npz')
		#if keys == None:
		#	keys = sorted(weights.keys())
		
		keys = weights.keys()
		
		for i, k in enumerate(keys):	
			pre_trained_weights_shape = []
			
			for j in range(np.shape(np.shape(weights[k]))[0]):
				pre_trained_weights_shape += [np.shape(weights[k])[j]]
			
			if pre_trained_weights_shape == parameters[k].get_shape().as_list():
				print ("[\033[1;32;40mO\033[0m]{KEY} {PRE}->{PARAMS}" .format(KEY=k, PRE=pre_trained_weights_shape, PARAMS=str(parameters[k].get_shape().as_list())))
			else:
				print ("[\033[1;31;40mX\033[0m]{KEY} {PRE}->{PARAMS}" .format(KEY=k, PRE=pre_trained_weights_shape, PARAMS=str(parameters[k].get_shape().as_list())))
			
			if pre_trained_weights_shape == parameters[k].get_shape().as_list():
				sess.run(parameters[k].assign(weights[k]))
		print ("") 

def save_pre_trained_weights(file_name, parameters, xs, batch_xs, sess):
	weights_to_be_saved = []
	keys = list(parameters.keys())
	for i, key in enumerate(keys):
		weights_to_be_saved += [sess.run(parameters[key], feed_dict={xs: batch_xs})]
		
	np.savez(file_name, **{keys[x]:weights_to_be_saved[x] for x in range(len(parameters))})

def assign_trained_mean_and_var(mean_collection, 
								var_collection, 
								trained_mean_collection, 
								trained_var_collection, 
								params,
								xs, 
								ys, 
								is_training, 
								is_testing, 
								batch_xs,
								sess):
	NUM = len(trained_mean_collection)
	for i in range(NUM):
		sess.run(trained_mean_collection[i].assign(sess.run(mean_collection[i], feed_dict={xs: batch_xs, is_training: False, is_testing: False})))
		sess.run(trained_var_collection[i].assign(sess.run(var_collection[i], feed_dict={xs: batch_xs, is_training: False, is_testing: False})))

def quantized_m_and_f(activation_collection, xs, is_training, is_testing, is_quantized_activation, batch_xs, sess):
	NUM = len(activation_collection)
	for i in range(NUM):
		activation = sess.run(activation_collection[i], feed_dict={xs: batch_xs, is_training: False, is_testing: True, is_quantized_activation: False})
		var = np.var(activation)
		mantissa = 16
		fraction = -int(np.log2((var*3)/pow(2, mantissa)))-1
		#fraction = 3
		#pdb.set_trace()
		if i==0:
			m = np.array([[mantissa]])
			f = np.array([[fraction]])
		else:
			m = np.concatenate([m, np.array([[mantissa]])], axis=0)
			f = np.concatenate([f, np.array([[fraction]])], axis=0)
	#pdb.set_trace()
	return m, f

def assign_quantized_m_and_f(mantissa_collection, fraction_collection, m, f, sess):
	NUM = len(mantissa_collection)
	for i in range(NUM):
		sess.run(mantissa_collection[i].assign(m[i]))
		sess.run(fraction_collection[i].assign(f[i]))

def assign_ternary_boundary(ternary_weights_bd_collection, 
							ternary_biases_bd_collection, 
							ternary_weights_bd,
							ternary_biases_bd,
							sess):
	NUM = len(ternary_weights_bd_collection)
	for i in range(NUM):
		sess.run(ternary_weights_bd_collection[i].assign(ternary_weights_bd[i]))
		sess.run( ternary_biases_bd_collection[i].assign(ternary_biases_bd [i]))

def tenarized_bd(weights_collection,
                 biases_collection,
				 #lower_bd # percentage. Ex 20=20%
				 #upper_bd # percentage. Ex 80=80%
				 weights_bd_ratio, # percentage. Ex 50=50%
				 biases_bd_ratio,    # percentage. Ex 50=50%
				 sess
				):
	NUM = len(weights_collection)

	for i in range(NUM):
		w     = np.absolute(sess.run(weights_collection[i]))
		b     = np.absolute(sess.run(biases_collection [i]))
		
		w_bd  = np.percentile(w, weights_bd_ratio) 
		b_bd  = np.percentile(b, biases_bd_ratio )


		if i==0:
			weights_bd = np.array([[-w_bd, w_bd]])
			biases_bd  = np.array([[-b_bd, b_bd]])
		else:
			weights_bd = np.concatenate([weights_bd, np.array([[-w_bd, w_bd]])], axis=0)
			biases_bd  = np.concatenate([biases_bd , np.array([[-b_bd, b_bd]])], axis=0)

	weights_table = [-1, 0, 1]
	biases_table  = [-1, 0, 1]

	return weights_bd, biases_bd, weights_table, biases_table

def quantizing_weight_and_biases(weights_collection, 
							     biases_collection,
							     weights_bd,
							     biases_bd,
							     weights_table,
							     biases_table,
							     sess):
	NUM = len(weights_collection)
	for i in range(NUM):
		w                 = sess.run(weights_collection[i])
		b                 = sess.run(biases_collection   [i])

		w_bd              = weights_bd[i]
		b_bd              = biases_bd   [i]

		index_num         = len(w_bd)
		

		for j in range(index_num):
			if j==0:
				quantized_weights =(w <= w_bd[j]) * weights_table[j]
				quantized_biases    =(b <= b_bd[j]) * biases_table   [j]
			else:
				quantized_weights = quantized_weights + ((w <= w_bd[j]) & (w > w_bd[j-1])) * weights_table[j]
				quantized_biases    = quantized_biases    + ((b <= b_bd[j]) & (b > b_bd[j-1])) * biases_table   [j]
			if j==(index_num-1):
				quantized_weights = quantized_weights + (w > w_bd[j]) * weights_table[j+1]
				quantized_biases    = quantized_biases    + (b > b_bd[j]) * biases_table   [j+1]


		sess.run(weights_collection[i].assign(quantized_weights))
		sess.run( biases_collection[i].assign(quantized_biases ))

def quantize_activation(float32_activation):
	m = tf.get_variable("mantissa", dtype=tf.float32, initializer=tf.constant([7], tf.float32))
	f = tf.get_variable("fraction", dtype=tf.float32, initializer=tf.constant([0], tf.float32))
	tf.add_to_collection("activation", float32_activation)
	tf.add_to_collection("mantissa"  , m)
	tf.add_to_collection("fraction"  , f)

	upper_bd  =  tf.pow(tf.constant([2], tf.float32),  m)
	lower_bd  = -tf.pow(tf.constant([2], tf.float32),  m)
	step_size =  tf.pow(tf.constant([2], tf.float32), -f)
	
	step = tf.cast(tf.cast(tf.divide(float32_activation, step_size), tf.int32), tf.float32)
	quantized_activation = tf.multiply(step, step_size)
	quantized_activation = tf.maximum(lower_bd, tf.minimum(upper_bd, quantized_activation))
	
	return quantized_activation

def ternarize_weights(float32_weights, ternary_weights_bd):
	ternary_weights = tf.multiply(tf.cast(tf.less_equal(float32_weights, ternary_weights_bd[0]), tf.float32), tf.constant(-1, tf.float32))
	ternary_weights = tf.add(ternary_weights, tf.multiply(tf.cast(tf.greater(float32_weights, ternary_weights_bd[1]), tf.float32), tf.constant( 1, tf.float32)))
	
	return ternary_weights
	
def ternarize_biases(float32_biases, ternary_biases_bd):
	ternary_biases = tf.multiply(tf.cast(tf.less_equal(float32_biases, ternary_biases_bd[0]), tf.float32), tf.constant(-1, tf.float32))
	ternary_biases = tf.add(ternary_biases, tf.multiply(tf.cast(tf.greater(float32_biases, ternary_biases_bd[1]), tf.float32), tf.constant( 1, tf.float32)))

	return ternary_biases

def Analyzer(Analysis, net, type, kernel_shape=None, stride=0, group=1,
			is_depthwise=False,
			padding='SAME',
			name=None):
	if net !=None:
		[B, H, W, D] = net.get_shape().as_list() # B:Batch // H:Height // W:Width // D:Depth
		h = 0 
		w = 0 
		i = 0 
		o = 0

	if kernel_shape!=None:
		[h, w, i, o] = kernel_shape # h:Kernel Height // w:Kernel Width // i:input channel // o:output channel
	
	if is_depthwise:
		name = name + '_Depthwise'
		o = 1
	else: 
		group = 1

	if type=='CONV':
		if padding=='SAME':
			if is_depthwise:
				macc = H*W*h*w*D*group / (stride*stride)
				comp = 0
				add  = 0
				div  = 0
				exp  = 0
				activation = H*W*D / (stride*stride)
				param = h*w*i
			else:
				macc = H*W*h*w*i*o / (stride*stride)
				comp = 0
				add  = 0
				div  = 0
				exp  = 0
				activation = H*W*o / (stride*stride)
				param = h*w*i*o
		elif padding=='VALID':
			if is_depthwise:
				macc = (H-h+1)*(W-w+1)*h*w*D*group / (stride*stride)
				comp = 0
				add  = 0
				div  = 0
				exp  = 0
				activation = (H-h+1)*(W-w+1)*D / (stride*stride)
				param = h*w*i
			else:
				macc = (H-h+1)*(W-w+1)*h*w*i*o / (stride*stride)
				comp = 0
				add  = 0
				div  = 0
				exp  = 0
				activation = (H-h+1)*(W-w+1)*o / (stride*stride) 
				param = h*w*i*o
	elif type=='POOL':
		macc = 0
		comp = h*w*H*W*D / (stride*stride)
		add  = 0
		div  = 0
		exp  = 0
		activation = H*W*D / (stride*stride)
		param = 0
	elif type=='UNPOOL':
		macc = 0
		comp = 0
		add  = 0
		div  = 0
		activation = H*W*D*stride*stride
		param = 0
	elif type=='RELU':
		macc = 0
		comp = H*W*D
		add  = 0
		div  = 0
		exp  = 0
		activation = H*W*D
		param = 0
	elif type=='DATA':
		macc = 0
		comp = 0
		add  = 0
		div  = 0
		exp  = 0
		activation = H*W*D
		param = 0
	elif type=='ADD':
		macc = 0
		comp = 0
		add  = H*W*D
		div  = 0
		exp  = 0
		activation = H*W*D
		param = 0
	elif type=='TOTAL':
		name   		= 'Total' 
		H      		= 'None'
		W      		= 'None'
		D      		= 'None'
		h      		= 'None'
		w      		= 'None'
		i      		= 'None' 
		o      		= 'None'
		stride 		= 'None'
		activation  = 'None' 
		param	    = 'None'

		for i, key in enumerate(Analysis.keys()):
			if i==0:
				macc   	     = Analysis[key]['Macc'] 
				comp   	     = Analysis[key]['Comp'] 
				add    	     = Analysis[key]['Add'] 
				div    	     = Analysis[key]['Div'] 
				exp    	     = Analysis[key]['Exp'] 
				PE_row       = Analysis[key]['PE Height']
				PE_col       = Analysis[key]['PE Width']
				Macc_Cycle   = Analysis[key]['Macc Cycle'] 
				Input_Cycle  = Analysis[key]['Input Cycle']
				Kernel_Cycle = Analysis[key]['Kernel Cycle']
				Output_Cycle = Analysis[key]['Output Cycle']
				Bottleneck   = Analysis[key]['Bottleneck']
			else:
				macc   	     = macc         + Analysis[key]['Macc'] 
				comp   	     = comp         + Analysis[key]['Comp'] 
				add    	     = add          + Analysis[key]['Add'] 
				div    	     = div          + Analysis[key]['Div'] 
				exp    	     = exp          + Analysis[key]['Exp'] 
				Macc_Cycle   = Macc_Cycle   + Analysis[key]['Macc Cycle'] 
				Input_Cycle  = Input_Cycle  + Analysis[key]['Input Cycle']
				Kernel_Cycle = Kernel_Cycle + Analysis[key]['Kernel Cycle']
				Output_Cycle = Output_Cycle + Analysis[key]['Output Cycle']
				Bottleneck   = Bottleneck   + Analysis[key]['Bottleneck']
	
	# Estimate the Cycle Times which will be taken in hardware
	## Hardware Environment
	if type!='TOTAL':
		PE_row = 15
		PE_col = 16
		
		Tile_row = 8
		Tile_col = 8

		Data_Bits = 32

		Memory_Bandwidth = 128

		## Some Meaningful Variable
		# initialization
		Input_Retake_Times  = 1
		Kernel_Retake_Times = 1
		Output_Retake_Times = 1
		# conv
		if type=='CONV':
			if is_depthwise:
				Input_Retake_Times  = 1
				Kernel_Retake_Times = (H / Tile_row) * (W / Tile_col)
			else:
				Input_Retake_Times  = o / ((PE_row / h) * (PE_col / Tile_row))
				Kernel_Retake_Times = (H / Tile_row) * (W / Tile_col)
		
			if Input_Retake_Times<1:
				Input_Retake_Times = 1
			if Kernel_Retake_Times<1: 
				Kernel_Retake_Times = 1

		# Data Access In Cycle
		Data_Access_In_One_Cycle = Memory_Bandwidth / Data_Bits

		## Cycle times
		if type=='CONV':
			Macc_Cycle   = macc / (PE_row * PE_col)
			Input_Cycle  = (H * W * D) * (Input_Retake_Times) / (Data_Access_In_One_Cycle)		 
			Kernel_Cycle = (h * w * i * o * group) * (Kernel_Retake_Times) / (Data_Access_In_One_Cycle)		
			Output_Cycle = (activation) * (Output_Retake_Times) / (Data_Access_In_One_Cycle)		 
		else:
			Macc_Cycle   = comp + add +div + exp 
			Input_Cycle  = (H * W * D) * (Input_Retake_Times) / (Data_Access_In_One_Cycle)		 
			Kernel_Cycle = 0 
			Output_Cycle = (activation) * (Output_Retake_Times) / (Data_Access_In_One_Cycle)		 
		
		# Bottleneck
		Bottleneck = max(Macc_Cycle, Input_Cycle + Kernel_Cycle + Output_Cycle)
		

	components = {'name'         : name, 
				  'Input Height' : H,
				  'Input Width'  : W,
				  'Input Depth'  : D,
				  'Type'         : type,
				  'Kernel Height': h,
				  'Kernel Width' : w,
				  'Kernel Depth' : i, 
				  'Kernel Number': o,
				  'Stride'       : stride,
				  'Macc'         : macc, 
				  'Comp'         : comp, 
				  'Add'          : add, 
				  'Div'          : div, 
				  'Exp'          : exp, 
				  'Activation'   : activation, 
				  'Param'        : param,
				  'PE Height'	 : PE_row,
				  'PE Width'	 : PE_col,
				  'Macc Cycle'	 : Macc_Cycle,
				  'Input Cycle'	 : Input_Cycle,
				  'Kernel Cycle' : Kernel_Cycle,
				  'Output Cycle' : Output_Cycle,
				  'Bottleneck'   : Bottleneck}

	if len(Analysis.keys())<10:
		layer_now = '00' + str(len(Analysis.keys()))
	elif len(Analysis.keys())<100:
		layer_now = '0' + str(len(Analysis.keys()))
	else:
		layer_now = str(len(Analysis.keys()))

	Analysis.update({'layer' + layer_now : components})

	#pdb.set_trace()

	return Analysis

def Save_Analyzsis_as_csv(Analysis, FILE):
	Analyzer(Analysis, net=None, type='TOTAL')

	keys = sorted(Analysis.keys())
	
	components = np.array(['name'          ,
				           'Input Height'  ,
				           'Input Width'   ,
				           'Input Depth'   ,
				           'Type'          ,
				           'Kernel Height' ,
				           'Kernel Width'  ,
				           'Kernel Depth'  ,
				           'Kernel Number' ,
				           'Stride'        ,
				           'Macc'          ,
				           'Comp'          ,
				           'Add'           ,
				           'Div'           ,
				           'Exp'           ,
				           'Activation'    , 
				           'Param'         ,
				  		   'PE Height'	   ,
				  		   'PE Width'	   ,
				  		   'Macc Cycle'	   ,
				  		   'Input Cycle'   ,
				  		   'Kernel Cycle'  ,
				  		   'Output Cycle'  ,
				  		   'Bottleneck'    ])

	Ana = np.expand_dims(components, axis=1)
	for i, key in enumerate(keys):
		#if i==0:
		#	Ana = np.expand_dims(np.array([Analysis[key][x] for x in components]), axis=1)
		#else:	
		Ana = np.concatenate([Ana, np.expand_dims(np.array([Analysis[key][x] for x in components]), axis=1)], axis=1)

	np.savetxt(FILE + '_Analysis.csv', Ana, delimiter=",", fmt="%s") 

def shortcut_Module( net, kernel_size, stride, input_channel, internal_channel, output_channel, rate,
			         initializer             ,
			         is_constant_init       , 
			         is_batch_norm	        ,
			         is_training		    ,
			         is_testing		        ,
			         is_dilated		        ,
			         is_depthwise		    ,
			         is_ternary		        ,
			         is_quantized_activation,
			         IS_TERNARY			    , 	
			         IS_QUANTIZED_ACTIVATION,
			         IS_SAVER				,
			         padding				,			     
					 Analysis				):

	with tf.variable_scope("shortcut"):
		if input_channel!=output_channel:
			# Variable define
			weights, biases = conv2D_Variable(kernel_size       = 1,
											  input_channel	    = input_channel,		 
											  output_channel    = output_channel, 
											  initializer       = initializer,
											  is_constant_init  = is_constant_init,
											  is_ternary        = is_ternary,
											  IS_TERNARY		= IS_TERNARY,
											  is_depthwise		= False)
			# convolution
			#   Analyzer   #
			Analysis = Analyzer(Analysis, net, type='CONV', kernel_shape=[1,1,input_channel, output_channel], stride=1, is_depthwise=False, name='Shortcut')

			if is_dilated: 
				shortcut = tf.nn.atrous_conv2d(net, weights, rate=rate, padding=padding)
			else:
				shortcut = tf.nn.conv2d(net, weights, strides=[1, 1, 1, 1], padding=padding)

			# add bias
			shortcut = tf.nn.bias_add(shortcut, biases)

			# batch normalization
			if is_batch_norm == True:
				shortcut = batch_norm(shortcut, is_training, is_testing, IS_SAVER)
			if is_depthwise:
				shortcut = tf.nn.relu(shortcut)
		else:
			shortcut = net

	return shortcut, Analysis

def SEP_Module(net, kernel_size, stride, input_channel, internal_channel, output_channel, rate,
			   initializer             ,
			   is_constant_init        ,
			   is_batch_norm	       ,
			   is_training		       ,
			   is_testing		       ,
			   is_dilated		       ,
			   is_depthwise		       ,
			   is_ternary		       ,
			   is_quantized_activation ,
			   IS_TERNARY			   ,		
			   IS_QUANTIZED_ACTIVATION ,
			   IS_SAVER			   	   ,
			   padding			       ,
			   Analysis				   ):

	with tf.variable_scope("SEP_Module"):
		with tf.variable_scope("Reduction"):
			# Variable define
			weights, biases = conv2D_Variable(kernel_size       = 1,
										      input_channel	    = input_channel,
										      output_channel    = internal_channel,
										      initializer       = initializer,
										      is_constant_init  = is_constant_init,
										      is_ternary        = is_ternary,
											  IS_TERNARY		= IS_TERNARY,
											  is_depthwise		= False)

			# convolution
			#   Analyzer   #
			Analysis = Analyzer(Analysis, net, type='CONV', kernel_shape=[1,1,input_channel, internal_channel], stride=1, is_depthwise=False, name='SEP_Module/Reduction/Conv')

			if is_dilated:
				net = tf.nn.atrous_conv2d(net, weights, rate=rate, padding="SAME")
			else:
				net = tf.nn.conv2d(net, weights, strides=[1, 1, 1, 1], padding="SAME")
			
			# add bias
			net = tf.nn.bias_add(net, biases)
			
			# batch normalization
			if is_batch_norm == True:
				net = batch_norm(net, is_training, is_testing, IS_SAVER)
			
			# Relu
			#   Analyzer   #
			#Analysis = Analyzer(Analysis, net, type='RELU', name='SEP_Module/Reduction/Activation')
			net = tf.nn.relu(net)


		with tf.variable_scope("PatternConv1"):
			with tf.variable_scope("Pattern"):
				# Variable define
				weights, biases = conv2D_Variable(kernel_size      = kernel_size,
												  input_channel	   = internal_channel,		
												  output_channel   = internal_channel, 
												  initializer      = initializer,
												  is_constant_init = is_constant_init,
												  is_ternary       = is_ternary,
												  IS_TERNARY	   = IS_TERNARY,
												  is_depthwise     = is_depthwise)
												  
				# convolution
				#   Analyzer   #
				Analysis = Analyzer(Analysis, net, type='CONV', kernel_shape=[kernel_size,kernel_size,internal_channel, internal_channel], stride=stride, is_depthwise=is_depthwise, name='SEP_Module/PatternConv1/Pattern/Conv')

				if is_dilated:
					if is_depthwise:
						Pattern = depthwise_atrous_conv2d(net, weights, rate, padding=padding)	
					else:
						Pattern = tf.nn.atrous_conv2d(net, weights, rate=rate, padding=padding)
				else:
					if is_depthwise:
						Pattern = tf.nn.depthwise_conv2d(net, weights, strides=[1, stride, stride, 1], padding=padding)	
					else:
						Pattern = tf.nn.conv2d(net, weights, strides=[1, stride, stride, 1], padding=padding)
				
				# add bias
				Pattern = tf.nn.bias_add(Pattern, biases)

				# batch normalization
				if is_batch_norm == True:
					Pattern = batch_norm(Pattern, is_training, is_testing, IS_SAVER)
				#relu
				#   Analyzer   #
				#Analysis = Analyzer(Analysis, net, type='RELU', name='SEP_Module/PatternConv1/Pattern/Activation')
				
				Pattern = tf.nn.relu(Pattern)

				if IS_QUANTIZED_ACTIVATION:
					quantized_net = quantize_activation(Pattern)
					Pattern = tf.cond(is_quantized_activation, lambda: quantized_net, lambda: Pattern)
					tf.add_to_collection("final_net", Pattern)

				if is_depthwise:
					Pattern = conv2D(Pattern, kernel_size=1, stride=1, internal_channel=internal_channel, output_channel=internal_channel, rate=rate,
						   	         initializer=tf.contrib.layers.variance_scaling_initializer(),
						   	  	     is_constant_init        = False,
						   	  	     is_shortcut		     = False, 		
						   	 	     is_bottleneck	         = False, 		
									 is_residual			 = False,
									 is_SEP					 = False,
						   	 	     is_batch_norm	         = True,  		
						   	 	     is_dilated		         = False, 		
						   	 	     is_depthwise			 = False,		
						   	 	     is_training		     = is_training,  		
						   	 	     is_testing		         = is_testing, 
						   	 	     is_ternary		         = is_ternary,		
						   	 	     is_quantized_activation = is_quantized_activation,		
						   	 	     IS_TERNARY			     = IS_TERNARY,		
						   	 	     IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
						   	 	     IS_SAVER				 = IS_SAVER,
						   	 	     padding			     = padding,
									 Analysis				 = Analysis,
						   	 	     scope			         = "depthwise_conv1x1")

			with tf.variable_scope("Pattern_Residual"):
				# Variable define
				weights, biases = conv2D_Variable(kernel_size       = 1,
											      input_channel	    = internal_channel,		 
											      output_channel    = internal_channel, 
											      initializer       = initializer,
											      is_constant_init  = is_constant_init,
											      is_ternary        = is_ternary,
												  IS_TERNARY		= IS_TERNARY,
												  is_depthwise		= False)

				# convolution
				#   Analyzer   #
				Analysis = Analyzer(Analysis, net, type='CONV', kernel_shape=[1,1,internal_channel, internal_channel], stride=1, is_depthwise=False, name='SEP_Module/PatternConv1/Pattern_Residual/Conv')

				if is_dilated:
					Pattern_Residual = tf.nn.atrous_conv2d(net, weights, rate=rate, padding="SAME")
				else:
					Pattern_Residual = tf.nn.conv2d(net, weights, strides=[1, 1, 1, 1], padding="SAME")
				
				# add bias
				Pattern_Residual = tf.nn.bias_add(Pattern_Residual, biases)
				
				# batch normalization
				if is_batch_norm == True:
					Pattern_Residual = batch_norm(Pattern_Residual, is_training, is_testing, IS_SAVER)
					
				Pattern_Residual = tf.nn.relu(Pattern_Residual)
				
				# Relu
				#   Analyzer   #
				#Analysis = Analyzer(Analysis, net, type='RELU', name='SEP_Module/PatternConv1/Pattern_Residual/Activation')
				Pattern_Residual = tf.nn.relu(Pattern_Residual)

		# Adding Pattern and Pattern Residual
		net = tf.add(Pattern, Pattern_Residual)

		with tf.variable_scope("PatternConv2"):
			with tf.variable_scope("Pattern"):
				# Variable define
				weights, biases = conv2D_Variable(kernel_size      = kernel_size,
												  input_channel	   = internal_channel,		
												  output_channel   = internal_channel/2, 
												  initializer      = initializer,
												  is_constant_init = is_constant_init,
												  is_ternary       = is_ternary,
												  IS_TERNARY	   = IS_TERNARY,
												  is_depthwise     = is_depthwise)
												  
				# convolution
				#   Analyzer   #
				Analysis = Analyzer(Analysis, net, type='CONV', kernel_shape=[kernel_size,kernel_size,internal_channel, internal_channel/2], stride=1, is_depthwise=is_depthwise, name='SEP_Module/PatternConv2/Pattern/Conv')

				if is_dilated:
					if is_depthwise:
						Pattern = depthwise_atrous_conv2d(net, weights, rate, padding=padding)	
					else:
						Pattern = tf.nn.atrous_conv2d(net, weights, rate=rate, padding=padding)
				else:
					if is_depthwise:
						Pattern = tf.nn.depthwise_conv2d(net, weights, strides=[1, 1, 1, 1], padding=padding)	
					else:
						Pattern = tf.nn.conv2d(net, weights, strides=[1, 1, 1, 1], padding=padding)
				
				# add bias
				Pattern = tf.nn.bias_add(Pattern, biases)

				
				# batch normalization
				if is_batch_norm == True:
					Pattern = batch_norm(Pattern, is_training, is_testing, IS_SAVER)

				#relu
				#   Analyzer   #
				#Analysis = Analyzer(Analysis, net, type='RELU', name='SEP_Module/PatternConv2/Pattern/Activation')

				Pattern = tf.nn.relu(Pattern)
				
				if IS_QUANTIZED_ACTIVATION:
					quantized_net = quantize_activation(Pattern)
					Pattern = tf.cond(is_quantized_activation, lambda: quantized_net, lambda: Pattern)
					tf.add_to_collection("final_net", Pattern)

				if is_depthwise:
					Pattern =  conv2D(Pattern, kernel_size=1, stride=1, internal_channel=internal_channel, output_channel=internal_channel/2, rate=rate,
						   	      	  initializer=tf.contrib.layers.variance_scaling_initializer(),
						   	  	  	  is_constant_init        = False,
						   	  	  	  is_shortcut		      = False, 		
						   	 	  	  is_bottleneck	          = False, 		
									  is_residual			  = False,
									  is_SEP			      = False,
						   	 	  	  is_batch_norm	          = True,  		
						   	 	  	  is_dilated		      = False, 		
						   	 	  	  is_depthwise			  = False,		
						   	 	  	  is_training		      = is_training,  		
						   	 	  	  is_testing		      = is_testing, 
						   	 	  	  is_ternary		      = is_ternary,		
						   	 	  	  is_quantized_activation = is_quantized_activation,		
						   	 	  	  IS_TERNARY			  = IS_TERNARY,		
						   	 	  	  IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
						   	 	  	  IS_SAVER				  = IS_SAVER,
						   	 	  	  padding			      = padding,
									  Analysis				  = Analysis,
						   	 	  	  scope			          = "depthwise_conv1x1")

			with tf.variable_scope("Pattern_Residual"):
				# Variable define
				weights, biases = conv2D_Variable(kernel_size       = 1,
											      input_channel	    = internal_channel,		 
											      output_channel    = internal_channel/2, 
											      initializer       = initializer,
											      is_constant_init  = is_constant_init,
											      is_ternary        = is_ternary,
												  IS_TERNARY		= IS_TERNARY,
												  is_depthwise		= False)

				# convolution
				#   Analyzer   #
				Analysis = Analyzer(Analysis, net, type='CONV', kernel_shape=[1,1,internal_channel, internal_channel/2], stride=1, is_depthwise=False, name='SEP_Module/PatternConv2/Pattern_Residual/Conv')

				if is_dilated:
					Pattern_Residual = tf.nn.atrous_conv2d(net, weights, rate=rate, padding="SAME")
				else:
					Pattern_Residual = tf.nn.conv2d(net, weights, strides=[1, 1, 1, 1], padding="SAME")
				
				# add bias
				Pattern_Residual = tf.nn.bias_add(Pattern_Residual, biases)
				
				# batch normalization
				if is_batch_norm == True:
					Pattern_Residual = batch_norm(Pattern_Residual, is_training, is_testing, IS_SAVER)

				# Relu	
				#   Analyzer   #
				#Analysis = Analyzer(Analysis, net, type='RELU', name='SEP_Module/PatternConv2/Pattern_Residual/Activation')
				Pattern_Residual = tf.nn.relu(Pattern_Residual)

		# Adding Pattern and Pattern Residual
		net = tf.add(Pattern, Pattern_Residual)

		with tf.variable_scope("Recovery"):
			# Variable define
			weights, biases = conv2D_Variable(kernel_size       = 1,
										      input_channel	    = internal_channel/2,
										      output_channel    = output_channel,
										      initializer       = initializer,
										      is_constant_init  = is_constant_init,
										      is_ternary        = is_ternary,
											  IS_TERNARY		= IS_TERNARY,
											  is_depthwise		= False)

			# convolution
			#   Analyzer   #
			Analysis = Analyzer(Analysis, net, type='CONV', kernel_shape=[1,1,internal_channel/2, output_channel], stride=1, is_depthwise=False, name='SEP_Module/Recovery/Conv')

			if is_dilated:
				net = tf.nn.atrous_conv2d(net, weights, rate=rate, padding="SAME")
			else:
				net = tf.nn.conv2d(net, weights, strides=[1, 1, 1, 1], padding="SAME")
			
			# add bias
			net = tf.nn.bias_add(net, biases)
			
			# batch normalization
			if is_batch_norm == True:
				net = batch_norm(net, is_training, is_testing, IS_SAVER)
				
			# relu	
			#   Analyzer   #
			#Analysis = Analyzer(Analysis, net, type='RELU', name='SEP_Module/Recovery/Activation')
			net = tf.nn.relu(net)

	return net	


def Residual_Block( net, kernel_size, stride, input_channel, internal_channel, output_channel, rate,
					initializer             ,
					is_constant_init        , 
					is_bottleneck	        , 
					is_batch_norm	        , 
					is_training		        , 
					is_testing		        , 
					is_dilated		        , 
					is_depthwise		    , 
					is_ternary		        , 
					is_quantized_activation , 
					IS_TERNARY			    ,  	
					IS_QUANTIZED_ACTIVATION , 
					IS_SAVER			    , 
					padding			        ,
					Analysis				): 

	#===============================#
	#	Bottleneck Residual Block	#
	#===============================#
	if is_bottleneck: 
		with tf.variable_scope("bottle_neck"):
			with tf.variable_scope("conv1_1x1"):
				# Variable define
				weights, biases = conv2D_Variable(kernel_size       = 1,
											      input_channel	    = input_channel,		 
											      output_channel    = internal_channel, 
											      initializer       = initializer,
											      is_constant_init  = is_constant_init,
											      is_ternary        = is_ternary,
												  IS_TERNARY		= IS_TERNARY,
												  is_depthwise		= False)

				# convolution
				#   Analyzer   #
				Analysis = Analyzer(Analysis, net, type='CONV', kernel_shape=[1,1,input_channel, internal_channel], stride=1, is_depthwise=False, name='/bottle_neck/conv1_1x1/Conv')

				if is_dilated:
					net = tf.nn.atrous_conv2d(net, weights, rate=rate, padding="SAME")
				else:
					net = tf.nn.conv2d(net, weights, strides=[1, 1, 1, 1], padding="SAME")
				
				# add bias
				net = tf.nn.bias_add(net, biases)
				
				# batch normalization
				if is_batch_norm == True:
					net = batch_norm(net, is_training, is_testing, IS_SAVER)

				# Relu
				#   Analyzer   #
				#Analysis = Analyzer(Analysis, net, type='RELU', name='/bottle_neck/conv1_1x1/Activation')				
				net = tf.nn.relu(net)	

			with tf.variable_scope("conv2_3x3"):
				# Variable define
				weights, biases = conv2D_Variable(kernel_size       = 3,
											      input_channel	    = internal_channel,		 
											      output_channel    = internal_channel, 
											      initializer       = initializer,
											      is_constant_init  = is_constant_init,
											      is_ternary        = is_ternary,
												  IS_TERNARY		= IS_TERNARY,
												  is_depthwise		= is_depthwise)					

				# convolution
				#   Analyzer   #
				Analysis = Analyzer(Analysis, net, type='CONV', kernel_shape=[3,3,internal_channel, internal_channel], stride=stride, is_depthwise=is_depthwise, name='/bottle_neck/conv2_3x3/Conv')

				if is_dilated:
					if is_depthwise:
						net = depthwise_atrous_conv2d(net, weights, rate, padding=padding)	
					else:
						net = tf.nn.atrous_conv2d(net, weights, rate=rate, padding=padding)
				else:
					if is_depthwise:
						net = tf.nn.depthwise_conv2d(net, weights, strides=[1, stride, stride, 1], padding=padding)	
					else:
						net = tf.nn.conv2d(net, weights, strides=[1, stride, stride, 1], padding=padding)

				# add bias
				net = tf.nn.bias_add(net, biases)
				
				# batch normalization
				if is_batch_norm == True:
					net = batch_norm(net, is_training, is_testing, IS_SAVER)

				# relu
				#   Analyzer   #
				#Analysis = Analyzer(Analysis, net, type='RELU', name='/bottle_neck/conv2_3x3/Activation')
				net = tf.nn.relu(net)

			with tf.variable_scope("conv3_1x1"):
				# Variable define
				weights, biases = conv2D_Variable(kernel_size       = 1,
											      input_channel	    = internal_channel,		 
											      output_channel    = output_channel, 
											      initializer       = initializer,
											      is_constant_init  = is_constant_init,
											      is_ternary        = is_ternary,
												  IS_TERNARY		= IS_TERNARY,
												  is_depthwise		= False)

				# convolution
				#   Analyzer   #
				Analysis = Analyzer(Analysis, net, type='CONV', kernel_shape=[1,1,internal_channel, output_channel], stride=1, is_depthwise=False, name='/bottle_neck/conv3_1x1/Conv')

				if is_dilated:
					net = tf.nn.atrous_conv2d(net, weights, rate=rate, padding="SAME")
				else:
					net = tf.nn.conv2d(net, weights, strides=[1, 1, 1, 1], padding="SAME")
				
				# add bias
				net = tf.nn.bias_add(net, biases)
					
				# batch normalization
				if is_batch_norm == True:
					net = batch_norm(net, is_training, is_testing, IS_SAVER)
				
				# relu
				#   Analyzer   #
				#Analysis = Analyzer(Analysis, net, type='RELU', name='/bottle_neck/conv3_1x1/Activation')
				if is_depthwise:
					net = tf.nn.relu(net)


	#===========================#
	#	Normal Residual Block	#
	#===========================#
	else: 
		with tf.variable_scope("conv1_3x3"):
			# Variable define
			weights, biases = conv2D_Variable(kernel_size       = 3,
											  input_channel	    = input_channel,		 
											  output_channel    = internal_channel, 
											  initializer       = initializer,
											  is_constant_init  = is_constant_init,
											  is_ternary        = is_ternary,
											  IS_TERNARY		= IS_TERNARY,
											  is_depthwise		= is_depthwise)
											  
			# convolution
			#   Analyzer   #
			Analysis = Analyzer(Analysis, net, type='CONV', kernel_shape=[3,3,input_channel, internal_channel], stride=stride, is_depthwise=is_depthwise, name='/conv1_3x3/Conv')

			if is_dilated:
				if is_depthwise:
					net = depthwise_atrous_conv2d(net, weights, rate, padding=padding)	
				else:
					net = tf.nn.atrous_conv2d(net, weights, rate=rate, padding=padding)
			else:
				if is_depthwise:
					net = tf.nn.depthwise_conv2d(net, weights, strides=[1, stride, stride, 1], padding=padding)	
				else:
					net = tf.nn.conv2d(net, weights, strides=[1, stride, stride, 1], padding=padding)

			# add bias
			net = tf.nn.bias_add(net, biases)
			
			# batch normalization
			if is_batch_norm == True:
				net = batch_norm(net, is_training, is_testing, IS_SAVER)
			
			# Relu
			#   Analyzer   #
			#Analysis = Analyzer(Analysis, net, type='RELU', name='/conv1_3x3/Activation')
			net = tf.nn.relu(net)

			if is_depthwise:
				net =  conv2D(net, kernel_size=1, stride=1, internal_channel=input_channel, output_channel=internal_channel, rate=rate,
					   	      initializer=tf.contrib.layers.variance_scaling_initializer(),
					   	  	  is_constant_init        = False,
					   	  	  is_shortcut		      = False, 		
					   	 	  is_bottleneck	          = False, 		
							  is_residual			  = False,
							  is_SEP			      = False,
					   	 	  is_batch_norm	          = True,  		
					   	 	  is_dilated		      = False, 		
					   	 	  is_depthwise			  = False,		
					   	 	  is_training		      = is_training,  		
					   	 	  is_testing		      = is_testing, 
					   	 	  is_ternary		      = is_ternary,		
					   	 	  is_quantized_activation = is_quantized_activation,		
					   	 	  IS_TERNARY			  = IS_TERNARY,		
					   	 	  IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
					   	 	  IS_SAVER				  = IS_SAVER,
					   	 	  padding			      = padding,
							  Analysis				  = Analysis,
					   	 	  scope			          = "depthwise_conv1x1")

		with tf.variable_scope("conv2_3x3"):
			# Variable define
			weights, biases = conv2D_Variable(kernel_size       = 3,
											  input_channel	    = internal_channel,		 
											  output_channel    = output_channel, 
											  initializer       = initializer,
											  is_constant_init  = is_constant_init,
											  is_ternary        = is_ternary,
											  IS_TERNARY		= IS_TERNARY,
											  is_depthwise		= is_depthwise)
			
			# convolution
			#   Analyzer   #
			Analysis = Analyzer(Analysis, net, type='CONV', kernel_shape=[3,3,internal_channel, output_channel], stride=stride, is_depthwise=is_depthwise, name='/conv2_3x3/Conv')

			if is_dilated:
				if is_depthwise:
					net = depthwise_atrous_conv2d(net, weights, rate, padding=padding)	
				else:
					net = tf.nn.atrous_conv2d(net, weights, rate=rate, padding=padding)
			else:
				if is_depthwise:
					net = tf.nn.depthwise_conv2d(net, weights, strides=[1, 1, 1, 1], padding=padding)	
				else:
					net = tf.nn.conv2d(net, weights, strides=[1, 1, 1, 1], padding=padding)

			# add bias
			net = tf.nn.bias_add(net, biases)
			
			
			# batch normalization
			if is_batch_norm == True:
				net = batch_norm(net, is_training, is_testing, IS_SAVER)
				

			if is_depthwise:
				# relu
				#   Analyzer   #
				#Analysis = Analyzer(Analysis, net, type='RELU', name='/conv2_3x3/Conv')
				net = tf.nn.relu(net)
				
				net =  conv2D(net, kernel_size=1, stride=1, internal_channel=internal_channel, output_channel=output_channel, rate=rate,
					   	      initializer=tf.contrib.layers.variance_scaling_initializer(),
					   	  	  is_constant_init        = False,
					   	  	  is_shortcut		      = False, 		
					   	 	  is_bottleneck	          = False, 		
							  is_residual			  = False,
							  is_SEP			      = False,
					   	 	  is_batch_norm	          = True,  		
					   	 	  is_dilated		      = False, 		
					   	 	  is_depthwise			  = False,		
					   	 	  is_training		      = is_training,  		
					   	 	  is_testing		      = is_testing, 
					   	 	  is_ternary		      = is_ternary,		
					   	 	  is_quantized_activation = is_quantized_activation,		
					   	 	  IS_TERNARY			  = IS_TERNARY,		
					   	 	  IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
					   	 	  IS_SAVER				  = IS_SAVER,
					   	 	  padding			      = padding,
							  Analysis				  = Analysis,
					   	 	  scope			          = "depthwise_conv1x1")

				
	return net

def conv2D_Variable(kernel_size,
					input_channel,
					output_channel, 
					initializer,
					is_constant_init,
					is_ternary,
					IS_TERNARY,
					is_depthwise
					):
	# float32 Variable
	if is_constant_init:
		if is_depthwise:
			float32_weights = tf.get_variable("weights", dtype=tf.float32, initializer=initializer)
		else:
			float32_weights = tf.get_variable("weights", dtype=tf.float32, initializer=initializer)
	else:
		if is_depthwise:
			float32_weights = tf.get_variable("weights", [kernel_size, kernel_size, input_channel, 1			 ], tf.float32, initializer=initializer)
		else:
			float32_weights = tf.get_variable("weights", [kernel_size, kernel_size, input_channel, output_channel], tf.float32, initializer=initializer)
	
	if is_depthwise:
		float32_biases = tf.Variable(tf.constant(0.0, shape=[input_channel], dtype=tf.float32), trainable=True, name='biases')
	else:
		float32_biases = tf.Variable(tf.constant(0.0, shape=[output_channel], dtype=tf.float32), trainable=True, name='biases')
	tf.add_to_collection("weights", float32_weights)
	tf.add_to_collection("biases" , float32_biases)
	tf.add_to_collection("params" , float32_weights)
	tf.add_to_collection("params" , float32_biases)
	
	if IS_TERNARY:
		# Ternary Variable boundary
		ternary_weights_bd = tf.get_variable("ternary_weights_bd", [2], tf.float32, initializer=initializer)
		ternary_biases_bd  = tf.get_variable("ternary_biases_bd" , [2], tf.float32, initializer=initializer)
		tf.add_to_collection("ternary_weights_bd", ternary_weights_bd)
		tf.add_to_collection("ternary_biases_bd" , ternary_biases_bd)
		
		# Choose Precision
		final_weights = tf.cond(is_ternary, lambda: ternarize_weights(float32_weights, ternary_weights_bd), lambda: float32_weights)
		final_biases  = tf.cond(is_ternary, lambda: ternarize_biases (float32_biases , ternary_biases_bd) , lambda: float32_biases )
	
	
		if is_depthwise:
			weights = tf.get_variable("final_weights", [kernel_size, kernel_size, input_channel, input_channel], tf.float32, initializer=initializer)
			biases = tf.Variable(tf.constant(0.0, shape=[input_channel], dtype=tf.float32), trainable=True, name='final_biases')
		else:
			weights = tf.get_variable("final_weights", [kernel_size, kernel_size, input_channel, output_channel], tf.float32, initializer=initializer)
			biases = tf.Variable(tf.constant(0.0, shape=[output_channel], dtype=tf.float32), trainable=True, name='final_biases')
		tf.add_to_collection("final_weights", weights)
		tf.add_to_collection("final_biases", biases)
		tf.add_to_collection("var_list", weights)
		tf.add_to_collection("var_list", biases)
	
		assign_final_weights = tf.assign(weights, final_weights)
		assign_final_biases  = tf.assign(biases , final_biases )
		tf.add_to_collection("assign_var_list", assign_final_weights)
		tf.add_to_collection("assign_var_list", assign_final_biases)
		
		return weights, biases
	else:
		tf.add_to_collection("var_list", float32_weights)
		tf.add_to_collection("var_list", float32_biases)
		return float32_weights, float32_biases
	
def conv2D(	net, kernel_size=3, stride=1, internal_channel=64, output_channel=64, rate=1, group=1,
			initializer=tf.contrib.layers.variance_scaling_initializer(),
			is_constant_init        = False, 		# For using constant value as weight initial; Only valid in Normal Convolution
			is_shortcut		        = False, 		# For Residual, SEP
			is_bottleneck	        = False, 		# For Residual
			is_residual				= False,		# For Residual
			is_SEP					= False,		# For SEP-Net
			is_batch_norm	        = True,  		# For Batch Normalization
			is_dilated		        = False, 		# For Dilated Convoution
			is_depthwise			= False,		# For Depthwise Convolution
			is_ternary		        = False,		# (tensor) For weight ternarization
			is_training		        = True,  		# (tensor) For Batch Normalization
			is_testing		        = False, 		# (tensor) For getting the pretrained from caffemodel
			is_quantized_activation = False,		# (tensor) For activation quantization
			IS_TERNARY				= False,		
			IS_QUANTIZED_ACTIVATION = False,
			IS_SAVER				= True,
			padding			        = "SAME",
			Analysis				= None,
			scope			        = "conv"):
		
	with tf.variable_scope(scope):
		input_channel = net.get_shape().as_list()[-1]
			
		#====================#
		#   SEP-Net Module   #
		#====================#	
		if is_SEP:
			net_SEP = SEP_Module(net, kernel_size, stride, input_channel, internal_channel, output_channel, rate,
						         initializer             ,
						         is_constant_init        ,
						         is_batch_norm	         ,
						         is_training		     ,
						         is_testing		         ,
						         is_dilated		         ,
						         is_depthwise		     ,
						         is_ternary		         ,
						         is_quantized_activation ,
						         IS_TERNARY			     ,		
						         IS_QUANTIZED_ACTIVATION ,
						         IS_SAVER			   	 ,
						         padding			     ,
								 Analysis				 )
		
		#===============================#
		#	Bottleneck Residual Block	#
		#===============================#
		if is_residual:
			net_Res = Residual_Block( net, kernel_size, stride, input_channel, internal_channel, output_channel, rate,
							          initializer             ,
							          is_constant_init        , 
							          is_bottleneck	          , 
							          is_batch_norm	          , 
							          is_training		      , 
							          is_testing		      , 
							          is_dilated		      , 
							          is_depthwise		      , 
							          is_ternary		      , 
							          is_quantized_activation , 
							          IS_TERNARY			  ,  	
							          IS_QUANTIZED_ACTIVATION , 
							          IS_SAVER			      , 
							          padding			      ,
									  Analysis				  ) 
		#===================#
		#	Shortcut Block	#
		#===================#
		if is_shortcut:
			shortcut, Analysis =  shortcut_Module( net, kernel_size, stride, input_channel, internal_channel, output_channel, rate,
						  		         		   initializer             ,
						  		         		   is_constant_init        , 
						  		         		   is_batch_norm	       ,
						  		         		   is_training		       ,
						  		         		   is_testing		       ,
						  		         		   is_dilated		       ,
						  		         		   is_depthwise			   ,
						  		         		   is_ternary		       ,
						  		         		   is_quantized_activation ,
						  		         		   IS_TERNARY			   , 	
						  		         		   IS_QUANTIZED_ACTIVATION ,
						  		         		   IS_SAVER				   ,
						  		         		   padding			       ,
										 		   Analysis				   )
			if is_SEP:
				net = net_SEP
			else:
				if is_residual:
					net = net_Res

			# adding shortcut
			#   Analyzer   #
			Analysis = Analyzer(Analysis, net, type='ADD' , name='/shortcut/ADD')
			#Analysis = Analyzer(Analysis, net, type='RELU', name='/shortcut/Activation')
			if is_depthwise:
				net = tf.add(net, shortcut)
			else:
				net = tf.nn.relu(tf.add(net, shortcut))
			
			

		#===========================================#
		#	Normal Convolution Block (No Shortcut)  #
		#===========================================#
		else:  
			if not is_depthwise:
				group=1

			#   Analyzer   #
			Analysin = Analyzer(Analysis, net, type='CONV', kernel_shape=[kernel_size,kernel_size, input_channel, output_channel], stride=stride, group=group, is_depthwise=is_depthwise, padding=padding, name='Conv')

			for g in range(group):
				with tf.variable_scope('group%d'%(g)):
					# Variable define
					weights, biases = conv2D_Variable(kernel_size      = kernel_size,
													  input_channel	   = input_channel,		
													  output_channel   = output_channel, 
													  initializer      = initializer,
													  is_constant_init = is_constant_init,
													  is_ternary       = is_ternary,
													  IS_TERNARY	   = IS_TERNARY,
													  is_depthwise     = is_depthwise)
													  
					# convolution
					if is_dilated:
						if is_depthwise:
							net_tmp = depthwise_atrous_conv2d(net, weights, rate, padding=padding)	
						else:
							net_tmp = tf.nn.atrous_conv2d(net, weights, rate=rate, padding=padding)
					else:
						if is_depthwise:
							net_tmp = tf.nn.depthwise_conv2d(net, weights, strides=[1, stride, stride, 1], padding=padding)	
						else:
							net_tmp = tf.nn.conv2d(net, weights, strides=[1, stride, stride, 1], padding=padding)
					
					# add bias
					net_tmp = tf.nn.bias_add(net_tmp, biases)

					# merge every group net together
					if g==0:
						net = net_tmp
					else:
						#net = tf.concat([net, net_tmp], axis=3)
						net = tf.add(net, net_tmp)

			# batch normalization
			if is_batch_norm == True:
				net = batch_norm(net, is_training, is_testing, IS_SAVER)
			# relu
			#   Analyzer   #
			#Analysin = Analyzer(Analysis, net, type='RELU', name='Activation')
			net = tf.nn.relu(net)
			
			

			if IS_QUANTIZED_ACTIVATION:
				quantized_net = quantize_activation(net)
				net = tf.cond(is_quantized_activation, lambda: quantized_net, lambda: net)
				tf.add_to_collection("final_net", net)

			if is_depthwise:
				net =  conv2D(net, kernel_size=1, stride=1, internal_channel=input_channel, output_channel=output_channel, rate=rate,
					   	      initializer=tf.contrib.layers.variance_scaling_initializer(),
					   	  	  is_constant_init        = False,
					   	  	  is_shortcut		      = False, 		
					   	 	  is_bottleneck	          = False, 		
							  is_residual			  = False,
							  is_SEP				  = False,
					   	 	  is_batch_norm	          = True,  		
					   	 	  is_dilated		      = False, 		
					   	 	  is_depthwise			  = False,		
					   	 	  is_training		      = is_training,  		
					   	 	  is_testing		      = is_testing, 
					   	 	  is_ternary		      = is_ternary,		
					   	 	  is_quantized_activation = is_quantized_activation,		
					   	 	  IS_TERNARY			  = IS_TERNARY,		
					   	 	  IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
					   	 	  IS_SAVER				  = IS_SAVER,
					   	 	  padding			      = padding,
							  Analysis				  = Analysis,
					   	 	  scope			          = "depthwise_conv1x1")


	return net


def depthwise_atrous_conv2d(net, weights, rate, padding):
	[H, W, D, _] = weights.get_shape().as_list()
	for i in range(D):
		x = tf.expand_dims(net    [:, :, :, i], 3)
		w = tf.expand_dims(weights[:, :, :, i], 3)
		if i==0:
			out = tf.nn.atrous_conv1d(x, w, rate=rate, padding=padding)
		else:
			out = tf.concat([out, tf.nn.atrous_conv2d(x, w, rate=rate, padding=padding)], axis=3)
	return out

def deconv2D(net, 
			kernel_size, 
			stride, 
			output_channel, 
			initializer=tf.contrib.layers.variance_scaling_initializer(), 
			scope="deconv"):
			
	with tf.variable_scope(scope):
		[batch_size, height, width, input_channel] = net.get_shape().as_list()
		
		output_shape = [BATCH_SIZE, height*stride, width*stride, output_channel]
		
		weights = tf.get_variable("weights", [kernel_size, kernel_size, output_channel, input_channel], tf.float32, initializer=initializer)
		net = tf.nn.conv2d_transpose(net, weights, output_shape, strides=[1, stride, stride, 1], padding="SAME", name=None)
	return net

def indice_pool(net, stride, Analysis, scope="Pool"):
	with tf.variable_scope(scope):
		output_shape = net.get_shape().as_list()
		################
		#   Analyzer   #
		################
		Analysis = Analyzer(Analysis, net, type='POOL', kernel_shape=[stride, stride, output_shape[3], output_shape[3]], stride=stride, name='Pool')
		
		net, indices = tf.nn.max_pool_with_argmax( 
			input=net, 
			ksize=[1, stride, stride, 1],
			strides=[1, stride, stride, 1],
			padding="SAME",
			Targmax=None,
			name=None
		)

	return net, indices, output_shape, Analysis
	
def indice_unpool(net, stride, output_shape, indices, scope="unPool"):
	with tf.variable_scope(scope):
		input_shape = net.get_shape().as_list()
		
		# calculation indices for batch, height, width and channel
		one_like_mask = tf.ones_like(indices)
		batch_range = tf.reshape(tf.range(output_shape[0], dtype=tf.int64), shape=[output_shape[0], 1, 1, 1])
		b = one_like_mask * batch_range
		h = indices // (output_shape[2] * output_shape[3])
		w = (indices-h*output_shape[2]*output_shape[3]) // output_shape[3]
		c = indices-(h*output_shape[2]+w)*output_shape[3]
		
		# transpose indices & reshape update values to one dimension
		updates_size = tf.size(net)
		indices = tf.transpose(tf.reshape(tf.stack([b, h, w, c]), [4, updates_size]))
		values = tf.reshape(net, [updates_size])
		net = tf.scatter_nd(indices, values, output_shape)
		return net
				
def batch_norm(net, is_training, is_testing, IS_SAVER):
	with tf.variable_scope("Batch_Norm"):
		if IS_SAVER:
			return tf.contrib.layers.batch_norm(
			   			 inputs 				= net, 
			   			 decay					= 0.95,
			   			 center					= True,
			   			 scale					= False,
			   			 epsilon				= 0.001,
			   			 activation_fn			= None,
			   			 param_initializers		= None,
			   			 param_regularizers		= None,
			   			 updates_collections	= tf.GraphKeys.UPDATE_OPS,
			   			 is_training			= True,
			   			 reuse					= None,
			   			 variables_collections	= "params",
			   			 outputs_collections	= None,
			   			 trainable				= True,
			   			 batch_weights			= None,
			   			 fused					= False,
			   			 #data_format			= DATA_FORMAT_NHWC,
			   			 zero_debias_moving_mean= False,
			   			 scope					= None,
			   			 renorm					= False,
			   			 renorm_clipping		= None,
			   			 renorm_decay			= 0.99)
		else:
			batch_mean, batch_var = tf.nn.moments(net, axes=[0, 1, 2])
			output_channel = net.get_shape().as_list()

			trained_mean = tf.Variable(tf.zeros([output_channel]))
			trained_var = tf.Variable(tf.zeros([output_channel]))
			
			ema = tf.train.ExponentialMovingAverage(decay=0.95, zero_debias=False)
			
			def mean_var_with_update():
				ema_apply_op = ema.apply([batch_mean, batch_var])
				with tf.control_dependencies([ema_apply_op]):
					return tf.identity(batch_mean), tf.identity(batch_var)
				
			mean, var = tf.cond(is_testing,
							lambda: (
								trained_mean, 
								trained_var
									),
							lambda: (
								tf.cond(is_training,    
									mean_var_with_update,
									lambda:(
										ema.average(batch_mean), 
										ema.average(batch_var)
											)    
										)
									)
								)
			
			scale = tf.Variable(tf.ones([output_channel]))
			shift = tf.Variable(tf.ones([output_channel])*0.01)
					
			tf.add_to_collection("batch_mean", mean)
			tf.add_to_collection("batch_var", var)
			tf.add_to_collection("batch_scale", scale)
			tf.add_to_collection("batch_shift", shift)
			tf.add_to_collection("trained_mean", trained_mean)
			tf.add_to_collection("trained_var", trained_var)
			tf.add_to_collection("params", trained_mean)
			tf.add_to_collection("params", trained_var)
			tf.add_to_collection("params", scale)
			tf.add_to_collection("params", shift)
	
			
			epsilon = 0.001
		return tf.nn.batch_normalization(net, mean, var, shift, scale, epsilon)
	

def softmax_with_weighted_cross_entropy(prediction, ys, class_weight):
	[NUM, HEIGHT, WIDTH, DIM] = ys.get_shape().as_list()
	weighted_ys_pos = []
	weighted_ys_neg = []
	loss = tf.nn.softmax(prediction, dim = -1)
	for i in range(DIM):
		weighted_ys_pos.append(ys[:,:,:,i] * class_weight[i])
		weighted_ys_neg.append(tf.subtract(tf.ones_like(ys[:,:,:,i]), ys[:,:,:,i]) * class_weight[i])
	weighted_ys_pos = tf.stack(weighted_ys_pos, axis=3)
	weighted_ys_neg = tf.stack(weighted_ys_neg, axis=3)
	loss = tf.add(tf.multiply(weighted_ys_pos, tf.log(loss)), tf.multiply(weighted_ys_neg, tf.log(tf.subtract(tf.ones_like(loss), loss))))
	loss = -tf.reduce_sum(loss, axis=3) 
	return loss
	

#=====================#
#	Post Processing	  #
#=====================#
def get_max_K_class(data, target, K):
	data_shape = np.shape(data)
	
	#top = np.argsort(-data, axis=-1)[:, :, :, 0:K]
	#top = np.sort(top, axis=3)
	for i in range(K):
		if i==0:
			top = np.expand_dims(np.argsort(-data, axis=-1)[:, :, :, 0], 3)
		else:
			top = np.concatenate([top, np.expand_dims(np.argsort(-data, axis=-1)[:, :, :, 0], 3)], axis=-1)

	bb = range(data_shape[1])
	hh = range(data_shape[0])
	ww = range(data_shape[2])
	
	mesh = np.meshgrid(bb, hh, ww)
	
	BATCH  = np.tile(np.expand_dims(mesh[1], axis=3), (1,1,1,K))
	HEIGHT = np.tile(np.expand_dims(mesh[0], axis=3), (1,1,1,K))
	WIDTH  = np.tile(np.expand_dims(mesh[2], axis=3), (1,1,1,K))
	
	top_K_data   = data[BATCH, HEIGHT, WIDTH, top]
	top_K_target = target[BATCH, HEIGHT, WIDTH, top]
	table 		 = top

	for i in range(K):
		if i==0:
			top_K_target_K = top_K_target[:, :, :, i]
		else:
			top_K_target_K = np.logical_or(top_K_target_K, top_K_target[:, :, :, i])
	top_K_target_K = ~top_K_target_K
	top_K_target_K = np.expand_dims(top_K_target_K, 3)
	top_K_target = np.concatenate([top_K_target, top_K_target_K], axis=3)

	table_K = np.random.randint(np.shape(target)[3], size=(data_shape[0], data_shape[1], data_shape[2]))
	table_K = np.expand_dims(table_K, 3)
	table = np.concatenate([table, table_K], axis=3)

	return top_K_data, top_K_target, table


def get_real_prediction(Prediction, batch_table):
	data_shape = np.shape(Prediction)

	top = np.argsort(-Prediction, axis=-1)[:, :, :, 0] 

	bb = range(data_shape[1])
	hh = range(data_shape[0])
	ww = range(data_shape[2])
	
	mesh = np.meshgrid(bb, hh, ww)

	BATCH  = mesh[1]
	HEIGHT = mesh[0]
	WIDTH  = mesh[2]
	
	y_pre = batch_table[BATCH, HEIGHT, WIDTH, top]

	return y_pre


#def Save_Y_pre(Path, Y_pre, target):
#	np.savez(Path , Y_pre)
#	np.savez(Path + '_annot', target)

#def load_Y_pre(Path):
#	y_pre_file = np.load(Path + '.npz')
#	target_file = np.load(Path + '_annot.npz')
#	
#	print("data ...")
#	Y_pre  = y_pre_file[y_pre_file.keys()[0]]
#
#	print("target ...")
#	target = target_file[target_file.keys()[0]]
#
#	return Y_pre, target

def read_Y_pre_file(data_index, Path, layer=None):
	for i, file_name in enumerate(data_index):
		file_name = file_name.split('.')[0]
		if layer==None:
			y_pre = np.load(Path + file_name + '.npz')
		else:
			y_pre = np.load(Path + file_name + '_' + str(layer) + '.npz')
		Y_pre = y_pre[y_pre.keys()[0]]

		if i==0:
			data = np.expand_dims(Y_pre, axis=0)
		else:
			data = np.concatenate([data, np.expand_dims(Y_pre, axis=0)], axis=0)
	return data, data_index

def compute_accuracy_TOP_K(xs, ys, is_training, is_testing, is_validation, prediction, v_xs, v_ys, table, BATCH_SIZE, sess):
	test_batch_size = BATCH_SIZE
	batch_num = len(v_xs) / test_batch_size
	result_accuracy_top1 = 0
	result_accuracy_top2 = 0
	result_accuracy_top3 = 0

	for i in range(batch_num):	
		v_xs_part  = v_xs [i*test_batch_size : (i+1)*test_batch_size]
		v_ys_part  = v_ys [i*test_batch_size : (i+1)*test_batch_size] # one-of-k
		table_part = table[i*test_batch_size : (i+1)*test_batch_size]
		Prediction = sess.run(prediction, feed_dict={xs: v_xs_part, is_training: False, is_testing: (not is_validation)})
				
		top1 = get_real_prediction(Prediction, table_part)
	#	top1 = np.argsort(-Y_pre, axis=-1)[:, :, :, 0] 
	#	top2 = np.argsort(-Y_pre, axis=-1)[:, :, :, 1] 
	#	top3 = np.argsort(-Y_pre, axis=-1)[:, :, :, 2] 
		
		correct_prediction_top1 = np.equal(top1, np.argmax(v_ys_part, -1))
	#	correct_prediction_top2 = np.equal(top2, np.argmax(v_ys_part, -1)) | correct_prediction_top1
	#	correct_prediction_top3 = np.equal(top3, np.argmax(v_ys_part, -1)) | correct_prediction_top2
		
		accuracy_top1 = np.mean(correct_prediction_top1.astype(float))
	#	accuracy_top2 = np.mean(correct_prediction_top2.astype(float))
	#	accuracy_top3 = np.mean(correct_prediction_top3.astype(float))
		
	#print("Part : {iter}, top1 Accuracy = {Accuracy_top1}, top2 Accuracy = {Accuracy_top2}, top3 Accuracy = {Accuracy_top3}"
	#	.format(iter = i, Accuracy_top1 = accuracy_top1, Accuracy_top2 = accuracy_top2, Accuracy_top3 = accuracy_top3))
		
		print("Part : {iter}, top1 Accuracy = {Accuracy_top1}" .format(iter = i, Accuracy_top1 = accuracy_top1))

		result_accuracy_top1 = result_accuracy_top1 + accuracy_top1
	#	result_accuracy_top2 = result_accuracy_top2 + accuracy_top2
	#	result_accuracy_top3 = result_accuracy_top3 + accuracy_top3
		
		# if you just want to see the result of top1 accuarcy, then using follwing codes
		if i==0:
			result = top1
		else:
			result = np.concatenate([result, top1], axis=0)
		
		## if you want to see the top2 result, then using the following codes
		#if i==0:
		#	result = np.multiply(~correct_prediction_top2, top1) + np.multiply(correct_prediction_top2, np.argmax(v_ys_part, -1))
		#else:
		#	result = np.concatenate([result, np.multiply(~correct_prediction_top2, top1) + np.multiply(correct_prediction_top2, np.argmax(v_ys_part, -1))], axis=0)
		
		## if you want to see the top3 result, then using the following codes
		#if i==0:
		#	result = np.multiply(~correct_prediction_top3, top1) + np.multiply(correct_prediction_top3, np.argmax(v_ys_part, -1))
		#else:
		#	result = np.concatenate([result, np.multiply(~correct_prediction_top3, top1) + np.multiply(correct_prediction_top3, np.argmax(v_ys_part, -1))], axis=0)
		
	result_accuracy_top1 = result_accuracy_top1 / batch_num
	#result_accuracy_top2 = result_accuracy_top2 / batch_num
	#result_accuracy_top3 = result_accuracy_top3 / batch_num
	
	#pdb.set_trace() 
	
	return result, result_accuracy_top1
	

def per_class_accuracy_TOP_K(prediction, batch_ys, table):
	print("  Per Class Accuracy	: ")
	[BATCH, HEIGHT, WIDTH, CLASS_NUM] = np.shape(batch_ys)
	correct_num = np.zeros([CLASS_NUM, 1])
	total_num = np.zeros([CLASS_NUM, 1])
	
	for i in range(CLASS_NUM):
		top1 = get_real_prediction(prediction, table)
		
		y_tmp = np.equal(np.argmax(batch_ys, -1), i)
		p_tmp = np.equal(top1, i)
		total_num = np.count_nonzero(y_tmp)
		zeros_num = np.count_nonzero( (p_tmp+y_tmp) == 0)
		correct_num = np.count_nonzero(np.equal(y_tmp, p_tmp)) - zeros_num
		if total_num == 0:
			accuracy = -1
		else:
			accuracy = float(correct_num) / float(total_num)
		
		print("    Class{Iter}	: {predict} / {target}".format(Iter = i, predict=correct_num, target=total_num))


def shuffle_image_TOP_K(w, x, y, z):
	[NUM, HEIGHT, WIDTH, DIM] = np.shape(w)
	
	shuffle_index = np.arange(NUM)
	np.random.shuffle(shuffle_index)
	
	#index_shuffle = index[shuffle_index]
	w_shuffle = w[shuffle_index, :, :, :]
	x_shuffle = x[shuffle_index, :, :, :]
	y_shuffle = y[shuffle_index, :, :, :]
	z_shuffle = z[shuffle_index, :, :, :]


	return w_shuffle, x_shuffle, y_shuffle, z_shuffle
