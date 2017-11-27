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
# Hperparameter Optimization
import argparse
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
# main
import Model
import sys
import os
import time
import csv

#=========#
#   Run   #
#=========#
def run(
	Hyperparameter			,
	# Info                  
	Dataset 				,
	Model_first_name 		,
	Model_second_name 		,
	IS_HYPERPARAMETER_OPT 	,
	IS_TRAINING 			,
	IS_TESTING 				,
	EPOCH_TIME              ,
	# Path                  
	Dataset_Path			,
	Y_pre_Path				,
	train_target_path		,
	valid_target_path		,
	test_target_path		,
	train_Y_pre_path		,
	valid_Y_pre_path		,
	test_Y_pre_path			,
	TESTING_WEIGHT_PATH     = None,
	TESTINGN_WEIGHT_MODEL   = None
	):
	
	Model_Name = Model_first_name + '_' + Model_second_name
	
	#===============#
	#   Data Info   #
	#===============#
	if Dataset=='CamVid':   # original : [360, 480, 12]
		class_num = 12
	elif Dataset=='ade20k': # original : All Different
		class_num = 151
	elif Dataset=='mnist':  # original : [28, 28, 10]
		class_num = 10
	
	#================#
	#   Model_dict   #
	#================#
	Model_dict = Model_dict_Generator('Model/'+Model_first_name+'_Model/'+Model_Name+'.csv', class_num)
	
	#====================#
	#   Hyperparameter   #
	#====================#
	#-----------------------------------#
	#   Hyperparameter : User Defined   #
	#-----------------------------------#
	if not IS_HYPERPARAMETER_OPT:
		# Basic info
		BATCH_SIZE                  = 4
		#EPOCH_TIME                  = 200
		H_resize 					= 28 
		W_resize 					= 28
		# Learning Rate
		LEARNING_RATE 			    = 1e-3	 # Learning Rate      
		LR_DECADE     				= 10	 # Learning Rate Decade Magnification           
		LR_DECADE_1st_EPOCH			= 200	 # 1st Learning Rate Decade Epoch  
		LR_DECADE_2nd_EPOCH			= 200	 # 2nd Learning Rate Decade Epoch 
		LAMBDA        				= 0.1	 # L2-Regularization parameter 
		# Optimization 
		OptMethod                   = 'ADAM' # 'ADAM' or "MOMENTUM"
		Momentum_Rate               = 0.9
		# Teacher-Student Strategy
		IS_STUDENT  			    = False	 # (Coming Soon)
		# Weights Ternarized
		TERNARY_EPOCH              	= 2
		# Activation Quantized
		QUANTIZED_ACTIVATION_EPOCH 	= 3
		# Dropout
		DROPOUT_RATE                = 0.0
		
	#--------------------------------------------------#
	#   Hyperparameter : Hyperparameter Optimization   #
	#--------------------------------------------------#
	else:
		HP_dict, Model_dict = Hyperparameter_Decoder(Hyperparameter, Model_dict)
		# Basic info
		BATCH_SIZE                  = int(HP_dict['BATCH_SIZE'])
		#EPOCH_TIME                  = 200
		H_resize 					= 28 
		W_resize 					= 28
		# Learning Rate
		LEARNING_RATE 			    = float(HP_dict['LEARNING_RATE'])
		LR_DECADE     				= int(HP_dict['LR_DECADE'])           
		LR_DECADE_1st_EPOCH			= int(HP_dict['LR_DECADE_1st_EPOCH'])
		LR_DECADE_2nd_EPOCH			= int(HP_dict['LR_DECADE_2nd_EPOCH'])
		LAMBDA        				= float(HP_dict['Weight_Decay_Lambda'])
		# Optimization 
		OptMethod                   = HP_dict['OptMethod']
		Momentum_Rate               = float(HP_dict['Momentum_Rate'])
		# Teacher-Student Strategy
		IS_STUDENT  			    = HP_dict['IS_STUDENT']	== 'TRUE'
		# Weights Ternarized
		TERNARY_EPOCH              	= int(HP_dict['TERNARY_EPOCH'])
		# Activation Quantized
		QUANTIZED_ACTIVATION_EPOCH 	= int(HP_dict['QUANTIZED_ACTIVATION_EPOCH'])
		# Dropout
		DROPOUT_RATE                = float(HP_dict['Dropout_Rate'])

	#------------------------------#
	#   Hyperparameter : Testing   #
	#------------------------------#
	if ((not IS_TRAINING) and IS_TESTING):
		HP_test = {}
		with open(TESTING_WEIGHT_PATH + 'Hyperparameter.csv') as csvfile:
			HPreader = csv.reader(csvfile, delimiter=',', quotechar='|')
			for iter, row in enumerate(HPreader):
				HP_test.update({row[0]: row[1]})
			
				if   row[1]==' ADAM'    : HP_tmp = 'ADAM'
				elif row[1]==' MOMENTUM': HP_tmp = 'MOMENTUM'
				elif row[1]==' ReLU'    : HP_tmp = 'ReLU'
				elif row[1]==' True'    : HP_tmp = 'True'
				elif row[1]==' False'   : HP_tmp = 'False'
				else                    : HP_tmp = float(row[1])
					
				if iter==0:
					HP = np.array([HP_tmp])
				else:
					HP = np.concatenate([HP, [HP_tmp]])
					
		# Basic info
		BATCH_SIZE                  = 1
		EPOCH_TIME                  =   int(HP_test['EPOCH_TIME'                ])
		H_resize                    =   int(HP_test['H_resize'                  ])
		W_resize                    =   int(HP_test['W_resize'                  ])
		# Learning Rate               
		LEARNING_RATE               = float(HP_test['LEARNING_RATE'             ])
		LR_DECADE                   = float(HP_test['LR_DECADE'                 ])
		LR_DECADE_1st_EPOCH         = float(HP_test['LR_DECADE_1st_EPOCH'       ])
		LR_DECADE_2nd_EPOCH         = float(HP_test['LR_DECADE_2nd_EPOCH'       ])
		LAMBDA                      = float(HP_test['LAMBDA'                    ])
		# Optimization                                                          
		OptMethod                   =       HP_test['OptMethod'                 ]
		Momentum_Rate               = flaot(HP_test['Momentum_Rate'             ])
		# Teacher-Student Strategy                                              
		IS_STUDENT                  =       HP_test['IS_STUDENT'                ]
		# Weights Ternarized                                                    
		TERNARY_EPOCH               =   int(HP_test['TERNARY_EPOCH'             ])
		# Activation Quantized            
		QUANTIZED_ACTIVATION_EPOCH  =   int(HP_test['QUANTIZED_ACTIVATION_EPOCH'])
		# Dropout
		DROPOUT_RATE                = float(HP_test['DROPOUT_RATE'              ])
               
	print("\033[1;32;40mBATCH_SIZE\033[0m : \033[1;37;40m{BS}\033[0m" .format(BS=BATCH_SIZE))
	
	
	#===========================#
	#    Hyperparameter Save    #
	#===========================#
	if IS_TRAINING and (not IS_HYPERPARAMETER_OPT):
		components = np.array(['BATCH_SIZE'                ,
		                       'EPOCH_TIME'                ,
		                       'H_resize'                  ,
		                       'W_resize'                  ,
		                       'LEARNING_RATE'             ,
		                       'LR_DECADE'                 ,
		                       'LR_DECADE_1st_EPOCH'       ,
		                       'LR_DECADE_2nd_EPOCH'       ,
		                       'LAMBDA'                    ,
		                       'OptMethod'                 ,
		                       'Momentum_Rate'             ,
		                       'IS_STUDENT'                ,
		                       'TERNARY_EPOCH'             ,
		                       'QUANTIZED_ACTIVATION_EPOCH',
		                       'DROPOUT_RATE'              ])     
							   
		HP =                     np.array([BATCH_SIZE                ])
		HP = np.concatenate([HP, np.array([EPOCH_TIME                ])], axis=0)
		HP = np.concatenate([HP, np.array([H_resize                  ])], axis=0)
		HP = np.concatenate([HP, np.array([W_resize                  ])], axis=0)
		HP = np.concatenate([HP, np.array([LEARNING_RATE             ])], axis=0)
		HP = np.concatenate([HP, np.array([LR_DECADE                 ])], axis=0)
		HP = np.concatenate([HP, np.array([LR_DECADE_1st_EPOCH       ])], axis=0)
		HP = np.concatenate([HP, np.array([LR_DECADE_2nd_EPOCH       ])], axis=0)
		HP = np.concatenate([HP, np.array([LAMBDA                    ])], axis=0)
		HP = np.concatenate([HP, np.array([OptMethod                 ])], axis=0)
		HP = np.concatenate([HP, np.array([Momentum_Rate             ])], axis=0)
		HP = np.concatenate([HP, np.array([IS_STUDENT                ])], axis=0)
		HP = np.concatenate([HP, np.array([TERNARY_EPOCH             ])], axis=0)
		HP = np.concatenate([HP, np.array([QUANTIZED_ACTIVATION_EPOCH])], axis=0)
		HP = np.concatenate([HP, np.array([DROPOUT_RATE              ])], axis=0)
		
		components = np.expand_dims(components, axis=1)
		HP = np.expand_dims(HP, axis=1)
		HP = np.concatenate([HP, components], axis=1)
	else:
		HP = None
	
	#==================#
	#    Placeholder   #
	#==================#
	data_shape = [None, H_resize, W_resize, 3]
	xs = tf.placeholder(tf.float32, [BATCH_SIZE, data_shape[1], data_shape[2], data_shape[3]])
	if Dataset!='mnist':
		ys = tf.placeholder(tf.float32, [BATCH_SIZE, data_shape[1], data_shape[2], class_num])
	else:
		ys = tf.placeholder(tf.float32, [BATCH_SIZE, 1, 1, class_num])
	
	learning_rate           = tf.placeholder(tf.float32)
	is_training             = tf.placeholder(tf.bool)
	is_testing 	            = tf.placeholder(tf.bool)

	# Data Preprocessing
	xImage = xs
	
	#===========#
	#   Model   #
	#===========#
	net = xImage
	if Hyperparameter is None:
		prediction, Analysis, max_parameter = Model_dict_Decoder(net, Model_dict, is_training, is_testing, DROPOUT_RATE)
	else:
		HP_dict, Model_dict = Hyperparameter_Decoder(Hyperparameter, Model_dict)
		prediction, Analysis, max_parameter = Model_dict_Decoder(net, Model_dict, is_training, is_testing, DROPOUT_RATE)
		
	#================#
	#   Collection   #
	#================#	
	weights_collection	 		  	   = tf.get_collection("weights"                , scope=None)
	biases_collection  	    	  	   = tf.get_collection("biases"                 , scope=None)
	ternary_weights_bd_collection 	   = tf.get_collection("ternary_weights_bd"     , scope=None)
	ternary_biases_bd_collection  	   = tf.get_collection("ternary_biases_bd"      , scope=None)
	final_weights_collection      	   = tf.get_collection("final_weights"          , scope=None)
	final_biases_collection       	   = tf.get_collection("final_biases"           , scope=None)
	var_list_collection		      	   = tf.get_collection("var_list"               , scope=None)
	assign_var_list_collection    	   = tf.get_collection("assign_var_list"        , scope=None)
	activation_collection	      	   = tf.get_collection("activation"             , scope=None)
	is_quantized_activation_collection = tf.get_collection("is_quantized_activation", scope=None)
	mantissa_collection		      	   = tf.get_collection("mantissa"               , scope=None)
	fraction_collection           	   = tf.get_collection("fraction"               , scope=None)
	final_net_collection	      	   = tf.get_collection("final_net"              , scope=None)
	params 							   = tf.get_collection("params"                 , scope=None) 
	
	#=======================#
	#   Training Strategy   #
	#=======================#	
	# KL divergence
	KL = tf.nn.softmax_cross_entropy_with_logits(labels = ys, logits = prediction) 
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(ys, -1), logits=prediction)
	
	# L2 Regularization
	l2_norm   = tf.reduce_mean(tf.stack([tf.nn.l2_loss(x) for x in weights_collection]))
	l2_lambda = tf.constant(LAMBDA)
	l2_norm   = tf.multiply(l2_lambda, l2_norm)

	# Loss
	loss = KL
	# Optimizer
	if   OptMethod=='ADAM': 
		opt = tf.train.AdamOptimizer(learning_rate, Momentum_Rate)
	elif OptMethod=='MOMENTUM':
		opt = tf.train.MomentumOptimizer(learning_rate, Momentum_Rate)

	# Update Weights
	gradients   = opt.compute_gradients(loss, var_list=var_list_collection)
	gra_and_var = [(gradients[i][0], params[i]) for i in range(np.shape(gradients)[0])]
	train_step  = opt.apply_gradients(gra_and_var)

	#===========#
	#   Saver   #
	#===========#	
	saver = tf.train.Saver()

	#=================#
	#   Session Run   #
	#=================#
	with tf.Session() as sess:
		# Initialize
		init = tf.global_variables_initializer()
		sess.run(init)
		if IS_TRAINING == True:
			# File Read
			layer = None
			
			# Initialize
			init = tf.global_variables_initializer()
			sess.run(init)
			
			# Learning Rate	
			lr = LEARNING_RATE

			Training_and_Validation( 
				# Model
				Model_dict                         = Model_dict                         ,
				Analysis                           = Analysis                           ,
				# (File Read) Path                                                     
				Dataset	                           = Dataset                            ,
				Dataset_Path                       = Dataset_Path                       ,
				Y_pre_Path                         = Y_pre_Path                         ,
				# (File Read) Variable                                                 
				class_num                          = class_num                          ,
				layer                              = layer                              ,
				H_resize                           = H_resize                           ,
				W_resize                           = W_resize                           ,
				# (File Read) Variable                                                 
				IS_STUDENT                         = IS_STUDENT                         ,
				# Parameter					       		                               
				EPOCH_TIME					       = EPOCH_TIME						    ,
				BATCH_SIZE					       = BATCH_SIZE						    ,
				LR_DECADE					       = LR_DECADE						    ,
				LR_DECADE_1st_EPOCH			       = LR_DECADE_1st_EPOCH                ,
				LR_DECADE_2nd_EPOCH			       = LR_DECADE_2nd_EPOCH                ,
				lr							       = lr		            			    ,
				# Tensor					       		                               
				train_step					       = train_step						    ,
				loss						       = loss							    ,
				prediction					       = prediction						    ,
				# Placeholder				       			                           
				xs							       = xs								    , 
				ys							       = ys								    ,
				learning_rate				       = learning_rate					    ,
				is_training					       = is_training						,
				is_testing					       = is_testing						    ,
				# (For Saving Trained Weight) Collection 		
				params						       = params							    ,
				# (For Saving Trained Weight) File Path 
				TRAINED_WEIGHT_FILE			       = None                               ,
				TRAINING_WEIGHT_FILE		       = None                   		    ,
				# (For Saving Trained Weight) Trained Weight Parameters 		
				saver						       = saver							    ,
				HP                                 = HP                                 ,
				Model_first_name                   = Model_first_name                   ,
				Model_Name                         = Model_Name                         ,
				# (Ternary) Parameter			                                        
				TERNARY_EPOCH			           = TERNARY_EPOCH					    ,
				# (Ternary) Collection		                                            
				weights_collection		           = weights_collection				    ,
				biases_collection			       = biases_collection				    ,
				ternary_weights_bd_collection      = ternary_weights_bd_collection	    ,
				ternary_biases_bd_collection       = ternary_biases_bd_collection	    ,
				# (Assign final weights)                                                
				assign_var_list_collection	       = assign_var_list_collection		    ,
				# (Quantize actvation) parameter                                        
				QUANTIZED_ACTIVATION_EPOCH	       = QUANTIZED_ACTIVATION_EPOCH		    ,
				# (Quantize actvation) collection
				activation_collection		       = activation_collection			    , 
				mantissa_collection			       = mantissa_collection 			    ,
				fraction_collection     	       = fraction_collection				,
				is_quantized_activation_collection = is_quantized_activation_collection ,
				# (Hyperparameter Optimization)
				IS_HYPERPARAMETER_OPT              = IS_HYPERPARAMETER_OPT              ,
				# (Debug)                          
				final_weights_collection	       = final_weights_collection		    ,
				final_net_collection		       = final_net_collection			    ,
				# Session                                                               
				sess						       = sess							    )
	
		if IS_TESTING == True:
			train_accuracy, valid_accuracy, test_accuracy = Testing(
				# (File Read) Path
				Dataset	                    = Dataset                   ,
				Dataset_Path                = Dataset_Path              ,
				Y_pre_Path                  = Y_pre_Path                ,
				# (File Read) Variable                                  
				class_num                   = class_num                 ,
				layer                       = None                      ,
				H_resize                    = H_resize                  ,
				W_resize                    = W_resize                  ,
				# Parameter	
				Model_dict                  = Model_dict                ,
				BATCH_SIZE					= BATCH_SIZE				,
				IS_SAVING_RESULT_AS_IMAGE	= False						,	
				IS_SAVING_RESULT_AS_NPZ		= False						,
				IS_SAVING_PARTIAL_RESULT	= False						,
				IS_TRAINING					= IS_TRAINING				,
				IS_HYPERPARAMETER_OPT       = IS_HYPERPARAMETER_OPT     ,
				# Tensor	
				prediction					= prediction				,
				# Placeholder
				xs							= xs						, 
				ys							= ys						,
				is_training					= is_training				,
				is_testing					= is_testing				,
				# File Path (For Loading Trained Weight)
				TESTING_WEIGHT_PATH         = TESTING_WEIGHT_PATH       ,
				TESTINGN_WEIGHT_MODEL       = TESTINGN_WEIGHT_MODEL     , 
				# Trained Weight Parameters (For Loading Trained Weight)
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
		else:
			train_accuracy = 0 
			valid_accuracy = 0 
			test_accuracy  = 0
	
	tf.reset_default_graph()
	
	#pdb.set_trace()
	#print("")
	#print("Works are All Done !")
	
	return train_accuracy, valid_accuracy, test_accuracy, max_parameter
	
def Training_and_Validation( 
		# Model
		Model_dict              ,
		Analysis                ,
		
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
		LR_DECADE_1st_EPOCH		,
		LR_DECADE_2nd_EPOCH		,
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
		params							   = None,
		
		# (Saving Trained Weight) File Path
		TRAINED_WEIGHT_FILE				   = None,
		TRAINING_WEIGHT_FILE			   = None, 
		
		# (Saving Trained Weight) Trained Weight Parameters 
		saver							   = None,
		HP                                 = None,
		Model_first_name                   = None,
		Model_Name                         = None,
		                                   
		# (ternary) Parameter              
		TERNARY_EPOCH					   = None,
				
		# (ternary) Collection		
		weights_collection				   = None,
		biases_collection				   = None,
		ternary_weights_bd_collection 	   = None,
		ternary_biases_bd_collection	   = None,
		                                   
		# (ternary) variable               
		weights_bd_ratio				   = 50,
		biases_bd_ratio					   = 50,
		                                   
		# (assign final weights)           
		assign_var_list_collection 		   = None,
		                                   
		# (quantize actvation) parameter   
		QUANTIZED_ACTIVATION_EPOCH		   = None,

		# (quantize actvation) collection
		activation_collection		       = None, 
		mantissa_collection			       = None, 
		fraction_collection     	       = None, 
		is_quantized_activation_collection = None,
		# (Hyperparameter Optimization)
		IS_HYPERPARAMETER_OPT			   = None,
		                                   
		# (debug)                          
		final_weights_collection		   = None,
		final_net_collection			   = None,
                                           
		# Session                          
		sess							   = None):

	#-------------------------------#
	#   Loading Pre-trained Model   #
	#-------------------------------#
	if TRAINED_WEIGHT_FILE!=None:
		print ""
		print("Loading Pre-trained weights ...")
		save_path = saver.save(sess, TRAINED_WEIGHT_FILE + ".ckpt")
		print(save_path)

	#-----------------------------#
	#   Some Control Parameters   #
	#-----------------------------#
	IS_TERNARY = False
	IS_QUANTIZED_ACTIVATION = False
	for layer in range(len(Model_dict)):
		if Model_dict['layer'+str(layer)]['IS_TERNARY'] == 'TRUE':
			IS_TERNARY = True
		if Model_dict['layer'+str(layer)]['IS_QUANTIZED_ACTIVATION'] == 'TRUE':
			IS_QUANTIZED_ACTIVATION = True
			
	TERNARY_NOW = False
	QUANTIZED_NOW = False
	
	#---------------#
	#   File Read   #
	#---------------#
	train_data_index   = np.array(open(Dataset_Path + '/train.txt', 'r').read().splitlines())
	train_target_index = np.array(open(Dataset_Path + '/trainannot.txt', 'r').read().splitlines())
	train_data_num = len(train_data_index)
	print("\033[1;32;40mTraining Data Number\033[0m = {DS}" .format(DS = train_data_num))
	
	shuffle_index = np.arange(train_data_num)
	np.random.shuffle(shuffle_index)

	train_data_index   = train_data_index  [shuffle_index]
	train_target_index = train_target_index[shuffle_index]

	val_data_index   = open(Dataset_Path + '/val.txt', 'r').read().splitlines()
	val_target_index = open(Dataset_Path + '/valannot.txt', 'r').read().splitlines()
	val_data_num = len(val_data_index)
	print("\033[1;32;40mValidation Data Number\033[0m = {DS}" .format(DS = val_data_num))
	
	Data_Size_Per_Iter = BATCH_SIZE

	#---------------#
	#   Per Epoch   #
	#---------------#
	########################
	tStart_All = time.time()
	########################
	for epoch in range(EPOCH_TIME): 
		Train_acc = 0
		Train_loss = 0
		iteration = 0
		
		#--------------------------#
		#   Quantizad Activation   #
		#--------------------------#
		if IS_QUANTIZED_ACTIVATION and (epoch+1)==QUANTIZED_ACTIVATION_EPOCH:
			# Training file
			# data_index_part
			if train_data_num<Data_Size_Per_Iter:
				train_data_index_part   = train_data_index
				train_target_index_part = train_target_index
			else:
				train_data_index_part   = train_data_index[0:Data_Size_Per_Iter]
				train_target_index_part = train_target_index[0:Data_Size_Per_Iter]
			# data parse
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
			
			batch_xs = train_data[0 : BATCH_SIZE]
			
			# Calculate Each Activation's appropriate mantissa and fractional bit
			m, f = quantized_m_and_f(activation_collection, is_quantized_activation_collection, xs, is_training, is_testing, Model_dict, batch_xs, sess)	
			
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
		
		
		####################
		tStart = time.time()
		####################
		print("\nTraining ... ")
		for data_iter in range((train_data_num/Data_Size_Per_Iter)):
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

			#print("Loading Training Data ...")
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

			#------------------------#
			#   Data Preprocessing   #
			#------------------------#
			data_shape = np.shape(train_data) 
			train_data, train_target = shuffle_image(train_data, train_target)
			
			##**********************##
			##    Start Training    ##
			##**********************##
			for i in range(int(data_shape[0]/BATCH_SIZE)):
				iteration = iteration + 1
				# Train data in BATCH SIZE
				batch_xs    = train_data   [i*BATCH_SIZE : (i+1)*BATCH_SIZE]
				batch_ys    = train_target [i*BATCH_SIZE : (i+1)*BATCH_SIZE]
				
				#-----------------------#
				#   Run Training Step   #
				#-----------------------#	
				dict_inputs = [xs      , ys      , learning_rate, is_training, is_testing]
				dict_data   = [batch_xs, batch_ys, lr           , True       , False     ]
				for layer in range(len(Model_dict)):
					dict_inputs.append(Model_dict['layer'+str(layer)]['is_ternary'])
					dict_inputs.append(Model_dict['layer'+str(layer)]['is_quantized_activation'])
					dict_data.append(Model_dict['layer'+str(layer)]['IS_TERNARY'] and TERNARY_NOW)
					dict_data.append(Model_dict['layer'+str(layer)]['IS_QUANTIZED_ACTIVATION'] and QUANTIZED_NOW)
				
				_, Loss, Prediction = sess.run([train_step, loss, prediction], 
				                               feed_dict={i: d for i, d in zip(dict_inputs, dict_data)})
				
				dict_inputs = []
				dict_data   = []
				for layer in range(len(Model_dict)):
					dict_inputs.append(Model_dict['layer'+str(layer)]['is_ternary'])
					dict_data.append(Model_dict['layer'+str(layer)]['IS_TERNARY'] and TERNARY_NOW)
					
				for assign_var_list_iter, assign_var_list in enumerate(assign_var_list_collection):
					sess.run(assign_var_list, 
					         feed_dict={i: d for i, d in zip(dict_inputs, dict_data)})
				
				#------------#
				#   Result   #
				#------------#
				y_pre = np.argmax(Prediction, -1)
				batch_accuracy = np.mean(np.equal(np.argmax(batch_ys, -1), y_pre))
				
				Train_acc  = Train_acc  + batch_accuracy
				Train_loss = Train_loss + np.mean(Loss)
				
				# DEBUG
				if iteration==1:
					ACC = np.array([batch_accuracy])
				else:
					ACC = np.concatenate([ACC, np.array([batch_accuracy])])
					
				# Print Training Result Per Batch Size
				"""
				print("\033[1;34;40mEpoch\033[0m : {ep}" .format(ep = epoch))
				print("\033[1;34;40mData Iteration\033[0m : {Iter}" .format(Iter = i*BATCH_SIZE+data_iter*Data_Size_Per_Iter))
				print("\033[1;32;40m  Batch Accuracy\033[0m : {acc}".format(acc = batch_accuracy))
				print("\033[1;32;40m  Loss          \033[0m : {loss}".format(loss = np.mean(Loss)))
				print("\033[1;32;40m  Learning Rate \033[0m : {LearningRate}".format(LearningRate = lr))
				"""
				
				# Per Class Accuracy
				"""
				per_class_accuracy(Prediction, batch_ys)
				"""
			#----------------------------------------------------------------------#
			#    Show Difference Between Quanitzed and Non-Quantized Activation    #
			#----------------------------------------------------------------------#
			"""
			if QUANTIZED_NOW:
				batch_xs = train_data[0 : BATCH_SIZE]
				for i, final_net in enumerate(final_net_collection):
					TRUE   = sess.run(final_net, feed_dict={xs: batch_xs, is_quantized_activation: True })
					FALSE  = sess.run(final_net, feed_dict={xs: batch_xs, is_quantized_activation: False})
					print("{iter}: {TRUE_MEAN}	{FALSE_MEAN}	{DIFF_MEAN}" .format(iter=i, TRUE_MEAN=np.mean(TRUE), FALSE_MEAN=np.mean(FALSE), DIFF_MEAN=np.mean(np.absolute(TRUE-FALSE))))
			"""
		
		# Record Per Epoch Training Result (Finally this will save as the .csv file)
		Train_acc  = Train_acc  / iteration 
		Train_loss = Train_loss / iteration 
		
		print("\033[1;34;40mEpoch\033[0m : {ep}" .format(ep = epoch))
		print("\033[1;32;40m  Training Accuracy\033[0m : {acc}".format(acc = Train_acc))
		print("\033[1;32;40m  Training Accuracy\033[0m : {acc}".format(acc = np.mean(ACC)))
		print("\033[1;32;40m  Training Loss    \033[0m : {loss}".format(loss = Train_loss))
		
		if epoch==0:
			Train_acc_per_epoch  = np.array([Train_acc ])
			Train_loss_per_epoch = np.array([Train_loss])
		else:
			Train_acc_per_epoch  = np.concatenate([Train_acc_per_epoch , np.array([Train_acc ])], axis=0)
			Train_loss_per_epoch = np.concatenate([Train_loss_per_epoch, np.array([Train_loss])], axis=0)
		
		##################
		tEnd = time.time()
		##################
		print("It cost {TIME} sec\n" .format(TIME=tEnd - tStart))
		
		#-------------------------#
		#   Learning Rate Decay   #
		#-------------------------#
		# 1st
		if ((epoch+1)==LR_DECADE_1st_EPOCH):
			lr = lr / LR_DECADE
		# 2nd
		if ((epoch+1)==LR_DECADE_2nd_EPOCH):
			lr = lr / LR_DECADE
		
		##----------------##
		##   Validation   ##
		##----------------##
		if ((not IS_HYPERPARAMETER_OPT) and ((epoch+1)%10==160)):
			print("Validation ... ")
			print("\033[1;34;40mEpoch\033[0m : {ep}".format(ep = epoch))
			total_valid_accuracy = 0
			for iter in range(val_data_num/Data_Size_Per_Iter):
				# data_index_part
				if val_data_num<Data_Size_Per_Iter:
					val_data_index_part   = val_data_index
					val_target_index_part = val_target_index
				else:
					val_data_index_part   = val_data_index[iter*Data_Size_Per_Iter:(iter+1)*Data_Size_Per_Iter]
					val_target_index_part = val_target_index[iter*Data_Size_Per_Iter:(iter+1)*Data_Size_Per_Iter]
	
				#print("Loading Validation Data ...")
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
	
				#print("\033[1;34;40mEpoch\033[0m : {ep}".format(ep = epoch))
				#print("\033[1;34;40mData Iteration\033[0m : {Iter}" .format(Iter = iter*Data_Size_Per_Iter))
				
				is_validation = True 
				_, valid_accuracy, _, _, _ = compute_accuracy(
							xs                      = xs, 
							ys                      = ys, 
							is_training             = is_training, 
							is_testing              = is_testing, 
							is_validation           = is_validation, 
							Model_dict              = Model_dict, 
							QUANTIZED_NOW           = QUANTIZED_NOW, 
							prediction              = prediction, 
							v_xs                    = val_data, 
							v_ys                    = val_target, 
							BATCH_SIZE              = BATCH_SIZE, 
							sess                    = sess)
				is_validation = False
				
				total_valid_accuracy = total_valid_accuracy + valid_accuracy
	
			if val_data_num<Data_Size_Per_Iter:
				total_valid_accuracy = total_valid_accuracy
			else:
				total_valid_accuracy = total_valid_accuracy / float(int(val_data_num / Data_Size_Per_Iter))
				
			print("  \033[1;32;40mValidation Accuracy\033[0m = {Valid_Accuracy}".format(Valid_Accuracy=total_valid_accuracy))
		
		#------------------------------#
		#   Training Directory Build   #
		#------------------------------#
		if ((not IS_HYPERPARAMETER_OPT) and ((epoch+1)==(EPOCH_TIME-40) or (epoch+1)==EPOCH_TIME)):
			if (not os.path.exists('Model/'+Model_first_name + '_Model/')) :
				print("\n\033[1;35;40m%s\033[0m is not exist!" %'Model/'+Model_first_name)
				print("\033[1;35;40m%s\033[0m is creating" %'Model/'+Model_first_name)
				os.mkdir(Model_first_name)
			
			#Dir = Model_first_name + '_Model/' + Model_Name + '_' + time.strftime("%Y.%m.%d_%H:%M")
			Dir = 'Model/' + Model_first_name + '_Model/' + Model_Name + '_' + str(int(Train_acc*100)) + '_' + str(int(total_valid_accuracy*100)) + '_' + time.strftime("%Y.%m.%d")
			
			if (not os.path.exists(Dir)):
				print("\n\033[1;35;40m%s\033[0m is not exist!" %Dir)
				print("\033[1;35;40m%s\033[0m is creating\n" %Dir)
				os.makedirs(Dir)
			
			#--------------------#
			#    Saving Model    #
			#--------------------#
			np.savetxt(Dir + '/Hyperparameter.csv', HP, delimiter=",", fmt="%s")
			Model_csv_Generator(Model_dict, Dir + '/model')
			Save_Analyzsis_as_csv(Analysis, Dir + '/Analysis')
			
		#----------------------------#
		#   Saving trained weights   #
		#----------------------------#
		if not IS_HYPERPARAMETER_OPT:
			if (((epoch+1)%10==0) and ((epoch+1)>=(EPOCH_TIME-40))) or (epoch+1)==EPOCH_TIME:
				print("Saving Trained Weights ... \n")
				save_path = saver.save(sess, Dir + '/' + str(epoch+1) + ".ckpt")
				print(save_path)

	######################
	tEnd_All = time.time()
	######################
	print("Total cost {TIME} sec\n" .format(TIME=tEnd_All - tStart_All))

	#-----------------------------------#
	#   Saving Train info as csv file   #
	#-----------------------------------#
	if not IS_HYPERPARAMETER_OPT:
		Train_acc_per_epoch  = np.expand_dims(Train_acc_per_epoch , axis=1)
		Train_loss_per_epoch = np.expand_dims(Train_loss_per_epoch, axis=1)
		Train_info = np.concatenate([Train_acc_per_epoch, Train_loss_per_epoch], axis=1)
		Save_file_as_csv(Dir+'/train_info' , Train_info)
		
	"""
	#***********#
	#   DEBUG   #
	#***********#
	
	train_data_index   = open(Dataset_Path + '/train.txt', 'r').read().splitlines()
	train_target_index = open(Dataset_Path + '/trainannot.txt', 'r').read().splitlines()
	train_data_num = len(train_data_index)
	print("Training   Data Number = {DS}" .format(DS = train_data_num))
	print("Training   Data Result ... ")
	Data_Size_Per_Iter = BATCH_SIZE
	for data_iter in range(train_data_num/Data_Size_Per_Iter):
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
			IS_STUDENT       = False ,
			IS_TRAINING      = True)
			
		#---------------------#
		#   Training Result   #
		#---------------------#
		train_result, train_accuracy, train_accuracy_top2, train_accuracy_top3, Y_pre_train = compute_accuracy(
					xs                      = xs, 
					ys                      = ys, 
					is_training             = is_training, 
					is_testing              = is_testing, 
					is_validation           = False, 
					Model_dict              = Model_dict, 
					QUANTIZED_NOW           = IS_QUANTIZED_ACTIVATION, 
					prediction              = prediction, 
					v_xs                    = train_data, 
					v_ys                    = train_target, 
					BATCH_SIZE              = BATCH_SIZE, 
					sess                    = sess)
		
		if data_iter==0:
			train_result_total        = np.array([train_result])
			train_accuracy_total      = np.array([train_accuracy])
			train_accuracy_top2_total = np.array([train_accuracy_top2])
			train_accuracy_top3_total = np.array([train_accuracy_top3])
			Y_pre_train_total         = np.array([Y_pre_train])
		else:
			train_result_total        = np.concatenate([train_result_total       , np.array([train_result])       ], axis=0)
			train_accuracy_total      = np.concatenate([train_accuracy_total     , np.array([train_accuracy])     ], axis=0)
			train_accuracy_top2_total = np.concatenate([train_accuracy_top2_total, np.array([train_accuracy_top2])], axis=0)
			train_accuracy_top3_total = np.concatenate([train_accuracy_top3_total, np.array([train_accuracy_top3])], axis=0)
			Y_pre_train_total         = np.concatenate([Y_pre_train_total        , np.array([Y_pre_train])        ], axis=0)
	
	train_result        = train_result_total
	train_accuracy      = np.mean(train_accuracy_total)
	train_accuracy_top2 = np.mean(train_accuracy_top2_total)
	train_accuracy_top3 = np.mean(train_accuracy_top3_total)
	Y_pre_train         = Y_pre_train_total
	print("Training   Data Accuracy = {Train_Accuracy}, top2 = {top2}, top3 = {top3}"
	.format(Train_Accuracy=train_accuracy, top2=train_accuracy_top2, top3=train_accuracy_top3))
	Save_file_as_csv('debug_train' , train_accuracy_total)
	"""
	
def Testing(
	# (File Read) Path
	Dataset	                    ,
	Dataset_Path                ,
	Y_pre_Path                  ,
	
	# (File Read) Variable      
	class_num                   ,
	layer                       ,
	H_resize                    ,
	W_resize                    ,
	
	# Parameter	
	Model_dict                  ,
	BATCH_SIZE					,
	IS_SAVING_RESULT_AS_IMAGE	,
	IS_SAVING_RESULT_AS_NPZ		,
	IS_SAVING_PARTIAL_RESULT	,
	IS_TRAINING					,
	IS_HYPERPARAMETER_OPT       ,
	
	# Tensor	
	prediction					,
	
	# Placeholder	
	xs							, 
	ys							,
	is_training					,
	is_testing					,
		
	# File Path (For Loading Trained Weight)
	TESTING_WEIGHT_PATH         ,
	TESTINGN_WEIGHT_MODEL       ,
		
	# Trained Weight Parameters (For Loading Trained Weight)
	saver						= None,

	# File Path (For Saving Result)
	train_target_path 			= None,
	train_Y_pre_path  			= None,
	valid_target_path 			= None,
	valid_Y_pre_path  			= None,
	test_target_path 			= None,
	test_Y_pre_path  			= None,

	# Session
	sess						= None):
	
	#-----------------------------#
	#   Some Control Parameters   #
	#-----------------------------#
	IS_TERNARY = False
	IS_QUANTIZED_ACTIVATION = False
	for layer in range(len(Model_dict)):
		if Model_dict['layer'+str(layer)]['IS_TERNARY'] == 'TRUE':
			IS_TERNARY = True
		if Model_dict['layer'+str(layer)]['IS_QUANTIZED_ACTIVATION'] == 'TRUE':
			IS_QUANTIZED_ACTIVATION = True
	
	is_validation = False
	
	#--------------------------#
	#   Load trained weights   #
	#--------------------------#
	if not IS_TRAINING:
		TESTING_WEIGHT_FILE = TESTING_WEIGHT_PATH + TESTINGN_WEIGHT_MODEL
		print("")
		print("Loading the trained weights ... ")
		print("\033[1;32;40mWeights File\033[0m : {WF}\n" .format(WF = TESTING_WEIGHT_FILE))
		save_path = saver.restore(sess, TESTING_WEIGHT_FILE + ".ckpt")
		
	#---------------#
	#   File Read   #
	#---------------#
	train_data_index   = open(Dataset_Path + '/train.txt', 'r').read().splitlines()
	train_target_index = open(Dataset_Path + '/trainannot.txt', 'r').read().splitlines()
	train_data_num = len(train_data_index)
	#print("Training   Data Number = {DS}" .format(DS = train_data_num))

	valid_data_index   = open(Dataset_Path + '/val.txt', 'r').read().splitlines()
	valid_target_index = open(Dataset_Path + '/valannot.txt', 'r').read().splitlines()
	valid_data_num = len(valid_data_index)
	#print("Validation Data Number = {DS}" .format(DS = valid_data_num))
	
	if Dataset != 'ade20k':
		test_data_index   = open(Dataset_Path + '/test.txt', 'r').read().splitlines()
		test_target_index = open(Dataset_Path + '/testannot.txt', 'r').read().splitlines()
		test_data_num = len(test_data_index)
		#print("Testing    Data Number = {DS}\n" .format(DS = test_data_num))
	
	Data_Size_Per_Iter = BATCH_SIZE
	
	#***********#
	#   TRAIN   #
	#***********#
	if not IS_HYPERPARAMETER_OPT:
		print("Training   Data Result ... ")
		for data_iter in range(train_data_num/Data_Size_Per_Iter):
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
				IS_STUDENT       = False ,
				IS_TRAINING      = True)
				
			#---------------------#
			#   Training Result   #
			#---------------------#
			train_result, train_accuracy, train_accuracy_top2, train_accuracy_top3, Y_pre_train = compute_accuracy(
						xs                      = xs, 
						ys                      = ys, 
						is_training             = is_training, 
						is_testing              = is_testing, 
						is_validation           = is_validation, 
						Model_dict              = Model_dict, 
						QUANTIZED_NOW           = True, 
						prediction              = prediction, 
						v_xs                    = train_data, 
						v_ys                    = train_target, 
						BATCH_SIZE              = BATCH_SIZE, 
						sess                    = sess)
			
			if data_iter==0:
				train_result_total        = np.array([train_result])
				train_accuracy_total      = np.array([train_accuracy])
				train_accuracy_top2_total = np.array([train_accuracy_top2])
				train_accuracy_top3_total = np.array([train_accuracy_top3])
				Y_pre_train_total         = np.array([Y_pre_train])
			else:
				train_result_total        = np.concatenate([train_result_total       , np.array([train_result])       ], axis=0)
				train_accuracy_total      = np.concatenate([train_accuracy_total     , np.array([train_accuracy])     ], axis=0)
				train_accuracy_top2_total = np.concatenate([train_accuracy_top2_total, np.array([train_accuracy_top2])], axis=0)
				train_accuracy_top3_total = np.concatenate([train_accuracy_top3_total, np.array([train_accuracy_top3])], axis=0)
				Y_pre_train_total         = np.concatenate([Y_pre_train_total        , np.array([Y_pre_train])        ], axis=0)
		
		train_result        = train_result_total
		train_accuracy      = np.mean(train_accuracy_total)
		train_accuracy_top2 = np.mean(train_accuracy_top2_total)
		train_accuracy_top3 = np.mean(train_accuracy_top3_total)
		Y_pre_train         = Y_pre_train_total
		print("Training   Data Accuracy = {Train_Accuracy}, top2 = {top2}, top3 = {top3}"
		.format(Train_Accuracy=train_accuracy, top2=train_accuracy_top2, top3=train_accuracy_top3))
		#Save_file_as_csv('debug_test' , train_accuracy_total)
	else:
		train_accuracy = 0
	
	#***********#
	#   VALID   #
	#***********#
	if not IS_HYPERPARAMETER_OPT:
		print("Validation Data Result ... ")
		for data_iter in range(valid_data_num/Data_Size_Per_Iter):
			#---------------------#
			#   Validation file   #
			#---------------------#
			# data_index_part
			if valid_data_num<Data_Size_Per_Iter:
				valid_data_index_part   = valid_data_index
				valid_target_index_part = valid_target_index
			else:
				valid_data_index_part   = valid_data_index[data_iter*Data_Size_Per_Iter:(data_iter+1)*Data_Size_Per_Iter]
				valid_target_index_part = valid_target_index[data_iter*Data_Size_Per_Iter:(data_iter+1)*Data_Size_Per_Iter]
	
			valid_data , valid_target = dataset_parser(
				# Path
				Dataset	         = Dataset,
				Path             = Dataset_Path, 
				Y_pre_Path       = Y_pre_Path,
				data_index       = valid_data_index_part,
				target_index     = valid_target_index_part,		
				
				# Variable
				class_num        = class_num,
				layer            = layer,
				H_resize         = H_resize,
				W_resize         = W_resize,
				# Parameter
				IS_STUDENT       = False,
				IS_TRAINING      = True)
				
			#-----------------------#
			#   Validation Result   #
			#-----------------------#
			valid_result, valid_accuracy, valid_accuracy_top2, valid_accuracy_top3, Y_pre_valid = compute_accuracy(
						xs                      = xs, 
						ys                      = ys, 
						is_training             = is_training, 
						is_testing              = is_testing, 
						is_validation           = is_validation, 
						Model_dict              = Model_dict, 
						QUANTIZED_NOW           = True, 
						prediction              = prediction, 
						v_xs                    = valid_data, 
						v_ys                    = valid_target, 
						BATCH_SIZE              = BATCH_SIZE, 
						sess                    = sess)
			
			if data_iter==0:
				valid_result_total        = np.array([valid_result])
				valid_accuracy_total      = np.array([valid_accuracy])
				valid_accuracy_top2_total = np.array([valid_accuracy_top2])
				valid_accuracy_top3_total = np.array([valid_accuracy_top3])
				Y_pre_valid_total         = np.array([Y_pre_valid])
			else:
				valid_result_total        = np.concatenate([valid_result_total       , np.array([valid_result])       ], axis=0)
				valid_accuracy_total      = np.concatenate([valid_accuracy_total     , np.array([valid_accuracy])     ], axis=0)
				valid_accuracy_top2_total = np.concatenate([valid_accuracy_top2_total, np.array([valid_accuracy_top2])], axis=0)
				valid_accuracy_top3_total = np.concatenate([valid_accuracy_top3_total, np.array([valid_accuracy_top3])], axis=0)
				Y_pre_valid_total         = np.concatenate([Y_pre_valid_total        , np.array([Y_pre_valid])        ], axis=0)
				
		valid_result        = valid_result_total
		valid_accuracy      = np.mean(valid_accuracy_total)
		valid_accuracy_top2 = np.mean(valid_accuracy_top2_total)
		valid_accuracy_top3 = np.mean(valid_accuracy_top3_total)
		Y_pre_valid         = Y_pre_valid_total
		print("validation Data Accuracy = {valid_Accuracy}, top2 = {top2}, top3 = {top3}"
		.format(valid_Accuracy=valid_accuracy, top2=valid_accuracy_top2, top3=valid_accuracy_top3))
	else:
		valid_accuracy = 0

	#***********#
	#   TEST   #
	#***********#
	if Dataset!='ade20k':
		print("Testing    Data Result ... ")
		for data_iter in range(test_data_num/Data_Size_Per_Iter):
			#------------------#
			#   Testing file   #
			#------------------#
			# data_index_part
			if test_data_num<Data_Size_Per_Iter:
				test_data_index_part   = test_data_index
				test_target_index_part = test_target_index
			else:
				test_data_index_part   = test_data_index[data_iter*Data_Size_Per_Iter:(data_iter+1)*Data_Size_Per_Iter]
				test_target_index_part = test_target_index[data_iter*Data_Size_Per_Iter:(data_iter+1)*Data_Size_Per_Iter]
	
			test_data , test_target = dataset_parser(
				# Path
				Dataset	         = Dataset,
				Path             = Dataset_Path, 
				Y_pre_Path       = Y_pre_Path,
				data_index       = test_data_index_part,
				target_index     = test_target_index_part,		
				
				# Variable
				class_num        = class_num,
				layer            = layer,
				H_resize         = H_resize,
				W_resize         = W_resize,
				# Parameter
				IS_STUDENT       = False,
				IS_TRAINING      = True)
				
			#--------------------#
			#   Testing Result   #
			#--------------------#
			test_result, test_accuracy, test_accuracy_top2, test_accuracy_top3, Y_pre_test = compute_accuracy(
						xs                      = xs, 
						ys                      = ys, 
						is_training             = is_training, 
						is_testing              = is_testing, 
						is_validation           = is_validation, 
						Model_dict              = Model_dict, 
						QUANTIZED_NOW           = True, 
						prediction              = prediction, 
						v_xs                    = test_data, 
						v_ys                    = test_target, 
						BATCH_SIZE              = BATCH_SIZE, 
						sess                    = sess)
						
			if data_iter==0:
				test_result_total        = np.array([test_result], )
				test_accuracy_total      = np.array([test_accuracy])
				test_accuracy_top2_total = np.array([test_accuracy_top2])
				test_accuracy_top3_total = np.array([test_accuracy_top3])
				Y_pre_test_total         = np.array([Y_pre_test])
			else:
				test_result_total        = np.concatenate([test_result_total       , np.array([test_result])       ], axis=0)
				test_accuracy_total      = np.concatenate([test_accuracy_total     , np.array([test_accuracy])     ], axis=0)
				test_accuracy_top2_total = np.concatenate([test_accuracy_top2_total, np.array([test_accuracy_top2])], axis=0)
				test_accuracy_top3_total = np.concatenate([test_accuracy_top3_total, np.array([test_accuracy_top3])], axis=0)
				Y_pre_test_total         = np.concatenate([Y_pre_test_total        , np.array([Y_pre_test])        ], axis=0)
		
		test_result        = test_result_total
		test_accuracy      = np.mean(test_accuracy_total)
		test_accuracy_top2 = np.mean(test_accuracy_top2_total)
		test_accuracy_top3 = np.mean(test_accuracy_top3_total)
		Y_pre_test         = Y_pre_test_total
		print("testing    Data Accuracy = {test_Accuracy}, top2 = {top2}, top3 = {top3}\n"
		.format(test_Accuracy=test_accuracy, top2=test_accuracy_top2, top3=test_accuracy_top3))
	
	#****************************#
	#   Saving Result as Image   #
	#****************************#
	if IS_SAVING_RESULT_AS_IMAGE:
		print("Coloring train result ... ")
		train_result = color_result(train_result)
		print("Saving the train data result as image ... ")
		Save_result_as_image(train_target_path, train_result, train_data_index)
		
		print("Coloring valid result ... ")
		valid_result = color_result(valid_result)
		print("Saving the validation data result as image ... ")
		Save_result_as_image(valid_target_path, valid_result, valid_data_index)
		
		if Dataset!='ade20k':
			print("Coloring test result ... ")
			test_result = color_result(test_result)
			print("Saving the test data result as image ... ")
			Save_result_as_image(test_target_path, test_result, test_data_index)

	#*******************************#
	#   Saving Result as NPZ File   #
	#*******************************#
	if IS_SAVING_RESULT_AS_NPZ:
		print("Saving the train Y_pre result as npz ... ")
		Save_result_as_npz(	train_Y_pre_path, Y_pre_train, train_data_index,
							# tensor
							xs						= xs,
							is_training				= is_training, 
							is_testing				= is_testing,              
							is_quantized_activation	= is_quantized_activation,
							# data
							v_xs					= train_data,
							# Parameter
							QUANTIZED_NOW 			= True,
							H_resize				= H_resize,
							W_resize				= W_resize,
							sess					= sess)
		print("Saving the valid Y_pre result as npz ... ")
		Save_result_as_npz(	valid_Y_pre_path, Y_pre_valid, valid_data_index, 
							# tensor
							xs						= xs,
							is_training				= is_training, 
							is_testing				= is_testing,              
							is_quantized_activation	= is_quantized_activation,
							# data
							v_xs					= valid_data,
							# Parameter
							QUANTIZED_NOW 			= True,
							H_resize				= H_resize,
							W_resize				= W_resize,
							sess					= sess)
		if Dataset!='ade20k':
			print("Saving the test Y_pre result as npz ... ")
			Save_result_as_npz(	test_Y_pre_path, Y_pre_test, test_data_index, 
								# tensor
								xs						= xs,
								is_training				= is_training, 
								is_testing				= is_testing,              
								is_quantized_activation	= is_quantized_activation,
								# data
								v_xs					= test_data,
								# Parameter
								QUANTIZED_NOW 			= True,
								H_resize				= H_resize,
								W_resize				= W_resize,
								sess					= sess)

	#===========================#
	#    Save Result as .csv    #
	#===========================#
	if not IS_TRAINING:
		accuracy_train = np.concatenate([[['Train Data  ']], [[train_accuracy]], [[train_accuracy_top2]], [[train_accuracy_top3]]], axis=1)
		accuracy_valid = np.concatenate([[['Valid Data  ']],[[valid_accuracy]], [[valid_accuracy_top2]], [[valid_accuracy_top3]]], axis=1)
		if Dataset!='ade20k':
			accuracy_test  = np.concatenate([[['Test  Data  ']],[[ test_accuracy]], [[ test_accuracy_top2]], [[ test_accuracy_top3]]], axis=1)
		
		title_col = np.expand_dims(np.array(['            ', 'TOP1	', 'TOP2	', 'TOP3	']),axis=0)
		if Dataset!='ade20k':
			accuracy       = np.concatenate([title_col, accuracy_train, accuracy_valid, accuracy_test], axis=0)
		else:
			accuracy       = np.concatenate([title_col, accuracy_train, accuracy_valid], axis=0)
			
		np.savetxt(TESTING_WEIGHT_FILE + '_' + Dataset + '.csv', accuracy, delimiter=",	", fmt="%5s")	
		#Save_file_as_csv(TESTING_WEIGHT_FILE+'_accuracy' , accuracy)
	
	#==============#
	#    output    #
	#==============#
	if Dataset!='ade20k':
		return train_accuracy, valid_accuracy, test_accuracy
	else:
		return train_accuracy, valid_accuracy, valid_accuracy

def Model_dict_Generator(
	csv_file = None,
	class_num = 1
	):
	
	#----------------------#
	#    Hyperparameter    #
	#----------------------#
	# Read Hyperparameter in .csv file 
	# Or, you can directly define it here.
	# --------------------------------------------------------
	# i.e. 
	#     HP_tmp.update({'type'       :['CONV', 'CONV', ...]})
	#     HP_tmp.update({'kernel_size':['3', '3', ...]})
	#         ... 
	#    (Until all hyperparameter is defined)
	# --------------------------------------------------------
	HP_tmp = {}
	keys = []
	with open(csv_file) as csvfile:
		reader = csv.reader(csvfile, delimiter=',', quotechar='|')
		for iter, row in enumerate(reader):
			keys.append(row[0])
			HP_tmp.update({row[0]:row[1:len(row)]})
	
	#------------#
	#    Info    #
	#------------#
	depth = len(HP_tmp['layer'])
	
	#------------------------------#
	#    Placeholder Definition    #
	#------------------------------#
	is_ternary = {}
	is_quantized_activation = {}
	for layer in range(depth):
		is_ternary.update             ({'layer' + str(layer) : tf.placeholder(tf.bool)}) 
		is_quantized_activation.update({'layer' + str(layer) : tf.placeholder(tf.bool)}) 
	
	#------------------------#
	#    Model Definition    #
	#------------------------#
	Model_dict = {}
	for layer in range(depth):
		HP = {}		
		for iter, key in enumerate(keys):
			if key == 'output_channel' and layer == (depth-1):
				HP.update({'output_channel': class_num})
			else:
				HP.update({key: HP_tmp[key][layer]})
		
		HP.update({'is_ternary'             : is_ternary             ['layer' + str(layer)]}) # Placeholder
		HP.update({'is_quantized_activation': is_quantized_activation['layer' + str(layer)]}) # Placeholder

		Model_dict.update({'layer'+str(layer):HP})

	return Model_dict	

def Hyperparameter_Decoder(Hyperparameter, Model_dict):
	"""
	Training Hyperparameter
	|=======================================================================================================|
	| Num|              Type                     |             0                |              1            |
	|=======================================================================================================|
	| 00 | Optimization Method                   | MOMENTUM                     | ADAM                      |
	| 01 | Momentum Rate                         | 0.9                          | 0.99                      |
	| 02 | Initial Learning Rate                 | < 0.01                       | >= 0.01                   |
	| 03 | Initial Learning Rate                 | < 0.001; <0.1;               | >= 0.001; >= 0.1          |
	| 04 | Initial Learning Rate                 | 0.0001; 0.001; 0.01; 0.1;    | 0.0003; 0.003; 0.03; 0.3  |
	| 05 | Learning Rate Drop                    | 5                            | 10                        |
	| 06 | Learning Rate First Drop Time         | Drop by 1/10 at Epoch 40     | Drop by 1/10 at Epoch 60  |
	| 07 | Learning Rate Second Drop Time        | Drop by 1/10 at Epoch 80     | Drop by 1/10 at Epoch 100 |
	| 08 | Weight Decay                          | No                           | Yes                       | 
	| 09 | Weight Decay Lambda                   | 1e-4                         | 1e-3                      |
	| 10 | Batch Size                            | Small                        | Big                       |
	| 11 | Batch Size                            | 32; 128                      | 64; 256                   |
	| 12 | Teacher-Student Strategy              | No                           | Yes                       |
	| 13 | Use Dropout                           | No                           | Yes                       |
	| 14 | Dropout Rate                          | Low                          | High                      | 
	| 15 | Dropout Rate                          | 0.05; 0.2                    | 0.1; 0.3                  |
	| 16 | Weight Ternary Epoch                  | 40                           | 60                        |
	| 17 | Activation Quantized Epoch            | 80                           | 100                       |
	|=======================================================================================================|

	Model Hyperparameter (Per Layer)
	|=======================================================================================================|
	| Num|              Type                     |             0                |              1            |
	|=======================================================================================================|
	| 00 | Weight Ternary                        | No                           | Yes                       |
	| 01 | Activation Quantized                  | No                           | Yes                       |
	| 02 | Batch Normalization                   | No                           | Yes                       |
	| 03 | Shortcut                              | No                           | Yes                       |
	| 04 | Shortcut Distance                     | 1                            | 2                         |
	| 05 | Bottlneck                             | No                           | Yes                       |
	| 06 | Inception                             | No                           | Yes                       |
	| 07 | Dilated (Rate=2)                      | No                           | Yes                       |
	| 08 | Depthwise                             | No                           | Yes                       |
	| 09 | Activation                            | Sigmoid                      | ReLU                      |
	|=======================================================================================================|
	"""
	
	
	HP_dict = {}
	Bit_Now = 0
	Bits = 0
	
	#-------------------------------#
	#    Training Hyperparameter    #
	#-------------------------------#
	# Optimization Method
	Bit_Now = Bit_Now + Bits
	Bits = 1
	if   np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [-1]): HP_dict.update({'OptMethod': 'MOMENTUM'})
	elif np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [ 1]): HP_dict.update({'OptMethod': 'ADAM'})
	# Momentum Rate 
	Bit_Now = Bit_Now + Bits
	Bits = 1
	if   np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [-1]): HP_dict.update({'Momentum_Rate': 0.9})
	elif np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [ 1]): HP_dict.update({'Momentum_Rate': 0.99})
	# Initial Learning Rate
	Bit_Now = Bit_Now + Bits
	Bits = 3
	if   np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [-1, -1, -1]): HP_dict.update({'LEARNING_RATE': 0.0001})
	elif np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [-1, -1,  1]): HP_dict.update({'LEARNING_RATE': 0.0003})
	elif np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [-1,  1, -1]): HP_dict.update({'LEARNING_RATE': 0.001})
	elif np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [-1,  1,  1]): HP_dict.update({'LEARNING_RATE': 0.003})
	elif np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [ 1, -1, -1]): HP_dict.update({'LEARNING_RATE': 0.01})
	elif np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [ 1, -1,  1]): HP_dict.update({'LEARNING_RATE': 0.03})
	elif np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [ 1,  1, -1]): HP_dict.update({'LEARNING_RATE': 0.1})
	elif np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [ 1,  1,  1]): HP_dict.update({'LEARNING_RATE': 0.3})
	# Learning Rate Drop
	Bit_Now = Bit_Now + Bits
	Bits = 1
	if   np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [-1]): HP_dict.update({'LR_DECADE':  5})
	elif np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [ 1]): HP_dict.update({'LR_DECADE': 10})
	# Learning Rate First Drop Time
	Bit_Now = Bit_Now + Bits
	Bits = 1
	if   np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [-1]): HP_dict.update({'LR_DECADE_1st_EPOCH': 40})
	elif np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [ 1]): HP_dict.update({'LR_DECADE_1st_EPOCH': 60})
	# Learning Rate Second Drop Time
	Bit_Now = Bit_Now + Bits
	Bits = 1
	if   np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [-1]): HP_dict.update({'LR_DECADE_2nd_EPOCH': 80})
	elif np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [ 1]): HP_dict.update({'LR_DECADE_2nd_EPOCH': 100})	
	# Weight Decay Lambda
	Bit_Now = Bit_Now + Bits
	Bits = 2
	if   np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [-1, -1]): HP_dict.update({'Weight_Decay_Lambda': 0.0})
	elif np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [-1,  1]): HP_dict.update({'Weight_Decay_Lambda': 0.0})
	elif np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [ 1, -1]): HP_dict.update({'Weight_Decay_Lambda': 1e-4})
	elif np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [ 1,  1]): HP_dict.update({'Weight_Decay_Lambda': 1e-3})
	# Batch Size
	Bit_Now = Bit_Now + Bits
	Bits = 2
	if   np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [-1, -1]): HP_dict.update({'BATCH_SIZE': 128})
	elif np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [-1,  1]): HP_dict.update({'BATCH_SIZE': 256})
	elif np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [ 1, -1]): HP_dict.update({'BATCH_SIZE': 512})
	elif np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [ 1,  1]): HP_dict.update({'BATCH_SIZE': 1024})
	# Teacher-Student Strategy
	Bit_Now = Bit_Now + Bits
	Bits = 1
	if   np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [-1]): HP_dict.update({'IS_STUDENT': 'FALSE'})
	elif np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [ 1]): HP_dict.update({'IS_STUDENT': 'TRUE'})
	# Dropout Rate
	Bit_Now = Bit_Now + Bits
	Bits = 3
	if   np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [-1, -1, -1]): HP_dict.update({'Dropout_Rate': 0.0})
	elif np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [-1, -1,  1]): HP_dict.update({'Dropout_Rate': 0.0})
	elif np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [-1,  1, -1]): HP_dict.update({'Dropout_Rate': 0.0})
	elif np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [-1,  1,  1]): HP_dict.update({'Dropout_Rate': 0.0})
	elif np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [ 1, -1, -1]): HP_dict.update({'Dropout_Rate': 0.05})
	elif np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [ 1, -1,  1]): HP_dict.update({'Dropout_Rate': 0.1})
	elif np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [ 1,  1, -1]): HP_dict.update({'Dropout_Rate': 0.2})
	elif np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [ 1,  1,  1]): HP_dict.update({'Dropout_Rate': 0.3})
	# Weight Ternary Epoch
	Bit_Now = Bit_Now + Bits
	Bits = 1
	if   np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [-1]): HP_dict.update({'TERNARY_EPOCH': 2})
	elif np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [ 1]): HP_dict.update({'TERNARY_EPOCH': 2})
	# Activation Quantized Epoch
	Bit_Now = Bit_Now + Bits
	Bits = 1
	if   np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [-1]): HP_dict.update({'QUANTIZED_ACTIVATION_EPOCH': 3})
	elif np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [ 1]): HP_dict.update({'QUANTIZED_ACTIVATION_EPOCH': 3})
	
	#-----------------------------------#
	#    Model Hyperparameter Update    #
	#-----------------------------------#
	for layer in range(len(Model_dict)):
		# IS_TERNARY
		Bit_Now = Bit_Now + Bits
		Bits = 1
		if   np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [-1]): Model_dict["layer"+str(layer)].update({'IS_TERNARY': 'FALSE'})
		elif np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [ 1]): Model_dict["layer"+str(layer)].update({'IS_TERNARY': 'TRUE' })
		# IS_QUANTIZED_ACTIVATION
		Bit_Now = Bit_Now + Bits
		Bits = 1
		if   np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [-1]): Model_dict["layer"+str(layer)].update({'IS_QUANTIZED_ACTIVATION': 'FALSE'})
		elif np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [ 1]): Model_dict["layer"+str(layer)].update({'IS_QUANTIZED_ACTIVATION': 'TRUE' })
		# is_batch_norm
		Bit_Now = Bit_Now + Bits
		Bits = 1
		if   np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [-1]): Model_dict["layer"+str(layer)].update({'is_batch_norm': 'FALSE'})
		elif np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [ 1]): Model_dict["layer"+str(layer)].update({'is_batch_norm': 'TRUE' })
		# is_shortcut
		Bit_Now = Bit_Now + Bits
		Bits = 1
		if   np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [-1]): Model_dict["layer"+str(layer)].update({'is_shortcut': 'FALSE'})
		elif np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [ 1]): Model_dict["layer"+str(layer)].update({'is_shortcut': 'TRUE' })
		# shortcut_destination
		Bit_Now = Bit_Now + Bits
		Bits = 1
		if   np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [-1]): Model_dict["layer"+str(layer)].update({'shortcut_destination': 'layer'+str(layer+1)})
		elif np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [ 1]): Model_dict["layer"+str(layer)].update({'shortcut_destination': 'layer'+str(layer+2) })
		# is_bottleneck
		Bit_Now = Bit_Now + Bits
		Bits = 1
		if   np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [-1]): Model_dict["layer"+str(layer)].update({'is_bottleneck': 'FALSE'})
		elif np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [ 1]): Model_dict["layer"+str(layer)].update({'is_bottleneck': 'TRUE' })
		# is_inception
		Bit_Now = Bit_Now + Bits
		Bits = 1
		if   np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [-1]): Model_dict["layer"+str(layer)].update({'is_inception': 'FALSE'})
		elif np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [ 1]): Model_dict["layer"+str(layer)].update({'is_inception': 'TRUE' })
		# is_dilated
		Bit_Now = Bit_Now + Bits
		Bits = 1
		if   np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [-1]): Model_dict["layer"+str(layer)].update({'is_dilated': 'FALSE'})
		elif np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [ 1]): Model_dict["layer"+str(layer)].update({'is_dilated': 'TRUE' })
		# is_depthwise
		Bit_Now = Bit_Now + Bits
		Bits = 1
		if   np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [-1]): Model_dict["layer"+str(layer)].update({'is_depthwise': 'FALSE'})
		elif np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [ 1]): Model_dict["layer"+str(layer)].update({'is_depthwise': 'TRUE' })
		# Activation
		Bit_Now = Bit_Now + Bits
		Bits = 1
		if   np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [-1]): Model_dict["layer"+str(layer)].update({'Activation': 'Sigmoid'})
		elif np.array_equal(Hyperparameter[Bit_Now : Bit_Now + Bits], [ 1]): Model_dict["layer"+str(layer)].update({'Activation': 'ReLU'   })
	
	return HP_dict, Model_dict

def Model_dict_Decoder(
	net, 
	Model_dict, 
	is_training, 
	is_testing,
	DROPOUT_RATE
	):
	
	Analysis = Analyzer({}, net, type='DATA', name='Input')
	past_layer = 0
	max_parameter = 0
	
	for layer in range(len(Model_dict)):
		with tf.variable_scope('layer%d' %(layer)):
			layer_now = Model_dict['layer'+str(layer)]

			if layer_now['type'] == 'POOL':
				net, indices, output_shape = indice_pool(net, 
														kernel_size             = int(layer_now['kernel_size']), 
														stride                  = int(layer_now['stride']     ), 
														Analysis                = Analysis,
														IS_QUANTIZED_ACTIVATION = layer_now['IS_QUANTIZED_ACTIVATION'] == 'TRUE',
														scope                   = layer_now['scope'])
				if layer_now['indice'] != 'None':
					Model_dict[layer_now['indice']].update({'indice':indices, 'output_shape':output_shape})
					
			elif layer_now['type'] == 'UNPOOL':
				net = indice_unpool(net, 
									stride       = int(layer_now['stride']), 
									output_shape = layer_now['output_shape'], 
									indices      = layer_now['indice'], 
									scope        = layer_now['scope'])
			elif layer_now['type'] == 'CONV':
				if layer==(len(Model_dict)-1):
					net = tf.cond(is_testing, lambda: net, lambda: tf.layers.dropout(net, DROPOUT_RATE))
					
				net  = conv2D( net, 
							kernel_size             = int(layer_now['kernel_size']     ), 
							stride                  = int(layer_now['stride']          ),
							internal_channel        = int(layer_now['internal_channel']),
							output_channel          = int(layer_now['output_channel']  ),
							rate                    = int(layer_now['rate']            ),
							group                   = int(layer_now['group']           ),
							initializer             = tf.contrib.layers.variance_scaling_initializer(),
							is_constant_init        = False,          
							is_bottleneck           = layer_now['is_bottleneck']           == 'TRUE',                  
							is_batch_norm           = layer_now['is_batch_norm']           == 'TRUE',      
							is_dilated              = layer_now['is_dilated']              == 'TRUE',      
							is_depthwise            = layer_now['is_depthwise']            == 'TRUE',  
							is_inception            = layer_now['is_inception']            == 'TRUE',
							is_ternary              = layer_now['is_ternary']                       ,
							is_quantized_activation = layer_now['is_quantized_activation']          ,
							IS_TERNARY              = layer_now['IS_TERNARY']              == 'TRUE',     
							IS_QUANTIZED_ACTIVATION = layer_now['IS_QUANTIZED_ACTIVATION'] == 'TRUE',						   
							is_training             = is_training,       
							is_testing              = is_testing,            
							Activation              = layer_now['Activation'],
							padding                 = "SAME",
							Analysis                = Analysis,
							scope                   = layer_now['scope'])
			
			# Shortcut
			if bool(layer_now.get('shortcut_num')):
				for shortcut_index in range(layer_now['shortcut_num']):
					with tf.variable_scope('shortcut%d' %(shortcut_index)):
						shortcut = layer_now['shortcut_input'][str(shortcut_index)]
						# Add shortcut 
						shortcut = shortcut_Module( net                     = shortcut, 
													destination             = net,
													initializer             = tf.contrib.layers.variance_scaling_initializer(),
													is_constant_init        = False, 
													is_batch_norm           = layer_now['is_batch_norm']           == 'TRUE',
													is_training             = is_training,
													is_testing              = is_testing,
													is_ternary              = layer_now['is_ternary']                       ,
													is_quantized_activation = layer_now['is_quantized_activation']          ,
													IS_TERNARY              = layer_now['IS_TERNARY']              == 'TRUE', 	
													IS_QUANTIZED_ACTIVATION = layer_now['IS_QUANTIZED_ACTIVATION'] == 'TRUE',
													padding                 = "SAME",			     
													Analysis                = Analysis)
						net = tf.add(net, shortcut)
						
						# Activation Quantization
						if layer_now['IS_QUANTIZED_ACTIVATION'] == 'TRUE':
							quantize_Module(net, layer_now['is_quantized_activation'])
							
						#   Analyzer   #
						Analysis = Analyzer(Analysis, 
											net, 
											type                    = 'ADD', 
											IS_QUANTIZED_ACTIVATION = layer_now['IS_QUANTIZED_ACTIVATION'] == 'TRUE', 
											name                    = 'shortcut_ADD')
						
			if layer_now['is_shortcut'] == 'TRUE':
				if bool(Model_dict[layer_now['shortcut_destination']].get('shortcut_num')):
					shortcut_num = Model_dict[layer_now['shortcut_destination']]['shortcut_num'] + 1
					Model_dict[layer_now['shortcut_destination']].update({'shortcut_num':shortcut_num})
					Model_dict[layer_now['shortcut_destination']]['shortcut_input'].update({str(shortcut_num-1):net})
				else:
					shortcut_num = 1
					Model_dict[layer_now['shortcut_destination']].update({'shortcut_num':shortcut_num})
					Model_dict[layer_now['shortcut_destination']].update({'shortcut_input':{str(shortcut_num-1):net}})
				

		##
		current_layer = len(Analysis)-1
		layer_num = current_layer - past_layer
		
		# parameter_in
		if past_layer>=0 and past_layer < 10:
			parameter_in  = Analysis['layer00'+str(past_layer)]['Activation'] * Analysis['layer00'+str(past_layer)]['Activation Bits']
		elif past_layer>=10 and past_layer < 100:
			parameter_in  = Analysis['layer0'+str(past_layer)]['Activation'] * Analysis['layer0'+str(past_layer)]['Activation Bits']
		elif past_layer > 100:
			parameter_in  = Analysis['layer'+str(past_layer)]['Activation'] * Analysis['layer'+str(past_layer)]['Activation Bits']
			
		# parameter_out
		if current_layer>=0 and current_layer < 10:
			parameter_out  = Analysis['layer00'+str(current_layer)]['Activation'] * Analysis['layer00'+str(current_layer)]['Activation Bits']
		elif current_layer>=10 and current_layer < 100:
			parameter_out  = Analysis['layer0'+str(current_layer)]['Activation'] * Analysis['layer0'+str(current_layer)]['Activation Bits']
		elif current_layer > 100:
			parameter_out  = Analysis['layer'+str(current_layer)]['Activation'] * Analysis['layer'+str(current_layer)]['Activation Bits']
		
		# parameter_kernel
		parameter_kernel = 0
		for i in range(layer_num):
			layer_now = past_layer + i + 1
			if layer_now>=0 and layer_now < 10:
				parameter_kernel = Analysis['layer00'+str(layer_now)]['Param'] * Analysis['layer00'+str(layer_now)]['Kernel Bits']
			elif layer_now>=10 and layer_now < 100:
				parameter_kernel = Analysis['layer0'+str(layer_now)]['Param'] * Analysis['layer0'+str(layer_now)]['Kernel Bits']
			elif layer_now > 100:
				parameter_kernel = Analysis['layer'+str(layer_now)]['Param'] * Analysis['layer'+str(layer_now)]['Kernel Bits']
				
			parameter = parameter_in + parameter_out + parameter_kernel

			if parameter > max_parameter:
				max_parameter = parameter

		past_layer = current_layer
		
	return net, Analysis, max_parameter

def Model_csv_Generator(
	Model_dict,
	csv_file
	):

	Model = np.array(['layer'                  ,
	                  'type'                   ,
	                  'kernel_size'            ,
	                  'stride'                 ,
	                  'internal_channel'       ,
	                  'output_channel'         ,
	                  'rate'                   ,
	                  'group'                  ,
	                  'is_shortcut'            ,
					  'shortcut_destination'   ,
	                  'is_bottleneck'          ,
	                  'is_batch_norm'          ,
	                  'is_dilated'             ,
	                  'is_depthwise'           ,
					  'is_inception'           ,
	                  'IS_TERNARY'             ,
	                  'IS_QUANTIZED_ACTIVATION',
	                  'Activation'             ,
	                  'indice'                 ,
					  'scope'                  ])     
					  
	Model = np.expand_dims(Model, axis=1)

	for layer in range(len(Model_dict)):
		for iter, key in enumerate(Model):
			if iter==0:
				Model_per_layer = np.array(['layer'+str(layer)])
			else:
				if key[0]=='indice':
					if Model_dict['layer'+str(layer)]['type']=='POOL':
						Model_per_layer = np.concatenate([Model_per_layer, np.array([Model_dict['layer'+str(layer)][key[0]]])], axis=0)
					else:
						Model_per_layer = np.concatenate([Model_per_layer,np.array(['None'])], axis=0)
				else:
					Model_per_layer = np.concatenate([Model_per_layer, np.array([Model_dict['layer'+str(layer)][key[0]]])], axis=0)
					
		Model_per_layer = np.expand_dims(Model_per_layer, axis=1)
		Model = np.concatenate([Model, Model_per_layer], axis=1)
		
	np.savetxt(csv_file + '.csv', Model, delimiter=",", fmt="%s")
	
#============#
#   Parser   #
#============#
def dataset_parser(# Path
	               Dataset,          # e.g. 'CamVid'
	               Path,             # e.g. '/Path_to_Dataset/CamVid'
	               Y_pre_Path,       # e.g. '/Path_to_Y_pre/CamVid'
	               data_index,
	               target_index,
                   
	               # Variable
	               class_num,
	               layer=None,       # Choose the needed layer 
	               H_resize=224,
	               W_resize=224,
                   
	               # Parameter
	               IS_STUDENT=False,
	               IS_TRAINING=False
	               ):
	
	# Dataset
	data  , data_shape = read_dataset_file(data_index, Path, Dataset, H_resize, W_resize)
	if IS_TRAINING and IS_STUDENT:	 
		target, _          = read_Y_pre_file(data_index, Y_pre_Path, H_resize, W_resize, layer)
	else:
		target, _          = read_dataset_file(target_index, Path, Dataset, H_resize, W_resize, True)
		target = one_of_k(target, class_num)
	
	#print("Class Num : {CN}" .format(CN=np.max(target)))

	#print("Shape of data   : {Shape}" .format(Shape = np.shape(data)))
	#print("Shape of target : {Shape}" .format(Shape = np.shape(target)))
	
	return data, target

def file_name_parser(filepath):
	filename = [f for f in listdir(filepath) if isfile(join(filepath, f))]

	return filename

#===============#
#   File Read   #
#===============#
def read_Y_pre_file(data_index, Path, H_resize, W_resize, layer=None):
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

def read_dataset_file(data_index, Path, Dataset, H_resize, W_resize, IS_TARGET=False):
	for i, file_name in enumerate(data_index):
		# Read Data
		data_tmp  = misc.imread(Path + file_name)
		shape_tmp = np.shape(data_tmp)[0:2]

		# Data Preprocessing
		#data_tmp = scipy.misc.imresize(data_tmp, (H_resize, W_resize))
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

def read_csv_file(file): # (To recall how to read csv file. Not to use.)
	with open(file) as csvfile:
		reader = csv.reader(csvfile, delimiter=',', quotechar='|')
		for iter, row in enumerate(reader):
			print(row)
	
#===============#
#   File Save   #
#===============#
def Save_file_as_csv(Path, file):
	np.savetxt(Path + '.csv', file, delimiter=",")

#==============#
#   Analyzer   #
#==============#
def Analyzer(Analysis, net, type, kernel_shape=None, stride=0, group=1,
			is_depthwise=False,
			IS_TERNARY=False,
			IS_QUANTIZED_ACTIVATION=False,
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
	
	if IS_QUANTIZED_ACTIVATION:
		activation_bits = 16
	else:
		activation_bits = 32
		
	weight_bits = 0
	
	if type=='CONV':
		if IS_TERNARY:
			weight_bits = 2
		else:
			weight_bits = 32
			
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
		

	components = {'name'            : name, 
				  'Input Height'    : H,
				  'Input Width'     : W,
				  'Input Depth'     : D,
				  'Type'            : type,
				  'Kernel Height'   : h,
				  'Kernel Width'    : w,
				  'Kernel Depth'    : i, 
				  'Kernel Number'   : o,
				  'Stride'          : stride,
				  'Kernel Bits'     : weight_bits,
				  'Macc'            : macc, 
				  'Comp'            : comp, 
				  'Add'             : add, 
				  'Div'             : div, 
				  'Exp'             : exp, 
				  'Activation'      : activation, 
				  'Activation Bits' : activation_bits,
				  'Param'           : param,
				  'PE Height'	    : PE_row,
				  'PE Width'	    : PE_col,
				  'Macc Cycle'	    : Macc_Cycle,
				  'Input Cycle'	    : Input_Cycle,
				  'Kernel Cycle'    : Kernel_Cycle,
				  'Output Cycle'    : Output_Cycle,
				  'Bottleneck'      : Bottleneck}

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
	
	components = np.array(['name'            ,
				           'Input Height'    ,
				           'Input Width'     ,
				           'Input Depth'     ,
				           'Type'            ,
				           'Kernel Height'   ,
				           'Kernel Width'    ,
				           'Kernel Depth'    ,
				           'Kernel Number'   ,
				           'Stride'          ,
						   'Kernel Bits'     ,
				           'Macc'            ,
				           'Comp'            ,
				           'Add'             ,
				           'Div'             ,
				           'Exp'             ,
				           'Activation'      ,
						   'Activation Bits' ,
				           'Param'           ,
				  		   'PE Height'	     ,
				  		   'PE Width'	     ,
				  		   'Macc Cycle'	     ,
				  		   'Input Cycle'     ,
				  		   'Kernel Cycle'    ,
				  		   'Output Cycle'    ,
				  		   'Bottleneck'      ])

	Ana = np.expand_dims(components, axis=1)
	for i, key in enumerate(keys):
		#if i==0:
		#	Ana = np.expand_dims(np.array([Analysis[key][x] for x in components]), axis=1)
		#else:	
		Ana = np.concatenate([Ana, np.expand_dims(np.array([Analysis[key][x] for x in components]), axis=1)], axis=1)

	np.savetxt(FILE + '.csv', Ana, delimiter=",", fmt="%s") 

#====================#
#   CNN Components   #
#====================#
# (No Use Now)
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

def residual_Module( net, kernel_size, stride, internal_channel, output_channel, rate, group,
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
					Activation              ,
					padding			        ,
					Analysis				): 

	#===============================#
	#   Bottleneck Residual Block   #
	#===============================#
	if is_bottleneck: 
		with tf.variable_scope("bottle_neck"):
			with tf.variable_scope("conv1_1x1"):
				net = conv2D_Module( net, kernel_size=1, stride=1, output_channel=internal_channel, rate=rate, group=1,
			                     initializer              = initializer              ,
			                     is_constant_init         = is_constant_init         ,
			                     is_batch_norm            = is_batch_norm            ,
			                     is_dilated               = is_dilated               ,
			                     is_depthwise             = False                    ,
			                     is_ternary               = is_ternary               ,
			                     is_training              = is_training              ,
			                     is_testing               = is_testing               ,
			                     is_quantized_activation  = is_quantized_activation  ,
			                     IS_TERNARY               = IS_TERNARY               ,
			                     IS_QUANTIZED_ACTIVATION  = IS_QUANTIZED_ACTIVATION  ,
			                     Activation               = Activation               ,
			                     padding                  = padding                  ,
			                     Analysis                 = Analysis                 )

			with tf.variable_scope("conv2_3x3"):
				net = conv2D_Module( net, kernel_size=3, stride=stride, output_channel=internal_channel, rate=rate, group=group,
			                     initializer              = initializer              ,
			                     is_constant_init         = is_constant_init         ,
			                     is_batch_norm            = is_batch_norm            ,
			                     is_dilated               = is_dilated               ,
			                     is_depthwise             = is_depthwise             ,
			                     is_ternary               = is_ternary               ,
			                     is_training              = is_training              ,
			                     is_testing               = is_testing               ,
			                     is_quantized_activation  = is_quantized_activation  ,
			                     IS_TERNARY               = IS_TERNARY               ,
			                     IS_QUANTIZED_ACTIVATION  = IS_QUANTIZED_ACTIVATION  ,
			                     Activation               = Activation               ,
			                     padding                  = padding                  ,
			                     Analysis                 = Analysis                 )
				
			with tf.variable_scope("conv3_1x1"):
				net = conv2D_Module( net, kernel_size=1, stride=stride, output_channel=output_channel, rate=rate, group=group,
			                     initializer              = initializer              ,
			                     is_constant_init         = is_constant_init         ,
			                     is_batch_norm            = is_batch_norm            ,
			                     is_dilated               = is_dilated               ,
			                     is_depthwise             = False                    ,
			                     is_ternary               = is_ternary               ,
			                     is_training              = is_training              ,
			                     is_testing               = is_testing               ,
			                     is_quantized_activation  = is_quantized_activation  ,
			                     IS_TERNARY               = IS_TERNARY               ,
			                     IS_QUANTIZED_ACTIVATION  = IS_QUANTIZED_ACTIVATION  ,
			                     Activation               = Activation               ,
			                     padding                  = padding                  ,
			                     Analysis                 = Analysis                 )

	#===========================#
	#   Normal Residual Block   #
	#===========================#
	else: 
		with tf.variable_scope("conv1_3x3"):
			net = conv2D_Module( net, kernel_size=3, stride=stride, output_channel=internal_channel, rate=rate, group=group,
			                     initializer              = initializer              ,
			                     is_constant_init         = is_constant_init         ,
			                     is_batch_norm            = is_batch_norm            ,
			                     is_dilated               = is_dilated               ,
			                     is_depthwise             = is_depthwise             ,
			                     is_ternary               = is_ternary               ,
			                     is_training              = is_training              ,
			                     is_testing               = is_testing               ,
			                     is_quantized_activation  = is_quantized_activation  ,
			                     IS_TERNARY               = IS_TERNARY               ,
			                     IS_QUANTIZED_ACTIVATION  = IS_QUANTIZED_ACTIVATION  ,
			                     Activation               = Activation               ,
			                     padding                  = padding                  ,
			                     Analysis                 = Analysis                 )
			if is_depthwise:
				with tf.variable_scope("depthwise_conv1x1"):
					net = conv2D_Module( net, kernel_size=1, stride=1, output_channel=internal_channel, rate=rate, group=1,
								         initializer              = initializer              ,
								         is_constant_init         = is_constant_init         ,
								         is_batch_norm            = is_batch_norm            ,
								         is_dilated               = False                    ,
								         is_depthwise             = False                    ,
								         is_ternary               = is_ternary               ,
								         is_training              = is_training              ,
								         is_testing               = is_testing               ,
								         is_quantized_activation  = is_quantized_activation  ,
								         IS_TERNARY               = IS_TERNARY               ,
								         IS_QUANTIZED_ACTIVATION  = IS_QUANTIZED_ACTIVATION  ,
								         Activation               = Activation               ,
								         padding                  = padding                  ,
								         Analysis                 = Analysis                 )

		with tf.variable_scope("conv2_3x3"):
			net = conv2D_Module( net, kernel_size=3, stride=1, output_channel=output_channel, rate=rate, group=group,
			                     initializer              = initializer              ,
			                     is_constant_init         = is_constant_init         ,
			                     is_batch_norm            = is_batch_norm            ,
			                     is_dilated               = is_dilated               ,
			                     is_depthwise             = is_depthwise             ,
			                     is_ternary               = is_ternary               ,
			                     is_training              = is_training              ,
			                     is_testing               = is_testing               ,
			                     is_quantized_activation  = is_quantized_activation  ,
			                     IS_TERNARY               = IS_TERNARY               ,
			                     IS_QUANTIZED_ACTIVATION  = IS_QUANTIZED_ACTIVATION  ,
			                     Activation               = Activation               ,
			                     padding                  = padding                  ,
			                     Analysis                 = Analysis                 )
			if is_depthwise:
				with tf.variable_scope("depthwise_conv1x1"):
					net = conv2D_Module( net, kernel_size=1, stride=1, output_channel=output_channel, rate=rate, group=1,
								         initializer              = initializer              ,
								         is_constant_init         = is_constant_init         ,
								         is_batch_norm            = is_batch_norm            ,
								         is_dilated               = False                    ,
								         is_depthwise             = False                    ,
								         is_ternary               = is_ternary               ,
								         is_training              = is_training              ,
								         is_testing               = is_testing               ,
								         is_quantized_activation  = is_quantized_activation  ,
								         IS_TERNARY               = IS_TERNARY               ,
								         IS_QUANTIZED_ACTIVATION  = IS_QUANTIZED_ACTIVATION  ,
								         Activation               = Activation               ,
								         padding                  = padding                  ,
								         Analysis                 = Analysis                 )
										
	return net

def SEP_Module(net, kernel_size, stride, internal_channel, output_channel, rate, group,
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
			   Activation              ,
			   padding			       ,
			   Analysis				   ):

	with tf.variable_scope("SEP_Module"):
		with tf.variable_scope("Reduction"):
			net = conv2D_Module( net, kernel_size=1, stride=1, output_channel=internal_channel, rate=rate, group=1,
			                     initializer              = initializer              ,
			                     is_constant_init         = is_constant_init         ,
			                     is_batch_norm            = is_batch_norm            ,
			                     is_dilated               = is_dilated               ,
			                     is_depthwise             = False                    ,
			                     is_ternary               = is_ternary               ,
			                     is_training              = is_training              ,
			                     is_testing               = is_testing               ,
			                     is_quantized_activation  = is_quantized_activation  ,
			                     IS_TERNARY               = IS_TERNARY               ,
			                     IS_QUANTIZED_ACTIVATION  = IS_QUANTIZED_ACTIVATION  ,
			                     Activation               = Activation               ,
			                     padding                  = padding                  ,
			                     Analysis                 = Analysis                 )

		with tf.variable_scope("PatternConv1"):
			with tf.variable_scope("Pattern"):
				Pattern = conv2D_Module( net, kernel_size=kernel_size, stride=stride, output_channel=internal_channel, rate=rate, group=group,
			                             initializer              = initializer              ,
			                             is_constant_init         = is_constant_init         ,
			                             is_batch_norm            = is_batch_norm            ,
			                             is_dilated               = is_dilated               ,
			                             is_depthwise             = is_depthwise             ,
			                             is_ternary               = is_ternary               ,
			                             is_training              = is_training              ,
			                             is_testing               = is_testing               ,
			                             is_quantized_activation  = is_quantized_activation  ,
			                             IS_TERNARY               = IS_TERNARY               ,
			                             IS_QUANTIZED_ACTIVATION  = IS_QUANTIZED_ACTIVATION  ,
			                             Activation               = Activation               ,
			                             padding                  = padding                  ,
			                             Analysis                 = Analysis                 )
					
				if is_depthwise:
					with tf.variable_scope("depthwise_conv1x1"):
						Pattern = conv2D_Module( Pattern, kernel_size=1, stride=1, output_channel=internal_channel, rate=rate, group=1,
									             initializer              = initializer              ,
									             is_constant_init         = is_constant_init         ,
									             is_batch_norm            = is_batch_norm            ,
									             is_dilated               = False                    ,
									             is_depthwise             = False                    ,
									             is_ternary               = is_ternary               ,
									             is_training              = is_training              ,
									             is_testing               = is_testing               ,
									             is_quantized_activation  = is_quantized_activation  ,
									             IS_TERNARY               = IS_TERNARY               ,
									             IS_QUANTIZED_ACTIVATION  = IS_QUANTIZED_ACTIVATION  ,
									             Activation               = Activation               ,
									             padding                  = padding                  ,
									             Analysis                 = Analysis                 )

			with tf.variable_scope("Pattern_Residual"):
				Pattern_Residual = conv2D_Module( net, kernel_size=1, stride=1, output_channel=internal_channel, rate=rate, group=1,
			                                      initializer              = initializer              ,
			                                      is_constant_init         = is_constant_init         ,
			                                      is_batch_norm            = is_batch_norm            ,
			                                      is_dilated               = is_dilated               ,
			                                      is_depthwise             = False                    ,
			                                      is_ternary               = is_ternary               ,
			                                      is_training              = is_training              ,
			                                      is_testing               = is_testing               ,
			                                      is_quantized_activation  = is_quantized_activation  ,
			                                      IS_TERNARY               = IS_TERNARY               ,
			                                      IS_QUANTIZED_ACTIVATION  = IS_QUANTIZED_ACTIVATION  ,
			                                      Activation               = Activation               ,
			                                      padding                  = padding                  ,
			                                      Analysis                 = Analysis                 )
				

			# Adding Pattern and Pattern Residual
			net = tf.add(Pattern, Pattern_Residual)
			
			# Activation Quantization
			if IS_QUANTIZED_ACTIVATION:
				quantize_Module(net, is_quantized_activation)
				
			##   Analyzer   ##
			#Analysis = Analyzer(Analysis, net, type='ADD' , name='SEP_ADD')
		
		with tf.variable_scope("PatternConv2"):
			with tf.variable_scope("Pattern"):
				Pattern = conv2D_Module( net, kernel_size=kernel_size, stride=1, output_channel=internal_channel/2, rate=rate, group=group,
			                             initializer              = initializer              ,
			                             is_constant_init         = is_constant_init         ,
			                             is_batch_norm            = is_batch_norm            ,
			                             is_dilated               = is_dilated               ,
			                             is_depthwise             = is_depthwise             ,
			                             is_ternary               = is_ternary               ,
			                             is_training              = is_training              ,
			                             is_testing               = is_testing               ,
			                             is_quantized_activation  = is_quantized_activation  ,
			                             IS_TERNARY               = IS_TERNARY               ,
			                             IS_QUANTIZED_ACTIVATION  = IS_QUANTIZED_ACTIVATION  ,
			                             Activation               = Activation               ,
			                             padding                  = padding                  ,
			                             Analysis                 = Analysis                 )
					
				if is_depthwise:
					with tf.variable_scope("depthwise_conv1x1"):
						Pattern = conv2D_Module( Pattern, kernel_size=1, stride=1, output_channel=internal_channel/2, rate=rate, group=1,
									             initializer              = initializer              ,
									             is_constant_init         = is_constant_init         ,
									             is_batch_norm            = is_batch_norm            ,
									             is_dilated               = False                    ,
									             is_depthwise             = False                    ,
									             is_ternary               = is_ternary               ,
									             is_training              = is_training              ,
									             is_testing               = is_testing               ,
									             is_quantized_activation  = is_quantized_activation  ,
									             IS_TERNARY               = IS_TERNARY               ,
									             IS_QUANTIZED_ACTIVATION  = IS_QUANTIZED_ACTIVATION  ,
									             Activation               = Activation               ,
									             padding                  = padding                  ,
									             Analysis                 = Analysis                 )
				
			with tf.variable_scope("Pattern_Residual"):
				Pattern_Residual = conv2D_Module( net, kernel_size=1, stride=1, output_channel=internal_channel/2, rate=rate, group=1,
			                                      initializer              = initializer              ,
			                                      is_constant_init         = is_constant_init         ,
			                                      is_batch_norm            = is_batch_norm            ,
			                                      is_dilated               = is_dilated               ,
			                                      is_depthwise             = False                    ,
			                                      is_ternary               = is_ternary               ,
			                                      is_training              = is_training              ,
			                                      is_testing               = is_testing               ,
			                                      is_quantized_activation  = is_quantized_activation  ,
			                                      IS_TERNARY               = IS_TERNARY               ,
			                                      IS_QUANTIZED_ACTIVATION  = IS_QUANTIZED_ACTIVATION  ,
			                                      Activation               = Activation               ,
			                                      padding                  = padding                  ,
			                                      Analysis                 = Analysis                 )

			# Adding Pattern and Pattern Residual
			net = tf.add(Pattern, Pattern_Residual)
			
			# Activation Quantization
			if IS_QUANTIZED_ACTIVATION:
				quantize_Module(net, is_quantized_activation)
			
			##   Analyzer   ##
			#Analysis = Analyzer(Analysis, net, type='ADD' , name='SEP_ADD')
		
		with tf.variable_scope("Recovery"):
			net = conv2D_Module( net, kernel_size=1, stride=1, output_channel=output_channel, rate=rate, group=1,
			                     initializer              = initializer              ,
			                     is_constant_init         = is_constant_init         ,
			                     is_batch_norm            = is_batch_norm            ,
			                     is_dilated               = is_dilated               ,
			                     is_depthwise             = False                    ,
			                     is_ternary               = is_ternary               ,
			                     is_training              = is_training              ,
			                     is_testing               = is_testing               ,
			                     is_quantized_activation  = is_quantized_activation  ,
			                     IS_TERNARY               = IS_TERNARY               ,
			                     IS_QUANTIZED_ACTIVATION  = IS_QUANTIZED_ACTIVATION  ,
			                     Activation               = Activation               ,
			                     padding                  = padding                  ,
			                     Analysis                 = Analysis                 )
			
	return net	
	
# (Using Now)
def Pyramid_Pooling(net, strides, output_channel,
	is_training, 
	is_testing ):
	input_shape = net.get_shape().as_list()
	for level, stride in enumerate(strides):
		with tf.variable_scope('pool%d' %(level)):
			net_tmp = tf.nn.avg_pool(net, ksize=[1, stride, stride, 1], strides=[1, stride, stride, 1], padding='SAME')
			net_tmp = conv2D(net_tmp, kernel_size=1, stride=1, output_channel=output_channel, rate=1,
							is_bottleneck	= False, 
							is_batch_norm	= True, 
							is_training		= is_training, 
							is_testing		= is_testing, 
							is_dilated		= False, 
							scope			= "1x1")
			net_tmp = tf.image.resize_images(net_tmp, [input_shape[1], input_shape[2]])
		net = tf.concat([net, net_tmp], axis=3)
	return net
	
def ternarize_weights(float32_weights, ternary_weights_bd):
	ternary_weights = tf.multiply(tf.cast(tf.less_equal(float32_weights, ternary_weights_bd[0]), tf.float32), tf.constant(-1, tf.float32))
	ternary_weights = tf.add(ternary_weights, tf.multiply(tf.cast(tf.greater(float32_weights, ternary_weights_bd[1]), tf.float32), tf.constant( 1, tf.float32)))
	
	return ternary_weights
	
def ternarize_biases(float32_biases, ternary_biases_bd):
	ternary_biases = tf.multiply(tf.cast(tf.less_equal(float32_biases, ternary_biases_bd[0]), tf.float32), tf.constant(-1, tf.float32))
	ternary_biases = tf.add(ternary_biases, tf.multiply(tf.cast(tf.greater(float32_biases, ternary_biases_bd[1]), tf.float32), tf.constant( 1, tf.float32)))

	return ternary_biases

def quantize_activation(float32_activation):
	m = tf.get_variable("manssstissa", dtype=tf.float32, initializer=tf.constant([7], tf.float32))
	f = tf.get_variable("fraction"   , dtype=tf.float32, initializer=tf.constant([0], tf.float32))
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

# Conv2D Module
def quantize_Module(net, is_quantized_activation):
	quantized_net = quantize_activation(net)
	net = tf.cond(is_quantized_activation, lambda: quantized_net, lambda: net)
	
	tf.add_to_collection("is_quantized_activation", is_quantized_activation)
	tf.add_to_collection("final_net", net)
	
def shortcut_Module( net, 
                     destination             ,
			         initializer             ,
			         is_constant_init        , 
			         is_batch_norm	         ,
			         is_training		     ,
			         is_testing		         ,
			         is_ternary		         ,
			         is_quantized_activation ,
			         IS_TERNARY			     , 	
			         IS_QUANTIZED_ACTIVATION ,
			         padding				 ,			     
					 Analysis				 ):
	
	[batch,  input_height,  input_width,  input_channel] = net.get_shape().as_list()
	[batch, output_height, output_width, output_channel] = destination.get_shape().as_list()
	
	with tf.variable_scope("shortcut"):
		# Depth
		if input_channel!=output_channel:
			shortcut = conv2D_Module( net, kernel_size=1, stride=1, output_channel=output_channel, rate=1, group=1,
			                          initializer              = initializer              ,
			                          is_constant_init         = is_constant_init         ,
			                          is_batch_norm            = is_batch_norm            ,
			                          is_dilated               = False                    ,
			                          is_depthwise             = False                    ,
			                          is_ternary               = is_ternary               ,
			                          is_training              = is_training              ,
			                          is_testing               = is_testing               ,
			                          is_quantized_activation  = is_quantized_activation  ,
			                          IS_TERNARY               = IS_TERNARY               ,
			                          IS_QUANTIZED_ACTIVATION  = IS_QUANTIZED_ACTIVATION  ,
			                          Activation               = None                     ,
			                          padding                  = padding                  ,
			                          Analysis                 = Analysis                 )
		else:
			shortcut = net
			
		# Height & Width
		if input_height!=output_height or input_width!=output_width:
			shortcut = tf.image.resize_images( images = shortcut, 
			                                   size   = [tf.constant(output_height), tf.constant(output_width)])
		
	return shortcut

def bottleneck_Module( net, kernel_size, stride, internal_channel, output_channel, rate, group,
					   initializer             ,
					   is_constant_init        , 
					   is_batch_norm           , 
					   is_training             , 
					   is_testing              ,
					   is_dilated              , 
					   is_depthwise            ,
                       is_inception            ,
					   is_ternary              , 
					   is_quantized_activation , 
					   IS_TERNARY              ,  	
					   IS_QUANTIZED_ACTIVATION ,  
					   Activation              ,
					   padding                 ,
					   Analysis                ): 

	with tf.variable_scope("bottle_neck"):
		with tf.variable_scope("conv1_1x1"):
			net = conv2D_Module( net, kernel_size=1, stride=1, output_channel=internal_channel, rate=rate, group=1,
		                     initializer              = initializer              ,
		                     is_constant_init         = is_constant_init         ,
		                     is_batch_norm            = is_batch_norm            ,
		                     is_dilated               = is_dilated               ,
		                     is_depthwise             = False                    ,
		                     is_ternary               = is_ternary               ,
		                     is_training              = is_training              ,
		                     is_testing               = is_testing               ,
		                     is_quantized_activation  = is_quantized_activation  ,
		                     IS_TERNARY               = IS_TERNARY               ,
		                     IS_QUANTIZED_ACTIVATION  = IS_QUANTIZED_ACTIVATION  ,
		                     Activation               = Activation               ,
		                     padding                  = padding                  ,
		                     Analysis                 = Analysis                 )

		with tf.variable_scope("conv2_3x3"):
			if is_inception:
				net = inception_Module( net, kernel_size=kernel_size, stride=stride, output_channel=internal_channel, rate=rate, group=group,
			                            initializer               = initializer              ,
			                            is_constant_init          = is_constant_init         ,
			                            is_batch_norm             = is_batch_norm            ,
			                            is_dilated                = is_dilated               ,
			                            is_depthwise              = is_depthwise             ,
			                            is_ternary                = is_ternary               ,
			                            is_training               = is_training              ,
			                            is_testing                = is_testing               ,
			                            is_quantized_activation   = is_quantized_activation  ,
			                            IS_TERNARY                = IS_TERNARY               ,
			                            IS_QUANTIZED_ACTIVATION   = IS_QUANTIZED_ACTIVATION  ,
			                            Activation                = Activation               ,
			                            padding                   = padding                  ,
			                            Analysis                  = Analysis                 )
			else :
				net = conv2D_Module( net, kernel_size=kernel_size, stride=stride, output_channel=internal_channel, rate=rate, group=group,
		                             initializer              = initializer              ,
		                             is_constant_init         = is_constant_init         ,
		                             is_batch_norm            = is_batch_norm            ,
		                             is_dilated               = is_dilated               ,
		                             is_depthwise             = is_depthwise             ,
		                             is_ternary               = is_ternary               ,
		                             is_training              = is_training              ,
		                             is_testing               = is_testing               ,
		                             is_quantized_activation  = is_quantized_activation  ,
		                             IS_TERNARY               = IS_TERNARY               ,
		                             IS_QUANTIZED_ACTIVATION  = IS_QUANTIZED_ACTIVATION  ,
		                             Activation               = Activation               ,
		                             padding                  = padding                  ,
		                             Analysis                 = Analysis                 )
			
		with tf.variable_scope("conv3_1x1"):
			net = conv2D_Module( net, kernel_size=1, stride=stride, output_channel=output_channel, rate=rate, group=group,
		                     initializer              = initializer              ,
		                     is_constant_init         = is_constant_init         ,
		                     is_batch_norm            = is_batch_norm            ,
		                     is_dilated               = is_dilated               ,
		                     is_depthwise             = False                    ,
		                     is_ternary               = is_ternary               ,
		                     is_training              = is_training              ,
		                     is_testing               = is_testing               ,
		                     is_quantized_activation  = is_quantized_activation  ,
		                     IS_TERNARY               = IS_TERNARY               ,
		                     IS_QUANTIZED_ACTIVATION  = IS_QUANTIZED_ACTIVATION  ,
		                     Activation               = Activation               ,
		                     padding                  = padding                  ,
		                     Analysis                 = Analysis                 )
									
	return net
			
def inception_Module( net, kernel_size, stride, output_channel, rate, group,
			          initializer              ,
			          is_constant_init         ,
			          is_batch_norm            ,
			          is_dilated               ,
			          is_depthwise             ,
			          is_ternary               ,
			          is_training              ,
			          is_testing               ,
			          is_quantized_activation  ,
			          IS_TERNARY               ,
			          IS_QUANTIZED_ACTIVATION  ,
			          Activation               ,
			          padding                  ,
			          Analysis                 
				      ):
	with tf.variable_scope('inception'):
		with tf.variable_scope('conv1x1'):
			net_1x1 = conv2D_Module( net, kernel_size=1, stride=stride, output_channel=output_channel, rate=1, group=1,
									 initializer              = initializer              ,
									 is_constant_init         = is_constant_init         ,
									 is_batch_norm            = is_batch_norm            ,
									 is_dilated               = False                    ,
									 is_depthwise             = False                    ,
									 is_ternary               = is_ternary               ,
									 is_training              = is_training              ,
									 is_testing               = is_testing               ,
									 is_quantized_activation  = is_quantized_activation  ,
									 IS_TERNARY               = IS_TERNARY               ,
									 IS_QUANTIZED_ACTIVATION  = IS_QUANTIZED_ACTIVATION  ,
									 Activation               = Activation               ,
									 padding                  = padding                  ,
									 Analysis                 = Analysis                 )
		 
		net = conv2D_Module( net, kernel_size=kernel_size, stride=stride, output_channel=output_channel, rate=rate, group=group,
							 initializer              = initializer              ,
							 is_constant_init         = is_constant_init         ,
							 is_batch_norm            = is_batch_norm            ,
							 is_dilated               = is_dilated               ,
							 is_depthwise             = is_depthwise             ,
							 is_ternary               = is_ternary               ,
							 is_training              = is_training              ,
							 is_testing               = is_testing               ,
							 is_quantized_activation  = is_quantized_activation  ,
							 IS_TERNARY               = IS_TERNARY               ,
							 IS_QUANTIZED_ACTIVATION  = IS_QUANTIZED_ACTIVATION  ,
							 Activation               = Activation               ,
							 padding                  = padding                  ,
							 Analysis                 = Analysis                 )
		if is_depthwise:
			with tf.variable_scope("depthwise_conv1x1"):
				net = conv2D_Module( net, kernel_size=1, stride=1, output_channel=output_channel, rate=rate, group=1,
							         initializer              = initializer              ,
							         is_constant_init         = is_constant_init         ,
							         is_batch_norm            = is_batch_norm            ,
							         is_dilated               = False                    ,
							         is_depthwise             = False                    ,
							         is_ternary               = is_ternary               ,
							         is_training              = is_training              ,
							         is_testing               = is_testing               ,
							         is_quantized_activation  = is_quantized_activation  ,
							         IS_TERNARY               = IS_TERNARY               ,
							         IS_QUANTIZED_ACTIVATION  = IS_QUANTIZED_ACTIVATION  ,
							         Activation               = Activation               ,
							         padding                  = padding                  ,
							         Analysis                 = Analysis                 )
		net = tf.add(net, net_1x1)
		
		# Activation Quantization
		if IS_QUANTIZED_ACTIVATION:
			quantize_Module(net, is_quantized_activation)
			
	return net

def conv2D_Module( net, kernel_size, stride, output_channel, rate, group,
			       initializer              ,
			       is_constant_init         ,
			       is_batch_norm            ,
			       is_dilated               ,
			       is_depthwise             ,
			       is_ternary               ,
			       is_training              ,
			       is_testing               ,
			       is_quantized_activation  ,
			       IS_TERNARY               ,
			       IS_QUANTIZED_ACTIVATION  ,
			       Activation               ,
			       padding                  ,
			       Analysis                 
				   ):
	
	input_channel = net.get_shape().as_list()[-1]

	if not is_depthwise:
		group=1
	
	##   Analyzer   ##
	Analysis = Analyzer(Analysis, 
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
					net_tmp = tf.nn.depthwise_conv2d(net, weights, strides=[1, stride, stride, 1], padding=padding, rate=[rate, rate])	
				else:
					net_tmp = tf.nn.atrous_conv2d(net, weights, rate=rate, padding=padding)
			else:
				if is_depthwise:
					net_tmp = tf.nn.depthwise_conv2d(net, weights, strides=[1, stride, stride, 1], padding=padding)	
				else:
					net_tmp = tf.nn.conv2d(net, weights, strides=[1, stride, stride, 1], padding=padding)
			
			# add bias
			net_tmp = tf.nn.bias_add(net_tmp, biases)

			# Ternary Scale
			if IS_TERNARY:
				with tf.variable_scope('Ternary_Scalse'):
					weights = tf.get_variable("weights", [1], tf.float32, initializer=initializer)
					net_tmp = tf.multiply(net_tmp, weights)
					
			# Batch Normalization
			if is_batch_norm == True:
				net_tmp = batch_norm(net_tmp, is_training, is_testing)
			
			# Activation
			if Activation == 'ReLU':
				net_tmp = tf.nn.relu(net_tmp)
			elif Activation == 'Sigmoid':
				net_tmp = tf.nn.sigmoid(net_tmp)
			else:
				net_tmp = net_tmp
			
			# Activation Quantization
			if IS_QUANTIZED_ACTIVATION:
				quantize_Module(net_tmp, is_quantized_activation)
				
			# merge every group net together
			if g==0:
				net = net_tmp
			else:
				#net = tf.concat([net, net_tmp], axis=3)
				net = tf.add(net, net_tmp)
		
	# Activation Quantization
	if IS_QUANTIZED_ACTIVATION and group>1:
		quantize_Module(net, is_quantized_activation)
		
	return net
	
# 
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
			float32_weights = tf.get_variable("weights", [kernel_size, kernel_size, input_channel,              1], tf.float32, initializer=initializer)
		else:
			float32_weights = tf.get_variable("weights", [kernel_size, kernel_size, input_channel, output_channel], tf.float32, initializer=initializer)
	
	if is_depthwise:
		float32_biases = tf.Variable(tf.constant(0.0, shape=[input_channel], dtype=tf.float32), trainable=True, name='biases')
	else:
		float32_biases = tf.Variable(tf.constant(0.0, shape=[output_channel], dtype=tf.float32), trainable=True, name='biases')
	
	if IS_TERNARY:
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
			weights = tf.get_variable("final_weights", [kernel_size, kernel_size, input_channel, 1], tf.float32, initializer=initializer)
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
            initializer             =tf.contrib.layers.variance_scaling_initializer(),
            is_constant_init        = False,      # For using constant value as weight initial; Only valid in Normal Convolution
            is_bottleneck           = False,      # For Residual
            is_batch_norm           = True,       # For Batch Normalization
            is_dilated              = False,      # For Dilated Convoution
            is_depthwise            = False,      # For Depthwise Convolution
			is_inception            = False,      # For Inception Convolution
            is_ternary              = False,      # (tensor) For weight ternarization
            is_training             = True,       # (tensor) For Batch Normalization
            is_testing              = False,      # (tensor) For getting the pretrained from caffemodel
            is_quantized_activation = False,      # (tensor) For activation quantization
            IS_TERNARY              = False,      
            IS_QUANTIZED_ACTIVATION = False,      
            Activation              = 'ReLU',
            padding                 = "SAME",
            Analysis                = None,
            scope                   = "conv"):
		
	with tf.variable_scope(scope):
		
		#===============================#
		#   Bottleneck Residual Block   #
		#===============================#
		if is_bottleneck:
			net = bottleneck_Module( net, kernel_size=kernel_size, stride=stride, internal_channel=internal_channel, output_channel=output_channel, rate=rate, group=group,
							         initializer             = initializer             ,
							         is_constant_init        = is_constant_init        , 
							         is_batch_norm           = is_batch_norm           , 
							         is_training             = is_training             , 
							         is_testing              = is_testing              , 
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

		#============================#
		#   Normal Convolution Block #
		#============================#
		else:  
			if is_inception:
				net = inception_Module( net=net, kernel_size=kernel_size, stride=stride, output_channel=output_channel, rate=rate, group=group,
			                            initializer              = initializer              ,
			                            is_constant_init         = is_constant_init         ,
			                            is_batch_norm            = is_batch_norm            ,
			                            is_dilated               = is_dilated               ,
			                            is_depthwise             = is_depthwise             ,
			                            is_ternary               = is_ternary               ,
			                            is_training              = is_training              ,
			                            is_testing               = is_testing               ,
			                            is_quantized_activation  = is_quantized_activation  ,
			                            IS_TERNARY               = IS_TERNARY               ,
			                            IS_QUANTIZED_ACTIVATION  = IS_QUANTIZED_ACTIVATION  ,
			                            Activation               = Activation               ,
			                            padding                  = padding                  ,
			                            Analysis                 = Analysis                 )
			else:
				net = conv2D_Module( net, kernel_size=kernel_size, stride=stride, output_channel=output_channel, rate=rate, group=group,
									 initializer              = initializer              ,
									 is_constant_init         = is_constant_init         ,
									 is_batch_norm            = is_batch_norm            ,
									 is_dilated               = is_dilated               ,
									 is_depthwise             = is_depthwise             ,
									 is_ternary               = is_ternary               ,
									 is_training              = is_training              ,
									 is_testing               = is_testing               ,
									 is_quantized_activation  = is_quantized_activation  ,
									 IS_TERNARY               = IS_TERNARY               ,
									 IS_QUANTIZED_ACTIVATION  = IS_QUANTIZED_ACTIVATION  ,
									 Activation               = Activation               ,
									 padding                  = padding                  ,
									 Analysis                 = Analysis                 )
				if is_depthwise:
					with tf.variable_scope("depthwise_conv1x1"):
						net = conv2D_Module( net, kernel_size=1, stride=1, output_channel=output_channel, rate=rate, group=1,
											initializer              = initializer              ,
											is_constant_init         = is_constant_init         ,
											is_batch_norm            = is_batch_norm            ,
											is_dilated               = False                    ,
											is_depthwise             = False                    ,
											is_ternary               = is_ternary               ,
											is_training              = is_training              ,
											is_testing               = is_testing               ,
											is_quantized_activation  = is_quantized_activation  ,
											IS_TERNARY               = IS_TERNARY               ,
											IS_QUANTIZED_ACTIVATION  = IS_QUANTIZED_ACTIVATION  ,
											Activation               = Activation               ,
											padding                  = padding                  ,
											Analysis                 = Analysis                 )
	return net

def indice_pool(net, kernel_size, stride, Analysis, IS_QUANTIZED_ACTIVATION, scope="Pool"):
	with tf.variable_scope(scope):
		output_shape = net.get_shape().as_list()
		################
		#   Analyzer   #
		################
		Analysis = Analyzer(Analysis, 
		                    net, 
							type                    = 'POOL', 
							kernel_shape            = [stride, stride, output_shape[3], output_shape[3]], 
							stride                  = stride, 
							IS_QUANTIZED_ACTIVATION = IS_QUANTIZED_ACTIVATION,
							name                    = 'Pool')
		
		net, indices = tf.nn.max_pool_with_argmax( 
			input=net, 
			ksize=[1, kernel_size, kernel_size, 1],
			strides=[1, stride, stride, 1],
			padding="SAME",
			Targmax=None,
			name=None
		)

	return net, indices, output_shape
	
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
				
def batch_norm(net, is_training, is_testing):
	with tf.variable_scope("Batch_Norm"):
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
		   			 variables_collections	= None,
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

#=========================#
#   Training Components   #
#=========================#
# (No Use Now)
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

def quantizing_weight_and_biases(weights_collection, 
							     biases_collection,
							     weights_bd,
							     biases_bd,
							     weights_table,
							     biases_table,
							     sess):
	NUM = len(weights_collection)
	for i in range(NUM):
		w         = sess.run(weights_collection[i])
		b         = sess.run(biases_collection [i])

		w_bd      = weights_bd[i]
		b_bd      = biases_bd [i]

		index_num = len(w_bd)
		

		for j in range(index_num):
			if j==0:
				quantized_weights =(w <= w_bd[j]) * weights_table[j]
				quantized_biases  =(b <= b_bd[j]) * biases_table   [j]
			else:
				quantized_weights = quantized_weights + ((w <= w_bd[j]) & (w > w_bd[j-1])) * weights_table[j]
				quantized_biases  = quantized_biases  + ((b <= b_bd[j]) & (b > b_bd[j-1])) * biases_table [j]
			if j==(index_num-1):
				quantized_weights = quantized_weights + (w > w_bd[j]) * weights_table[j+1]
				quantized_biases  = quantized_biases  + (b > b_bd[j]) * biases_table [j+1]


		sess.run(weights_collection[i].assign(quantized_weights))
		sess.run( biases_collection[i].assign(quantized_biases ))

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
	
# (Using Now)
def one_of_k(target, class_num):
	target.astype('int64')
	one_of_k_target = np.zeros([np.shape(target)[0], np.shape(target)[1], np.shape(target)[2], class_num])
	
	meshgrid_target = np.meshgrid(np.arange(np.shape(target)[1]), np.arange(np.shape(target)[0]), np.arange(np.shape(target)[2]))
	
	one_of_k_target[meshgrid_target[1], meshgrid_target[0], meshgrid_target[2], target] = 1
	
	return one_of_k_target

def shuffle_image(x, y):
	[NUM, HEIGHT, WIDTH, DIM] = np.shape(x)
	
	shuffle_index = np.arange(NUM)
	np.random.shuffle(shuffle_index)
	
	#index_shuffle = index[shuffle_index]
	x_shuffle = x[shuffle_index, :, :, :]
	y_shuffle = y[shuffle_index, :, :, :]
	
	return x_shuffle, y_shuffle

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

# Activation Quantization
def quantized_m_and_f(activation_collection, is_quantized_activation_collection, xs, is_training, is_testing, Model_dict, batch_xs, sess):
	#is_quantized_activation = []
	#for layer in range(len(Model_dict)):
	#	if Model_dict['layer'+str(layer)]['IS_QUANTIZED_ACTIVATION']=='TRUE':
	#		is_quantized_activation.append(Model_dict['layer'+str(layer)]['is_quantized_activation'])
			
	NUM = len(activation_collection)
	for i in range(NUM):
		activation = sess.run(activation_collection[i], feed_dict={xs: batch_xs, is_training: False, is_testing: True, is_quantized_activation_collection[i]: False})
		var = np.var(activation)+1e-4

		mantissa = 16
		fraction = -int(np.log2((var*3)/pow(2, mantissa)))-1
		#fraction = 3

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

# Weight Ternary
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

#========================#
#   Testing Components   #
#========================#
# (No Use Now)
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

# (Using Now)
def compute_accuracy(xs, ys, is_training, is_testing, is_validation, Model_dict, QUANTIZED_NOW, prediction, v_xs, v_ys, BATCH_SIZE, sess):
	test_batch_size = BATCH_SIZE
	batch_num = len(v_xs) / test_batch_size
	result_accuracy_top1 = 0
	result_accuracy_top2 = 0
	result_accuracy_top3 = 0
	for i in range(int(batch_num)):	
		v_xs_part = v_xs[i*test_batch_size : (i+1)*test_batch_size, :]
		v_ys_part = v_ys[i*test_batch_size : (i+1)*test_batch_size, :]
			
		dict_inputs = [xs       , is_training, is_testing         ]
		dict_data   = [v_xs_part, False      , (not is_validation)]
		for layer in range(len(Model_dict)):
			dict_inputs.append(Model_dict['layer'+str(layer)]['is_quantized_activation'])
			dict_data.append(Model_dict['layer'+str(layer)]['IS_QUANTIZED_ACTIVATION'] and QUANTIZED_NOW)
				
		Y_pre = sess.run(prediction, feed_dict={i: d for i, d in zip(dict_inputs, dict_data)})				
		# for post-processing
		if i==0:
			SegNet_Y_pre = Y_pre;
			#SegNet_Y_pre_annot = v_ys_part
		else:
			SegNet_Y_pre = np.concatenate([SegNet_Y_pre, Y_pre], axis=0)
	    	#SegNet_Y_pre_annot = np.concatenate([SegNet_Y_pre_annot, v_ys_part], axis=0)

		top1 = np.argsort(-Y_pre, axis=-1)[:, :, :, 0] 
		top2 = np.argsort(-Y_pre, axis=-1)[:, :, :, 1] 
		top3 = np.argsort(-Y_pre, axis=-1)[:, :, :, 2] 
		
		correct_prediction_top1 = np.equal(top1, np.argmax(v_ys_part, -1))
		correct_prediction_top2 = np.equal(top2, np.argmax(v_ys_part, -1)) | correct_prediction_top1
		correct_prediction_top3 = np.equal(top3, np.argmax(v_ys_part, -1)) | correct_prediction_top2
		
		accuracy_top1 = np.mean(correct_prediction_top1.astype(float))
		accuracy_top2 = np.mean(correct_prediction_top2.astype(float))
		accuracy_top3 = np.mean(correct_prediction_top3.astype(float))
		
		#print("\033[1;32;40mtop1 Accuracy\033[0m = {Accuracy_top1}, \033[1;32;40mtop2 Accuracy\033[0m = {Accuracy_top2}, \033[1;32;40mtop3 Accuracy\033[0m = {Accuracy_top3}"
		#	.format(iter = i, Accuracy_top1 = accuracy_top1, Accuracy_top2 = accuracy_top2, Accuracy_top3 = accuracy_top3))
		
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

def Save_result_as_image(Path, result, file_index):
	for i, target in enumerate(result):
		scipy.misc.imsave(Path + file_index[i], target)
	
def Save_result_as_npz(	
	Path, result, file_index,
	#--------------------------#
	# Saving Each Layer Result #
	#--------------------------#
	# tensor
	xs,
	is_training,
	is_testing,              
	is_quantized_activation,
	# data
	v_xs,
	# Parameter
	QUANTIZED_NOW,
	H_resize,
	W_resize,
	sess,
	last_x_layer = 5
	):

	for i, target in enumerate(result):
		file_index[i] = file_index[i].split('.')[0]
		np.savez(Path + file_index[i] + '_' + H_resize + '_' + W_resize, target)
	
#============#
#   others   #
#============#
#(No Use Now)
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

def calculate_class_weight(y):
	[NUM, H, W, DIM] = np.shape(y)
	class_num = np.zeros([DIM])
	for i in range(DIM):
		class_num[i] = np.count_nonzero(np.equal(np.argmax(y, -1), i))
	class_num = 1 / class_num
	total = np.sum(class_num)
	return class_num / total
			
# (Using Now)	
	

