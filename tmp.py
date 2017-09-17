import tensorflow as tf
import numpy as np
import scipy.ndimage
import pdb
import math
from PIL import Image
from scipy import misc
import utils
import SegNet

#=======================#
#	Global Parameter	#
#=======================#
BATCH_SIZE = 4
EPOCH_TIME = 30
IS_TRAINING = False
IS_TESTING = True
if IS_TESTING:	
	BATCH_SIZE = 1
K = 6 # TOP "K" accuracy class predicted by SegNet

LEARNING_RATE = 0.01
EPOCH_DECADE = 100
LR_DECADE = 10
LAMBDA = 0.001

#=======================#
#	Training File Name	#
#=======================#
if LEARNING_RATE <= 1e-5:
	TRAINING_WEIGHT_FILE = 'SegNet_Model/SegNet_TOP_K_0' + str(LEARNING_RATE).split('.')[0]
else:
	TRAINING_WEIGHT_FILE = 'SegNet_Model/SegNet_TOP_K_0' + str(LEARNING_RATE).split('.')[1]
TRAINING_WEIGHT_FILE = TRAINING_WEIGHT_FILE + '_epoch' + str(EPOCH_DECADE)
TRAINING_WEIGHT_FILE = TRAINING_WEIGHT_FILE + '_divide' + str(LR_DECADE)
TRAINING_WEIGHT_FILE = TRAINING_WEIGHT_FILE + '_L20' + str(LAMBDA).split('.')[1]	


#=======================#
#	Testing File Name	#
#=======================#
if LEARNING_RATE <= 1e-5:
	TESTING_WEIGHT_FILE = 'SegNet_Model/SegNet_TOP_K_0' + str(LEARNING_RATE).split('.')[0]
else:
	TESTING_WEIGHT_FILE = 'SegNet_Model/SegNet_TOP_K_0' + str(LEARNING_RATE).split('.')[1]
TESTING_WEIGHT_FILE = TESTING_WEIGHT_FILE + '_epoch' + str(EPOCH_DECADE)
TESTING_WEIGHT_FILE = TESTING_WEIGHT_FILE + '_divide' + str(LR_DECADE)
TESTING_WEIGHT_FILE = TESTING_WEIGHT_FILE + '_L20' + str(LAMBDA).split('.')[1]
TESTING_WEIGHT_FILE = TESTING_WEIGHT_FILE + '_' + str(EPOCH_TIME)
TESTING_WEIGHT_FILE = TESTING_WEIGHT_FILE + '.npz'


#===========# 
#	Define	#
#===========#
def tmp(net, is_training, is_testing, reuse=None, scope="tmp"):
	with tf.variable_scope(scope, reuse=reuse):
		net = utils.conv2D(net, kernel_size=3, stride=1, internal_channel=64, output_channel=64, rate=1,
#			initializer=tf.zeros([3, 3, 3, 64], tf.float32),
			is_contant_init=False, # For using constant value as weight initial
			is_shortcut=False, # For Residual
			is_bottleneck=False, # For Residual
			is_batch_norm=True, # For Batch Normalization
			is_training=is_training, # For Batch Normalization
			is_testing=is_testing, # For getting the pretrained from caffemodel
			is_dilated=False, # For Dilated Convolution
			padding="SAME",
			scope="conv1")

		net = utils.conv2D(net, kernel_size=3, stride=1, internal_channel=64, output_channel=64, rate=2,
#			initializer=tf.zeros([3, 3, 64, 64], tf.float32),
			is_contant_init=False, # For using constant value as weight initial
			is_shortcut=False, # For Residual
			is_bottleneck=False, # For Residual
			is_batch_norm=True, # For Batch Normalization
			is_training=is_training, # For Batch Normalization
			is_testing=is_testing, # For getting the pretrained from caffemodel
			is_dilated=True, # For Dilated Convolution
			padding="SAME",
			scope="conv2")

		net = utils.conv2D(net, kernel_size=3, stride=1, internal_channel=64, output_channel=K+1, rate=1,
#			initializer=tf.concat([tf.ones([3, 3, 64, 1], tf.float32), tf.zeros([3, 3, 64, K-1], tf.float32)], axis=-1),
			is_contant_init=False, # For using constant value as weight initial
			is_shortcut=False, # For Residual
			is_bottleneck=False, # For Residual
			is_batch_norm=False, # For Batch Normalization
			is_training=is_training, # For Batch Normalization
			is_testing=is_testing, # For getting the pretrained from caffemodel
			is_dilated=False, # For Dilated Convolution
			padding="SAME",
			scope="conv3")
	return net
	
def main(argv=None):
#===============#
#	File Read	#
#===============#
	CamVid_Path = "/home/2016/b22072117/ObjectSegmentation/codes/dataset/CamVid"
	SegNet_Y_pre_Path = '/home/2016/b22072117/ObjectSegmentation/codes/nets/SegNet_Y_pre/student' 
# Training file
	print("")
	print("Loading Training Data ...")
	# CamVid
	CamVid_train_data, _ = utils.read_dataset_file(CamVid_Path, '/train.txt')
	CamVid_train_target, _ = utils.read_dataset_file(CamVid_Path, '/trainannot.txt')

	# SegNet Prediction Output
	SegNet_train_data, SegNet_train_data_index = utils.read_Y_pre_file(SegNet_Y_pre_Path, '/train.txt')

	class_num = np.max(CamVid_train_target)+1
	CamVid_train_target = utils.one_of_k(CamVid_train_target, class_num)
	SegNet_train_target = CamVid_train_target
	
	print("Shape of train data	: {Shape}" .format(Shape = np.shape(CamVid_train_data)))
	print("Shape of train target	: {Shape}" .format(Shape = np.shape(CamVid_train_target)))
	
# Validation file
	print("")
	print("Loading Validation Data ...")
	# CamVid
	CamVid_valid_data, _ = utils.read_dataset_file(CamVid_Path, '/val.txt')
	CamVid_valid_target, _ = utils.read_dataset_file(CamVid_Path, '/valannot.txt')

	# SegNet Prediction output 
	SegNet_valid_data, SegNet_valid_data_index = utils.read_Y_pre_file(SegNet_Y_pre_Path, '/val.txt')

	CamVid_valid_target = utils.one_of_k(CamVid_valid_target, class_num)
	SegNet_valid_target = CamVid_valid_target
	print("Shape of valid data	: {Shape}" .format(Shape = np.shape(CamVid_valid_data)))
	print("Shape of valid target	: {Shape}" .format(Shape = np.shape(CamVid_valid_target)))

# Testing file
	print("")
	print("Loading Testing Data ...")
	# CamVid 
	CamVid_test_data, _ = utils.read_dataset_file(CamVid_Path, '/test.txt')
	CamVid_test_target, _ = utils.read_dataset_file(CamVid_Path, '/testannot.txt')
	
	# SegNet Preidict output
	SegNet_test_data, SegNet_test_data_index = utils.read_Y_pre_file(SegNet_Y_pre_Path, '/test.txt')
	
	CamVid_test_target = utils.one_of_k(CamVid_test_target, class_num)
	SegNet_test_target = CamVid_test_target
	print("Shape of test data	: {Shape}" .format(Shape = np.shape(CamVid_test_data)))
	print("Shape of test target	: {Shape}" .format(Shape = np.shape(CamVid_test_target)))
	print("")

#=======================#
#	Data Preprocessing	#
#=======================#
	# Get TOP K Data
	SegNet_train_TOP_K_data, SegNet_train_TOP_K_target, SegNet_train_TOP_K_table = utils.get_max_K_class(SegNet_train_data, SegNet_train_target, K)
	SegNet_valid_TOP_K_data, SegNet_valid_TOP_K_target, SegNet_valid_TOP_K_table = utils.get_max_K_class(SegNet_valid_data, SegNet_valid_target, K)
	SegNet_test_TOP_K_data , SegNet_test_TOP_K_target , SegNet_test_TOP_K_table  = utils.get_max_K_class(SegNet_test_data , SegNet_test_target , K)
	
#===========#
#	Data	#
#===========#
# Placeholder
	data_shape = np.shape(SegNet_train_TOP_K_data)
	xs = tf.placeholder(tf.float32, [BATCH_SIZE, data_shape[1], data_shape[2], data_shape[3]]) 
	ys = tf.placeholder(tf.float32, [BATCH_SIZE, data_shape[1], data_shape[2], K+1])
	learning_rate = tf.placeholder(tf.float32)
	is_training = tf.placeholder(tf.bool)
	is_testing = tf.placeholder(tf.bool)
	
# Data Preprocessing
	xImage = xs
	
#===========#
#	Graph	#
#===========#
	net = xImage
	prediction = tmp(net, is_training=is_training, is_testing=is_testing, reuse=None, scope="tmp")
	

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
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(ys, -1), logits=prediction)
	l2_norm = tf.reduce_mean(tf.stack([tf.nn.l2_loss(x) for x in weights_collection]))
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
			'conv1_3', 'conv1_3_b']
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
		# Loading Pre-trained Model
			#print ""
			#print("Loading Pre-trained weights ...")
			#utils.load_pre_trained_weights(parameters, pre_trained_weight_file=TESTING_WEIGHT_FILE, sess=sess) 
			
		# Training & Validation
			for epoch in range(EPOCH_TIME):
				SegNet_train_TOP_K_data, SegNet_train_TOP_K_target, SegNet_train_TOP_K_table, SegNet_train_target = utils.shuffle_image_TOP_K(SegNet_train_TOP_K_data, SegNet_train_TOP_K_target, SegNet_train_TOP_K_table, SegNet_train_target)
				print("")
				print("Training ... ")
				for i in range(data_shape[0]/BATCH_SIZE):
				
					batch_xs    = SegNet_train_TOP_K_data   [i*BATCH_SIZE : (i+1)*BATCH_SIZE]
					batch_ys    = SegNet_train_TOP_K_target [i*BATCH_SIZE : (i+1)*BATCH_SIZE]
					batch_table = SegNet_train_TOP_K_table  [i*BATCH_SIZE : (i+1)*BATCH_SIZE]
					
					real_batch_ys = SegNet_train_target	[i*BATCH_SIZE : (i+1)*BATCH_SIZE]

					_, CrossEntropy, Prediction = sess.run([train_step, cross_entropy, prediction], 
						feed_dict={xs: batch_xs, ys: batch_ys, learning_rate: lr, is_training: True, is_testing: False})
					
														
					y_pre = utils.get_real_prediction(Prediction, batch_table) 
					batch_accuracy = np.mean(np.equal(np.argmax(real_batch_ys, -1), y_pre))
					
					print("Epoch : {ep}" .format(ep = epoch))
					print("Batch Iteration : {Iter}" .format(Iter = i))
					print("  Batch Accuracy : {acc}".format(acc = batch_accuracy))
					print("  Cross Entropy : {CrossEntropy}".format(CrossEntropy = np.mean(CrossEntropy)))
					print("  Learning Rate : {LearningRate}".format(LearningRate = lr))
					

					utils.per_class_accuracy_TOP_K(Prediction, real_batch_ys, batch_table)
					
			# Validation
				print("")
				print("Epoch : {ep}".format(ep = epoch))
				print("Validation ... ")
				is_validation = True
				_, valid_accuracy = utils.compute_accuracy_TOP_K(xs, ys, is_training, is_testing, is_validation, prediction, SegNet_valid_TOP_K_data, SegNet_valid_target,SegNet_valid_TOP_K_table, BATCH_SIZE, sess)
				is_validation = False
				print("")
				print("Validation Accuracy = {Valid_Accuracy}".format(Valid_Accuracy=valid_accuracy))
				
				if ((epoch%EPOCH_DECADE==0) and (epoch!=0)):
					lr = lr / LR_DECADE
					
			# Save trained weights
				if (((epoch+1)%10==0) and ((epoch+1)>=10)):
					print("Saving Trained Weights ... ")
					batch_xs = SegNet_train_TOP_K_data[0 : BATCH_SIZE]
					utils.assign_trained_mean_and_var(mean_collection, var_collection, trained_mean_collection, trained_var_collection, params, 
												xs, ys, is_training, is_testing, batch_xs, sess)
					utils.save_pre_trained_weights( (TRAINING_WEIGHT_FILE+'_'+str(epoch+1)), parameters, xs, batch_xs, sess)
		
	
		if IS_TESTING == True:
			is_validation = False

		# Load trained weights and save all the results of the wanted image
			print("")
			print("Loading the trained weights ... ")
			#utils.load_pre_trained_weights(parameters, pre_trained_weight_file=TESTING_WEIGHT_FILE, sess=sess) 
			
			print("Saving the train data result as image ... ")
			train_target_path = '/home/2016/b22072117/ObjectSegmentation/codes/result/CamVid/SegNet/'
			train_result, train_accuracy = utils.compute_accuracy_TOP_K(xs, ys, is_training, is_testing, is_validation, prediction, SegNet_train_TOP_K_data, SegNet_train_target, SegNet_train_TOP_K_table, BATCH_SIZE, sess)
			#train_result = utils.color_result(train_result)
			#utils.Save_result_as_image(train_target_path, train_result, SegNet_train_data_index)
			print("Train Data Accuracy = {Train_Accuracy}"
			.format(Train_Accuracy=train_accuracy))
			
			print("Saving the validation data result as image ... ")
			valid_target_path = '/home/2016/b22072117/ObjectSegmentation/codes/result/CamVid/SegNet/'
			valid_result, valid_accuracy = utils.compute_accuracy_TOP_K(xs, ys, is_training, is_testing, is_validation, prediction, SegNet_valid_TOP_K_data, SegNet_valid_target, SegNet_valid_TOP_K_table, BATCH_SIZE, sess)
			#valid_result = utils.color_result(valid_result)
			#utils.Save_result_as_image(valid_target_path, valid_result, SegNet_valid_data_index)
			print("Valid Data Accuracy = {Valid_Accuracy}"
			.format(Valid_Accuracy=valid_accuracy))
			
			print("Saving the test data result as image ... ")
			test_target_path = '/home/2016/b22072117/ObjectSegmentation/codes/result/CamVid/SegNet/'
			test_result, test_accuracy = utils.compute_accuracy_TOP_K(xs, ys, is_training, is_testing, is_validation, prediction, SegNet_test_TOP_K_data, SegNet_test_target, SegNet_test_TOP_K_table, BATCH_SIZE, sess)
			#test_result = utils.color_result(test_result)
			#utils.Save_result_as_image(test_target_path, test_result, SegNet_test_data_index)
			print("Test Data Accuracy = {Test_Accuracy}"
			.format(Test_Accuracy=test_accuracy))
			
			print("")
			
			print("Train Data Accuracy = {Train_Accuracy}"
			.format(Train_Accuracy=train_accuracy))
			print("Valid Data Accuracy = {Valid_Accuracy}"
			.format(Valid_Accuracy=valid_accuracy))
			print("Test Data Accuracy = {Test_Accuracy}"
			.format(Test_Accuracy=test_accuracy))
			
		print("")
		print("Works are All Done !")
if __name__ == "__main__":
	tf.app.run()

