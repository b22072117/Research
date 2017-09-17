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
BATCH_SIZE = 1
EPOCH_TIME = 100
IS_TRAINING = False
IS_TESTING = True
if IS_TESTING:	
	BATCH_SIZE = 1
T = 5 # Softer output for student 

LEARNING_RATE = 1e-4
EPOCH_DECADE = 100
LR_DECADE = 10
LAMBDA = 0.001

PRE_TRAINED_WEIGHT_FILE = 'segnet.npz'


#=======================#
#	Training File Name	#
#=======================#
if LEARNING_RATE <= 1e-5:
	TRAINING_WEIGHT_FILE = 'SegNet_Model/SegNet0' + str(LEARNING_RATE).split('.')[0]
else:
	TRAINING_WEIGHT_FILE = 'SegNet_Model/SegNet0' + str(LEARNING_RATE).split('.')[1]
TRAINING_WEIGHT_FILE = TRAINING_WEIGHT_FILE + '_epoch' + str(EPOCH_DECADE)
TRAINING_WEIGHT_FILE = TRAINING_WEIGHT_FILE + '_divide' + str(LR_DECADE)
TRAINING_WEIGHT_FILE = TRAINING_WEIGHT_FILE + '_L20' + str(LAMBDA).split('.')[1]	

#=======================#
#	Testing File Name	#
#=======================#
if LEARNING_RATE <= 1e-5:
	TESTING_WEIGHT_FILE = 'SegNet_Model/SegNet0' + str(LEARNING_RATE).split('.')[0]
else:
	TESTING_WEIGHT_FILE = 'SegNet_Model/SegNet0' + str(LEARNING_RATE).split('.')[1]
TESTING_WEIGHT_FILE = TESTING_WEIGHT_FILE + '_epoch' + str(EPOCH_DECADE)
TESTING_WEIGHT_FILE = TESTING_WEIGHT_FILE + '_divide' + str(LR_DECADE)
TESTING_WEIGHT_FILE = TESTING_WEIGHT_FILE + '_L20' + str(LAMBDA).split('.')[1]
TESTING_WEIGHT_FILE = TESTING_WEIGHT_FILE + '_' + str(EPOCH_TIME)
TESTING_WEIGHT_FILE = TESTING_WEIGHT_FILE + '.npz'

	
def SegNet_VGG_16(net, class_num, is_training, is_testing, reuse=None, scope="SegNet_VGG_16"):
	with tf.variable_scope(scope, reuse=reuse):
		with tf.variable_scope("encoder"):
			with tf.variable_scope("224X224"): # 1/1
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut=False, is_bottleneck=False, is_batch_norm=True, 
							is_training=is_training, is_testing=is_testing, is_dilated=False, scope="conv1")
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut=False, is_bottleneck=False, is_batch_norm=True, 
							is_training=is_training, is_testing=is_testing, is_dilated=False, scope="conv2")		
				net, indices1, output_shape1 = utils.indice_pool(net, stride=2, scope="Pool1")
				
			with tf.variable_scope("112X112"): # 1/2
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=128, rate=1,
							is_shortcut=False, is_bottleneck=False, is_batch_norm=True, 
							is_training=is_training, is_testing=is_testing, is_dilated=False, scope="conv1")
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=128, rate=1,
							is_shortcut=False, is_bottleneck=False, is_batch_norm=True, 
							is_training=is_training, is_testing=is_testing, is_dilated=False, scope="conv2")	
				net, indices2, output_shape2 = utils.indice_pool(net, stride=2, scope="Pool2")
				
			with tf.variable_scope("56X56"): # 1/4
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=256, rate=1, 
							is_shortcut=False, is_bottleneck=False, is_batch_norm=True, 
							is_training=is_training, is_testing=is_testing, is_dilated=False, scope="conv1")
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=256, rate=1,
							is_shortcut=False, is_bottleneck=False, is_batch_norm=True, 
							is_training=is_training, is_testing=is_testing, is_dilated=False, scope="conv2")	
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=256, rate=1,
							is_shortcut=False, is_bottleneck=False, is_batch_norm=True, 
							is_training=is_training, is_testing=is_testing, is_dilated=False, scope="conv3")			
				net, indices3, output_shape3 = utils.indice_pool(net, stride=2, scope="Pool3")
				
			with tf.variable_scope("28X28"): # 1/8
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=512, rate=1,
							is_shortcut=False, is_bottleneck=False, is_batch_norm=True, 
							is_training=is_training, is_testing=is_testing, is_dilated=False, scope="conv1")
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=512, rate=1,
							is_shortcut=False, is_bottleneck=False, is_batch_norm=True, 
							is_training=is_training, is_testing=is_testing, is_dilated=False, scope="conv2")	
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=512, rate=1,
							is_shortcut=False, is_bottleneck=False, is_batch_norm=True, 
							is_training=is_training, is_testing=is_testing, is_dilated=False, scope="conv3")			
				net, indices4, output_shape4 = utils.indice_pool(net, stride=2, scope="Pool4")
				
			with tf.variable_scope("14X14"): # 1/16
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=512, rate=1,
							is_shortcut=False, is_bottleneck=False, is_batch_norm=True, 
							is_training=is_training, is_testing=is_testing, is_dilated=False, scope="conv1")
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=512, rate=1,
							is_shortcut=False, is_bottleneck=False, is_batch_norm=True, 
							is_training=is_training, is_testing=is_testing, is_dilated=False, scope="conv2")	
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=512, rate=1,
							is_shortcut=False, is_bottleneck=False, is_batch_norm=True, 
							is_training=is_training, is_testing=is_testing, is_dilated=False, scope="conv3")			
				net, indices5, output_shape5 = utils.indice_pool(net, stride=2, scope="Pool5")
						
		with tf.variable_scope("decoder"):
			with tf.variable_scope("14X14_D"): # 1/ # conv5_D
				net = utils.indice_unpool(net, stride=2, output_shape=output_shape5, indices=indices5, scope="unPool5")
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=512, rate=1,
							is_shortcut=False, is_bottleneck=False, is_batch_norm=True, 
							is_training=is_training, is_testing=is_testing, is_dilated=False, scope="conv1")
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=512, rate=1,
							is_shortcut=False, is_bottleneck=False, is_batch_norm=True, 
							is_training=is_training, is_testing=is_testing, is_dilated=False, scope="conv2")	
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=512, rate=1,
							is_shortcut=False, is_bottleneck=False, is_batch_norm=True, 
							is_training=is_training, is_testing=is_testing, is_dilated=False, scope="conv3")							
				
			with tf.variable_scope("28X28_D"): # 1/8 # conv4_D
				net = utils.indice_unpool(net, stride=2, output_shape=output_shape4, indices=indices4, scope="unPool4")
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=512, rate=1,
							is_shortcut=False, is_bottleneck=False, is_batch_norm=True, 
							is_training=is_training, is_testing=is_testing, is_dilated=False, scope="conv1")
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=512, rate=1,
							is_shortcut=False, is_bottleneck=False, is_batch_norm=True, 
							is_training=is_training, is_testing=is_testing, is_dilated=False, scope="conv2")	
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=256, rate=1,
							is_shortcut=False, is_bottleneck=False, is_batch_norm=True, 
							is_training=is_training, is_testing=is_testing, is_dilated=False, scope="conv3")
				fc16 = net
				
			with tf.variable_scope("56X56_D"): # 1/4 # conv3_D
				net = utils.indice_unpool(net, stride=2, output_shape=output_shape3, indices=indices3, scope="unPool3")
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=256, rate=1,
							is_shortcut=False, is_bottleneck=False, is_batch_norm=True, 
							is_training=is_training, is_testing=is_testing, is_dilated=False, scope="conv1")
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=256, rate=1,
							is_shortcut=False, is_bottleneck=False, is_batch_norm=True, 
							is_training=is_training, is_testing=is_testing, is_dilated=False, scope="conv2")
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=128, rate=1,
							is_shortcut=False, is_bottleneck=False, is_batch_norm=True, 
							is_training=is_training, is_testing=is_testing, is_dilated=False, scope="conv3")
							
			with tf.variable_scope("112X112_D"): # 1/2 # conv2_D
				net = utils.indice_unpool(net, stride=2, output_shape=output_shape2, indices=indices2, scope="unPool2")
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=128, rate=1,
							is_shortcut=False, is_bottleneck=False, is_batch_norm=True, 
							is_training=is_training, is_testing=is_testing, is_dilated=False, scope="conv1")
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut=False, is_bottleneck=False, is_batch_norm=True, 
							is_training=is_training, is_testing=is_testing, is_dilated=False, scope="conv2")	
				
			with tf.variable_scope("224X224_D"): # 1/1 # conv1_D
				net = utils.indice_unpool(net, stride=2, output_shape=output_shape1, indices=indices1, scope="unPool1")
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1,
							is_shortcut=False, is_bottleneck=False, is_batch_norm=True, 
							is_training=is_training, is_testing=is_testing, is_dilated=False, scope="conv1")
				net = utils.conv2D(net, kernel_size=3, stride=1, output_channel=class_num, rate=1,
							is_shortcut=False, is_bottleneck=False, is_batch_norm=False, 
							is_training=is_training, is_testing=is_testing, is_dilated=False, scope="conv2")	
	return net
	
	
def main(argv=None):
#===============#
#	File Read	#
#===============#
	CamVid_Path = "/home/2016/b22072117/ObjectSegmentation/codes/dataset/CamVid"

# Training file
	print("")
	print("Loading Training Data ...")
	CamVid_train_data, CamVid_train_data_index = utils.read_dataset_file(CamVid_Path, '/train.txt')
	CamVid_train_target, _ = utils.read_dataset_file(CamVid_Path, '/trainannot.txt')
	class_num = np.max(CamVid_train_target)+1
	CamVid_train_target = utils.one_of_k(CamVid_train_target, class_num)
	print("Shape of train data	: {Shape}" .format(Shape = np.shape(CamVid_train_data)))
	print("Shape of train target	: {Shape}" .format(Shape = np.shape(CamVid_train_target)))
	
# Validation file
	print("")
	print("Loading Validation Data ...")
	CamVid_valid_data, CamVid_valid_data_index = utils.read_dataset_file(CamVid_Path, '/val.txt')
	CamVid_valid_target, _ = utils.read_dataset_file(CamVid_Path, '/valannot.txt')
	if class_num < np.max(CamVid_valid_target)+1:
		print("[\033[1;33;40mTraining class number is lower than Validation class number!\033[0m") 
		exit()
	CamVid_valid_target = utils.one_of_k(CamVid_valid_target, class_num)
	print("Shape of valid data	: {Shape}" .format(Shape = np.shape(CamVid_valid_data)))
	print("Shape of valid target	: {Shape}" .format(Shape = np.shape(CamVid_valid_target)))

# Testing file
	print("")
	print("Loading Testing Data ...")
	CamVid_test_data, CamVid_test_data_index = utils.read_dataset_file(CamVid_Path, '/test.txt')
	CamVid_test_target, _ = utils.read_dataset_file(CamVid_Path, '/testannot.txt')
	if class_num < np.max(CamVid_test_target)+1:
		print("[\033[1;33;40mTraining class number is lower than Validation class number!\033[0m") 
		exit()
	CamVid_test_target = utils.one_of_k(CamVid_test_target, class_num)
	print("Shape of test data	: {Shape}" .format(Shape = np.shape(CamVid_test_data)))
	print("Shape of test target	: {Shape}" .format(Shape = np.shape(CamVid_test_target)))
	print("")
	
#===========#
#	Data	#
#===========#

# Placeholder
	data_shape = np.shape(CamVid_train_data)
	xs = tf.placeholder(tf.float32, [BATCH_SIZE, data_shape[1], data_shape[2], data_shape[3]]) 
	ys = tf.placeholder(tf.float32, [BATCH_SIZE, data_shape[1], data_shape[2], class_num])
	learning_rate = tf.placeholder(tf.float32)
	is_training = tf.placeholder(tf.bool)
	is_testing = tf.placeholder(tf.bool)
	
# Data Preprocessing
	class_weight = utils.calculate_class_weight(CamVid_train_target) # For Loss Function
#	y_b = boundary_finder(ys)
	xs_image = xs
	
#===========#
#	Graph	#
#===========#
	net = xs_image
	prediction = Model.SegNet_VGG_16(net, class_num=class_num, is_training=is_training, is_testing=is_testing)

	if IS_TESTING:
		prediction = tf.nn.softmax(tf.truediv(prediction, tf.constant(T)))
	
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
	
#	labels_bd = tf.multiply(y_b, tf.argmax(ys, -1))
#	logits_bd = tf.multiply(tf.tile(tf.expend_dims(y_b, 3), [1, 1, 1, class_num]), prediction)
#	cross_entropy_bd = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_bd, logits=logits_bd)
	l2_norm = tf.reduce_mean(tf.stack([tf.nn.l2_loss(x) for x in weights_collection]))
	l2_lambda = tf.constant(LAMBDA)
	#cross_entropy = softmax_with_weighted_cross_entropy(prediction, ys, class_weight)
	#train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
	loss = cross_entropy
#	loss = tf.add(loss, cross_entropy_bd)
	loss = tf.add(loss, tf.multiply(l2_lambda, l2_norm))
	
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
			'conv4_1', 'conv4_1_b', 'conv4_1_mean', 'conv4_1_var', 'conv4_1_scale', 'conv4_1_shift',
			'conv4_2', 'conv4_2_b', 'conv4_2_mean', 'conv4_2_var', 'conv4_2_scale', 'conv4_2_shift',
			'conv4_3', 'conv4_3_b', 'conv4_3_mean', 'conv4_3_var', 'conv4_3_scale', 'conv4_3_shift',
			'conv5_1', 'conv5_1_b', 'conv5_1_mean', 'conv5_1_var', 'conv5_1_scale', 'conv5_1_shift',
			'conv5_2', 'conv5_2_b', 'conv5_2_mean', 'conv5_2_var', 'conv5_2_scale', 'conv5_2_shift',
			'conv5_3', 'conv5_3_b', 'conv5_3_mean', 'conv5_3_var', 'conv5_3_scale', 'conv5_3_shift',
			'conv5_3_D', 'conv5_3_D_b', 'conv5_3_D_mean', 'conv5_3_D_var', 'conv5_3_D_scale', 'conv5_3_D_shift',
			'conv5_2_D', 'conv5_2_D_b', 'conv5_2_D_mean', 'conv5_2_D_var', 'conv5_2_D_scale', 'conv5_2_D_shift',
			'conv5_1_D', 'conv5_1_D_b', 'conv5_1_D_mean', 'conv5_1_D_var', 'conv5_1_D_scale', 'conv5_1_D_shift',
			'conv4_3_D', 'conv4_3_D_b', 'conv4_3_D_mean', 'conv4_3_D_var', 'conv4_3_D_scale', 'conv4_3_D_shift',
			'conv4_2_D', 'conv4_2_D_b', 'conv4_2_D_mean', 'conv4_2_D_var', 'conv4_2_D_scale', 'conv4_2_D_shift',
			'conv4_1_D', 'conv4_1_D_b', 'conv4_1_D_mean', 'conv4_1_D_var', 'conv4_1_D_scale', 'conv4_1_D_shift',
			'conv3_3_D', 'conv3_3_D_b', 'conv3_3_D_mean', 'conv3_3_D_var', 'conv3_3_D_scale', 'conv3_3_D_shift',
			'conv3_2_D', 'conv3_2_D_b', 'conv3_2_D_mean', 'conv3_2_D_var', 'conv3_2_D_scale', 'conv3_2_D_shift',
			'conv3_1_D', 'conv3_1_D_b', 'conv3_1_D_mean', 'conv3_1_D_var', 'conv3_1_D_scale', 'conv3_1_D_shift',
			'conv2_2_D', 'conv2_2_D_b', 'conv2_2_D_mean', 'conv2_2_D_var', 'conv2_2_D_scale', 'conv2_2_D_shift',
			'conv2_1_D', 'conv2_1_D_b', 'conv2_1_D_mean', 'conv2_1_D_var', 'conv2_1_D_scale', 'conv2_1_D_shift',
			'conv1_2_D', 'conv1_2_D_b', 'conv1_2_D_mean', 'conv1_2_D_var', 'conv1_2_D_scale', 'conv1_2_D_shift',
			'conv1_1_D', 'conv1_1_D_b']
	parameters = {keys[x]: params[x] for x in range(len(params))}
	
#===========#
#	Saver	#
#===========#
	saver = tf.train.Saver()
	save_path = "SegNet.ckpt"
	
	restore = tf.train.Saver()
	restore_path = "SegNet.ckpt"
	
#===============#
#	Session Run	#
#===============#
	with tf.Session() as sess:
	# Initialize
		sess.run(init)
		
	# Learning Rate
		"""
		Policy : base_lr * gamma ^ (floor(iter / step))
		-----------------------------------------------
		base_lr = 0.001
		gamma = 1.0
		stepsize: 10000000
		"""
		
		lr = LEARNING_RATE
	
		if IS_TRAINING == True:
		# Loading Pre-trained Model
			print ""
			print("Loading Pre-trained weights ...")
			utils.load_pre_trained_weights(parameters, pre_trained_weight_file=PRE_TRAINED_WEIGHT_FILE, sess=sess)
			
		# Training & Validation
			per_class_prediction_num = np.zeros([class_num, 1])
			per_class_target_num = np.zeros([class_num, 1])
			
			for epoch in range(EPOCH_TIME):
				CamVid_train_data, CamVid_train_target = utils.shuffle_image(CamVid_train_data, CamVid_train_targetx)
				print("")
				print("Training ... ")
				for i in range(data_shape[0]/BATCH_SIZE):
				
					batch_xs = CamVid_train_data[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
					batch_ys = CamVid_train_target[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
					
					_, CrossEntropy, Prediction = sess.run([train_step, cross_entropy, prediction], 
															feed_dict={xs: batch_xs, ys: batch_ys, learning_rate: lr, is_training: True, is_testing: False})
					
														
					batch_accuracy = np.mean(np.equal(np.argmax(batch_ys, -1), np.argmax(Prediction, -1)))
					
					print("Epoch : {ep}" .format(ep = epoch))
					print("Batch Iteration : {Iter}" .format(Iter = i))
					print("  Batch Accuracy : {acc}".format(acc = batch_accuracy))
					print("  Cross Entropy : {CrossEntropy}".format(CrossEntropy = np.mean(CrossEntropy)))
					print("  Learning Rate : {LearningRate}".format(LearningRate = lr))
					utils.per_class_accuracy(Prediction, batch_ys)
					
			# Validation
				print("")
				print("Epoch : {ep}".format(ep = epoch))
				print("Validation ... ")
				is_validation = True
				_, valid_accuracy, _, _, _, _ = utils.compute_accuracy(xs, ys, is_training, is_testing, is_validation, prediction, CamVid_valid_data, CamVid_valid_target, sess)
				is_validation = False
				print("")
				print("Validation Accuracy = {Valid_Accuracy}".format(Valid_Accuracy=valid_accuracy))
				
				if ((epoch%EPOCH_DECADE==0) and (epoch!=0)):
					lr = lr / LR_DECADE
					
			# Save trained weights
				if (((epoch+1)%10==0) and ((epoch+1)>=10)):
					print("Saving Trained Weights ... ")
					batch_xs = CamVid_train_data[0 : BATCH_SIZE]
					utils.assign_trained_mean_and_var(mean_collection, var_collection, trained_mean_collection, trained_var_collection, params, 
												xs, ys, is_training, is_testing, batch_xs, sess)
					utils.save_pre_trained_weights( (TRAINING_WEIGHT_FILE+'_'+str(epoch+1)), parameters, xs, batch_xs, sess)
		
	
		if IS_TESTING == True:
			is_validation = False

		# Load trained weights and save all the results of the wanted image
			print("")
			print("Loading the trained weights ... ")
			utils.load_pre_trained_weights(parameters, pre_trained_weight_file=TESTING_WEIGHT_FILE, sess=sess) 

			#***********#
			#	TRAIN	#
			#***********#
			print("train result ... ")
			train_target_path = '/home/2016/b22072117/ObjectSegmentation/codes/result/CamVid/SegNet/'
			train_Y_pre_path  = '/home/2016/b22072117/ObjectSegmentation/codes/nets/SegNet_Y_pre'

			train_result, train_accuracy, train_accuracy_top2, train_accuracy_top3, SegNet_Y_pre_train = utils.compute_accuracy(xs, ys, is_training, is_testing, is_validation, prediction, CamVid_train_data, CamVid_train_target, BATCH_SIZE, sess)

			print("Coloring train result ... ")
			#train_result = utils.color_result(train_result)
			
			print("Saving the train data result as image ... ")
			#utisl.Save_result_as_image(train_target_path, train_result, CamVid_train_data_index)

			print("Saving the train Y_pre result as npz ... ")
			utils.Save_result_as_npz(train_Y_pre_path, SegNet_Y_pre_train, CamVid_train_data_index)
			
			print("Train Data Accuracy = {Train_Accuracy}, top2 = {top2}, top3 = {top3}"
			.format(Train_Accuracy=train_accuracy, top2=train_accuracy_top2, top3=train_accuracy_top3))
			

			#***********#
			#	VALID	#
			#***********#
			print(" valid result ... ")
			valid_target_path = '/home/2016/b22072117/ObjectSegmentation/codes/result/CamVid/SegNet/'
			valid_Y_pre_path  = '/home/2016/b22072117/ObjectSegmentation/codes/nets/SegNet_Y_pre'

			valid_result, valid_accuracy, valid_accuracy_top2, valid_accuracy_top3, SegNet_Y_pre_valid = utils.compute_accuracy(xs, ys, is_training, is_testing, is_validation, prediction, CamVid_valid_data, CamVid_valid_target, BATCH_SIZE, sess)

			print("Coloring valid result ... ")
			#valid_result = utils.color_result(valid_result)

			print("Saving the valid data result as image ... ")
			#utils.Save_result_as_image(valid_target_path, valid_result, CamVid_valid_data_index)

			print("Saving the valid Y_pre result as npz ... ")
			utils.Save_result_as_npz (valid_Y_pre_path, SegNet_Y_pre_valid, CamVid_valid_data_index)

			print("Valid Data Accuracy = {Valid_Accuracy}, top2 = {top2}, top3 = {top3}"
			.format(Valid_Accuracy=valid_accuracy, top2=valid_accuracy_top2, top3=valid_accuracy_top3))
			
			#***********#
			#	TEST	#
			#***********#
			print(" test result ... ")
			test_target_path = '/home/2016/b22072117/ObjectSegmentation/codes/result/CamVid/SegNet/'
			test_Y_pre_path  = '/home/2016/b22072117/ObjectSegmentation/codes/nets/SegNet_Y_pre'

			test_result, test_accuracy, test_accuracy_top2, test_accuracy_top3, SegNet_Y_pre_test = utils.compute_accuracy(xs, ys, is_training, is_testing, is_validation, prediction, CamVid_test_data, CamVid_test_target, BATCH_SIZE, sess)

			print("Coloring test result ... ")
			#test_result = utils.color_result(test_result)

			print("Saving the test data result as image ... ")
			#utils.Save_result_as_image(test_target_path, test_result, CamVid_test_data_index)

			print("Saving the test Y_pre result as npz ... ")
			utils.Save_result_as_npz(test_Y_pre_path, SegNet_Y_pre_test, CamVid_test_data_index)

			print("Test Data Accuracy = {Test_Accuracy}, top2 = {top2}, top3 = {top3}"
			.format(Test_Accuracy=test_accuracy, top2=test_accuracy_top2, top3=test_accuracy_top3))
			
			print("")
			print("Train Data Accuracy = {Train_Accuracy}, top2 = {top2}, top3 = {top3}"
			.format(Train_Accuracy=train_accuracy, top2=train_accuracy_top2, top3=train_accuracy_top3))
			print("Valid Data Accuracy = {Valid_Accuracy}, top2 = {top2}, top3 = {top3}"
			.format(Valid_Accuracy=valid_accuracy, top2=valid_accuracy_top2, top3=valid_accuracy_top3))
			print("Test Data Accuracy = {Test_Accuracy}, top2 = {top2}, top3 = {top3}"
			.format(Test_Accuracy=test_accuracy, top2=test_accuracy_top2, top3=test_accuracy_top3))

			
		print("")
		print("Works are All Done !")
			
if __name__ == "__main__":
	tf.app.run()

