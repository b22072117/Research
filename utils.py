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


#def GAN_discriminator():
	

def CamVid_data_parser(
	# Path
	CamVid_Path, # CamVid_Path = '/"Path_to_CamVid"/CamVid'
	Y_pre_Path , # Y_pre_Path  = '/"Path_to_Y_pre_Path"/PSPNet_Y_pre'

	# Parameter
	IS_STUDENT,
	IS_TRAINING
):

# Training file
	print("")
	print("Loading Training Data ...")
	# CamVid
	CamVid_train_data  , CamVid_train_data_index = read_dataset_file(CamVid_Path, '/train.txt')
	CamVid_train_target, _                       = read_dataset_file(CamVid_Path, '/trainannot.txt')
	
	# Teacher Model Prediction Output
	if IS_TRAINING and IS_STUDENT:	
		Y_pre_train_data = CamVid_train_data
		Y_pre_train_target, Y_pre_train_data_index = read_Y_pre_file(Y_pre_Path, '/train.txt')
	else:
		Y_pre_train_data       = None
		Y_pre_train_target     = None
		Y_pre_train_data_index = None
		

	class_num = np.max(CamVid_train_target)+1
	CamVid_train_target = one_of_k(CamVid_train_target, class_num)
	print("Shape of train data	: {Shape}" .format(Shape = np.shape(CamVid_train_data)))
	print("Shape of train target	: {Shape}" .format(Shape = np.shape(CamVid_train_target)))
	
# Validation file
	print("")
	print("Loading Validation Data ...")
	# CamVid
	CamVid_valid_data  , CamVid_valid_data_index = read_dataset_file(CamVid_Path, '/val.txt')
	CamVid_valid_target, _                       = read_dataset_file(CamVid_Path, '/valannot.txt')
	
	# Teacher Model Prediction Output
	if IS_TRAINING and IS_STUDENT:	
		Y_pre_valid_data = CamVid_valid_data
		Y_pre_valid_target, Y_pre_valid_data_index = read_Y_pre_file(Y_pre_Path, '/val.txt')
	else:
		Y_pre_valid_data       = None
		Y_pre_valid_target     = None
		Y_pre_valid_data_index = None

	CamVid_valid_target = one_of_k(CamVid_valid_target, class_num)
	print("Shape of valid data	: {Shape}" .format(Shape = np.shape(CamVid_valid_data)))
	print("Shape of valid target	: {Shape}" .format(Shape = np.shape(CamVid_valid_target)))

# Testing file
	print("")
	print("Loading Testing Data ...")
	# CamVid 
	CamVid_test_data  , CamVid_test_data_index = read_dataset_file(CamVid_Path, '/test.txt')
	CamVid_test_target, _                      = read_dataset_file(CamVid_Path, '/testannot.txt')
	
	# Teacher Model Prediction Output
	if IS_TRAINING and IS_STUDENT:	
		Y_pre_test_data = CamVid_test_data
		Y_pre_test_target, Y_pre_test_data_index = read_Y_pre_file(Y_pre_Path, '/test.txt')
	else:
		Y_pre_test_data       = None
		Y_pre_test_target     = None
		Y_pre_test_data_index = None
	
	CamVid_test_target = one_of_k(CamVid_test_target, class_num)
	print("Shape of test data	: {Shape}" .format(Shape = np.shape(CamVid_test_data)))
	print("Shape of test target	: {Shape}" .format(Shape = np.shape(CamVid_test_target)))
	print("")
	
	return CamVid_train_data, CamVid_train_data_index, CamVid_train_target, CamVid_valid_data, CamVid_valid_data_index, CamVid_valid_target, CamVid_test_data, CamVid_test_data_index, CamVid_test_target, Y_pre_train_data, Y_pre_train_data_index, Y_pre_train_target, Y_pre_valid_data, Y_pre_valid_data_index, Y_pre_valid_target, Y_pre_test_data  , Y_pre_test_data_index, Y_pre_test_target, class_num
		   

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
	IS_TRAINING					,
	
	# Tensor	
	prediction					,
	
	# Placeholder	
	xs							, 
	ys							,
	is_training					,
	is_testing					,
		
	# File Path (For Loading Trained Weight)
	TESTING_WEIGHT_FILE			,
		
	# Trained Weight Parameters (For Loading Trained Weight)
	parameters					,
	
	# File Path (For Saving Result)
	train_target_path 			,
	train_Y_pre_path  			,
	valid_target_path 			,
	valid_Y_pre_path  			,
	test_target_path 			,
	test_Y_pre_path  			,
		
	# Session
	sess						):
	
	is_validation = False
# Load trained weights and save all the results of the wanted image
	if IS_TRAINING == False:
		print("")
		print("Loading the trained weights ... ")
		load_pre_trained_weights(parameters, pre_trained_weight_file=TESTING_WEIGHT_FILE, sess=sess) 
	
	#***********#
	#	TRAIN	#
	#***********#
	print("train result ... ")
	train_result, train_accuracy, train_accuracy_top2, train_accuracy_top3, Y_pre_train = compute_accuracy(xs, ys, is_training, is_testing, is_validation, prediction, train_data, train_target, BATCH_SIZE, sess)
	print("Train Data Accuracy = {Train_Accuracy}, top2 = {top2}, top3 = {top3}"
	.format(Train_Accuracy=train_accuracy, top2=train_accuracy_top2, top3=train_accuracy_top3))

	
	#***********#
	#	VALID	#
	#***********#
	print(" valid result ... ")
	valid_result, valid_accuracy, valid_accuracy_top2, valid_accuracy_top3, Y_pre_valid = compute_accuracy(xs, ys, is_training, is_testing, is_validation, prediction, valid_data, valid_target, BATCH_SIZE, sess)
	print("Valid Data Accuracy = {Valid_Accuracy}, top2 = {top2}, top3 = {top3}"
	.format(Valid_Accuracy=valid_accuracy, top2=valid_accuracy_top2, top3=valid_accuracy_top3))
	

	#***********#
	#	TEST	#
	#***********#
	print(" test result ... ")
	test_result, test_accuracy, test_accuracy_top2, test_accuracy_top3, Y_pre_test = compute_accuracy(xs, ys, is_training, is_testing, is_validation, prediction, test_data, test_target, BATCH_SIZE, sess)
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
		print("Saving the valid Y_pre result as npz ... ")
		Save_result_as_npz (valid_Y_pre_path, Y_pre_valid, CamVid_valid_data_index)
		print("Saving the train Y_pre result as npz ... ")
		Save_result_as_npz(train_Y_pre_path, Y_pre_train, CamVid_train_data_index)
		print("Saving the test Y_pre result as npz ... ")
		Save_result_as_npz(test_Y_pre_path, Y_pre_test, CamVid_test_data_index)

def Training_and_Validation( 
		# Training & Validation Data
		train_data				, 
		train_target			,
		valid_data				,
		valid_target			,
		
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
		
		# Collection (For Saving Trained Weight)
		mean_collection			,
		var_collection			,
		trained_mean_collection	,
		trained_var_collection	,
		params					,
		
		# File Path (For Saving Trained Weight)
		TRAINED_WEIGHT_FILE		,
		TRAINING_WEIGHT_FILE	,
		
		# Trained Weight Parameters (For Saving Trained Weight)
		parameters				,
		
		# Session
		sess					):

	# Loading Pre-trained Model
	if TRAINED_WEIGHT_FILE!=None:
		print ""
		print("Loading Pre-trained weights ...")
		load_pre_trained_weights(parameters, pre_trained_weight_file=TRAINED_WEIGHT_FILE, sess=sess)

	for epoch in range(EPOCH_TIME):
		data_shape = np.shape(train_data)
		train_data, train_target = shuffle_image(train_data, train_target)

		Train_acc = 0
		Train_loss = 0
		
		print("")
		print("Training ... ")
		for i in range(int(data_shape[0]/BATCH_SIZE)):
			# Train data in BATCH SIZR
			batch_xs    = train_data   [i*BATCH_SIZE : (i+1)*BATCH_SIZE]
			batch_ys    = train_target [i*BATCH_SIZE : (i+1)*BATCH_SIZE]
			
			# Run Training Step
			_, Loss, Prediction = sess.run([train_step, loss, prediction], 
													feed_dict={xs: batch_xs, ys: batch_ys, learning_rate: lr, is_training: True, is_testing: False})
			
			# Result
			y_pre = np.argmax(Prediction, -1)
			batch_accuracy = np.mean(np.equal(np.argmax(batch_ys, -1), y_pre))
			
			Train_acc  = Train_acc  + batch_accuracy
			Train_loss = Train_loss + np.mean(Loss)

			# Print Training Process
			print("Epoch : {ep}" .format(ep = epoch))
			print("Batch Iteration : {Iter}" .format(Iter = i))
			print("  Batch Accuracy : {acc}".format(acc = batch_accuracy))
			print("  Loss : {loss}".format(loss = np.mean(Loss)))
			print("  Learning Rate : {LearningRate}".format(LearningRate = lr))
			
			# Per Class Accuracy
			per_class_accuracy(Prediction, batch_ys)

		# Per Epoch Training Result
		if epoch==0:
			Train_acc_per_epoch  = np.array([Train_acc ])
			Train_loss_per_epoch = np.array([Train_loss])
		else:
			Train_acc_per_epoch  = np.concatenate([Train_acc_per_epoch , np.array([Train_acc ])], axis=0)
			Train_loss_per_epoch = np.concatenate([Train_loss_per_epoch, np.array([Train_loss])], axis=0)

		

	# Validation
		print("")
		print("Epoch : {ep}".format(ep = epoch))
		print("Validation ... ")
		is_validation = True 
		_, valid_accuracy, _, _, _ = compute_accuracy(xs, ys, is_training, is_testing, is_validation, prediction, valid_data, valid_target, BATCH_SIZE, sess)
		is_validation = False
		print("")
		print("Validation Accuracy = {Valid_Accuracy}".format(Valid_Accuracy=valid_accuracy))
		
		if ((epoch%EPOCH_DECADE==0) and (epoch!=0)):
			lr = lr / LR_DECADE
			
	# Save trained weights
		print("Saving Trained Weights ... ")
		batch_xs = train_data[0 : BATCH_SIZE]
		assign_trained_mean_and_var(mean_collection, var_collection, trained_mean_collection, trained_var_collection, params, 
										xs, ys, is_training, is_testing, batch_xs, sess)
		if (((epoch+1)%10==0) and ((epoch+1)>=10)):
			save_pre_trained_weights( (TRAINING_WEIGHT_FILE+'_'+str(epoch+1)), parameters, xs, batch_xs, sess)
	
	# Save Train info as csv file
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

def read_dataset_file(Path, text_file_name):
	data_index = open(Path + text_file_name, 'r').read().splitlines()
			
	for i, file_name in enumerate(data_index):
		if i==0:
			data = np.expand_dims(misc.imread(Path + file_name), axis=0)
		else:
			data = np.concatenate([data, np.expand_dims(misc.imread(Path + file_name), axis=0)], axis=0)
	return data, data_index

def Save_result_as_image(Path, result, file_index):
	for i, target in enumerate(result):
		scipy.misc.imsave(Path + file_index[i], target)
	
def Save_result_as_npz(Path, result, file_index):
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
		
		print("    Class{Iter}	: {predict} / {target}".format(Iter = i, predict=correct_num, target=total_num))
	
		
def compute_accuracy(xs, ys, is_training, is_testing, is_validation, predicton, v_xs, v_ys, BATCH_SIZE, sess):
	test_batch_size = BATCH_SIZE
	batch_num = len(v_xs) / test_batch_size
	result_accuracy_top1 = 0
	result_accuracy_top2 = 0
	result_accuracy_top3 = 0
	for i in range(int(batch_num)):	
		v_xs_part = v_xs[i*test_batch_size : (i+1)*test_batch_size, :]
		v_ys_part = v_ys[i*test_batch_size : (i+1)*test_batch_size, :]
			
		Y_pre = sess.run(predicton, feed_dict={xs: v_xs_part, is_training: False, is_testing: (not is_validation)})
				
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
		weights = np.load(pre_trained_weight_file)
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
	
def conv2D(	net, kernel_size=3, stride=1, internal_channel=64, output_channel=64, rate=1,
			initializer=tf.contrib.layers.variance_scaling_initializer(),
			is_contant_init	= False, 		# For using constant value as weight initial; Only valid in Normal Convolution
			is_shortcut		= False, 		# For Residual
			is_bottleneck	= False, 		# For Residual
			is_batch_norm	= True,  		# For Batch Normalization
			is_training		= True,  		# For Batch Normalization
			is_testing		= False, 		# For getting the pretrained from caffemodel
			is_dilated		= False, 		# For Dilated Convolution
			padding			= "SAME",
			scope			= "conv"):
			
	with tf.variable_scope(scope):
		input_channel = net.get_shape()[-1]

		#===================#
		#	Resdual Block	#
		#===================#
		if is_shortcut:
			with tf.variable_scope("shortcut"):
				weights = tf.get_variable("weights", [1, 1, input_channel, output_channel], tf.float32, initializer=initializer)		
				biases = tf.Variable(tf.constant(0.0, shape=[output_channel], dtype=tf.float32), trainable=True, name='biases')
				if input_channel!=output_channel:

					tf.add_to_collection("weights", weights)
					tf.add_to_collection("bias", biases)
					tf.add_to_collection("params", weights)
					tf.add_to_collection("params", biases)

					if stride != 1:
						# convolution
						shortcut = tf.nn.conv2d(net, weights, strides=[1, stride, stride, 1], padding="SAME")
					elif is_dilated:
						# convolution
						shortcut = tf.nn.atrous_conv2d(net, weights, rate=rate, padding="SAME")

					# add bias
					shortcut = tf.nn.bias_add(shortcut, biases)

					# batch normalization
					if is_batch_norm == True:
						shortcut = batch_norm(shortcut, output_channel, is_training, is_testing)
				else:
					shortcut = net	

			#-------------------------------#
			#	Bottleneck Residual Block	#
			#-------------------------------#
			if is_bottleneck: 
				with tf.variable_scope("bottle_neck"):
					with tf.variable_scope("conv1_1x1"):
						weights = tf.get_variable("weights", [1, 1, input_channel, internal_channel], tf.float32, initializer=initializer)
						biases = tf.Variable(tf.constant(0.0, shape=[internal_channel], dtype=tf.float32), trainable=True, name='biases')

						tf.add_to_collection("weights", weights)
						tf.add_to_collection("bias", biases)
						tf.add_to_collection("params", weights)
						tf.add_to_collection("params", biases)
						
						# convolution
						if is_dilated:
							net = tf.nn.atrous_conv2d(net, weights, rate=rate, padding="SAME")
						else:
							net = tf.nn.conv2d(net, weights, strides=[1, stride, stride, 1], padding="SAME")
						
						# add bias
						net = tf.nn.bias_add(net, biases)
						
						# batch normalization
						if is_batch_norm == True:
							net = batch_norm(net, internal_channel, is_training, is_testing)
							
						net = tf.nn.relu(net)

					with tf.variable_scope("conv2_3x3"):
						weights = tf.get_variable("weights", [3, 3, internal_channel, internal_channel], tf.float32, initializer=initializer)
						biases = tf.Variable(tf.constant(0.0, shape=[internal_channel], dtype=tf.float32), trainable=True, name='biases')

						tf.add_to_collection("weights", weights)	
						tf.add_to_collection("bias", biases)
						tf.add_to_collection("params", weights)
						tf.add_to_collection("params", biases)
						
						# convolution
						if is_dilated:
							net = tf.nn.atrous_conv2d(net, weights, rate=rate, padding="SAME")
						else:
							net = tf.nn.conv2d(net, weights, strides=[1, 1, 1, 1], padding="SAME")
						
						# add bias
						net = tf.nn.bias_add(net, biases)
						
						# batch normalization
						if is_batch_norm == True:
							net = batch_norm(net, internal_channel, is_training, is_testing)
							
						net = tf.nn.relu(net)
					with tf.variable_scope("conv3_1x1"):
						weights = tf.get_variable("weights", [1, 1, internal_channel, output_channel], tf.float32, initializer=initializer)
						biases = tf.Variable(tf.constant(0.0, shape=[output_channel], dtype=tf.float32), trainable=True, name='biases')

						tf.add_to_collection("weights", weights)
						tf.add_to_collection("bias", biases)
						tf.add_to_collection("params", weights)
						tf.add_to_collection("params", biases)
						
						# convolution
						if is_dilated:
							net = tf.nn.atrous_conv2d(net, weights, rate=rate, padding="SAME")
						else:
							net = tf.nn.conv2d(net, weights, strides=[1, 1, 1, 1], padding="SAME")
						
						# add bias
						net = tf.nn.bias_add(net, biases)
						
						# batch normalization
						if is_batch_norm == True:
							net = batch_norm(net, output_channel, is_training, is_testing)
							
					net = tf.nn.relu(net + shortcut)

			#---------------------------#
			#	Normal Residual Block	#
			#---------------------------#
			else: 
				with tf.variable_scope("conv1_3x3"):
					weights = tf.get_variable("weights", [3, 3, input_channel, internal_channel], tf.float32, initializer=initializer)
					biases = tf.Variable(tf.constant(0.0, shape=[internal_channel], dtype=tf.float32), trainable=True, name='biases')

					tf.add_to_collection("weights", weights)
					tf.add_to_collection("bias", biases)
					tf.add_to_collection("params", weights)
					tf.add_to_collection("params", biases)
						
					# convolution
					if is_dilated:
						net = tf.nn.atrous_conv2d(net, weights, rate=rate, padding="SAME")
					else:
						net = tf.nn.conv2d(net, weights, strides=[1, stride, stride, 1], padding="SAME")
					
					# add bias
					net = tf.nn.bias_add(net, biases)
					
					# batch normalization
					if is_batch_norm == True:
						net = batch_norm(net, internal_channel, is_training, is_testing)
						
					net = tf.nn.relu(net)

				with tf.variable_scope("conv2_3x3"):
					weights = tf.get_variable("weights", [3, 3, internal_channel, output_channel], tf.float32, initializer=initializer)
					biases = tf.Variable(tf.constant(0.0, shape=[output_channel], dtype=tf.float32), trainable=True, name='biases')
					tf.add_to_collection("weights", weights)
					tf.add_to_collection("bias", biases)
					tf.add_to_collection("params", weights)
					tf.add_to_collection("params", biases)
					
					# convolution
					if is_dilated:
						net = tf.nn.atrous_conv2d(net, weights, rate=rate, padding="SAME")
					else:
						net = tf.nn.conv2d(net, weights, strides=[1, 1, 1, 1], padding="SAME")
					
					# add bias
					net = tf.nn.bias_add(net, biases)
					
					# batch normalization
					if is_batch_norm == True:
						net = batch_norm(net, internal_channel, is_training, is_testing)
				
				#relu
				net = tf.nn.relu(net + shortcut)

		#=======================#
		#	Normal Convolution	#
		#=======================#
		else:  
			if is_contant_init:
				weights = tf.get_variable("weights", dtype=tf.float32, initializer=initializer)
			else:
				weights = tf.get_variable("weights", [kernel_size, kernel_size, input_channel, output_channel], tf.float32, initializer=initializer)

			biases = tf.Variable(tf.constant(0.0, shape=[output_channel], dtype=tf.float32), trainable=True, name='biases')
			tf.add_to_collection("weights", weights)
			tf.add_to_collection("bias", biases)
			tf.add_to_collection("params", weights)
			tf.add_to_collection("params", biases)
			
			# convolution
			if is_dilated:
				net = tf.nn.atrous_conv2d(net, weights, rate=rate, padding=padding)
			else:
				net = tf.nn.conv2d(net, weights, strides=[1, stride, stride, 1], padding=padding)
			
			# add bias
			net = tf.nn.bias_add(net, biases)
			
			# batch normalization
			if is_batch_norm == True:
				net = batch_norm(net, output_channel, is_training, is_testing)
			
			#relu
			net = tf.nn.relu(net)

	return net

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

def indice_pool(net, stride, scope="Pool"):
	with tf.variable_scope(scope):
		output_shape = net.get_shape().as_list()
		net, indices = tf.nn.max_pool_with_argmax(input=net, 
												ksize=[1, stride, stride, 1],
												strides=[1, stride, stride, 1],
												padding="SAME",
												Targmax=None,
												name=None)
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
				
def batch_norm(net, output_channel, is_training, is_testing):
	with tf.variable_scope("Batch_Norm", reuse=None):
		batch_mean, batch_var = tf.nn.moments(net, axes=[0, 1, 2])
		
		ema = tf.train.ExponentialMovingAverage(decay=0.95)
		
		def mean_var_with_update():
			ema_apply_op = ema.apply([batch_mean, batch_var])
			with tf.control_dependencies([ema_apply_op]):
				return tf.identity(batch_mean), tf.identity(batch_var)
		
		trained_mean = tf.Variable(tf.zeros([output_channel]))
		trained_var = tf.Variable(tf.zeros([output_channel]))
		
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

def read_Y_pre_file(Path, text_file_name):
	data_index = open(Path + text_file_name, 'r').read().splitlines()

	for i, file_name in enumerate(data_index):
		file_name = file_name.split('.')[0]
		y_pre = np.load(Path + file_name + '.npz')
		Y_pre = y_pre[y_pre.keys()[0]]

		if i==0:
			data = np.expand_dims(Y_pre, axis=0)
		else:
			data = np.concatenate([data, np.expand_dims(Y_pre, axis=0)], axis=0)
	return data, data_index

def compute_accuracy_TOP_K(xs, ys, is_training, is_testing, is_validation, predicton, v_xs, v_ys, table, BATCH_SIZE, sess):
	test_batch_size = BATCH_SIZE
	batch_num = len(v_xs) / test_batch_size
	result_accuracy_top1 = 0
	result_accuracy_top2 = 0
	result_accuracy_top3 = 0

	for i in range(batch_num):	
		v_xs_part  = v_xs [i*test_batch_size : (i+1)*test_batch_size]
		v_ys_part  = v_ys [i*test_batch_size : (i+1)*test_batch_size] # one-of-k
		table_part = table[i*test_batch_size : (i+1)*test_batch_size]
		Prediction = sess.run(predicton, feed_dict={xs: v_xs_part, is_training: False, is_testing: (not is_validation)})
				
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
