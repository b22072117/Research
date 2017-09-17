import tensorflow as tf
import numpy as np
import scipy.ndimage
import pdb
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

#===============#
#	Parameter	#
#===============#
BATCH_SIZE = 20

#===========#
#	Define	#
#===========#
def compute_accuracy(xs, ys, is_training, predicton, v_xs, v_ys, sess):
	test_batch_size = 20
	batch_num = len(v_xs) / test_batch_size
	result = 0
	for i in range(batch_num):
	
		print("Testing Part {Iter}" .format(Iter=i))
		
		v_xs_part = v_xs[i*test_batch_size : (i+1)*test_batch_size, :]
		v_ys_part = v_ys[i*test_batch_size : (i+1)*test_batch_size, :]
		
		Y_pre = sess.run(predicton, feed_dict={xs: v_xs_part, is_training: False})
		
		#pdb.set_trace()
		correct_prediction = np.equal(np.argmax(Y_pre,1), np.argmax(v_ys_part,1))
		accuracy = np.mean(correct_prediction.astype(float))
		print("	Target Class  : {Target}".format(Target = np.argmax(v_ys_part,1)))
		print("	Predict Class : {Y_pre}".format(Y_pre = np.argmax(Y_pre,1)))
		#print("	Accuracy = {Accuracy}".format(Accuracy = accuracy))
		result = result + accuracy
	result = result / batch_num
	return result

def load_pre_trained_weights(parameters, pre_trained_weight_file=None, sess=None):
	if pre_trained_weight_file != None and sess != None:
		weights = np.load(pre_trained_weight_file)
		keys = sorted(weights.keys())
		for i, k in enumerate(keys):
			pre_trained_weights_shape = []
			for j in range(np.shape(np.shape(weights[k]))[0]):
				pre_trained_weights_shape += [np.shape(weights[k])[j]]
			
			print i, pre_trained_weights_shape, str(parameters[i].get_shape().as_list()), pre_trained_weights_shape == parameters[i].get_shape().as_list()
	
			if pre_trained_weights_shape == parameters[i].get_shape().as_list():
				#pdb.set_trace()
				sess.run(parameters[i].assign(weights[k]))

def save_pre_trained_weights(parameters, pre_trained_weight_file, sess):
	if pre_trained_weight_file != None and sess != None:
		if pre_trained_weights_shape == parameters[i].get_shape().as_list():
			#pdb.set_trace()
			sess.run(parameters[i].assign(weights[k]))
			
def conv2D(	net, 
			kernel_size=3, 
			stride=1, 
			internal_channel=64, 
			output_channel=64,
			rate=1,
			parameters=[],
			initializer=tf.contrib.layers.variance_scaling_initializer(),
			is_biases=False,
			is_shortcut=False, # For Residual
			is_bottleneck=False, # For Residual
			is_batch_norm=True, # For Batch Normalization
			is_training=True, # For Batch Normalization
			is_dialted=False, # For Dialted Convolution
			padding="SAME",
			scope="conv"):
			
	with tf.variable_scope(scope):
		input_channel = net.get_shape()[-1]
		if is_shortcut:
			with tf.variable_scope("shortcut"):
				if stride != 1:
					weights = tf.get_variable("weights", [1, 1, input_channel, output_channel], tf.float32, initializer=initializer)
					parameters += [weights]
					tf.add_to_collection("all_weights", weights)
					shortcut = tf.nn.conv2d(net, weights, strides=[1, stride, stride, 1], padding="SAME")
					shortcut = batch_norm(shortcut, output_channel, is_training=is_training)
					#shortcut = tf.nn.relu(shortcut)
				else:
					shortcut = net
					
			if is_bottleneck:
				with tf.variable_scope("bottle_neck"):
					with tf.variable_scope("conv1_1x1"):
						weights = tf.get_variable("weights", [1, 1, input_channel, internal_channel], tf.float32, initializer=initializer)
						parameters += [weights]
						tf.add_to_collection("all_weights", weights)
						if is_dialted == False:
							net = tf.nn.conv2d(net, weights, strides=[1, stride, stride, 1], padding="SAME")
						else:
							net = tf.nn.conv2d(net, weights, rate=rate, padding="SAME")
						if is_batch_norm == True:
							net = batch_norm(net, internal_channel, is_training=is_training)
						net = tf.nn.relu(net)
					with tf.variable_scope("conv2_3x3"):
						weights = tf.get_variable("weights", [3, 3, internal_channel, internal_channel], tf.float32, initializer=initializer)
						parameters += [weights]
						tf.add_to_collection("all_weights", weights)	
						if is_dialted == False:
							net = tf.nn.conv2d(net, weights, strides=[1, 1, 1, 1], padding="SAME")
						else:
							net = tf.nn.conv2d(net, weights, rate=rate, padding="SAME")
						if is_batch_norm == True:
							net = batch_norm(net, internal_channel, is_training=is_training)
						net = tf.nn.relu(net)
					with tf.variable_scope("conv3_1x1"):
						weights = tf.get_variable("weights", [1, 1, internal_channel, output_channel], tf.float32, initializer=initializer)
						parameters += [weights]
						tf.add_to_collection("all_weights", weights)
						if is_dialted == False:
							net = tf.nn.conv2d(net, weights, strides=[1, 1, 1, 1], padding="SAME")
						else:
							net = tf.nn.conv2d(net, weights, rate=rate, padding="SAME")
						if is_batch_norm == True:
							net = batch_norm(net, output_channel, is_training=is_training)
					net = tf.nn.relu(net + shortcut)
			else:
				with tf.variable_scope("conv1_3x3"):
					weights = tf.get_variable("weights", [3, 3, input_channel, internal_channel], tf.float32, initializer=initializer)
					parameters += [weights]
					tf.add_to_collection("all_weights", weights)
					if is_dialted == False:
						net = tf.nn.conv2d(net, weights, strides=[1, stride, stride, 1], padding="SAME")
					else:
						net = tf.nn.conv2d(net, weights, rate=rate, padding="SAME")
					if is_batch_norm == True:
						net = batch_norm(net, internal_channel, is_training=is_training)
					net = tf.nn.relu(net)
				with tf.variable_scope("conv2_3x3"):
					weights = tf.get_variable("weights", [3, 3, internal_channel, output_channel], tf.float32, initializer=initializer)
					parameters += [weights]
					tf.add_to_collection("all_weights", weights)	
					if is_dialted == False:
						net = tf.nn.conv2d(net, weights, strides=[1, 1, 1, 1], padding="SAME")
					else:
						net = tf.nn.conv2d(net, weights, rate=rate, padding="SAME")
					if is_batch_norm == True:
						net = batch_norm(net, internal_channel, is_training=is_training)
				net = tf.nn.relu(net + shortcut)
		else:
			weights = tf.get_variable("weights", [kernel_size, kernel_size, input_channel, output_channel], tf.float32, initializer=initializer)
			parameters += [weights]
			tf.add_to_collection("all_weights", weights)
			if is_dialted == False:
				net = tf.nn.conv2d(net, weights, strides=[1, stride, stride, 1], padding=padding)
			else:
				net = tf.nn.conv2d(net, weights, rate=rate, padding=padding)
			if is_biases:
				biases = tf.Variable(tf.constant(0.0, shape=[output_channel], dtype=tf.float32), trainable=True, name='biases')
				parameters += [biases]
				tf.add_to_collection("all_weights", biases)
				net = tf.nn.bias_add(net, biases)
			if is_batch_norm == True:
				net = batch_norm(net, output_channel, is_training=is_training)
			net = tf.nn.relu(net)
	return net, parameters

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
			
	
	
def batch_norm(net, output_channel, is_training):
	with tf.variable_scope("Batch_Norm", reuse=None):
		batch_mean, batch_var = tf.nn.moments(net, axes=[0, 1, 2])
		
		ema = tf.train.ExponentialMovingAverage(decay=0.999)
		
		def mean_var_with_update():
			ema_apply_op = ema.apply([batch_mean, batch_var])
			with tf.control_dependencies([ema_apply_op]):
				return tf.identity(batch_mean), tf.identity(batch_var)
				
		mean, var = tf.cond(is_training,    
							mean_var_with_update,
							lambda: (               
								ema.average(batch_mean), 
								ema.average(batch_var)
								)    
							) 
		scale = tf.Variable(tf.ones([output_channel]))
		shift = tf.Variable(tf.zeros([output_channel]))
		tf.add_to_collection("mean", mean)
		tf.add_to_collection("var", var)
		tf.add_to_collection("scale", scale)
		tf.add_to_collection("shift", shift)
		epsilon = 0.001
		
	return tf.nn.batch_normalization(net, mean, var, shift, scale, epsilon)

	
def SegNet_VGG_16(net, is_training, reuse=None, scope="SegNet_VGG_16"):
	with tf.variable_scope(scope, reuse=reuse):
		with tf.variable_scope("224X224"), tf.device("/gpu:0"): # 1/1
			parameters=[]
			net, parameters = conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1, parameters=parameters,
						is_biases=True, is_shortcut=False, is_bottleneck=False, is_batch_norm=False, is_training=is_training, is_dialted=False, scope="conv1")
			net, parameters = conv2D(net, kernel_size=3, stride=1, output_channel=64, rate=1, parameters=parameters,
						is_biases=True, is_shortcut=False, is_bottleneck=False, is_batch_norm=False, is_training=is_training, is_dialted=False, scope="conv2")	
			net, indices1, output_shape1 = indice_pool(net, stride=2, scope="Pool1")
			#net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
			
		with tf.variable_scope("112X112"), tf.device("/gpu:1"): # 1/2
			net, parameters = conv2D(net, kernel_size=3, stride=1, output_channel=128, rate=1, parameters=parameters,
						is_biases=True, is_shortcut=False, is_bottleneck=False, is_batch_norm=False, is_training=is_training, is_dialted=False, scope="conv1")
			net, parameters = conv2D(net, kernel_size=3, stride=1, output_channel=128, rate=1, parameters=parameters,
						is_biases=True, is_shortcut=False, is_bottleneck=False, is_batch_norm=False, is_training=is_training, is_dialted=False, scope="conv2")	
			net, indices2, output_shape2 = indice_pool(net, stride=2, scope="Pool2")
			#net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
			
		with tf.variable_scope("56X56"), tf.device("/gpu:2"): # 1/4
			net, parameters = conv2D(net, kernel_size=3, stride=1, output_channel=256, rate=1, parameters=parameters,
						is_biases=True, is_shortcut=False, is_bottleneck=False, is_batch_norm=False, is_training=is_training, is_dialted=False, scope="conv1")
			net, parameters = conv2D(net, kernel_size=3, stride=1, output_channel=256, rate=1, parameters=parameters,
						is_biases=True, is_shortcut=False, is_bottleneck=False, is_batch_norm=False, is_training=is_training, is_dialted=False, scope="conv2")	
			net, parameters = conv2D(net, kernel_size=3, stride=1, output_channel=256, rate=1, parameters=parameters,
						is_biases=True, is_shortcut=False, is_bottleneck=False, is_batch_norm=False, is_training=is_training, is_dialted=False, scope="conv3")			
			net, indices3, output_shape3 = indice_pool(net, stride=2, scope="Pool3")
			#net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
			
		with tf.variable_scope("28X28"), tf.device("/gpu:3"): # 1/8
			net, parameters = conv2D(net, kernel_size=3, stride=1, output_channel=512, rate=1, parameters=parameters,
						is_biases=True, is_shortcut=False, is_bottleneck=False, is_batch_norm=False, is_training=is_training, is_dialted=False, scope="conv1")
			net, parameters = conv2D(net, kernel_size=3, stride=1, output_channel=512, rate=1, parameters=parameters,
						is_biases=True, is_shortcut=False, is_bottleneck=False, is_batch_norm=False, is_training=is_training, is_dialted=False, scope="conv2")	
			net, parameters = conv2D(net, kernel_size=1, stride=1, output_channel=512, rate=1, parameters=parameters,
						is_biases=True, is_shortcut=False, is_bottleneck=False, is_batch_norm=False, is_training=is_training, is_dialted=False, scope="conv3")			
			net, indices4, output_shape4 = indice_pool(net, stride=2, scope="Pool4")
			#net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
			
		with tf.variable_scope("14X14"), tf.device("/gpu:0"): # 1/16
			net, parameters = conv2D(net, kernel_size=3, stride=1, output_channel=512, rate=1, parameters=parameters,
						is_biases=True, is_shortcut=False, is_bottleneck=False, is_batch_norm=False, is_training=is_training, is_dialted=False, scope="conv1")
			net, parameters = conv2D(net, kernel_size=3, stride=1, output_channel=512, rate=1, parameters=parameters,
						is_biases=True, is_shortcut=False, is_bottleneck=False, is_batch_norm=False, is_training=is_training, is_dialted=False, scope="conv2")	
			net, parameters = conv2D(net, kernel_size=1, stride=1, output_channel=512, rate=1, parameters=parameters,
						is_biases=True, is_shortcut=False, is_bottleneck=False, is_batch_norm=False, is_training=is_training, is_dialted=False, scope="conv3")			
			net, indices5, output_shape5 = indice_pool(net, stride=2, scope="Pool5")
			#net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
			
		with tf.variable_scope("7X7"), tf.device("/gpu:1"): # 1/32 Fully Connected Layers
			net, parameters = conv2D(net, kernel_size=7, stride=1, output_channel=4096, rate=1, parameters=parameters,
						is_biases=True, is_shortcut=False, is_bottleneck=False, is_batch_norm=False, is_training=is_training, is_dialted=False, padding="VALID", scope="conv1")
			net, parameters = conv2D(net, kernel_size=1, stride=1, output_channel=4096, rate=1, parameters=parameters,
						is_biases=True, is_shortcut=False, is_bottleneck=False, is_batch_norm=False, is_training=is_training, is_dialted=False, scope="conv2")	
			net, parameters = conv2D(net, kernel_size=1, stride=1, output_channel=10, rate=1, parameters=parameters,
						is_biases=True, is_shortcut=False, is_bottleneck=False, is_batch_norm=False, is_training=is_training, is_dialted=False, scope="conv3")			
	return net, parameters

def main(argv=None):
#===========#
#	Data	#
#===========#
	print("shape of training data : {Shape}" .format(Shape = np.shape(mnist.train.images)))
	print("shape of training target : {Shape}" .format(Shape = np.shape(mnist.train.labels)))
	print("shape of testing data : {Shape}" .format(Shape = np.shape(mnist.test.images)))
	print("shape of testing target : {Shape}" .format(Shape = np.shape(mnist.test.labels)))

# Placeholder
	xs = tf.placeholder(tf.float32, [None, 784])  # 28*28 = 784
	ys = tf.placeholder(tf.float32, [None, 10])
	learning_rate = tf.placeholder(tf.float32)
	is_training = tf.placeholder(tf.bool)
	
# Data Preprocessing
	xs_image = tf.reshape(xs, [-1, 28, 28, 1])
	xs_image = tf.image.resize_images(xs_image, [224, 224])	
	
#===========#
#	Graph	#
#===========#
	net = xs_image
	output, parameters = SegNet_VGG_16(net, is_training=is_training)
	
	for i in range(np.shape(parameters)[0]):
		print i, parameters[i].get_shape()
	
	#pdb.set_trace()
	
	prediction = tf.reshape(output, [-1, 10])
	
#=======================#
#	Training Strategy	#
#=======================#
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(ys,1), logits=prediction)
	train_step = tf.train.MomentumOptimizer(learning_rate, momentum=tf.constant(0.9)).minimize(cross_entropy)
	#train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
	#train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
	init = tf.global_variables_initializer()

#===============#
#	Collection	#
#===============#	
	all_weights_collection = tf.get_collection("all_weights", scope=None)
	mean_collection = tf.get_collection("mean", scope=None)
	var_collection = tf.get_collection("var", scope=None)
	scale_collection = tf.get_collection("scale", scope=None)
	shift_collection = tf.get_collection("shift", scope=None)
	
#===============#
#	Session Run	#
#===============#
	with tf.Session() as sess: 
		sess.run(init)
		lr = 0.00001
		# training
		for i in range(2000): 
			batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
			
			_, CrossEntropy, Prediction = sess.run([train_step, cross_entropy, prediction], 
													feed_dict={xs: batch_xs, ys: batch_ys, learning_rate: lr, is_training: True})
			print("Batch Iteration : {Iter}" .format(Iter = i))
			print("	Target Class  : {Target}".format(Target = np.argmax(batch_ys, 1)))
			print("	Predict Class : {y_pre}".format(y_pre = np.argmax(Prediction, 1)))
			print("	Cross Entropy : {CrossEntropy}".format(CrossEntropy = np.mean(CrossEntropy)))
			print("	Learning Rate : {LearningRate}".format(LearningRate = lr))
		
		# testing 
		test_accuracy = compute_accuracy(xs, ys, is_training, prediction, mnist.test.images, mnist.test.labels, sess)
		print("Test Accuracy = {Test_Accuracy}".format(Test_Accuracy=test_accuracy))
		
		#pdb.set_trace()
		
		# save weights
		#saver = tf.train.Saver()
		#save_path = saver.save(sess, "/VGG16_mnist_model.ckpt")
		#print("Model saved in file: %s" % save_path)
	
if __name__ == "__main__":
	tf.app.run()


