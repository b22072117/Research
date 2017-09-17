import tensorflow as tf
import numpy as np
import scipy.ndimage
import pdb
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#===============#
#	Parameter	#
#===============#
BATCH_SIZE = 20

#===========#
#	Define	#
#===========#
def compute_accuracy(sess, xs, ys, is_training, predicton, v_xs, v_ys):
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

def conv2D(	net, 
			kernel_size=3, 
			stride=1, 
			internal_channel=64, 
			output_channel=64,
			rate=1,
			initializer=tf.contrib.layers.variance_scaling_initializer(),
			is_shortcut=False, # For Residual
			is_bottleneck=False, # For Residual
			is_batch_norm=True, # For Batch Normalization
			is_training=True, # For Batch Normalization
			is_dialted=False, # For Dialted Convolution
			scope="conv"):
			
	with tf.variable_scope(scope):
		input_channel = net.get_shape()[-1]
		if is_shortcut:
			with tf.variable_scope("shortcut"):
				if stride != 1:
					weights = tf.get_variable("weights", [1, 1, input_channel, output_channel], tf.float32, initializer=initializer)
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
			tf.add_to_collection("all_weights", weights)
			if is_dialted == False:
				net = tf.nn.conv2d(net, weights, strides=[1, stride, stride, 1], padding="SAME")
			else:
				net = tf.nn.conv2d(net, weights, rate=rate, padding="SAME")
			if is_batch_norm == True:
				net = batch_norm(net, output_channel, is_training=is_training)
			net = tf.nn.relu(net)
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

	
def residual_50(net, is_training, reuse=None, scope="Residual_50"):
	with tf.variable_scope(scope, reuse=reuse):
		with tf.variable_scope("conv1"):
			net = conv2D(net, kernel_size=7, stride=2, output_channel=64, is_training=is_training)
		with tf.variable_scope("conv2_x"), tf.device("/gpu:0"):
			net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
			net = conv2D(net, stride=1, internal_channel=64, output_channel=256, 
							is_shortcut=True, is_bottleneck=True, is_training=is_training, scope="C1")
							
			net = conv2D(net, stride=1, internal_channel=64, output_channel=256, 
							is_shortcut=True, is_bottleneck=True, is_training=is_training, scope="C2")
							
			net = conv2D(net, stride=1, internal_channel=64, output_channel=256, 
							is_shortcut=True, is_bottleneck=True, is_training=is_training, scope="C3")
							
		with tf.variable_scope("conv3_x"), tf.device("/gpu:1"):
			net = conv2D(net, stride=2, internal_channel=128, output_channel=512, 
							is_shortcut=True, is_bottleneck=True, is_training=is_training, scope="C1")
							
			net = conv2D(net, stride=1, internal_channel=128, output_channel=512, 
							is_shortcut=True, is_bottleneck=True, is_training=is_training, scope="C2")
							
			net = conv2D(net, stride=1, internal_channel=128, output_channel=512, 
							is_shortcut=True, is_bottleneck=True, is_training=is_training, scope="C3")
							
		with tf.variable_scope("conv4_x"), tf.device("/gpu:2"):
			net = conv2D(net, stride=2, internal_channel=256, output_channel=1024, 
							is_shortcut=True, is_bottleneck=True, is_training=is_training, scope="C1")
							
			net = conv2D(net, stride=1, internal_channel=256, output_channel=1024, 
							is_shortcut=True, is_bottleneck=True, is_training=is_training, scope="C2")
							
			net = conv2D(net, stride=1, internal_channel=256, output_channel=1024, 
							is_shortcut=True, is_bottleneck=True, is_training=is_training, scope="C3")
							
		with tf.variable_scope("conv5_x"), tf.device("/gpu:3"):
			net = conv2D(net, stride=2, internal_channel=512, output_channel=2048, 
							is_shortcut=True, is_bottleneck=True, is_training=is_training, scope="C1")
							
			net = conv2D(net, stride=1, internal_channel=512, output_channel=2048, 
							is_shortcut=True, is_bottleneck=True, is_training=is_training, scope="C2")
							
			net = conv2D(net, stride=1, internal_channel=512, output_channel=2048, 
							is_shortcut=True, is_bottleneck=True, is_training=is_training, scope="C3")
							
		with tf.variable_scope("Average_Pooling"):
			net = tf.reduce_mean(net, [1, 2], keep_dims=True)
		with tf.variable_scope("FC1"):
			net = conv2D(net, kernel_size=1, stride=1, output_channel=10, is_training=is_training)
	return net

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
	
	#train_output = residual_50(net, is_training=is_training)
	#train_prediction = tf.reshape(train_output, [-1, 10])
	#
	#test_output = residual_50(net, is_training=is_training, reuse=True)
	#test_prediction = tf.reshape(test_output, [-1, 10])
	
	output = residual_50(net, is_training=is_training)
	prediction = tf.reshape(output, [-1, 10])
	
#=======================#
#	Training Strategy	#
#=======================#
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(ys,-1), logits=prediction)
	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
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
	sess = tf.Session()
	sess.run(init)
	lr = 1e-1
	Past_CrossEntropy = np.zeros([1])
	for i in range(2000):
		
		batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
		_, CrossEntropy, Prediction = sess.run([train_step, cross_entropy, prediction], 
												feed_dict={xs: batch_xs, ys: batch_ys, learning_rate: lr, is_training: True})
		
		print("Batch Iteration : {Iter}" .format(Iter = i))
		print("	Target Class  : {Target}".format(Target = np.argmax(batch_ys, -1)))
		print("	Predict Class : {y_pre}".format(y_pre = np.argmax(Prediction, -1)))
		print("	Cross Entropy : {CrossEntropy}".format(CrossEntropy = np.mean(CrossEntropy)))
		print("	Learning Rate : {LearningRate}".format(LearningRate = lr))
		
	#pdb.set_trace()
	
	test_accuracy = compute_accuracy(sess, xs, ys, is_training, prediction, mnist.test.images, mnist.test.labels)
	print("Test Accuracy = {Test_Accuracy}".format(Test_Accuracy=test_accuracy))
	
	#pdb.set_trace()
	
if __name__ == '__main__':
	tf.app.run()




