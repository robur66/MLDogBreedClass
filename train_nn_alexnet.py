import numpy as np
import tensorflow as tf
import time
import sys
import math
import os.path
import tflearn

"""

"""

OUTFILETRAIN = '/root/kaggle/dogs/cache_train.npy'
TRAINEDPARAMETERS = '/root/kaggle/dogs/trained_parameters.npy'
TRAINEDMODEL = '/root/kaggle/dogs/model.ckpt'
PRETRAINEDAXELNET = '/root/kaggle/dogs/bvlc_alexnet.npy'


image_width = 124 # 227
image_size = image_width * image_width * 3
starter_learning_rate = 0.001
# learning_rate = 0.000001
mini_batch_size = 128
epoch_control = 10
num_epochs = 5001
epoch_learning_rate_decay = 100
# L2 regularisation
reg_constant=0.08

# Real-time data augmentation

img_aug = tflearn.ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_crop([image_width, image_width], padding=4)

# Real-time data preprocessing
img_prep = tflearn.ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

def dense(x, size, scope):
	return tf.contrib.layers.fully_connected(x, size,activation_fn=None,scope=scope)
    
	


def dense_batch_relu(x, size, phase, scope):
    with tf.variable_scope(scope):
        h1 = tf.contrib.layers.fully_connected(x, size, 
                                               activation_fn=None,
                                               scope='dense')
        h2 = tf.contrib.layers.batch_norm(h1, 
                                          center=True, 
                                          scale=True, 
                                          is_training=phase,
                                          scope='bn')
        return tf.nn.relu(h2, 'relu')


data = np.load(OUTFILETRAIN)

m = data.shape[1]
X = data[0]
y = data[1]
x1 = X[0]
y1 = y[0]
n_y = y1.shape[0]

num_complete_minibatches = int(math.floor(m/mini_batch_size))


n_x = image_size


X_t = tf.placeholder(shape = [ None, image_width,image_width,3],dtype = tf.float32, name = "X_t")
y_t = tf.placeholder(shape =[ None, n_y],dtype = tf.float32, name = "y_t")
phase = tf.placeholder(tf.bool, name='phase')
global_step = tf.placeholder(tf.int32, name='global_step')

def convGroup(input, kernel, biases,  c_o, s_h, s_w,  padding="VALID", group=2):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    
    input_groups = tf.split(axis=3, num_or_size_splits=group, value=input)
    kernel_groups = tf.split(axis=3, num_or_size_splits=group, value=kernel)
    output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
    conv = tf.concat(axis=3, values=output_groups)
    return  tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])



 # Input data.
X_t = tf.placeholder(shape = [ None, image_width,image_width,3],dtype = tf.float32, name = "X_t")
y_t = tf.placeholder(shape =[ None, n_y],dtype = tf.float32, name = "y_t")
phase = tf.placeholder(tf.bool, name='phase')
global_step = tf.placeholder(tf.int32, name='global_step')


# Variables.

#pre trained Constants
net_data = np.load(PRETRAINEDAXELNET).item()
layerC1_weights_alex = tf.constant(net_data["conv1"][0])
layerC1_biases_alex = tf.constant(net_data["conv1"][1])
layerC2_weights_alex = tf.constant(net_data["conv2"][0])
layerC2_biases_alex = tf.constant(net_data["conv2"][1])
layerC3_weights_alex = tf.constant(net_data["conv3"][0])
layerC3_biases_alex = tf.constant(net_data["conv3"][1])
layerC4_weights_alex = tf.constant(net_data["conv4"][0])
layerC4_biases_alex = tf.constant(net_data["conv4"][1])
layerC5_weights_alex = tf.constant(net_data["conv5"][0])
layerC5_biases_alex = tf.constant(net_data["conv5"][1])	
layerF1_weights_alex = tf.constant(net_data["fc6"][0])
layerF1_biases_alex = tf.constant(net_data["fc6"][1])
layerF2_weights_alex = tf.constant(net_data["fc7"][0])
layerF2_biases_alex = tf.constant(net_data["fc7"][1])

print np.array(net_data["conv1"][0]).shape
print np.array(net_data["conv1"][1]).shape
print np.array(net_data["conv2"][0]).shape
print np.array(net_data["conv2"][1]).shape
print np.array(net_data["conv3"][0]).shape
print np.array(net_data["conv3"][1]).shape
print np.array(net_data["conv4"][0]).shape
print np.array(net_data["conv4"][1]).shape
print np.array(net_data["conv5"][0]).shape
print np.array(net_data["conv5"][1]).shape	
print np.array(net_data["fc6"][0]).shape
print np.array(net_data["fc6"][1]).shape
print np.array(net_data["fc7"][0]).shape
print np.array(net_data["fc7"][1]).shape


layerF3_weights_alex = tf.Variable(tf.truncated_normal([4096, 256], stddev=0.01))
layerF3_biases_alex = tf.Variable(tf.constant(1.0, shape=[256]))
layerOutput_weights_alex = tf.Variable(tf.truncated_normal([256, n_y], stddev=0.01))
layerOutput_biases_alex = tf.Variable(tf.constant(1.0, shape=[n_y]))

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Model.
shape_X_t = tf.shape(X_t)
conv1 = tf.nn.conv2d(X_t, layerC1_weights_alex, [1, 4, 4, 1], padding='SAME')
conv1 = tf.nn.relu(conv1 + layerC1_biases_alex)
shape_conv1 =  tf.shape(conv1)
radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
lrn1 = tf.nn.local_response_normalization(conv1,
	depth_radius=radius,
	alpha=alpha,
	beta=beta,
	bias=bias)
shape_lrn1 =  tf.shape(lrn1)
maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
shape_maxpool1 =  tf.shape(maxpool1)
conv2_in = convGroup(maxpool1, layerC2_weights_alex, layerC2_biases_alex, 256, 1, 1, padding="SAME")
conv2 = tf.nn.relu(conv2_in)
shape_conv2 =  tf.shape(conv2)
radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
lrn2 = tf.nn.local_response_normalization(conv2,
	depth_radius=radius,
	alpha=alpha,
	beta=beta,
	bias=bias)
shape_lrn2 =   tf.shape(lrn2)
maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
shape_maxpool2 =  tf.shape(maxpool2)
conv3 = tf.nn.conv2d(maxpool2, layerC3_weights_alex, [1, 1, 1, 1], padding='SAME')
conv3 = tf.nn.relu(conv3 + layerC3_biases_alex)
shape_conv3 =  tf.shape(conv3)
conv = tf.nn.depthwise_group_conv2d(conv3, layerC4_weights_alex, layerC4_biases_alex, strides, padding, name)
conv4_in = convGroup(conv3, layerC4_weights_alex, layerC4_biases_alex, 384, 1, 1, padding="SAME")
conv4 = tf.nn.relu(conv4_in)
shape_conv4 =  tf.shape(conv4)
conv5_in = convGroup(conv4, layerC5_weights_alex, layerC5_biases_alex, 256, 1, 1, padding="SAME")
conv5 = tf.nn.relu(conv5_in)
shape_conv5 =  tf.shape(conv5)
maxpool3 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
shape_maxpool3 =  tf.shape(maxpool3)
shape_mp3 = tf.shape(maxpool3)
reshape_maxpool3 = tf.reshape(maxpool3, [shape_mp3[0], shape_mp3[1] * shape_mp3[2] * shape_mp3[3]])
hiddenF1 = tf.nn.relu(tf.matmul(reshape_maxpool3, layerF1_weights_alex) + layerF1_biases_alex)
hiddenF2 = tf.nn.relu(tf.matmul(hiddenF1, layerF2_weights_alex) + layerF2_biases_alex)
dropout2 = tf.nn.dropout(hiddenF2, 0.5)
hiddenF3 = tf.nn.relu(tf.matmul(dropout2, layerF3_weights_alex) + layerF3_biases_alex)
dropout3 = tf.nn.dropout(hiddenF3, 0.5)
logits = tf.matmul(dropout3, layerOutput_weights_alex) + layerOutput_biases_alex


# Training computation.
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = y_t))

# learning rate decay 
learning_rate = tf.train.exponential_decay(starter_learning_rate, num_epochs, epoch_learning_rate_decay, 0.96, staircase=True)

# optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
# Ensures that we execute the update_ops before performing the train_step
	optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)

# Initialize all the variables
init = tf.global_variables_initializer()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

def prepare_mini_batch_with_augmenter(k):
	first = k * mini_batch_size
	last = min((k + 1) * mini_batch_size, m)
	X_train = np.asarray(X[k * mini_batch_size : (k + 1) * mini_batch_size].tolist(), dtype=np.float32) / 255
	y_train = np.asarray(y[k * mini_batch_size : (k + 1) * mini_batch_size].tolist(), dtype=np.float32)
	return X_train,y_train


previous_cost = 9999.0

with tf.Session() as sess:
# Run the initialization
	tmps1=time.clock()
	X_train, y_train = prepare_mini_batch_with_augmenter(0)
	print sess.run([
			shape_X_t,
			shape_conv1,
			shape_lrn1,
			shape_maxpool1,
			shape_conv2,
			shape_lrn2,
			shape_maxpool2,
			shape_conv2,
			shape_conv3,
			shape_conv4,
			shape_conv5,
			shape_maxpool3

		], feed_dict={X_t: X_train, y_t: y_train})
	# Before training, run this:
	tflearn.is_training(True, session=sess)
	if os.path.isfile('/root/kaggle/dogs/model.ckpt.index'):
		print "parameters loaded"
		saver.restore(sess, TRAINEDMODEL)
	else:
		print "parameters initialized"
		sess.run(init)
	tt = time.time()
	for epoch in range(num_epochs):
		tmps1=time.clock()
		epoch_cost = 0.0
		for k in range(num_complete_minibatches):
			X_train, y_train = prepare_mini_batch_with_augmenter(k)
			_ , minibatch_cost = sess.run([optimizer, loss], feed_dict={X_t: X_train, y_t: y_train,phase: 1, global_step: epoch})

		        epoch_cost += minibatch_cost * y_train.shape[0]/ m   
		tmps2=time.clock()
  			
  		if epoch % epoch_control == 0:
  			print ("Cost mean epoch %i: %f" % (epoch, epoch_cost))
  			print "execution time epoch = %f" %(tmps2-tmps1)
  			print "total execution time = %f\n" %(time.time() - tt)
  			if epoch_cost < previous_cost:
				previous_cost = epoch_cost
				save_path = saver.save(sess, TRAINEDMODEL)
  			
	#parameters = sess.run(parameters)
	
# np.save(TRAINEDPARAMETERS, parameters) 

# Load
# read_dictionary = np.load('my_file.npy').item()
