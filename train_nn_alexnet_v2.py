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


image_width = 227
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

 # Input data.
X_t = tf.placeholder(shape = [ None, image_width,image_width,3],dtype = tf.float32, name = "X_t")
y_t = tf.placeholder(shape =[ None, n_y],dtype = tf.float32, name = "y_t")
phase = tf.placeholder(tf.bool, name='phase')
global_step = tf.placeholder(tf.int32, name='global_step')


# Variables.

#pre trained Constants
net_data = np.load(PRETRAINEDAXELNET).item()




# Model.
def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    
    
    if group==1:
        conv = convolve(input, kernel)
    else:
        input_groups =  tf.split(input, group, 3)   #tf.split(3, group, input)
        kernel_groups = tf.split(kernel, group, 3)  #tf.split(3, group, kernel) 
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(output_groups, 3)          #tf.concat(3, output_groups)
    return  tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])


#conv1
#conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
conv1W = tf.constant(net_data["conv1"][0])
conv1b = tf.constant(net_data["conv1"][1])
conv1_in = conv(X_t, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
conv1 = tf.nn.relu(conv1_in)

#lrn1
#lrn(2, 2e-05, 0.75, name='norm1')
radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
lrn1 = tf.nn.local_response_normalization(conv1,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)

#maxpool1
#max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)


#conv2
#conv(5, 5, 256, 1, 1, group=2, name='conv2')
k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
conv2W = tf.constant(net_data["conv2"][0])
conv2b = tf.constant(net_data["conv2"][1])
conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv2 = tf.nn.relu(conv2_in)


#lrn2
#lrn(2, 2e-05, 0.75, name='norm2')
radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
lrn2 = tf.nn.local_response_normalization(conv2,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)

#maxpool2
#max_pool(3, 3, 2, 2, padding='VALID', name='pool2')                                                  
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

#conv3
#conv(3, 3, 384, 1, 1, name='conv3')
k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
conv3W = tf.constant(net_data["conv3"][0])
conv3b = tf.constant(net_data["conv3"][1])
conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv3 = tf.nn.relu(conv3_in)

#conv4
#conv(3, 3, 384, 1, 1, group=2, name='conv4')
k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
conv4W = tf.constant(net_data["conv4"][0])
conv4b = tf.constant(net_data["conv4"][1])
conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv4 = tf.nn.relu(conv4_in)


#conv5
#conv(3, 3, 256, 1, 1, group=2, name='conv5')
k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
conv5W = tf.constant(net_data["conv5"][0])
conv5b = tf.constant(net_data["conv5"][1])
conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv5 = tf.nn.relu(conv5_in)

#maxpool5
#max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

#fc6
#fc(4096, name='fc6')
fc6W = tf.constant(net_data["fc6"][0])
fc6b = tf.constant(net_data["fc6"][1])
#fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(tf.prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)
fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, tf.cast((tf.reduce_prod(maxpool5.get_shape()[1:])), tf.int32)]), fc6W, fc6b)

#fc7
#fc(4096, name='fc7')
fc7W = tf.constant(net_data["fc7"][0])
fc7b = tf.constant(net_data["fc7"][1])
fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)

#fc8
#fc(1000, relu=False, name='fc8')
"""
fc8W = tf.constant(net_data["fc8"][0])
fc8b = tf.constant(net_data["fc8"][1])
fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)
"""
# greffe sur un AlexnET existant
h1 = dense_batch_relu(fc7, 100, phase,'layer1')
h2 = dense_batch_relu(h1, 100, phase, 'layer2')
logits = dense(h2, n_y, 'logits')

# Training computation.
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = y_t))

reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
loss = cost  + reg_constant * sum(reg_losses)

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
