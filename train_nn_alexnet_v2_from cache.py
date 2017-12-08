import numpy as np
import tensorflow as tf
import time
import sys
import math
import os.path
import tflearn

"""

"""

INPUTYTRAIN = '/root/kaggle/dogs/cache_train_y.npy'
INPUTXTRAIN = '/root/kaggle/dogs/cache_train_x_axel_fc7.npy'
TRAINEDPARAMETERS = '/root/kaggle/dogs/trained_parameters.npy'
TRAINEDMODEL = '/root/kaggle/dogs/model.ckpt'
PRETRAINEDAXELNET = '/root/kaggle/dogs/bvlc_alexnet.npy'


image_width = 227
image_size = image_width * image_width * 3
starter_learning_rate = 0.01
# learning_rate = 0.000001
mini_batch_size = 128
epoch_control = 10
num_epochs = 5001
epoch_learning_rate_decay = 100
# L2 regularisation
reg_constant=0.08


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


y = np.load(INPUTYTRAIN)
X = np.load(INPUTXTRAIN)
m = y.shape[0]
n_x = X[0].shape[0]
y1 = y[0]
n_y = y1.shape[0]

num_complete_minibatches = int(math.floor(m/mini_batch_size))


X_t = tf.placeholder(shape = [ None, n_x],dtype = tf.float32, name = "X_t")
y_t = tf.placeholder(shape =[ None, n_y],dtype = tf.float32, name = "y_t")
phase = tf.placeholder(tf.bool, name='phase')
global_step = tf.placeholder(tf.int32, name='global_step')

# greffe sur un AlexnET existant
# dropout1 = tf.nn.dropout(X_t, 0.5)
h1 = dense_batch_relu(X_t, 100, phase,'layer1')
#dropout2 = tf.nn.dropout(h1, 0.5)
h2 = dense_batch_relu(h1, 100, phase, 'layer2')
#dropout3 = tf.nn.dropout(h2, 0.5)
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

def prepare_mini_batch(k):
	first = k * mini_batch_size
	last = min((k + 1) * mini_batch_size, m)
	X_train = np.asarray(X[first : last].tolist(), dtype=np.float32)
	y_train = np.asarray(y[first : last].tolist(), dtype=np.float32)
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
		for k in range(num_complete_minibatches + 1):
			X_train, y_train = prepare_mini_batch(k)
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
