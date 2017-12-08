import numpy as np
import tensorflow as tf
import time
import sys
import math
import gc
"""
('Train Accuracy:', 0.77841365)
('Test Accuracy:', 0.10850229)

fc7 - 256 - 120
('Train Accuracy:', 0.95450652)
('Test Accuracy:', 0.11344744)

fc7 - 100 - 60
('Train Accuracy:', 0.36920631)
('Test Accuracy:', 0.13154034)


"""

INPUTYTRAIN = '/root/kaggle/dogs/cache_train_y.npy'
INPUTXTRAIN = '/root/kaggle/dogs/cache_train_x_axel_fc7.npy'
INPUTYTEST = '/root/kaggle/dogs/cache_test_y.npy'
INPUTXTEST = '/root/kaggle/dogs/cache_test_x_axel_fc7.npy'
# TRAINEDPARAMETERS = '/root/kaggle/painter/trained_parameters.npy'
TRAINEDMODEL = '/root/kaggle/dogs/model.ckpt'

image_width = 124
image_size = image_width * image_width * 3

def dense(x, size, scope):
	return tf.contrib.layers.fully_connected(x, size,activation_fn=None,scope=scope)

def dense_batch_relu(x, size, phase, scope):
    with tf.variable_scope(scope):
        h1 = tf.contrib.layers.fully_connected(x, size, 
                                               activation_fn=None,
                                               scope='dense')
        h2 = tf.contrib.layers.batch_norm(h1, 
                                          center=True, scale=True, 
                                          is_training=phase,
                                          scope='bn')
        return tf.nn.relu(h2, 'relu')
  
        
def prepare_data(X,y):
	X = X.tolist()
	y = y.tolist()
	X = np.asarray(X, dtype=np.float32)
	y = np.asarray(y, dtype=np.float32)
	return X,y


# Add ops to save and restore all the variables.

y = np.load(INPUTYTRAIN)
X = np.load(INPUTXTRAIN)
m = y.shape[0]
n_x = X[0].shape[0]
y1 = y[0]
n_y = y1.shape[0]

X_train,y_train = prepare_data(X,y)
print X_train.shape
print y_train.shape

X_t = tf.placeholder(shape = [ None, n_x],dtype = tf.float32, name = "X_t")
y_t = tf.placeholder(shape =[ None, n_y],dtype = tf.float32, name = "y_t")
phase = tf.placeholder(tf.bool, name='phase')

# greffe sur un AlexnET existant
h1 = dense_batch_relu(X_t, 100, phase,'layer1')
h2 = dense_batch_relu(h1, 60, phase, 'layer2')
logits = dense(h2, n_y, 'logits')



saver = tf.train.Saver()


# Ensures that we execute the update_ops before performing the train_step
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_t, 1), tf.argmax(logits, 1)),'float32'))




with tf.Session() as sess:
	# Restore variables from disk.
  	saver.restore(sess, TRAINEDMODEL)
	# Before training, run this:
	print ("Train Accuracy:", accuracy.eval({X_t: X_train, y_t: y_train,phase: 0}))

del X
del y
del X_train
del y_train
gc.collect()

y = np.load(INPUTYTEST)
X = np.load(INPUTXTEST)
m = y.shape[0]
n_x = X[0].shape[0]
y1 = y[0]
n_y = y1.shape[0]

X_test,y_test = prepare_data(X,y)
print X_test.shape
print y_test.shape

with tf.Session() as sess:
	# Restore variables from disk.
  	saver.restore(sess, TRAINEDMODEL)
	# Before training, run this:
	print ("Test Accuracy:", accuracy.eval({X_t: X_test, y_t: y_test,phase: 0}))

# print ("Test Accuracy:", accuracy.eval({X_t: X_test, y_t: y_test}))
