import numpy as np
import tensorflow as tf
import time
import sys
import math
import gc
import tflearn
"""
('Train Accuracy:', 0.77841365)
('Test Accuracy:', 0.10850229)

"""

OUTFILETEST = '/root/kaggle/dogs/cache_test.npy'
OUTFILETRAIN = '/root/kaggle/dogs/cache_train.npy'
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

# Real-time data augmentation
img_aug = tflearn.ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_crop([image_width, image_width], padding=4)
# Real-time data preprocessing
img_prep = tflearn.ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()       
        
def prepare_data(X,y):
	X = X.tolist()
	y = y.tolist()
	X = np.asarray(X, dtype=np.float32)
	y = np.asarray(y, dtype=np.float32)
	return X,y


# Add ops to save and restore all the variables.

data_train = np.load(OUTFILETRAIN)

y1 = data_train[1][0]
n_y = y1.shape[0]
n_x = image_size

X_train,y_train = prepare_data(data_train[0],data_train[1])
print X_train.shape
print y_train.shape

X_t = tf.placeholder(shape = [ None, image_width,image_width,3],dtype = tf.float32, name = "X_t")
y_t = tf.placeholder(shape =[ None, n_y],dtype = tf.float32, name = "y_t")
phase = tf.placeholder(tf.bool, name='phase')

inp_data = tflearn.input_data(shape = [None, image_width,image_width,3], placeholder = X_t, data_preprocessing=img_prep)
reshape_X = tf.reshape(inp_data, [tf.shape(X_t)[0], image_size])
h1 = dense_batch_relu(reshape_X, 100, phase,'layer1')
h2 = dense_batch_relu(reshape_X, 50, phase,'layer2')
h3 = dense_batch_relu(h1, 25, phase, 'layer3')
logits = dense(h2, n_y, 'logits')



saver = tf.train.Saver()


# Ensures that we execute the update_ops before performing the train_step
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_t, 1), tf.argmax(logits, 1)),'float32'))




with tf.Session() as sess:
	# Restore variables from disk.
  	saver.restore(sess, TRAINEDMODEL)
	# Before training, run this:
	tflearn.is_training(False, session=sess)
	print ("Train Accuracy:", accuracy.eval({X_t: X_train, y_t: y_train,phase: 0}))

del data_train
del X_train
del y_train
gc.collect()

data_test = np.load(OUTFILETEST)
X_test,y_test = prepare_data(data_test[0],data_test[1])
print X_test.shape
print y_test.shape

with tf.Session() as sess:
	# Restore variables from disk.
  	saver.restore(sess, TRAINEDMODEL)
	# Before training, run this:
	tflearn.is_training(False, session=sess)
	print ("Test Accuracy:", accuracy.eval({X_t: X_test, y_t: y_test,phase: 0}))

# print ("Test Accuracy:", accuracy.eval({X_t: X_test, y_t: y_test}))
