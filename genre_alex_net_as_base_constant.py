import pandas as pd
from datetime import time
from datetime import datetime
from datetime import timedelta
from sklearn import preprocessing
from sklearn.decomposition import PCA
import matplotlib.pylab as plt
import numpy as np
import gc
import os
from sklearn.cross_validation import StratifiedShuffleSplit
import tensorflow as tf
from sklearn import cross_validation
import pickle
import Image, numpy


image_size = 224 # 227 pour pretrained ?

FTRAIN = '/root/kaggle/painter/train_info.csv'
IMGPATH = '/root/kaggle/painter/train_1/'
THUMBNAILPATH = '/root/kaggle/painter/thumbnail/'
CACHEPATH = '/root/kaggle/painter/cache/'
TRAINED_VARIABLES = '/root/kaggle/painter/genre_alex_net.ckpt'

[os.remove(os.path.join(CACHEPATH,f)) for f in os.listdir(CACHEPATH)]

num_channels = 3 # grayscale = image_file.convert('RGB')

def image_to_thumbnail(filename):
    size = image_size, image_size
    infile = os.path.join(IMGPATH,filename)
    if (os.path.isfile(infile)):
        outfile = os.path.join(THUMBNAILPATH,filename)
        if (not os.path.isfile(outfile)):
		try:
		    im = Image.open(infile).convert('RGB')
		    im.thumbnail(size, Image.ANTIALIAS)
		    im.save(outfile, "JPEG")
		    return True
		except IOError:
		    return False
	else:
		return True
    else:
        return False

def loadInfo():
    df = pd.read_csv(FTRAIN)
    df['hasImage'] =  df["filename"].map(lambda filename: 1 if (image_to_thumbnail(filename)) else 0)
    return  df[df['hasImage'] == 1]


df = loadInfo()

print(len(df))
print(len(df['artist'].unique()))
print(len(df['style'].unique()))
print(len(df['genre'].unique()))
print(len(df['date'].unique()))
print(len(df['title'].unique()))
print(len(df['filename'].unique()))
filter = df["genre"] != ""
df = df[["genre","filename"]]
df = df.dropna()
print df['genre'].unique()
print(len(df['genre'].unique()))
print(len(df))
print df.groupby(["genre"]).size().sort_values()

def randomize(data,label):
   permutations = np.random.permutation(data.index)
   return data.reindex(permutations),label.reindex(permutations)

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])
          

labels = pd.get_dummies(df['genre'])
num_labels = len(df['genre'].unique())
batch_size = 128
test_batch_size = 500

def image_to_nparray(filename):
    infile = os.path.join(THUMBNAILPATH,filename)
    img = Image.open(infile)
    img_as_np = np.asarray(img.getdata()).reshape(img.size[1], img.size[0], num_channels) / 255
    x_left_pad =  (image_size - img.size[1]) // 2
    x_right_pad = image_size - x_left_pad - img.size[1]
    y_left_pad =  (image_size - img.size[0]) // 2
    y_right_pad = image_size - y_left_pad - img.size[0]
    img_as_np = np.lib.pad(img_as_np, ((x_left_pad,x_right_pad),(y_left_pad,y_right_pad),(0,0)),  'constant',constant_values=(0, 0))
    return img_as_np

def prepare_batch_data(df_train,df_labels,step):
    offset = (step * batch_size) % (df_train["filename"].shape[0] - batch_size) 
    return np.array(df_labels[offset:(offset + batch_size)]),np.array([image_to_nparray(filename) for filename in df_train[offset:(offset + batch_size)]["filename"]])

def prepare_test_data(df_test,df_labels,step):
    offset = (step * test_batch_size) % (df_test["filename"].shape[0] - test_batch_size)
    matfile = os.path.join(CACHEPATH,"test" + str(offset))
    matfilewithext = os.path.join(CACHEPATH,"test" + str(offset)+".npy")
    if (os.path.isfile(matfilewithext)):
    	matrice = numpy.load(matfilewithext)
    else:
    	matrice = np.array([image_to_nparray(filename) for filename in df_test[offset:(offset + test_batch_size)]["filename"]])
    	numpy.save(matfile,matrice)  
    return np.array(df_labels[offset:(offset + test_batch_size)]),matrice

def prepare_data(df_test):
    return np.array([image_to_nparray(filename) for filename in df_test["filename"]]).astype(np.float32)
    
    
#image_to_nparray("1.jpg")

X_train, X_test, y_train, y_test = cross_validation.train_test_split(df, labels, test_size=0.2)


train_size = len(X_train)

print train_size

def convGroup(input, kernel, biases,  c_o, s_h, s_w,  padding="VALID", group=2):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    
    input_groups = tf.split(3, group, input)
    kernel_groups = tf.split(3, group, kernel)
    output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
    conv = tf.concat(3, output_groups)
    return  tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])

graph = tf.Graph()

with graph.as_default():
	 # Input data.
	tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
	tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
	# tf_valid_dataset = tf.constant(valid_dataset)
	# tf_test_dataset = tf.constant(prepare_data(X_test))
	# tf_test_labels = tf.constant(np.array(y_test))
	tf_test_dataset = tf.placeholder(tf.float32, shape=(test_batch_size, image_size, image_size, num_channels))
	tf_test_labels = tf.placeholder(tf.float32, shape=(test_batch_size, num_labels))
	
	# Variables.

	#pre trained Variables
	net_data = np.load("bvlc_alexnet.npy").item()
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
	layerF3_weights_alex = tf.Variable(tf.truncated_normal([4096, 4096], stddev=0.01))
	layerF3_biases_alex = tf.Variable(tf.constant(1.0, shape=[4096]))
	layerOutput_weights_alex = tf.Variable(tf.truncated_normal([4096, num_labels], stddev=0.01))
	layerOutput_biases_alex = tf.Variable(tf.constant(1.0, shape=[num_labels]))
	
	# Add ops to save and restore all the variables.
	saver = tf.train.Saver()
	
	# Model.
	def model_alexnet_train(data):
	  conv1 = tf.nn.conv2d(data, layerC1_weights_alex, [1, 4, 4, 1], padding='SAME')
	  conv1 = tf.nn.relu(conv1 + layerC1_biases_alex)
	  radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
	  lrn1 = tf.nn.local_response_normalization(conv1,
          	depth_radius=radius,
          	alpha=alpha,
          	beta=beta,
          	bias=bias)
          maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
          conv2_in = convGroup(maxpool1, layerC2_weights_alex, layerC2_biases_alex, 256, 1, 1, padding="SAME")
	  conv2 = tf.nn.relu(conv2_in)
	  radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
	  lrn2 = tf.nn.local_response_normalization(conv2,
          	depth_radius=radius,
          	alpha=alpha,
          	beta=beta,
          	bias=bias)
          maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
          conv3 = tf.nn.conv2d(maxpool2, layerC3_weights_alex, [1, 1, 1, 1], padding='SAME')
	  conv3 = tf.nn.relu(conv3 + layerC3_biases_alex)
	  conv4_in = convGroup(conv3, layerC4_weights_alex, layerC4_biases_alex, 384, 1, 1, padding="SAME")
	  conv4 = tf.nn.relu(conv4_in)
	  conv5_in = convGroup(conv4, layerC5_weights_alex, layerC5_biases_alex, 256, 1, 1, padding="SAME")
	  conv5 = tf.nn.relu(conv5_in)
	  maxpool3 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
	  shape = maxpool3.get_shape().as_list()
	  print shape
	  reshape = tf.reshape(maxpool3, [shape[0], shape[1] * shape[2] * shape[3]])
	  hiddenF1 = tf.nn.relu(tf.matmul(reshape, layerF1_weights_alex) + layerF1_biases_alex)
	  dropout1 = tf.nn.dropout(hiddenF1, 0.5)
	  hiddenF2 = tf.nn.relu(tf.matmul(dropout1, layerF2_weights_alex) + layerF2_biases_alex)
	  dropout2 = tf.nn.dropout(hiddenF2, 0.5)
	  hiddenF3 = tf.nn.relu(tf.matmul(dropout2, layerF3_weights_alex) + layerF3_biases_alex)
	  dropout3 = tf.nn.dropout(hiddenF3, 0.5)
	  return tf.matmul(dropout3, layerOutput_weights_alex) + layerOutput_biases_alex
	  
	  
	# Model.
	def model_alexnet_test(data):
	  conv1 = tf.nn.conv2d(data, layerC1_weights_alex, [1, 4, 4, 1], padding='SAME')
	  conv1 = tf.nn.relu(conv1 + layerC1_biases_alex)
	  radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
	  lrn1 = tf.nn.local_response_normalization(conv1,
          	depth_radius=radius,
          	alpha=alpha,
          	beta=beta,
          	bias=bias)
          maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
          conv2 = tf.nn.conv2d(maxpool1, layerC2_weights_alex, [1, 1, 1, 1], padding='SAME')
	  conv2 = tf.nn.relu(conv2 + layerC2_biases_alex)
	  radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
	  lrn2 = tf.nn.local_response_normalization(conv2,
          	depth_radius=radius,
          	alpha=alpha,
          	beta=beta,
          	bias=bias)
          maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
          conv3 = tf.nn.conv2d(maxpool2, layerC3_weights_alex, [1, 1, 1, 1], padding='SAME')
	  conv3 = tf.nn.relu(conv3 + layerC3_biases_alex)
	  conv4 = tf.nn.conv2d(conv3, layerC4_weights_alex, [1, 1, 1, 1], padding='SAME')
	  conv4 = tf.nn.relu(conv4 + layerC4_biases_alex)
	  conv5 = tf.nn.conv2d(conv4, layerC5_weights_alex, [1, 1, 1, 1], padding='SAME')
	  conv5 = tf.nn.relu(conv5 + layerC5_biases_alex)
	  maxpool3 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
	  shape = maxpool3.get_shape().as_list()
	  reshape = tf.reshape(maxpool3, [shape[0], shape[1] * shape[2] * shape[3]])
	  hiddenF1 = tf.nn.relu(tf.matmul(reshape, layerF1_weights_alex) + layerF1_biases_alex)
	  hiddenF2 = tf.nn.relu(tf.matmul(hiddenF1, layerF2_weights_alex) + layerF2_biases_alex)
	  return tf.matmul(hiddenF2, layerOutput_weights_alex) + layerOutput_biases_alex
	  
	
	# Training computation.
	
	logits = model_alexnet_train(tf_train_dataset)

	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
	
	
	# Optimizer: set up a variable that's incremented once per batch and
	# controls the learning rate decay.
	global_step = tf.Variable(0)
	"""
	learning_rate = tf.train.exponential_decay(
	  0.001,                # Base learning rate.
	  global_step * batch_size,  # Current index into the dataset.
	  train_size,          # Decay step.
	  0.8,                # Decay rate.
	  staircase=True)
	"""
	# Optimizer.
	learning_rate = 0.01
	learning_momentum = 0.9
	optimizer = tf.train.MomentumOptimizer(learning_rate,learning_momentum).minimize(loss, global_step=global_step)
	# optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss)
	# optimizer = tf.train.AdadeltaOptimizer(learning_rate).minimize(loss)
	# Predictions for the training, validation, and test data.
	train_prediction = tf.nn.softmax(logits)
	test_prediction = tf.nn.softmax(model_alexnet_train(tf_test_dataset))
	def testPrediction(X_test,y_test):
		r = 0.0
		for step in range(len(X_test)//test_batch_size):
			test_labels,test_data = prepare_test_data(X_test,y_test,step)
			feed_dict = {tf_test_dataset : test_data}
			r += accuracy(session.run(test_prediction, feed_dict=feed_dict),test_labels)
		r = r/(len(X_test)//test_batch_size)
		return r


nb_epochs = 30 
num_steps = train_size//batch_size

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print('Initialized')
  if (os.path.isfile(TRAINED_VARIABLES)):
  	saver.restore(session, TRAINED_VARIABLES)
  	print("Model restored.")
  for epoch in range(nb_epochs):
      X_train,y_train = randomize(X_train,y_train)
      mean_train_loss = 0.0
      mean_train_accuracy = 0.0
      for step2 in range(num_steps):
	    batch_labels,batch_data = prepare_batch_data(X_train,y_train,step2)
	    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
	    _, l, predictions = session.run(
	      [optimizer, loss, train_prediction], feed_dict=feed_dict)
	    mean_train_loss += l
	    mean_train_accuracy += accuracy(predictions, batch_labels)
      offset = (step2 * batch_size) % (X_train["filename"].shape[0] - batch_size) 
      filenames = X_train[offset:(offset + batch_size)]["filename"]
      np.savetxt("labels.csv", batch_labels, delimiter=";")
      np.savetxt("predictions.csv", predictions, delimiter=";")
      np.savetxt("filename.csv", filenames,fmt=('%15s'))
      print('Minibatch mean train loss at epoch %d : %f' % (epoch, mean_train_loss/num_steps))
      print('Minibatch mean train accuracy: %.1f%%' % (mean_train_accuracy/num_steps))
      predictions = testPrediction(X_test,y_test)
      print('Test accuracy at epoch %d : %f' % (epoch, predictions))     
      save_path = saver.save(session, TRAINED_VARIABLES)
      print("Model saved in file: %s" % save_path)
# print('Validation accuracy: %.1f%%' % accuracy(test_prediction.eval(), tf_test_labels))
# print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
  
