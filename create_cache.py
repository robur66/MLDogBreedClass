import os
import Image
Image.MAX_IMAGE_PIXELS = 100000000000
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
import scipy
from scipy import ndimage
import types

IMGPATH = '/root/kaggle/dogs/train/'
FDATA = '/root/kaggle/dogs/labels.csv'
OUTFILETEST = '/root/kaggle/dogs/cache_test'
OUTFILETRAIN = '/root/kaggle/dogs/cache_train'

mini_batch_size = 124
image_size = 227
num_channels = 3




def mini_batch_image(filename):
	infile = os.path.join(IMGPATH,filename[0] +'.jpg')
	if (os.path.isfile(infile)):
		try:
			image = np.array(ndimage.imread(infile, flatten=False,mode="RGB"))
			if (image.shape[0] * image.shape[1] * image.shape[2] < 400000000000):
				my_image = scipy.misc.imresize(image, size=(image_size,image_size))
			else:
				print my_image
				my_image = False
			return my_image
		except (IOError,MemoryError):
			print "memoryError"
			return False
	else:
		print infile 
		return False	
        
        

def prepare_images(mini_batch_X, mini_batch_Y):
	L = mini_batch_Y.shape[0]
	X = []
	y = []
	for l in range(0,L):
		img = mini_batch_image(mini_batch_X[l])
		if type(img) != types.BooleanType:
			X.append(img)
			y.append(mini_batch_Y[l])
	return (X, y)
	
def prepare_minibatch( X, y, file):
	m1 = X.shape[0]
	print X.shape
	# Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
	num_complete_minibatches = int(math.floor(m1/mini_batch_size)) # number of mini batches of size mini_batch_size in your partitionning
	print num_complete_minibatches
	for k in range(0, num_complete_minibatches):
		mini_batch = prepare_images(X[k * mini_batch_size : (k + 1) * mini_batch_size,0],y[k * mini_batch_size : (k + 1) * mini_batch_size,:])
		mini_batches.append(mini_batch)

	# Handling the end case (last mini-batch < mini_batch_size)
	if (m1 % mini_batch_size) != 0:
		mini_batch = prepare_images(X[num_complete_minibatches * mini_batch_size : m1,0],y[num_complete_minibatches * mini_batch_size : m1,:])
		mini_batches.append(mini_batch)

	np.save(file, mini_batches)
	
def prepare_data( X, y, file):
	m1 = X.shape[0]
	data = prepare_images(X,y)
	np.save( file, data )
	
	

# Step 1: Shuffle (X, Y)

# permutation = list(np.random.permutation(m))
# shuffled_X = filenames[permutation,:]
# shuffled_Y = labels[permutation,:]

df = pd.read_csv(FDATA)
df = df[["id","breed"]]
df = df.dropna()

labels = pd.get_dummies(df["breed"]).as_matrix()
filenames = df[["id"]].as_matrix()

# m = labels.shape[0]                  # number of training examples
mini_batches = []

print df["id"].shape
print labels.shape

X_train, X_test, y_train, y_test = train_test_split( filenames, labels, test_size=0.2)

print X_test.shape


prepare_data( X_train, y_train, OUTFILETRAIN)
prepare_data( X_test, y_test, OUTFILETEST)


