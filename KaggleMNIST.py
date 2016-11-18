#############

# This program is a simple digit classifier using the MNIST dataset. 
# I will be using the Keras neural network library with a Theano 
# backend. I will be submitting this to the competition on Kaggle.
# This competition has 42,000 training images and 28,000 test
# images. At the end of the program, a text file (csv format) named results
# will be created. It will have ImageId in one column (numbers from
# 1 to 28,000), and the prediction labels for each test image
# in another column. 

#############

# Imports of the different Keras layers and optimizers, as well as
# importing numpy and csv
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD
from keras.utils import np_utils
import csv
import numpy as np
import sys

# VARIABLES
img_rows, img_cols = 28, 28
img_channels = 1
num_pixels = img_cols*img_rows
num_classes = 10
num_trainImages = 42000
num_testImages = 28000

# CREATE TRAINING DATASET
yTrain = np.ones((num_trainImages))
xTrain = np.ones((num_trainImages,img_channels,img_cols,img_rows))
counter = 0
# We want to skip the first row in the csv file because it just 
# contains column header
skip = True

# train.csv should be in the same folder as this file
trainFile = open('train.csv')
csv_file = csv.reader(trainFile)
for row in csv_file:
	if (skip == True):
		skip = False
		continue
	yTrain[counter] = row[0]
	temp = np.ones((1,num_pixels))
	for num in range(1,num_pixels):
		temp[0,num - 1] = row[num]
	temp = (temp - np.mean(temp))/(np.max(temp) - np.min(temp))
	temp = np.reshape(temp, (img_rows,img_cols))
	xTrain[counter,0,:,:] = temp
	counter = counter + 1

# CREATE TEST DATASET
yTest = np.ones((num_testImages))
xTest = np.ones((num_testImages,img_channels,img_cols,img_rows))
skip2 = True
counter2 = 0

testFile = open('test.csv')
csv_file2 = csv.reader(testFile)
for row in csv_file2:
	if (skip2 == True):
		skip2 = False
		continue
	yTest[counter2] = row[0]
	temp = np.ones((1,num_pixels))
	for num in range(1,num_pixels):
		temp[0,num - 1] = row[num]
	temp = (temp - np.mean(temp))/(np.max(temp) - np.min(temp))
	temp = np.reshape(temp, (img_rows,img_cols))
	xTest[counter2,0,:,:] = temp
	counter2 = counter2 + 1

# Convert class vectors to binary class matrices
yTrain = np_utils.to_categorical(yTrain, num_classes)
yTest = np_utils.to_categorical(yTest, num_classes)

# NETWORK ARCHITECTURE
model = Sequential()
model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(img_channels,img_rows, img_cols)))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3, border_mode='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(96, 3, 3, border_mode='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.15))
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# TRAINING
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
model.fit(xTrain, yTrain, batch_size=32, nb_epoch=8,validation_data=(xTest, yTest),shuffle=True)

results = np.zeros((num_testImages,2))
for num in range(1,num_testImages + 1):	
	results[num - 1,0] = num

# TESTING
temp = model.predict_classes( xTest, batch_size=32, verbose=1)
for num in range(0,num_testImages):	
	results[num,1] = temp[num]
# Results saved in this text file
np.savetxt('result.csv', results, delimiter=',', fmt = '%i')  
results = pd.np.array(results)
firstRow = [[0 for x in range(2)] for x in range(1)]
firstRow[0][0] = 'ImageId'
firstRow[0][1] = 'Label'
with open("result.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(firstRow)
    writer.writerows(results)