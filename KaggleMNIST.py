from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD
import csv
import numpy as np

# input image dimensions
img_rows, img_cols = 28, 28
# the MNIST images are black and white
img_channels = 1
num_classes = 10
yTrain = np.ones((1,42000))
xTrain = np.ones((42000,img_cols,img_rows))
counter = 0
skip = True

trainFile = open('train.csv')
csv_file = csv.reader(trainFile)
for row in csv_file:
	if (skip == True):
		skip = False
		continue
	yTrain[0,counter] = row[0]
	temp = np.ones((1,784))
	for num in range(1,784):
		temp[0,num - 1] = row[num]
	temp = np.reshape(temp, (img_rows,img_cols))
	xTrain[counter,:,:] = temp
	counter = counter + 1

testFile = open('test.csv')
csv_file2 = csv.reader(testFile)
yTest = np.ones((1,28000))
xTest = np.ones((28000,img_cols,img_rows))
skip2 = True
counter2 = 0
for row in csv_file2:
	if (skip2 == True):
		skip2 = False
		continue
	yTest[0,counter2] = row[0]
	temp = np.ones((1,784))
	for num in range(1,784):
		temp[0,num - 1] = row[num]
	temp = np.reshape(temp, (img_rows,img_cols))
	xTest[counter2,:,:] = temp
	counter2 = counter2 + 1

model = Sequential()
model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(img_channels, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(Dropout(0.05))

model.add(Convolution2D(64, 3, 3, border_mode='valid'))
model.add(Activation('relu'))
model.add(Dropout(0.10))

model.add(Dense(num_classes))
model.add(Activation('softmax'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=32, nb_epoch=1,validation_data=(X_test, Y_test),shuffle=True)