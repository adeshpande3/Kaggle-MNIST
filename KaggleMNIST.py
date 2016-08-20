from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD
import csv
import numpy as np

yTrain = np.ones((1,42000))
xTrain = np.ones((42000,28,28))
counter = 0
skip = True
# input image dimensions
img_rows, img_cols = 28, 28
# the MNIST images are black and white
img_channels = 1

file = open('train.csv')
csv_file = csv.reader(file)
for row in csv_file:
	if (skip == True):
		skip = False
		continue
	yTrain[0,counter] = row[0]
	temp = np.ones((1,784))
	for num in range(1,784):
		temp[0,num - 1] = row[num]
	temp = np.reshape(temp, (28,28))
	xTrain[counter,:,:] = temp
	counter = counter + 1
print yTrain


#model = Sequential()
#model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(img_channels, img_rows, img_cols)))
#model.add(Activation('relu'))
#model.add(Dropout(0.05))

#model.add(Convolution2D(64, 3, 3, border_mode='valid'))
#model.add(Activation('relu'))
#model.add(Dropout(0.10))

#sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(loss='categorical_crossentropy', optimizer=sgd)

#model.fit(X_train, Y_train, batch_size=32, nb_epoch=1)