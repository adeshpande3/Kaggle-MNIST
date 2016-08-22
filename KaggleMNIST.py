from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD
from keras.utils import np_utils
import csv
import numpy as np

# input image dimensions
img_rows, img_cols = 28, 28
# the MNIST images are black and white
img_channels = 1
num_classes = 10
yTrain = np.ones((42000))
xTrain = np.ones((42000,img_channels,img_cols,img_rows))
counter = 0
skip = True

trainFile = open('train.csv')
csv_file = csv.reader(trainFile)
for row in csv_file:
	if (skip == True):
		skip = False
		continue
	yTrain[counter] = row[0]
	temp = np.ones((1,784))
	for num in range(1,784):
		temp[0,num - 1] = row[num]
	temp = np.reshape(temp, (img_rows,img_cols))
	xTrain[counter,0,:,:] = temp
	counter = counter + 1

testFile = open('test.csv')
csv_file2 = csv.reader(testFile)
yTest = np.ones((28000))
xTest = np.ones((28000,img_channels,img_cols,img_rows))
skip2 = True
counter2 = 0
for row in csv_file2:
	if (skip2 == True):
		skip2 = False
		continue
	yTest[counter2] = row[0]
	temp = np.ones((1,784))
	for num in range(1,784):
		temp[0,num - 1] = row[num]
	temp = np.reshape(temp, (img_rows,img_cols))
	xTest[counter2,0,:,:] = temp
	counter2 = counter2 + 1

# convert class vectors to binary class matrices
yTrain = np_utils.to_categorical(yTrain, num_classes)
yTest = np_utils.to_categorical(yTest, num_classes)

#Network Architecture
model = Sequential()
model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(img_channels,img_rows, img_cols)))

model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(64, 2, 2, border_mode='valid'))
model.add(Dropout(0.05))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Convolution2D(192, 3, 3, border_mode='valid'))
#model.add(Dropout(0.10))
#model.add(Activation('relu'))
#model.add(Convolution2D(256, 3, 3, border_mode='valid'))
#model.add(Activation('relu'))
model.add(Dropout(0.15))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

#Optimizers and Testing
sgd = SGD(lr=0.2, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.fit(xTrain, yTrain, batch_size=32, nb_epoch=1,validation_data=(xTest, yTest),shuffle=True)

#Saving predictions into a test file that can be uploaded to Kaggle
#NOTE: You have to add a header row before submitting the txt file
results = np.zeros((28000,2))
for num in range(1,28001):	
	results[num - 1,0] = num
temp = model.predict_classes(self, xTest, batch_size=32, verbose=1)
#temp is a 
for num in range(0,28000):	
	results[num,1] = temp[num]
np.savetxt('result.csv', results, delimiter=',',format='%.3e')  