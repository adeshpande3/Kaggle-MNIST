from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import csv
import numpy as np

yTrain = np.ones((1,42000))
counter = 0
skip = 
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
	counter = counter + 1
print yTrain