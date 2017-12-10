# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 13:52:49 2017

@author: junfeng
"""

# import some library 
import os

##########
# change the image name by using os.rename and os.walk, os.listdir() functions
# the image name format as classname.index.jpg
##########

# =============================================================================
# PATH = 'F:\DeeplearningTest\data\IMG'
# =============================================================================

PATH = os.getcwd() + '/IMG'# this code used for hpc computation, if it run locally,
#  use above code and comment this code 

# =============================================================================
# for root, subdir, files in os.walk(PATH):
#     for classname in subdir:
#         datadir = os.path.join(PATH,classname)
#         datafiles = os.listdir(datadir)
#         i = 1
#         for file in datafiles:
#             filedir = os.path.join(datadir, file)
#             os.rename(filedir,os.path.join(datadir, classname +'.'+str(i)+'.jpg'))
#             i += 1
# =============================================================================
  
##############################################################################
            # resie image to be 512 * 512

import numpy as np
#import cv2
import scipy 
from scipy import ndimage
from matplotlib import pyplot as plt
data_list = []
classfold = os.listdir(PATH)            
for fold in classfold:
     print('loadind data + {}\n'.format(fold))
     i = 0
     fullClassfold = os.path.join(PATH,fold)
     files = os.listdir(fullClassfold)
     for file in files:
          fullfilefold = os.path.join(fullClassfold, file)
          img = ndimage.imread(fullfilefold)
          i += 1
          imgResized = scipy.misc.imresize(img, (256,256))
          data_list.append(imgResized)
     print ('---{}--{} images were found in total \n'.format(i, fold))
     
# =============================================================================
# pre-processing image and scale pixel value to be [0,1]
# =============================================================================
image_data_raw = np.array(data_list)
image_data = image_data_raw.astype('float32')
image_data /= 255
print(image_data.shape) 

# =============================================================================
# construct correspondance labels for iamge data
# =============================================================================
from keras.utils import np_utils

num_classes = 5
label = np.zeros(image_data_raw.shape[0]) # initialize value for label
label[0:36] = 0
label[36:66] = 1
label[66:96] = 2
label[96:126] = 3
label[126:156] = 4

names = ['bindweed', 'lambquarter', 'maize', 'sow thistle', 'volunter potato']
# convert class to be one-hot encoding
Y = np_utils.to_categorical(label, num_classes)
# =============================================================================
# prepareing training data and test data for model
# =============================================================================
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

x,y = shuffle(image_data, Y, random_state = 0)
trian_x, test_x, train_y,test_y = train_test_split(x,y,test_size = 0.2, 
                                                   stratify = label)

# =============================================================================
# shuffle dataset and split to be training and testing dataset
# =============================================================================
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import adam, SGD, RMSprop

input_shape = image_data[0].shape

model = Sequential()
model.add(Convolution2D(32,5,5,border_mode = 'same', input_shape = input_shape))

model.add(Activation ('relu'))

model.add(Convolution2D(32,3,3))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.5))

model.add(Convolution2D(64,3,3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(num_classes))
model.add(Activation('softmax'))

# comple model
model.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics=['accuracy'])
# =============================================================================
# trian model
# =============================================================================
import time # counting the processing time
time1 = time.time()
num_epoch = 40
hist = model.fit(trian_x,train_y,batch_size = 15, epochs = num_epoch,
                 verbose = 1, validation_data = (test_x, test_y))


time2 = time.time()
with open('precessingtime.txt','w') as f:
     f.write(str(time2-time1))

# =============================================================================
# draw plot to see the result of accuracy in training data and test data
# =============================================================================
train_loss =  hist.history['loss']
val_loss = hist.history['val_loss']

train_accuracy = hist.history['acc']
val_accuracy = hist.history['val_acc']

x = np.arange(num_epoch)
plt.figure()
plt.plot(x, train_loss, x, val_loss)
plt.legend(['train','val'])
plt.savefig('loss.png', dpi = 200)

plt.figure()
plt.plot(x, train_accuracy, x, val_accuracy)
plt.legend(['trian','val'])
plt.savefig('accuracy.png',dpi = 400)

# =============================================================================
# evaluation model
# =============================================================================
































