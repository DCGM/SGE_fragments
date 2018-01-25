'''Trains a simple deep NN on the MNIST dataset.
Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import os
import subprocess
import sys
def checkGPU():
	freeGpu = subprocess.check_output('nvidia-smi -q | grep "Minor\|Processes" | grep "None" -B1 | tr -d " " | cut -d ":" -f2 | sed -n "1p"', shell=True)
	print(freeGpu)
	if len(freeGpu) == 0:
  		print ('No free GPU available!')
  		sys.exit(1)
	os.environ['CUDA_VISIBLE_DEVICES'] = freeGpu.decode().strip()


def setGPU(state):
	""" Turn GPU on or off. If True, GPU is on, if false, GPU is off.
	Needs to be called prior keras or tensorflow is first imported."""

	import tensorflow as tf
	from keras import backend as K

	checkGPU()

	num_cores = 1
	num_CPU = 1
	num_GPU = 0
	if state:
		num_GPU = 1

	config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,\
	        inter_op_parallelism_threads=num_cores, allow_soft_placement=True,\
	        device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})
	session = tf.Session(config=config)
	K.set_session(session)

setGPU(True)


import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

batch_size = 128
num_classes = 10
epochs = 20

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
