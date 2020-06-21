#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 01:57:40 2020

@author: Manzanita
"""

# https://towardsdatascience.com/mnist-cnn-python-c61a5bce7a19

import keras
from keras.datasets import mnist 
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

(train_X,train_Y), (test_X,test_Y) = mnist.load_data()

# %%

train_X = train_X.reshape(-1, 28,28, 1)
test_X = test_X.reshape(-1, 28,28, 1)

train_X = train_X.astype('float32')
test_X = test_X.astype('float32')
train_X = train_X / 255
test_X = test_X / 255

train_Y_one_hot = to_categorical(train_Y)
test_Y_one_hot = to_categorical(test_Y)

a = []
for i in range(10):
    b = []
    for j in range(10):
        c=1 if i==j else 0
        b.append(c)
    a.append(b)
    
train_Y_one_hot_2 = np.identity(10)[train_Y,np.newaxis].reshape(-1,10)

# %%
model = Sequential()
model.add(Conv2D(64, (3,3), input_shape=(28, 28, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(64))
model.add(Dense(10))
model.add(Activation('softmax'))
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])


# %%

model.fit(train_X, train_Y_one_hot, batch_size=64, epochs=3)

# %%
pred = np.argmax(model.predict(test_X),axis=1)

# %%
# Problembsp
case_3 = np.where(pred==3)[0][2]
plt.imshow(test_X[case_3].reshape(28,28))
plt.imshow(np.transpose(test_X[case_3]).reshape(28,28))
print(np.argmax(model.predict(test_X[case_3:(case_3+1)])))
print(np.argmax(model.predict(np.transpose(test_X[case_3:(case_3+1)]))))
