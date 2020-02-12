#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 14:30:07 2019

Tensorflow example solution for training a simple regression model for the boston housing price dataset

@author: Benjamin Milde
"""

import tensorflow as tf
from sklearn.datasets import load_boston
from sklearn.preprocessing import scale
import numpy as np
import pylab as pl

data_X, data_Y = load_boston(True)

input_dim = data_X.shape[1]

print('input_dim:', input_dim)

print('X/Y example of the training data:')
print(data_X[0])
print(data_Y[0])

# This normalizes the mean to 0 and variance to 1
data_X = scale(data_X)

# You can check the mean and variance in numpy now.
# The mean vector should now be close to a 0, variance close to 1:
print(np.mean(data_X, axis=0))
print(np.var(data_X))

data_Y = data_Y.reshape(len(data_Y), 1)

# This splits the training data manually:
cutoff = 400 
data_X_train = data_X[:cutoff]
data_Y_train = data_Y[:cutoff]
data_X_test = data_X[cutoff:]
data_Y_test =data_Y[cutoff:]

# Construct the model:
# placeholders for data inputs:
X = tf.placeholder(tf.float32, [None, input_dim])
Y = tf.placeholder(tf.float32, [None, 1])

w = tf.Variable(tf.ones([input_dim,1]))
b = tf.Variable(tf.zeros([1]))

yhat = tf.add(tf.matmul(X,w),b)

## Hyperparameters - you can try to tune these to get better results
learning_rate = 0.02
epochs = 200

# Loss function is the squared error measure:
loss = tf.reduce_mean(tf.square(yhat-Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    test_losses = []
    train_losses = []
    for i in range(epochs):
        train_loss2, _ = sess.run([loss, optimizer], {X:data_X_train, Y:data_Y_train})

        test_loss = sess.run(loss, {X:data_X_test, Y:data_Y_test})
        train_loss = sess.run(loss, {X:data_X_train, Y:data_Y_train})

        print("train loss2:", train_loss2)
        print("train loss:", train_loss)
        
        print("test loss:", test_loss)
        
        test_losses.append(test_loss)
        train_losses.append(train_loss)
    
    pl.plot(test_losses)
    pl.plot(train_losses)
    pl.legend(["test","train"])
    pl.show()
    print(sess.run(b))