#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 14:21:58 2019

@author: AliveIT
"""

import tensorflow as tf
from sklearn.datasets import load_boston
from sklearn.preprocessing import scale

data_X, data_Y = load_boston(True)

print(data_Y)

"""
print(data_X[0])
print(data_Y[0])
"""

data_X = scale(data_X)
data_Y = data_Y.reshape(len(data_Y), 1)


input_dim = 13

X = tf.placeholder(tf.float32, [None, input_dim])
Y = tf.placeholder(tf.float32, [None])

w = tf.Variable(tf.ones([input_dim, 10]))
b = tf.Variable(tf.zeros(10))
#data_Y = data_Y.reshape(len(data_Y), 1)

yhat = tf.add(tf.matmul(X, w), b)
loss = tf.reduce_mean(tf.square(yhat - Y))

optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(50):
        sess.run(optimizer, {X:data_X, Y:data_Y})
        print(sess.run(w))

