#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 01:53:18 2019

@author: AliveIT
"""

import tensorflow as tf
import pylab as pl

tf.compat.v1.reset_default_graph()

graph = tf.get_default_graph()

x = tf.constant(2.0, name='input')
w = tf.Variable(0.8, name='weight')

y = tf.multiply(w, x, name='output')
y_ = tf.constant(5.0)

loss = (y - y_)**2

sess = tf.Session()
sess.run(tf.global_variables_initializer())

train_step = tf.train.GradientDescentOptimizer(learning_rate=0.025) \
             .minimize(loss)
function = []
losses = []

for i in range(20):
    sess.run(train_step)
    function.append(sess.run(y))
    losses.append(sess.run(loss))
    
pl.plot(function)
pl.plot(losses)
pl.legend(['function', 'losses'])
