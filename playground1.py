#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 02:58:17 2019

@author: AliveIT
"""

import numpy as np
import tensorflow as tf

"""
np.random.seed(101)
tf.set_random_seed(101)

rand_a = np.random.uniform(0, 100, (5,5))

rand_b = np.random.uniform(0, 100, (5,1))

a = tf.placeholder(tf.float32)

b = tf.placeholder(tf.float32)

add_op = a + b

mul_op = a * b

with tf.Session() as sess:
    add_result = sess.run(mul_op, feed_dict={a:rand_a, b:rand_b})
    print(add_result)
    
"""

n_features = 10
n_dense_neurons = 3

x = tf.placeholder(tf.float32, (None, n_features))

W = tf.Variable(tf.random_normal([n_features, n_dense_neurons]))

b = tf.Variable(tf.ones([n_dense_neurons]))