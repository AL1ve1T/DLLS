import numpy as np
import tensorflow as tf

X = tf.placeholder(tf.float32, [None])
Y = tf.placeholder(tf.float32, [None])

w = tf.Variable(tf.ones([]), name="weight")
b = tf.Variable(tf.zeros([]), name="bias")

yhat = tf.add(tf.multiply(X, w), b)

loss = tf.reduce_mean(tf.square(yhat - Y))

x_data = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], np.float32)
y_data = 2.0*x_data+1.0

optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print(sess.run(w))

    for i in range(500):
        sess.run(optimizer, {X:x_data, Y:y_data})
        print(sess.run(w))
