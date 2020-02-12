import numpy as np
import tensorflow as tf

a_data = np.random.rand(256, 128)
b_data = np.random.rand(128, 256)

a = tf.placeholder(tf.float32, [256, 128])
b = tf.placeholder(tf.float32, [128, 256])

print(a_data)
print(b_data)
x = tf.matmul(a, b)

with tf.device('/cpu'):
    with tf.Session() as sess:
        x_data = sess.run(x, {a: a_data, b: b_data})
        print(x_data)
