import tensorflow as tf
import numpy as np

data = np.loadtxt('data/housing.txt', unpack=True, dtype='float32')
x = data[0:-1]
y = data[-1]

# normalize
x_transpose = np.transpose(x)
x_min = np.transpose(np.min(x, 1))
x_max = np.transpose(np.max(x, 1))
x = np.transpose((x_transpose - x_min) / (x_max - x_min))

W = tf.Variable(tf.random_uniform([1, len(x)], -1, 1))
b = tf.Variable(tf.random_uniform([1], -1, 1))

h = tf.matmul(W, x) + b
cost = tf.reduce_mean(tf.square(h - y))

optimizer = tf.train.GradientDescentOptimizer(tf.Variable(0.1))
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1001):
    sess.run(train)
    if i % 10 == 0:
        print(i, sess.run(cost), sess.run(W), sess.run(b))
