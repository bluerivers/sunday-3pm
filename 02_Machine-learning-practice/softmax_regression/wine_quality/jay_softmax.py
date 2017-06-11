import tensorflow as tf
import numpy as np


def normalize(mat):
    transposed = np.transpose(mat)
    mat_mean = np.transpose(np.mean(mat, 1))
    mat_std = np.transpose(np.std(mat, 1))
    mat = np.transpose((transposed - mat_mean) / mat_std)
    return mat


def one_hot(arr, n):
    mat = np.zeros((len(arr), n))
    mat[np.arange(len(arr)), arr.astype(int)] = 1
    return mat

num_of_class = 11

data = np.loadtxt('winequality-red.csv', unpack=True, dtype='float32', delimiter=';')
data = np.transpose(data)
np.random.shuffle(data)
train_count = int(len(data) * 0.8)
data_train = np.transpose(data[:train_count])
data_test = np.transpose(data[train_count:])
x_train = np.transpose(normalize(data_train[0:-1]))
y_train = one_hot(data_train[-1], num_of_class)
x_test = np.transpose(normalize(data_test[0:-1]))
y_test = one_hot(data_test[-1], num_of_class)

num_of_feature = len(x_train[0])

x = tf.placeholder("float", [None, num_of_feature])
y = tf.placeholder("float", [None, num_of_class])

W = tf.Variable(tf.random_uniform([num_of_feature, num_of_class], -1, 1))
b = tf.Variable(tf.random_uniform([num_of_class], -1, 1))

h = tf.nn.softmax(tf.matmul(x, W) + b)
cost = tf.reduce_mean(-tf.reduce_mean(y * tf.log(h), axis=1))

optimizer = tf.train.AdamOptimizer(tf.Variable(0.1))
train = optimizer.minimize(cost)

# training
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(1001):
    sess.run(train, feed_dict={x: x_train, y: y_train})
    if i % 100 == 0:
        print(sess.run(cost, feed_dict={x: x_train, y: y_train}))

#testing
predicted = sess.run(tf.arg_max(h, 1), feed_dict={x: x_test})
measured = sess.run(tf.arg_max(y_test, 1))

print("test:", np.sum(predicted == measured) / len(measured))