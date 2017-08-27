import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def normalize(mat):
    transposed = np.transpose(mat)
    mat_mean = np.transpose(np.mean(mat, 1))
    mat_std = np.transpose(np.std(mat, 1))
    mat = np.transpose((transposed - mat_mean) / mat_std)
    return mat

data = np.loadtxt('winequality-red.csv', unpack=True, dtype='float32', delimiter=';')
data = np.transpose(data)
np.random.shuffle(data)
train_count = int(len(data) * 0.8)
data_train = np.transpose(data[:train_count])
data_test = np.transpose(data[train_count:])
x_train = normalize(data_train[0:-1])
y_train = data_train[-1]
x_test = normalize(data_test[0:-1])
y_test = data_test[-1]

num_of_feature = len(x_train)

x = tf.placeholder("float", [num_of_feature, None])
y = tf.placeholder("float", [None])

W = tf.Variable(tf.random_uniform([1, num_of_feature], -1, 1))
b = tf.Variable(tf.random_uniform([1], -1, 1))

h = tf.matmul(W, x) + b
cost = tf.reduce_mean(tf.square(h - y))

optimizer = tf.train.AdamOptimizer(tf.Variable(0.1))
train = optimizer.minimize(cost)

# training
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(1001):
    sess.run(train, feed_dict={x: x_train, y: y_train})
print("cost:", sess.run(cost, feed_dict={x: x_train, y: y_train}), "W:", sess.run(W, feed_dict={x: x_train}), "b:", sess.run(b, feed_dict={x: x_train}))

#testing
predicted = sess.run(h, feed_dict={x: x_test})

print("test:", np.sum(np.round(predicted).astype(int)[0] == y_test.astype(int)) / len(y_test))

fig, ax = plt.subplots()
ax.scatter(y_test, predicted)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
