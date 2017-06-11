import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# %matplotlib inline

data = np.loadtxt('./housing.data', delimiter=',', dtype=np.float32)

# 마지막 열을 label로 하고 나머지는 feature로 구분함

print(data.shape)
features = data[:, :-1]
labels = data[:, [-1]]

print(features.shape)
print(labels.shape)

mu = np.mean(features, axis=0)
sigma = np.std(features, axis=0)
features = (features - mu) / sigma

n_training_samples = features.shape[0]
n_dim = features.shape[1] + 1

features = np.c_[np.ones(n_training_samples), features]

rnd_indices = np.random.rand(len(features)) < 0.80

train_x = features[rnd_indices]
train_y = labels[rnd_indices]
test_x = features[~rnd_indices]
test_y = labels[~rnd_indices]

learning_rate = 0.1
training_epochs = 1000
cost_history = np.empty(shape=[1], dtype=float)

X = tf.placeholder(tf.float32, [None, n_dim])
Y = tf.placeholder(tf.float32, [None, 1])
W = tf.Variable(tf.ones([n_dim, 1]))

init = tf.global_variables_initializer()

Y_Hat = tf.matmul(X, W)
cost = tf.reduce_mean(tf.square(Y_Hat - Y))
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

sess = tf.Session()
sess.run(init)

for epoch in range(training_epochs):
    sess.run(training_step, feed_dict={X: train_x, Y: train_y})
    c = sess.run(cost, feed_dict={X: train_x, Y: train_y})
    cost_history = np.append(cost_history, c)
    w = sess.run(W)
    print(epoch, c, w)


plt.plot(range(len(cost_history)), cost_history)
plt.axis([0, training_epochs, 0, np.max(cost_history)])
plt.show()


pred_y = sess.run(Y_Hat, feed_dict={X: test_x})
mse = tf.reduce_mean(tf.square(pred_y - test_y))
print("MSE: %.4f" % sess.run(mse))

fig, ax = plt.subplots()
ax.scatter(test_y, pred_y)
ax.plot([test_y.min(), test_y.max()], [test_y.min(), test_y.max()], 'k--', lw=3)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

sess.close()