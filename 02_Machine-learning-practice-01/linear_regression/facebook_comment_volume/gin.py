import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# %matplotlib inline


# set up

data = None
for index in range(1, 6):
    tmp = np.loadtxt('./data/Training/Features_Variant_' + str(index) + '.csv', delimiter=',', dtype=np.float32)
    if data is None:
        data = tmp
    else:
        data = np.r_[data, tmp]

test = np.loadtxt('./data/Testing/Features_TestSet.csv', delimiter=',', dtype=np.float32)


print(data.shape)
training_features = data[:, :-1]
training_labels = data[:, [-1]]
print(training_features.shape)
print(training_labels.shape)

test_features = data[:, :-1]
test_labels = data[:, [-1]]

# normalize
all = np.r_[training_features, test_features]
mu = np.mean(all, axis=0)
sigma = np.std(all, axis=0)

sigma[sigma == 0] += 1
training_features = (training_features - mu) / sigma
test_features = (test_features - mu) / sigma

n_training_samples = training_features.shape[0]
training_features = np.c_[np.ones(n_training_samples), training_features]

n_testing_samples = test_features.shape[0]
test_features = np.c_[np.ones(n_testing_samples), test_features]

n_dim = training_features.shape[1]

cost_history = np.empty(shape=[1], dtype=float)

X = tf.placeholder(tf.float32, [None, n_dim])
Y = tf.placeholder(tf.float32, [None, 1])
W = tf.Variable(tf.ones([n_dim, 1]))

Y_Hat = tf.matmul(X, W)
cost = tf.reduce_mean(tf.square(Y_Hat - Y))

learning_rate = 0.05
training_epochs = 100
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

# training


with tf.Session() as sess:
    sess.run(init)

    for epoch in range(training_epochs):

        sess.run(training_step, feed_dict={X: training_features, Y: training_labels})
        c = sess.run(cost, feed_dict={X: training_features, Y: training_labels})
        cost_history = np.append(cost_history, c)
        w = sess.run(W)
        print(epoch, c, w)

    pred_y = sess.run(Y_Hat, feed_dict={X: test_features})
    mse = tf.reduce_mean(tf.square(pred_y - test_labels))
    print("MSE: %.4f" % sess.run(mse))

    fig, ax = plt.subplots()
    ax.scatter(test_labels, pred_y)
    ax.plot([test_labels.min(), test_labels.max()], [test_labels.min(), test_labels.max()], 'k--', lw=3)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show()
