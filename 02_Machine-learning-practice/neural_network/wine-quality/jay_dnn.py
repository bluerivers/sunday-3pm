import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def normalize(mat):
    transposed = np.transpose(mat)
    mat_mean = np.transpose(np.mean(mat, 1))
    mat_std = np.transpose(np.std(mat, 1))
    mat = np.transpose((transposed - mat_mean) / mat_std)
    return mat

data = np.loadtxt('data/winequality-red.csv', unpack=True, dtype='float32', delimiter=';')
data = np.transpose(data)

# tf.set_random_seed(777)   # reproducibility
np.random.shuffle(data)

train_count = int(len(data) * 0.8)
data_train = np.transpose(data[:train_count])
data_test = np.transpose(data[train_count:])
x_train = np.transpose(normalize(data_train[0:-1]))
y_train = np.reshape(data_train[-1], [-1, 1])
x_test = np.transpose(normalize(data_test[0:-1]))
y_test = np.reshape(data_test[-1], [-1, 1])

num_of_class = 11
num_of_feature = len(x_train[0])

X = tf.placeholder(tf.float32, [None, num_of_feature])
Y = tf.placeholder(tf.float32, [None, 1])

W1 = tf.Variable(tf.random_normal([num_of_feature, 16]))
b1 = tf.Variable(tf.random_normal([16]))
layer1 = tf.nn.relu(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal([16, num_of_class]))
b2 = tf.Variable(tf.random_normal([num_of_class]))
layer2 = tf.nn.relu(tf.matmul(layer1, W2) + b2)

W3 = tf.Variable(tf.random_normal([num_of_class, 8]))
b3 = tf.Variable(tf.random_normal([8]))
layer3 = tf.nn.relu(tf.matmul(layer2, W3) + b3)

W_final = tf.Variable(tf.random_normal([8, 1]))
b_final = tf.Variable(tf.random_normal([1]))
hypothesis = tf.matmul(layer3, W_final) + b_final

cost = tf.reduce_mean(tf.square(hypothesis - Y))
train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(3001):
    sess.run(train, feed_dict={X: x_train, Y: y_train})
    if step % 100 == 0:
        print(step, sess.run(cost, feed_dict={X: x_train, Y: y_train}))

### TEST ###
# x_test = x_train
# y_test = y_train
predicted = sess.run(hypothesis, feed_dict={X: x_test})
print("Hypothesis: ", predicted, "Accuracy: ", np.sum(np.round(predicted).astype(int) ==  y_test.astype(int)) / len(y_test))
fig, ax = plt.subplots()
ax.scatter(y_test, predicted)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()