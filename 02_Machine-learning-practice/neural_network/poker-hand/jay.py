import tensorflow as tf
import numpy as np


def normalize(mat):
    transposed = np.transpose(mat)
    mat_mean = np.transpose(np.mean(mat, 1))
    mat_std = np.transpose(np.std(mat, 1))
    mat = np.transpose((transposed - mat_mean) / mat_std)
    return mat


# string to num
def convert_string(mat):
    new_mat = np.empty(mat.shape)
    for i in range(len(mat)):
        d, idx = np.unique(mat[i], return_inverse=True)
        new_mat[i] = idx
    return new_mat


def num_of_hidden_neuron(f, c):
    n = int(round((f + c) * 2 / 3))
    return n if f * 2 > n else f * 2

data_type = 'float32'
data_train = np.loadtxt('poker-hand-training-true.data', unpack=True, dtype=data_type, delimiter=',')
x_train = np.transpose(normalize(data_train[0:-1]))
y_train = np.reshape(data_train[-1], [-1, 1])

num_of_class = 10
num_of_feature = len(x_train[0])
num_of_neuron = num_of_hidden_neuron(num_of_feature, num_of_class)
keep_prob = tf.placeholder(tf.float32)

X = tf.placeholder(tf.float32, [None, num_of_feature])
Y = tf.placeholder(tf.int32, [None, 1])

W1 = tf.Variable(tf.random_normal([num_of_feature, num_of_neuron]))
b1 = tf.Variable(tf.random_normal([num_of_neuron]))
layer1 = tf.nn.relu(tf.matmul(X, W1) + b1)
layer1 = tf.nn.dropout(layer1, keep_prob)

W2 = tf.Variable(tf.random_normal([num_of_neuron, num_of_neuron]))
b2 = tf.Variable(tf.random_normal([num_of_neuron]))
layer2 = tf.nn.relu(tf.matmul(layer1, W2) + b2)
layer2 = tf.nn.dropout(layer2, keep_prob)

W_final = tf.Variable(tf.random_normal([num_of_neuron, num_of_class]))
b_final = tf.Variable(tf.random_normal([num_of_class]))
hypothesis = tf.matmul(layer2, W_final) + b_final

Y_one_hot = tf.one_hot(Y, num_of_class)  # one hot
Y_one_hot = tf.reshape(Y_one_hot, [-1, num_of_class])
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y_one_hot))
train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(cost)

training_dropout_rate = 1.0
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(30001):
    sess.run(train, feed_dict={X: x_train, Y: y_train, keep_prob: training_dropout_rate})
    if step % 100 == 0:
        predicted_train = sess.run(tf.argmax(hypothesis, 1), feed_dict={X: x_train, keep_prob: training_dropout_rate})
        print("Step: ", step,
              "Cost: ", sess.run(cost, feed_dict={X: x_train, Y: y_train, keep_prob: training_dropout_rate}),
              "Accuracy: ", np.sum(predicted_train == np.squeeze(y_train).astype(int)) / len(y_train))

### TEST ###
data_test = np.loadtxt('poker-hand-testing.data', unpack=True, dtype=data_type, delimiter=',')
x_test = np.transpose(normalize(data_test[0:-1]))
y_test = np.reshape(data_test[-1], [-1, 1])
predicted = sess.run(tf.argmax(hypothesis, 1), feed_dict={X: x_test, keep_prob: 1.0})
print("Hypothesis: ", predicted,
      "Accuracy: ", np.sum(predicted == np.squeeze(y_test).astype(int)) / len(y_test))