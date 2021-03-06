import numpy as np
import pandas as pd
import pandas.plotting as pdplt
import tensorflow as tf
import matplotlib.pyplot as plt


# set up
def load_data(file_name):
    data = pd.read_csv(file_name, header=None, sep=',')
    print(data.shape)

    features, labels = split_features_and_labels(data)

    # labels.groupby(['G3']).size().plot('bar')
    # plt.show()

    print(features.shape, labels.shape)
    return features, labels


def split_features_and_labels(data):
    last_column_index = data.shape[1]
    features = data.iloc[:, 1:]
    labels = data.iloc[:, :1]
    return features, labels


whole_features, whole_labels = load_data('./agaricus-lepiota.data')

for col in whole_features.select_dtypes(['object']).columns:
    whole_features[col] = whole_features[col].astype('category').cat.codes
for col in whole_labels.select_dtypes(['object']).columns:
    whole_labels[col] = whole_labels[col].astype('category').cat.codes

for col in whole_features.select_dtypes(['int']).columns:
    whole_features[col] = (whole_features[col] - np.mean(whole_features[col], axis=0)) / np.std(whole_features[col], axis=0)


rnd_indices = np.random.rand(len(whole_features)) < 0.80

training_features = whole_features[rnd_indices]
training_labels = whole_labels[rnd_indices]
test_features = whole_features[~rnd_indices]
test_labels = whole_labels[~rnd_indices]

print('# of train set / # of test set : ', training_features.shape[0], '/', test_features.shape[0])

# parameters
learning_rate = 0.001
training_epochs = 5000
batch_size = 100

number_of_features = training_features.shape[1]

nb_classes = 10

# input place holders
X = tf.placeholder(tf.float32, [None, number_of_features])
Y = tf.placeholder(tf.int32, [None, 1])
Y_one_hot = tf.reshape(tf.one_hot(Y, nb_classes), [-1, nb_classes])

keep_prob = tf.placeholder(tf.float32)

# weights & bias for nn layers
number_of_w1_output = 128
W1 = tf.get_variable("W1", shape=[number_of_features, number_of_w1_output],
                     initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([number_of_w1_output]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

number_of_w2_output = 64
W2 = tf.get_variable("W2", shape=[number_of_w1_output, number_of_w2_output],
                     initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([number_of_w2_output]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)


number_of_w3_output = 32
W3 = tf.get_variable("W3", shape=[number_of_w2_output, number_of_w3_output],
                     initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([number_of_w3_output]))
L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

W4 = tf.get_variable("W4", shape=[number_of_w3_output, nb_classes],
                     initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([nb_classes]))
hypothesis = tf.matmul(L3, W4) + b4

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y_one_hot)) \
       + tf.reduce_mean(tf.square(W1)) + tf.reduce_mean(tf.square(W2)) + \
       tf.reduce_mean(tf.square(W3)) + tf.reduce_mean(tf.square(W4))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(training_epochs + 1):
        sess.run(optimizer, feed_dict={X: training_features, Y: training_labels, keep_prob: 0.5})
        loss, acc = sess.run([cost, accuracy], feed_dict={X: training_features, Y: training_labels, keep_prob: 0.5})
        if step % 1000 == 0:
            print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(step, loss, acc))

    loss_test, acc_test = sess.run([cost, accuracy], feed_dict={X: test_features, Y: test_labels, keep_prob: 1.0})
    print("Loss: {:.3f}\tAcc: {:.2%}".format(loss_test, acc_test))
