import numpy as np
import pandas as pd
import pandas.plotting as pdplt
import tensorflow as tf
import matplotlib.pyplot as plt


# set up
def load_data():
    data = pd.read_csv('./winequality-red.csv', sep=';')

    print(data.shape)
    print(data.describe())

    # data.loc[:, 'free sulfur dioxide':'total sulfur dioxide'].plot.box()
    # data.plot.box()
    # pdplt.scatter_matrix(data, alpha=0.2, figsize=(6, 6), diagonal='kde')
    plt.show()

    features = data.loc[:, :'alcohol']
    labels = data.loc[:, 'quality':].copy()

    # normalization
    for col in features.select_dtypes(['int']).columns:
        features[col] = (features[col] - np.mean(features[col], axis=0)) / np.std(features[col], axis=0)

    # 3, 4, 5, 6, 7, 8
    # 3, 4 -> low / 5, 6 -> medium / 7, 8 -> high
    # labels.groupby(['quality']).size().plot('bar')
    # plt.show()
    bins = [3, 5, 7, 9]
    new_labels = []
    for lower in range(0, len(bins) - 1):
        new_labels.append("{0} - {1}".format(bins[lower], bins[lower + 1] - 1))

    print(new_labels)

    labels['quality'] = pd.cut(labels['quality'].values, bins, right=False, labels=new_labels)

    cat_columns = labels.select_dtypes(['category']).columns
    labels[cat_columns] = labels[cat_columns].apply(lambda x: x.cat.codes)

    return features, labels


features, labels = load_data()


rnd_indices = np.random.rand(len(features)) < 0.80

train_x = features[rnd_indices]
train_y = labels[rnd_indices]
test_x = features[~rnd_indices]
test_y = labels[~rnd_indices]

print('# of train set / # of test set : ', train_x.shape[0], '/', test_x.shape[0])

# parameters
learning_rate = 0.001
training_epochs = 10000
batch_size = 100

number_of_features = train_x.shape[1]

nb_classes = 3

# input place holders
X = tf.placeholder(tf.float32, [None, number_of_features])
Y = tf.placeholder(tf.int32, [None, 1])
Y_one_hot = tf.reshape(tf.one_hot(Y, nb_classes), [-1, nb_classes])

keep_prob = tf.placeholder(tf.float32)

# weights & bias for nn layers
# http://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
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
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y_one_hot))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(training_epochs + 1):
        sess.run(optimizer, feed_dict={X: train_x, Y: train_y, keep_prob: 0.5})
        loss, acc = sess.run([cost, accuracy], feed_dict={X: train_x, Y: train_y, keep_prob: 0.5})
        if step % 1000 == 0:
            print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(step, loss, acc))

    loss_test, acc_test = sess.run([cost, accuracy], feed_dict={X: test_x, Y: test_y, keep_prob: 1.0})
    print("Loss: {:.3f}\tAcc: {:.2%}".format(loss_test, acc_test))
