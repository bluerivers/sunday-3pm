import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# %matplotlib inline


# set up

data = np.loadtxt('./winequality-red.csv', delimiter=';', dtype=np.float32)
# data = np.r_[data, np.loadtxt('./winequality-white.csv', delimiter=';', dtype=np.float32)]

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

# features = np.c_[np.ones(n_training_samples), features]

rnd_indices = np.random.rand(len(features)) < 0.80

train_x = features[rnd_indices]
train_y = labels[rnd_indices]
test_x = features[~rnd_indices]
test_y = labels[~rnd_indices]


X = tf.placeholder(tf.float32, [None, 11])  # x_data와 같은 크기의 열 가짐. 행 크기는 모름.
Y = tf.placeholder(tf.float32, [None, 1])  # tf.float32라고 써도 됨

W = tf.Variable(tf.random_normal([11, 11]))       # 3x3 행렬. 전체 0.

# softmax 알고리듬 적용. X*W = (8x3) * (3x3) = (8x3)
nb_classes = 11

b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)

# 0...10

labels_encoding = tf.one_hot(train_y, nb_classes)
labels_encoding = tf.reshape(labels_encoding, [-1, nb_classes])



# cross-entropy cost 함수
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_encoding))

learning_rate = 0.01
train = tf.train.AdamOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

print(train_x.shape, train_y.shape)

prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(labels_encoding, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


with tf.Session() as sess:
    sess.run(init)

    for step in range(10001):
        sess.run(train, feed_dict={X: train_x, Y: train_y})
        if step % 100 == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict={X: train_x, Y: train_y})
            print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(step, loss, acc))

    print('-'  * 25)

    Test_X = tf.placeholder(tf.float32, [None, 11])
    Test_Y = tf.placeholder(tf.float32, [None, 1])

    Y_Hat = tf.nn.softmax(tf.matmul(Test_X, W) + b)

    test_labels_encoding = tf.one_hot(test_y, nb_classes)
    test_labels_encoding = tf.reshape(test_labels_encoding, [-1, nb_classes])

    pred = tf.argmax(hypothesis, 1)
    correct_prediction = tf.equal(prediction, tf.argmax(test_labels_encoding, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print(sess.run(accuracy, feed_dict={X: test_x, Y: test_y}))



