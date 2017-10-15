import tensorflow as tf
import pandas as pandas


def load_data(file_name):
    data = pandas.read_csv(file_name, sep=',')

    features = data.iloc[:, 1:]
    labels = data.iloc[:, :1]

    return features, labels


X, Y = load_data('data/fashion-mnist_train.csv')
print(X.shape)
X_test, Y_test = load_data('data/fashion-mnist_test.csv')
print(X_test.shape)

nb_classes = 10
learning_rate = 0.001

# input place holders
X_in = tf.placeholder(tf.float32, [None, 784])
training = tf.placeholder(tf.bool)

# img 28x28x1 (black/white), Input Layer
X_img = tf.reshape(X_in, [-1, 28, 28, 1])
Y_in = tf.placeholder(tf.int32, [None, 1])
Y_one_hot = tf.reshape(tf.one_hot(Y_in, nb_classes), [-1, nb_classes])

# Convolutional Layer #1
conv1 = tf.layers.conv2d(inputs=X_img,
                         filters=32,
                         kernel_size=[5, 5],
                         padding="SAME",
                         activation=tf.nn.relu)

# Pooling Layer #1
pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                pool_size=[2, 2],
                                padding="SAME",
                                strides=2)
dropout1 = tf.layers.dropout(inputs=pool1, rate=0.5, training=training)

# Convolutional Layer #2 and Pooling Layer #2
conv2 = tf.layers.conv2d(inputs=dropout1,
                         filters=64,
                         kernel_size=[5, 5],
                         padding="SAME",
                         activation=tf.nn.relu)
pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                pool_size=[2, 2],
                                padding="SAME",
                                strides=2)
dropout2 = tf.layers.dropout(inputs=pool2,
                             rate=0.5, training=training)

# Convolutional Layer #3 and Pooling Layer #3
# conv3 = tf.layers.conv2d(inputs=dropout2, filters=128, kernel_size=[3, 3],
#                          padding="same", activation=tf.nn.relu)
# pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2],
#                                 padding="same", strides=2)
# dropout3 = tf.layers.dropout(inputs=pool3,
#                              rate=0.7, training=training)

# Dense Layer with Relu
flat = tf.reshape(dropout2, [-1, 64 * 7 * 7])
dense4 = tf.layers.dense(inputs=flat, units=1024, activation=tf.nn.relu)
dropout4 = tf.layers.dropout(inputs=dense4, rate=0.5, training=training)

# Logits (no activation) Layer: L5 Final FC 625 inputs -> 10 outputs
logits = tf.layers.dense(inputs=dropout4, units=nb_classes)

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# hyper parameters
learning_rate = 0.002
training_epochs = 30
batch_size = 100


# initialize
print('Learning Started!')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # train my model
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(Y.shape[0] / batch_size)

        for i in range(total_batch):
            lower = i * batch_size
            upper = (i + 1) * batch_size
            batch_xs = X.iloc[lower: upper, :]
            batch_ys = Y.iloc[lower: upper, :]
            c, _ = sess.run([cost, optimizer], feed_dict={X_in: batch_xs, Y_in: batch_ys, training: True})
            avg_cost += c / total_batch

        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

    print('Learning Finished!')

    # Test model and check accuracy
    prediction = sess.run(tf.argmax(logits, 1), feed_dict={X_in: X_test, training: False})
    print('prediction:', prediction)
    acc_test = sess.run(accuracy, feed_dict={X_in: X_test, Y_in: Y_test, training: False})
    print('accuracy:', acc_test)

    # for index in range(len(prediction)):
    #     print(index + 1, ',', prediction[index], sep='')
