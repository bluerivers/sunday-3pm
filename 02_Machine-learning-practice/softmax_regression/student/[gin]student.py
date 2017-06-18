import pandas
import numpy
import tensorflow as tf
import matplotlib.pyplot as plt


def load_student_mat_data():
    student_mat = pandas.read_csv('./student-mat.csv', sep=";")
    print(student_mat.shape)

    for col in student_mat.select_dtypes(['object']).columns:
        student_mat[col] = student_mat[col].astype('category')

    student_mat.drop(student_mat[student_mat['G3'] == 0].index, inplace=True)

    features = student_mat.loc[:, :'G2']
    labels = student_mat.loc[:, 'G3':]

    # labels.groupby(['G3']).size().plot('bar')
    # plt.show()

    bins = [0, 6, 11, 16, 21]
    new_labels = []
    for lower in range(0, len(bins) - 1):
        new_labels.append("{0} - {1}".format(bins[lower], bins[lower + 1] - 1))

    print(new_labels)

    features['G1'] = pandas.cut(features['G1'].values, bins, right=False, labels=new_labels)
    features['G2'] = pandas.cut(features['G2'].values, bins, right=False, labels=new_labels)
    labels['G3'] = pandas.cut(labels['G3'].values, bins, right=False, labels=new_labels)

    print(features.shape, labels.shape)

    for col in features.select_dtypes(['int']).columns:
        features[col] = (features[col] - numpy.mean(features[col], axis=0)) / numpy.std(features[col], axis=0)

    cat_columns = features.select_dtypes(['category']).columns
    features[cat_columns] = features[cat_columns].apply(lambda x: x.cat.codes)

    cat_columns = labels.select_dtypes(['category']).columns
    labels[cat_columns] = labels[cat_columns].apply(lambda x: x.cat.codes)

    rnd_indices = numpy.random.rand(len(features)) < 0.80

    train_x = features[rnd_indices]
    train_y = labels[rnd_indices]
    test_x = features[~rnd_indices]
    test_y = labels[~rnd_indices]

    print('# of train set / # of test set : ', train_x.shape[0], '/', test_x.shape[0])

    return train_x, train_y, test_x, test_y


train_feature, train_label, test_feature, test_label = load_student_mat_data()

training_epochs = 5000
learning_rate = 0.01
cost_history = numpy.empty(shape=[1], dtype=float)

# grade 0 ~ 20
nb_classes = 7

number_of_features = train_feature.shape[1]

# Y = X W + B => [n, 32] X [32,  21] + [n, 21] = [n, 21]
X = tf.placeholder(tf.float32, [None, number_of_features])
Y = tf.placeholder(tf.int32, [None, 1])

Y_one_hot = tf.reshape(tf.one_hot(Y, nb_classes), [-1, nb_classes])

print("Y_one_hot : ", Y_one_hot)

W = tf.Variable(tf.random_normal([number_of_features, nb_classes]), name='weight')

B = tf.Variable(tf.random_normal([nb_classes]), name='bias')

logits = tf.matmul(X, W) + B

hypothesis = tf.nn.softmax(logits)

regularization = 0.001 * tf.reduce_sum(tf.square(W))
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y_one_hot)) + regularization
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(training_epochs + 1):
        sess.run(optimizer, feed_dict={X: train_feature, Y: train_label})
        loss, r, acc = sess.run([cost, regularization, accuracy], feed_dict={X: train_feature, Y: train_label})
        cost_history = numpy.append(cost_history, acc)
        if step % 1000 == 0:
            print("Step: {:5}\tLoss: {:.3f}\tReg: {:.3f}\tAcc: {:.2%}".format(step, loss, r, acc))

    loss_test, acc_test = sess.run([cost, accuracy], feed_dict={X: test_feature, Y: test_label})
    print("Loss: {:.3f}\tAcc: {:.2%}".format(loss_test, acc_test))



