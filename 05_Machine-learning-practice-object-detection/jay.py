import os
from PIL import Image
import random
import math
import tensorflow as tf
import numpy as np


def resize_img(img, new_size, offset_x, offset_y):
    width, height = img.size
    left = math.floor((width - new_size)/2) + offset_x
    top = math.floor((height - new_size)/2) + offset_y
    right = math.floor((width + new_size)/2) + offset_x
    bottom = math.floor((height + new_size)/2) + offset_y
    img = img.crop((left, top, right, bottom))
    return img.resize((base_size, base_size), Image.ANTIALIAS)


base_size = 250

true_img_path = "true_img/"
false_img_path = "false_img/"
true_images = os.listdir(true_img_path)
false_images = os.listdir(false_img_path)

sample_size = 100
true_images = random.sample(true_images, sample_size)
false_images = random.sample(false_images, sample_size)

labels = np.array([[0, 1]] * (sample_size*2))
labels[:sample_size] = [1, 0]

train_img = []
for img in true_images:
    f = Image.open(true_img_path + img).convert('LA')
    train_img.append([t[0] for t in list(f.getdata())])
for img in false_images:
    f = Image.open(false_img_path + img).convert('LA')
    width, height = f.size
    f = resize_img(f, min(width, height), 0, 0)
    train_img.append([t[0] for t in list(f.getdata())])

keep_prob = tf.placeholder(tf.float32)
X = tf.placeholder(tf.float32, [None, base_size*base_size])
X_img = tf.reshape(X, [-1, base_size, base_size, 1])
Y = tf.placeholder(tf.float32, [None, 2])

# 250
W1 = tf.Variable(tf.random_normal([3, 3, 1, 4], stddev=0.01))
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

# 125
W2 = tf.Variable(tf.random_normal([3, 3, 4, 8], stddev=0.01))
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

# 63
W3 = tf.Variable(tf.random_normal([3, 3, 8, 16], stddev=0.01))
L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)
L3_flat = tf.reshape(L3, [-1, 16 * 32 * 32])

W4 = tf.get_variable("W4", shape=[16 * 32 * 32, 512], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([512]))
L4 = tf.nn.relu(tf.matmul(L3_flat, W4) + b4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

W5 = tf.get_variable("W5", shape=[512, 2], initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([2]))
logits = tf.matmul(L4, W5) + b5

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for epoch in range(64):
    feed_dict = {X: train_img, Y: labels, keep_prob: 0.7}
    c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(c))

# accuracy for training set
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
logit = sess.run(logits, feed_dict={X: train_img, Y: labels, keep_prob: 1})
print('Accuracy:', sess.run(accuracy, feed_dict={X: train_img, Y: labels, keep_prob: 1}))

# image detection test
initial_window_size = 34
sample_path = "sample/"
random_samples = os.listdir(sample_path)
random_sample = random.sample(random_samples, 1)[0]
f = Image.open(sample_path + random_sample).convert('LA')

max_face = f
diff = 0
width, height = f.size
max_window_size = min(width, height)
iter = 10
for i in range(iter):
    window_size = initial_window_size + (i * math.floor((max_window_size - initial_window_size) / iter))
    for j in range(iter):
        for k in range(iter):
            face = resize_img(f, window_size, k * math.floor((width - window_size)/ iter), j * math.floor((height - window_size) / iter))
            result = sess.run(logits, feed_dict={X: [[t[0] for t in list(face.getdata())]], keep_prob: 1})
            cur_diff = result[0][0] - result[0][1]
            if cur_diff > diff:
                print(result)
                diff = cur_diff
                max_face = face
f.show()
max_face.show()
