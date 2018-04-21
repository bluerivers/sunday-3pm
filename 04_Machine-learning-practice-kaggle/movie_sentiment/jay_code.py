import numpy as np
import pandas as pd
import tensorflow as tf
import sys
from bs4 import BeautifulSoup
import re
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter


stopwords1 = ["i","me","my","myself","we","our","ours","ourselves","you","your","yours","yourself","yourselves","he","him","his","himself","she","her","hers","herself","it","its","itself","they","them","their","theirs","themselves","what","which","who","whom","this","that","these","those","am","is","are","was","were","be","been","being","have","has","had","having","do","does","did","doing","a","an","the","and","but","if","or","because","as","until","while","of","at","by","for","with","about","against","between","into","through","during","before","after","above","below","to","from","up","down","in","out","on","off","over","under","again","further","then","once","here","there","when","where","why","how","all","any","both","each","few","more","most","other","some","such","no","nor","not","only","own","same","so","than","too","very","s","t","can","will","just","don","should","now"]
stopwords2 = ["a","about","above","after","again","against","all","am","an","and","any","are","aren't","as","at","be","because","been","before","being","below","between","both","but","by","can't","cannot","could","couldn't","did","didn't","do","does","doesn't","doing","don't","down","during","each","few","for","from","further","had","hadn't","has","hasn't","have","haven't","having","he","he'd","he'll","he's","her","here","here's","hers","herself","him","himself","his","how","how's","i","i'd","i'll","i'm","i've","if","in","into","is","isn't","it","it's","its","itself","let's","me","more","most","mustn't","my","myself","no","nor","not","of","off","on","once","only","or","other","ought","our","ours	ourselves","out","over","own","same","shan't","she","she'd","she'll","she's","should","shouldn't","so","some","such","than","that","that's","the","their","theirs","them","themselves","then","there","there's","these","they","they'd","they'll","they're","they've","this","those","through","to","too","under","until","up","very","was","wasn't","we","we'd","we'll","we're","we've","were","weren't","what","what's","when","when's","where","where's","which","while","who","who's","whom","why","why's","with","won't","would","wouldn't","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves"]


def num_of_hidden_neuron(f, c):
    n = int(round((f + c) * 2 / 3))
    return n if f * 2 > n else f * 2


def review_to_words(data):
    clean_reviews = []
    for i in range(data["review"].size):
        review_text = BeautifulSoup(data["review"][i]).get_text()
        letters_only = re.sub("[^a-zA-Z]", " ", review_text)
        words = letters_only.lower().split()
        stops = set(stopwords1)
        meaningful_words = [w for w in words if not w in stops]
        meaningful_words = " ".join(meaningful_words)
        clean_reviews.append(meaningful_words)
    return clean_reviews


def vectorize(data):
    clean_reviews = review_to_words(data)
    vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None, max_features=2000)
    features = vectorizer.fit_transform(clean_reviews)
    return vectorizer.get_feature_names(), features.toarray()


data_train = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
data_test = pd.read_csv("testData.tsv", header=0, delimiter="\t", quoting=3)

feature_names, train_features = vectorize(data_train)
x_train = train_features
y_train = data_train["sentiment"]

x_test = np.zeros((data_test["review"].size, len(feature_names)))
test_reviews = review_to_words(data_test)
for i in range(len(test_reviews)):
    counts = Counter(test_reviews[i].split())
    for key, value in dict(counts).items():
        try:
            idx = feature_names.index(key)
            x_test[i][idx] = value
        except ValueError:
            idx = -1


keep_prob = tf.placeholder(tf.float32)

num_of_feature = len(x_train[0])
num_of_neuron = num_of_hidden_neuron(num_of_feature, 2)
print("num of neurons: ", num_of_neuron)

X = tf.placeholder(tf.float32, [None, num_of_feature])
Y = tf.placeholder(tf.float32, [None])

layer1 = tf.contrib.layers.fully_connected(X, num_of_neuron)
layer1 = tf.nn.dropout(layer1, keep_prob=keep_prob)

layer2 = tf.contrib.layers.fully_connected(layer1, num_of_neuron)
layer2 = tf.nn.dropout(layer2, keep_prob=keep_prob)

hypothesis = tf.contrib.layers.fully_connected(layer2, 1, activation_fn=None)
hypothesis = tf.transpose(hypothesis)
hypothesis = tf.div(1., 1.+tf.exp(-hypothesis))

cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y)*tf.log(1-hypothesis))
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

result = []
best_epoch = 0
best_cost = sys.maxsize
epoch_size = 100
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(epoch_size):
    sess.run(train, feed_dict={X: x_train, Y: y_train, keep_prob: 0.9})
    print("Step: ", step, "Cost: ", sess.run(cost, feed_dict={X: x_train, Y: y_train, keep_prob: 1}))
    cur_cost = sess.run(cost, feed_dict={X: x_train, Y: y_train, keep_prob: 1})
    if np.isnan(cur_cost):
        break
    if cur_cost < best_cost:
        best_cost = cur_cost
        best_epoch = step
        result = sess.run(hypothesis, feed_dict={X: x_test, keep_prob: 1})


print("Best - Step:", best_epoch, "Cost: ", best_cost)

# test
print("Test")
file = open("submission.csv", 'w')
file.write('"id","sentiment"\n')
for i in range(len(x_test)):
    line = str(data_test["id"][i]) + ',' + str(1 if result[0][i] >= 0.5 else 0) + '\n'
    file.write(line)
file.close()
