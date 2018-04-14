import numpy as np
import pandas as pd
import tensorflow as tf
import sys


def num_of_hidden_neuron(f, c):
    n = int(round((f + c) * 2 / 3))
    return n if f * 2 > n else f * 2


def split_na(data):
    buf = []
    for value in na_list:
        buf.append(np.isnan(data[value]).astype(int))
    for i in range(len(buf)):
        data[na_list[i] + "_NA"] = buf[i]


def to_numeric(data, mapping):
    for key, value in mapping.items():
        idx_dict = {}
        for i in range(len(value)):
            idx_dict[value[i]] = i
        data[key] = data[key].map(idx_dict)


categorical_columns_to_numeric = {"Street": ["Grvl", "Pave"],
                                  "LotShape": ["Reg", "IR1", "IR2", "IR3"],
                                  "Utilities": ["AllPub", "NoSewr", "NoSeWa", "ELO"],
                                  "LandSlope": ["Gtl", "Mod", "Sev"],
                                  "ExterQual": ["Ex", "Gd", "TA", "Fa", "Po"],
                                  "ExterCond": ["Ex", "Gd", "TA", "Fa", "Po"],
                                  "BsmtQual": ["NA", "Po", "Fa", "TA", "Gd", "Ex"],
                                  "BsmtCond": ["NA", "Po", "Fa", "TA", "Gd", "Ex"],
                                  "BsmtExposure": ["NA", "No", "Mn", "Av", "Gd"],
                                  "BsmtFinType1": ["NA", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
                                  "BsmtFinType2": ["NA", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
                                  "HeatingQC": ["Ex", "Gd", "TA", "Fa", "Po"],
                                  "CentralAir": ["N", "Y"],
                                  "KitchenQual": ["Ex", "Gd", "TA", "Fa", "Po"],
                                  "Functional": ["Typ", "Min1", "Min2", "Mod", "Maj1", "Maj2", "Sev", "Sal"],
                                  "FireplaceQu": ["NA", "Po", "Fa", "TA", "Gd", "Ex"],
                                  "GarageFinish": ["NA", "Unf", "RFn", "Fin"],
                                  "GarageQual": ["NA", "Po", "Fa", "TA", "Gd", "Ex"],
                                  "GarageCond": ["NA", "Po", "Fa", "TA", "Gd", "Ex"],
                                  "PavedDrive": ["Y", "P", "N"],
                                  "PoolQC": ["NA", "Fa", "TA", "Gd", "Ex"]
                                  }
na_list = ["LotFrontage","MasVnrArea","GarageYrBlt"]
categorical_columns = ["MSZoning","Alley","LandContour","LotConfig",
                       "Neighborhood","Condition1","Condition2","BldgType","HouseStyle",
                       "RoofStyle","RoofMatl","Exterior1st","Exterior2nd","MasVnrType",
                       "Foundation","Heating","Electrical","GarageType",
                       "Fence","MiscFeature","SaleType","SaleCondition"]
data_train = pd.read_csv('./train.csv')
data_test = pd.read_csv('./test.csv')
train_offset = 1460
x_train = data_train.iloc[:, 1:-1]
x_test = data_test.iloc[:, 1:]
x_buf = pd.get_dummies(pd.concat([x_train, x_test]), columns=categorical_columns)
to_numeric(x_buf, categorical_columns_to_numeric)
split_na(x_buf)
x_buf = x_buf.fillna(0)
x_train = x_buf[:train_offset]
y_train = data_train.iloc[:, -1]
x_test = x_buf[train_offset:]

keep_prob = tf.placeholder(tf.float32)

num_of_feature = len(x_train.iloc[0])
num_of_neuron = num_of_hidden_neuron(num_of_feature, num_of_feature)
print("num of neurons: ", num_of_neuron)

X = tf.placeholder(tf.float32, [None, num_of_feature])
Y = tf.placeholder(tf.float32, [None])

W1 = tf.get_variable("W1", shape=[num_of_feature, num_of_neuron],
                     initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([num_of_neuron]))
layer1 = tf.nn.relu(tf.matmul(X, W1) + b1)
layer1 = tf.nn.dropout(layer1, keep_prob=keep_prob)

W2 = tf.get_variable("W2", shape=[num_of_neuron, num_of_neuron],
                     initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([num_of_neuron]))
layer2 = tf.nn.relu(tf.matmul(layer1, W2) + b2)
layer2 = tf.nn.dropout(layer2, keep_prob=keep_prob)

W_final = tf.get_variable("W_final", shape=[num_of_neuron, 1],
                          initializer=tf.contrib.layers.xavier_initializer())
b_final = tf.Variable(tf.random_normal([1]))
hypothesis = tf.transpose(tf.matmul(layer2, W_final) + b_final)

cost = tf.reduce_mean(tf.square(hypothesis - Y)) + 0.01 * (tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W_final))
train = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

result = []
best_epoch = 0
best_cost = sys.maxsize
epoch_size = 20000
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(epoch_size):
    sess.run(train, feed_dict={X: x_train, Y: y_train, keep_prob: 0.9})
    if step % 100 == 0 and step <= epoch_size - 1000:
        print("Step: ", step, "Cost: ", sess.run(tf.sqrt(tf.reduce_mean(tf.square(tf.log(hypothesis) - tf.log(Y)))), feed_dict={X: x_train, Y: y_train, keep_prob: 1}))
    if step > epoch_size - 1000:
        cur_cost = sess.run(cost, feed_dict={X: x_train, Y: y_train, keep_prob: 1})
        if cur_cost < best_cost:
            best_cost = cur_cost
            best_epoch = step
            print("Step: ", step, "Cost: ", sess.run(tf.sqrt(tf.reduce_mean(tf.square(tf.log(hypothesis) - tf.log(Y)))), feed_dict={X: x_train, Y: y_train, keep_prob: 1}))
            result = sess.run(hypothesis, feed_dict={X: x_test, keep_prob: 1})


print("Cost: ", sess.run(tf.sqrt(tf.reduce_mean(tf.square(tf.log(hypothesis) - tf.log(Y)))), feed_dict={X: x_train, Y: y_train, keep_prob: 1}))

# test
print("Test")
fh = open("result.txt", "a")
file = open("submission.csv", 'w')
file.write("Id,SalePrice\n")
for i in range(len(x_test)):
    line = str(train_offset + i + 1) + ',' + str(result[0][i]) + '\n'
    file.write(line)
file.close()
