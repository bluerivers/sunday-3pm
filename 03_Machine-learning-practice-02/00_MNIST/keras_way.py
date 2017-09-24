import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, MaxPool2D, Conv2D, Flatten, Dropout
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.utils.np_utils import to_categorical

output_shape = 58

model = Sequential()
model.add(Conv2D(output_shape, activation='relu', kernel_size=(3, 3), input_shape=(28, 28, 1)))
model.add(MaxPool2D((2, 2)))
model.add(Dropout(0.3))
model.add(Conv2D(output_shape, activation='relu', kernel_size=(2, 2), input_shape=(13, 13, 1)))
model.add(MaxPool2D((2, 2)))
model.add(Dropout(0.3))
model.add(Conv2D(output_shape, activation='relu', kernel_size=(2, 2), input_shape=(6, 6, 1)))
model.add(MaxPool2D((2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(output_shape, activation='relu', kernel_size=(2, 2), input_shape=(2, 2, 1)))
model.add(MaxPool2D((1, 1)))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))

model.compile(loss=categorical_crossentropy,
              optimizer=Adam(lr=0.001),
              metrics=['accuracy'])

model.summary()

dataset = pd.read_csv("./fashion-mnist_train.csv")
test_dataset = pd.read_csv("./fashion-mnist_test.csv")

train = dataset.iloc[:, 1:].values.astype("float32")
label = dataset.iloc[:, :1].values.astype("int32")

test_x = test_dataset.iloc[:, 1:].values.astype("float32")
test_y = test_dataset.iloc[:, :1].values.astype("int32")

train = train.reshape(train.shape[0], 28, 28, 1)
test_x = test_x.reshape(test_x.shape[0], 28, 28, 1)

y_one_hot = to_categorical(label)
print(train.shape)
print(label.shape)

num_of_classes = y_one_hot.shape[1]

history = model.fit(train, y_one_hot,
                    batch_size=200,
                    epochs=200,
                    verbose=1)

predictions = model.predict_classes(test_x, verbose=0)

correct = 0

for i in range(len(predictions)):
    if predictions[i] == test_y[i]:
        correct = correct + 1


print(correct/len(predictions))

submissions = pd.DataFrame({"ImageId": list(range(1, len(predictions) + 1)),
                            "Label": predictions})
submissions.to_csv("DR.csv", index=False, header=True)
