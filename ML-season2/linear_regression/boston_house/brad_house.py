import numpy as np

def read_dataset(file):
	dataset = np.genfromtxt('/Users/midas/PythonProject/tensorflow/boston_house/housing.data', delimiter=',')
	dataset_length = len(dataset)
	return dataset, dataset_length

def sampling_data(dataset, dataset_length):
	rnd_indices = np.random.rand(dataset_length) < 0.80
	training_data = [dataset[i] for i in range(dataset_length) if rnd_indices[i]]
	validation_data = [dataset[i] for i in range(dataset_length) if not(rnd_indices[i])]

	training_data = np.transpose(training_data)
	validation_data = np.transpose(validation_data)
	return training_data, validation_data

def linear_regression(training_data, validation_data):
	import tensorflow as tf
	x_data = training_data[0] #CRIME
	y_data = training_data[13]

	with tf.Session() as sess:
		W = tf.Variable(tf.random_normal([1]), name='weight')
		b = tf.Variable(tf.random_normal([1]), name='bias')

		H = W * x_data + b

		cost = tf.reduce_mean(tf.square(H - y_data))
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
		train = optimizer.minimize(cost)

		sess.run(tf.global_variables_initializer())

		print("step", "cost", "weight", "bias")
		for step in range(5000):
			sess.run(train)
			print(step, sess.run(cost), sess.run(W), sess.run(b))
		

if __name__ == "__main__":
	with open('/Users/midas/PythonProject/tensorflow/boston_house/housing.data', 'rt') as file:
		print("read csv file..")
		dataset, dataset_length = read_dataset(file)
		print("read done. Total %d records loaded." % (dataset_length,))

	print("Start sampling..(0.80)")
	training_data, validation_data = sampling_data(dataset, dataset_length)
	print("Sampling is done.")

	linear_regression(training_data, validation_data)


