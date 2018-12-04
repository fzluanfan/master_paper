import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
from sklearn.preprocessing import MinMaxScaler

# hyper parameters
n_steps = 100
n_inputs = 3
n_outputs = 1
num_units = [128, 64]

lr = 0.00001
epochs = 100
batch_size = 8
batch_start = 0

my_data = genfromtxt('dataset2.csv', delimiter=',')
label = genfromtxt('label.csv', delimiter=',', skip_header=1)


def split_scale(data):
    datalen1 = data[:, :1]
    data1 = np.array(data[:1, :n_inputs])
    data2 = np.array(data[:1, :n_inputs])
    data3 = np.array(data[:1, :n_inputs])
    merge_data = np.array(data[:1, :n_inputs])

    for i in range(len(datalen1)):
        tmp = i % 100
        if tmp < 50:
            data1 = np.vstack((data1, data[i:i+1, :]))
        elif 50 <= tmp < 80:
            data2 = np.vstack((data[i:i+1, :], data2))
        elif 80 <= tmp < 100:
            data3 = np.vstack((data3, data[i:i+1, :]))

    datalen2 = data2[:, :1]
    data1 = data1[1:, :]
    data1 = scalerx.fit_transform(data1)
    data2 = data2[:len(datalen2) - 1, :]
    data2 = data2[::-1, :]
    data2 = scalerx.fit_transform(data2)
    data3 = data3[1:, :]
    data3 = scalerx.fit_transform(data3)
    datatmp1, datatmp2, datatmp3 = 0, 0, 0
    for i in range(len(datalen1)):
        tmp = i % 100
        if tmp < 50:
            merge_data = np.vstack((merge_data, data1[datatmp1:datatmp1 + 1, :]))
            datatmp1 += 1
        elif 50 <= tmp < 80:
            merge_data = np.vstack((merge_data, data2[datatmp2:datatmp2 + 1, :]))
            datatmp2 += 1
        elif 80 <= tmp < 100:
            merge_data = np.vstack((merge_data, data3[datatmp3:datatmp3 + 1, :]))
            datatmp3 += 1

    merge_data = merge_data[1:, :]
    return merge_data


def Rnn(X, Weights, Biases):
    # input
    X = tf.reshape(X, [-1, n_inputs])
    X_in = tf.matmul(X, Weights['in']) + Biases['in']
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    # cell
    cells = [tf.contrib.rnn.BasicLSTMCell(num_units=n) for n in num_units]
    stacked_rnn_cell = tf.contrib.rnn.MultiRNNCell(cells)
    init_state = cells.zero_state(batch_size, dtype=tf.float32)
    outputs, final_state = tf.nn.dynamic_rnn(cells, X_in, initial_state=init_state, time_major=False)

    # output
    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    results = tf.matmul(outputs[-1], Weights['out']) + Biases['out']

    return results


x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_outputs])

my_data = my_data[:, :n_inputs]
scalerx = MinMaxScaler(feature_range=(0, 1))
split_scale(my_data)
label = label[:, :-1]
scalery = MinMaxScaler(feature_range=(0, 1))
label = label.reshape(-1, 1)
label = scalery.fit_transform(label)
print(my_data.shape, label.shape)

weights = {
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_outputs]))
}
biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[n_outputs, ]))
}

pred = Rnn(x, weights, biases)
cost = tf.reduce_mean(tf.abs(tf.subtract(pred, y)))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    i = 0
    while i < epochs+1:
        step = 0
        while step < len(label[:100, :1])/batch_size+1:
            batch_xs, batch_ys = my_data[batch_start:batch_start+batch_size*n_steps, :], label[batch_start:batch_start+batch_size, :]
            batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
            sess.run([train_op], feed_dict={
                x: batch_xs,
                y: batch_ys,
            })
            if step == 0:
                print(i, step, sess.run(cost, feed_dict={
                    x: batch_xs,
                    y: batch_ys,
                }))
            step += 1
        i += 1
    prediction = sess.run(pred, feed_dict={x: my_data[-batch_size * n_steps:, :].reshape([batch_size, n_steps, n_inputs])})
    prediction = scalery.inverse_transform(prediction)
    label[-batch_size:, :] = scalery.inverse_transform(label[-batch_size:, :])
    print(prediction)
    print(label[-batch_size:, :])
    plt.plot(prediction, 'r', label='fitted line', lw=3)
    plt.plot(label[-batch_size:, :], 'b', label='ori line', lw=3)
    plt.show()
    plt.clf()
