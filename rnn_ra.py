import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
from sklearn.preprocessing import MinMaxScaler

# hyperparameters
n_steps = 100
n_inputs = 6
n_outputs = 1
n_hidden_units = 30

lr = 0.001
epochs = 20
batch_size = 10
batch_start = 0

my_data = genfromtxt('dataset2.csv', delimiter=',')
label = genfromtxt('label.csv', delimiter=',', skip_header=1)


def split_scale(data):
    datalen1 = data[:, :1]
    data1 = np.array(data[:1, :6])
    data2 = np.array(data[:1, :6])
    data3 = np.array(data[:1, :6])
    merge_data = np.array(data[:1, :6])

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
    cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units)
    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)

    # output
    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    results = tf.matmul(outputs[-1], Weights['out']) + Biases['out']

    return results


x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_outputs])

my_data = my_data[:, :6]
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
    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
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
    prediction = sess.run(pred, feed_dict={x: my_data[-10 * n_steps:, :].reshape([batch_size, n_steps, n_inputs])})
    prediction = scalery.inverse_transform(prediction)
    label[-10:, :] = scalery.inverse_transform(label[-10:, :])
    print(prediction)
    print(label[-10:, :])
    plt.plot(prediction, 'r', label='fitted line', lw=3)
    plt.plot(label[-10:, :], 'b', label='ori line', lw=3)
    plt.show()
    plt.clf()
