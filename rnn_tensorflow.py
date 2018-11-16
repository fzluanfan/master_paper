import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# hyperparameters
n_steps = 100
n_inputs = 6
n_outputs = 1
n_hidden_units = 90

lr = 0.001
epochs = 100
batch_size = 2

x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_outputs])

weights = {
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_outputs]))
}
biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[n_outputs, ]))
}


def Rnn(X, Weights, Biases):
    # input
    X = tf.reshape(X, [-1, n_inputs])
    X_in = tf.matmul(X, Weights['in']) + Biases['in']
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    # cell
    cell = tf.nn.rnn.BasicLSTMCell(n_hidden_units)
    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)

    # output
    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    results = tf.matmul(outputs[-1], Weights['out']) + Biases['out']

    return results

pred = Rnn(x, weights, biases)
cost = tf.reduce_mean()