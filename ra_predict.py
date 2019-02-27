from __future__ import print_function
import tensorflow as tf
import csv
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
import time
from sklearn.preprocessing import MinMaxScaler


# hyper parameters
n_steps = 1
n_inputs = 37
n_outputs = 1
n_hidden_units_1 = 64
n_hidden_units_2 = 128
n_hidden_units_3 = 256
n_hidden_units_4 = 128
n_hidden_units_5 = 64
n_hidden_units_6 = 32

lr = 0.0005
epochs = 1500
batch_size = 8
batch_start = 0


def add_layer(inputs, in_size, out_size, activation_function):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_uniform([in_size, out_size], minval=0.085, maxval=0.095))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


x_data = genfromtxt('me_NCU/realwork/feature.csv', delimiter=',')
y_data = genfromtxt('me_NCU/realwork/label.csv', delimiter=',')


scalerx = MinMaxScaler(feature_range=(0, 1))
tmp_data = scalerx.fit_transform(x_data[:, -5:].reshape(-1, 1))
x_data[:, -5:] = tmp_data.reshape(-1, 5)
scalery = MinMaxScaler(feature_range=(0, 1))
y_data = y_data.reshape(-1, 1)
y_data = scalery.fit_transform(y_data)
print(x_data.shape, y_data.shape)

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, n_inputs])
ys = tf.placeholder(tf.float32, [None, n_outputs])

# add hidden layer
l1 = add_layer(xs, n_inputs, n_hidden_units_1, activation_function=tf.nn.selu)
l2 = add_layer(l1, n_hidden_units_1, n_hidden_units_2, activation_function=tf.nn.selu)
l3 = add_layer(l2, n_hidden_units_2, n_hidden_units_3, activation_function=tf.nn.selu)
l4 = add_layer(l3, n_hidden_units_3, n_hidden_units_4, activation_function=tf.nn.selu)
l5 = add_layer(l4, n_hidden_units_4, n_hidden_units_5, activation_function=tf.nn.selu)
l6 = add_layer(l5, n_hidden_units_5, n_hidden_units_6, activation_function=tf.nn.selu)
# add output layer
pred = add_layer(l6, n_hidden_units_6, n_outputs, activation_function=None)

# the error between prediction and real data
loss = tf.reduce_mean(tf.abs(tf.subtract(pred, ys)))
train_step = tf.train.AdamOptimizer(lr).minimize(loss)

saver = tf.train.Saver()
now = time.strftime("%Y%m%d%H%M", time.localtime())


with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    i = 0
    while i < epochs+1:
        step = 0
        while step < len(y_data[:120, :1])/batch_size+1:
            batch_xs, batch_ys = x_data[batch_start:batch_start+batch_size*n_steps, :], \
                                 y_data[batch_start:batch_start+batch_size, :]
            sess.run([train_step], feed_dict={
                xs: batch_xs,
                ys: batch_ys,
            })
            if step == 0:
                print(i, sess.run(loss, feed_dict={
                    xs: batch_xs,
                    ys: batch_ys,
                }))
            step += 1
        if i % 10 == 0:
            prediction_epoch = sess.run(pred, feed_dict={xs: x_data[-batch_size * n_steps:, :]})
            prediction_epoch = scalery.inverse_transform(prediction_epoch)
            y_data_epoch = scalery.inverse_transform(y_data[-batch_size:, :])
            plt.plot(prediction_epoch, 'r', label='fitted line', lw=3)
            plt.plot(y_data_epoch, 'b', label='ori line', lw=3)
            plt.savefig('picture/ra_markov/' + str(i))
            plt.clf()
        i += 1
    save_path = saver.save(sess, "model/ra_markov/model_"+now+'_e'+str(epochs)+'_b'+str(batch_size)+".ckpt")
    prediction1 = sess.run(pred, feed_dict={xs: x_data[-batch_size * n_steps*3:-batch_size * n_steps*2, :]})
    prediction2 = sess.run(pred, feed_dict={xs: x_data[-batch_size * n_steps * 2:-batch_size * n_steps, :]})
    prediction3 = sess.run(pred, feed_dict={xs: x_data[-batch_size * n_steps:, :]})
    prediction = np.hstack([prediction1, prediction2, prediction3]).reshape(-1, 1)
    prediction = scalery.inverse_transform(prediction)
    y_data_pred = scalery.inverse_transform(y_data[-batch_size*3:, :])
    print(prediction.shape)
    print(y_data_pred.shape)
    with open('result/ra_Markov/result_'+now+'_e'+str(epochs)+'_b'+str(batch_size)+'.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        prediction = prediction.reshape(1, -1)
        y_data_pred = y_data_pred.reshape(1, -1)
        for i in range(len(prediction)):
            writer.writerow(prediction[i])
        for i in range(len(y_data_pred)):
            writer.writerow(y_data_pred[i])
        for i in range(len(y_data_pred)):
            writer.writerow(abs((prediction[i]-y_data_pred[i])/y_data_pred[i]))
    plt.plot(prediction.reshape(-1, 1), 'r', label='fitted line', lw=3)
    plt.plot(y_data_pred.reshape(-1, 1), 'b', label='ori line', lw=3)
    plt.savefig('picture/ra_markov/'+now)
    plt.show()
    plt.clf()
