import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

tf.set_random_seed(108)

# path
prod_path = './data/P9_WOPR.txt'
day_path = './data/days.dat'

# train Parameters
input_data_dim = 1
output_data_dim = 1

seq_length = 15
hidden_dim = 15

num_stacked_layers = 2
keep_prob = 1

learning_rate = 0.004
num_epoch = 100
check_step = 1

TEST_IDX = 40  # 0 ~ 103
TRAIN_SIZE = 103


def data_standardization(x):
    x_np = np.array(x)
    return (x_np - x_np.mean()) / x_np.std()


def min_max_scaling(x):
    x_np = np.array(x)
    min_np = np.reshape(x_np.min(axis=1), (np.shape(x_np.min(axis=1))[0], 1))
    max_np = np.reshape(x_np.max(axis=1), (np.shape(x_np.max(axis=1))[0], 1))
    return (x_np - min_np) / (max_np - min_np + 1e-7)



def reverse_min_max_scaling(org_x, x):
    org_x_np = np.array(org_x)
    min_np = np.min(org_x_np)
    max_np = np.max(org_x_np)
    x_np = np.array(x)
    return (x_np * (max_np - min_np + 1e-7)) + min_np


def lstm_cell(ReLu=False):
    if ReLu:
        return tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True, activation=tf.nn.relu)
    return tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)


def rnn_cell():
    return tf.contrib.rnn.BasicRNNCell(num_units=hidden_dim, activation=tf.nn.relu)


def gru_cell(ReLu=False):
    if ReLu:
        return tf.contrib.rnn.GRUCell(num_units=hidden_dim, activation=tf.nn.relu)
    return tf.contrib.rnn.GRUCell(num_units=hidden_dim, activation=tf.nn.tanh)


prod = np.loadtxt(prod_path)
day = np.loadtxt(day_path)
prod = prod[85:, ].T
day = day[85:, ]

DAYS = np.shape(day)[0]
INPUT_DAYS = 65
# min max scaling for training
prod_sc = min_max_scaling(prod).T

# for idx in range(0, 103):
#     plt.figure(1)
#     plt.plot(day, prod[:, idx])
# plt.show()

dataX = []
dataY = []

for num in range(0, len(prod_sc[0])):
    x_n = []
    y_n = []
    for idx in range(0, len(prod_sc) - seq_length):
        _x = prod_sc[idx: idx + seq_length, num]
        _y = prod_sc[idx + seq_length, num]
        '''
        if idx == 0:
            print(_x, "->", _y)
        '''
        x_n.append(_x)
        y_n.append(_y)
    dataX.append(x_n)
    dataY.append(y_n)

# print(len(prod[0]))
# print(len(prod))
#
X_shape = np.shape(dataX)
Y_shape = np.shape(dataY)

dataX = np.reshape(dataX, (X_shape[0], X_shape[1], X_shape[2], 1))
dataY = np.reshape(dataY, (Y_shape[0], Y_shape[1], 1))

testX = dataX[TEST_IDX]
testY = dataY[TEST_IDX]

trainX = np.delete(dataX, TEST_IDX, axis=0)
trainY = np.delete(dataY, TEST_IDX, axis=0)

print(np.shape(testX), "/", np.shape(testY))
print(np.shape(trainX), "/", np.shape(trainY))


X = tf.placeholder(tf.float32, [None, seq_length, input_data_dim])
Y = tf.placeholder(tf.float32, [None, 1])

multi_cells = tf.contrib.rnn.MultiRNNCell([gru_cell(True) for _ in range(num_stacked_layers)], state_is_tuple=True)
outputs, _ = tf.nn.dynamic_rnn(multi_cells, X, dtype=tf.float32)
Y_prediction = tf.contrib.layers.fully_connected(outputs[:, -1], output_data_dim, activation_fn=None)

loss = tf.reduce_sum(tf.square(Y_prediction - Y))

optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

# for calculate rmse
targets = tf.placeholder(tf.float32, [None, 1])
predictions = tf.placeholder(tf.float32, [None, 1])

rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Train Session
    print("Train Start")
    train_start_time = time.time()
    for epoch in range(1, num_epoch + 1):
        total_loss = 0.0
        for n in range(0, TRAIN_SIZE):
            if n != TEST_IDX:
                _, step_loss = sess.run([train, loss], feed_dict={X: trainX[n], Y: trainY[n]})
                total_loss += (step_loss / TRAIN_SIZE)
        if epoch % check_step == 0:
            print("[step: {}] loss: {}".format(epoch, total_loss))
    print("Train Finish, Collapse Time: {}s".format(time.time() - train_start_time))

    # Test Session
    print("Test Start")
    test_start_time = time.time()
    test_predict = np.zeros((np.shape(testY)[0] + seq_length, np.shape(testY)[1]))
    test_predict[seq_length:INPUT_DAYS+1] = sess.run(Y_prediction, feed_dict={X: testX[0: INPUT_DAYS-seq_length+1,]})
    for days in range(INPUT_DAYS - seq_length + 1, DAYS - seq_length):
        feedX = np.array([test_predict[days: days + seq_length]])
        test_predict[days + seq_length] = sess.run(Y_prediction, feed_dict={X: feedX})

    rmse_val = sess.run(rmse, feed_dict={targets: testY, predictions: test_predict[seq_length:]})
    #
    # print("RMSE: {}".format(rmse_val))
    print("Test Finish, Collapse Time: {}s".format(time.time() - test_start_time))

    test_predict_reverse = np.reshape(reverse_min_max_scaling(prod[TEST_IDX], test_predict[seq_length:]), (-1))
    # predict data plot(red)
    plt.figure(1)
    plt.plot(day[seq_length:], test_predict_reverse, 'r')
    # real data plot(blue)
    plt.figure(1)
    plt.plot(day[seq_length:], prod[TEST_IDX, seq_length:], 'b')

    plt.show()
