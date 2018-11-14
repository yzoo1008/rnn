import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

tf.set_random_seed(777)

# path
prod_path = './data/P9_WOPR.txt'
day_path = './data/days.dat'

# train Parameters
input_data_dim = 1
output_data_dim = 1

seq_length = 15
hidden_dim = 15

num_stacked_layers = 2

learning_rate = 0.005
num_epoch = 100
check_step = 1

TEST_IDX = 6  # 0 ~ 103
TRAIN_SIZE = 104
DAY_SIZE = 413
DAY_INPUT = 100  # how many times used real value


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
    x_np = np.array(x)
    return (x_np * (org_x_np.max() - org_x_np.min() + 1e-7)) + org_x_np.min()


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


prod = np.loadtxt(prod_path)[85:, ].T  # transpose array to reshape (413, 104) -> (104, 413)
day = np.loadtxt(day_path)[85:, ]

print("Production shape: ", np.shape(prod), " / day shape: ", np.shape(day))

# min max scaling for training
prod_sc = min_max_scaling(prod).T

# reshape
train_X = prod_sc[0:DAY_INPUT, ].T
train_Y = prod_sc.T
X_shape = np.shape(train_X)
Y_shape = np.shape(train_Y)
dataX = np.reshape(train_X, (X_shape[0], X_shape[1], 1))   # shape(dataX) = (104, DAY_INPUT, 1)
dataY = np.reshape(train_Y, (Y_shape[0], Y_shape[1], 1))   # shape(dataY) = (104, 413, 1)

print(np.shape(dataX), "/", np.shape(dataY))


# Deep Learning Model
X = tf.placeholder(tf.float32, [None, seq_length, input_data_dim])
Y = tf.placeholder(tf.float32, [None, 1])

multi_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell(False) for _ in range(num_stacked_layers)], state_is_tuple=True)
outputs, _ = tf.nn.dynamic_rnn(multi_cells, X, dtype=tf.float32)
Y_prediction = tf.contrib.layers.fully_connected(outputs[:, -1], output_data_dim, activation_fn=None)

loss = tf.reduce_sum(tf.square(Y_prediction - Y))

optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

# for calculate RMSE(Root Mean Square Error)
targets = tf.placeholder(tf.float32, [None, 1])
predictions = tf.placeholder(tf.float32, [None, 1])

rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print("Train Start")
    train_start_time = time.time()
    tempX = np.zeros(np.shape(dataY))
    for epoch in range(1, num_epoch + 1):
        total_loss = 0.0
        for model_num in range(0, TRAIN_SIZE):
            if model_num != TEST_IDX:
                for day in range(0, DAY_SIZE - seq_length):
                    feedY = np.array([dataY[model_num, day + seq_length]])
                    if day <= DAY_INPUT - seq_length:
                        feedX = [dataX[model_num, day:day+seq_length]]
                        # print("Case 1: ", np.shape(feedX), " / ", np.shape(feedY), " at day ", day)
                        _, step_loss, tempX[model_num, day + seq_length, 0] = sess.run([train, loss, Y_prediction], feed_dict={X: feedX, Y: feedY})
                    elif (day < DAY_INPUT) & (day > DAY_INPUT - seq_length):
                        feedX = [np.concatenate((dataX[model_num, day:DAY_INPUT], (tempX[model_num, DAY_INPUT:day+seq_length])), axis=0)]
                        # print("Case 2: ", np.shape(feedX), " / ", np.shape(feedY), " at day ", day)
                        _, step_loss, tempX[model_num, day + seq_length, 0] = sess.run([train, loss, Y_prediction],
                                                                                       feed_dict={X: feedX, Y: feedY})
                    else:
                        feedX = [tempX[model_num, day:day + seq_length]]
                        # print("Case 3: ", np.shape(feedX), " / ", np.shape(feedY), " at day ", day)
                        _, step_loss, tempX[model_num, day + seq_length, 0] = sess.run([train, loss, Y_prediction],
                                                                                       feed_dict={X: feedX, Y: feedY})
                print("Model ", model_num, " finished! Loss: ", step_loss)
                total_loss += (step_loss / (TRAIN_SIZE - 1))
        if epoch % check_step == 0:
            print("[step: {}] loss: {}".format(epoch, total_loss))
    print("Train Finish, Collapse Time: {}s".format(time.time() - train_start_time))

'''
    print("Test Start")
    test_start_time = time.time()
    test_predict = sess.run(Y_prediction, feed_dict={X: testX})
    rmse_val = sess.run(rmse, feed_dict={targets: testY, predictions: test_predict})
    print("RMSE: {}".format(rmse_val))
    print("Test Finish, Collapse Time: {}s".format(time.time() - test_start_time))
    test_predict_reverse = np.reshape(reverse_min_max_scaling(prod, test_predict), (-1))
    # predict data plot(red)
    plt.figure(1)
    plt.plot(day[seq_length:], test_predict_reverse, 'r')
    # real data plot(blue)
    plt.figure(1)
    plt.plot(day[seq_length:], prod[seq_length:, TEST_IDX], 'b')
    plt.show()
'''
