import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os


# path
prod_path = './data/P9_WOPR.txt'
day_path = './data/days.dat'
base_path = './tf_models'
if not os.path.isdir(base_path):
    os.mkdir(base_path)
save_file = base_path + '/model.ckpt'

# parameters
input_data_dim = 1
output_data_dim = 1
seq_length = 20
hidden_dim = 20
num_stacked_layers = 3
learning_rate = 0.005
num_epochs = 150
check_step = 1

# macro
TEST_IDX = 40
MAX_INPUT_DATE = 149
NUM_DAYS = -1       # be specified later
NUM_MODELS = -1     # be specified later

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


def rnn_cell():
    return tf.contrib.rnn.BasicRNNCell(num_units=hidden_dim, activation=tf.nn.relu)


def lstm_cell(relu=False):
    if relu:
        return tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True, activation=tf.nn.relu)
    return tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)


def gru_cell(relu=False):
    if relu:
        return tf.contrib.rnn.GRUCell(num_units=hidden_dim, activation=tf.nn.relu)
    return tf.contrib.rnn.GRUCell(num_units=hidden_dim, activation=tf.nn.tanh)


# load data
print("Load Data...")
load_start = time.time()
prod = np.loadtxt(prod_path)    # shape = (498, 104)
day = np.reshape(np.loadtxt(day_path), (-1, 1))      # shape = (498, 1)
# remove redundant day
prod = prod[85:, ].T                    # shape = (104, 413)
day = day[85:, ]                        # shape = (413, 1)
MAX_INPUT_DATE -= 85
NUM_DAYS = np.shape(day)[0]
NUM_MODELS = np.shape(prod)[0]
# min max scaling for training
prod_scaled = min_max_scaling(prod).T   # shape = (413, 104)

# make train set & test set
dataX = []
dataY = []
for model_n in range(0, NUM_MODELS):
    x_n = []
    y_n = []
    for day_n in range(0, NUM_DAYS - seq_length):
        _x = prod_scaled[day_n: day_n + seq_length, model_n]
        _y = prod_scaled[day_n + seq_length, model_n]
        x_n.append(_x)
        y_n.append(_y)
    dataX.append(x_n)
    dataY.append(y_n)

X_shape = np.shape(dataX)  # shape = (104, 393, 20)
Y_shape = np.shape(dataY)  # shape = (104, 393)

dataX = np.reshape(dataX, (X_shape[0], X_shape[1], X_shape[2], 1))  # shape = (104, 393, 20, 1)
dataY = np.reshape(dataY, (Y_shape[0], Y_shape[1], 1))              # shape = (104, 393, 1)

testX = dataX[TEST_IDX]     # shape = (393, 20, 1)
testY = dataY[TEST_IDX]     # shape = (393, 1)
trainX = np.delete(dataX, TEST_IDX, axis=0)
trainY = np.delete(dataY, TEST_IDX, axis=0)

print(np.shape(trainX), np.shape(trainY))
print(np.shape(testX), np.shape(testY))

print("Load Data Finished for {:.3f}s\n".format(time.time() - load_start))
time.sleep(1)

print("Make Model...")
make_model_start = time.time()
X = tf.placeholder(tf.float32, [None, seq_length, input_data_dim])
Y = tf.placeholder(tf.float32, [None, 1])

multi_cells = tf.contrib.rnn.MultiRNNCell([gru_cell(True) for _ in range(num_stacked_layers)], state_is_tuple=True)
outputs, _ = tf.nn.dynamic_rnn(multi_cells, X, dtype=tf.float32)
Y_prediction = tf.contrib.layers.fully_connected(outputs[:, -1], output_data_dim, activation_fn=None)

loss = tf.reduce_sum(tf.square(Y_prediction - Y))
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

# for calculate Root Mean Square Error
targets = tf.placeholder(tf.float32, [None, 1])
predictions = tf.placeholder(tf.float32, [None, 1])
rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))

print("Make Model Finished for {:.3f}s\n".format(time.time() - make_model_start))
time.sleep(1)

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("\nRestore model in {}".format(save_file))
    saver.restore(sess, save_file)

    # Train Session
    print("Train Start")
    train_start_time = time.time()
    for epoch in range(1, num_epochs + 1):
        total_loss = 0.0
        for n in range(0, NUM_MODELS-1):  # 1 is for test
            _, step_loss = sess.run([train, loss], feed_dict={X: trainX[n], Y: trainY[n]})
            total_loss += (step_loss / (NUM_MODELS - 1))
        if epoch % check_step == 0:
            print("[Epoch: {}] Loss: {:.7f} at {:.3f}s".format(epoch, total_loss, time.time() - train_start_time))
    print("Train Finish for {:.3f}s".format(time.time() - train_start_time))
    saver.save(sess, save_file)

    # Test Session
    print("Test model_{} Start".format(TEST_IDX))
    test_start_time = time.time()
    # test_predict = sess.run(Y_prediction, feed_dict={X: testX})
    test_predict = np.zeros((np.shape(testY)[0], np.shape(testY)[1]))   # shape = (393, 1)
    test_predict[0:MAX_INPUT_DATE - seq_length + 1] \
        = sess.run(Y_prediction, feed_dict={X: testX[0: MAX_INPUT_DATE-seq_length+1, ]})
    for days in range(MAX_INPUT_DATE - seq_length + 1, NUM_DAYS - seq_length):
        feedX = np.array([test_predict[days - seq_length: days]])
        test_predict[days] = sess.run(Y_prediction, feed_dict={X: feedX})
    rmse_error = sess.run(rmse, feed_dict={targets: testY, predictions: test_predict})
    print("Error: {:.7f}".format(rmse_error))
    print("Train Finish for {:.3f}s".format(time.time() - test_start_time))

    # Reversing data to original scale & Plot
    test_predict_reverse = np.reshape(reverse_min_max_scaling(prod[TEST_IDX], test_predict), (-1))  # shape = (393, )
    # predict data plot(Red)
    plt.figure(1)
    plt.plot(day[seq_length:], test_predict_reverse, 'r')
    # real data plot(Blue)
    plt.figure(1)
    plt.plot(day[seq_length:], prod[TEST_IDX, seq_length:], 'b')

    plt.show()
