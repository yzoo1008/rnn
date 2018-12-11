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
seq_length = 40
hidden_dim = 30
num_stacked_layers = 2
learning_rate = 0.005
num_epochs = 30
check_step = 1

# macro
TEST_IDX = 39
MAX_INPUT_DATE = 200  # 149 => 100 days
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


def rnn_cell(keep_prob=1.0):
    cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_dim, activation=tf.nn.relu)
    return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)


def lstm_cell(relu=False, keep_prob=1.0):
    cell = tf.contrib.rnn.LSTMCell(num_units=hidden_dim, activation=tf.tanh)
    if relu:
        cell = tf.contrib.rnn.LSTMCell(num_units=hidden_dim, activation=tf.nn.relu)
    return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)


def gru_cell(relu=False, keep_prob=1.0):
    cell = tf.contrib.rnn.GRUCell(num_units=hidden_dim, activation=tf.nn.tanh)
    if relu:
        cell = tf.contrib.rnn.GRUCell(num_units=hidden_dim, activation=tf.nn.relu)
    return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)


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
trainX = np.delete(dataX, TEST_IDX, axis=0)  # shape = (103, 393, 20, 1)
trainY = np.delete(dataY, TEST_IDX, axis=0)  # shape = (103, 393, 1)

print(np.shape(trainX), np.shape(trainY))
print(np.shape(testX), np.shape(testY))

print("Load Data Finished for {:.3f}s\n".format(time.time() - load_start))
time.sleep(1)


print("Make Model...")
make_model_start = time.time()


class Model:
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build()

    def _build(self):
        with tf.variable_scope(self.name):
            self.X = tf.placeholder(tf.float32, [None, seq_length, input_data_dim])
            self.Y = tf.placeholder(tf.float32, [None, 1])
            self.keep_prob = tf.placeholder(tf.float32)

            stacked_rnn = [rnn_cell(self.keep_prob) for _ in range(num_stacked_layers)]
            multi_cells = tf.contrib.rnn.MultiRNNCell(stacked_rnn, state_is_tuple=True)
            outputs, _ = tf.nn.dynamic_rnn(multi_cells, self.X, dtype=tf.float32)
            self.Y_prediction = tf.contrib.layers.fully_connected(outputs[:, -1], output_data_dim, activation_fn=None)

        self.loss = tf.reduce_sum(tf.square(self.Y_prediction - self.Y))
        optimizer = tf.train.AdamOptimizer(learning_rate)
        self.train = optimizer.minimize(self.loss)

        # for calculate Root Mean Square Error
        self.targets = tf.placeholder(tf.float32, [None, 1])
        self.predictions = tf.placeholder(tf.float32, [None, 1])
        self.rmse = tf.sqrt(tf.reduce_mean(tf.square(self.targets - self.predictions)))

        self.saver = tf.train.Saver()

    def predict(self, x_test):
        return self.sess.run(self.Y_prediction, feed_dict={self.X: x_test, self.keep_prob: 1.0})

    def training(self, x_data, y_data):
        return self.sess.run([self.train, self.loss], feed_dict={self.X: x_data, self.Y: y_data, self.keep_prob: 0.7})

    def error(self, test, predict):
        return self.sess.run(self.rmse, feed_dict={self.targets: test, self.predictions: predict})

    def save(self, save_file_path):
        print("Saving Training model")
        self.saver.save(self.sess, save_file_path)


print("Make Model Finished for {:.3f}s\n".format(time.time() - make_model_start))
time.sleep(1)

sess = tf.Session()
models = []
NUM_M = 7
for num_m in range(NUM_M):
    models.append(Model(sess, "model" + str(num_m)))

sess.run(tf.global_variables_initializer())

# Train Session
print("\nTrain Start")
train_start_time = time.time()

for epoch in range(1, num_epochs + 1):
    avg_loss = 0.0
    for m_idx, m in enumerate(models):
        total_loss = 0.0
        for n in range(0, (NUM_MODELS - 1)):  # 1 is for test
            _, step_loss = m.training(trainX[n], trainY[n])
            total_loss += (step_loss / (NUM_MODELS - 1))
        print("[Epoch: {}] [Model: {}] Loss: {:.7f}".format(epoch, m_idx, total_loss))
        avg_loss += total_loss / NUM_M
    if epoch % check_step == 0:
        print("[Epoch: {}] Finished! Loss: {:.7f} at {:.3f}s".format(epoch, avg_loss, time.time() - train_start_time))
print("Train Finish for {:.3f}s".format(time.time() - train_start_time))

for m_idx, m in enumerate(models):
    m.save(base_path + '/model_' + str(m_idx) + '.ckpt')

# Test Session
print("Test model_{} Start".format(TEST_IDX))
test_start_time = time.time()

error = 0.
result_predict = np.zeros((np.shape(testY)[0], np.shape(testY)[1]))
for m_idx, m in enumerate(models):
    test_predict = np.zeros((np.shape(testY)[0], np.shape(testY)[1]))   # shape = (393, 1)
    test_predict[0:MAX_INPUT_DATE - seq_length + 1] = m.predict(testX[0: MAX_INPUT_DATE-seq_length+1, ])
    for days in range(MAX_INPUT_DATE - seq_length + 1, MAX_INPUT_DATE + 1):
        sd = MAX_INPUT_DATE - seq_length
        feedX = testX[days][0:seq_length-(days-sd)]
        # print(np.shape(feedX), np.shape(test_predict[days - (days - sd): days]))
        # print(feedX, test_predict[days - (days - sd): days])
        feedX = np.append(feedX, test_predict[days - (days - sd): days])
        # print(np.shape(feedX))
        # print(feedX)
        feedX = np.reshape(feedX, (-1, np.shape(testX)[1], np.shape(testX)[2]))
        test_predict[days] = m.predict(feedX)
        # print(np.shape(feedX))
    for days in range(MAX_INPUT_DATE + 1, NUM_DAYS - seq_length):
        feedX = np.array([test_predict[days - seq_length: days]])
        test_predict[days] = m.predict(feedX)
    error += m.error(testY, test_predict)/NUM_M
    result_predict += test_predict/NUM_M

print("Error: {:.7f}".format(error))
print("Train Finish for {:.3f}s".format(time.time() - test_start_time))

# Reversing data to original scale & Plot
result_predict_reverse = np.reshape(reverse_min_max_scaling(prod[TEST_IDX], result_predict), (-1))  # shape = (393, )

# predict data plot(Red)
plt.figure(1)
plt.plot(day[seq_length:], result_predict_reverse, 'r')

# real data plot(Blue)
plt.figure(1)
plt.plot(day[seq_length:], prod[TEST_IDX, seq_length:], 'b')

plt.show()
