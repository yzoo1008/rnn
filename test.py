import numpy as np
# import matplotlib.pyplot as plt
#
# data_dir = './data/'
# P9_WOPR = 'P9_WOPR.txt'
# DAY = 'days.dat'
#
# production_path = data_dir + P9_WOPR
# day_path = data_dir + DAY
#
# production = np.loadtxt(production_path)
# day = np.loadtxt(day_path)
#
# print(np.shape(production[:, 0]))
# print(np.shape(day))
#
# for idx in range(0, 103):
#     plt.figure(1)
#     plt.plot(day, production[:, idx])
# plt.show()

a = [[1, 2, 3],
     [4, 6, 8],
     [7, 10, 13]]

np_a = np.array(a)

np_a_min = np.reshape(np_a.min(axis=1), (np.shape(np_a.min(axis=1))[0], 1))
np_a_max = np.reshape(np_a.max(axis=1), (np.shape(np_a.max(axis=1))[0], 1))

print(np_a_min)
print(np_a_max)
print(np_a - np_a_min)

print((np_a - np_a_min)/(np_a_max-np_a_min + 1e-7))
