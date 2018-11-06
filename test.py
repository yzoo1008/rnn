import numpy as np
import matplotlib.pyplot as plt

data_dir = './data/'
P9_WOPR = 'P9_WOPR.txt'
DAY = 'days.dat'

production_path = data_dir + P9_WOPR
day_path = data_dir + DAY

production = np.loadtxt(production_path)
day = np.loadtxt(day_path)

print(np.shape(production[:, 0]))
print(np.shape(day))

for idx in range(0, 103):
    plt.figure(1)
    plt.plot(day, production[:, idx])
plt.show()
