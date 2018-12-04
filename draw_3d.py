# -*- coding: UTF-8 -*-
from numpy import genfromtxt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import scale
import matplotlib
matplotlib.rcParams.update({'font.size': 15})
my_data = genfromtxt('./realwork/10.1.10.202GX430L/20180201_realwork.csv', delimiter=',')
my_data = my_data[2272:, 3:6]
my_data = my_data
print(my_data)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = my_data[:, 0]
y = my_data[:, 1]
z = my_data[:, 2]
ax.scatter(x, y, z, c='g', marker='o')
# plt.title('加工路線圖')
plt.show()
