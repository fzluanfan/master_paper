# -*- coding: UTF-8 -*-
from numpy import genfromtxt
import csv
import os
import numpy as np

# YA
# 10.1.10.202GX430L 211
# 10.1.10.203Q3020  55

files = os.listdir(r"./10.1.10.203Q3020/")
for file in files:
    load_data = genfromtxt('./10.1.10.203Q3020/'+file, delimiter=',', skip_header=1)
    program = genfromtxt('./10.1.10.203Q3020/'+file, delimiter=',', dtype=str, skip_header=1)
    # load_data = genfromtxt('20180201.csv', delimiter=',', skip_header=1)
    # program = genfromtxt('20180201.csv', delimiter=',', dtype=str, skip_header=1)
    print(load_data.shape)

    i = 0
    with open("./realwork/10.1.10.203Q3020/"+file[:-4]+"_realwork.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for numi in range(len(load_data[:, 4:5])):
            if load_data[numi:numi+1, 6:7] == 4 and load_data[numi:numi+1, 7:8] == 2 \
                    and (load_data[numi:numi+1, 9:10] == 1 or load_data[numi:numi+1, 9:10] == 3):
                    # and program[numi:numi+1, 14:15] == '30T-3          ':
                print(program[numi:numi+1, 14:15])
                GAP = load_data[numi:numi + 1, 10:13].reshape(3).tolist()
                MPX = load_data[numi:numi + 1, 20:23].reshape(3).tolist()
                my_data = GAP + MPX
                i += 1
                print(my_data, numi, i)
                writer.writerow(my_data)
