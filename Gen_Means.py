from statistics import mean
import matplotlib
from matplotlib import testing
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import os
print("Imports done.")

print("Generating just some means...")

testing_mode = False # Write to sep file
run_all = True # Let's you pick and choose

# Sort out variables
loc = "Means"
mean_files = os.listdir(loc)
mean_files.sort()
accs = []
losses = []
names = []

# Narrow down the list if necessary
if not run_all:
    removals = []
    if picks:
        print(mean_files)
        for m in mean_files:
            print("Accessing", m, end='')
            if m not in picks:
                print(" | removed", end='')
                removals.append(m)
            print("")
    for r in removals:
        mean_files.remove(r)

# Append to dict
for m in mean_files:
    names.append(str(m))
    means = np.load(loc+"/"+m)
    accs.append(means['arr_0'])
    losses.append(means['arr_1'])
    del means

# Data
roundto = 2

for i in range(len(names)):
    print("Set", i+1, ":", names[i], ":", round(mean(accs[i]), roundto))

print("Done!")