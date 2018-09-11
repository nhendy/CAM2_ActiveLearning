import re
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


with open('logs', 'r') as logsFile:
    data = logsFile.read()


train_acc_pattern = r'train Loss: [-+]?[0-9]*\.?[0-9]+ Acc: ([-+]?[0-9]*\.?[0-9]+)'
val_acc_pattern  = r'val Loss: [-+]?[0-9]*\.?[0-9]+ Acc: ([-+]?[0-9]*\.?[0-9]+)'

train_acc_match = re.findall(train_acc_pattern, data)
val_acc_match = re.findall(val_acc_pattern, data)

print(type(train_acc_match))


epoch_axis = np.arange(0, 25, 1)
fig, ax = plt.subplots()

ax.plot(epoch_axis, train_acc_match, '.-',epoch_axis, val_acc_match, '+--')

ax.set(xlabel='epoch', ylabel='accuracy')
ax.grid()

fig.savefig("result.png")





