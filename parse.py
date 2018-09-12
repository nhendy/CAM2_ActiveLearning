import re
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.axes_divider import make_axes_area_auto_adjustable
# import matplotlib.ticker as ticker
with open('logs', 'r') as logsFile:
    data = logsFile.read()


train_acc_pattern = r'train Loss: [-+]?[0-9]*\.?[0-9]+ Acc: ([-+]?[0-9]*\.?[0-9]+)'
val_acc_pattern  = r'val Loss: [-+]?[0-9]*\.?[0-9]+ Acc: ([-+]?[0-9]*\.?[0-9]+)'

train_acc_match = re.findall(train_acc_pattern, data)
val_acc_match = re.findall(val_acc_pattern, data)

print(type(train_acc_match))


epoch_axis = np.arange(0, 25, 1)
fig, ax = plt.subplots()


plt.title('Accuracy vs Epoch')
ax.plot(epoch_axis, train_acc_match, '.--', label='Training accuracy')
ax.plot(epoch_axis,val_acc_match, '+--', label='Validation accuracy')

ax.set_xlabel('epochs')
ax.set_ylabel('accuracy')

ymin, ymax = ax.get_ylim()
ax.set_yticks(np.round(np.linspace(ymin, ymax, 10), 2))

count = 0
for i,j in zip(epoch_axis,train_acc_match):
    if(count % 2 == 0):
        ax.annotate(str(j), xy=(i, j), xytext=(-10, 10), textcoords='offset points')
    count += 1

count = 0
for i,j in zip(epoch_axis,val_acc_match):
    if(count % 2 == 0):
        ax.annotate(str(j), xy=(i, j), xytext=(-10, 10), textcoords='offset points')
    count += 1


plt.legend()

plt.savefig('result3.png')





