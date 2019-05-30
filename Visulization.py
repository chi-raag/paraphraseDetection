import numpy as np
import matplotlib.pyplot as plt
from baseline import create_training_data
import sys

data = create_training_data("data/msr_paraphrase_test.txt")

tags = []

for i in range(1, len(data) + 1):
    tags.append(data[i]['quality'])
    tags.append(data[i]['quality'])

data = create_training_data("data/msr_paraphrase_train.txt")


for i in range(1, len(data) + 1):
    tags.append(data[i]['quality'])
    tags.append(data[i]['quality'])


x = np.arange(2)
y = [tags.count(1), tags.count(0)]

plt.figure()
plt.bar(x, y)
plt.xticks(x, ('Paraphrase', 'Non-Paraphrase'))
plt.savefig("data_bar_chart.png")
plt.show()

print(tags.count(1) / (tags.count(1)+tags.count(0)))

sys.exit()

