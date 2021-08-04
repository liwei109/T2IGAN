import pandas as pd

path = "../data/birds/train/class_info.pickle"

data = pd.read_csv(path, delim_whitespace=True, header=None)
print(len(data[0]))
# print(data[0])

classes = []
count = 0

for i in range(len(data[0])):
    if data[0][i] not in classes:
        classes.append(data[0][i])
        count += 1

print(classes)
print(count)
