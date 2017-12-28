import matplotlib.pyplot as plt
import numpy as np
import re
from matplotlib.ticker import MaxNLocator

with open('./data/src-train_dev_0.8_common.txt') as f:
    src_train = np.array(f.readlines())
with open('./data/tgt-train_dev_0.8_common.txt') as f:
    tgt_train = np.array(f.readlines())

print(len(tgt_train))
print('Number of unique equations:',len(np.unique(tgt_train)))

operations = [len(re.findall(r'\+|\-|\*|\/|\^', d)) for d in tgt_train]
lengths = [len(''.join(d.split())) for d in tgt_train]
total_op_occur = np.array([
    ['+', np.sum([len(re.findall(r'\+', d)) for d in tgt_train])],
    ['-', np.sum([len(re.findall(r'\-', d)) for d in tgt_train])],
    ['*', np.sum([len(re.findall(r'\*', d)) for d in tgt_train])],
    ['/', np.sum([len(re.findall(r'\/', d)) for d in tgt_train])],
    ['^', np.sum([len(re.findall(r'\^', d)) for d in tgt_train])]
])

cont = np.array([
    ['+', np.sum([len(re.findall(r'\+', d)) > 0 for d in tgt_train])],
    ['-', np.sum([len(re.findall(r'\-', d)) > 0 for d in tgt_train])],
    ['*', np.sum([len(re.findall(r'\*', d)) > 0 for d in tgt_train])],
    ['/', np.sum([len(re.findall(r'\/', d)) > 0 for d in tgt_train])],
    ['^', np.sum([len(re.findall(r'\^', d)) > 0 for d in tgt_train])]
])

# equation frequency
equation, count= np.unique(tgt_train, return_counts=True)
indices = np.flip(np.asarray((equation, count)).T[:,1].astype(int).argsort(), axis=0)
result = np.asarray([[equation[i], count[i]] for i in indices])
np.set_printoptions(threshold=np.nan)
#print(result[:,1])
#print(equation[indices[0]])

# the histogram of equation frequency
fig = plt.figure().gca()
fig.grid()
plt.plot(result[:,1])
plt.ylabel('# of occurences')
plt.xlabel('equation')
plt.show()


# the histogram of operations
fig = plt.figure().gca()
fig.grid()
n, bins, patches = plt.hist(operations,bins=range(min(operations), max(operations) + 1, 1))
fig.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlabel('# of operations')
plt.ylabel('# examples')
plt.show()

# the histogram of lengths
fig = plt.figure().gca()
fig.grid()
n, bins, patches = plt.hist(lengths,bins=range(min(lengths), max(lengths) + 1, 1))
fig.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlabel('length')
plt.ylabel('# examples')
plt.show()

# histogram of operation occurences
fig = plt.figure().gca()
fig.grid()

plt.bar(np.arange(len(total_op_occur[:,0])), total_op_occur[:,1].astype(int))
plt.xticks(np.arange(len(total_op_occur[:,0])), total_op_occur[:,0])

plt.xlabel('operation')
plt.ylabel('total occurences')
plt.show()

# histogram of cont
fig = plt.figure().gca()
fig.grid()

plt.bar(np.arange(len(cont[:,0])), cont[:,1].astype(int))
plt.xticks(np.arange(len(cont[:,0])), cont[:,0])

plt.xlabel('operation')
plt.ylabel('# examples containg operation')
plt.show()
