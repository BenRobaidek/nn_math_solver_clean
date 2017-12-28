import matplotlib.pyplot as plt
import numpy as np
import re
from matplotlib.ticker import MaxNLocator

def main():
    showStats('./data/train.tsv', file_type='tsv')
    showStats('./data/val.tsv', file_type='tsv')
    showStats('./data/test.tsv', file_type='tsv')

    showStats('./data/train_0.2.tsv', file_type='tsv')
    showStats('./data/train_0.4.tsv', file_type='tsv')
    showStats('./data/train_0.6.tsv', file_type='tsv')
    showStats('./data/train_0.8.tsv', file_type='tsv')

    showStats('./data/val_0.2.tsv', file_type='tsv')
    showStats('./data/val_0.4.tsv', file_type='tsv')
    showStats('./data/val_0.6.tsv', file_type='tsv')
    showStats('./data/val_0.8.tsv', file_type='tsv')

def showStats(path, file_type='txt'):
    if file_type == 'txt':
        tgt_train = np.array(open(path).readlines())
    elif file_type == 'tsv':
        tgt_train = np.array(open(path).readlines())
        tgt_train = [example.split('\t')[1].strip() for example in tgt_train]
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
    plt.title(path)
    plt.ylabel('# of occurences')
    plt.xlabel('equation')
    plt.show()

    """
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
    """

if __name__ == '__main__':
    main()
