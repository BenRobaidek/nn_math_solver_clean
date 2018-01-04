import sys
import os
from random import shuffle

surfd = "surfaces/"

with open("ceq.txt") as f:
  eqs = f.read().strip().split("\n")
surfs = []
with open("cidx.txt") as f:
  idx = f.read().strip().split("\n")
for i in range(len(eqs)):

  fn = idx[i].split(".")[0]+".txt"
  with open(surfd+fn) as f:
    surfs.append(f.read())

assert(len(eqs)==len(surfs))

data = [surfs[i]+"\t"+eqs[i] for i in range(len(eqs))]
shuffle(data)
l = len(data)//10

with open("kdata_dev.tsv",'w') as f:
  f.write("\n".join(data[:l]))

with open("kdata_test.tsv",'w') as f:
  f.write("\n".join(data[l:l*2]))
with open("kdata_train.tsv",'w') as f:
  f.write("\n".join(data[l*2:]))
