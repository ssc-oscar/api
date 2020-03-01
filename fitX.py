import numpy as np
import sys,gzip,collections,gensim.models.doc2vec
from gensim.models import Doc2Vec
from collections import OrderedDict, namedtuple
import multiprocessing, random,unicodedata, re
from random import shuffle
import datetime
cores = multiprocessing.cpu_count()-2


tags = list()
ms = list()
nl = 0
# First read into tags and ms
sys.argv.pop(0)
lst=sys.argv.pop(0)
mCnt = int (sys.argv.pop(0))
ns = int (sys.argv.pop(0))
vs = int (sys.argv.pop(0))

f=gzip.open(lst)
for line in f:
 l = line.rstrip().decode('ascii', 'ignore')
 all = l.split(';')
 p = all .pop(0)
 a = all .pop(0)
 m = all
 tag = [ p, a ]
 tags.append(tag)
 ms.append(";".join(m))
 nl+=1

print ('records:' + str(nl))

ldoc = namedtuple('ldoc', 'words tags split')
docs = []
for l in enumerate([ (w,t) for w,t in zip(ms, tags) ]):
  words=l[1][0].split(";")
  tags=l[1][1]
  split=random.choice(['test','train','validate'])
  docs.append(ldoc(words, tags, split))

mod = Doc2Vec(dm=0, dbow_words=1, dm_mean=1, vector_size=vs, window=1, negative=ns, min_count=mCnt, workers=cores)

mod.build_vocab(docs)
dl = docs[:]

for epoch in range(20):
  mod.train(dl, epochs=mod.epochs, total_examples=mod.corpus_count)
  mod.alpha -= 0.002
  mod.min_alpha = mod.alpha
  mod.save("doc2vec."+str(vs)+"."+str(ns)+"."+str(mCnt)+"."+lst+"."+str(epoch))

