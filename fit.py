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
lst="";
while (len(sys.argv):
 la = sys.argv.pop(0)
 lst=lst+la
 f=gzip.open("PtaPkgQ"+la+".s")
 for line in f:
  l = line.rstrip().decode('ascii', 'ignore')
  all = l.split(';')
  p = all .pop(0)
  t = all .pop(0)
  a = all .pop(0)
  m = all
  tag = [ la, p, a ]
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

mod = Doc2Vec(dm=1, dbow_words=1, dm_mean=1, vector_size=100, window=2, negative=100, min_count=2, workers=cores)

mod.build_vocab(docs)
dl = docs[:]

#mod = Doc2Vec.load ("doc2vec.model1.13")

for epoch in range(10):
  mod.train(dl, epochs=mod.epochs, total_examples=mod.corpus_count)
  mod.alpha -= 0.002
  mod.min_alpha = mod.alpha
  mod.save("doc2vec.Q"+lst+"."+str(epoch))

