import numpy as np
import math,sys,gzip,collections,gensim.models.doc2vec
from gensim.models import Doc2Vec
from collections import OrderedDict, namedtuple
import random, unicodedata, re
import datetime

f='/da4_data/play/api/doc2vecR.200.30.20.5.1518784533.eA.trained'
mod = Doc2Vec.load (f)

def dist (av, bv):
 return (sum(av*bv)/math.sqrt(sum(av*av)*sum(bv*bv)))

#experts in JS
na = 0
f=open("exA.csv","rb")
for line in f:
 h, m, nc, s, a = line.rstrip().decode('ascii', 'ignore').split(';')
 if a in mod .docvecs:          
  av= mod .docvecs[a]
  mv = mod.wv.get_vector(m)
  st = m +';'+nc+';'+s+';'+str(dist(av,mv))
  print(st)
  na += 1
 else: sys.stderr.write(a+'\n')

sys.stderr.write(str(na)+'\n')
  #print (m +';'+str(dist(av,mv))+';'+ str(mod.wv.most_similar([av])))
