import numpy as np
import sys,gzip,collections,gensim.models.doc2vec
from gensim.models import Doc2Vec
from collections import OrderedDict, namedtuple
import multiprocessing, random,unicodedata, re
from random import shuffle
import datetime, mmap, os, pickle
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.WARNING)

cores = multiprocessing.cpu_count()


mod = Doc2Vec .load ("doc2vec.QR.1518784533.9")
#mod = Doc2Vec .load ("doc2vec.QRust.0")
#mod = Doc2Vec .load ("all.d2v.200.50.20.5.1618784533.PtAPkgQ.0")


for line in sys.stdin:
  l = line.rstrip()
  (type, key) = l .split(';')
  if (type == 'api2api'):
    print (type + ";" + key + ";" +  str(mod.wv.most_similar (key)))
  else:  
    if (type == 'p2p'):
      print (type + ";" + key + ";" +  str(mod.docvecs.most_similar (key)))
    else:
      if (type == 'p2api'):
        print (type + ";" + key + ";" +  str(mod.wv.similar_by_vector (mod.docvecs[key])))
