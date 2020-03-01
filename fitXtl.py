from gensim import corpora, similarities
import numpy as np
import sys,gzip,collections,gensim.models.doc2vec
from gensim.models import LsiModel, TfidfModel
from collections import OrderedDict, namedtuple
import multiprocessing, random,unicodedata, re
from random import shuffle
import datetime
cores = multiprocessing.cpu_count()-2


ms = list()
nl = 0
# First read into tags and ms
sys.argv.pop(0)
lst=sys.argv.pop(0)
nTopics = int (sys.argv.pop(0))
f=gzip.open(lst)
for l in f:
 all = l.rstrip().decode('ascii', 'ignore').split(';')
 p = all .pop(0)
 a = all .pop(0)
 m = all
 nl+=1
 ms.append (m)

print ('records:' + str(nl))
dictionary = corpora.Dictionary(ms)
dictionary.save('dict.'+lst)
doc = [ dictionary.doc2bow(text) for text in ms ]
modt = TfidfModel (corpus=doc, normalize=True)
modt.save("tfidf."+lst)
modl = LsiModel (corpus=modt[doc], num_topics=nTopics)
modl.save("tlsi."+lst+"."+str(nTopics))

