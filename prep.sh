#!/bin/bash

#prepare data mapping projects/time/author/apis for the following languages
for LA in jl pl R F Go Scala Rust Cs PY ipy JS C java rb
do zcat PtaPkgQ$LA.*.gz | lsort 500G -t\| | uniq | gzip >  PtaPkgQ$LA.s
  zcat PtaPkgQ$LA.s | perl -ane 'chop();($p,$t,$a,@ms) = split(/;/); for $m (@ms){print "$p;$m\n"}' | lsort 500G -t\; -k1,2 -u | gzip > P2Pkg$LA.s
  zcat PtaPkgQ$LA.s | perl -ane 'chop();($p,$t,$a,@ms) = split(/;/); for $m (@ms){print "$a;$m\n"}' | lsort 500G -t\; -k1,2 -u | gzip > a2Pkg$LA.s
done

#try on several small languages 'F', 'R', 'jl', 'pl', 'ipy'
(time python3 fit.py F R jl pl ipy) &

#one iteration takes 5 hr on da4 (see /da4_data/play/api)
# lets look at the second iteration


import gzip,collections,gensim.models.doc2vec,math
from gensim.models import Doc2Vec
mod = Doc2Vec.load ("doc2vec.QFRjlplipy.2")
mod = Doc2Vec.load ("doc2vec.QFipy.3")

#get most similar packages to language, project, author
mod.wv.similar_by_vector(mod.docvecs['R'])
[('extrafont', 0.9955250024795532), ('csnorm', 0.9952453374862671), ('knitr', 0.9948492050170898), ('stringr', 0.9943090081214905), ('matrixStats', 0.9934355020523071), ('building.h', 0.9933176636695862), ('scam', 0.9915322065353394), ('gridExtra', 0.9907146096229553), ('shinystan', 0.9894420504570007), ('esprdbfile.h', 0.9891785979270935)]

mod.wv.similar_by_vector(mod.docvecs['cran_tidyquery'])
[('HuffmanDecoder.jl', 0.46050825715065), ('general.fh', 0.4211460053920746), ('bokeh.palettes.all_palettes', 0.4210362434387207), ('OPN', 0.4131404757499695), ('qubarqu_nInJququbar_465_Sq1_specs.h', 0.41047632694244385), ('flask_pymongo.PyMongo', 0.39284461736679077), ('mapStats', 0.3890906870365143), ('curlib', 0.38832515478134155), ('cctk_Faces.h', 0.38744601607322693), ('PhageR', 0.38713282346725464)]

mod.wv.similar_by_vector(mod.docvecs['Yannick Spill <yannick.spill@crg.eu>']);
[('csnorm', 0.9973132610321045), ('extrafont', 0.9968918561935425), ('stringr', 0.9961603283882141), ('matrixStats', 0.9951667785644531), ('jiebaRD', 0.9946102499961853), ('knitr', 0.9934378862380981), ('flowCore', 0.9919644594192505), ('rhdf5', 0.9909929037094116), ('mgcv', 0.9899401664733887), ('scam', 0.9892599582672119)]

#get most similar languages, projects,authors to language, project, author
mod.docvecs.most_similar('R');
('F', 0.9947392344474792), ('Yannick Spill <yannick.spill@crg.eu>', 0.9929205179214478), ('AsaEE_ESP-rSource', 0.992476224899292), ('jhand <jhand@7d53e970-de11-0410-8a54-3d01b9da36cf>', 0.9924437999725342), ('2DegreesInvesting_PortCheck', 0.990721583366394), ('Clare2D <32903584+Clare2D@users.noreply.github.com>', 0.9905468225479126), ('Taylor Posey <taylor.m.posey@outlook.com>', 0.9884799718856812), ('tinaGNAW <tina@2degrees-investing.org>', 0.9880185127258301), ('Paul Fischer <fischerp@illinois.edu>', 0.987598180770874), ('12379Monty_scRNASeq', 0.9872961044311523)]

mod.docvecs.most_similar('cran_tidyquery')
it2:[('kungeinus_Prediction_Assignment_Writeup', 0.49396124482154846), ('parserpro_db_update', 0.48060914874076843), ('adisarid <adisarid@gmail.com>', 0.47186779975891113), ('danthemango <danthemango@gmail.com>', 0.4652522802352905), ('arnarg_plex_exporter', 0.46454471349716187), ('colin-combe_CLMS-UI', 0.4438340663909912), ('alanaw1_CulturalHitchhiking', 0.4377615451812744), ('lavanyaj09_BE223A', 0.4339529275894165), ('jnarhan_Kaggle-Pneumonia', 0.43245214223861694), ('gxe778_Trajectory-Inference-Methods-applied-on-early-cell-lines-from-human-embryo', 0.42638999223709106)]
it1:[('parserpro_db_update', 0.4985978603363037), ('kungeinus_Prediction_Assignment_Writeup', 0.49396124482154846), ('adisarid <adisarid@gmail.com>', 0.47186779975891113), ('danthemango <danthemango@gmail.com>', 0.4674111604690552), ('arnarg_plex_exporter', 0.46454471349716187), ('PeterHenell_goora', 0.4478600025177002), ('colin-combe_CLMS-UI', 0.44383400678634644), ('danthemango_ClientRG', 0.4412115514278412), ('alanaw1_CulturalHitchhiking', 0.4377615451812744), ('jnarhan_Kaggle-Pneumonia', 0.43245214223861694)]			       

mod.docvecs.most_similar('Yannick Spill <yannick.spill@crg.eu>')
it2:[('R', 0.992920458316803), ('3schwartz_SpecialeScrAndFun', 0.9900994300842285), ('Francois <briatte@gmail.com>', 0.9879549145698547), ('215ALab4_lab4', 0.9871786832809448), ('12379Monty_scRNASeq', 0.986408531665802), ('12379Monty <francois.collin@gmail.com>', 0.9862282872200012), ('3wen_elus', 0.9861852526664734), ('52North_tamis', 0.9858419299125671), ('tinaGNAW <tina@2degrees-investing.org>', 0.9855629205703735), ('2DegreesInvesting_PortCheck', 0.984999418258667)]
it1:[('3DGenomes_binless', 0.9939588904380798), ('R', 0.9929205179214478), ('3schwartz_SpecialeScrAndFun', 0.9900994300842285), ('Francois <briatte@gmail.com>', 0.9879549145698547), ('215ALab4_lab4', 0.9871785640716553), ('12379Monty_scRNASeq', 0.986408531665802), ('12379Monty <francois.collin@gmail.com>', 0.986228346824646), ('3wen_elus', 0.9861852526664734), ('52North_tamis', 0.9858419299125671), ('tinaGNAW <tina@2degrees-investing.org>', 0.9855630397796631)]

#get most similar packages to a package
mod.wv.most_similar('ggplot2')
it2: [('nortest', 0.9667873382568359), ('reshape2', 0.9574425220489502), ('tidyr', 0.9566653966903687), ('partykit', 0.9542033076286316), ('purrr', 0.9534142017364502), ('knitr', 0.9527696371078491), ('aim2_parameters.h', 0.9526889324188232), ('matrixStats', 0.9519108533859253), ('FlowSOM', 0.9512166976928711), ('Obspars.com', 0.9505057334899902)]
it1: ('tidyr', 0.9896135330200195), ('reshape2', 0.9894652366638184), ('nortest', 0.9879751205444336), ('dplyr', 0.9831089973449707), ('knitr', 0.9812949895858765), ('partykit', 0.9795119762420654), ('RColorBrewer', 0.9769814610481262), ('purrr', 0.9767657518386841), ('lubridate', 0.9766140580177307), ('data.table', 0.9752072095870972)]

mod.wv.most_similar('keras_learn')
it2:[('tensorbayes.nputils.log_sum_exp', 0.7950129508972168), ('neural_network_decision_tree.nn_decision_tree', 0.774245023727417), ('TechnicalAnalysis.TechnicalAnalysis', 0.768464207649231), ('models.naive_convnet.NaiveConvColoringModel', 0.762176513671875), ('model_VAE.VAE_mnist', 0.761232316493988), ('batch_generator.dir.DirIterator', 0.7570154070854187), ('optimizer.learing_rate_scheduling', 0.7539881467819214), ('models.yolov3_gpu_head.inference.restore_model', 0.7510530948638916), ('flickr8k_parse', 0.7491798400878906), ('ppo.NNValueFunction', 0.7474846839904785)]
it1:[('tensorbayes.nputils.log_sum_exp', 0.8044129610061646), ('models.yolov3_gpu_head.inference.restore_model', 0.7988119125366211), ('model_VAE.VAE_mnist', 0.7955332398414612), ('antTrainEnv_class.antTrainEnv_class', 0.784370481967926), ('models.naive_convnet.NaiveConvColoringModel', 0.7827848196029663), ('batch_generator.dir.DirIterator', 0.7793075442314148), ('neural_network_decision_tree.nn_decision_tree', 0.7765824198722839), ('TechnicalAnalysis.TechnicalAnalysis', 0.7733845710754395), ('envs.economy.jesusfv', 0.773059606552124), ('model.audio_u_net_dnn', 0.7707473039627075)]

#no most similar language,project.author from package, need to write a function

for la in ('F', 'R', 'jl', 'pl', 'ipy'):
 for lb in ('F', 'R', 'jl', 'pl', 'ipy'):
   print (la+":"+lb+" "+str(mod.docvecs.distance(la,lb))
it2:		  
F:R 0.005260765552520752
F:jl 0.5120232105255127
F:pl 0.18782222270965576
F:ipy 0.24761343002319336
R:jl 0.5035586059093475
R:pl 0.1807081699371338
R:ipy 0.2535156011581421
jl:pl 0.429531455039978
jl:ipy 0.8230383545160294
pl:ipy 0.4515225887298584

it1:
F:R 0.005260765552520752
F:jl 0.518935889005661
F:pl 0.160944402217865
F:ipy 0.2259724736213684
R:jl 0.5141101777553558
R:pl 0.14899468421936035
R:ipy 0.23450106382369995
jl:pl 0.46918046474456787
jl:ipy 0.8400852829217911
pl:ipy 0.41011691093444824


#measure distance between package and project/author/language
def dist (a, b):
 av = mod.wv.get_vector(a) 
 bv = mod.docvecs[b]
 return (sum(av*bv)/math.sqrt(sum(av*av)*sum(bv*bv)))

# save document and word vectors 
f = open('outDocs','w')  
for t in mod.docvecs.doctags.keys():
  f.write(t)
  for v in mod.docvecs[t]:
    f.write(';'+"{:1.12e}".format(v))
    f.write('\n')
f.close()
f = open('outWords','w')
for t in mod.wv.vocab.keys():
  f.write(t)
  for v in mod.wv[t]:
    f.write(';'+"{:1.12e}".format(v))
    f.write('\n')
f.close()



