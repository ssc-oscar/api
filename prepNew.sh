#!/bin/bash
####################
#new ver R
####################
##################################################################################################################
# report stats for the ICSE submission Table 1-2
##################################################################################################################
for la in F jl R ipy pl Rust Dart Kotlin TypeScript Cs Go Scala rb java C rb PY JS ; do zcat  PtAPkgR$la.s| perl -e 'while(<STDIN>){chop();($p,$t,$a,@ms)=split(/;/);$as{$a}++;$ps{$p}++;$ls{$#ms}++, $n++;} print STDERR "'$la';$n;".(scalar(keys %as)).";".(scalar(keys %ps))."\n"; for $nl (keys %ls){print "$nl;$ls{$nl}\n"}' | gzip > PtAPkgR$la.nm;   done
######################
#lang;chages;authors;projects
F;1628760;24898;15623
jl;1297134;18666;35723
R;6822662;361754;516678
ipy;12160775;793261;1154120
pl;18780774;480615;547115
Rust;13599452;95712;148327
Dart;7036000;116317;164360
Kotlin;28129485;281469;429071
TypeScript;239416852;1605563;2253291
Cs;220871444;2092316;3092761
Go;123432323;490967;662355
Scala;36361141;176414;210175
rb;74618824;1222886;2343825
JS;55609812;3362191;7347050
PY;612708423;4795735;6820899
C;1780602124;3656965;4704446
java;1106084606;5063200;7512800

#lang;apis
for la in F jl R ipy pl Rust Dart Kotlin TypeScript Cs Go Scala rb java C rb PY JS ;do zcat PtAPkgR$la.s| perl -e 'while(<STDIN>){chop();($p,$t,$a,@ms)=split(/;/);for $m (@ms){$mm{$m}++;}} print "'$la';".(scalar(keys %mm))."\n";';  done
F;59349
jl;104725
R;85255
ipy;687085
pl;58942
Rust;818686
Dart;467863
Kotlin;6233673
TypeScript;7324019
Cs;6648357
Go;245102
Scala;3571593
java;85079403
JS;1105918
PY;17227676
rb;669297
C;2553521

#quantiles on the number of APIs per blob counting % of delta with <=30 APIs
for la in F jl R ipy pl Rust Dart Kotlin TypeScript Cs Go Scala rb java C rb PY JS; do zcat PtAPkgR$la.nm|lsort 1G -t\; -k2 -rn | awk -F\; '{n+=$2; c[$1]=$2} END {num=0;for (k=0;k<=30;k++){num+=c[k]};print "'$la'",num/n,n,$1}'; done
F 0.984606 1628760 106
jl 0.988739 1297134 108
R 0.998431 6822662 117
ipy 0.990262 12160775 1158
pl 0.999785 18780774 109
Rust 0.992667 13599452 118
Dart 0.993704 7036000 165
Kotlin 0.964331 28129485 1096
TypeScript 0.989924 239416852 1013
Cs 0.997572 220871444 150
Go 0.995657 123432323 1207
Scala 0.990467 36361141 1288
rb 0.997044 74618824 1002
java 0.916324 1106084606 1004
C 0.986453 1780602124 1007
rb 0.997044 74618824 1002
PY 0.989345 612708423 1001
JS 0.667983 55609812 10014
##################################################################################################################
##################################################################################################################



for la in F jl R ipy pl Rust Dart Kotlin TypeScript Cs Go Scala rb C java PY JS 
do zcat PtAPkgR$la.s | perl -e 'while(<STDIN>){chop();($p,$t,$a)=split(/;/);$pre=0; $pre=1 if $t>= 1518784533+3600*24*365.25; $pn{$p}{$pre}++; $an{$a}{$pre}++;}; for my $p (keys %pn){print "p;$p;$pn{$p}{1};$pn{$p}{0}\n";} for my $a (keys %an){print "a;$a;$an{$a}{1};$an{$a}{0}\n";}' | gzip > PtAPkgR$la.cnt
done

for la in F jl R ipy pl Rust Dart Kotlin TypeScript Cs Go Scala rb C java PY JS 
do zcat PtAPkgR$la.cnt| grep ^a | awk -F\; '{if($4>100 && $3>100)print $0}' > PtAPkgR$la.cnt100
done

for la in F jl R ipy pl Rust Dart Kotlin TypeScript Cs Go Scala rb C java PY JS 
do cut -d\; -f2 PtAPkgR$la.cnt100
done | lsort 1G -u | gzip > AuR100.gz


for la in F jl R ipy pl Rust Dart Kotlin TypeScript Cs Go Scala rb C java PY JS 
do zcat PtAPkgR$la.s 
done | perl ~/bin/grepField.perl AuR100.gz 3 | gzip > PtAPkgRAllA100.s


#fit in da5:/data/play/forks
ls -f /da0_data/play/*thruMaps/b2cPtaPkgR*.s| while read f; do la=$(echo $f|sed 's|.*b2cPtaPkgR||;s|\.[0-9]*\.s$||'); zcat $f | cut -d\; -f3- | ~/lookup/grepField.perl eap 1 | awk '{print "'$la';"$0}'; done | gzip > eap.api
zcat eap.api | perl ~/lookup/mp.perl 3 /da0_data/basemaps/gz/a2AQ.s | gzip > eAp.api
python3 fitXldRea.py eAp.api 200 30 20 5 1550908281 eAp eAp
python3 fitXldRea.py eA.api 200 30 20 5 1550908281 eA eA


#prepare api prediction
#first count delta for authors
zcat /da4_data/play/api/eAp.api|cut -d\; -f4|lsort 100G |uniq -c | sed 's|^\s*||;s|\s|;|' |perl -ane 's/\r//g;print'| gzip > eAp.c2a.gz
zcat eAp.c2a.gz|perl -ane 'chop();($n,$a)=split(/;/);print "$a\n" if $n >=100 && $n < 25000;' | gzip > eAp.a100
zcat /da4_data/play/api/eAp.api| ~/bin/grepField.perl eAp.a100 4 | gzip > /da4_data/play/api/eAp.api100

python3 fitXldRea.py eAp.api100 200 30 20 5 1518784533 eAp100 eAp100


##################################################################################################################
# report stats for the ICSE submission Table 3
##################################################################################################################
#need eAp.api
perl prepPredApi.perl | gzip > eAp.sAD
python3 measureAPIvR.py| gzip > measureAPIvR.gz
f='measureAPIvR.gz'
aa = read.table("eAp.c2a.gz",sep=";",quote="",comment.char="");
amed = as.character(aa[aa[,1]>100&aa[,1]<25000,2]);
x = read.table(f,sep=";",quote="",comment.char="");
ind = match(x[,1], amed,nomatch=0);
a = tapply(x$V5[ind>0], list(x$V1[ind>0],x$V3[ind>0],x$V2[ind>0]), mean,na.rm=T);

a1 = tapply(x$V5[ind>-1], list(x$V1[ind>-1],x$V3[ind>-1],x$V2[ind>-1]), mean,na.rm=T);

las=c("Dart","jl","R","ipy","pl","Rust","Kotlin","TypeScript","Cs","Go","Scala","rb","java","C","PY","JS");
res = c();
for (la in las){
 res=rbind(res, c(t.test(a[,2,la]-a[,3,la])$estimate,t.test(a[,2,la]-a[,3,la])$p.value))
}
dimnames(res)[[1]]=las;	       
res
Dart       0.41207559  3.120130e-92
jl         0.20955929  8.565540e-05
R          0.14442871  1.455249e-06
ipy        0.19954272  6.677312e-65
pl         0.04645639  2.852958e-13
Rust       0.20947185 2.010680e-151
Kotlin     0.20606213 1.090052e-139
TypeScript 0.23007271  0.000000e+00
Cs         0.24571956 6.162232e-137
Go         0.14883848  0.000000e+00
Scala      0.20382756  8.451967e-89
rb         0.16819598 3.796952e-188
java       0.12770313  0.000000e+00
C          0.13112611  0.000000e+00
PY         0.11885238  0.000000e+00
JS         0.09861961  0.000000e+00



#prepare project prediction
perl prepPredPrj.perl | gzip > eAp.sAPD
python3 measureAPvR.py| gzip > measureAPvR.gz

aa = read.table("eAp.c2a.gz",sep=";",quote="",comment.char="");
amed = as.character(aa[aa[,1]>100&aa[,1]<25000,2]);
f='measureAPvR.gz'
x = read.table(f,sep=";",quote="",comment.char="");
#zz=table(x[,1]);ind = match(x[,1],names(zz)[zz>5],nomatch=0)
ind = match(x[,1], amed,nomatch=0);
a = tapply(x$V4[ind>0], list(x$V1[ind>0],x$V2[ind>0]), mean,na.rm=T);
t.test(a[,2]-a[,3])
data:  a[, 2] - a[, 3]
t = 8.8863, df = 8614, p-value < 2.2e-16
alternative hypothesis: true mean is not equal to 0
95 percent confidence interval:
 0.01347707 0.02110568
sample estimates:
 mean of x 
0.01729138 


#prepare Author prediction
perl prepPredAuth.perl | gzip > eAp.sPAD
python3 measurePAvR.py|perl -ane 's/\r//g;print'| gzip > measurePAvR.gz
f='measurePAvR.gz'
x = read.table(f,sep=";",quote="",comment.char="");
ind = match(x[,3], amed,nomatch=0);
a = tapply(x$V4[ind>0], list(x$V1[ind>0],x$V2[ind>0]), mean,na.rm=T);
t.test(a[,2]-a[,3])
data:  a[, 2] - a[, 3]
t = 18.216, df = 513, p-value < 2.2e-16
alternative hypothesis: true mean is not equal to 0
95 percent confidence interval:
 0.1253809 0.1556942
sample estimates:
mean of x 
0.1405376

##################################################################################################################
#do PRs
# Try new PR data
perl joinPrs.perl > joinedPrs.csv
cut -d\; -f2 joinedPrs.csv | sort -u -t\; | gzip > au.prs.new
zcat au.prs.new | perl ~/lookup/mp.perl 0 /da0_data/basemaps/gz/a2AQ.s | lsort 1G -t\; -u | gzip > Au.prs.new

for la in F jl R ipy pl Rust Dart Kotlin TypeScript Cs Go Scala rb C java PY JS 
do zcat PtAPkgR$la.s | perl ~/lookup/grepField.perl Au.prs.new 3 | gzip > PtAPkgR$la.prs.s
done

cat joinedPrs.csv | sed 's|https://github.com/||;s|/pull/|;|;s|/|_|;' | perl ~/lookup/mp.perl 0 /da0_data/basemaps/gz/p2PR.s | perl ~/lookup/mp.perl 2 /da0_data/basemaps/gz/a2AQ.s  >  joinedPrsAP.csv


cut=1518784533
for la in F jl R ipy pl Rust Dart Kotlin TypeScript Go Scala rb Cs C java PY JS 
do zcat PtAPkgR$la.prs.s | awk '{print "'$la';"$0}'
done | perl -e 'while(<STDIN>){chop(); ($la,$p,$t,$a,@ms)=split(/;/);if ($t < '$cut'){ for $m (@ms){$k{"$p;$a;$la"}{$m}++}}};while (($p, $v)=each %k){@ms=sort keys %{$v}; print "$p;".(join ";", @ms)."\n";}' | gzip > prs.Rnew.s2.$cut
for la in F jl R ipy pl Rust Dart Kotlin TypeScript Go Scala rb Cs C java PY JS 
do zcat PtAPkgR$la.prs.s
done | perl -e 'while(<STDIN>){chop(); ($p,$t,$a,@ms)=split(/;/);if ($t >= '$cut'){ for $m (@ms){$k{"$p;$a"}{$m}++}}};while (($p, $v)=each %k){@ms=sort keys %{$v}; print "$p;".(join ";", @ms)."\n";}' | gzip > prs.Rnew.s4.$cut
perl cmpAprsvRnew.perl prs.Rnew $cut | gzip > prs.Rnew.sAD.$cut

python3 measureAPprsvRnew.py /da4_data/play/api/doc2vecR.200.30.20.5.$cut.JS.trained prs.Rnew.sAD.$cut | perl -ane 's/\r//g;print' > out.prs.JSRnew.$cut

python3 fitXldRprs.py PtAPkgR 200 30 20 5 1518784533 PRs F jl R ipy pl Rust Dart Kotlin TypeScript Cs Go Scala rb C java PY JS 2> fitPRs.err
python3 measureAPprsvRnew.py doc2vecR.200.30.20.5.1518784533.PRs.trained prs.Rnew.sAD.$cut 2> missPRs | perl -ane 's/\r//g;print' > out.prs.PRsRnew.$cut

x=read.table("out.prs.PRsRnew.1518784533",sep=";",quote="",comment.char="");
if (length(grep('\\bbot\\b', x$V1,perl=T,ignore.case=T) > 0)) x=x[-grep('\\(bot\\)', x$V1,perl=T,ignore.case=T),]
if (length(grep('Automation', x$V1,perl=T,ignore.case=T) > 0)) x=x[-grep('Automation', x$V1,perl=T,ignore.case=T),]
sim=x[,dim(x)[2]];
y = x[,3]=='True';
prev = x[,4]>0;
z=x[,-c(1:4,20,23)]
summary(glm(y~sim,family=binomial))$coefficients
             Estimate Std. Error  z value      Pr(>|z|)
(Intercept) 0.2851478 0.01129879 25.23701 1.572824e-140
sim         0.5007773 0.02464404 20.32043  8.483674e-92

#JS specific, where most of PRs are
x=read.table("out.prs.JSRnew.1518784533",sep=";",quote="",comment.char="");
if (length(grep('\\bbot\\b', x$V1,perl=T,ignore.case=T) > 0)) x=x[-grep('\\(bot\\)', x$V1,perl=T,ignore.case=T),]
if (length(grep('Automation', x$V1,perl=T,ignore.case=T) > 0)) x=x[-grep('Automation', x$V1,perl=T,ignore.case=T),]
#nn = table(as.character(x[,1]));ind = match(x[,1], names(nn[nn<3]),nomatch=0); x=x[ind > 0,];
#ind = match(x[,1], amed,nomatch=0);x=x[ind > 0,];
#x=x[x$V8+x$V7>0,]
#response
#y=cbind(x$V8,x$V7)
sim=x[,dim(x)[2]];
y = x[,3]=='True';
prev = x[,4]>0;
z=x[,-c(1:4,20:23)]

summary(glm(y~sim,family=binomial))$coefficients
             Estimate Std. Error  z value      Pr(>|z|)
(Intercept) 0.2584875 0.01106767 23.35519 1.220673e-120
sim         0.7322888 0.02614693 28.00669 1.346974e-172


form=as.formula(paste(c("y~sim",names(z)),collapse="+"));
mod = glm(form,family=binomial,data=z,subs=!prev)
summary(mod);
Coefficients:
              Estimate Std. Error  z value Pr(>|z|)    
(Intercept) -1.582e+00  5.351e-02  -29.565  < 2e-16 ***
sim          5.044e-01  9.093e-02    5.547 2.90e-08 ***
V5          -1.084e-04  1.946e-05   -5.572 2.51e-08 ***
V6           1.132e+00  3.422e-02   33.076  < 2e-16 ***
V7          -7.296e-06  9.040e-07   -8.071 6.98e-16 ***
V8           3.209e+00  5.957e-02   53.864  < 2e-16 ***
V9          -2.346e-01  2.291e-02  -10.240  < 2e-16 ***
V10         -1.436e-06  1.286e-08 -111.726  < 2e-16 ***
V11          6.754e-03  1.363e-03    4.956 7.18e-07 ***
V12          1.505e-02  1.014e-03   14.841  < 2e-16 ***
V13         -2.105e-02  7.262e-04  -28.993  < 2e-16 ***
V14          6.316e-06  1.569e-06    4.026 5.67e-05 ***
V15         -7.116e-06  1.921e-06   -3.704 0.000212 ***
V16         -1.608e-03  1.918e-04   -8.384  < 2e-16 ***
V17          2.648e-01  2.344e-02   11.299  < 2e-16 ***
V18         -4.830e-01  3.881e-01   -1.244 0.213322    
V19          9.170e-01  2.867e-02   31.980  < 2e-16 ***
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

(Dispersion parameter for binomial family taken to be 1)

    Null deviance: 86374  on 62419  degrees of freedom
Residual deviance: 50436  on 62403  degrees of freedom
AIC: 50470

library(car)
> vif(mod)
     sim       V5       V6       V7       V8       V9      V10      V11 
1.065648 1.031543 1.414611 1.242080 1.101948 1.033086 1.288278 1.246721 
     V12      V13      V14      V15      V16      V17      V18      V19 
1.276105 1.395016 1.938039 1.824200 1.265469 1.083477 1.001566 1.527913 
		

mod = glm(y~sim+V5+V6+ V7 + V8 + V9 + V10 + V11 + V12 + V17+V18+V19+V21+V22,family=binomial,data=z,subs=!prev)
summary(mod)
              Estimate Std. Error  z value Pr(>|z|)    
(Intercept) -1.648e+00  5.429e-02  -30.363  < 2e-16 ***
sim          6.020e-01  9.332e-02    6.450 1.12e-10 ***
V5          -1.154e-03  1.084e-04  -10.641  < 2e-16 ***
V6           1.204e+00  3.529e-02   34.117  < 2e-16 ***
V7          -4.674e-06  9.073e-07   -5.152 2.58e-07 ***
V8           3.055e+00  5.997e-02   50.944  < 2e-16 ***
V9          -2.546e-01  2.317e-02  -10.988  < 2e-16 ***
V10         -1.426e-06  1.302e-08 -109.481  < 2e-16 ***
V11          2.421e-03  1.351e-03    1.792   0.0732 .  
V12          4.903e-03  9.239e-04    5.306 1.12e-07 ***
V17          2.615e-01  2.365e-02   11.057  < 2e-16 ***
V18         -4.678e-01  3.971e-01   -1.178   0.2388    
V19          7.501e-01  2.872e-02   26.119  < 2e-16 ***
V21         -1.510e-06  1.229e-06   -1.228   0.2194    
V22          1.375e-05  2.133e-06    6.445 1.15e-10 ***
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

(Dispersion parameter for binomial family taken to be 1)

    Null deviance: 81054  on 58590  degrees of freedom
Residual deviance: 49093  on 58576  degrees of freedom


### old PR data
for la in F jl R ipy pl Rust Dart Kotlin TypeScript Cs Go Scala rb C java PY JS 
do zcat PtaPkgR$la.prs.s
done | perl -e 'while(<STDIN>){chop(); ($p,$t,$a,@ms)=split(/;/);if ($t >= '$cut'){ for $m (@ms){$k{"$p;$a"}{$m}++}}};while (($p, $v)=each %k){@ms=sort keys %{$v}; print "$p;".(join ";", @ms)."\n";}' | gzip > prs.R.s4.$cut
cat PRdata_newA.csv | perl ~/lookup/mp.perl 1 /da0_data/basemaps/gz/p2PR.s > PRdata_newAR.csv

for la in F jl R ipy pl Rust Dart Kotlin TypeScript Cs Go Scala rb java C PY JS 
do zcat PtAPkgR$la.s | perl ~/bin/grepField.perl au.prs 3 | gzip > PtaPkgR$la.prs.s
done
cut=1550908281
cut=1518784533
for la in F jl R ipy pl Rust Dart Kotlin TypeScript Cs Go Scala rb C java PY JS 
do zcat PtaPkgR$la.prs.s | awk '{print "'$la';"$0}'
done | perl -e 'while(<STDIN>){chop(); ($la,$p,$t,$a,@ms)=split(/;/);if ($t < '$cut'){ for $m (@ms){$k{"$p;$a;$la"}{$m}++}}};while (($p, $v)=each %k){@ms=sort keys %{$v}; print "$p;".(join ";", @ms)."\n";}' | gzip > prs.R.s2.$cut
for la in F jl R ipy pl Rust Dart Kotlin TypeScript Cs Go Scala rb C java PY JS 
do zcat PtaPkgR$la.prs.s
done | perl -e 'while(<STDIN>){chop(); ($p,$t,$a,@ms)=split(/;/);if ($t >= '$cut'){ for $m (@ms){$k{"$p;$a"}{$m}++}}};while (($p, $v)=each %k){@ms=sort keys %{$v}; print "$p;".(join ";", @ms)."\n";}' | gzip > prs.R.s4.$cut
perl cmpAprsvR.perl prs.R $cut | gzip > prs.R.sAD.$cut

x=read.table("out.prs.R100.1518784533",sep=";",quote="",comment.char="");
if (length(grep('\\bbot\\b', x$V1,perl=T,ignore.case=T) > 0)) x=x[-grep('\\(bot\\)', x$V1,perl=T,ignore.case=T),]
if (length(grep('Automation', x$V1,perl=T,ignore.case=T) > 0)) x=x[-grep('Automation', x$V1,perl=T,ignore.case=T),]
#nn = table(as.character(x[,1]));ind = match(x[,1], names(nn[nn<3]),nomatch=0); x=x[ind > 0,];
#ind = match(x[,1], amed,nomatch=0);x=x[ind > 0,];
x=x[x$V8+x$V7>0,]
#response
y=cbind(x$V8,x$V7)
sim=x$V9
summary(glm(y~sim,family=binomial))$coefficients


for la in JS
do zcat PtaPkgR$la.prs.s | awk '{print "'$la';"$0}'
done | perl -e 'while(<STDIN>){chop(); ($la,$p,$t,$a,@ms)=split(/;/);if ($t < '$cut'){ for $m (@ms){$k{"$p;$a;$la"}{$m}++}}};while (($p, $v)=each %k){@ms=sort keys %{$v}; print "$p;".(join ";", @ms)."\n";}' | gzip > prs.JSR.s2.$cut
for la in JS
do zcat PtaPkgR$la.prs.s
done | perl -e 'while(<STDIN>){chop(); ($p,$t,$a,@ms)=split(/;/);if ($t >= '$cut'){ for $m (@ms){$k{"$p;$a"}{$m}++}}};while (($p, $v)=each %k){@ms=sort keys %{$v}; print "$p;".(join ";", @ms)."\n";}' | gzip > prs.JSR.s4.$cut

perl cmpAprsvR.perl prs.JSR $cut | gzip > prs.JSR.sAD.$cut
#python3 measureAPprsvR.py /da4_data/play/api/doc2vecR.200.30.20.5.$cut.eAp.trained prs.R.sAD.$cut |perl -ane 's/\r//g;print' > out.prs.R.$cut
#python3 measureAPprsvR.py /da4_data/play/api/doc2vecR.200.30.20.5.1550908281.eAp.trained prs.R.sAD.$cut |perl -ane 's/\r//g;print' > out.prs.R.$cut

python3 measureAPprsvR.py /da4_data/play/api/doc2vecR.200.30.20.5.1550908281.eAp.trained prs.JSR.sAD.$cut |perl -ane 's/\r//g;print' > out.prs.JSR.$cut

python3 measureAPprsvR.py /da4_data/play/api/doc2vecR.200.30.20.5.1550908281.eA.trained prs.JSR.sAD.$cut |perl -ane 's/\r//g;print' > out.prs.JSR1.$cut

python3 measureAPprsvR.py /da4_data/play/api/doc2vecR.200.30.20.5.$cut.eA.trained prs.R.sAD.$cut |perl -ane 's/\r//g;print' > out.prs.R1.$cut
python3 measureAPprsvR.py /da4_data/play/api/doc2vecR.200.30.20.5.$cut.eAp100.trained prs.R.sAD.$cut |perl -ane 's/\r//g;print' > out.prs.R100.$cut


aa = read.table("eAp.c2a.gz",sep=";",quote="",comment.char="");
amed = as.character(aa[aa[,1]>100&aa[,1]<25000,2]);
x=read.table("out.prs.R100.1518784533",sep=";",quote="",comment.char="");
if (length(grep('\\bbot\\b', x$V1,perl=T,ignore.case=T) > 0)) x=x[-grep('\\(bot\\)', x$V1,perl=T,ignore.case=T),]
if (length(grep('Automation', x$V1,perl=T,ignore.case=T) > 0)) x=x[-grep('Automation', x$V1,perl=T,ignore.case=T),]
#nn = table(as.character(x[,1]));ind = match(x[,1], names(nn[nn<3]),nomatch=0); x=x[ind > 0,];
#ind = match(x[,1], amed,nomatch=0);x=x[ind > 0,];
x=x[x$V8+x$V7>0,]
#response
y=cbind(x$V8,x$V7)
sim=x$V9
summary(glm(y~sim,family=binomial))$coefficients
               Estimate Std. Error    z value      Pr(>|z|)
(Intercept) -0.39494073 0.01360379 -29.031662 2.622661e-185
sim          0.09917047 0.03403742   2.913572  3.573196e-03

#
#nn = table(as.character(x[,1]));ind = match(x[,1], names(nn[nn<3]),nomatch=0); x=x[ind > 0,];


#############################################
#do self-assessment Table 5-6 ICSE
#############################################
python3 m675vR.py > out675.vR

mttp = function (x) t.test(x)$p.value
mtte = function (x) t.test(x)$estimate

z=read.table('out675.vR',sep=";",quote="",comment.char="")

##################################################
# Table 5 ICSE
##################################################
summary(lm(V4~-1+V1+log(V2)+V3,data=z))
             Estimate Std. Error t value Pr(>|t|)    
V1mongodb   0.2494751  0.0130248  19.154  < 2e-16 ***
V1react     0.3070126  0.0111510  27.532  < 2e-16 ***
V1socketio  0.4220339  0.0118403  35.644  < 2e-16 ***
log(V2)    -0.0002471  0.0015631  -0.158    0.874    
V3          0.0143766  0.0030008   4.791 1.81e-06 ***
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Residual standard error: 0.1232 on 1602 degrees of freedom
Multiple R-squared:  0.8992,    Adjusted R-squared:  0.8988 
F-statistic:  2857 on 5 and 1602 DF,  p-value: < 2.2e-16



##################################################
# Table 6 ICSE
##################################################
summary(lm(V3~-1+V1+log(V2)+V4,data=z))
           Estimate Std. Error t value Pr(>|t|)    
V1mongodb   2.54606    0.10101  25.207  < 2e-16 ***
V1react     2.94646    0.08426  34.969  < 2e-16 ***
V1socketio  1.93059    0.12187  15.841  < 2e-16 ***
log(V2)     0.11060    0.01262   8.762  < 2e-16 ***
V4          0.98250    0.20508   4.791 1.81e-06 ***
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Residual standard error: 1.018 on 1602 degrees of freedom
Multiple R-squared:  0.919,     Adjusted R-squared:  0.9187 
F-statistic:  3633 on 5 and 1602 DF,  p-value: < 2.2e-16


######################################################################################
######################################################################################


#prepare data mapping projects/time/author/apis for the following languages
for LA in R Rust
do zcat /da0_data/play/${LA}thruMaps/b2cPtaPkgR${LA}.*.s | cut -d\; -f3- | lsort 100G -t\; -k2  | uniq | gzip >  PtaPkgR$LA.s
done

#this is for R and Rust
python3 fitXldR.py /fast/PtaPkgR 200 30 20 5 1618784533 Rust Rust
cut -d\; -f1 ChrisRust.P2p | awk '{print "p2p;"$1}' | grep -v HyeonuPark_srtp | grep -v Isan-Rivkin_rsocket-rs | python3 predXpclF.py 100

for LA in jl F Dart ipy pl Kotlin Scala Go; do zcat  /da0_data/play/${LA}thruMaps/b2cPtaPkgR${LA}.*.s | cut -d\; -f3- | perl -ane 'chop();($a,$b,$c,@ms)=split(/;/);%o=();for $m (@ms){$m=~s|^\s+||;$m=~s|\s+$||;$m=~s|^.*/||;$o{$m}++ if $m ne ""}; print "$a;$b;$c;".(join ";", sort keys %o)."\n";' | lsort 100G -t\; -k2,3 | uniq | gzip > PtaPkgR$LA.s; done

for la in rb Go TypesScript JS Cs PY java C; do zcat  /da0_data/play/${LA}thruMaps/b2cPtaPkgR${LA}.*.s | cut -d\; -f3- | perl -ane 'chop();($a,$b,$c,@ms)=split(/;/);%o=();for $m (@ms){$m=~s|^\s+||;$m=~s|\s+$||;$m=~s|^.*/||;$o{$m}++ if $m ne ""}; print "$a;$b;$c;".(join ";", sort keys %o)."\n";' | lsort 100G -t\; -k2,3 | uniq | gzip > PtaPkgR$LA.s; done

#just rust
python3 fitXldR.py /fast/PtaPkgR 200 30 20 5 1618784533 RRust R Rust
python3 fitXldR.py /fast/PtaPkgR 200 30 20 5 1618784533 Rust Rust

cut -d\; -f1 ChrisRust.P2p | awk '{print "p2p;"$1}' | grep -v HyeonuPark_srtp| python3 predXpclF.py 1000 .3 > dist
cut -d\; -f2- dist | awk -F\; '{print $3";"$2";"$1}' | perl connectExportVwP2a.perl dist
143751 nodes and 1152000

cp -p dist.versions ~/src/networkit
zcat dist.weights > ~/src/networkit/w
zcat dist.versions | ./clusterw  143751 1152000 | gzip > dist.PLM
modularity=0.611794 nver=143751 clusters=36032 largest=24916
modularity=0.616274 nver=143751 clusters=36032 largest=29823

zcat ~/src/networkit/dist.PLM | perl rank1.perl  dist | gzip > dist.crank.map
zcat dist.crank.map | lsort 1G -t\; -k3 -rn | awk -F\; '{if ($1 != $2) print $0}' | head
therealprof_mkw41z-hal;eldruin_hdc20xx-rs;1.486938;1.508500
m9s_xmc1000;eldruin_hdc20xx-rs;1.479339;1.508500
TeXitoi_bme280-rs;eldruin_hdc20xx-rs;1.445716;1.508500
no111u3_serialio;eldruin_hdc20xx-rs;1.433340;1.508500
thenewwazoo_elatec-twn4-simple;eldruin_hdc20xx-rs;1.427218;1.508500
fionawhim_cortex-m-systick-countdown;eldruin_hdc20xx-rs;1.409276;1.508500
smart-leds-rs_apa102-spi-rs;eldruin_hdc20xx-rs;1.392477;1.508500
JoshMcguigan_tsl256x;eldruin_hdc20xx-rs;1.391639;1.508500
therealprof_stm32f767-hal;eldruin_hdc20xx-rs;1.389100;1.508500
lucazulian_l298n;eldruin_hdc20xx-rs;1.387941;1.508500

#if mixed with R
m9s_xmc1000;therealprof_mkw41z-hal;1.438271;1.477410
eldruin_hdc20xx-rs;therealprof_mkw41z-hal;1.421558;1.477410
TeXitoi_bme280-rs;therealprof_mkw41z-hal;1.406271;1.477410
thenewwazoo_elatec-twn4-simple;therealprof_mkw41z-hal;1.405293;1.477410
no111u3_serialio;therealprof_mkw41z-hal;1.396393;1.477410
smart-leds-rs_apa102-spi-rs;therealprof_mkw41z-hal;1.380515;1.477410
fionawhim_cortex-m-systick-countdown;therealprof_mkw41z-hal;1.379802;1.477410
JoshMcguigan_shift-register-driver;therealprof_mkw41z-hal;1.368159;1.477410
mathk_mfxstm32l152;therealprof_mkw41z-hal;1.362509;1.477410
richardeoin_stm32h7-fmc;therealprof_mkw41z-hal;1.362391;1.477410

zcat dist.crank.map | lsort 1G -t\; -k3 -rn | awk -F\; '{if ($1 != $2) print $0}' | grep '<' | head
Henk Dieter Oordt <henkdieter@oordt.net>;eldruin_hdc20xx-rs;0.489912;1.508500
Albert Moravec <albert.moravec@keenmate.com>;eldruin_hdc20xx-rs;0.487719;1.508500
PinkNoize <21967246+PinkNoize@users.noreply.github.com>;eldruin_hdc20xx-rs;0.465511;1.508500
cmoran <cmoran@eri.ucsb.edu>;eldruin_hdc20xx-rs;0.454001;1.508500
Roma Sokolov <roma.sokolov@imc.com>;eldruin_hdc20xx-rs;0.449869;1.508500
Trond Hbertz Emaus <trondhe@gmail.com>;eldruin_hdc20xx-rs;0.423623;1.508500
inazarenko <inazarenko@google.com>;eldruin_hdc20xx-rs;0.422698;1.508500
Igor Nazarenko <inazarenko@google.com>;eldruin_hdc20xx-rs;0.414672;1.508500
Neil Goldader <cgoldader@zipcar.com>;eldruin_hdc20xx-rs;0.408192;1.508500
irwineffect <irwineffect@users.noreply.github.com>;eldruin_hdc20xx-rs;0.405343;1.508500

#if mixed with R
inazarenko <inazarenko@google.com>;therealprof_mkw41z-hal;0.464728;1.477410
Henk Dieter Oordt <henkdieter@oordt.net>;therealprof_mkw41z-hal;0.452849;1.477410
PinkNoize <21967246+PinkNoize@users.noreply.github.com>;therealprof_mkw41z-hal;0.451155;1.477410
Roma Sokolov <roma.sokolov@imc.com>;therealprof_mkw41z-hal;0.430368;1.477410
Igor Nazarenko <inazarenko@google.com>;therealprof_mkw41z-hal;0.415571;1.477410
Felipe Lalanne <flalanne@niclabs.cl>;therealprof_mkw41z-hal;0.402307;1.477410
Anderson Nascimento <anascime@gmail.com>;therealprof_mkw41z-hal;0.394570;1.477410
cjbe <chris.ballance@physics.ox.ac.uk>;therealprof_mkw41z-hal;0.390449;1.477410
Neil Goldader <cgoldader@zipcar.com>;therealprof_mkw41z-hal;0.389320;1.477410
Garrett Greenwood <garrettagreenwood@gmail.com>;therealprof_mkw41z-hal;0.388886;1.477410




####################
#old ver Q
####################
#Investigate joint frequencies
zcat PtAPkgQR.s0 | cut -d\; -f4- | perl -e 'while(<STDIN>){chop();@m=sort split(/;/);for $i (0..$#m){$a{$m[$i]}++;for $j (($i+1)..$#m){$n{$m[$i]}{$m[$j]}++;$n{$m[$j]}{$m[$i]}++}}};for $i (keys %a){for $j (keys %a){ print "$i;$j;$a{$i};$a{$j};$n{$i}{$j}\n" if ($i cmp $j)<0 && $a{$j}>5000 && $a{$i} > 5000 }}' | gzip > crosstab.gz


x=read.table("crosstab.gz", sep=";",quote="",comment.char="")
names(x)=c("a","b","na","nb","nab")
x$mn = apply(x[,c("na","nb")],1,min)
x$mor = x$mn/(x$nab+1);
x$tot=x$na+x$nb-x$nab;
x$ind=(x$na/x$tot * x$nb/x$tot);
x$pab = x$nab/x$tot;
x$or = x$ind/(1-x$ind)*(1-x$pab)/x$pab

#x=x[x$na>1000&x$nb>1000&x$or>3,]


myftestl = function(y){
  y=as.integer(y)
  res = fisher.test(matrix(c(y[1]-y[3],  y[3], y[3], y[2]-y[3]),ncol=2))
  res$conf.int[1];
}
myftestu = function(y){
  y=as.integer(y)
  res = fisher.test(matrix(c(y[1]-y[3],  y[3], y[3], y[2]-y[3]),ncol=2))
  res$conf.int[2];
}
x$orl = apply(x[,3:5],1,myftestl)
x$oru = apply(x[,3:5],1,myftestu)

quantile(x$oru)

y = x[x$a=='tidyr'&x$b=='readr',3:5]
fisher.test(matrix(as.integer(c(y[1]-y[3],  y[3], y[3], y[2]-y[3])),ncol=2))

 Fisher's Exact Test for Count Data

data:  matrix(as.integer(c(y[1] - y[3], y[3], y[3], y[2] - y[3])), ncol = 2)
p-value < 2.2e-16
alternative hypothesis: true odds ratio is not equal to 1
95 percent confidence interval:
 13.59858 13.99002
sample estimates:
odds ratio 
   13.7906 



#prepare data mapping projects/time/author/apis for the following languages
for LA in jl pl R F Go Scala Rust Cs PY ipy JS C java rb
do zcat PtaPkgQ$LA.*.gz | lsort 500G -t\| | uniq | gzip >  PtaPkgQ$LA.s
  zcat PtaPkgQ$LA.s | perl -ane 'chop();($p,$t,$a,@ms) = split(/;/); for $m (@ms){print "$p;$m\n"}' | lsort 500G -t\; -k1,2 -u | gzip > P2Pkg$LA.s
  zcat PtaPkgQ$LA.s | perl -ane 'chop();($p,$t,$a,@ms) = split(/;/); for $m (@ms){print "$a;$m\n"}' | lsort 500G -t\; -k1,2 -u | gzip > a2Pkg$LA.s
done

#Select ML/AI 
zcat PtaPkgQPY.s | grep -iE 'systemml|cntk|opennn|pandas|numpy|tensorflow|random|sklearn|gensim|nltk|scipy|skimage|datacube|matplotlib|face_recognition|fastai|keras|torch|basicnn|DecisionTree|baseline_cnn|pyaicnn|mtcnn_detector|nnclf|cnn|clustering|svm|caffe|scikit|mlib|torch|theano|veles|h2o' | cut -d\; -f1 | uniq | gzip > b.gz
zcat PtaPkgQPY.s | perl ~/lookup/grepField.perl b.gz 1 | gzip > PtaPkgQPYml.s


#try on several small languages 'F', 'R', 'jl', 'pl', 'ipy'
(time python3 fit.py F R jl pl ipy) &

#one iteration takes 5 hr on da4 (see /da4_data/play/api)
# lets look at the second iteration


import gzip,collections,gensim.models.doc2vec,math
from gensim.models import Doc2Vec
mod = Doc2Vec.load ("doc2vec.QFRjlplipy.2")
mod = Doc2Vec.load ("doc2vec.QFipy.7")

#get most similar packages to language, project, author
mod.wv.similar_by_vector(mod.docvecs['R'])
it1-7: [('extrafont', 0.9955250024795532), ('csnorm', 0.9952453374862671), ('knitr', 0.9948492050170898), ('stringr', 0.9943090081214905), ('matrixStats', 0.9934355020523071), ('building.h', 0.9933176636695862), ('scam', 0.9915322065353394), ('gridExtra', 0.9907146096229553), ('shinystan', 0.9894420504570007), ('esprdbfile.h', 0.9891785979270935)]

mod.wv.similar_by_vector(mod.docvecs['cran_tidyquery'])
it7:   [('HuffmanDecoder.jl', 0.46050825715065), ('general.fh', 0.4211460053920746), ('OPN', 0.4155998229980469), ('qubarqu_nInJququbar_465_Sq1_specs.h', 0.41047632694244385), ('arcgis.geocode', 0.4010382294654846), ('dataset_export.jl', 0.39810460805892944), ('NQS_Header', 0.3891255855560303), ('mapStats', 0.3890906870365143), ('curlib', 0.38832515478134155), ('cctk_Faces.h', 0.38744601607322693)]
ii1-3: [('HuffmanDecoder.jl', 0.46050825715065), ('general.fh', 0.4211460053920746), ('bokeh.palettes.all_palettes', 0.4210362434387207), ('OPN', 0.4131404757499695), ('qubarqu_nInJququbar_465_Sq1_specs.h', 0.41047632694244385), ('flask_pymongo.PyMongo', 0.39284461736679077), ('mapStats', 0.3890906870365143), ('curlib', 0.38832515478134155), ('cctk_Faces.h', 0.38744601607322693), ('PhageR', 0.38713282346725464)]

mod.wv.similar_by_vector(mod.docvecs['Yannick Spill <yannick.spill@crg.eu>']);
it1-7: [('csnorm', 0.9973132610321045), ('extrafont', 0.9968918561935425), ('stringr', 0.9961603283882141), ('matrixStats', 0.9951667785644531), ('jiebaRD', 0.9946102499961853), ('knitr', 0.9934378862380981), ('flowCore', 0.9919644594192505), ('rhdf5', 0.9909929037094116), ('mgcv', 0.9899401664733887), ('scam', 0.9892599582672119)]

#get most similar languages, projects,authors to language, project, author
mod.docvecs.most_similar('R');
it7:   [('F', 0.9947392344474792), ('Yannick Spill <yannick.spill@crg.eu>', 0.9929205179214478), ('AsaEE_ESP-rSource', 0.992476224899292), ('jhand <jhand@7d53e970-de11-0410-8a54-3d01b9da36cf>', 0.9924437999725342), ('2DegreesInvesting_PortCheck', 0.990721583366394), ('Clare2D <32903584+Clare2D@users.noreply.github.com>', 0.9905468225479126), ('Taylor Posey <taylor.m.posey@outlook.com>', 0.9884799718856812), ('tinaGNAW <tina@2degrees-investing.org>', 0.9880185127258301), ('Paul Fischer <fischerp@illinois.edu>', 0.987598180770874), ('12379Monty_scRNASeq', 0.9872961044311523)]
it:1-3: ('F', 0.9947392344474792), ('Yannick Spill <yannick.spill@crg.eu>', 0.9929205179214478), ('AsaEE_ESP-rSource', 0.992476224899292), ('jhand <jhand@7d53e970-de11-0410-8a54-3d01b9da36cf>', 0.9924437999725342), ('2DegreesInvesting_PortCheck', 0.990721583366394), ('Clare2D <32903584+Clare2D@users.noreply.github.com>', 0.9905468225479126), ('Taylor Posey <taylor.m.posey@outlook.com>', 0.9884799718856812), ('tinaGNAW <tina@2degrees-investing.org>', 0.9880185127258301), ('Paul Fischer <fischerp@illinois.edu>', 0.987598180770874), ('12379Monty_scRNASeq', 0.9872961044311523)]

mod.docvecs.most_similar('cran_tidyquery')
it2:[('kungeinus_Prediction_Assignment_Writeup', 0.49396124482154846), ('parserpro_db_update', 0.48060914874076843), ('adisarid <adisarid@gmail.com>', 0.47186779975891113), ('danthemango <danthemango@gmail.com>', 0.4652522802352905), ('arnarg_plex_exporter', 0.46454471349716187), ('colin-combe_CLMS-UI', 0.4438340663909912), ('alanaw1_CulturalHitchhiking', 0.4377615451812744), ('lavanyaj09_BE223A', 0.4339529275894165), ('jnarhan_Kaggle-Pneumonia', 0.43245214223861694), ('gxe778_Trajectory-Inference-Methods-applied-on-early-cell-lines-from-human-embryo', 0.42638999223709106)]
it1:[('parserpro_db_update', 0.4985978603363037), ('kungeinus_Prediction_Assignment_Writeup', 0.49396124482154846), ('adisarid <adisarid@gmail.com>', 0.47186779975891113), ('danthemango <danthemango@gmail.com>', 0.4674111604690552), ('arnarg_plex_exporter', 0.46454471349716187), ('PeterHenell_goora', 0.4478600025177002), ('colin-combe_CLMS-UI', 0.44383400678634644), ('danthemango_ClientRG', 0.4412115514278412), ('alanaw1_CulturalHitchhiking', 0.4377615451812744), ('jnarhan_Kaggle-Pneumonia', 0.43245214223861694)]			       

mod.docvecs.most_similar('Yannick Spill <yannick.spill@crg.eu>')
it2:[('R', 0.992920458316803), ('3schwartz_SpecialeScrAndFun', 0.9900994300842285), ('Francois <briatte@gmail.com>', 0.9879549145698547), ('215ALab4_lab4', 0.9871786832809448), ('12379Monty_scRNASeq', 0.986408531665802), ('12379Monty <francois.collin@gmail.com>', 0.9862282872200012), ('3wen_elus', 0.9861852526664734), ('52North_tamis', 0.9858419299125671), ('tinaGNAW <tina@2degrees-investing.org>', 0.9855629205703735), ('2DegreesInvesting_PortCheck', 0.984999418258667)]
it1:[('3DGenomes_binless', 0.9939588904380798), ('R', 0.9929205179214478), ('3schwartz_SpecialeScrAndFun', 0.9900994300842285), ('Francois <briatte@gmail.com>', 0.9879549145698547), ('215ALab4_lab4', 0.9871785640716553), ('12379Monty_scRNASeq', 0.986408531665802), ('12379Monty <francois.collin@gmail.com>', 0.986228346824646), ('3wen_elus', 0.9861852526664734), ('52North_tamis', 0.9858419299125671), ('tinaGNAW <tina@2degrees-investing.org>', 0.9855630397796631)]

#get most similar packages to a package
mod.wv.most_similar('pandas')
[('song_data.songs', 0.6327548623085022), ('context.plot.plot.plot_points.plot_points', 0.6035584211349487), ('ax.storage.sqa_store.save.save_experiment', 0.585919976234436), ('emperor', 0.5831856727600098), ('geograph.term_profile.get_term_profile', 0.5705782175064087), ('negmas.apps.scml.utils.anac2019_world', 0.5701258778572083), ('pymove.conversions', 0.5696786642074585), ('learning_curve.learning_curve', 0.5638871192932129), ('ax.Data', 0.5624120831489563), ('starutils.populations.Raghavan_BinaryPopulation', 0.5612468123435974)]
mod.wv.most_similar('numpy')
[('tigre.utilities.plotimg.plotimg', 0.6538034677505493), ('cs231n.classifiers.linear_classifier.LinearSVM', 0.6469931602478027), ('Test_data.data_loader.load_head_phantom', 0.6307787895202637), ('PsyNeuLink.Components.Projections.TransmissiveProjections.MappingProjection.MappingProjection', 0.6261978149414062), ('section3_1_heatingday', 0.6204843521118164), ('tigre.Utilities.plotproj.ppslice', 0.617354154586792), ('tigre.demos.Test_data.data_loader.load_head_phantom', 0.6111389994621277), ('hmtk.hazard.HMTKHazardCurve', 0.6096319556236267), ('pyshtools.spectralanalysis.SHBias', 0.6086275577545166), ('agentnet.learning.n_step', 0.6063860058784485)]


mod.wv.most_similar('ggplot2')
it7: [('bsearchtools', 0.8243459463119507), ('cp_common_uses.h', 0.8239778876304626), ('PTRACERS_FIELDS.h', 0.8211128115653992), ('filnames.h', 0.8199384212493896), ('jelira.h', 0.8191816210746765), ('soilsnow.h', 0.8181809186935425), ('parmhor.h', 0.8178006410598755), ('da_transform_xtoy_pilot_adj.inc', 0.8176625967025757), ('ebbyeb.blk', 0.8171699047088623), ('SCHROD', 0.8164928555488586)]
it2: [('nortest', 0.9667873382568359), ('reshape2', 0.9574425220489502), ('tidyr', 0.9566653966903687), ('partykit', 0.9542033076286316), ('purrr', 0.9534142017364502), ('knitr', 0.9527696371078491), ('aim2_parameters.h', 0.9526889324188232), ('matrixStats', 0.9519108533859253), ('FlowSOM', 0.9512166976928711), ('Obspars.com', 0.9505057334899902)]
it1: ('tidyr', 0.9896135330200195), ('reshape2', 0.9894652366638184), ('nortest', 0.9879751205444336), ('dplyr', 0.9831089973449707), ('knitr', 0.9812949895858765), ('partykit', 0.9795119762420654), ('RColorBrewer', 0.9769814610481262), ('purrr', 0.9767657518386841), ('lubridate', 0.9766140580177307), ('data.table', 0.9752072095870972)]

mod.wv.most_similar('keras_learn')
it2:[('tensorbayes.nputils.log_sum_exp', 0.7950129508972168), ('neural_network_decision_tree.nn_decision_tree', 0.774245023727417), ('TechnicalAnalysis.TechnicalAnalysis', 0.768464207649231), ('models.naive_convnet.NaiveConvColoringModel', 0.762176513671875), ('model_VAE.VAE_mnist', 0.761232316493988), ('batch_generator.dir.DirIterator', 0.7570154070854187), ('optimizer.learing_rate_scheduling', 0.7539881467819214), ('models.yolov3_gpu_head.inference.restore_model', 0.7510530948638916), ('flickr8k_parse', 0.7491798400878906), ('ppo.NNValueFunction', 0.7474846839904785)]
it1:[('tensorbayes.nputils.log_sum_exp', 0.8044129610061646), ('models.yolov3_gpu_head.inference.restore_model', 0.7988119125366211), ('model_VAE.VAE_mnist', 0.7955332398414612), ('antTrainEnv_class.antTrainEnv_class', 0.784370481967926), ('models.naive_convnet.NaiveConvColoringModel', 0.7827848196029663), ('batch_generator.dir.DirIterator', 0.7793075442314148), ('neural_network_decision_tree.nn_decision_tree', 0.7765824198722839), ('TechnicalAnalysis.TechnicalAnalysis', 0.7733845710754395), ('envs.economy.jesusfv', 0.773059606552124), ('model.audio_u_net_dnn', 0.7707473039627075)]

#no most similar language,project.author from package, need to write a function

# get a doc vector based on the set of words and find most closely related terms
mod.wv.similar_by_vector(mod.infer_vector(['ggplot2','data.table']))
[('analyticlab.LaTeX', 0.9497971534729004), ('knitr', 0.9475012421607971), ('lubridate', 0.9463158845901489), ('mafdecls.fh', 0.9458824992179871), ('tidyr', 0.9457533359527588), ('ggthemes', 0.9443897008895874), ('demos.sampling_freq_demo1', 0.9438977837562561), ('meyer.basic_constructs.MRest', 0.9436166286468506), ('errquit.fh', 0.9411042332649231), ('scam', 0.9407658576965332)]

mod.wv.most_similar('data.table')
[('tidyr', 0.9825125336647034), ('scam', 0.9800063967704773), ('knitr', 0.9797146320343018), ('lubridate', 0.9793548583984375), ('purrr', 0.976327657699585), ('espriou.h', 0.9762163758277893), ('plant.h', 0.9760875105857849), ('gksenu.h', 0.9760852456092834), ('ggthemes', 0.9758648872375488), ('g01wsl.h', 0.9758262634277344)]



#similarities among languages
for la in ('F', 'R', 'jl', 'pl', 'ipy'):
 for lb in ('F', 'R', 'jl', 'pl', 'ipy'):
   print (la+":"+lb+" "+str(mod.docvecs.distance(la,lb))
it7:
F:R 0.005260765552520752
F:jl 0.5538360178470612
F:pl 0.26047611236572266
F:ipy 0.2711484432220459
R:jl 0.5432112514972687
R:pl 0.25471168756484985
R:ipy 0.2826273441314697
jl:pl 0.4799261689186096
jl:ipy 0.7023965418338776
pl:ipy 0.3867550492286682

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



for la in F R jl ipy pl Cs Go PYml Rust Scala PY JS java rb; do zcat PtaPkgQ$la.s | perl -e 'while(<STDIN>){chop();($p,$t,$a)=split(/;/);$pre=0; $pre=1 if $t>= 1518784533+3600*24*365.25; $pn{$p}{$pre}++; $an{$a}{$pre}++;}; for my $p (keys %pn){print "p;$p;$pn{$p}{1};$pn{$p}{0}\n";} for my $a (keys %an){print "a;$a;$an{$a}{1};$an{$a}{0}\n";}' | gzip > PtaPkgQ$la.cnt; done &



for la in F jl R ipy pl Cs Go PYml Rust Scala PY PYml JS java rb; do
    zcat PtaPkgQ$la.cnt| grep ^a | awk -F\; '{if($4>10 && $3>10)print $0}' > PtaPkgQ$la.cnt10
done

for la in F jl R ipy pl Cs Go PYml Rust Scala PY PYml JS java rb; do
    zcat PtaPkgQ$la.cnt| grep ^p | awk -F\; '{if($4>100 && $3>100)print $0}' > PtaPkgQ$la.cnt100
done


for la in F jl R ipy pl Cs Go PYml Rust Scala PY PYml JS java rb; do     zcat PtaPkgQ$la.s | perl ~/lookup/mp.perl 2 /da0_data/basemaps/gz/a2AQ.s | gzip >  PtAPkgQ$la.s ; done &

for la in F R jl ipy pl Cs Go PYml Rust Scala PY JS java rb; do zcat PtAPkgQ$la.s | perl -e 'while(<STDIN>){chop();($p,$t,$a)=split(/;/);$pre=0; $pre=1 if $t>= 1518784533+3600*24*365.25; $pn{$p}{$pre}++; $an{$a}{$pre}++;}; for my $p (keys %pn){print "p;$p;$pn{$p}{1};$pn{$p}{0}\n";} for my $a (keys %an){print "a;$a;$an{$a}{1};$an{$a}{0}\n";}' | gzip > PtAPkgQ$la.cnt; done 

for la in F jl R ipy pl Cs Go  Rust Scala PY PYml JS java rb; do
    zcat PtAPkgQ$la.cnt| grep ^a | awk -F\; '{if($4>10 && $3>10)print $0}' > PtAPkgQ$la.cnt10
done

for la in F jl R ipy pl Cs Go PYml Rust Scala PY JS java rb; do
    zcat PtAPkgQ$la.cnt| grep ^p | awk -F\; '{if($4>100 && $3>100)print $0}' > PtAPkgQ$la.cntp100
done

#Figure out commit counts for authors
for la in F jl R ipy pl Cs Go PYml Rust Scala PY JS java rb
do cut -d\; -f2 PtAPkgQ$la.cnt10
done | lsort 1G -u | gzip > au10.gz

for la in F jl R ipy pl Cs Go Rust Scala PY JS java rb
do zcat PtAPkgQ$la.s | perl ~/bin/grepField.perl au10.gz 3
done | gzip > all.a10.gz

for la in F jl R ipy pl Cs Go  Rust Scala PY PYml JS java rb; do
    zcat PtAPkgQ$la.cnt| grep ^a | awk -F\; '{if($4>100 && $3>100)print $0}' > PtAPkgQ$la.cnt100
done

for la in F jl R ipy pl Cs Go PYml Rust Scala PY JS java rb
do cut -d\; -f2 PtAPkgQ$la.cnt100
done | lsort 1G -u | gzip > au100.gz

for la in C F jl R ipy pl Cs Go Rust Scala PY JS java rb
do zcat PtAPkgQ$la.s | perl ~/bin/grepField.perl au100.gz 3
done | gzip > all.a100.gz

####################################
#Prepare PR data for FSE submission: Table 4
####################################
cat PRdata_new.csv | perl ~/lookup/mp.perl 0 /da0_data/basemaps/gz/a2AQ.s > PRdata_newA.csv
cut -d\; -f1 PRdata_newA.csv | lsort 1G -u | gzip > au.prs
for la in JS C F jl R ipy pl Cs Go Rust Scala PY java rb
do zcat /da4_data/play/api/PtAPkgQ$la.s | perl ~/bin/grepField.perl au.prs 3 | gzip > PtaPkgQ$la.prs.s
done 
####################################

zcat *A*.cnt | grep ^a | awk -F\; '{print $4+$3";"$2}' | lsort 30G -t\; -rn |gzip > topA
zcat topA|awk -F\; '{if ($1>50000){print $2}}' | gzip | lsort 10G -u > topA.50K
zcat au100.gz | lsort 10G -t\; -k1,1 -u | join -t\; -v1 - <(cat topA.50K | lsort 1G -t\; -k1,1)| gzip > au100-50k.gz
zcat au10.gz | lsort 10G -t\; -k1,1 -u | join -t\; -v1 - <(cat topA.50K | lsort 1G -t\; -k1,1)| gzip > au10-50k.gz

for la in F jl R ipy pl Cs Go PYml Rust Scala PY JS java rb; do
   zcat PtAPkgQ$la.s | perl ~/bin/grepField.perl au100-50K.gz 3 | gzip >  PtAPkgQ$la.a100.s
done


#reproduce import2vec

import gzip,collections,gensim.models.doc2vec,math
from gensim.models import Doc2Vec, Word2Vec
mod = Doc2Vec.load ("doc2vec.QAJS.a100.1558784533.1")
#get most similar packages to language, project, author
mod.wv.most_similar('http')


import gzip,collections,gensim.models.doc2vec,math
from gensim.models import Doc2Vec
mod = Doc2Vec.load ("doc2vec.QAJS.a100.1558784533.5")
mod.wv.most_similar('http')
[('firebase-admin', 0.7661893367767334), ('koa-bodyparser', 0.733301043510437), ('mysql2', 0.7215847969055176), ('react-loadable', 0.6763203144073486), ('vuelidate', 0.665387749671936), ('vue-style-loader', 0.6605743765830994), ('marklar', 0.6552571654319763), ('ipfs-mdns', 0.6538937091827393), ('diap', 0.6523748636245728), ('jQuery', 0.6493134498596191)]


for f in ('doc2vec.QML.2', 'doc2vec.QFRjlipyml.1518784533.9', 'doc2vec.Qipy.9', 'doc2vec.QR.1518784533.9'):
 mod = Doc2Vec.load (f)
 mod.wv.most_similar('data.table')


mod.wv.similar_by_vector(mod.docvecs['R'])
mod.wv.similar_by_vector(mod.docvecs['R'])


zcat PtAPkgQJS.a100.s | grep ';http;' | wc -l  
15375
zcat PtAPkgQJS.a100.s | grep ';http;' | grep -v ';https;' | wc -l
12852
zcat PtAPkgQJS.a100.s | grep ';https\b' > s & 
zcat PtAPkgQJS.a100.s | grep ';http\b' > p & 

wc -l p s
   1239802     3189600 15785383911 s
   1899811     4873782 18410486749 p


grep -v ';http\b' s | wc -l
 56677
grep -v ';https\b' p | wc -l
716686


#doc2vec (binary, author+project)
f='doc2vec.PAPkgQR.a100b.9'
mod = Doc2Vec.load (f)
mod.wv.most_similar('data.table')
[('ggtree', 0.45562177896499634), ('datastorr', 0.45432671904563904), ('koRpus', 0.45120853185653687), ('emmeans', 0.45097100734710693), ('fansi', 0.44699180126190186), ('datasets', 0.442926287651062), ('ellipse', 0.4332200288772583), ('mlmRev', 0.43092772364616394), ('ddalpha', 0.4287028908729553), ('extrafont', 0.42746877670288086)]

mod.wv.most_similar('readr')
[('reshape2', 0.5896936655044556), ('rgdal', 0.5330021381378174), ('rlist', 0.5308029651641846), ('scales', 0.5262876749038696), ('slam', 0.5214910507202148), ('rstanarm', 0.5202317833900452), ('readstata13', 0.517525315284729), ('reshape', 0.5141236186027527), ('tidyverse', 0.511005163192749), ('pracma', 0.5025790929794312)]

#doc2vec (binary, author only)
f='doc2vecA01.20.1.20.3.PAPkgQR.a100b.10'
mod = Doc2Vec.load (f)
mod.wv.most_similar('data.table')
[('dplyr', 0.9619144797325134), ('devtools', 0.9410998821258545), ('stringr', 0.938940703868866), ('gridExtra', 0.9301508069038391), ('tidyverse', 0.9286336302757263), ('tidyr', 0.9281747341156006), ('ggplot2', 0.924401581287384), ('RColorBrewer', 0.9191794395446777), ('Hmisc', 0.9127570986747742), ('foreach', 0.9072998762130737)]

mod.wv.most_similar('readr')
[('tidyr', 0.9521045684814453), ('tidyverse', 0.9518229961395264), ('lubridate', 0.9460780024528503), ('ggthemes', 0.9348131418228149), ('rvest', 0.9196170568466187), ('scales', 0.912832498550415), ('RColorBrewer', 0.8977759480476379), ('gridExtra', 0.8954176902770996), ('corrplot', 0.8899544477462769), ('stringr', 0.8898525238037109)]

f='doc2vecA01.20.1.20.3.PAPkgQR.a100b.17'
mod = Doc2Vec.load (f)
mod.wv.most_similar('data.table')
[('dplyr', 0.9987048506736755), ('devtools', 0.9978946447372437), ('ggplot2', 0.9974669814109802), ('tidyr', 0.9973721504211426), ('knitr', 0.9970445036888123), ('reshape2', 0.996809720993042), ('tidyverse', 0.996529221534729), ('gridExtra', 0.9961731433868408), ('scales', 0.9960485696792603), ('plyr', 0.9955645203590393)]

mod.wv.most_similar('readr')
[('scales', 0.9980828166007996), ('tidyr', 0.9975719451904297), ('ggthemes', 0.9971051812171936), ('magrittr', 0.9970694780349731), ('lubridate', 0.9966671466827393), ('RColorBrewer', 0.9965049028396606), ('gridExtra', 0.9964038133621216), ('tidyverse', 0.9963114261627197), ('rpart', 0.9957792162895203), ('ggplot2', 0.9957766532897949)]

for f in ('doc2vec.20.30.3.PAPkgQR.a100b.19','doc2vec.20.3.3.PAPkgQR.a100b.19','doc2vec.40.30.3.PAPkgQR.a100b.19', 'doc2vec.40.3.3.PAPkgQR.a100b.19', 'doc2vec.80.30.3.PAPkgQR.a100b.19', 'doc2vec.80.3.3.PAPkgQR.a100b.19', 'doc2vec.120.30.3.PAPkgQR.a100b.19', 'doc2vec.120.3.3.PAPkgQR.a100b.19'):
 mod = Doc2Vec.load (f)
 #mod.wv.most_similar('data.table')
 mod.wv.most_similar('readr')
[('RColorBrewer', 0.9983182549476624), ('scales', 0.998315155506134), ('tidyverse', 0.9982290267944336), ('magrittr', 0.9981428980827332), ('gridExtra', 0.9980251789093018), ('reshape2', 0.9979166388511658), ('knitr', 0.9977608919143677), ('lubridate', 0.997739315032959), ('tidyr', 0.9972754120826721), ('e1071', 0.9971935749053955)]
[('knitr', 0.9893307685852051), ('parallel', 0.9887491464614868), ('data.table', 0.9852929711341858), ('DESeq2', 0.9849643707275391), ('ggplot2', 0.9848182201385498), ('devtools', 0.9846794605255127), ('tidyr', 0.9845887422561646), ('rmarkdown', 0.9837965965270996), ('readxl', 0.9831156134605408), ('tools', 0.9823206663131714)]
[('lubridate', 0.9961996078491211), ('scales', 0.9947738647460938), ('knitr', 0.9944191575050354), ('RColorBrewer', 0.994251012802124), ('magrittr', 0.9941045045852661), ('reshape2', 0.9939345121383667), ('gridExtra', 0.9935609102249146), ('tidyr', 0.9930714964866638), ('ggthemes', 0.9929649233818054), ('readxl', 0.9924424886703491)]
[('tidyr', 0.9897856116294861), ('magrittr', 0.9842157959938049), ('lubridate', 0.9824410676956177), ('tidyverse', 0.9819707870483398), ('scales', 0.9788222908973694), ('RColorBrewer', 0.976738452911377), ('knitr', 0.9757811427116394), ('dplyr', 0.973181426525116), ('reshape2', 0.9710521697998047), ('gridExtra', 0.9706953763961792)]
[('tidyverse', 0.9914058446884155), ('lubridate', 0.9820787906646729), ('ggplot2', 0.981021523475647), ('reshape2', 0.9763184189796448), ('dplyr', 0.96944260597229), ('stringr', 0.9606151580810547), ('scales', 0.9509478807449341), ('ggthemes', 0.9472478032112122), ('rmarkdown', 0.9466075897216797), ('gridExtra', 0.946486234664917)]
[('tidyr', 0.979855477809906), ('tidyverse', 0.9716714024543762), ('dplyr', 0.9697237014770508), ('lubridate', 0.9694627523422241), ('magrittr', 0.9675484895706177), ('ggplot2', 0.9638528823852539), ('knitr', 0.9620657563209534), ('scales', 0.959942638874054), ('data.table', 0.9549976587295532), ('gridExtra', 0.9542347192764282)]
[('magrittr', 0.9877380132675171), ('lubridate', 0.983036994934082), ('tidyverse', 0.9830121994018555), ('scales', 0.9816074967384338), ('knitr', 0.9798682928085327), ('devtools', 0.9794678688049316), ('RColorBrewer', 0.9773341417312622), ('jsonlite', 0.9698127508163452), ('readxl', 0.9695264101028442), ('ggthemes', 0.9690042734146118)]
[('tidyr', 0.9725443720817566), ('dplyr', 0.9542319774627686), ('data.table', 0.9513483047485352), ('magrittr', 0.9475299119949341), ('ggplot2', 0.9475277662277222), ('jsonlite', 0.9473656415939331), ('lubridate', 0.9440580606460571), ('tidyverse', 0.9436643123626709), ('stringr', 0.9377952814102173), ('devtools', 0.9372730255126953)]

for f in ('doc2vecA.20.30.3.PAPkgQR.a100b.19','doc2vecA.20.3.3.PAPkgQR.a100b.19','doc2vecA.40.30.3.PAPkgQR.a100b.19', 'doc2vecA.40.3.3.PAPkgQR.a100b.19', 'doc2vecA.80.30.3.PAPkgQR.a100b.19', 'doc2vecA.80.3.3.PAPkgQR.a100b.19', 'doc2vecA.120.30.3.PAPkgQR.a100b.19', 'doc2vecA.120.3.3.PAPkgQR.a100b.19'):
  mod = Doc2Vec.load (f)
  #mod.wv.most_similar('data.table')
  mod.wv.most_similar('readr')

[('magrittr', 0.9991831183433533), ('tidyverse', 0.9989697933197021), ('gridExtra', 0.9982430934906006), ('tidyr', 0.9982177019119263), ('scales', 0.9980565309524536), ('jsonlite', 0.9979439973831177), ('data.table', 0.9979092478752136), ('reshape2', 0.9978592395782471), ('knitr', 0.9976184368133545), ('stringi', 0.9969038367271423)]
[('tidyverse', 0.9947269558906555), ('tidyr', 0.9942873120307922), ('magrittr', 0.9933727979660034), ('lubridate', 0.9930378794670105), ('knitr', 0.9925771951675415), ('gridExtra', 0.9892634749412537), ('scales', 0.9878233671188354), ('RColorBrewer', 0.9874245524406433), ('devtools', 0.9865281581878662), ('dplyr', 0.9864441752433777)]
[('tidyr', 0.9988787174224854), ('magrittr', 0.9987964630126953), ('knitr', 0.9982733726501465), ('scales', 0.9981702566146851), ('lubridate', 0.9979217648506165), ('gridExtra', 0.9973997473716736), ('ggthemes', 0.9967532157897949), ('reshape2', 0.9961109161376953), ('rpart.plot', 0.995823860168457), ('tidyverse', 0.9956599473953247)]
[('tidyr', 0.9956086277961731), ('tidyverse', 0.9895380735397339), ('magrittr', 0.9875149726867676), ('dplyr', 0.9868128299713135), ('jsonlite', 0.9848260879516602), ('lubridate', 0.9833686947822571), ('devtools', 0.9828656315803528), ('data.table', 0.9797936677932739), ('gridExtra', 0.9791477918624878), ('ggplot2', 0.979009747505188)]
[('scales', 0.997490644454956), ('tidyverse', 0.9954097270965576), ('e1071', 0.9951643943786621), ('knitr', 0.9948922991752625), ('cluster', 0.9930667877197266), ('devtools', 0.9929601550102234), ('rpart.plot', 0.9902037382125854), ('RColorBrewer', 0.9877432584762573), ('ggfortify', 0.9876832365989685), ('DBI', 0.9824402928352356)]
[('tidyr', 0.9836821556091309), ('dplyr', 0.9753010272979736), ('magrittr', 0.9750666618347168), ('lubridate', 0.9709521532058716), ('tidyverse', 0.9692929983139038), ('ggplot2', 0.9688870906829834), ('knitr', 0.9656769037246704), ('jsonlite', 0.965003490447998), ('devtools', 0.9634120464324951), ('RColorBrewer', 0.960491955280304)]
[('magrittr', 0.991066038608551), ('dplyr', 0.9904848337173462), ('ggplot2', 0.9897637963294983), ('lubridate', 0.9878308773040771), ('knitr', 0.9876624941825867), ('data.table', 0.9873613119125366), ('reshape2', 0.9868736267089844), ('tidyverse', 0.9868265390396118), ('gridExtra', 0.9866074919700623), ('stringr', 0.9863458871841431)]
[('tidyr', 0.982008695602417), ('dplyr', 0.9632465243339539), ('magrittr', 0.9609993696212769), ('ggplot2', 0.9522542953491211), ('lubridate', 0.9505354762077332), ('tidyverse', 0.9469977021217346), ('data.table', 0.9440603256225586), ('devtools', 0.9428290128707886), ('stringr', 0.9426917433738708), ('jsonlite', 0.9396312832832

																										    #W2V

f='word2vec.20.1.3.PAPkgQR.a100b'
mod = Word2Vec.load (f)
mod.most_similar('data.table')
[('devtools', 0.9298685789108276), ('ggplot2', 0.9192495346069336), ('dplyr', 0.9015513062477112), ('reshape2', 0.8996330499649048), ('RColorBrewer', 0.8813983201980591), ('gridExtra', 0.876798689365387), ('knitr', 0.8706860542297363), ('scales', 0.8666546940803528), ('readr', 0.86326003074646), ('magrittr', 0.8588861227035522)]

mod.most_similar('readr')
[('tidyr', 0.9588572978973389), ('magrittr', 0.9268977642059326), ('dplyr', 0.9099311828613281), ('tidyverse', 0.875988781452179), ('patchwork', 0.8729138374328613), ('data.table', 0.86326003074646), ('knitr', 0.8573061227798462), ('forcats', 0.8519827723503113), ('stringi', 0.8486554622650146), ('jsonlite', 0.8443582057952881)]

####################################
# LSI for FSE submission
####################################
python3 fitXtl.py PAPkgQR.a100.s3
records:20700
data.table;1.0
hierinf;0.96815044
MixtureInf;0.9635873
data.cube;0.9635765
macrobenchmark;0.9634209
RcppAPT;0.96161354
sykdomspulscompartmentalinfluenza;0.9587247
JFuncs;0.9568475
antaresWeeklyMargin;0.95673203

readr;0.99999994
stuko;0.9718676
imrParsers;0.97154045
ctsmr;0.9688772
targetscan.Hs.eg.db;0.9636713
wyntonquery;0.9597609
scdhlm;0.9505869
farms;0.9332409
sde;0.9197796

python3 fitXl.py PAPkgQR.a100.s3
records:20700
data.table;0.9999999
macrobenchmark;0.9698489
data.cube;0.96984845
MixtureInf;0.96984583
RcppAPT;0.9698335
hierinf;0.96799344
antaresWeeklyMargin;0.96222705
antaresRead;0.9620175
spatialdatatable;0.9574404

readr;1.0
ctsmr;0.9638993
imrParsers;0.96388185
stuko;0.9625468
wyntonquery;0.9605134
targetscan.Hs.eg.db;0.95924646
scdhlm;0.92792475
stlcsb;0.9269278
NameNeedle;0.92489654
####################################


#JS
f='doc2vecA.30.100.3.PAPkgQJS.0.b.1'

mod = Doc2Vec.load (f)
mod.wv.most_similar('http')

import gzip,collections,gensim.models.doc2vec,math
from gensim.models import Doc2Vec, Word2Vec

python3 fitXw.py PAPkgQR.s1 1 100 50 20 100 200

mod = Word2Vec(docs,sg=dm,size=vs, window=ws, negative=ns, min_count=mc, workers=cores,iter=iter)
mod.save("word2vec."+str(dm)+"."+str(vs)+"."+str(ws)+"."+str(ns)+"."+str(mc)+"."+str(iter)+"."+lst)
f='word2vec.100.50.20.100.200.0.PAPkgQR.s1' #garbage sg=0
f='word2vec.100.50.20.100.200.1.PAPkgQR.s1' #decent sg=1
>>> mod.most_similar('data.table')
[('dplyr', 0.8867905139923096), ('stringr', 0.8839394450187683), ('plyr', 0.8830145001411438), ('magrittr', 0.8689486384391785), ('magclass', 0.8649991154670715), ('readr', 0.8637726902961731), ('tidyr', 0.8611730337142944), ('lubridate', 0.8585350513458252), ('gWidgetsWWW2', 0.8545087575912476), ('lucode', 0.852830171585083)]
>>> mod.most_similar('readr')
[('magrittr', 0.9195265769958496), ('tidyr', 0.8952105641365051), ('dplyr', 0.8914402723312378), ('ggplot2', 0.8837970495223999), ('stringr', 0.8687031865119934), ('data.table', 0.8637727499008179), ('plotly', 0.8532767295837402), ('magclass', 0.852198600769043), ('lucode', 0.8505121469497681), ('plyr', 0.841021716594696)]


f='word2vec.1.100.50.20.100.200.PAPkgQR.s1' # OK
mod = Word2Vec.load (f)
mod.most_similar('data.table')
mod.most_similar('readr')
[('dplyr', 0.838019609451294), ('stringr', 0.7983173131942749), ('plyr', 0.7869787216186523), ('ggplot2', 0.7796253561973572), ('magrittr', 0.7705162167549133), ('reshape2', 0.7650773525238037), ('tidyr', 0.7562291622161865), ('readr', 0.7183531522750854), ('scales', 0.7113677263259888), ('tidyverse', 0.7113019227981567)]
>>> mod.most_similar('readr')
[('dplyr', 0.8551141023635864), ('magrittr', 0.8088136911392212), ('tidyr', 0.8009055852890015), ('stringr', 0.7933487892150879), ('ggplot2', 0.7563961148262024), ('data.table', 0.7183531522750854), ('tidyverse', 0.7072793841362), ('readxl', 0.6874278783798218), ('plotly', 0.6379566192626953), ('scales', 0.6288162469863892)]
f='word2vec.0.100.50.20.100.200.PAPkgQR.s1'
mod = Word2Vec.load (f)
mod.most_similar('data.table')  # OK
mod.most_similar('readr')
[('dplyr', 0.6934608221054077), ('stringr', 0.6257824301719666), ('plyr', 0.6186755895614624), ('readr', 0.5545837879180908), ('tidyr', 0.5514156818389893), ('ggplot2', 0.5356006622314453), ('reshape2', 0.5290185809135437), ('lubridate', 0.5161994695663452), ('scales', 0.5158300399780273), ('magrittr', 0.4615442454814911)]
>>> mod.most_similar('readr')
[('dplyr', 0.6838119626045227), ('stringr', 0.6082045435905457), ('tidyverse', 0.5811571478843689), ('tidyr', 0.5662992596626282), ('data.table', 0.5545837879180908), ('lubridate', 0.531836748123169), ('magrittr', 0.5190625190734863), ('ggplot2', 0.4508250951766968), ('readxl', 0.42567265033721924), ('forcats', 0.3814961314201355)]




f='word2vec.20.1.3.PAPkgQJS.0.b'
mod.most_similar('http')
[('color-namer', 0.8891410827636719), ('easyyoutubedownload', 0.8820397257804871), ('socketio', 0.8606370091438293), ('https', 0.858386754989624), ('ffmetadata', 0.8539266586303711), ('tress', 0.8495419025421143), ('lwip', 0.8478525280952454), ('data-utils', 0.847440242767334), ('render', 0.8417345285415649), ('sharedb-mingo-memory', 0.836542546749115)]
mod.most_similar('https')
[('http', 0.858386754989624), ('google-search-scraper', 0.8478592038154602), ('restc', 0.8463708758354187), ('lwip', 0.8457597494125366), ('sanitize', 0.8455460071563721), ('skipper', 0.842998206615448), ('dom-parser', 0.842816948890686), ('mongoose-auto-increment', 0.8401623368263245), ('guid', 0.8306612968444824), ('easyyoutubedownload', 0.8301177024841309)]



#### Compare various LSI methods
#all tl - tfidf + lsi
python3 fitXtl.py PAPkgQ.all1.a100.0.s2
data.table;1.0
lubridate;0.9246081
magrittr;0.92408097
glmnet;0.90353286
dplyr;0.8974145
reshape;0.86968565
tibble;0.8680916
shinydashboard;0.8583525
shinythemes;0.85728675

readr;1.0
synapser;0.915734
RJSONIO;0.9078114
stringr;0.9041679
dplyr;0.8992814
tidyr;0.8970244
gridGraphics;0.8929982
tibble;0.8897572
magrittr;0.8888842

https;1.0000001
facebook-chat-api;0.9842278
shrink-ray;0.9786956
@sanity/mutator;0.9766238
groq;0.9766238
mead;0.9766238
sse-channel;0.9766238
epoll;0.976463
promise-mysql;0.97581285


python3 fitXl.py PAPkgQ.all1.a100.0.s2
data.table;1.0
yum;0.9950353
string.split;0.9948384
report;0.9942483
dbus.mainloop.glib.DBusGMainLoop;0.9939707
Reporter;0.99381834
ScanView;0.99381834
startfile;0.99381834
webbrowser._iscommand;0.99381834

readr;0.99999994
cfa/chrony_conf;0.9650319
yast2/target_file" # required to cfa work on changed scr;0.9650319
all;0.96063083
y2ntp_client/dialog/add_pool;0.9601435
deep_merge/core;0.95998293
yast/logger;0.9599484
cwm/dialog;0.95994604
cwm/popup;0.95994604

https;1.0000001
@sanity/mutator;0.94792485
groq;0.94792485
mead;0.94792485
sse-channel;0.94792485
pretty-log;0.9450143
@luminati-io/socksv5;0.9416811
hutil;0.9416811


python3 fitXtl.py PAPkgQ.all1.a100.s2
records:1614998

data.table;1.0
DESeq2;0.989663
RColorBrewer;0.9851666
reshape2;0.98426837
gplots;0.9818575
cowplot;0.9815432
ggplot2;0.9800583
lme4;0.9800026
ggrepel;0.9798994

readr;1.0
dplyr;0.9935278
GGally;0.99229115
magrittr;0.9897382
dendextend;0.9892954
pROC;0.9880314
pandas ;0.98731506
rdkit.Chem.PandasTools;0.986886
allensdk.core.cell_types_cache.CellTypesCache;0.98675674

https;0.99999994
express-fileupload;0.9932468
express-promise-router;0.9930251
mongodb;0.9930065
express-validator;0.9927011
passport-local-mongoose;0.9923985
body-parse;0.9923428
monk;0.992233
multer-s3;0.99221814

####################################
#Market basket:
####################################
zcat PAPkgQ.all1.a100.0.s2 | cut -d\; -f3- > tr

R --no-save
library(arules);
tr = read.transactions("tr",sep=";",quote="");
summary(tr);
itemFrequencyPlot(tr, topN=15);

res = apriori(tr, parameter = list(support=0.006, confidence = 0.25, minlen=2,maxtime=200));
summary(res);
set of 2399140 rules

rule length distribution (lhs + rhs):sizes
      2 
2399140 

   Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
      2       2       2       2       2       2 

summary of quality measures:
    support           confidence          lift              count     
 Min.   :0.006015   Min.   :0.2500   Min.   :  0.9964   Min.   : 304  
 1st Qu.:0.009576   1st Qu.:0.4796   1st Qu.:  9.8844   1st Qu.: 484  
 Median :0.016778   Median :0.7383   Median : 11.6512   Median : 848  
 Mean   :0.023824   Mean   :0.6956   Mean   : 22.7509   Mean   :1204  
 3rd Qu.:0.029994   3rd Qu.:0.9292   3rd Qu.: 20.4901   3rd Qu.:1516  
 Max.   :0.162001   Max.   :1.0000   Max.   :161.9968   Max.   :8188  

mining info:
 data ntransactions support confidence
   tr         50543   0.006       0.25


inspect(sort(res, by="lift")[1:7])
    lhs                                           rhs                                              support confidence     lift count
[1] {mdast-util-compact}                       => {is-alphanumeric}                            0.006172962          1 161.9968   312
[2] {is-alphanumeric}                          => {mdast-util-compact}                         0.006172962          1 161.9968   312
[3] {setuptools.extern.six.moves.urllib.parse} => {setuptools.extern.six.moves.html_parser}    0.006212532          1 160.9650   314
[4] {setuptools.extern.six.moves.html_parser}  => {setuptools.extern.six.moves.urllib.parse}   0.006212532          1 160.9650   314
[5] {setuptools.extern.six.moves.urllib.parse} => {setuptools.extern.pyparsing.ZeroOrMore}     0.006212532          1 160.9650   314
[6] {setuptools.extern.pyparsing.ZeroOrMore}   => {setuptools.extern.six.moves.urllib.parse}   0.006212532          1 160.9650   314
[7] {setuptools.extern.six.moves.urllib.parse} => {setuptools.extern.pyparsing.ParseException} 0.006212532          1 160.9650   314


res85 = apriori(tr, parameter = list(support=0.006, confidence = 0.85, minlen=2,maxtime=200))
inspect(sort(res85, by="lift")[1:7])
    lhs                           rhs                                support confidence     lift count
[1] {functools.update_wrapper,                                                                        
     memcache}                 => {pylibmc}                      0.006014681  1.0000000 165.1732   304
[2] {memcache,                                                                                        
     pprint.pformat}           => {pylibmc}                      0.006014681  0.9967213 164.6317   304
[3] {dummy_threading,                                                                                 
     memcache}                 => {pylibmc}                      0.006014681  0.9934641 164.0936   304
[4] {code,                                                                                            
     memcache}                 => {pylibmc}                      0.006014681  0.9902280 163.5591   304
[5] {decimal,                                                                                         
     memcache}                 => {pylibmc}                      0.006014681  0.9902280 163.5591   304
[6] {datetime.date,                                                                                   
     memcache}                 => {pylibmc}                      0.006014681  0.9902280 163.5591   304
[7] {sw-toolbox,                                                                                      
     trim-newlines}            => {serviceworker-cache-polyfill} 0.006074036  1.0000000 163.0419   307



res850.1 = apriori(tr, parameter = list(support=0.1, confidence = 0.85, minlen=2,maxlen=5,maxtime=200))
inspect(sort(res850.1, by="lift")[1:7])
    lhs                                     rhs                    support  
[1] {ajv,fast-deep-equal}                => {json-schema-traverse} 0.1002711
[2] {json-schema-traverse}               => {fast-deep-equal}      0.1003700
[3] {fast-deep-equal}                    => {json-schema-traverse} 0.1003700
[4] {ajv,json-schema-traverse}           => {fast-deep-equal}      0.1002711
[5] {is-buffer,kind-of,source-map}       => {repeat-string}        0.1000930
[6] {is-buffer,repeat-string,source-map} => {kind-of}              0.1000930
[7] {is-buffer,kind-of}                  => {repeat-string}        0.1006865
    confidence lift     count
[1] 1.0000000  9.961175 5068 
[2] 0.9998029  9.943534 5073 
[3] 0.9982290  9.943534 5073 
[4] 0.9998027  9.943532 5068 
[5] 0.9964546  9.861720 5059 
[6] 0.9996048  9.854306 5059 
[7] 0.9953061  9.850354 5089 


res850.05 = apriori(tr, parameter = list(support=0.05, confidence = 0.85, minlen=2,maxlen=3,maxtime=200))
inspect(sort(res850.05, by="lift")[1:7])
    lhs                             rhs                support    confidence
[1] {homedir-polyfill,isexe}     => {parse-passwd}     0.05037295 1.0000000 
[2] {homedir-polyfill,semver}    => {parse-passwd}     0.05047188 1.0000000 
[3] {parse-passwd}               => {homedir-polyfill} 0.05051145 1.0000000 
[4] {homedir-polyfill}           => {parse-passwd}     0.05051145 0.9996085 
[5] {is-glob,parse-passwd}       => {homedir-polyfill} 0.05009596 1.0000000 
[6] {is-extglob,parse-passwd}    => {homedir-polyfill} 0.05013553 1.0000000 
[7] {is-extendable,parse-passwd} => {homedir-polyfill} 0.05017510 1.0000000 
    lift     count
[1] 19.79749 2546 
[2] 19.79749 2551 
[3] 19.78974 2553 
[4] 19.78974 2553 
[5] 19.78974 2532 
[6] 19.78974 2534 
[7] 19.78974 2536 

####################################
####################################
####################################
####################################
####################################

####################################
# The results for FSE evaluation
####################################
#do evaluation: get diffs, new apis, projects, authors
a100 - authors that had between 100 and 25K blobs changed

for i in {0..31}; do zcat PAPkgQ.all2.a100.$i.s[24] | perl -e 'while(<STDIN>){chop(); ($p,$la,$a,@ms)=split(/;/);for $m (@ms){$m =~ s/^\s+//; $m =~ s/\s+$//; $k{"$p;$la"}{$m}++}};while (($p, $v)=each %k){@ms=sort keys %{$v}; print "$p;".(join ";", @ms)."\n";}' | gzip > PAPkgQ.all2.a100.$i.sPA; done &
for i in {0..31}; do perl cmp.perl $i | gzip > PAPkgQ.all2.a100.$i.sPD; done


#do author api prediction

for i in {0..31}; do zcat PAPkgQ.all2.a100.$i.s2; done | perl -e 'while(<STDIN>){chop(); ($p,$la,$a,@ms)=split(/;/);for $m (@ms){$m =~ s/^\s+//; $m =~ s/\s+$//; $k{"$a;$la"}{$m}++}};while (($p, $v)=each %k){@ms=sort keys %{$v}; print "$p;".(join ";", @ms)."\n";}' | gzip > PAPkgQ.all2.a100.$i.sA2 
for i in {0..31}; do zcat PAPkgQ.all2.a100.$i.s[24]; done | perl -e 'while(<STDIN>){chop(); ($p,$la,$a,@ms)=split(/;/);for $m (@ms){$m =~ s/^\s+//; $m =~ s/\s+$//; $k{"$a;$la"}{$m}++}};while (($p, $v)=each %k){@ms=sort keys %{$v}; print "$p;".(join ";", @ms)."\n";}' | gzip > PAPkgQ.all2.a100.$i.sAA 
perl cmpA.perl | gzip > PAPkgQ.all2.a100.sAD

zcat APPkgQ.all2.a100.s2.0.gz | cut -d\; -f1 | gzip > APPkgQall2.a100.0.s2.a
zcat PAPkgQ.all2.a100.sAD | perl ~/lookup/grepField.perl APPkgQ.all2.a100.0.s2.a 2 | gzip > APPkgQ.all2.a100.sAD.0


###########
# it looks as if the doc2vec overfits after the first iteration
###########
for (i in 0:17){
x = read.table(paste("out.",i,".1",sep=""),sep=";",quote="",comment.char="");
a = tapply(x$V5, list(x$V1, x$V3), mean,na.rm=T);
print (c(i, mean(a[,1],na.rm=T),t.test(a[,1]-a[,2],na.rm=T)$conf.int[1:2], t.test(a[,2]-a[,3],na.rm=T)$conf.int[1:2]));
}
[1] 0.00000000 0.50927680 0.09067491 0.10093593 0.18090885 0.19335262
[1] 1.00000000 0.44869976 0.08790081 0.09649559 0.08724058 0.09609265
[1] 2.00000000 0.42212822 0.08993144 0.09809864 0.06047655 0.06825218
[1] 3.00000000 0.40641982 0.09011352 0.09808722 0.04716476 0.05433207
[1] 4.00000000 0.39646346 0.09043100 0.09829863 0.03969543 0.04650274
[1] 5.00000000 0.39025141 0.09065208 0.09841595 0.03526486 0.04181648
[1] 6.00000000 0.38500585 0.09046420 0.09818283 0.03178368 0.03820169
[1] 7.00000000 0.38154876 0.09033579 0.09799816 0.02943122 0.03567698
[1] 8.00000000 0.37869745 0.09019635 0.09783633 0.02834118 0.03452642
[1] 9.00000000 0.37661727 0.09015368 0.09777063 0.02703326 0.03317761
[1] 10.000000  0.37553708 0.09035486 0.09793701 0.02664744 0.03276708
[1] 11.000000  0.37455881 0.09032891 0.09790877 0.02617130 0.03224815
[1] 12.000000  0.37403331 0.09060562 0.09817693 0.02633169 0.03245005
[1] 13.000000  0.49103977 0.06520619 0.07053818 0.06103873 0.06537175
[1] 14.000000  0.52866034 0.04765265 0.05206281 0.04579535 0.04934915
[1] 15.000000  0.54368915 0.04360702 0.04800054 0.00185246 0.005180105
[1] 16.000000  0.54855054 0.04256544 0.0469666 -0.02046332 -0.01718081
[1] 17.000000  0.54995806 0.04208497 0.0464946 -0.02926308 -0.02602597

for (i in 0:14){
x = read.table(paste("out.",i,".0",sep=""),sep=";",quote="",comment.char="");
a = tapply(x$V5, list(x$V1, x$V3), mean);
print (c(i, mean(a[,1]),t.test(a[,1]-a[,2])$conf.int[1:2], t.test(a[,2]-a[,3])$conf.int[1:2]));
}
[1] 0.00000000 0.50449317 0.08559275 0.09438839 0.18037605 0.19138137
[1] 1.00000000 0.44622477 0.08535391 0.09280849 0.08754807 0.09538888
[1] 2.00000000 0.41799572 0.08583718 0.09302845 0.06024928 0.06713268
[1] 3.00000000 0.40280881 0.08656080 0.09359742 0.04696934 0.05334605
[1] 4.00000000 0.39329022 0.08686626 0.09380557 0.03994251 0.04601570
[1] 5.00000000 0.38565571 0.08581838 0.09273360 0.03513862 0.04099686
[1] 6.00000000 0.38082778 0.08592939 0.09279577 0.03214265 0.03786785
[1] 7.00000000 0.37717005 0.08587037 0.09270433 0.03005095 0.03572047
[1] 8.00000000 0.37450212 0.08571395 0.09252385 0.02864018 0.03423129
[1] 9.00000000 0.37289926 0.08587650 0.09266644 0.02735387 0.03286991
[1] 10.000000  0.37146300 0.08569699 0.09247655 0.02718105 0.03267695
[1] 11.000000  0.37067109 0.08585872 0.09263314 0.02689559 0.03244588
[1] 12.000000  0.37011623 0.08606692 0.09284306 0.02672185 0.03223387
[1] 13.000000  0.48766617 0.06082190 0.06551233 0.06231336 0.06603381
[1] 14.000000  0.52486957 0.04465219 0.04876194 0.04607511 0.04923602
[1] 15.000000  0.53952873 0.04020725 0.04428654 0.00192170 0.005003169

for (i in 0:17){
x = read.table(paste("out.",i,".3",sep=""),sep=";",quote="",comment.char="");
a = tapply(x$V5, list(x$V1, x$V3), mean);
print (c(i, mean(a[,1]),t.test(a[,1]-a[,2])$conf.int[1:2], t.test(a[,2]-a[,3])$conf.int[1:2]));
}
[1] 0.00000000 0.51228422 0.08423109 0.09421903 0.18961387 0.20200018
[1] 1.00000000 0.45414432 0.08676788 0.09509846 0.09162590 0.10043595
[1] 2.00000000 0.42591008 0.08840818 0.09638070 0.06372909 0.07150438
[1] 3.00000000 0.41047527 0.08921806 0.09701840 0.04994462 0.05715492
[1] 4.00000000 0.40040532 0.08941896 0.09710931 0.04211002 0.04891717
[1] 5.00000000 0.39285663 0.08880580 0.09644760 0.03752624 0.04411451
[1] 6.00000000 0.38795940 0.08885461 0.09644423 0.03407689 0.04049291
[1] 7.00000000 0.38439066 0.08893528 0.09646439 0.03221137 0.03851050
[1] 8.00000000 0.38162160 0.08886911 0.09637950 0.03080327 0.03703007
[1] 9.00000000 0.37985002 0.08880199 0.09627510 0.02974078 0.03586236
[1] 10.000000  0.37892498 0.08915898 0.09660508 0.02977044 0.03595569
[1] 11.000000  0.37798348 0.08910305 0.09654585 0.02966858 0.03579533
[1] 12.000000  0.37741404 0.08938410 0.09682284 0.02925239 0.03542766
[1] 13.000000  0.49398193 0.06519341 0.07050073 0.06147903 0.06576454
[1] 14.000000  0.53190766 0.04968853 0.05420056 0.04470187 0.04835752
[1] 15.0000000 0.54696112 0.04496077 0.0494012 -0.0003606  0.0030147588
[1] 16.000000  0.55164709 0.04354439 0.0479578 -0.0215186 -0.01821583
[1] 17.000000  0.55291163 0.04290569 0.0472929 -0.0290859 -0.02576941
[1] 18.00000000  0.55315557  0.04267271  0.04704862 -0.03195601 -0.02864087

for (i in 0:12){
x = read.table(paste("out.",i,".2",sep=""),sep=";",quote="",comment.char="");
a = tapply(x$V5, list(x$V1, x$V3), mean,na.rm=T);
print (c(i, mean(a[,1],na.rm=T),t.test(a[,1]-a[,2],na.rm=T)$conf.int[1:2], t.test(a[,2]-a[,3],na.rm=T)$conf.int[1:2]));
}
[1] 0.00000000 0.49785506 0.07669858 0.08579238 0.18172363 0.19316974
[1] 1.00000000 0.44261004 0.07811053 0.08591582 0.09038415 0.09859895
[1] 2.00000000 0.41638081 0.08035304 0.08781969 0.06356984 0.07085234
[1] 3.00000000 0.40178666 0.08130004 0.08858155 0.05072390 0.05746753
[1] 4.00000000 0.39176749 0.08122574 0.08844958 0.04328591 0.04975070
[1] 5.00000000 0.38515961 0.08153330 0.08867910 0.03890856 0.04517659
[1] 6.00000000 0.38081594 0.08198846 0.08907676 0.03558160 0.04174849
[1] 7.00000000 0.37792258 0.08253919 0.08957826 0.03355376 0.03962063
[1] 8.00000000 0.37481701 0.08189974 0.08893458 0.03209322 0.03801004
[1] 9.00000000 0.37295145 0.08193768 0.08894917 0.03135597 0.03722180
[1] 10.000000  0.37174513 0.08208734 0.08907893 0.03071343 0.03660824
[1] 11.000000  0.37082015 0.08216424 0.08914824 0.03024330 0.03610591
[1] 12.000000  0.37022541 0.08241546 0.08939464 0.03013885 0.03599387

for (i in 0:13){
x = read.table(paste("out.",i,".4",sep=""),sep=";",quote="",comment.char="");
a = tapply(x$V5, list(x$V1, x$V3), mean,na.rm=T);
print (c(i, mean(a[,1],na.rm=T),t.test(a[,1]-a[,2],na.rm=T)$conf.int[1:2], t.test(a[,2]-a[,3],na.rm=T)$conf.int[1:2]));
}
[1] 0.00000000 0.50201980 0.07987561 0.08916248 0.18060162 0.19158181
[1] 1.00000000 0.44391281 0.08300614 0.09081614 0.08638836 0.09425660
[1] 2.00000000 0.41660976 0.08470465 0.09220151 0.06000864 0.06696254
[1] 3.00000000 0.40158190 0.08521869 0.09254401 0.04658003 0.05299151
[1] 4.00000000 0.39212684 0.08571876 0.09294464 0.03974222 0.04593746
[1] 5.00000000 0.38514603 0.08543596 0.09260558 0.03490646 0.04089176
[1] 6.00000000 0.38027639 0.08524057 0.09237278 0.03206088 0.03785095
[1] 7.00000000 0.37701049 0.08545423 0.09254990 0.03061003 0.03634893
[1] 8.00000000 0.37440156 0.08519508 0.09225728 0.02834649 0.03394094
[1] 9.00000000 0.37249239 0.08519084 0.09222946 0.02764028 0.03322387
[1] 10.000000  0.37108960 0.08493468 0.09197349 0.02752469 0.03306693
[1] 11.000000  0.37028994 0.08498972 0.09201713 0.02748844 0.03302571
[1] 12.000000  0.36986789 0.08519832 0.09221641 0.02678472 0.03228770
[1] 13.000000  0.48675934 0.06097694 0.06585658 0.06026705 0.06401915

for (i in 0:14){
x = read.table(paste("out.",i,".5",sep=""),sep=";",quote="",comment.char="");
a = tapply(x$V5, list(x$V1, x$V3), mean,na.rm=T);
print (c(i, mean(a[,1],na.rm=T),t.test(a[,1]-a[,2],na.rm=T)$conf.int[1:2], t.test(a[,2]-a[,3],na.rm=T)$conf.int[1:2]));
}
[1] 0.00000000 0.50267151 0.08600955 0.09632277 0.17849202 0.19114564
[1] 1.00000000 0.44218779 0.08239234 0.09104467 0.08791098 0.09704787
[1] 2.00000000 0.41486685 0.08387749 0.09206987 0.06065273 0.06858225
[1] 3.00000000 0.39991191 0.08414283 0.09208297 0.04886143 0.05630194
[1] 4.00000000 0.38959406 0.08372344 0.09153724 0.04083363 0.04786534
[1] 5.00000000 0.38285499 0.08351071 0.09122461 0.03697885 0.04380045
[1] 6.00000000 0.37839685 0.08361938 0.09124596 0.03406560 0.04067783
[1] 7.00000000 0.37465376 0.08337533 0.09096221 0.03179169 0.03826294
[1] 8.00000000 0.37260513 0.08378574 0.09132284 0.03066911 0.03704302
[1] 9.00000000 0.37041890 0.08355600 0.09106802 0.02981507 0.03618065
[1] 10.000000  0.36934699 0.08388878 0.09137486 0.02955678 0.03593405
[1] 11.000000  0.36870231 0.08412754 0.09160140 0.02956867 0.03589636
[1] 12.000000  0.36819066 0.08449028 0.09195601 0.02890331 0.03517397
[1] 13.000000  0.48408286 0.06204632 0.06726454 0.06454006 0.06901382
[1] 14.000000  0.51933219 0.04638277 0.05088548 0.05114478 0.05489391

for (i in 0:15){
x = read.table(paste("out.",i,".6",sep=""),sep=";",quote="",comment.char="");
a = tapply(x$V5, list(x$V1, x$V3), mean,na.rm=T);
print (c(i, mean(a[,1],na.rm=T),t.test(a[,1]-a[,2],na.rm=T)$conf.int[1:2], t.test(a[,2]-a[,3],na.rm=T)$conf.int[1:2]));
}
[1] 0.00000000 0.50194590 0.07802904 0.08722640 0.18307867 0.19440272
[1] 1.00000000 0.44191769 0.07892995 0.08676495 0.08899817 0.09697803
[1] 2.00000000 0.41517657 0.08203017 0.08954691 0.06153415 0.06861204
[1] 3.00000000 0.39924938 0.08213974 0.08957466 0.04791676 0.05440171
[1] 4.00000000 0.39020489 0.08310714 0.09043139 0.04041467 0.04652071
[1] 5.00000000 0.38355582 0.08362871 0.09086342 0.03575334 0.04174329
[1] 6.00000000 0.37902522 0.08404463 0.09122902 0.03312081 0.03896850
[1] 7.00000000 0.37519196 0.08374163 0.09090996 0.03071490 0.03646530
[1] 8.00000000 0.37258031 0.08369725 0.09084405 0.02923016 0.03489168
[1] 9.00000000 0.37086885 0.08370747 0.09083969 0.02901673 0.03465719
[1] 10.000000  0.3696916  0.08377138 0.09089033 0.02787652 0.03347392
[1] 11.000000  0.3689507  0.08393354 0.09104211 0.02782000 0.03341200
[1] 12.000000  0.3681606  0.08404009 0.09115488 0.02761429 0.03315016
[1] 13.000000  0.4852373  0.06116680 0.06610586 0.06109427 0.06482499
[1] 14.000000  0.5228279  0.04592885 0.05020548 0.04461072 0.04784755
[1] 15.000000  0.5373017  0.04143105 0.04569960 0.00103443 0.004078491

for (i in 0:15){
x = read.table(paste("out.",i,".7",sep=""),sep=";",quote="",comment.char="");
a = tapply(x$V5, list(x$V1, x$V3), mean,na.rm=T);
print (c(i, mean(a[,1],na.rm=T),t.test(a[,1]-a[,2],na.rm=T)$conf.int[1:2], t.test(a[,2]-a[,3],na.rm=T)$conf.int[1:2]));
}
[1] 0.00000000 0.50396787 0.07947480 0.08936423 0.18162355 0.19398814
[1] 1.00000000 0.44450430 0.07831268 0.08675250 0.09060640 0.09934813
[1] 2.00000000 0.41814557 0.08194505 0.08995940 0.06255808 0.07034907
[1] 3.00000000 0.40236771 0.08241262 0.09026814 0.04968048 0.05684223
[1] 4.00000000 0.39252629 0.08289184 0.09064631 0.04213635 0.04892226
[1] 5.00000000 0.38564438 0.08284525 0.09052531 0.03695521 0.04354209
[1] 6.00000000 0.38066470 0.08303947 0.09068435 0.03391318 0.04034644
[1] 7.00000000 0.37736891 0.08346400 0.09106224 0.03174895 0.03805528
[1] 8.00000000 0.37510235 0.08374403 0.09130274 0.03073049 0.03696002
[1] 9.00000000 0.37266407 0.08352685 0.09108348 0.02966367 0.03585700
[1] 10.000000  0.3716950  0.08391375 0.09144281 0.02928881 0.03547189
[1] 11.000000  0.3704244  0.08384022 0.09137753 0.02868524 0.03480612
[1] 12.000000  0.3697260  0.08412862 0.09166525 0.02847076 0.03464066
[1] 13.000000  0.4842518  0.06187893 0.06720188 0.06334633 0.06755593
[1] 14.000000  0.5215503  0.04675143 0.05130571 0.05097473 0.05456828
[1] 15.000000  0.5357240  0.04265098 0.04710515 0.00803823 0.011404840

#lets look at the performance by language
i=7
x = read.table("out.gz",sep=";",quote="",comment.char="");
a = tapply(x$V5, list(x$V1, x$V3, x$V2), mean,na.rm=T);
print (apply(a, c(2,3),mean,na.rm=T));
print (apply(a, c(2,3),mean,na.rm=T));
          C        Cs         F        Go        JS        PY      PYml
0 0.3571373 0.5189378 0.2278115 0.3515255 0.3386374 0.3755978 0.1295043
1 0.3874602 0.5061611 0.4108476 0.4010556 0.3477729 0.3159948 0.3361127
2 0.2280350 0.2268121 0.2768938 0.2245667 0.2239590 0.2281462 0.2301409
          R      Rust     Scala       ipy      java        jl        pl
0 0.4039998 0.2178510 0.3608376 0.3047590 0.3847986 0.3876581 0.2629120
1 0.4546809 0.2959062 0.3415147 0.3921432 0.3510327 0.4497684 0.3492407
2 0.2343047 0.2165532 0.2121240 0.2311565 0.2275455 0.1987026 0.2134258
         rb
0 0.3405758
1 0.3314413
2 0.2217781

###################################################################
# lets do project api prediction: what apis project will introduce?
for (i in 0:3){
x = read.table(paste("outP.",i,".0",sep=""),sep=";",quote="",comment.char="");
a = tapply(x$V5, list(x$V1, x$V3), mean,na.rm=T);
print (c(i, mean(a[,1],na.rm=T),t.test(a[,1]-a[,2],na.rm=T)$conf.int[1:2], t.test(a[,2]-a[,3],na.rm=T)$conf.int[1:2]));
}
[1] 0.00000000 0.50220484 0.03750815 0.04034000 0.06212110 0.06565460
[1] 1.00000000 0.42417264 0.03249091 0.03481417 0.03105006 0.03386180
[1] 3.00000000 0.37842667 0.03051798 0.03257089 0.01372966 0.01610825
i=0
x = read.table(paste("outP.",i,".0",sep=""),sep=";",quote="",comment.char="");
a = tapply(x$V5, list(x$V1, x$V3), mean,na.rm=T);
print (c(i, mean(a[,1],na.rm=T),t.test(a[,1]-a[,2],na.rm=T)$conf.int[1:2], t.test(a[,2]-a[,3],na.rm=T)$conf.int[1:2]));
[1] 0.00000000 0.50220484 0.03750815 0.04034000 0.06212110 0.06565460

a = tapply(x$V5, list(x$V1, x$V3, x$V2), mean,na.rm=T);
print (apply(a, c(2,3),mean,na.rm=T));
          C        Cs         F        Go        JS        PY      PYml
0 0.4038616 0.5008812 0.3221772 0.4173387 0.4109708 0.3785951 0.2043879
1 0.4206692 0.5185743 0.4671599 0.4672491 0.4405812 0.3692081 0.3704627
2 0.3976459 0.4082502 0.3635964 0.3969003 0.3989239 0.3974906 0.3968144
          R      Rust     Scala       ipy      java        jl        pl
0 0.4202732 0.3414117 0.3948754 0.3510758 0.3808703 0.4874737 0.3557524
1 0.4765751 0.4091097 0.4168996 0.4032985 0.3810110 0.5234631 0.4483392
2 0.3942742 0.3935141 0.3863154 0.3974065 0.3947928 0.3813477 0.3979783
         rb
0 0.4213470
1 0.4464474
2 0.3960902

#does having more dimensions help (no diff)
outP300.0.0.gz
i=0
x = read.table(paste("outP300.",i,".0.gz",sep=""),sep=";",quote="",comment.char="");
a = tapply(x$V5, list(x$V1, x$V3), mean,na.rm=T);
print (c(i, mean(a[,1],na.rm=T),t.test(a[,1]-a[,2],na.rm=T)$conf.int[1:2], t.test(a[,2]-a[,3],na.rm=T)$conf.int[1:2]));
[1] 0.00000000 0.50253115 0.03709295 0.03992245 0.06241946 0.06595332

a = tapply(x$V5, list(x$V1, x$V3, x$V2), mean,na.rm=T);

print (apply(a, c(2,3),mean,na.rm=T));
          C        Cs         F        Go        JS        PY      PYml
0 0.4059530 0.4996751 0.3033180 0.4175354 0.4083223 0.3767376 0.1888257
1 0.4243197 0.5202884 0.4710519 0.4708684 0.4424237 0.3692582 0.3695199
2 0.3964593 0.4094849 0.3811432 0.3971758 0.3992997 0.3972929 0.3973700
          R      Rust     Scala       ipy      java        jl        pl
0 0.4065591 0.3391558 0.3871812 0.3421830 0.3797190 0.4773927 0.3497007
1 0.4641753 0.4128587 0.4137810 0.4012505 0.3801242 0.5147258 0.4495260
2 0.3940716 0.3881544 0.3837782 0.3972767 0.3947031 0.3729810 0.3981757
         rb
0 0.4173108
1 0.4443876
2 0.3973568

# Do project similarity prediction
#can we predict new authors for a project?

x = read.table("outPA.gz",sep=";",quote="",comment.char="");
a = tapply(x$V4, list(x$V1,x$V2), mean,na.rm=T);
print (apply(a, 2, mean, na.rm=T));
0.5455169 0.4287622 0.3155700 

#can we predit new projects for an author?

x = read.table("outAP.gz",sep=";",quote="",comment.char="");
a = tapply(x$V4, list(x$V1,x$V2), mean,na.rm=T);
print (apply(a, 2,mean,na.rm=T));
0.6459501 0.3842832 0.2867446 
##################################################
##################################################
##################################################



#fit an overall model instead of on 1/32 of the data
# first model without projects
xA = read.table("outA.gz",sep=";",quote="",comment.char="");
aA = tapply(xA$V5, list(xA$V1, xA$V3), mean,na.rm=T);
print (c(mean(aA[,1],na.rm=T),t.test(aA[,1]-aA[,2],na.rm=T)$conf.int[1:2], t.test(aA[,2]-aA[,3],na.rm=T)$conf.int[1:2]));
[1] 0.45467152 0.05725422 0.05858767 0.14957094 0.15119570
aA = tapply(xA$V5, list(xA$V1, xA$V3, xA$V2), mean,na.rm=T);
print (apply(aA, c(2,3),mean,na.rm=T));
          C        Cs         F        Go        JS        PY      PYml
0 0.3372032 0.4443690 0.2833624 0.3148605 0.3272042 0.3324191 0.1087180
1 0.3779736 0.4413446 0.3941996 0.3667553 0.3732069 0.2934193 0.3089333
2 0.2410490 0.2515277 0.2318679 0.2307177 0.2393430 0.2431279 0.2432443
          R      Rust     Scala       ipy      java        jl        pl
0 0.3576188 0.2324307 0.3308925 0.2608123 0.3621232 0.4033044 0.2310343
1 0.4388650 0.3207504 0.3580214 0.3403137 0.3427869 0.4454717 0.3475361
2 0.2498877 0.2302580 0.2312663 0.2464753 0.2441400 0.2346712 0.2402494
         rb
0 0.3001650
1 0.3266216
2 0.2365377

# now model with projects
x = read.table("out.gz",sep=";",quote="",comment.char="");
a = tapply(x$V5, list(x$V1, x$V3), mean,na.rm=T);
print (c(mean(a[,1],na.rm=T),t.test(a[,1]-a[,2],na.rm=T)$conf.int[1:2], t.test(a[,2]-a[,3],na.rm=T)$conf.int[1:2]));
[1] 0.42057762 0.06031639 0.06163822 0.12221982 0.12375388
a = tapply(x$V5, list(x$V1, x$V3, x$V2), mean,na.rm=T);
print (apply(a, c(2,3),mean,na.rm=T));
          C        Cs         F        Go        JS        PY       PYml
0 0.2971185 0.4002471 0.2653794 0.2841707 0.2934137 0.3129954 0.08763108
1 0.3305545 0.3918358 0.3886875 0.3296505 0.3354792 0.2624989 0.28077446
2 0.2307742 0.2443252 0.2345744 0.2193816 0.2295895 0.2320019 0.23174294
          R      Rust     Scala       ipy      java        jl        pl
0 0.3279675 0.2043637 0.3113372 0.2359361 0.3330258 0.3766405 0.2024342
1 0.4009443 0.2859266 0.3292285 0.3048130 0.3068544 0.4191881 0.3123098
2 0.2368511 0.2168607 0.2182584 0.2336722 0.2350468 0.2188272 0.2332850
         rb
0 0.2739018
1 0.2968572
2 0.2278331

# Do collaborator (new cop-project) similarity prediction?
#See how lsi works
# Do all the evaluations using tfidf + lsi
python3 fitXtl.py PAPkgQ.a100.s2
python3 measureTL.py PAPkgQ.all1.a100.s2 | gzip > outTL.gz

python3 fitXtl.py PAPkgQ.all1.a100.s2 200
python3 measureTL.py PAPkgQ.all1.a100.s2.200 | gzip > outTL200.gz


#########################
# Table 3 Row 1
#########################
#pure LSI
/da3_data/play/api/
python3 fitXl.py PAPkgQ.all1.a100.s2 200 #tfidf + lsi
python3 /da4_data/play/api/measureL.py PAPkgQ.all1.a100.s2 200 | gzip > PAPkgQ.all1.a100.s2.200.l.gz
python3 /da4_data/play/api/measureLw.py PAPkgQ.all1.a100.s2.200 200 | gzip > PAPkgQ.all1.a100.s2.200.lw.gz &

x = read.table("PAPkgQ.all1.a100.s2.200.l.gz",sep=";",quote="",comment.char="");
x=x[x$V2!="PYml",]
x$la=as.character(x$V2)

mttp = function (x) t.test(x)$p.value
mtte = function (x) t.test(x)$estimate

tapply(x$V3-x$V4,x$la,mttp)
tapply(x$V3-x$V4,x$la,mtte)

            C            Cs             F            Go            JS 
 0.000000e+00  0.000000e+00  6.677127e-21  0.000000e+00  0.000000e+00 
           PY             R          Rust         Scala           ipy 
 0.000000e+00 1.561532e-271  0.000000e+00  0.000000e+00  0.000000e+00 
         java            jl            pl            rb 
 0.000000e+00 1.925726e-166  0.000000e+00  0.000000e+00 
#########################
# Table 3 Row 1, not clear where Table 3 Row 2 comes from
#########################
> tapply(x$V3-x$V4,x$la,mtte)
         C         Cs          F         Go         JS         PY          R 
0.04632535 0.28855093 0.10609692 0.28309173 0.05586361 0.27987579 0.13212877 
      Rust      Scala        ipy       java         jl         pl         rb 
0.17607847 0.29022274 0.17760522 0.26892660 0.32896277 0.15910557 0.24831300 



x = read.table("outTL200.gz",sep=";",quote="",comment.char="");
x=x[x$V2!="PYml",]
x$la=as.character(x$V2)
tapply(x$V3-x$V4,x$la,mttp)
tapply(x$V3-x$V4,x$la,mtte)
            C            Cs             F            Go            JS 
 0.000000e+00  0.000000e+00  5.367864e-15  0.000000e+00 7.749541e-261 
           PY             R          Rust         Scala           ipy 
 0.000000e+00  0.000000e+00  0.000000e+00  0.000000e+00 2.434030e-106 
         java            jl            pl            rb 
 0.000000e+00 3.571484e-309  0.000000e+00  0.000000e+00 
> tapply(x$V3-x$V4,x$la,mtte)
          C          Cs           F          Go          JS          PY 
-0.03125579  0.11405128  0.08592825  0.20873489  0.02757071  0.07878625 
          R        Rust       Scala         ipy        java          jl 
 0.31942955  0.14897410  0.22820712  0.02918253  0.22073817  0.65453597 
         pl          rb 
 0.15449072  0.08948351 




#see if athors with fewer apis matter
sel = x$V2=='PY'
a=tapply(x$V5[sel], list(as.character(x$V1[sel]),as.character(x$V3[sel])), length)
ss = a[,1]<87;
nn = names(a[ss,1])
x1=x[sel,]
x2=x1[match(x$V1,nn,nomatch=0)>0,]
tapply(x2$V5, x2$V3, mean)
         0          1 
0.05748117 0.02799630 


for i in 0 2
do 
python3 measureR.py s1 100 10 3 $i > s1.100.10.3.$i
python3 measureR.py s1 200 10 3 $i > s1.200.10.3.$i
python3 measureR.py s1 200 10 20 $i > s1.200.10.20.$i
python3 measureR.py s2 100 10 3 $i > s2.100.10.3.$i
python3 measureR.py s2 200 10 3 $i > s2.200.10.3.$i
python3 measureR.py s2 200 10 20 $i > s2.200.10.20.$i

python3 measureR.py s1 200 20 20 $i > as1.200.20.20.$i &
python3 measureR.py s1 200 20 3 $i > as1.200.20.3.$i &
python3 measureR.py s2 200 20 20 $i > as2.200.20.20.$i &
python3 measureR.py s2 200 20 3 $i > as2.200.20.3.$i &
done

for (f in c("as2.200.20.20.0","as2.200.20.3.0","as2.200.20.20.2","as1.200.20.3.0","as1.200.20.3.2")){
  x = read.table(f,sep=";",quote="",comment.char="");
  print (c(f, tapply(x$V5,x$V3,mean)))
}

python3 measureRdm.py s1 100 30 20 0 > dms1.100.30.20.0
python3 measureRdm.py s1 100 30 20 1 > dms1.100.30.20.1
python3 measureRdm.py s1 100 30 20 3 > dms1.100.30.20.3
python3 measureRdm.py s2 100 30 20 0 > dms2.100.30.20.0
python3 measureRdm.py s2 100 30 20 16 > dms2.100.30.20.16

python3 measureR.py s1 60 30 20 0 > s1.60.30.20.0
python3 measureR.py s1 60 30 20 1 > s1.60.30.20.1


f='dms1.100.30.20.0'
x = read.table(f,sep=";",quote="",comment.char="");
print (tapply(x$V5,x$V3,mean))
         0          1          2 
0.04370712 0.04802327 0.25997248 


f='dms1.100.30.20.1'
x = read.table(f,sep=";",quote="",comment.char="");
print (tapply(x$V5,x$V3,mean))
0.04024299 0.03221836 0.16051011 

f='dms1.100.30.20.3'
x = read.table(f,sep=";",quote="",comment.char="");
print (tapply(x$V5,x$V3,mean))
-0.003097008 -0.020132358  0.130780851

f='dms2.100.30.20.0'
x = read.table(f,sep=";",quote="",comment.char="");
print (tapply(x$V5,x$V3,mean))
           0            1            2 
6.159344e-06 9.600476e-03 4.704827e-01 

f='dms2.100.30.20.16'
x = read.table(f,sep=";",quote="",comment.char="");
print (tapply(x$V5,x$V3,mean))
          0           1           2 
-0.03067361 -0.08871602  0.09374274 


f='s2.60.30.20.0'
x = read.table(f,sep=";",quote="",comment.char="");
print (tapply(x$V5,x$V3,mean))
0.4756433 0.5527332 0.4325226

##################
f='s2.60.30.20.25'
x = read.table(f,sep=";",quote="",comment.char="");
print (tapply(x$V5,x$V3,mean))
0.2703879 0.3146884 0.1867165
##################


f='s1.60.30.20.0'
x = read.table(f,sep=";",quote="",comment.char="");
print (tapply(x$V5,x$V3,mean))
0.3979975 0.4791224 0.3991035 

f='s1.60.30.20.1'
x = read.table(f,sep=";",quote="",comment.char="");
print (tapply(x$V5,x$V3,mean))
0.3185505 0.3947943 0.2949547 

python3 measureR.py s1 100 100 20 0 > s1.100.100.20.0
python3 measureR.py s1 100 100 3 0 > s1.100.100.3.0
python3 measureRdm.py s1 100 100 20 0 > dms1.100.100.20.0
python3 measureRdm.py s1 100 100 3 0 > dms1.100.100.3.0
python3 measureRdm.py s1 200 100 20 6 > dms1.200.100.20.6

f='dms1.100.100.20.0'
x = read.table(f,sep=";",quote="",comment.char="");
print (tapply(x$V5,x$V3,mean))
0.05413300 0.05514304 0.25653314 

f='dms1.100.100.3.0'
x = read.table(f,sep=";",quote="",comment.char="");
print (tapply(x$V5,x$V3,mean))
0.02718822 0.05469516 0.17476730

f='dms1.200.100.20.6'
x = read.table(f,sep=";",quote="",comment.char="");
print (tapply(x$V5,x$V3,mean))
0.028869464 -0.006965821  0.102283144
f='dms1.200.100.20.24'
x = read.table(f,sep=";",quote="",comment.char="");
print (tapply(x$V5,x$V3,mean))
0.1462254 0.1024042 0.1406151

##################
f='s1.100.100.20.0'
x = read.table(f,sep=";",quote="",comment.char="");
print (tapply(x$V5,x$V3,mean))
0.3496247 0.4253337 0.3716085

f='s1.100.100.3.0'
x = read.table(f,sep=";",quote="",comment.char="");
print (tapply(x$V5,x$V3,mean))
0.2759571 0.3595913 0.2865257 
##################


python3 measureR.py s1 200 50 50 0 > s1.200.50.50.0
python3 measureR.py s1 200 100 20 0 > s1.200.100.20.0
python3 measureRnm.py s1 100 100 20 0 > nms1.100.100.20.0

python3 measureR.py s1 100 100 3 4 > s1.100.100.3.4
python3 measureRnm.py s1 100 100 20 1 > nms1.100.100.20.1

f='s1.200.50.50.0'
x = read.table(f,sep=";",quote="",comment.char="");
print (tapply(x$V5,x$V3,mean))
0.3855742 0.4399539 0.4472965
f='s1.200.100.20.0'
x = read.table(f,sep=";",quote="",comment.char="");
print (tapply(x$V5,x$V3,mean))
0.2897465 0.3533489 0.3480027 

f='nms1.100.100.20.0'
x = read.table(f,sep=";",quote="",comment.char="");
print (tapply(x$V5,x$V3,mean))
0.3486340 0.4246614 0.3711284 

f='s1.100.100.20.0'
x = read.table(f,sep=";",quote="",comment.char="");
print (tapply(x$V5,x$V3,mean))
0.3496247 0.4253337 0.3716085

f='s1.100.10.3.0'
x = read.table(f,sep=";",quote="",comment.char="");
print (tapply(x$V5,x$V3,mean))
0.5264899 0.5609438 0.6013127

f='s1.100.100.3.0'
x = read.table(f,sep=";",quote="",comment.char="");
print (tapply(x$V5,x$V3,mean))
0.5264899 0.5609438 0.6013127


f='s1.200.50.50.0'
x = read.table(f,sep=";",quote="",comment.char="");
print (tapply(x$V5,x$V3,mean))
0.3855742 0.4399539 0.4472965 

f='s1.200.50.50.5'
x = read.table(f,sep=";",quote="",comment.char="");
print (tapply(x$V5,x$V3,mean))
0.2145872 0.2632262 0.2146869
f='s1.200.50.50.21'
x = read.table(f,sep=";",quote="",comment.char="");
print (tapply(x$V5,x$V3,mean))
0.0924019 0.1303114 0.1005268 

f='s1.200.10.20.0'
x = read.table(f,sep=";",quote="",comment.char="");
print (tapply(x$V5,x$V3,mean))
0.6147229 0.6447941 0.7171350


f='s1.60.30.20.0'
x = read.table(f,sep=";",quote="",comment.char="");
print (tapply(x$V5,x$V3,mean))
0.3979975 0.4791224 0.3991035

f='nms1.100.100.20.1'
x = read.table(f,sep=";",quote="",comment.char="");
print (tapply(x$V5,x$V3,mean))
0.2666719 0.3349231 0.2682732

f='s1.100.100.3.4'
x = read.table(f,sep=";",quote="",comment.char="");
print (tapply(x$V5,x$V3,mean))
0.2054330 0.2879944 0.1965531 

f='s1.100.100.3.7'
x = read.table(f,sep=";",quote="",comment.char="");
print (tapply(x$V5,x$V3,mean))
0.2039945 0.2862087 0.1899200


#PR resolution
cd /da3_data/play/api/
python3 measureAPR.py > outAPR
x = read.table("outAPR",sep=";",quote="",comment.char="");

y = x$V3/(x$V3+x$V4)


y = cbind( x$V3, x$V4)
m = glm(y ~ x$V5,family=binomial)
summary(m)
Call:
glm(formula = y ~ x$V5, family = binomial)

Deviance Residuals: 
    Min       1Q   Median       3Q      Max  
-3.1070  -1.3704  -0.0973   0.9925   3.7900  

Coefficients:
            Estimate Std. Error z value Pr(>|z|)    
(Intercept)  -1.0295     0.3790  -2.717   0.0066 ** 
x$V5          3.6884     0.7495   4.921  8.6e-07 ***
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

(Dispersion parameter for binomial family taken to be 1)

    Null deviance: 345.97  on 170  degrees of freedom
Residual deviance: 320.09  on 169  degrees of freedom
AIC: 361.18

Number of Fisher Scoring iterations: 4

#Joongi Kim <joongi@an.kaist.ac.kr>;abbr_deasync;0;1;0.2763929839195558
##################################################
##################################################

# do a full model not on pairs
python3 fitXa100.py PtAPkgQ 100 50 20 3 1518784533

records so far after C:428089712
records so far after Cs:454179878
records so far after F:454428020
records so far after Go:477059731
records so far after JS:483668796
records so far after PY:545977933
records so far after R:546676125
records so far after Rust:549318575
records so far after Scala:558835749
records so far after ipy:559437749
records so far after java:742917966
records so far after jl:743145989
records so far after pl:747243704
records so far after rb:754892793
records:754892793



#still calculating, do binary instead
python3 measureAPI.py 0 | gzip > doc2vec.200.1.5.PAPkgQ.all2.a100.b50.0.api.gz
python3 measureAPI.py 1 | gzip > doc2vec.200.1.5.PAPkgQ.all2.a100.b50.1.api.gz
f='doc2vec.200.1.5.PAP:kgQ.all2.a100.b50.0.api.gz'
x = read.table(f,sep=";",quote="",comment.char="");
x=x[x$V2!='PYml',]
x$V2=as.character(x$V2);
#do by language - V3
a = tapply(x$V4, list(x$V1,x$V2,x$V3), mean,na.rm=T);

t.test(a[,2]-a[,3])



python3 measureAP.py 0 | gzip > doc2vec.200.1.5.PAPkgQ.all2.a100.b50.0.AP.gz &
python3 measureAP.py 1 | gzip > doc2vec.200.1.5.PAPkgQ.all2.a100.b50.1.AP.gz &
f='doc2vec.200.1.5.PAPkgQ.all2.a100.b50.0.AP.gz'
x = read.table(f,sep=";",quote="",comment.char="");
a = tapply(x$V4, list(x$V1,x$V2), mean,na.rm=T);
t.test(a[,2]-a[,3])
t = 123.18, df = 36917, p-value < 2.2e-16
alternative hypothesis: true mean is not equal to 0
95 percent confidence interval:
 0.1117263 0.1153392
sample estimates:
mean of x 
0.1135327 



python3 measurePA.py 0 | gzip > doc2vec.200.1.5.PAPkgQ.all2.a100.b50.0.PA.gz &
python3 measurePA.py 1 | gzip > doc2vec.200.1.5.PAPkgQ.all2.a100.b50.1.PA.gz &
f='doc2vec.200.1.5.PAPkgQ.all2.a100.b50.0.PA.gz'
x = read.table(f,sep=";",quote="",comment.char="");
a = tapply(x$V4, list(x$V1,x$V2), mean,na.rm=T);

t.test(a[,2]-a[,3])
t = 138.26, df = 55112, p-value < 2.2e-16
alternative hypothesis: true mean is not equal to 0
95 percent confidence interval:
 0.1113763 0.1145795
sample estimates:
mean of x 
0.1129779 



##################################################
# FSE Submission PR
#PR resolution
#try a more careful Table 4
#####################################
#do full model using all languages on past data, predict acceptance on future
# Put this in the paper
for cut in 1487226933 1503005733 1518784533
do 
zcat prs.all.s1 | perl -e 'while(<STDIN>){chop(); ($p,$la,$t,$a,@ms)=split(/;/);if ($t < '$cut'){ for $m (@ms){$k{"$p;$la;$a"}{$m}++}}};while (($p, $v)=each %k){@ms=sort keys %{$v}; print "$p;".(join ";", @ms)."\n";}' | gzip > prs.all.s2.$cut
zcat prs.all.s1 | perl -e 'while(<STDIN>){chop(); ($p,$la,$t,$a,@ms)=split(/;/);if ($t >= '$cut'){ for $m (@ms){$k{"$p;$la;$a"}{$m}++}}};while (($p, $v)=each %k){@ms=sort keys %{$v}; print "$p;".(join ";", @ms)."\n";}' | gzip > prs.all.s4.$cut
done


for cut in 1487226933 1503005733 1518784533
do perl cmpAprs.perl prs.all $cut | gzip > prs.all.sAD.$cut
done

for cut in 1518784533
do python3 measureAPprs.py doc2vec.100.50.20.3.prs.all.s1.0 prs.all.sAD.$cut > out.prs.$cut.3.50
 python3 measureAPprs.py doc2vec.100.50.20.100.$cut.prs.all.s1.0 prs.all.sAD.1518784533 > out.prs.$cut.100.50
 python3 measureAPprs.py doc2vec.100.50.20.3.prs.all.s1.2 prs.all.sAD.$cut > out.prs.$cut.3.50.2
 python3 measureAPprs.py doc2vec.100.50.20.3.prs.all.s1.3 prs.all.sAD.$cut > out.prs.$cut.3.50.3
 python3 measureAPprs.py doc2vec.100.50.20.3.$cut.prs.all.s1.5 prs.all.sAD.1518784533 > out.prs.$cut.3.50.5
done
x=read.table("out.prs.1518784533.100.50.1",sep=";",quote="",comment.char="");
#this does not seem to capture enough specificity, removing rere (< 100 instances) APIs appears to hurt  

x=read.table("out.prs.1518784533.3.50.1",sep=";",quote="",comment.char="");
x=read.table("out.prs.1518784533.3.50.2",sep=";",quote="",comment.char="");
x=x[x$V8+x$V9>0,]
#response
y=cbind(x$V9,x$V8)
sim=x$V10
summary(glm(y~sim,family=binomial))$coefficients
             Estimate Std. Error   z value     Pr(>|z|)
(Intercept) -1.021836  0.1279698 -7.984975 1.405512e-15
sim          1.283114  0.2942174  4.361109 1.294048e-05
>  summary(glm(y~sim +I(x$V6 > 0),family=binomial))$coefficients
                  Estimate Std. Error   z value     Pr(>|z|)
(Intercept)     -1.0091211  0.1281618 -7.873804 3.440166e-15
sim              0.9553116  0.3065614  3.116216 1.831879e-03
I(x$V6 > 0)TRUE  0.3659381  0.0944885  3.872832 1.075779e-04

#overfit by now?
x=read.table("out.prs.1518784533.3.50.5",sep=";",quote="",comment.char="");
summary(glm(y~sim,family=binomial))$coefficients
              Estimate Std. Error   z value     Pr(>|z|)
(Intercept) -0.9738149  0.1199521 -8.118362 4.725167e-16
sim          1.2446302  0.2920199  4.262141 2.024773e-05
>  summary(glm(y~sim +I(x$V6 > 0),family=binomial))$coefficients
                  Estimate Std. Error   z value     Pr(>|z|)
(Intercept)     -0.9742389 0.12008804 -8.112706 4.950477e-16
sim              0.9249281 0.30346857  3.047855 2.304814e-03
I(x$V6 > 0)TRUE  0.3707065 0.09424937  3.933252 8.380424e-05


dim(x)
[1] 766  10
sum(x$V8+x$V9)
[1] 2334
######################################

Fit JS model only on past data, predict acceptance on future
la=JS 
for cut in 1487226933 1503005733 1518784533
do 
zcat PtAPkgQ$la.prs.s1 | perl -e 'while(<STDIN>){chop(); ($p,$la,$t,$a,@ms)=split(/;/);if ($t < '$cut'){ for $m (@ms){$k{"$p;$la;$a"}{$m}++}}};while (($p, $v)=each %k){@ms=sort keys %{$v}; print "$p;".(join ";", @ms)."\n";}' | gzip > PAPkgQ$la.prs.s2.$cut
zcat PtAPkgQ$la.prs.s1 | perl -e 'while(<STDIN>){chop(); ($p,$la,$t,$a,@ms)=split(/;/);if ($t >= '$cut'){ for $m (@ms){$k{"$p;$la;$a"}{$m}++}}};while (($p, $v)=each %k){@ms=sort keys %{$v}; print "$p;".(join ";", @ms)."\n";}' | gzip > PAPkgQ$la.prs.s4.$cut
done

for cut in 1487226933 1503005733 1518784533
do perl cmpAprs.perl JS $cut | gzip > PAPkgQJS.prs.sAD.$cut
done


for cut in 1487226933 1503005733 1518784533
do for mn in 3 10 100
 do python3 measureAPprs.py $cut $mn 0 50 > outJSprs.$cut.$mn.0.50
done; done
for cut in 1518784533
do for mn in 3 10 100
 do python3 measureAPprs.py $cut $mn 0 20 > outJSprs.$cut.$mn.0.20
done; done


summary(glm(y~sim +I(x$V6 > 0),family=binomial))$coefficients
cut=1518784533
mn=3
python3 measureAPprs.py $cut $mn 7 20 > outJSprs.$cut.$mn.7.20
x=read.table("outJSprs.1518784533.3.7.20",sep=";",quote="")
summary(glm(y~sim,family=binomial))$coefficients
              Estimate Std. Error   z value     Pr(>|z|)
(Intercept) -0.9536817  0.1373422 -6.943838 3.815887e-12
sim          1.2666900  0.2946566  4.298869 1.716719e-05
> summary(glm(y~sim +I(x$V6 > 0),family=binomial))$coefficients
                  Estimate Std. Error   z value     Pr(>|z|)
(Intercept)     -0.9832588  0.1377844 -7.136210 9.593913e-13
sim              1.0959133  0.3019879  3.628997 2.845242e-04
I(x$V6 > 0)TRUE  0.2738509  0.1098017  2.494049 1.262951e-02



> dim(x)
[1] 819  10
> sum(x$V8+x$V9)
[1] 2663
x=read.table("outJSprs.1518784533.3.0.50",sep=";",quote="")

x=read.table("outJSprs.1518784533.100.0.20",sep=";",quote="")
summary(glm(y~sim,family=binomial))$coefficients
              Estimate Std. Error   z value     Pr(>|z|)
(Intercept) -0.9000813  0.1474172 -6.105672 1.023688e-09
sim          0.9594344  0.2695859  3.558919 3.723848e-04
> summary(glm(y~sim +I(x$V6 > 0),family=binomial))$coefficients
                  Estimate Std. Error   z value     Pr(>|z|)
(Intercept)     -0.9293249  0.1481234 -6.273990 3.519110e-10
sim              0.7989229  0.2763595  2.890883 3.841611e-03
I(x$V6 > 0)TRUE  0.2952756  0.1093530  2.700207 6.929626e-03


x=read.table("outJSprs.1518784533.10.0.20",sep=";",quote="")

x=read.table("outJSprs.1518784533.3.0.20",sep=";",quote="")
summary(glm(y~sim,family=binomial))$coefficients
              Estimate Std. Error   z value     Pr(>|z|)
(Intercept) -0.9615074  0.1606088 -5.986643 2.142162e-09
sim          1.0347529  0.2845941  3.635890 2.770228e-04
summary(glm(y~sim +I(x$V6 > 0),family=binomial))$coefficients
                  Estimate Std. Error   z value     Pr(>|z|)
(Intercept)     -0.9984659  0.1617249 -6.173853 6.664567e-10
sim              0.8898408  0.2899991  3.068426 2.151898e-03
I(x$V6 > 0)TRUE  0.3033015  0.1085834  2.793258 5.218000e-03


x=read.table("outJSprs.1518784533.100.0.50",sep=";",quote="")
x=read.table("outJSprs.1518784533.0.0.50",sep=";",quote="")

x=read.table("outJSprs.1518784533.3.0.50",sep=";",quote="")
x=x[x$V8+x$V9>0,]
#response
y=cbind(x$V9,x$V8)
sim=x$V10
summary(glm(y~sim,family=binomial))$coefficients
              Estimate Std. Error   z value     Pr(>|z|)
(Intercept) -0.9081947  0.1523620 -5.960768 2.510556e-09
sim          0.9777273  0.2806686  3.483565 4.947826e-04

summary(glm(y~sim +I(x$V6 > 0),family=binomial))$coefficients
                  Estimate Std. Error   z value     Pr(>|z|)
(Intercept)     -0.9393741  0.1530750 -6.136691 8.425798e-10
sim              0.8169845  0.2869315  2.847316 4.408964e-03
I(x$V6 > 0)TRUE  0.3002647  0.1090680  2.753006 5.905081e-03
> dim(x)
[1] 477  10
> sum(x$V8+x$V9)
[1] 1567




######################
# Table 1: report stats for the FSE submission 
########################
for la in F jl R ipy pl Rust Cs Go Scala PY JS rb java C; do zcat  /da?_data/play/api/PtAPkgQ$la.s| perl -e 'while(<STDIN>){chop();($p,$t,$a,@ms)=split(/;/);$as{$a}++;$ps{$p}++;$ls{$#ms}++, $n++;} print STDERR "'$la';$n;".(scalar(keys %as)).";".(scalar(keys %ps))."\n"; for $nl (keys %ls){print "$nl;$ls{$nl}\n"}' | gzip > PtAPkgQ$la.nm;   done
lang;delta;authors;projects
F;1714314;23179;16084
jl;1173066;16029;32875
R;6591806;325797;501196
ipy;10480954;630743;983169
pl;21561320;456107;558712
Rust;12400022;257072;305284
Cs;219984011;1864720;2923567
Go;106791380;392871;597846
Scala;37173969;164111;215178
PY;560726046;4185716;6509696
JS;140972726;3291058;7527168
rb;85990225;1164335;2386418
java;1119433526;4518005;7049986
C;2019398881;3339816;4580319

#total number of delta
cat |cut -d\; -f2 | awk '{print i+=$1}'|tail -1
4,344,392,246

#count distinct APIs
for la in F jl R ipy pl Rust Cs Go Scala PY JS rb java C; do zcat  /da?_data/play/api/PtAPkgQ$la.s| perl -e 'while(<STDIN>){chop();($p,$t,$a,@ms)=split(/;/);for $m (@ms){$mm{$m}++;}} print "'$la';".(scalar(keys %mm))."\n";'   done
F;55266
jl;114038
R;79190
ipy;544464
pl;57495
Rust;97125
Cs;6005837
Go;1540313
Scala;2895075
PY;16032127
JS;1180985
rb;2053615
java;77586461
C;2483135
cat | cut -d\; -f2 | awk '{print i+=$1}' | tail -1
110,725,126

#what fraction of delta has 10 or fewer APIs?
for la in F jl R ipy pl Rust Cs Go Scala PY JS rb java C; do zcat PtAPkgQ$la.nm|lsort 1G -t\; -k2 -rn | awk -F\; '{n+=$2; c[$1]=$2} END {print "'$la'",(c[0]+c[1]+c[2]+c[3]+c[4]+c[5]+c[6]+c[7]+c[8]+c[9])/n,n,$1}'; done
lanf fraction total maxAPisPerDelta
F 0.864287 1714314 106
jl 0.918882 1173066 108
R 0.953017 6591806 117
ipy 0.76009 10480954 117
pl 0.958241 21561320 109
Rust 0.944941 12400022 53
Cs 0.844412 219984011 150
Go 0.841157 106791380 1362
Scala 0.765806 37173969 124
PY 0.814464 560726046 1001
JS 0.791007 140972726 10014
rb 0.978319 85990225 1002
java 0.574622 1119433526 1004
C 0.731285 2019398881 1007

for la in F jl R ipy pl Rust Cs Go Scala PY JS rb java C; do zcat PtAPkgQ$la.nm|lsort 1G -t\; -k2 -rn | awk -F\; '{n+=$2; c[$1]=$2} END {for (i=0; i<25;i++)v+=c[i]; print "'$la'",v/n,n,$1}';done
ipy 0.981094 10480954 117
Scala 0.978922 37173969 124
JS 0.889159 140972726 10014
java 0.87325 1119433526 1004

#what fraction of delta has 50 or fewer APIs?
for la in F jl R ipy pl Rust Cs Go Scala PY JS rb java C; do zcat PtAPkgQ$la.nm|lsort 1G -t\; -k2 -rn | awk -F\; '{n+=$2; c[$1]=$2} END {for (i=0; i<50;i++)v+=c[i]; print "'$la'",v/n,n,$1}';done
F 0.996817 1714314 106
jl 0.994654 1173066 108
R 0.999581 6591806 117
ipy 0.999316 10480954 117
pl 0.999989 21561320 109
Rust 0.999999 12400022 53
Cs 0.99971 219984011 150
Go 0.999108 106791380 1362
Scala 0.998829 37173969 124
PY 0.997004 560726046 1001
JS 0.920681 140972726 10014
rb 0.999357 85990225 1002
java 0.973206 1119433526 1004
C 0.998161 2019398881 1007

for la in F jl R ipy pl Rust Cs Go Scala PY JS rb java C; do zcat PtAPkgQ$la.nm|lsort 1G -t\; -k2 -rn | awk -F\; '{n+=$2; c[$1]=$2} END {for (i=0; i<50;i++)v+=c[i]; for (i=0; i<25;i++)v25+=c[i];for (i=0; i<10;i++)v10+=c[i];print "'$la'",v10/n,v25/n,v/n,n,$1}';done
F 0.864287 0.97831 0.996817 1714314 106
jl 0.918882 0.982022 0.994654 1173066 108
R 0.953017 0.996757 0.999581 6591806 117
ipy 0.76009 0.981094 0.999316 10480954 117
pl 0.958241 0.999547 0.999989 21561320 109
Rust 0.944941 0.997445 0.999999 12400022 53
Cs 0.844412 0.993558 0.99971 219984011 150
Go 0.841157 0.988415 0.999108 106791380 1362
Scala 0.765806 0.978922 0.998829 37173969 124
PY 0.814464 0.977265 0.997004 560726046 1001
JS 0.791007 0.889159 0.920681 140972726 10014
rb 0.978319 0.996 0.999357 85990225 1002
java 0.574622 0.87325 0.973206 1119433526 1004
C 0.731285 0.965926 0.998161 2019398881 1007

for la in F jl R ipy pl Rust Cs Go Scala PY JS rb java C; do zcat  /da4_data/play/api/PtAPkgQ$la.a100.s| perl -e 'while(<STDIN>){chop();($p,$t,$a,@ms)=split(/;/);$as{$a}++;$ps{$p}++;$ls{$#ms}++, $n++;} print STDERR "'$la';$n;".(scalar(keys %as)).";".(scalar(keys %ps))."\n"; for $nl (keys %ls){print "$nl;$ls{$nl}\n"}' | gzip > PtAPkgQ$la.a100.nm;   done
F;344552;1431;2219
jl;504320;1160;7722
R;1158873;7347;36983
ipy;1811746;20687;79585
pl;5113871;21069;68769
Rust;6550473;18195;55199
Cs;52313421;38619;281015
Go;51279675;20645;124218
Scala;12455617;8937;34664
PY;128114928;82223;730013
JS;15118273;58081;611156
rb;12014953;24989;162232
java;290099632;73443;546557
C;602688210;65323;302328
1,179,568,544

for la in F jl R ipy pl Rust Cs Go Scala PY JS rb java C; do zcat  /da4_data/play/api/PtAPkgQ$la.a10.s| perl -e 'while(<STDIN>){chop();($p,$t,$a,@ms)=split(/;/);$as{$a}++;$ps{$p}++;$ls{$#ms}++, $n++;} print STDERR "'$la';$n;".(scalar(keys %as)).";".(scalar(keys %ps))."\n"; for $nl (keys %ls){print "$nl;$ls{$nl}\n"}' | gzip > PtAPkgQ$la.a10.nm;   done
F;561214;3796;5190
jl;753150;3512;13534
R;2732365;28897;119376
ipy;4202017;82677;257966
pl;7920200;62996;148274
Rust;8782373;48580;107879
Cs;79621191;156816;708727
Go;66153157;64152;248059
Scala;17425527;24847;71368
PY;207647121;344269;1866940
JS;35773587;250915;1791320
rb;24196791;97982;497732
java;390303008;295725;1429521
C;822861651;242355;833121


zcat PAPkgQ.all2.a100.b50 | cut -d\; -f2 | uniq -c > PAPkgQ.all2.a100.b50.cnts
awk '{n[$2]+=$1}END{for (i in n){print i,n[i]}}' PAPkgQ.all2.a100.b50.cnts
Rust 1852525
java 68945988
Go 15083296
PY 73122267
Cs 30037469
JS 24352936
C 36866524
ipy 3816574
rb 10385313
R 1137280
F 90103
jl 492493
Scala 5070321
pl 3932819



dm=0
The doc-vectors are obtained by training a neural network on the synthetic task of predicting a center word based an average of both context word-vectors and the full document’s doc-vector.

##################################################
##################################################
##################################################
# FSE Submission Evaluating against qualitative data
##################################################
##################################################
##################################################
cd /da0_data/play/idRes/
zcat /da0_data/basemaps/gz/a2AQ.s | cut -d\; -f1 | perl -I /home/audris/lib64/perl5 -I /home/audris/lib/x86_64-linux-gnu/perl -I /home/audris/lookup -ane 'use cmt; chop();@x=splitSignature($_);@n = split(/ /, $x[0]); @e = split(/\@/, $x[1]); print "$_;$x[0];$x[1];$n[0];$n[$#n];$e[0];$e[1]\n";' | gzip > asQ.split
wget https://zenodo.org/record/1484498/files/socetio.csv
wget https://zenodo.org/record/1484498/files/react.csv
wget https://zenodo.org/record/1484498/files/mongodb.csv

cut -d\; -f3 {socketio,mongodb,react}.csv|grep -v email | sed 's|"||g' | lsort 1G -u > survey.email

cp -p /da4_data/play/Andrey/experts.csv .
cat experts.csv | perl ~/lookup/mp.perl 3 /da0_data/basemaps/gz/a2AQ.s > expertsA.csv
perl -ane 'chop();@x=split(/;/); $z=shift @x; $a=pop @x; print "$a;".(join ";", @x).";$z\n";' < expertsA.csv | lsort 1G -t\; -k1,1 > expertsA.csv.s

#in da0_data/play/idRes
zcat asQ.split|cut -d\; -f1,3 | python md5.py | lsort 20G | gzip > emd52a

for i in socketio mongodb react
do cut -d\; -f3,6,17 $i.csv | grep -v email | sed 's|"||g'|sed "s|;|;"$i";|" 
done | lsort 1G -t\; -k1 > exa.csv

zcat emd52a | join -t\; exa.csv - > exa1.csv
cat exa1.csv | perl ~/lookup/mp.perl 4 /da0_data/basemaps/gz/a2AQ.s > exA.csv


python3 m675.py > m675.full
568

#########################
# for Table 6
/da3_data/play/api/m675.py

python3 m675.py > out675.a10.2
568

#x=read.table('out675.full',sep=";")
x=read.table('out675.a10.2',sep=";")
x$rv = apply(x[,-c(1:4)],1,mean)
tapply(x$V4-x$rv,x$V1,mtte)
  mongodb     react  socketio 
0.1320529 0.1640949 0.3011829 

x$y=x$V4-x$rv
summary(lm(y~V1+V2+V3,data=x))
              Estimate Std. Error t value Pr(>|t|)    
(Intercept)  8.312e-02  1.845e-02   4.504  8.1e-06 ***
V1react      1.789e-02  1.416e-02   1.263 0.207006    
V1socketio   1.770e-01  1.649e-02  10.732  < 2e-16 ***
V2          -1.742e-05  1.447e-05  -1.204 0.229065    
V3           1.725e-02  4.420e-03   3.903 0.000107 ***
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Residual standard error: 0.0968 on 563 degrees of freedom
Multiple R-squared:  0.2452,    Adjusted R-squared:  0.2399 

##################################################
# Table 5
##################################################
summary(lm(V4~V1+log(V2)+V3,data=x))
           Estimate Std. Error t value Pr(>|t|)    
(Intercept) 0.151631   0.016344   9.278  < 2e-16 ***
V1react     0.018500   0.012213   1.515 0.130377    
V1socketio  0.173256   0.014407  12.026  < 2e-16 ***
log(V2)     0.003201   0.001707   1.875 0.061251 .  
V3          0.014425   0.003842   3.755 0.000191 ***
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Residual standard error: 0.08424 on 563 degrees of freedom
Multiple R-squared:  0.3036,    Adjusted R-squared:  0.2987 
F-statistic: 61.37 on 4 and 563 DF,  p-value: < 2.2e-16

##################################################
# Table 6
##################################################
summary(lm(V3~V1+log(V2)+V4,data=x))
             Estimate Std. Error t value Pr(>|t|)    
(Intercept)  2.48619    0.15868  15.668  < 2e-16 ***
V1react      0.67528    0.12951   5.214 2.60e-07 ***
V1socketio  -0.81566    0.17161  -4.753 2.55e-06 ***
log(V2)      0.07209    0.01830   3.938 9.23e-05 ***
V4           1.69367    0.45106   3.755 0.000191 ***
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Residual standard error: 0.9128 on 563 degrees of freedom
Multiple R-squared:  0.2433,    Adjusted R-squared:  0.2379 


##################################################
##################################################
##################################################
##################################################

##########################################
f='doc2vec.200.1.5.PAPkgQ.all2.a100.b50.0'
import gzip,collections,gensim.models.doc2vec,math
from gensim.models import Doc2Vec
f='doc2vec.200.1.5.PAPkgQ.all2.a100.b50.0'
mod = Doc2Vec.load (f)

mod.most_similar('readr')
mod.most_similar('data.table')
mod.most_similar('pandas')
mod.wv.similar_by_vector(mod.docvecs['R'])
mod.wv.similar_by_vector(mod.docvecs['JS'])
mod.wv.similar_by_vector(-mod.docvecs['PY']+mod.docvecs['R']+mod.wv.get_vector('pandas'))

mod.most_similar('readr')
[('tidyr', 0.9841052889823914), ('tidyverse', 0.9832087159156799), ('stringi', 0.9801042675971985), ('stringr', 0.9781765937805176), ('lubridate', 0.9763726592063904), ('purrr', 0.9756897687911987), ('shinydashboard', 0.9756823182106018), ('magrittr', 0.9755414724349976), ('shiny', 0.975053608417511), ('randomForest', 0.9747953414916992)]


>>> mod.most_similar('data.table')
[('reshape2', 0.9711512327194214), ('doMC', 0.9693106412887573), ('stringi', 0.9675365686416626), ('randomForest', 0.9670689105987549), ('Rcpp', 0.9667882919311523), ('gridExtra', 0.9667866826057434), ('doParallel', 0.9664762020111084), ('microbenchmark', 0.9648315906524658), ('roxygen2', 0.9642760753631592), ('dplyr', 0.964144766330719)]

>>>mod.most_similar('pandas')
[('matplotlib.pyplot', 0.8937591910362244), ('seaborn', 0.8759922981262207), ('numpy', 0.8473187685012817), ('scipy.stats', 0.8471274971961975), ('statsmodels.api', 0.8400012850761414), ('matplotlib', 0.8307245969772339), ('pandas.DataFrame', 0.827376127243042), ('scipy.stats.norm', 0.8176226615905762), ('scipy', 0.8164981603622437), ('sklearn.decomposition.PCA', 0.8095367550849915)]

>>> mod.wv.similar_by_vector(mod.docvecs['R'])
[('ggplot2', 0.9416438341140747), ('dplyr', 0.9274526238441467), ('testthat', 0.922480583190918), ('reshape2', 0.917689323425293), ('stringr', 0.9147644639015198), ('magrittr', 0.9097719192504883), ('knitr', 0.9066928625106812), ('data.table', 0.905428409576416), ('tidyr', 0.9049832820892334), ('readr', 0.9037255048751831)]



>>> mod.wv.similar_by_vector(mod.docvecs['JS'])
[('lodash', 0.8152533769607544), ('moment', 0.8062593936920166), ('underscore', 0.789423942565918), ('rimraf', 0.7869737148284912), ('socket.io', 0.7839827537536621), ('npm', 0.7823067903518677), ('webpack', 0.7808771133422852), ('marked', 0.7798334360122681), ('jquery', 0.776277482509613), ('express', 0.7752944231033325)]
>>> mod.wv.similar_by_vector(-mod.docvecs['PY']+mod.docvecs['R']+mod.wv.get_vector('pandas'))
[('ggplot2', 0.6462166905403137), ('jsonlite', 0.6377792358398438), ('dplyr', 0.6358301043510437), ('knitr', 0.6333364844322205), ('stringr', 0.6324319839477539), ('RColorBrewer', 0.6322579383850098), ('testthat', 0.6297568678855896), ('readr', 0.6281371116638184), ('RCurl', 0.6270754337310791), ('tidyr', 0.6259021162986755)]



