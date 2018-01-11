#coding:utf8

# 这个测试脚本测试在LongHit集合上，使用charmean、charKmeans、compMean、compKMeans4个方法的AUC值

import sys
import random
import numpy as np
import gensim
from gensim import matutils

reload(sys)
sys.setdefaultencoding('utf-8')

voc_file = open('../../data/wiki_phrase2/wiki_voc')
model = gensim.models.Word2VecWC.load("../../data/wiki_phrase2/wiki_model")
voc_list = list(voc_file)

uniSet = set()
biTriSet = set()
quadSet = set()
longSet = set()

for line in voc_list:
    word = line.split(' ')[0].decode()
    l = len(word)
    if l == 1:
        uniSet.add(word)
    elif l == 2 or l == 3:
        biTriSet.add(word)
    elif l == 4:
        quadSet.add(word)
    else:
        longSet.add(word)

print "uniSet", len(uniSet)
print "biTriSet", len(biTriSet)
print "quadSet", len(quadSet)
print "longSet", len(longSet)

wordSet = biTriSet.union(quadSet).union(longSet)



phraseList = list()
plist = list()
longSetCopy = longSet.copy()
for p in longSetCopy:
    pSubList = list()
    # print p
    len_p = len(p)
    for start in xrange(len_p):
        for length in xrange(len_p - start + 1):
            if p[start: start+length] in wordSet and not p[start: start+length] == p:

                pSubList.append(p[start: start+length])

    if len(pSubList) > 0:#This phrase has many sub parts
        phraseList.append(p)
        plist.append(pSubList)
    else:
        longSet.remove(p)

pDict = dict(zip(phraseList, plist))

testF = open('../../data/redi2')
testLines = list(testF)
testF.close()

#两个都是短词，在bitri中
#包含长词，且都在集合中
#包含长词，且长词不在集合中

# 把以上三个东西搞成集合

#          |shortHit | shortMiss | longHit | longMiss
# shortHit |shortSet | ignore    | longHit | longMiss
# shortMiss|ignore   | ignore    | ignore  | ignore
# longHit  |longHit  | ignore    | longHit | longMiss
# longMiss |longMiss | ignore    |longMiss | longMiss

shortSet = set()
longHitSet = set()
longMissSet = set()
doubleLongMissSet = set()
ignoreSet = set()

for line in testLines:
    word1, word2 = [i.strip() for i in  line.decode().split('\t')]
    if word1.find(word2) == -1 and  word2.find(word1) == -1:

        if word1 in biTriSet and word2 in biTriSet:#shortSet1
            shortSet.add((word1, word2))
            continue

        if word1 in biTriSet and word2 in longSet:
            longHitSet.add((word1, word2))
            continue

        if word2 in biTriSet and word1 in longSet:
            longHitSet.add((word1, word2))
            continue

        if word1 in longSet and word2 in longSet:
            longHitSet.add((word1, word2))
            continue

        if word1 in biTriSet and len(word2) > 4:
            longMissSet.add((word1, word2))
            continue

        if word2 in biTriSet and len(word1) > 4:
            longMissSet.add((word1, word2))
            continue

        if len(word1) > 4 and len(word) > 4:
            doubleLongMissSet.add((word1, word2))
            continue
        ignoreSet.add((word1, word2))


print "shortSet          :",len(shortSet)
print "longHitSet        :",len(longHitSet)
print "longMissSet       :",len(longMissSet)
print "doubleLongMissSet :",len(doubleLongMissSet)
print "ignoreSet         :",len(ignoreSet)

# for i in longHitSet:
#     print i[0], i[1]

# 首先我们需要产生一个负例
# 负例的长度与正例相同
# 随机取两个正例S11和S22
# 注意：还是至少取一个长的才有意义
#

longHitSet_neg = set()

while len(longHitSet_neg) < len(longHitSet):
    s1, s2 = random.sample(longHitSet, 2)
    if len(s1[0]) < 5 and len(s2[1]) < 5:
        continue

    longHitSet_neg.add((s1[0], s2[1]))

# for i in longHitSet_neg:
#     print i[0], i[1]


#下面生成complist
#一个函数将返回三个相似度，分别是标准答案，charmean， compmean
#注意1： 短词语的相似度总是用标准答案
#注意2： compmean是用普遍匹配的方法得来的，它没有使用前向最长匹配

def vecMean(model, compList):
    # 一个生成compList的简单平均值的简单函数
    vecList= model[compList]
    return np.asarray(vecList).mean(axis=0)

from scipy.cluster.vq import vq,kmeans,whiten
from scipy.stats import mode

def vecKmean(model, word, compList):
    # 一个根据compList查找相近词，并求KMeans 的向量的函数
    sim_words = list()
    for char in compList:
        simrst = model.wv.most_similar(char, topn=1000)
        for sim_word, _ in simrst:
            if sim_word == word:
                continue
            sim_words.append(sim_word)
    kmeans_method = len(word)

    data = whiten(model[sim_words])
    centoids = kmeans(data, kmeans_method)

    labels = vq(data, centoids[0])[0]
    max_label = mode(labels).mode[0]

    return centoids[0][max_label]

def combTest(model, w1, w2, pDict):
    rstStandard = model.similarity(w1, w2) #标准答案

    vec1c = model[w1] if len(w1) < 5 else vecMean(model, [i for i in w1 if i in model])
    vec2c = model[w2] if len(w2) < 5 else vecMean(model, [i for i in w2 if i in model])
    rstc = np.dot(matutils.unitvec(vec1c), matutils.unitvec(vec2c))

    vec1s = model[w1] if len(w1) < 5 else vecMean(model, pDict[w1])
    vec2s = model[w2] if len(w2) < 5 else vecMean(model, pDict[w2])
    rsts = np.dot(matutils.unitvec(vec1s), matutils.unitvec(vec2s))

    # 下面是两个K-Means
    vec1kc = model[w1] if len(w1) < 5 else vecKmean(model, w1, [i for i in w1 if i in model])
    vec2kc = model[w2] if len(w2) < 5 else vecKmean(model, w2, [i for i in w2 if i in model])
    rstkc = np.dot(matutils.unitvec(vec1kc), matutils.unitvec(vec2kc))

    vec1ks = model[w1] if len(w1) < 5 else vecKmean(model, w1, pDict[w1])
    vec2ks = model[w2] if len(w2) < 5 else vecKmean(model, w2, pDict[w2])
    rstks = np.dot(matutils.unitvec(vec1ks), matutils.unitvec(vec2ks))

    return rstStandard, rstc, rsts, rstkc, rstks
rstList = list()
rstLabel =list()
cnt = 0
for i in longHitSet:
    # print i[0], i[1]
    cnt += 1

    r = combTest(model, i[0], i[1], pDict)
    rstList.append(r)
    rstLabel.append(True)
    print cnt
    # if cnt > 10:
    #     break
    # print "====="
cnt = 0
for i in longHitSet_neg:
    # print i[0], i[1]
    # print combTest(model, i[0], i[1], pDict)
    cnt += 1
    r = combTest(model, i[0], i[1], pDict)
    rstList.append(r)
    rstLabel.append(False)
    print cnt
    # if cnt > 10:
    #     break
    print "====="


print rstList
from sklearn import metrics
rst = np.asarray(rstList)
print metrics.roc_auc_score(rstLabel,rst[:,0])
print metrics.roc_auc_score(rstLabel,rst[:,1])
print metrics.roc_auc_score(rstLabel,rst[:,2])
print metrics.roc_auc_score(rstLabel,rst[:,3])
print metrics.roc_auc_score(rstLabel,rst[:,4])
