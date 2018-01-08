#coding:utf8

#从voc中间读出来所有的词
#分成单字词、2-3字词、4字词、5字以上词4个集合
#这个文件开始统计带attention的逼近方法
#计算五个attention
#charmean
#comp * charmean
#comp * cheat
#comp_near * charmean
#comp_near * cheat

# 计算四种方式和


import sys
import random
import gensim
import numpy as np
from numpy import dot, float32 as REAL
from gensim import matutils

import matplotlib.pyplot as plt
import seaborn as sns

import cPickle as pickle
from scipy.cluster.vq import vq,kmeans,whiten
from scipy.stats import mode

reload(sys)
sys.setdefaultencoding('utf-8')

voc_file = open('../../data/wiki_phrase2/wiki_voc')
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

wordSet = biTriSet.union(quadSet).union(longSet)

phraseList = list()
plist = list()

for p in longSet:
    pSubList = list()
    # print p
    len_p = len(p)
    for start in xrange(len_p):
        for length in xrange(len_p - start + 1):
            if p[start: start+length] in wordSet and not p[start: start+length] == p:

                pSubList.append(p[start: start+length])

    if len(pSubList) > 1:#This phrase has many sub parts
        phraseList.append(p)
        plist.append(pSubList)

pDict = dict(zip(phraseList, plist))

pop_num = 1000
samP = random.sample(pDict.keys(), pop_num)
samS = random.sample(biTriSet, pop_num)

model = gensim.models.Word2VecWC.load("../../data/wiki_phrase2/wiki_model")

def combTest(model, word, compList, cheatF = True, noCheatF = True, charmeanF = True):
    # 这个函数将一个phrase/word的一份comp的三个参数一起返回。
    # 我们依然设置两个FLAG 以防万一
    #cheat 和no cheat

    #returns:

    # charmean
    # comp * charmean
    # comp * cheat
    # comp_near * charmean
    # comp_near * cheat

    compList_q = list()
    #print compList
    for i in compList:
        # print i, i in model
        if i in model:
            compList_q.append(i)
        else:
            print "ALARM!, char", i, "in word not in model"

    compList = compList_q

    kmeans_method = len(word)

    word_vec = model.wv.word_vec(word, use_norm=True)
    baseline = model.most_similar(word)[0][1]

    compCheatT = 0
    compMeanT = 0
    nearCheatT = 0
    nearMeanT = 0

    if baseline < 0:
        print "WOW, Single Boy!"

    char_mean_vec_list = [model.wv.word_vec(char, use_norm=True) for char in compList]
    char_mean_vec = np.array(char_mean_vec_list).mean(axis=0)


    charmeanT = dot(matutils.unitvec(char_mean_vec).astype(REAL), word_vec) - baseline

    sim_words = list()
    for char in compList:
        simrst = model.wv.most_similar(char, topn=1000)
        for sim_word, _ in simrst:
            if sim_word == word:
                continue
            sim_words.append(sim_word)

    #至此sim_words就是根据所有组成词求出来的相似词表

    if cheatF:
        near_list = list(model[sim_words])
        near_sim_list = [model.wv.similarity(i, word) for i in sim_words]
        nearCheatVec = np.average(np.array(near_list), axis=0, weights = near_sim_list)
        nearCheatT = dot(matutils.unitvec(nearCheatVec).astype(REAL), word_vec) - baseline

        comp_list = list(model[compList])
        comp_sim_list = [model.wv.similarity(i, word) for i in compList]
        compCheatVec = np.average(np.array(comp_list), axis=0, weights = comp_sim_list)
        compCheatT = dot(matutils.unitvec(compCheatVec).astype(REAL), word_vec) - baseline

    if noCheatF:
        near_list = list(model[sim_words])
        near_sim_list = [dot(matutils.unitvec(char_mean_vec).astype(REAL), np.asarray(i)) for i in near_list]
        nearMeanVec = np.average(np.array(near_list), axis=0, weights = near_sim_list)
        nearMeanT = dot(matutils.unitvec(nearMeanVec).astype(REAL), word_vec) - baseline

        comp_list = list(model[compList])
        comp_sim_list = [dot(matutils.unitvec(char_mean_vec).astype(REAL), np.asarray(i)) for i in comp_list]
        compMeanVec = np.average(np.array(comp_list), axis=0, weights = comp_sim_list)
        compMeanT = dot(matutils.unitvec(compMeanVec).astype(REAL), word_vec) - baseline

    return [charmeanT, compCheatT, compMeanT, nearCheatT, nearMeanT]

pCnt = 0

diffList_c = list()
diffList_s = list()
diffList_l = list()
model.init_sims()
for phrase in samP:
    pCnt+=1
    print pCnt, phrase

    charList = list(phrase)
    sonList = pDict[phrase]

    leftList = list()
    for char in charList:
        leftList.append(char)

    for son in sonList:
        leftList.append(son)
        for char in son:
            if char in leftList:
                leftList.remove(char)

    diffList_s.append(combTest(model, phrase, sonList))
    diffList_c.append(combTest(model, phrase, charList))
    diffList_l.append(combTest(model, phrase, leftList))

print diffList_s
print diffList_c
print diffList_l

diffList_w = list()
sCnt = 0
for word in samS: # This is a short word
    sCnt += 1
    print sCnt, word
    charList = list(word)

    diffList_w.append(combTest(model, word, charList))

print diffList_w

print('Dump Start')
f1 = file('../../data/phraseClusterMeanV1.pkl', 'wb')
pickle.dump(samP, f1, True)
pickle.dump(samS, f1, True)
pickle.dump(diffList_s, f1, True)
pickle.dump(diffList_c, f1, True)
pickle.dump(diffList_l, f1, True)
pickle.dump(diffList_w, f1, True)
f1.close()
print('Dump Done')



