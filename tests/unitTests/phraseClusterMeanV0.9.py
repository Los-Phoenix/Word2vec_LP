#coding:utf8

#从voc中间读出来所有的词
#分成单字词、2-3字词、4字词、5字以上词4个集合

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

#先看answer at

model = gensim.models.Word2VecWC.load("../../data/wiki_phrase2/wiki_model")

def answerAt(model, word, compList):


    nearest = 9999999
    for comp in compList:
        if comp in model:
            nearest_num_temp = len(model.wv.words_closer_than(comp, word))
            nearest = nearest if nearest < nearest_num_temp else nearest_num_temp
    return nearest

def combTest(model, word, compList, kmeansF = True, nearestF = True, charmeanF = True):#这个函数将一个phrase/word的一份comp的三个参数一起返回。

    compList_q = list()
    # print compList
    for i in compList:
        # print i, i in model
        if i in model:
            compList_q.append(i)
        else:
            print "ALARM!, char", i, "in word not in model"

    compList = compList_q
    if len(compList) == 0:
        return [-1, -1, -1]

    kmeans_method = len(word)

    word_vec = model.wv.word_vec(word, use_norm=True)
    baseline = model.most_similar(word)[0][1]

    charmeanT = 0
    kmeansT = 0
    nearestT = 0

    if baseline < 0:
        print "WOW, Single Boy!"

    char_mean_vec_list = [model.wv.word_vec(char, use_norm=True) for char in compList]
    char_mean_vec = np.array(char_mean_vec_list).mean(axis=0)

    if charmeanF:
        charmeanT = dot(matutils.unitvec(char_mean_vec).astype(REAL), word_vec) - baseline

    sim_words = list()
    for char in compList:
        simrst = model.wv.most_similar(char, topn=1000)
        for sim_word, _ in simrst:
            if sim_word == word:
                continue
            sim_words.append(sim_word)

    #至此sim_words就是根据所有组成词求出来的相似词表
    if nearestF:
        model_list = list(model[sim_words])

        model_list_value = [
            baseline if dot(matutils.unitvec(d), word_vec) >= baseline else dot(matutils.unitvec(d), word_vec)
            for d in model_list]

        nearestT = np.array(model_list_value).max() - baseline
        if nearestT > 0.001:
            print word
            print nearestT, np.array(model_list_value).max(), baseline

    if kmeansF:
        data = whiten(model[sim_words])
        centoids = kmeans(data, kmeans_method)

        labels = vq(data, centoids[0])[0]
        max_label = mode(labels).mode[0]

        kmeansT = dot(matutils.unitvec(centoids[0][max_label]).astype(REAL), word_vec) - baseline

    return [kmeansT, nearestT, charmeanT]

pCnt = 0

p_s_list = list()
p_c_list = list()
p_l_list = list()

s_c_list = list()
#
#
# for phrase in samP:
#     pCnt+=1
#     print pCnt, phrase
#
#     charList = list(phrase)
#     sonList = pDict[phrase]
#
#     leftList = list()
#     for char in charList:
#         leftList.append(char)
#
#     for son in sonList:
#         leftList.append(son)
#         for char in son:
#             if char in leftList:
#                 leftList.remove(char)
#
#     p_c_list.append(answerAt(model, phrase, charList))
#     p_s_list.append(answerAt(model, phrase, sonList))
#     p_l_list.append(answerAt(model, phrase, leftList))
#
#     # for chars in leftList:
#     #     print chars
#     #Here we got those three words
#     #we need a func to caculate nearest according to a given list
# sCnt = 0
# for word in samS: # This is a short word
#     sCnt += 1
#     print sCnt, word
#     charList = list(word)
#     s_c_list.append(answerAt(model, phrase, charList))
#
# mat_1 = np.asarray(p_c_list)
# mat_2 = np.asarray(p_s_list)
# mat_3 = np.asarray(p_l_list)
# mat_4 = np.asarray(s_c_list)
#
# sns.kdeplot(np.log10(mat_1), cumulative=True,vertical = True, label = 'phrase_char',  color = '#000000')
# sns.kdeplot(np.log10(mat_2), cumulative=True,vertical = True, label = 'phrase_son',  color = '#FF0000')
# sns.kdeplot(np.log10(mat_3), cumulative=True,vertical = True, label = 'phrase_left',  color = '#00FF00')
# sns.kdeplot(np.log10(mat_4), cumulative=True,vertical = True, label = 'short_char',  color = '#0000FF')
# plt.show()

#设窗口为1K
# 接下来考察nearest、 k-means 和charmean的情况
#比方说先看

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
f1 = file('../../data/phraseClusterMean.pkl', 'wb')
pickle.dump(samP, f1, True)
pickle.dump(samS, f1, True)
pickle.dump(diffList_s, f1, True)
pickle.dump(diffList_c, f1, True)
pickle.dump(diffList_l, f1, True)
pickle.dump(diffList_w, f1, True)
f1.close()
print('Dump Done')



