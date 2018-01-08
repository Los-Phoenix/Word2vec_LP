#coding:utf8

#从voc中间读出来所有的词
#分成单字词、2-3字词、4字词、5字以上词4个集合

import sys
import random
import gensim
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

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
samS.append(u"汉和帝")
#先看answer at

model = gensim.models.Word2VecWC.load("../../data/wiki_phrase2/wiki_model")

def answerAt(model, word, compList):
    nearest = 9999999
    for comp in compList:
        if comp in model:
            nearest_num_temp = len(model.wv.words_closer_than(comp, word))
            nearest = nearest if nearest < nearest_num_temp else nearest_num_temp
    return nearest


pCnt = 0

p_s_list = list()
p_c_list = list()
p_l_list = list()

s_c_list = list()


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

    p_c_list.append(answerAt(model, phrase, charList))
    p_s_list.append(answerAt(model, phrase, sonList))
    p_l_list.append(answerAt(model, phrase, leftList))

    # for chars in leftList:
    #     print chars
    #Here we got those three words
    #we need a func to caculate nearest according to a given list
sCnt = 0
for word in samS: # This is a short word
    sCnt += 1
    print sCnt, word
    charList = list(word)
    s_c_list.append(answerAt(model, phrase, charList))

mat_1 = np.asarray(p_c_list)
mat_2 = np.asarray(p_s_list)
mat_3 = np.asarray(p_l_list)
mat_4 = np.asarray(s_c_list)

sns.kdeplot(np.log10(mat_1), cumulative=True,vertical = True, label = 'phrase_char',  color = '#000000')
sns.kdeplot(np.log10(mat_2), cumulative=True,vertical = True, label = 'phrase_son',  color = '#FF0000')
sns.kdeplot(np.log10(mat_3), cumulative=True,vertical = True, label = 'phrase_left',  color = '#00FF00')
sns.kdeplot(np.log10(mat_4), cumulative=True,vertical = True, label = 'short_char',  color = '#0000FF')
plt.show()






