#coding:utf8

#从voc中间读出来所有的词
#分成单字词、2-3字词、4字词、5字以上词4个集合

import sys
import random

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

pop_num = 2000

samP = random.sample(pDict.keys(), pop_num)
samS = random.sample(biTriSet, pop_num)
#先看answer at
for phrase in samP:
    print phrase

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

    # for chars in leftList:
    #     print chars







