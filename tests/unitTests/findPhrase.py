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

print "uniSet", len(uniSet)
print "biTriSet", len(biTriSet)
print "quadSet", len(quadSet)
print "longSet", len(longSet)

wordSet = biTriSet.union(quadSet).union(longSet)

#longSet 中 包含两个以上bitri的才叫词组
#我们先看看有多少个词组：
phraseList = list()
plist = list()
# for p in longSet:
#     pSubList = list()
#     # print p
#     len_p = len(p)
#     for start in xrange(len_p):
#         for length in xrange(len_p - start + 1):
#             if p[start: start+length] in biTriSet:
#                 pSubList.append(p[start: start+length])
#
#     if len(pSubList) > 1:#This phrase has many sub parts
#         phraseList.append(p)
#         plist.append(pSubList)
#
# pDict = dict(zip(phraseList, plist))

#longSet = random.sample(longSet, 20)

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

print len(pDict)
for k in pDict.keys():
    print k
    for s in pDict[k]:
        print "  ", s

