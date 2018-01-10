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

for i in longHitSet:
    print i[0], i[1]
