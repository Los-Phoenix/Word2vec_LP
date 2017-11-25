#coding:utf-8
#这个函数从三份词表当中生成map
#再把map写成文件
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

fPos = open("../../data/woodWikiTestPos1000")
linesPos = list(fPos)
fConf = open("../../data/woodWikiTestConfused1000")
linesConf = list(fConf)
fDiff = open("../../data/woodWikiTestDiffClass1000")
linesDiff = list(fDiff)

fDictOut = open("../../data/unionDict1000", 'w')

wordsPos = [i.split('\t')[1].decode() for i in linesPos]
print len(wordsPos)
posSet = set(wordsPos)
print len(posSet)

wordsConf = [i.split('\t')[2].strip().decode() for i in linesConf]
print len(wordsConf)
confSet = set(wordsConf)
print len(confSet)

wordsDiff = [i.split('\t')[2].strip().decode() for i in linesDiff]
print len(wordsDiff)
diffSet = set(wordsDiff)
print len(diffSet)

unionSet = posSet.union(confSet).union(diffSet)
print len(unionSet)

for i in unionSet:
    print i