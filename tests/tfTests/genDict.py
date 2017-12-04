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

fCha = open("../../data/wiki_cha/zh_model_200_all_voc_cha")
linesCha = list(fCha)
chaSet =set([i.split(' ')[0].decode() for i in linesCha])

fDictOut = open("../../data/unionDict1000", 'w')

wordsPos = [i.split('\t')[1].decode() for i in linesPos]
print 'listPos:',len(wordsPos)
posSet = set(wordsPos)
print 'setPos:',len(posSet)

wordsConf = [i.split('\t')[2].strip().decode() for i in linesConf]
print 'listConf:',len(wordsConf)
confSet = set(wordsConf)
print 'setConf:',len(confSet)

wordsDiff = [i.split('\t')[2].strip().decode() for i in linesDiff]
print 'listDiff:',len(wordsDiff)
diffSet = set(wordsDiff)
print 'setDiff:',len(diffSet)

unionSet = posSet.union(diffSet)
print 'setUnion:',len(unionSet)

legalSet = set([])
legalCharSet = set([])
for word in unionSet:
    if all(i in chaSet for i in word):
        legalCharSet = legalCharSet.union(set([char for char in word]))
        fDictOut.write(word+'\n')
        legalSet.add(word)

print 'setLegal:',len(legalSet)
print 'setCharLegal:', len(legalCharSet)
    #那么，词都在字字典里的有多少呢？
