#coding:utf-8
#这个函数从三份词表当中生成map
#再把map写成文件
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
fCha = open("../../data/wiki_cha/zh_model_200_all_voc_cha")
linesCha = list(fCha)
chaSet =set([i.split(' ')[0].decode() for i in linesCha])

fWord = open("../../data/wiki_word/zh_model_200_all_voc")
linesWord = list(fWord)
wordSet =set([i.split(' ')[0].decode() for i in linesWord])

fDictOut = open("../../data/interDictCharWord", 'w')

interSet = chaSet.intersection(wordSet)
multiWordSet = wordSet.difference(chaSet)
multiWordCharSet = set([])

for word in multiWordSet:
    for cha in word:
        multiWordCharSet.add(cha)

interSet = interSet.intersection(multiWordCharSet)

print 'setUnion:',len(interSet)

cnt = 0
for word in interSet:
    cnt += 1
    print cnt
    fDictOut.write(word+'\n')

    #那么，词都在字字典里的有多少呢？
