#coding:UTF8
import gensim
import sys
from myWord2vec import Word2VecWC

reload(sys)
sys.setdefaultencoding('utf-8')

model = gensim.models.Word2VecWC.load("../data/wikiDummy/Dummy_model")

fPos = open("../../data/woodWikiTestPos1000")
linesPos = list(fPos)
fConf = open("../../data/woodWikiTestDiffClass1000")
linesConf = list(fConf)
fDiff = open("../../data/woodWikiTestConfused1000")
linesDiff = list(fDiff)

posList = list()

cnt = 0
for line in linesPos:
    title, w1, w2 = line.split('\t')
    w1 = w1.decode()
    w2 = w2.strip().decode()

    xy.append(list(model[w1]) + list(model[w2]) + [1])

    cnt += 1
    if cnt > num * 3:
        break
    if cnt % 1000 == 0:
        print cnt

for line in linesConf:
    title, w1, w2 = line.split('\t')
    w1 = w1.decode()
    w2 = w2.strip().decode()
    # print title
    # print w1
    # print w2
    # print len(list(model[w1]) + list(model[w2]) + [0])

    xy.append(list(model[w1]) + list(model[w2]) + [0])

    cnt += 1
    if cnt > num * 9:
        break
    if cnt % 1000 == 0:
        print cnt

for line in linesDiff:
    title, w1, w2 = line.split('\t')
    w1 = w1.decode()
    w2 = w2.strip().decode()
    # print title
    # print w1
    # print w2
    # print len(list(model[w1]) + list(model[w2]) + [0])
    # if len(xy) > num * 30:
    #     break
    xy.append(list(model[w1]) + list(model[w2]) + [0])

    cnt += 1
    if cnt > num * 12:
        break
    if cnt % 1000 == 0:
        print cnt

#
# charList = list(open("../../data/wiki_cha/zh_model_200_all_voc_cha"))
# charClearList = [i.split(" ")[0] for i in charList]
# charSet = set(charClearList)
# for char in charSet:
#     char = char.decode()#这里注意，只有Unicode编码才行
#     print len(model[char])

# i = u"这是一个贤良之人"
# for j in i :
#     print j, model[j]

# model2 = Word2VecWC.load("../data/wiki_word/zh_model_200_all")
# print model2[u'贤良']
model.pushpull([[u'人', u'人民']], [[u'人', u'拖拉机']])