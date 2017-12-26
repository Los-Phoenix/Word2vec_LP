#coding:UTF8
import gensim
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

reload(sys)
sys.setdefaultencoding('utf-8')

model = gensim.models.Word2VecWC.load("../../data/novel/novel2_model")

fPos = open("../../data/woodWikiTestPos")
linesPos = list(fPos)

fDiff = open("../../data/woodWikiTestDiffClass")
linesDiff = list(fDiff)

posList = list()
negList = list()

simListPos = list()
simListNeg = list()

cnt = 0
num = 9999999

for line in linesPos:
    title, w1, w2 = line.split('\t')
    w1 = w1.decode()
    w2 = w2.strip().decode()
    if w1 in model and w2 in model:
        posList.append([w1, w2])
        simListPos.append(model.similarity(w1, w2))
        cnt += 1
    else:
        print
    if cnt > num * 3:
        break
    if cnt % 1000 == 0:
        print cnt

for line in linesDiff:
    title, w1, w2 = line.split('\t')
    w1 = w1.decode()
    w2 = w2.strip().decode()

    if w1 in model and w2 in model:
        negList.append([w1, w2])
        simListNeg.append(model.similarity(w1, w2))
        cnt += 1

    if cnt > num * 6:
        break
    if cnt % 1000 == 0:
        print cnt


def test():
    #每到训练次数进行一下测试：
    simListPos = list()
    simListNeg = list()
    for w1, w2 in posList:
        simListPos.append(model.similarity(w1, w2))

    for w1, w2 in negList:
        simListNeg.append(model.similarity(w1, w2))

    plt.hist(np.asarray(simListPos), color="#FF0000", alpha=.5)
    plt.hist(np.asarray(simListNeg), color="#0000FF", alpha=.5)
    plt.show()




# model.pushpull([[u'人', u'人民']], [[u'人', u'拖拉机']])
for i in xrange(1000):
    model.pushpull(posList, [])
    if i %100 == 0 or i in [1,5,10,50]:
        # print i
        test()

print len(simListPos), len(simListNeg)




def test():
    #每到训练次数进行一下测试：
    simListPos = list()
    simListNeg = list()
    for w1, w2 in posList:
        simListPos.append(model.similarity(w1, w2))

    for w1, w2 in negList:
        simListNeg.append(model.similarity(w1, w2))

    plt.hist(np.asarray(simListPos), color="#FF0000", alpha=.5)
    plt.hist(np.asarray(simListNeg), color="#0000FF", alpha=.5)
    plt.show()


