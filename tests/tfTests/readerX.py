#coding:utf8
#这个函数最终解决的是X和对应的Y
#方法是构造201维的List，利用shuffle，再返回就行了

#请一步到位地写好
#返回值是：XList，YList，然后你自己切分去吧
import gensim
import sys
import numpy as np
import random
import gc

reload(sys)
sys.setdefaultencoding('utf-8')

model = gensim.models.Word2Vec.load("../../v1/zh_model_200_all")
fPos = open("../../data/woodWikiTestPos1000")
linesPos = list(fPos)
fConf = open("../../data/woodWikiTestDiffClass1000")
linesConf = list(fConf)
fDiff = open("../../data/woodWikiTestConfused1000")
linesDiff = list(fDiff)

def genXY(num):#传进来一个数就行，按着这个数产生列表
    model = gensim.models.Word2Vec.load("../../v1/zh_model_200_all")
    # if num > len(linesConf) + len(linesDiff) + len(linesPos):
    #     print 'NOT ENOUGH DATA!!!!'
    #     return [],[]

    xy = list()

    cnt = 0
    for line in linesPos:
        title, w1, w2 = line.split('\t')
        w1 = w1.decode()
        w2 = w2.strip().decode()
        # print title
        # print w1
        # print w2
        # print len(list(model[w1]) + list(model[w2]) + [1])
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
    del(model)
    gc.collect()
    randXY = random.sample(xy, num)
    print 'XY Ready!'
    return [i[:200]for i in randXY], [i[200]for i in randXY]


def genAllXY():#传进来一个数就行，按着这个数产生列表
    model = gensim.models.Word2Vec.load("../../v1/zh_model_200_all")
    # if num > len(linesConf) + len(linesDiff) + len(linesPos):
    #     print 'NOT ENOUGH DATA!!!!'
    #     return [],[]

    xy = list()

    cnt = 0
    for line in linesPos:
        title, w1, w2 = line.split('\t')
        w1 = w1.decode()
        w2 = w2.strip().decode()
        # print title
        # print w1
        # print w2
        # print len(list(model[w1]) + list(model[w2]) + [1])
        xy.append(list(model[w1]) + list(model[w2]) + [1])

        cnt += 1
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
        if cnt % 1000 == 0:
            print cnt

    del(model)
    gc.collect()
    randXY = random.sample(xy, len(xy))
    print 'XY Ready!'
    return [i[:200]for i in randXY], [i[200]for i in randXY]




# a, b = genXY(5)
# for i in a:
#     print len(i)
# for i in b:
#     print i

