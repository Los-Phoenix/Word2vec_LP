#coding:UTF8
#这个文档
import gensim
import sys
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')

import random
import logging
import os
import cPickle as pickle

program = os.path.basename(sys.argv[0])

logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)

reload(sys)
sys.setdefaultencoding('utf-8')
if __name__ == '__main__':
    model = gensim.models.Word2VecWC.load("../../data/novel/novelS_model")
    # model = gensim.models.Word2VecWC(model)

    fPos = open("../../data/Same2")
    linesPos = list(fPos)

    fDiff = open("../../data/woodWikiTestDiffClass1000")
    linesDiff = list(fDiff)

    posList = list()
    negList = list()

    simListPos = list()
    simListNeg = list()

    cnt = 0
    num = 1000000
    # num = 1000
    negNum = 10000
    samplesize = 3 * num
    batchsize = 1024
    posSet = set()
    negSet = set()

    for line in linesPos:
        title, w1, w2 = line.split('\t')
        w1 = w1.decode()
        w2 = w2.strip().decode()
        if w1 in model and w2 in model:
            posSet.add(w1)
            posSet.add(w2)
            posList.append([w1, w2])
            simListPos.append(model.similarity(w1, w2))
            cnt += 1
        else:
            print
        if cnt > num * 3:
            break
        if cnt % 1000 == 0:
            print cnt

    print"posSet:", len(posSet)

    for line in linesDiff:
        title, w1, w2 = line.split('\t')
        w1 = w1.decode()
        w2 = w2.strip().decode()

        if w1 in model and w2 in model:
            negSet.add(w1)
            negSet.add(w2)
            negList.append([w1, w2])
            simListNeg.append(model.similarity(w1, w2))
            cnt += 1

        if cnt > num * 6:
            break
        if cnt % 1000 == 0:
            print "Neg:", cnt

    print"negSet:",len(negSet)

    posWordList = list()
    posValue = list()

    simSet = set()
    fSimDict240 = list(open("../../data/240.txt"))
    fSimDict297 = list(open("../../data/297.txt"))
    for i in fSimDict240:
        w0, w1, sim = i.decode().split('\t')
        if w0 in model and w1 in model:
            simSet.add(w0)
            simSet.add(w1)
    for i in fSimDict297:
        w0, w1, sim = i.decode().split('\t')
        if w0 in model and w1 in model:
            simSet.add(w0)
            simSet.add(w1)
    print "simSet:", len(simSet)
    print "interSection:", len(simSet.intersection(posSet))
    posSet = posSet.union(simSet)
    print "New posSet", len(posSet)
    cnt = 0
    for i in posSet:
        if cnt % 1000 == 0:
            print "Found words", cnt
        posWordList.append(i)
        posValue.append(model.most_similar(i)[0][1])
        cnt += 1

    postDict = dict(zip(posWordList, posValue))

    #print postDict

    print('Dump Start')
    f1 = file('../../data/CombDataN.pkl', 'wb')
    pickle.dump(posList, f1, True)
    pickle.dump(postDict, f1, True)
    pickle.dump(negList, f1, True)
    f1.close()
    print('Dump Done')

def loadAll():
    #returns vocab, embd, y, x, x_other
    print "load pushPull start"
    f = file('../../data/CombDataN.pkl', 'rb')
    posList = pickle.load(f)
    posDict = pickle.load(f)
    negList = pickle.load(f)
    f.close()
    print "load pushPull done!"
    return posList, posDict, negList





