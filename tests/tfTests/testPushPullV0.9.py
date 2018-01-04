#coding:UTF8
import gensim
import sys
import numpy as np
import time
import matplotlib
#matplotlib.use('Agg')


import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
import seaborn as sns
import random
import logging
import os

import picklePushPull

program = os.path.basename(sys.argv[0])

logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)

reload(sys)
sys.setdefaultencoding('utf-8')

#====================END OF LOADING=======================#
batchsize = 640
th = 50

model = gensim.models.Word2VecWC.load("../../data/novel/novelS_model")

posList, simDict, negList = picklePushPull.loadAll()
posNum = len(posList)
negNum = len(negList)

def test(i):
    #每到训练次数进行一下测试：
    simListPos = list()
    simListNeg = list()
    for w1, w2 in random.sample(posSample, 10000):
        simListPos.append(model.similarity(w1, w2))

    for w1, w2 in negList:
        simListNeg.append(model.similarity(w1, w2))

    print simListPos[0:batchsize]
    print simListNeg[0:batchsize]

    plt.clf()
    plt.hist(np.asarray(simListPos), color="#FF0000", alpha=.5)
    plt.hist(np.asarray(simListNeg), color="#0000FF", alpha=.5)
    fileName = "figK" + str(i) + ".png"
    # savefig("thisfig.png")
    savefig(fileName)


def test240(i):
#this fuction reads 240.txt and test coef between sim and score
    fSimDict240 = list(open("../../data/240.txt"))
    fSimDict297 = list(open("../../data/297.txt"))

    list240 = list()
    list297 = list()
    listsim240 = list()
    listsim297 = list()

    for q in fSimDict240:
        w0, w1, sim = q.decode().split('\t')
        if w0 in model and w1 in model:
            list240.append(float(sim.strip()))
            listsim240.append(model.wv.similarity(w0, w1)/(1.))
            # listsim240.append(model.wv.similarity(w0, w1)/(simDict[w0]*simDict[w1]))

    for q in fSimDict297:
        w0, w1, sim = q.decode().split('\t')
        if w0 in model and w1 in model:
            list297.append(float(sim.strip()))
            listsim297.append(model.wv.similarity(w0, w1)/(1.))
            # listsim297.append(model.wv.similarity(w0, w1)/(simDict[w0]*simDict[w1]))

    a = np.array([list240, listsim240])
    b = np.array([list297, listsim297])

    corr240 = np.corrcoef(a)
    corr297 = np.corrcoef(b)

    print"Corr at i:", i
    print corr240[0][1]
    print corr297[0][1]

    corrY240.append(corr240[0][1])
    corrY297.append(corr297[0][1])
    corrX.append(i)
# posSample = random.sample(posList, samplesize)
# negSample = random.sample(negList, samplesize)
#下面是全量训练
posSample = posList
negSample = negList

corrY240 = list()
corrY297 = list()
tPct = list()
corrX = list()

accuLow = 0
for i in xrange(10000):
    model.wv.init_sims()
    t = time.time()
    #print "Train No.", i
    posBatch = random.sample(posSample, batchsize)
    negBatch = random.sample(negSample, batchsize)
    trainSizePos = model.pushpullCC(posBatch, negBatch, simDict, sample_size = 800, alpha = 0.01, alpha2 = 0.0002)
    #print "used:", time.time() - t, "Seconds"
    if i %50 == 0 or i < 10:
    # if True:
        print "Trained", trainSizePos, "out of", batchsize
        tPct.append(0.02 * float(trainSizePos)/float(batchsize) + 0.55)

        test240(i)
        # test(i)
        if trainSizePos < th:
            accuLow += 1
            if accuLow == 5:
                break
        else:
            accuLow = 0

    if i == 499:
        model.save("../../data/novel/novelS_model_500")
    if i == 999:
        model.save("../../data/novel/novelS_model_1000")

print corrX
print corrY240
print corrY297
print tPct

plt.plot(np.asarray(corrX), np.asarray(corrY240), marker='o', mec='r')
plt.plot(np.asarray(corrX), np.asarray(corrY297), marker='x', mec='b')
plt.plot(np.asarray(corrX), np.asarray(tPct), marker='*', mec='g')
plt.show()



