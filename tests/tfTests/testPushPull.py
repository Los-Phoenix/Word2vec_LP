#coding:UTF8
import gensim
import sys
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')


import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
import seaborn as sns
import random
import logging
import os

program = os.path.basename(sys.argv[0])

logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)

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
num = 40000
samplesize = 2 * num
batchsize = 512

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


def test(i):
    #每到训练次数进行一下测试：
    simListPos = list()
    simListNeg = list()
    for w1, w2 in posList:
        simListPos.append(model.similarity(w1, w2))

    for w1, w2 in negList:
        simListNeg.append(model.similarity(w1, w2))

    print simListPos[0:batchsize]
    print simListNeg[0:batchsize]

    plt.clf()
    plt.hist(np.asarray(simListPos), color="#FF0000", alpha=.5)
    plt.hist(np.asarray(simListNeg), color="#0000FF", alpha=.5)
    fileName = "figt" + str(i) + ".png"
    # savefig("thisfig.png")
    savefig(fileName)

posSample = random.sample(posList, samplesize)
negSample = random.sample(negList, samplesize)

model.wv.init_sims()
for i in xrange(5):
    t = time.time()
    print "Train No.", i
    posBatch = random.sample(posSample, batchsize)
    negBatch = random.sample(negSample, batchsize)
    # model.pushpull(posBatch, negBatch, sample_size = 500, alpha = 0.001)
    model.pushpullCC(posBatch, [], sample_size = 50000, alpha = 0.01)
    # model.pushpull(posBatch, negBatch, sample_size = 500, alpha = 0.001)
    print "used:", time.time() - t, "Seconds"
    if i %100 == 0 or i < 10:
        test(i)

print len(simListPos), len(simListNeg)

model = model.save("../../data/novel/novel2_model_pushpull_savetest")






