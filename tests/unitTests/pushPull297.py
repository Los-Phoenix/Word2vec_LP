#coding:utf-8
#这个文件是为了：
#读出SIM297
#转换成测试集
#计算协方差
#计算皮尔逊相关系数
import sys, getopt
import gensim
import numpy as np

reload(sys)
sys.setdefaultencoding('utf-8')

fSimDict297 = list(open("../../data/297.txt"))

model0 = gensim.models.Word2VecWC.load("../../data/novel/novelS_model")
model500 = gensim.models.Word2VecWC.load("../../data/novel/novelS_model_500")
model1000 = gensim.models.Word2VecWC.load("../../data/novel/novelS_model_1000")

listWords = list()
list297 = list()
listsim0 = list()
listsim500 = list()
listsim1000 = list()

for i in fSimDict297:

    w0, w1, sim = i.decode().split('\t')

    if w0 in model0 and w1 in model0:
        listWords.append([w0, w1])
        list297.append(float(sim.strip()))
        listsim0.append(model0.wv.similarity(w0, w1))
        listsim500.append(model500.wv.similarity(w0, w1))
        listsim1000.append(model1000.wv.similarity(w0, w1))

import json

result = json.dumps(listWords, encoding='UTF-8', ensure_ascii=False)

print result
print list297
print listsim0
print listsim500
print listsim1000

for i in xrange(len(listWords)):
    print "\t".join(["\t".join(listWords[i]), str(list297[i]), str(listsim0[i]), str(listsim500[i]), str(listsim1000[i])])

a = np.array([list297, listsim0, listsim500, listsim1000])
print a



corr = np.corrcoef(a)
print corr

