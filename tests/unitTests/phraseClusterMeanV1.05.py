#coding:utf8

#从voc中间读出来所有的词
#分成单字词、2-3字词、4字词、5字以上词4个集合

import sys
import random
import gensim
import numpy as np
from numpy import dot, float32 as REAL
from gensim import matutils

import matplotlib.pyplot as plt
import seaborn as sns

import cPickle as pickle
from scipy.cluster.vq import vq,kmeans,whiten
from scipy.stats import mode

reload(sys)
sys.setdefaultencoding('utf-8')


print('Load Start')
f1 = file('../../data/phraseClusterMeanV1.pkl', 'rb')
# pickle.dump(samP, f1, True)
# pickle.dump(samS, f1, True)
# pickle.dump(diffList_s, f1, True)
# pickle.dump(diffList_c, f1, True)
# pickle.dump(diffList_l, f1, True)
# pickle.dump(diffList_w, f1, True)

samP = pickle.load(f1)
samS = pickle.load(f1)
diffList_s = pickle.load(f1)
diffList_c = pickle.load(f1)
diffList_l = pickle.load(f1)
diffList_w = pickle.load(f1)
f1.close()
print('Dump Done')

mats = np.asarray(diffList_s)
matc = np.asarray(diffList_c)
matl = np.asarray(diffList_l)
matw = np.asarray(diffList_w)

# print mats[:,1]
sns.set_style("whitegrid")
plt.plot([-1, 0, 1], [0,0,0])
# sns.kdeplot(np.zeros(len(mats[:,0])), cumulative=True,vertical = True,label = 'zero', color = '#F0F0F0')

sns.kdeplot(mats[:, 0], cumulative=True,vertical = True, label = 's_mean')
# sns.kdeplot(mats[:, 1], cumulative=True,vertical = True, label = 's_compCheat')
sns.kdeplot(mats[:, 2], cumulative=True,vertical = True, label = 's_compMean')
# sns.kdeplot(mats[:, 3], cumulative=True,vertical = True, label = 's_nearCheat')
sns.kdeplot(mats[:, 4], cumulative=True,vertical = True, label = 's_nearMean')
# plt.show()

sns.kdeplot(matc[:, 0], cumulative=True,vertical = True, label = 'c_mean')
# sns.kdeplot(matc[:, 1], cumulative=True,vertical = True, label = 'c_compCheat')
sns.kdeplot(matc[:, 2], cumulative=True,vertical = True, label = 'c_compMean')
# sns.kdeplot(matc[:, 3], cumulative=True,vertical = True, label = 'c_nearCheat')
sns.kdeplot(matc[:, 4], cumulative=True,vertical = True, label = 'c_nearMean')
# plt.show()

# sns.kdeplot(matl[:, 0], cumulative=True,vertical = True, label = 'l_mean')
# sns.kdeplot(matl[:, 1], cumulative=True,vertical = True, label = 'l_compCheat')
# sns.kdeplot(matl[:, 2], cumulative=True,vertical = True, label = 'l_compMean')
# sns.kdeplot(matl[:, 3], cumulative=True,vertical = True, label = 'l_nearCheat')
# sns.kdeplot(matl[:, 4], cumulative=True,vertical = True, label = 'l_nearMean')
# plt.show()

sns.kdeplot(matw[:, 0], cumulative=True,vertical = True, label = 'w_mean')
# sns.kdeplot(matw[:, 1], cumulative=True,vertical = True, label = 'w_compCheat')
sns.kdeplot(matw[:, 2], cumulative=True,vertical = True, label = 'w_compMean')
# sns.kdeplot(matw[:, 3], cumulative=True,vertical = True, label = 'w_nearCheat')
sns.kdeplot(matw[:, 4], cumulative=True,vertical = True, label = 'w_nearMean')
plt.show()


