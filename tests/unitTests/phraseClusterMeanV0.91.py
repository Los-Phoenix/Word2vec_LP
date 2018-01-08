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
f1 = file('../../data/phraseClusterMean.pkl', 'rb')
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
with sns.color_palette(sns.cubehelix_palette(8, start=.5, rot=-.75)):
    sns.kdeplot(mats[:, 0], cumulative=True,vertical = True, label = 's_kmeans')
    # sns.kdeplot(mats[:, 1], cumulative=True,vertical = True, label = 's_nearest')
    sns.kdeplot(mats[:, 2], cumulative=True,vertical = True, label = 's_mean')
with sns.color_palette("Blues"):
    sns.kdeplot(matc[:, 0], cumulative=True,vertical = True, label = 'c_kmeans')#,  color = '#00FF00')
    # sns.kdeplot(matc[:, 1], cumulative=True,vertical = True, label = 'c_nearest')#,  color = '#08FF08')
    sns.kdeplot(matc[:, 2], cumulative=True,vertical = True, label = 'c_mean')#,  color = '#0FFF0F')
# with sns.color_palette("Greens"):
#     sns.kdeplot(matl[:, 0], cumulative=True,vertical = True, label = 'l_kmeans')#,  color = '#FF0000')
#     sns.kdeplot(matl[:, 1], cumulative=True,vertical = True, label = 'l_nearest')#,  color = '#FF0808')
#     sns.kdeplot(matl[:, 2], cumulative=True,vertical = True, label = 'l_mean')#,  color = '#FF0F0F')
with sns.color_palette("Oranges"):
    sns.kdeplot(matw[:, 0], cumulative=True,vertical = True, label = 'w_kmeans')#,  color = '#000000')
    # sns.kdeplot(matw[:, 1], cumulative=True,vertical = True, label = 'w_nearest')#,  color = '#0F0F0F')
    sns.kdeplot(matw[:, 2], cumulative=True,vertical = True, label = 'w_mean')#,  color = '#404040')
plt.show()


