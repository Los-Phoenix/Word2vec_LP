#coding:UTF-8
#这个文件用tf实现一个单层神经网络，用来判断两个词是否是同义词
#输入是readerX产生的X@200 和Y@1
#划分测试集和监督集、测试集

#使用三层神经网进行训练，n个隐藏层和1个输出层？？
import readerX
import tensorflow as tf
import numpy as np
import gc

data_dim = 200#输入数据的维度

piece = 1
sample_num = piece * 49

xInTemp, yInTemp = readerX.genAllXY()
print sum(yInTemp), len(yInTemp)
yInTemp = [[i,1 - i] for i in yInTemp]
print 'Reader Done'



xIn = np.matrix(xInTemp)
yIn = np.matrix(yInTemp)

del(yInTemp)
del(xInTemp)
gc.collect()
print 'Matrix Prepared'

print yIn.reshape(-1,2)
