#coding:utf-8
#最小二乘法拟合，用y=ax+b  a=weight b=biases
from __future__ import print_function
import tensorflow as tf
import numpy as np

# create data 生成100个0-1之间的随机数   np.random.rand(100) 1*100的矩阵
#np.random.rand(3,3) 3*3的矩阵，其每个元素为0-1的随机数
x_data = np.random.rand(100, 3).astype(np.float32)
y_data = [0.2, 0.4, 0.3] * x_data + 0.5

### create tensorflow structure start ###对权进行赋值 在-1到一之间随机数
#uniform([1]为1*1的矩阵，即一个数
Weights = tf.Variable(tf.random_uniform([1, 3], -1.0, 1.0))
#偏差为零，zeros([1]为一个1*1的零矩阵，即初始偏差为零
biases = tf.Variable(tf.zeros([1, 3]))
#权值与x相乘并加偏差
y = Weights*x_data + biases
#方差，(y-y_data)平方，求和，取均值
loss = tf.reduce_mean(tf.square(y-y_data))
#定义梯度下降法优化函数，优化，步长为0.5
optimizer = tf.train.GradientDescentOptimizer(0.2)


train = optimizer.minimize(loss)


init = tf.initialize_all_variables()
### create tensorflow structure end ###


sess = tf.Session()
sess.run(init)          # Very important


for step in range(3000):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases))