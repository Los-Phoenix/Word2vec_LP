#coding:UTF-8
#这个文件用tf实现一个单层神经网络，用来判断两个词是否是同义词
#先做一个分类的任务，输入是虚拟的
#暂时的输入是比较大小，200维输入，1000个样例

#数据准备：
#读取词模型、字模型、词典

#写一个函数叫genBatch
#X是一个每一行是一个字的矩阵
#Y是一个每一行是一个词的行

import tensorflow as tf
import numpy as np

sample_num = 2000
test_num = 200

data_dim = 28*28
out_dim = 5
xIn = np.random.uniform(0, 1, [sample_num+test_num, data_dim])#这是个假的输入
W = np.random.randint(0, 3, [data_dim, out_dim])
# print W
xIn2prev = xIn.dot(W)
# xIn2prev = xIn * W
xIn2 = np.mat(xIn2prev)
# xIn3 = xIn2.reshape((-1, 1))/50
#那这个就是假的输出了
# x2In =
# print xIn2
# print yIn
# print tf.Session().run(xIn[0:, 1:2])
x = tf.placeholder(tf.float32, shape = (None, data_dim))
y = tf.placeholder(tf.float32, shape = (None, out_dim))
x_image = tf.reshape(x, [-1, 28, 28, 1])
num = 10
with tf.variable_scope("Ez_flat"):
    W1 = tf.Variable(tf.truncated_normal([28,5,1,10], stddev = 0.1), 'weight1', dtype=tf.float32)
    b1 = tf.Variable(np.random.rand(1, 10), 'bias1', dtype=tf.float32)

    conv1 = tf.nn.conv2d(x_image, W1, strides=[1,1,1,1], padding='SAME') + b1

    # h_conv1 = tf.nn.relu(conv1)
    h_conv1 = conv1
    pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    W2 = tf.Variable(tf.truncated_normal([14, 5, 10, 8], stddev=0.1), 'weight1', dtype=tf.float32)
    b2 = tf.Variable(np.random.rand(1, 8), 'bias1', dtype=tf.float32)

    conv2 = tf.nn.conv2d(pool1, W2, strides=[1, 1, 1, 1], padding='SAME') + b2

    # h_conv2 = tf.nn.relu(conv2)
    h_conv2 = conv2
    pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    #以下7*7的意思是输入为28*28， 经过两次2*2的maxpool，得到了
    W3 = tf.Variable(tf.truncated_normal([7*7*8, out_dim], stddev=0.1), 'weight1', dtype=tf.float32)
    # b3 = tf.Variable(np.random.rand(1, out_dim), 'bias1', dtype=tf.float32)

    L2_out = tf.matmul(tf.reshape(pool2, [-1, 7*7*8]), W3)# + b3

    loss = tf.reduce_sum((y - L2_out) ** 2)
    # loss = tf.reduce_mean(loss)

with tf.name_scope("training-accuracy") as scope:
    correct_prediction = (L2_out-y)**2
    train_accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    train_accuracy_summary = tf.summary.scalar("training accuracy", train_accuracy)

opt = tf.train.AdamOptimizer(0.1)
train_op = opt.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for cnt in xrange(500000):
        randList = np.random.randint(0, sample_num, size=(1, 50))
        # inputSam = random.sample(inputLong,577)
        # print inputSam
        xSam = xIn[randList, :]
        ySam = xIn2[randList, :]
        _, loss_val, W1_val, b1_val = sess.run([train_op, loss, W1, b1],
                                               feed_dict={x: xSam.reshape(-1, data_dim),
                                                          y: ySam.reshape(-1, out_dim)})


        if cnt%100 == 0:#!!!!!!!!!!!在这里！！！！！！！！
            accu = train_accuracy.eval(feed_dict={x: xIn[:sample_num],
                                                  y: xIn2[:sample_num]})
            accu_test = train_accuracy.eval(feed_dict={x: xIn[sample_num:],
                                                       y: xIn2[sample_num:]})
            # print W1_val
            print '#' * 20
            print loss_val
            print accu
            print accu_test
            print cnt



