#coding:UTF-8
#这个文件用tf实现一个单层神经网络，用来判断两个词是否是同义词
#输入是readerX产生的X@200 和Y@1
#划分测试集和监督集、测试集

#使用三层神经网进行训练，n个隐藏层和1个输出层？？
import readerX
import tensorflow as tf
import numpy as np
import random
import gc

data_dim = 200#输入数据的维度

# piece = 999999
# sample_num = piece * 49

xInTemp, yInTemp = readerX.genAllXY()
yInTemp = [[i,1 - i] for i in yInTemp]
print 'Reader Done'

xIn = np.matrix(xInTemp)
yIn = np.matrix(yInTemp)
sample_num = len(yIn) - 1000
print 'Sample size is : ',sample_num

del(yInTemp)
del(xInTemp)
gc.collect()
print 'Matrix Prepared'

x = tf.placeholder(tf.float32, shape = (None, data_dim))
y = tf.placeholder(tf.float32, shape = (None, 2))

num = 30
num2 = 20
print  num, num2
with tf.variable_scope("Ez_flat"):
    W1 = tf.Variable(np.random.rand(data_dim, num), 'weight1', dtype=tf.float32)
    b1 = tf.Variable(np.random.rand(1, num), 'bias1', dtype=tf.float32)
    # b1 = tf.Variable(np.random.rand(1, num), 'bias1', dtype=tf.float32)
    L1_in = tf.matmul(x, W1) + b1
    #L1_out = tf.nn.softmax(L1_in)
    L1_out = tf.nn.sigmoid(L1_in)
    #
    W2 = tf.Variable(np.random.rand(num, num2), 'weight2', dtype=tf.float32)
    b2 = tf.Variable(np.random.rand(1, num2), 'bias2', dtype=tf.float32)
    #
    L2_in = tf.matmul(L1_out, W2) + b2
    L2_out = tf.nn.sigmoid(L2_in)

    W3 = tf.Variable(np.random.rand(num2, 2), 'weight2', dtype=tf.float32)
    # b3 = tf.Variable(np.random.rand(1, 2), 'bias2', dtype=tf.float32)
    Lf_in = tf.matmul(L2_out, W3)
    Lf_out = tf.nn.softmax(Lf_in)

    # rst = L2_out * 50 + 50
    # loss = tf.contrib.losses.mean_squared_error(L1_out, y)
    # loss = tf.contrib.losses.mean_squared_error(L1_out, y)
    loss = -tf.reduce_sum(y * tf.log(Lf_out))
    # loss = tf.reduce_mean(loss)

with tf.name_scope("training-accuracy") as scope:
    correct_prediction = tf.equal(tf.argmax(Lf_out,1), tf.argmax(y,1))
    train_accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    train_accuracy_summary = tf.summary.scalar("training accuracy", train_accuracy)

opt = tf.train.AdamOptimizer(0.01)
train_op = opt.minimize(loss)

with tf.Session() as sess:
    print 'Training Start'
    sess.run(tf.global_variables_initializer())

    for cnt in xrange(100000000):
        randList = np.random.randint(0,sample_num, size=(1,200))
        # inputSam = random.sample(inputLong,577)
        # print inputSam
        xSam = xIn[randList, :]
        ySam = yIn[randList, :]
        # print ySam
        # _, loss_val, W_val = sess.run([train_op, loss, W3],
        #                               feed_dict={x: xIn[:sample_num],
        #                                          y: yIn[:sample_num].reshape((-1, 2))})
        _, loss_val, W_val = sess.run([train_op, loss, W3],
                                      feed_dict={x: xSam.reshape(-1, data_dim),
                                                 y: ySam.reshape(-1, 2)})



        if cnt%1000 == 0 or( cnt < 2000 and cnt%100 == 0):#!!!!!!!!!!!在这里！！！！！！！！
            accu = train_accuracy.eval(feed_dict={x: xIn[:sample_num],
                                                  y: yIn[:sample_num].reshape((-1, 2))})
            accu_test = train_accuracy.eval(feed_dict={x: xIn[sample_num:],
                                                       y: yIn[sample_num:].reshape((-1, 2))})
            # print W1_val
            print '#' * 20
            print 'cnt', cnt
            print loss_val
            print accu
            print accu_test
            # print W_val

            if accu > 0.99:
                print '#' * 20
                print accu
                print accu_test
                print cnt
                print 'Done'
                break

