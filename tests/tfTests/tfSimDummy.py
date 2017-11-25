#coding:UTF-8
#这个文件用tf实现一个单层神经网络，用来判断两个词是否是同义词
#先做一个分类的任务，输入是虚拟的
#暂时的输入是比较大小，200维输入，1000个样例

#数据准备：这里应该是读取样例文件，但是我们不管
#数据准备：随机生成1000个200维的实数数组，x
#对应的y：奇数比偶数的大，求和 >0 的是1

import tensorflow as tf
import numpy as np

sample_num = 20000
test_num = 20000

data_dim = 20
data_dim_half = data_dim/2
data_dim_quad = data_dim_half/2.
xIn = np.random.uniform(-1, 1, [sample_num+test_num, data_dim])#这是个假的输入
# W = np.random.uniform(-10, 10, [data_dim, 1])
# print W
xIn2prev = [[i > data_dim_quad, i<=data_dim_quad] for i in np.sum(xIn[:, 0:data_dim_half] > xIn[:, data_dim_half:], 1)]#Woc 我居然把这个蛋疼的东西实现了
# xIn2prev = [xIn2prev, 1- xIn2prev]
# xIn2prev = xIn.dot(W)
# xIn2prev = xIn * W
xIn2 = np.mat(xIn2prev)
# xIn3 = xIn2.reshape((-1, 1))/50
#那这个就是假的输出了
# x2In =
print xIn2
# print yIn
# print tf.Session().run(xIn[0:, 1:2])
x = tf.placeholder(tf.float32, shape = (None, data_dim))
y = tf.placeholder(tf.float32, shape = (None, 2))

num = 10
with tf.variable_scope("Ez_flat"):
    W1 = tf.Variable(np.random.rand(data_dim, num), 'weight1', dtype=tf.float32)
    b1 = tf.Variable(np.random.rand(1, num), 'bias1', dtype=tf.float32)
    # b1 = tf.Variable(np.random.rand(1, num), 'bias1', dtype=tf.float32)
    L1_in = tf.matmul(x, W1) + b1
    #L1_out = tf.nn.softmax(L1_in)
    L1_out = tf.nn.sigmoid(L1_in)
    #
    W2 = tf.Variable(np.random.rand(num, 2), 'weight2', dtype=tf.float32)
    b2 = tf.Variable(np.random.rand(1, 2), 'bias2', dtype=tf.float32)
    #
    L2_in = tf.matmul(L1_out, W2) + b2
    L2_out = tf.nn.softmax(L2_in)

    # rst = L2_out * 50 + 50
    # loss = tf.contrib.losses.mean_squared_error(L1_out, y)
    # loss = tf.contrib.losses.mean_squared_error(L1_out, y)
    loss = -tf.reduce_sum(y * tf.log(L2_out))
    # loss = tf.reduce_mean(loss)

with tf.name_scope("training-accuracy") as scope:
    correct_prediction = tf.equal(tf.argmax(L2_out,1), tf.argmax(y,1))
    train_accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    train_accuracy_summary = tf.summary.scalar("training accuracy", train_accuracy)

opt = tf.train.AdamOptimizer(0.01)
train_op = opt.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for cnt in xrange(50000):
        _, loss_val, W1_val, b1_val = sess.run([train_op, loss, W1, b1],
                               feed_dict ={ x: xIn[:sample_num],
                                            y: xIn2[:sample_num].reshape((-1, 2))})


        if cnt%1000 == 0:#!!!!!!!!!!!在这里！！！！！！！！
            accu = train_accuracy.eval(feed_dict={x: xIn[:sample_num],
                                                  y: xIn2[:sample_num].reshape((-1, 2))})
            accu_test = train_accuracy.eval(feed_dict={x: xIn[sample_num:],
                                                       y: xIn2[sample_num:].reshape((-1, 2))})
            # print W1_val
            print '#' * 20
            print loss_val
            print accu
            print accu_test

            if accu > 0.99:
                print '#' * 20
                print accu
                print accu_test
                print cnt
                print 'Done'
                break


