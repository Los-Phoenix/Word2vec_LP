#coding:utf-8
#这个文件是一层的组合网络：
import sys
import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
import gensim
import gc
reload(sys)
sys.setdefaultencoding('utf-8')
#gensim文件的读取

#思路是这样的：
#先读取字典：word
#对于每一个word
#拆成字
#字查找词
#词组织成一行（最多40个）

import pickleXY as pxy

print "load embeddings"
vocab, embd, y, x, x_other = pxy.loadAll()
print "load Done"
embedding = np.asarray(embd)
vocab_size = len(vocab) + 1
embedding_dim = len(embd[0])
sim_num = len(x_other[0])/2
max_document_length = len(x[0])

whole_size = y.shape[0]
samplesize = 32
f_num = 400
f_num2 = 400

print("NN Start")
with tf.variable_scope("Ez_flat"):
    wordEmbed = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_dim]),
                    trainable=False, name="Word")
    charEmbed = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_dim]),
                            trainable=False, name="Char")
    embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_dim])
    char_embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_dim])

    embedding_init = wordEmbed.assign(embedding_placeholder)
    char_embedding_init = charEmbed.assign(char_embedding_placeholder)

    xIn = tf.placeholder(tf.int32, [samplesize, max_document_length])
    xIn2 = tf.placeholder(tf.int32, [samplesize, max_document_length])
    yIn = tf.placeholder(tf.int32, [samplesize,1])

    xEmbedRaw = tf.nn.embedding_lookup(charEmbed, xIn)
    xEmbed = tf.reshape(xEmbedRaw, [-1, max_document_length, embedding_dim])

    xEmbedRaw2 = tf.nn.embedding_lookup(charEmbed, xIn2)
    xEmbed2 = tf.reshape(xEmbedRaw2, [-1, max_document_length, embedding_dim])

    yEmbedRaw = tf.nn.embedding_lookup(wordEmbed, yIn)
    yEmbed = tf.reshape(yEmbedRaw, [-1, embedding_dim])

    alpha= tf.reduce_sum(xEmbed * xEmbed2, axis = 2)
    # alpha_mat = alpha
    alpha = tf.expand_dims(alpha, -1)
    # alpha_mat = tf.expand_dims(alpha_mat, -1)
    # for i in xrange(embedding_dim):
    #     alpha_mat = tf.stack(alpha, 2)
    print alpha.get_shape()
    xAttention_temp = tf.matmul(tf.transpose(xEmbed,perm=[0,2,1]), alpha)
    print xAttention_temp.get_shape()
    xAttention = tf.squeeze(xAttention_temp)
    W1 = tf.Variable(tf.truncated_normal([embedding_dim, f_num], stddev=0.1), 'weight1', dtype=tf.float32)
    b1 = tf.Variable(tf.truncated_normal([1, f_num], stddev=0.1), 'bias1', dtype=tf.float32)
    L1_out = tf.nn.sigmoid(tf.matmul(xAttention, W1) + b1)

    W2 = tf.Variable(tf.truncated_normal([f_num, f_num2], stddev=0.1), 'weight1', dtype=tf.float32)
    b2 = tf.Variable(tf.truncated_normal([1, f_num2], stddev=0.1), 'bias1', dtype=tf.float32)

    L2_out = tf.nn.sigmoid(tf.matmul(L1_out, W2) + b2)

    W3 = tf.Variable(tf.truncated_normal([f_num2, embedding_dim], stddev=0.1), 'weight1', dtype=tf.float32)
    b3 = tf.Variable(tf.truncated_normal([1, embedding_dim], stddev=0.1), 'bias1', dtype=tf.float32)

    L3_out = tf.nn.sigmoid(tf.matmul(L2_out, W3) + b3)


    self_var = tf.reduce_mean((yEmbed) ** 2)
    loss = tf.reduce_mean((yEmbed - L3_out) ** 2)
    # loss = tf.reduce_mean(alpha ** 2)

opt = tf.train.AdamOptimizer(0.01)
train_op = opt.minimize(loss)

with tf.name_scope("training-accuracy") as scope:
    self_var2 = tf.reduce_mean((yEmbed) ** 2)
    correct_prediction = (L3_out - yEmbed)**2
    train_accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    train_accuracy_summary = tf.summary.scalar("training accuracy", train_accuracy)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run([embedding_init,char_embedding_init], feed_dict={embedding_placeholder: embedding,
                                                              char_embedding_placeholder:embedding})
    #xR, yR = sess.run([xEmbed, yEmbed], feed_dict={xIn: x[:samplesize],
    #                                               yIn: y[:samplesize]})
    for cnt in xrange(5000000):
        randList = np.random.randint(0, whole_size-samplesize, size=(1, samplesize))

        xSam = x[randList, :]
        xSam2 = x_other[randList, :]
        ySam = y[randList, :]
        _, loss_val, y_std, y_out, self_var1= sess.run([train_op, loss, yEmbed, L2_out, self_var],feed_dict={xIn: xSam.reshape(-1,max_document_length),
                                                                                        xIn2: xSam2.reshape(-1,max_document_length),
                                                                                        yIn: ySam.reshape(-1,1)})
        if cnt % 100   == 0:
            print 'Loss', loss_val, 'self_var', self_var1
        if cnt % 1000 == 0:
            #print y_out-y_std
            #print alpha_
            #randList = np.random.randint(whole_size-1000, whole_size, size=(1, samplesize))
            xSam = x[whole_size - samplesize: whole_size, :]
            xSam2 = x_other[whole_size - samplesize: whole_size, :]
            ySam = y[whole_size - samplesize: whole_size, :]
            accu_test = train_accuracy.eval(feed_dict={xIn: xSam.reshape(-1,max_document_length),
                                                       xIn2: xSam2.reshape(-1, max_document_length),
                                                        yIn: ySam.reshape(-1,1)})
            self_var21 = self_var2.eval(feed_dict={xIn: xSam.reshape(-1,max_document_length),
                                                       xIn2: xSam2.reshape(-1, max_document_length),
                                                        yIn: ySam.reshape(-1,1)})
            print 'testAccu', accu_test, 'self_var', self_var21

    # print xR
    # print yR

