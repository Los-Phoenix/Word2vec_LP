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

sim_num = 1000
model = gensim.models.Word2Vec.load("../../data/wikiDummy2/Dummy_model")

fDictWord = open("../../data/simWoodDict")
listWords_raw =list(fDictWord)

def loadWord2Vec(filename, dictSet):
    vocab = []
    embd = []
    cnt = 0
    fr = open(filename,'r')
    line = fr.readline().decode('utf-8').strip()
    #print line
    word_dim = int(line.split(' ')[1])
    # vocab.append("unk")
    embd.append([0]*word_dim)
    for line in fr :
        row = line.strip().split(' ')
        if dictSet == None or row[0].decode() in dictSet:
            vocab.append(row[0].decode())
            embd.append(row[1:])
    print "loaded word2vec"
    fr.close()
    return vocab,embd

vocab,embd = loadWord2Vec('../../data/wikiDummy/Dummy_model_vec', None)
vocab_size = len(vocab) + 1
embedding_dim = len(embd[0])
embedding = np.asarray(embd)
print vocab_size,embedding_dim

listWords = list()
for i in listWords_raw:
    word_temp = i.strip().decode()
    if len(word_temp) == 2 and word_temp in vocab:
        #print word_temp
        listWords.append(word_temp)
#listWords = listWords[0:150]
#
# #init vocab processor
max_document_length = 1
processor = learn.preprocessing.VocabularyProcessor(max_document_length)
pretrain = processor.fit(vocab)
y = np.array(list(processor.transform(listWords)))
print y.shape
# print y

max_document_length = 2 * sim_num
processor = learn.preprocessing.VocabularyProcessor(max_document_length)
pretrain = processor.fit(vocab)
sim_list = list()
other_list = list()

for i in listWords:#每一个i代表一个词，结果需要组织在一个字符串里
    str = ''
    other_str = ''
    j = i[0] #j代表一个字
    j_other = i[1]
    if j in model.wv.vocab:
        for j_sim in model.wv.most_similar(j,topn=sim_num):
            # print(j_sim[0] in vocab)
            if j_sim == i:
                print "Found self "+i
                continue
            str += j_sim[0] + ' '
            other_str += j_other + ' '

    j = i[1]  # j代表一个字
    j_other = i[0]
    if j in model.wv.vocab:
        for j_sim in model.wv.most_similar(j, topn=sim_num):
            if j_sim == i:
                print "Found self "+ i
                continue
            # print(j_sim[0] in vocab)
            str += j_sim[0] + ' '
            other_str += j_other + ' '

    sim_list.append(str)
    other_list.append(other_str)

x = np.array(list(processor.transform(sim_list)))
x_other = np.array(list(processor.transform(other_list)))

print x.shape
print x_other.shape
#
# print x
# print x_other

whole_size = y.shape[0]
samplesize = 128
f_num = 5

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
    xAttention = tf.squeeze(xAttention_temp)
    W1 = tf.Variable(tf.truncated_normal([embedding_dim, embedding_dim], stddev=0.1), 'weight1', dtype=tf.float32)
    b1 = tf.Variable(tf.truncated_normal([1, embedding_dim], stddev=0.1), 'bias1', dtype=tf.float32)
    L2_out = tf.nn.sigmoid(tf.matmul(xAttention, W1) + b1)

    loss = tf.reduce_mean((yEmbed - L2_out) ** 2)
    # loss = tf.reduce_mean(alpha ** 2)

opt = tf.train.AdamOptimizer(0.001)
train_op = opt.minimize(loss)

with tf.name_scope("training-accuracy") as scope:
    correct_prediction = (L2_out-yEmbed)**2
    train_accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    train_accuracy_summary = tf.summary.scalar("training accuracy", train_accuracy)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run([embedding_init,char_embedding_init], feed_dict={embedding_placeholder: embedding,
                                                              char_embedding_placeholder:embedding})
    #xR, yR = sess.run([xEmbed, yEmbed], feed_dict={xIn: x[:samplesize],
    #                                               yIn: y[:samplesize]})
    for cnt in xrange(500000):
        randList = np.random.randint(0, whole_size-1000, size=(1, samplesize))

        xSam = x[randList, :]
        xSam2 = x_other[randList, :]
        ySam = y[randList, :]
        _, loss_val, y_std, y_out, alpha_= sess.run([train_op, loss, yEmbed, L2_out, alpha],feed_dict={xIn: xSam.reshape(-1,max_document_length),
                                                                                        xIn2: xSam.reshape(-1,max_document_length),
                                                                                        yIn: ySam.reshape(-1,1)})
        if cnt % 100   == 0:
            print loss_val
        if cnt % 1000 == 0 or cnt < 100:
            #print y_out-y_std
            #print alpha_
            #randList = np.random.randint(whole_size-1000, whole_size, size=(1, samplesize))
            xSam = x[whole_size - samplesize: whole_size, :]
            xSam2 = x_other[whole_size - samplesize: whole_size, :]
            ySam = y[whole_size - samplesize: whole_size, :]
            accu_test = train_accuracy.eval(feed_dict={xIn: xSam.reshape(-1,max_document_length),
                                                       xIn2: xSam2.reshape(-1, max_document_length),
                                                        yIn: ySam.reshape(-1,1)})
            print 'testAccu', accu_test

    # print xR
    # print yR

