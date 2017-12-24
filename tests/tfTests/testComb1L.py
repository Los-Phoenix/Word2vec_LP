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


print()

model = gensim.models.Word2Vec.load("../../data/wikiDummy2/Dummy_model")

fDictWord = open("../../data/unionDict1000")
listWords =list(fDictWord)
listWords = [i.strip().decode() for i in listWords]
# listWords = listWords[0:100]

wordSet = set(listWords)
chaSet = set([])
for word in wordSet:
    chaSet = chaSet.union(set([char for char in word]))

# print 'setLegal:',len(wordSet)
#
# for word in wordSet:
#     print word
#
# print 'setCharLegal:', len(charSet)
# print listWords

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
#
# #init vocab processor
max_document_length = 1
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
pretrain = vocab_processor.fit(vocab)
y = np.array(list(vocab_processor.transform(listWords)))
print y

max_document_length = 40
processor = learn.preprocessing.VocabularyProcessor(max_document_length)
processor.fit(vocab)
sim_list = list()
for i in listWords:#每一个i代表一个词，结果需要组织在一个字符串里
    str = ''
    for j in i:#j代表一个字
        if not j in model.wv.vocab:
            continue
        for j_sim in model.wv.most_similar(j):
            # print(j_sim[0] in vocab)
            str += j_sim[0] + ' '
    sim_list.append(str)

# print sim_list

qq = list(processor.transform(sim_list))
print(qq)
x = np.array(qq)
samplesize = 500
f_num = 5

del (model)
gc.collect()

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
    yIn = tf.placeholder(tf.int32, [samplesize,1])

    xEmbedRaw = tf.nn.embedding_lookup(charEmbed, xIn)
    xEmbed = tf.reshape(xEmbedRaw, [-1, max_document_length, embedding_dim, 1])
    yEmbedRaw = tf.nn.embedding_lookup(wordEmbed, yIn)
    yEmbed = tf.reshape(yEmbedRaw, [-1, embedding_dim])

    W1 = tf.Variable(tf.truncated_normal([2, 1, 1, f_num], stddev=0.1), 'weight1', dtype=tf.float32)
    #b1 = tf.Variable(np.random.rand(1, 5), 'bias1', dtype=tf.float32)
    conv1 = tf.nn.conv2d(xEmbed, W1, strides=[1, 1, 1, 1], padding='VALID')#  + b1



    W3 = tf.Variable(tf.truncated_normal([embedding_dim *( max_document_length - 1) * f_num, embedding_dim], stddev=0.1), 'weight1', dtype=tf.float32)
    L2_in = tf.matmul(tf.reshape(conv1, [-1, embedding_dim * ( max_document_length - 1) * f_num]), W3)
    L2_out = tf.nn.sigmoid(L2_in)

    loss = tf.reduce_mean((yEmbed - L2_out) ** 2)

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
        randList = np.random.randint(0, 6500, size=(1, samplesize))
        xSam = x[randList, :]
        ySam = y[randList, :]
        _, loss_val, y_std, y_out= sess.run([train_op, loss, yEmbed, L2_out],feed_dict={xIn: xSam.reshape(-1,max_document_length),
                                                   yIn: ySam.reshape(-1,1)})
        if cnt % 100 == 0:
            print loss_val
        if cnt % 1000 == 0:
            print y_out-y_std
            randList = np.random.randint(0, vocab_size - 1, size=(1, samplesize))
            xSam = x[6501:7001]
            ySam = y[6501:7001]
            accu_test = train_accuracy.eval(feed_dict={xIn: xSam.reshape(-1,max_document_length),
                                                        yIn: ySam.reshape(-1,1)})
            print 'testAccu', accu_test

    # print xR
    # print yR

