#coding:utf-8
#这个文件是单个字的线性转移网络：
import sys
import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
reload(sys)
sys.setdefaultencoding('utf-8')

fDictWord = open("../../data/interDictCharWord")
listWords =list(fDictWord)
listWords = [i.strip().decode() for i in listWords]

wordSet = set(listWords)
chaSet  = set(listWords)

def loadWord2Vec(filename, dictSet):
    vocab = []
    embd = []

    fr = open(filename,'r')
    line = fr.readline().decode('utf-8').strip()
    #print line
    word_dim = int(line.split(' ')[1])
    # vocab.append("unk")
    embd.append([0]*word_dim)
    for line in fr :
        row = line.strip().split(' ')
        if row[0].decode() in dictSet:
            vocab.append(row[0].decode())
            embd.append(row[1:])
    print "loaded word2vec"
    fr.close()
    return vocab,embd

vocab,embd = loadWord2Vec('../../data/wiki_word/zh_model_200_all_vec', wordSet)
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

cha_vocab,char_embd = loadWord2Vec('../../data/wiki_cha/zh_model_200_all_vec_cha', chaSet)
char_size = len(cha_vocab) + 1
cha_embedding_dim = len(char_embd[0])
cha_embedding = np.asarray(char_embd)
print char_size,cha_embedding_dim
max_document_length = 1
cha_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
cha_processor.fit(cha_vocab)

x = np.array(list(cha_processor.transform([' '.join(i) for i in listWords])))
print x

samplesize = 500
mid_dim = 100
with tf.variable_scope("Ez_flat"):
    wordEmbed = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_dim]),
                    trainable=False, name="Word")
    charEmbed = tf.Variable(tf.constant(0.0, shape=[char_size, cha_embedding_dim]),
                            trainable=False, name="Char")
    embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_dim])
    char_embedding_placeholder = tf.placeholder(tf.float32, [char_size, cha_embedding_dim])

    embedding_init = wordEmbed.assign(embedding_placeholder)
    char_embedding_init = charEmbed.assign(char_embedding_placeholder)

    xIn = tf.placeholder(tf.int32, [samplesize, max_document_length])
    yIn = tf.placeholder(tf.int32, [samplesize,1])

    xEmbedRaw = tf.nn.embedding_lookup(charEmbed, xIn)
    xEmbed = tf.reshape(xEmbedRaw, [-1, cha_embedding_dim])
    yEmbedRaw = tf.nn.embedding_lookup(wordEmbed, yIn)
    yEmbed = tf.reshape(yEmbedRaw, [-1, embedding_dim])

    W1 = tf.Variable(tf.truncated_normal([cha_embedding_dim, mid_dim], stddev=0.1), 'weight1', dtype=tf.float32)
    b1 = tf.Variable(tf.truncated_normal([1, mid_dim], stddev=0.1), 'bias1', dtype=tf.float32)
    # conv1 = tf.nn.conv2d(xEmbed, W1, strides=[1, 1, 1, 1], padding='VALID')#  + b1
    L1_in = tf.matmul(xEmbed, W1) + b1
    L1_out = tf.nn.relu(L1_in)

    W2 = tf.Variable(tf.truncated_normal([mid_dim, embedding_dim], stddev=0.1), 'weight2', dtype=tf.float32)
    b2 = tf.Variable(tf.truncated_normal([1, embedding_dim], stddev=0.1), 'bias2', dtype=tf.float32)

    # W3 = tf.Variable(tf.truncated_normal([cha_embedding_dim *( max_document_length - 1) * 10, embedding_dim], stddev=0.1), 'weight1', dtype=tf.float32)
    L2_out = L1_in#tf.matmul(L1_out, W2)# + b2
    loss = tf.reduce_mean((yEmbed - L2_out) ** 2)

opt = tf.train.AdamOptimizer(0.001)
train_op = opt.minimize(loss)

with tf.name_scope("training-accuracy") as scope:
    correct_prediction = (L2_out-yEmbed)**2
    train_accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    train_accuracy_summary = tf.summary.scalar("training accuracy", train_accuracy)

with tf.Session() as sess:
    shuffleList = np.random.randint(0, len(x), size=(1, len(x)))
    x = x[shuffleList, :].reshape(-1, max_document_length)
    y = y[shuffleList, :].reshape(-1, 1)
    sess.run(tf.global_variables_initializer())
    sess.run([embedding_init,char_embedding_init], feed_dict={embedding_placeholder: embedding,
                                                              char_embedding_placeholder:char_embd})
    #xR, yR = sess.run([xEmbed, yEmbed], feed_dict={xIn: x[:samplesize],
    #                                               yIn: y[:samplesize]})
    for cnt in xrange(5000000000):
        randList = np.random.randint(0, 500, size=(1, samplesize))
        xSam = x[randList, :]
        ySam = y[randList, :]
        _, loss_val, y_std, y_out, bias = sess.run([train_op, loss, yEmbed, L2_out, b1],feed_dict={xIn: xSam.reshape(-1,max_document_length),
                                                   yIn: ySam.reshape(-1,1)})
        if cnt % 1000 == 0:
            print 'loss:',loss_val
            print 'avg Loss:',bias.max() - bias.min()
            print cnt
        if cnt % 10000 == 0:
            print y_out-y_std
            randList = np.random.randint(0, vocab_size - 1, size=(1, samplesize))
            xSam = x[4501:5001]
            ySam = y[4501:5001]
            accu_test = train_accuracy.eval(feed_dict={xIn: xSam.reshape(-1,max_document_length),
                                                        yIn: ySam.reshape(-1,1)})
            print 'testAccu', accu_test

    # print xR
    # print yR

