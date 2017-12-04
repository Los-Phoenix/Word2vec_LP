#coding:utf-8
#这个文件是逐渐进行embedding的练习：
import sys
import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
reload(sys)
sys.setdefaultencoding('utf-8')
#gensim文件的读取

#思路是这样的：
#先读取两个字典：word 和 char
#再打开word 的embedding，构建wordEmbedding
#再打开char的embedding，构建charEmbedding

#再查一下
#把输入层和输出层打印出来就好了

fDictWord = open("../../data/unionDict1000")
listWords =list(fDictWord)
listWords = [i.strip().decode() for i in listWords]

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
# #fit the vocab from glove
pretrain = vocab_processor.fit(vocab)
# print vocab
# #transform inputs
# input = [u'贤良 新锐', u'口哨 庞焜元']
# for i in vocab[2992:2994]:
#      print i
# y = np.array(list(vocab_processor.transform(listWords)))
embedding = embedding.astype('float32')
print embedding.std() ** 2

