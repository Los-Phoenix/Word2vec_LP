#coding:UTF-8

#这是一个用来把测试样例存起来的脚本
#前半段与testComb1L相同
#直到生成了y x x_Other
#有用的有y x x_Other vocab embed
#再写一个函数用来读这些东西

import sys
import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
import gensim
import gc
import cPickle as pickle

reload(sys)
sys.setdefaultencoding('utf-8')

#gensim文件的读取

#思路是这样的：
#先读取字典：word
#对于每一个word
#拆成字
#字查找词
#词组织成一行（最多40个）

if __name__ == '__main__':
    sim_num = 1000
    model = gensim.models.Word2Vec.load("../../data/novel/novel2_model")

    fDictWord = open("../../data/simWoodDict")
    # fDictWord = open("../../data/unionDict1000")
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

    vocab,embd = loadWord2Vec('../../data/novel/novel2_vec', None)
    vocab_size = len(vocab) + 1
    embedding_dim = len(embd[0])
    embedding = np.asarray(embd)
    print vocab_size,embedding_dim

    listWords = list()
    for i in listWords_raw:
        word_temp = i.strip().decode()
        if len(word_temp) == 2 and word_temp in vocab:
            print len(listWords)
            listWords.append(word_temp)
    # listWords = listWords[0:1]
    # print listWords
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

    print('Dump Start')
    f1 = file('../../data/CombData3.pkl', 'wb')
    pickle.dump(vocab, f1, True)
    pickle.dump(embd, f1, True)
    pickle.dump(y, f1, True)
    pickle.dump(x, f1, True)
    pickle.dump(x_other, f1, True)
    f1.close()
    print('Dump Done')


def loadAll():
    #returns vocab, embd, y, x, x_other
    f = file('../../data/CombData3.pkl', 'rb')
    vocab1 = pickle.load(f)
    embd1 = pickle.load(f)
    y1 = pickle.load(f)
    x1 = pickle.load(f)
    x_other1 = pickle.load(f)
    f.close()
    return vocab1, embd1, y1, x1, x_other1





