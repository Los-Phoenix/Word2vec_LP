#coding:utf-8
import sys
import numpy as np
import gensim

reload(sys)
sys.setdefaultencoding('utf-8')

fDictWord = open("../../data/unionDict1000")
listWords =list(fDictWord)
listWords = [i.strip().decode() for i in listWords]

model = gensim.models.Word2Vec.load("../../data/wikiDummy/Dummy_model")
vec_dim = 100


wordSet = set(listWords)
chaSet = set([])

model_vocab = model
word_voc_set = set(model.wv.vocab.keys())

cnt_found = 0
cnt_notfound =0
# word_vec = np.zeros(shape=vec_dim)
diff = list(np.ndarray([1,vec_dim]))

for word in wordSet:

    print word
    if word in word_voc_set:
        cnt_found += 1
    else:
        cnt_notfound += 1
        continue

    word_vec = model[word]

    chars = set([char for char in word])#word 里面的所有字
    char_num = 0 #找到的字的个数

    char_vec_sum = np.zeros(shape=vec_dim)
    for char in chars:
        if not char in word_voc_set:
            continue

        print char
        # print model[char]

        simrst = model.wv.most_similar(char)
        for sim_word,_ in simrst:
            print sim_word
            char_vec_sum += model[sim_word]
            char_num += 1
            # print ''
        #
        # char_vec_sum += model[char]
        # char_num += 1

    if char_num == 0:#所有字都不在
        continue

    print char_num
    diff.append(word_vec-char_vec_sum/char_num)

    print diff[len(diff)-1]
    if cnt_found > 10:
        break


    # print word_vec
    # result = model.wv.most_similar(word)
    # print(result)
    # for e in result:
    #     print e[0], e[1]

    # print 'Found:%d, notFound:%d', cnt_found, cnt_notfound