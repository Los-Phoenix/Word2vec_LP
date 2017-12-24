#coding:utf-8
import sys
import numpy as np
import gensim
from gensim import matutils
import scipy
from numpy import dot
from scipy.cluster.vq import vq,kmeans,whiten
from scipy.stats import mode
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import dot, zeros, dtype, float32 as REAL,\
    double, array, vstack, fromstring, sqrt, newaxis,\
    ndarray, sum as np_sum, prod, ascontiguousarray

reload(sys)
sys.setdefaultencoding('utf-8')

fDictWord = open("../../data/unionDict1000")
listWords =list(fDictWord)
listWords = [i.strip().decode() for i in listWords]

model = gensim.models.Word2Vec.load("../../data/wikiDummy4/Dummy_model")
vec_dim = 100
leave_out = 0
kmeans_method = 3
nearest = 1
char_mean = 1


wordSet = set(listWords)
chaSet = set([])


model_vocab = model
word_voc_set = set(model.wv.vocab.keys())

cnt_found = 0
cnt_notfound =0
# word_vec = np.zeros(shape=vec_dim)
diff = list()
diff2 = list()
diff3 = list()

word_vec_list = list()
word_another_list = list()
model.wv.init_sims()

nearest_list = list()
for word in wordSet:

    print word
    if word in word_voc_set and not len(word) == 1:
        cnt_found += 1
    else:
        cnt_notfound += 1
        continue

    # print word
    word_vec = model.wv.word_vec(word, use_norm = True)
    baseline = model.most_similar(word)[0][1]
    if baseline < 0:
        print "WOW, Single Boy!"

    chars = set([char for char in word])#word 里面的所有字
    char_num = 0 #找到的字的个数

    char_vec_sum = np.zeros(shape=vec_dim)
    sim_words = list()


    nearest_num = 9999999
    for char in chars:
        if not char in word_voc_set:
            continue

        # print char
        # print model[char]

        simrst = model.wv.most_similar(char,topn=1000)
        nearest_num_temp = len(model.wv.words_closer_than(char,word))
        print nearest_num_temp
        nearest_num = nearest_num if nearest_num < nearest_num_temp else nearest_num_temp

        for sim_word,_ in simrst:
            # print sim_word
            sim_words.append(sim_word)
            char_vec_sum += model[sim_word]
            char_num += 1
            # print ''
        #
        # char_vec_sum += model[char]
        # char_num += 1

    nearest_list.append(nearest_num)
    if char_num <= leave_out:#所有字都不在
        continue
    # print char_num

    if leave_out > 0:
        for i in xrange(leave_out):
            o_word = model.doesnt_match(sim_words)
            sim_words.remove(o_word)
            char_num -= 1
            char_vec_sum -= model[o_word]
        diff_value = word_vec-char_vec_sum/char_num

    if kmeans_method > 0:

        # diff1 = ((word_vec - centoids[0][0])**2).mean()
        # diff2 = ((word_vec - centoids[0][1]) ** 2).mean()
        data = whiten(model[sim_words])
        centoids = kmeans(data, kmeans_method)

        labels = vq(data, centoids[0])[0]
        max_label = mode(labels).mode[0]

        # diff_value = word_vec - centoids[0][0]  if diff1 < diff2  else word_vec - centoids[0][1]
        diff_value2 = dot(matutils.unitvec(centoids[0][max_label]).astype(REAL), word_vec) - baseline

    if nearest > 0:
        # print 'used nearest!'
        model_list = list(model[sim_words])

        model_list_value = [baseline if dot(matutils.unitvec(d), word_vec) >= baseline else dot(matutils.unitvec(d), word_vec)
                            for d in model_list]

        diff_value = np.array(model_list_value).max() - baseline
        if diff_value > 0:
            print word
            print diff_value, np.array(model_list_value).max(), baseline

    if char_mean >0:#使用字符的平均值来进行预测
        char_mean_vec_list = [model.wv.word_vec(char, use_norm= True) for char in chars.intersection(word_voc_set)]
        char_mean_vec = np.array(char_mean_vec_list).mean(axis=0)
        diff_value3 = dot(matutils.unitvec(char_mean_vec).astype(REAL), word_vec) - baseline




    diff.append(diff_value)
    diff2.append(diff_value2)
    diff3.append(diff_value3)
    word_vec_list.append(word_vec)
    # print diff[len(diff)-1]
    if cnt_found > 200:
         break

    if cnt_found % 100 == 0:
        print('Found:%d, notFound:%d', cnt_found, cnt_notfound)

mat = np.array(diff)
mat2 = np.array(diff2)
mat3 = np.array(diff3)
mat4 = np.array([mat, mat2, mat3]).max(0)
mat_neatest = np.array(nearest_list)

word_mat = np.array(word_vec_list)
# print word_mat.std()**2
mat_neatest.sort()
print mat_neatest
# sns.kdeplot(mat, shade=True)

mat_sort = [i > -0.01 for i in mat3]
print(mat_sort.count(True))
# plt.hist(mat, color= "#FF0000", alpha = .5)
# plt.hist(mat2, color= "#00FF00", alpha = .5)
plt.plot([-1, 0, 1], [0,0,0])

sns.kdeplot(mat, cumulative=True,vertical = True, label = 'nearest', color = '#FF0000', grid= True)
sns.kdeplot(mat2, cumulative=True,vertical = True, label = 'kmeans',  color = '#00FF00')
sns.kdeplot(mat3, cumulative=True,vertical = True, label = 'charmean',  color = '#0000FF')
sns.kdeplot(mat4, cumulative=True,vertical = True, label = 'bestOf3',  color = '#00FFFF')
sns.kdeplot(np.zeros(len(mat)), cumulative=True,vertical = True, label = 'zero')

plt.show()

sns.kdeplot(np.log10(mat_neatest), cumulative=True,vertical = True, label = 'bestOf3',  color = '#0000FF')
plt.show()