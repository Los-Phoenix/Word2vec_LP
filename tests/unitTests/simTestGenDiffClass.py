#coding:utf-8

#这个脚本从同义词词林中生成负例子
#生成与Pos同样数量的负例

#对于每一行
#数出Hit的总数
#对每一个Hit的词
#找词典里两倍数量的随机词
#找前Hit个反例词输出为负例（可能包含正例，这是同义词词林撰写的不完全性，但是我们不管）
import sys
import gensim
import random
reload(sys)
sys.setdefaultencoding('utf-8')

dictSet = set([])

#加载词典
vocab = open("../../v1/zh_model_200_all_voc")
listVocab = list(vocab)
vocab.close()
vocabSet = set(list([i.split(' ')[0] for i in listVocab]))

#加载同义词词林
f_wood = open("../../v1/simWood.txt")#这是同义词词林
listWood = list(f_wood)
f_wood.close()

for line in listWood:#词林中的每一行

    words = line.split();
    title = words[0]
    words = words[1:]
    # 对于一行
    # 不是同义行跳出
    if not title.endswith('='):
        continue

    wordSet = set(words)
    wordSet = wordSet.intersection(vocabSet)
    dictSet = dictSet.union(wordSet)#把这一行的所有词插入vocabSet
    print len(dictSet)


#打开写入文件
f_rst = open("woodWikiTestDiffClass1000", 'w')

cnt = 0
#遍历同义词词林
cntline = 0
for line in listWood:#词林中的每一行

    words = line.split();
    title = words[0]
    words = words[1:]
    # 对于一行
    # 不是同义行跳出
    if not title.endswith('='):
        continue
    cntline += 1
    if cntline > 1000:
        break;
    wordSet = set(words)
    wordSet = wordSet.intersection(vocabSet)
    remainSet = dictSet.difference(wordSet)
    wordNum = len(wordSet)

    if wordNum <= 1:
        continue

    randomSet = random.sample(remainSet, wordNum - 1)
    # print title
    #对于一个词
    for word in wordSet:
        word = word.strip()
        for word2 in randomSet:
            rst = ''
            word2 = word2.strip()
            # 是同义词的不要，跳出

            cnt += 1
            rst += title
            rst += '\t' + word
            rst += '\t' + word2
            if cnt % 1000 == 0:
                print cnt
                print rst
            rst += '\n'
            # 生成结果，写入
            f_rst.write(rst)
f_rst.flush()
f_rst.close()