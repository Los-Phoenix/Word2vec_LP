#coding:utf-8
#这个小程序测试词向量训练的结果是否能用于词向量相似。
#同类词是否在同义词里
#先从同义词林里找一行
#删除不在里面的
#用mostsimilar找最近
#然后看一倍、两倍、三倍覆盖率

import gensim
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

# model = gensim.models.Word2Vec.load("../../v1/zh_model_200_all")
model = gensim.models.Word2Vec.load("../../v1/zh_novel_model_100")
# result = model.most_similar(u"本人", topn = 100)
# for e in result:
#     print e[0]

# wiki_vocab = open("../../v1/zh_model_200_all_voc")#这是维基单词表
wiki_vocab = open("../../v1/zh_novel_model_100_voc")
listWikiVocab = list(wiki_vocab)
wiki_vocab.close()

f_wood = open("../../v1/simWood.txt")#这是同义词词林
listWood = list(f_wood)
f_wood.close()

f_woodTime = open("simTestCountNovel", 'w')

words = []
times = []
for line in listWikiVocab:#这是维基词典集合
    word, time = line.split(" ")
    words.append(word)
    times.append(time)
vocabWikiMap = dict(zip(words, times))#这是维基词典集合，用来计数

cnt = 0
for line in listWood:#词林中的每一行
    words = line.split();
    title = words[0]
    words = words[1:]
    totalCount = len(words)
    rst = ''

    if not title.endswith('='):
        continue

    #先数这一行有多少个Hit
    rst = rst + title + '\t' + str(totalCount)

    wikiHit = 0
    newWords = []
    for word in words:
        word = word.strip()
        timeWiki = vocabWikiMap.get(word, 0)
        if not timeWiki == 0:
            newWords.append(word.decode())
            wikiHit += 1

    if wikiHit < 2:
        continue
    cnt+=1
    if cnt % 100 == 0:
        print cnt
        print newWords[0]

    rst += '\t' +  str(wikiHit)

    # print newWords[0]
    simResult1 = model.most_similar(newWords[0], topn=wikiHit)
    simResult2 = model.most_similar(newWords[0], topn=wikiHit*2)
    simResult3 = model.most_similar(newWords[0], topn=wikiHit*3)
    simResult4 = model.most_similar(newWords[0], topn=200)

    rstList1 = list(rst[0] for rst in simResult1)
    rstList2 = list(rst[0] for rst in simResult2)
    rstList3 = list(rst[0] for rst in simResult3)
    rstList4 = list(rst[0] for rst in simResult4)

    hitSet1 = set(rstList1)
    hitSet2 = set(rstList2)
    hitSet3 = set(rstList3)
    hitSet4 = set(rstList4)

    hitCount1 = 0
    hitCount2 = 0
    hitCount3 = 0
    hitCount4 = 0

    for word in newWords:
        word = word.strip()
        if word in hitSet1:
            hitCount1 += 1
        if word in hitSet2:
            hitCount2 += 1
        if word in hitSet3:
            hitCount3 += 1
        if word in hitSet4:
            hitCount4 += 1

    rst += '\t' + str(hitCount1)
    rst += '\t' + str(hitCount2)
    rst += '\t' + str(hitCount3)
    rst += '\t' + str(hitCount4)
    rst += '\n'

    f_woodTime.write(rst)

f_woodTime.flush()
f_woodTime.close()

