#coding:UTF8
import gensim
import sys


reload(sys)
sys.setdefaultencoding('utf-8')

import myWord2vec
import myWord2vec.Word2VecWC

model = myWord2vec.Word2VecWC.load("../data/wikiDummy/Dummy_model")
# model = gensim.models.Word2Vec.load("../data/wikiDummy/Dummy_model")

testList = [u"钾", u"扒鸡", u"计算机", u"工作", u"本", u"子", u"死", u"美", u"澳",  u"加",  u"求和",  u"历史",  u"惠普"]
for test in testList:
    print test

    result = model.wv.most_similar(test)
    #print(result)
    for e in result:
        print e[0], e[1]

    print ' '