#coding:UTF8
import gensim
import sys


reload(sys)
sys.setdefaultencoding('utf-8')

import myWord2vec
import myWord2vec.Word2VecWC

model = myWord2vec.Word2VecWC.load("../data/wikiNew/wikiNew_model")
model2 = gensim.models.Word2Vec.load("../data/wikiNew/wikiNew_model__pp")

testList = [u"手机", u"伏特加", u"法律", u"李白", u"本", u"子", u"死", u"美", u"澳",  u"加",  u"求和",  u"历史",  u"惠普"]
#testList = [u'拖拉机']
for test in testList:
    print test
    try:
        result = model.wv.most_similar(test)
        result2 = model2.wv.most_similar(test)
        #print(result)
        for e in result:
            print e[0], e[1]

        print '==='

        for e in result2:
            print e[0], e[1]

        print ' '
    except:
        print '?'