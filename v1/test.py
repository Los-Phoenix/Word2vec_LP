#coding:UTF8
import gensim
import sys


reload(sys)
sys.setdefaultencoding('utf-8')

import myWord2vec
import myWord2vec.Word2VecWC

model = gensim.models.Word2Vec.load("../data/wikiDummy4/Dummy_model")
model2 = gensim.models.Word2Vec.load("../data/wikiDummy2/Dummy_model")

print model.wv.vocab[u"互联网"]

testList = [u"互联网",
u"美国人",
u"猛扑",
u"给以",
u"惊心掉胆",
u"行李架",
u"检察官法",
u"初审",
u"江"]
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