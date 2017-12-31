#coding:UTF8
import gensim
import sys


reload(sys)
sys.setdefaultencoding('utf-8')

import myWord2vec
import myWord2vec.Word2VecWC

model = gensim.models.Word2VecWC.load("../data/wikiNew2/wikiNew_model")
model2 = gensim.models.Word2VecWC.load("../data/wikiNew2/wikiNew_model__pp")
model3 = gensim.models.Word2VecWC.load("../data/wikiNew2/wikiNew_model_pp_deep")

fSimDict240 = list(open("../data/297.txt"))
legalSimSet240 = set()
for i in fSimDict240:
    w0, w1, _ = i.decode().split('\t')
    legalSimSet240.add(w0)
    legalSimSet240.add(w1)

#testList = [u'拖拉机']
for test in legalSimSet240:
    print test
    try:
        result = model.wv.most_similar(test)
        result2 = model2.wv.most_similar(test)
        result3 = model3.wv.most_similar(test)
        #print(result)
        for e in result:
            print e[0], e[1]

        print '==='

        for e in result2:
            print e[0], e[1]

        print '==='

        for e in result3:
            print e[0], e[1]

        print ' '
    except:
        print '?\n\n\n'