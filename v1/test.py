#coding:UTF8
import gensim
import sys


reload(sys)
sys.setdefaultencoding('utf-8')

import myWord2vec
import myWord2vec.Word2VecWC

folder_path = "../data/wiki_phrase/"
ori_name = "wiki"
model_suffix = "_model"
vec_suffix = "_vec"
voc_suffix = "_voc"

model = gensim.models.Word2VecWC.load(folder_path + ori_name + model_suffix)

#print model.wv.vocab[u"互联网"]

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

        #print(result)
        for e in result:
            print e[0], e[1]

        print '==='

    except:
        print '?'