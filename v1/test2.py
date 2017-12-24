#coding:UTF8
import gensim
import sys
from myWord2vec import Word2VecWC

reload(sys)
sys.setdefaultencoding('utf-8')

model = gensim.models.Word2Vec.load("../data/wikiDummy/Dummy_model")
#
# charList = list(open("../../data/wiki_cha/zh_model_200_all_voc_cha"))
# charClearList = [i.split(" ")[0] for i in charList]
# charSet = set(charClearList)
# for char in charSet:
#     char = char.decode()#这里注意，只有Unicode编码才行
#     print len(model[char])

# i = u"这是一个贤良之人"
# for j in i :
#     print j, model[j]

# model2 = Word2VecWC.load("../data/wiki_word/zh_model_200_all")
# print model2[u'贤良']

for i in model.wv.vocab.keys():
    print i