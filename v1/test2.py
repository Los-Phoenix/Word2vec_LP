#coding:UTF8
import gensim
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

model = gensim.models.Word2Vec.load("../data/wiki_cha/zh_model_100_all_cha")
#
# charList = list(open("../../data/wiki_cha/zh_model_200_all_voc_cha"))
# charClearList = [i.split(" ")[0] for i in charList]
# charSet = set(charClearList)
# for char in charSet:
#     char = char.decode()#这里注意，只有Unicode编码才行
#     print len(model[char])
print model[u'贤']

model2 = gensim.models.Word2Vec.load("../data/wiki_word/zh_model_200_all")
print model2[u'贤良']