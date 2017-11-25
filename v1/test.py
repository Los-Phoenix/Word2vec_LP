#coding:UTF8
import gensim
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

model = gensim.models.Word2Vec.load("zh_model_100")
result = model.most_similar(u"元气")
    #print(result)
for e in result:
    print e[0]

print ' '

result = model.most_similar(u"侦探")
    #print(result)
for e in result:
    print e[0]

print ' '

result = model.most_similar(u"激光")
    #print(result)
for e in result:
    print e[0]