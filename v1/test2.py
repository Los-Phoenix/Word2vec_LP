#coding:UTF8
import gensim
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

model = gensim.models.Word2Vec.load("zh_model_100_all_cha")

charList = list(open("zh_model_200_all_voc_cha"))
charClearList = [i.split(" ")[0] for i in charList]
charSet = set(charClearList)
for char in charSet:
    char = char.decode()#这里注意，只有Unicode编码才行
    print len(model[char])
