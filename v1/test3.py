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

print u"的" in model
print u"阿基米德" in model



for i in model.wv.vocab.keys():
    if i.find(u"阿基") >-1:
        print i