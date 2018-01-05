#coding:utf8

#从voc中间读出来所有的词
#分成单字词、2-3字词、4字词、5字以上词4个集合

voc_file = open('../../data/wiki_phrase2/wiki_voc')

uniSet = set()
biTriSet = set()
quadSet = set()
longSet = set()
