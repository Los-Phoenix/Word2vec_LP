#coding:utf-8

from cStringIO import StringIO

import os
import sys, getopt

reload(sys)
sys.setdefaultencoding('utf-8')

maxLength = 10000

def is_chinese(uchar):
    """判断一个unicode是否是汉字"""
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False

#这个脚本将读取一个字典先
#再读wikiDummy
#再每一行跟字典进行对比
#结果输出到同一目录下得wikiNew

fLegal = open("../../data/legalWords.txt")
linesLegal = list(fLegal)
legalWords = [i.decode().split('\t') for i in linesLegal]
intcnt = 0
fLegal.close()
legalSet = set()

for i in legalWords:
    legalSet = legalSet.union(set(i))

legalSet.add(u"阿爸")
legalSet.add(u"阿基米德")




# for i in legalSet:
#     print i
print len(legalSet)

fSimDict = list(open("../../data/simWoodDict"))
legalSimSet = set([i.decode().strip() for i in fSimDict])
# for i in legalSimSet:
#     print i
print len(legalSimSet)

fSimDict240 = list(open("../../data/240.txt"))
legalSimSet240 = set()
for i in fSimDict240:
    w0, w1, _ = i.decode().split('\t')
    legalSimSet240.add(w0)
    legalSimSet240.add(w1)

print len(legalSimSet240)

fSimDict297 = list(open("../../data/297.txt"))
legalSimSet297 = set()
for i in fSimDict297:
    w0, w1, _ = i.decode().split('\t')
    legalSimSet297.add(w0)
    legalSimSet297.add(w1)

print len(legalSimSet297)

print"ori        :", len(legalSet)
legalSet = legalSet.union(legalSimSet)
print"add simWood:", len(legalSet)
legalSet = legalSet.union(legalSimSet240)
print"add 240:", len(legalSet)
legalSet = legalSet.union(legalSimSet297)
print"add 297:", len(legalSet)

hitSet = set()
wikiF = open("../../data/wikiNew2/wikiDummy", 'r')
outF = open("../../data/wikiNew2/wikiNew", 'w')

wikiL = list(wikiF)

sent_cnt = 0
line_cnt = 0
for linesRaw in wikiL:
    line_cnt += 1
    lines = linesRaw.decode()
    str = ""
    cnt = 0
    words = lines.split(" ")

    for word in words:
        word = word.strip()

        if len(word) == 1:
            if is_chinese(word):
                str += word
                str += " "
                cnt += 1
        else:#Longer words
            if word in legalSet:
                hitSet.add(word)
                str += word
                str += " "
                cnt += 1
        if cnt > maxLength:
            # print "Long Sent!"
            str += "\n"
            outF.write(str)
            str = ""
            cnt = 0
            sent_cnt += 1

    if len(str) >= 100:#Too short
        str += "\n"
        outF.write(str)
        sent_cnt += 1
    # else:
    #     print "Short Line"
    #     print str

    str = ""
    cnt = 0

    if sent_cnt % 1000 == 0:
        print 'Cleaned sents:', sent_cnt, 'at Line', line_cnt
        outF.flush()
outF.close()

print len(wikiL)

inter = legalSet.difference(hitSet)
print "not Hit:"
q = inter.copy()
for i in q:
    if len(i) == 1:
        inter.remove(i)
    else:
        print i
print len(inter)



