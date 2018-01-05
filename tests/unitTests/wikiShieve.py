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

fLegal = open("../../data/slimDict")
linesLegal = list(fLegal)
legalSet = set([i.decode().strip() for i in linesLegal])
fLegal.close()



print len(legalSet)

hitSet = set()
wikiF = open("../../data/wikiSegged", 'r')
outF = open("../../data/wikiShieved", 'w')

wikiL = list(wikiF)

sent_cnt = 0
line_cnt = 0

testStr = u"阿基米德"
testStr = u"阿斯顿发生"

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
            if word.find(testStr) > -1:
                print word
                print testStr in legalSet
                print"URIKA!!\n\n"

            if word in legalSet:
                if word == testStr:
                    print"URIKA!!\n\n"
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

    if len(str) >= 100:#Not Too short
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



