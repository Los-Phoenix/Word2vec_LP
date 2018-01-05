#coding:utf-8

import sys, getopt

reload(sys)
sys.setdefaultencoding('utf-8')

def is_chinese(uchar):
    """判断一个unicode是否是汉字"""
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False


fLegal = open("../../data/legalWords.txt")
outFile = open("../../data/legalDict", 'w')
linesLegal = list(fLegal)
legalWords = [i.decode().split('\t') for i in linesLegal]
intcnt = 0
fLegal.close()
legalSet = set()

for i in legalWords:
    legalSet = legalSet.union(set(i))

legalSet.add(u"阿爸")
legalSet.add(u"阿基米德")
legalSet.add(u"座无虚席")
legalSet.add(u"做习题")
legalSet.add(u"做作")

cnt = 0
for i in legalSet:
    save = True
    for q in i:
        if not is_chinese(q):
            print(i)
            save = False
            break
    if save:
        cnt += 1
        outFile.write(i+"\n")

print cnt
outFile.close()


