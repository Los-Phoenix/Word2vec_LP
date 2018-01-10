#coding:utf-8

#把redi2q清洗一下
# 首先分段
# 分段

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

#这个脚本将读取wikiTitle
#每一行:先分掉' ' 再做strip
#再取全是汉字的行
#粘到title之后
#还是先读240和297吧

wikiF = open("../../data/redi1", 'r')
outF = open("../../data/redi2", 'w')

wikiL = list(wikiF)

line_cnt = 0
title_cnt = 0
for lineRaw in wikiL:
    line_cnt += 1
    lineRaw = lineRaw.decode()

    titleRaw1, titleRaw2 = lineRaw.split("\t")

    titleRaw2 = titleRaw2.strip()
    title1 = titleRaw1.split(" ")[0]
    title2 = titleRaw2.split(" ")[0]

    nonChineseFlag = False
    for word in title1:#title中的每一个字
        if not is_chinese(word):#不全是汉字
            nonChineseFlag = True
            break

    for word in title2:#title中的每一个字
        if not is_chinese(word):#不全是汉字
            nonChineseFlag = True
            break

    if title1 == title2:
        nonChineseFlag = True

    if not nonChineseFlag:
        outF.write(title1 + '\t' + title2 + '\n')#打印出来
        title_cnt += 1
    if title_cnt % 1000 == 0:
        print 'Cleaned titles:', title_cnt, 'at Line', line_cnt
        outF.flush()
outF.close()




