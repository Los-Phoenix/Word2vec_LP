#coding:utf-8

#把wikititle

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

wikiF = open("../../data/wiki.title.jian", 'r')
outF = open("../../data/titleDict", 'w')

wikiL = list(wikiF)

line_cnt = 0
title_cnt = 0
for titleRaw in wikiL:
    line_cnt += 1
    titleRaw = titleRaw.decode()

    title = titleRaw.split(" ")[0]#用空格分割
    title = title.strip()
    nonChineseFlag = False
    for word in title:#title中的每一个字
        if not is_chinese(word):#不全是汉字
            nonChineseFlag = True
            break

    if not nonChineseFlag:
        outF.write(title + '\n')#打印出来
        title_cnt += 1
    if title_cnt % 1000 == 0:
        print 'Cleaned titles:', title_cnt, 'at Line', line_cnt
        outF.flush()
outF.close()




