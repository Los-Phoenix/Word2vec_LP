#coding:utf-8

#这个脚本从同义词词林中生成例子
fPos = open("../../data/woodWikiTestPos1000")
linesPos = list(fPos)
fConf = open("../../data/woodWikiTestDiffClass1000")
linesConf = list(fConf)
fDiff = open("../../data/woodWikiTestConfused1000")
linesDiff = list(fDiff)

dictMap = map()