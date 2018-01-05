#coding:utf-8

import sys, getopt

reload(sys)
sys.setdefaultencoding('utf-8')

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

f_out = open('../../data/wordSDict', 'w')
for i in legalSimSet240:
    f_out.write(i + "\n")
for i in legalSimSet297:
    f_out.write(i + "\n")
f_out.close()