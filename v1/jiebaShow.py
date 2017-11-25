#coding:utf8

import jieba
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("python jiebaShow inFile outFile")
        sys.exit(1)
    inp, outp = sys.argv[1:3]
    space = " "
    i = 0

    jieba.load_userdict("simWoodDict")

    output = open(outp, 'w')
    lineArray = list(open(inp, "r").readlines());
    i = 0
    for line1 in lineArray:
        i+=1
        if (i % 1000 == 0):
           print("Jiebad " + str(i) + " articles")

        seg_list = jieba.cut(line1)
        segLine = " ".join(seg_list)
        output.write(segLine)

    output.close()


