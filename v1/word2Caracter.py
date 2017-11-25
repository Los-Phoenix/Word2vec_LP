#coding:utf8
#这个文件将结巴之后的文本每个词进行过滤
#汉语词拆成字
#英语词直接删除这个词

import jieba
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("python word2Character inFile outFile")
        sys.exit(1)
    inp, outp, articleNum = sys.argv[1:4]
    space = " "
    i = 0

    output = open(outp, 'w')
    lineArray = list(open(inp, "r").readlines());
    i = 0

    print articleNum
    artNum = eval(articleNum)

    for line1 in lineArray:
        i+=1
        if i > artNum:
            break
        if (i % 1000 == 0):
           print("word2Character " + str(i) + " articles out of "+ articleNum )
        words = line1.strip().split(' ')
        newline = ""
        for word in words:
            word = unicode(word, 'utf-8')
            for character in word:
                #print character
                if not(character.isupper()|character.islower()):
                    #print character
                    newline += character + " "

        output.write(newline + "\n")

    output.close()


