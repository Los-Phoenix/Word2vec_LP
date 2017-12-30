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

fLegal = open("legalWords.txt")
linesLegal = list(fLegal)
legalWords = [i.decode().split('\t') for i in linesLegal]
intcnt = 0
fLegal.close()
legalSet = set()

for i in legalWords:
    legalSet = legalSet.union(set(i))

legalSet.add(u"阿爸")
legalSet.add(u"阿基米德")




for i in legalSet:
    print i
print len(legalSet)

fSimDict = open("")


exit(0)
wikiF = open("wikiDummy", 'r')
outF = open("wikiNew", 'w')

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

#
#
# for i in legalSet:
#     print i
# def convert(fname):#把一个文件中的内容变成一行字符串。注意是一行！！
#
#     infile = file(fname, 'rb')
#     sents = infile.read();
#     infile.close()
#     strBuffer = sents.decode('gbk', 'ignore')
#     strs = []
#     str = ""
#     cnt = 0
#     for oneWord in strBuffer:
#         cnt += 1
#         if cnt % 100000 == 0:
#             print "read " + cnt.__str__() + " words"
#         if cnt % 2000 == 0:
#             str = ' '.join(str.split())+"\n"
#             strs.append(str.encode('UTF-8'))
#             str = ""
#
#         if is_chinese(oneWord):
#             str += oneWord
#         else:
#             str += " "
#
#     return strs
#
#
# # converts all pdfs in directory pdfDir, saves all resulting txt files to txtdir
# def convertMultiple(textFileDir, txtName):
#     # if textFileDir == "":
#     #     textFileDir = os.getcwd() + "\\"  # if no textFileDir passed in
#
#     fileDirList = os.listdir(textFileDir)
#     for textFile in fileDirList:  # iterate through pdfs in pdf directory
#         fileExtension = textFile.split(".")[-1]
#         temptextFileDir = textFileDir + "/" + textFile
#         if fileExtension == "txt":
#             textFileFilename = textFileDir + "/" + textFile
#
#             print "Converting" + textFileFilename.decode('GBK')
#
#             strs = convert(textFileFilename)  # get string of text content of textFile
#             #textFilename = txtDir + "/" + textFile + ".txt"
#             textFile = open(txtName, "a")  # make text file
#             for str in strs:
#                 textFile.write(str)  # write text to text file
#             textFile.close()
#
#         elif os.path.isdir(temptextFileDir):
#             print "Entering Dir: " + temptextFileDir.decode('GBK')
#             textFileDirName = temptextFileDir
#             #txtDirName = txtDir + "/" + textFile
#             # create same Dir for txt
#             #if not os.path.isdir(txtDirName):
#             #    os.mkdir(txtDirName)
#             # Do same in son Dir, not too many layers
#             convertMultiple(textFileDirName, txtName)
#
#
# # i : info
# # p : pdfDir
# # t = txtDir
# def main(argv):
#     # pdfDir = ""
#     # txtDir = ""
#     # try:
#     #     opts, args = getopt.getopt(argv, "ip:t:")
#     # except getopt.GetoptError:
#     #     print("pdfToT.py -p <pdfdirectory> -t <textdirectory>")
#     #     sys.exit(2)
#     # for opt, arg in opts:
#     #     if opt == "-i":
#     #         print("pdfToT.py -p <pdfdirectory> -t <textdirectory>")
#     #         sys.exit()
#     #     elif opt == "-p":
#     #         pdfDir = arg
#     #     elif opt == "-t":
#     #         txtDir = arg
#
#     pdfDir = "./novel"# + unicode(pdfDir, "UTF-8")
#     txtName = "./novelOne/novelOne.text"# + unicode(txtDir, "UTF-8")
#
#     convertMultiple(pdfDir, txtName)
#
# if __name__ == "__main__":
#     main(sys.argv[1:])





