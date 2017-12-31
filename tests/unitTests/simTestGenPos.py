#coding:utf-8

#这个脚本从同义词词林中生成正例
#一个正例有三个字段：类 同义词一 同义词二
#只选取词典中有的例子，所以要加载词典，而且和词典也相关
#估计生成一次也不会花太多时间。
import sys

reload(sys)
sys.setdefaultencoding('utf-8')

#加载词典
vocab = open("../../data/wikiNew2/wikiNew_voc")
listVocab = list(vocab)
vocab.close()
vocabSet = set(list([i.split(' ')[0] for i in listVocab]))

maxLine = 999999
#加载同义词词林
f_wood = open("../../data/simWood.txt")#这是同义词词林
listWood = list(f_wood)
f_wood.close()

#打开写入文件
f_rst = open("../../data/Same2", 'w')
cnt = 0
#遍历同义词词林
cntline = 0
for line in listWood:#词林中的每一行

    words = line.split();
    title = words[0]
    words = words[1:]
    # 对于一行
    # 不是同义行跳出
    if title.endswith('@'):
        continue
    cntline += 1
    if cntline > maxLine:
        break;
    # print title
    #对于一个词
    for word in words:

        word = word.strip()

        # 不在词典内跳出
        if not word in vocabSet:
            continue

        # print word
        # 对于另一个词
        for word2 in words:
            rst = ''
            word2 = word2.strip()
            # 相同跳出
            if word2 == word:
                continue
            # 不在词典内跳出
            if not word2 in vocabSet:
                continue

            cnt += 1
            rst += title
            rst += '\t' + word
            rst += '\t' + word2
            if cnt % 1000 == 0:
                print cnt
                print rst
            rst += '\n'
            # 生成结果，写入
            f_rst.write(rst)

f_rst.flush()
f_rst.close()



