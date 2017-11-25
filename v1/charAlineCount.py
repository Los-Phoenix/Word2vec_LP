#coding:utf-8
#这个文件统计 字向量和词向量里面的重合部分。
#计划使用集合的交并等操作。
#python集合在哪里呢？
#自带set类，似乎很好用

wordList = list(open("zh_model_200_all_voc"))
wordClearList = [i.split(" ")[0] for i in wordList]
wordSet = set(wordClearList)

charList = list(open("zh_model_200_all_voc_cha"))
charClearList = [i.split(" ")[0] for i in charList]
charSet = set(charClearList)

interSet = wordSet.intersection(charSet)

print len(wordSet)
print len(charSet)
print len(interSet)

for interChar in charSet.difference(wordSet):#这是300多个不能单独作为词的字
     print interChar
