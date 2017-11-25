#coding:utf-8
#这个脚本统计词频对字典的覆盖情况
#在第一步试验中：
#字典是simWoodDict
#词频表是zh_model_200_all_voc

#先把词频表装入map
#再对字典遍历

f_vocab = open("zh_novel_model_100_voc")
listVocab = list(f_vocab)
f_vocab.close()

f_dict = open("simWoodDict")
listDict = list(f_dict)
f_dict.close()

f_dicttime = open("dictCountNovel", 'w')

words = []
times = []

for line in listVocab:
    word, time = line.split(" ")
    words.append(word)
    times.append(time)

vocabMap = dict(zip(words, times))
#get方法是取得，后面可以加词序，这是我们需要的！
#print vocabMap.get("pangkunyuan", 0)

cntZero = 0
for word in listDict:#词林中的每个单词
    word = word.strip()
    time = vocabMap.get(word, 0)
    if time == 0:
        cntZero += 1
    f_dicttime.write(word+ "\t" + time.__str__().strip() + "\n")
print cntZero
f_dicttime.flush()
f_dicttime.close()
