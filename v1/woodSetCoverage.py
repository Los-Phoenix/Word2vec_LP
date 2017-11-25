#coding:utf-8
#这个脚本统计同义词同类词集合的覆盖情况
#输出同义词集合总数
#输出同类词集合总数
#集合名 type total novelHit novelMiss wikiHit wikiMiss totalHit totalMiss

#在第一步试验中：
#字典是simWoodDict
#词频表是zh_model_200_all_voc

#先把词频表装入map
#再对字典遍历

f_vocab = open("zh_novel_model_100_voc")
listVocab = list(f_vocab)
f_vocab.close()

wiki_vocab = open("zh_model_200_all_voc")
listWikiVocab = list(wiki_vocab)
wiki_vocab.close()

f_wood = open("simWood.txt")
listWood = list(f_wood)
f_wood.close()

f_woodTime = open("woodCount", 'w')

words = []
times = []

for line in listVocab:
    word, time = line.split(" ")
    words.append(word)
    times.append(time)

vocabMap = dict(zip(words, times))

words = []
times = []
for line in listWikiVocab:
    word, time = line.split(" ")
    words.append(word)
    times.append(time)
vocabWikiMap = dict(zip(words, times))

#get方法是取得，后面可以加词序，这是我们需要的！
#print vocabMap.get("pangkunyuan", 0)

for line in listWood:#词林中的每一行
    words = line.split();
    title = words[0]
    words = words[1:]
    totalCount = len(words)
    rst = ''
    if title.endswith('='):
        type = 'equal'
    elif  title.endswith('#'):
        type = 'class'
    else:
        continue

    rst = rst + title + '\t' + type + '\t' + str(totalCount)

    novelHit = 0
    novelMiss = 0
    wikiHit = 0
    wikiMiss = 0
    totalHit = 0
    totalMiss = 0

    for word in words:
        word = word.strip()
        timeNovel = vocabMap.get(word, 0)
        timeWiki = vocabWikiMap.get(word, 0)
        if timeNovel == 0:
            novelMiss += 1
        else:
            novelHit += 1
        if timeWiki == 0:
            wikiMiss += 1
        else:
            wikiHit += 1

        if timeNovel == 0 and timeWiki == 0:
            print word
            totalMiss += 1
        else:
            totalHit += 1
    rst += '\t' +  str(novelMiss)
    rst += '\t' +  str(novelHit)
    rst += '\t' +  str(wikiMiss)
    rst += '\t' +  str(wikiHit)
    rst += '\t' +  str(totalMiss)
    rst += '\t' +  str(totalHit)
    rst += '\n'

    f_woodTime.write(rst)

f_woodTime.flush()
f_woodTime.close()
