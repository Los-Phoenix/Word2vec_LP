#coding:UTF8
#这个文档
import gensim
import sys

reload(sys)
sys.setdefaultencoding('utf-8')

if __name__ == '__main__':
    model = gensim.models.Word2VecWC.load("../../data/wikiNew2/wikiNew_model")

    fPos = open("../../data/Same2")
    linesPos = list(fPos)

    posList = list()

    simListPos = list()

    cnt = 0
    num = 1000000

    posSet = set()

    for line in linesPos:
        title, w1, w2 = line.split('\t')
        w1 = w1.decode()
        w2 = w2.strip().decode()
        if w1 in model and w2 in model:
            posSet.add(w1)
            posSet.add(w2)
            posList.append([w1, w2])
            simListPos.append(model.similarity(w1, w2))
            cnt += 1
        if cnt % 1000 == 0:
            print cnt

    print"posSet:", len(posSet)

    posWordList = list()

    simSet240 = set()
    simSet297 = set()
    fSimDict240 = list(open("../../data/240.txt"))
    fSimDict297 = list(open("../../data/297.txt"))
    for i in fSimDict240:
        w0, w1, sim = i.decode().split('\t')
        if w0 in model and w1 in model:
            simSet240.add(w0)
            simSet240.add(w1)
    print "simSet240:", len(simSet240)
    for i in fSimDict297:
        w0, w1, sim = i.decode().split('\t')
        if w0 in model and w1 in model:
            simSet297.add(w0)
            simSet297.add(w1)
    print "simSet297:", len(simSet297)

    print "interset240 and 297:", len(simSet297.intersection(simSet240))

    print "interSection240:", len(simSet240.intersection(posSet))
    print "interSection297:", len(simSet297.intersection(posSet))
    print "all:", len(simSet240.intersection(simSet297.intersection(posSet)))

