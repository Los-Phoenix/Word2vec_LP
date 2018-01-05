#coding:utf-8

#测验：为什么阿基米德不在里面
#是不在哪个文本里面
import sys

reload(sys)
sys.setdefaultencoding('utf-8')

def test(str, file):
    listF = list(open(file, 'r'))
    for i in listF:
        ii = i.split(' ')
        for wordRaw in ii:
            word = wordRaw.strip()
            if word == str:
                print word
                return True
    return False

path = "../../data/"
str = u'阿基米德'
# str = 'google'

print "simWoodDict", test(str, path + "simWoodDict")
print "wordSDict", test(str, path + "wordSDict")
print "titleDict", test(str, path + "titleDict")
print "legalDict", test(str, path + "legalDict")
print "slimDict", test(str, path + "slimDict")
print "wikiSegged", test(str, "../../data/wikiShieved")
print "voc", test(str, "../../data/wiki_phrase/wiki_voc")