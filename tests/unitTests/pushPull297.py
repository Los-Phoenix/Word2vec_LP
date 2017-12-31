#coding:utf-8
#这个文件是为了：
#读出SIM297
#转换成测试集
#计算协方差
#计算皮尔逊相关系数

fSimDict297 = list(open("../../data/297.txt"))
legalSimSet297 = set()
for i in fSimDict297:
    w0, w1, sim = i.decode().split('\t')
    legalSimSet297.add(w0)
    legalSimSet297.add(w1)
    print sim

print len(legalSimSet297)