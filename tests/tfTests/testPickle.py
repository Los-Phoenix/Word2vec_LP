
import pickleXY as pxy

vocab1, embd1, y1, x1, x_other1 = pxy.loadAll()

for i in vocab1:
    print i

print(embd1[0])
print len(vocab1)

print y1[1]
print x1[0]
print(x_other1[0])
