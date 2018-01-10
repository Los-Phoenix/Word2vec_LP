#coding=utf-8

import MySQLdb

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

conn= MySQLdb.connect(
        host='192.168.8.109',
        port = 3306,
        user='root',
        passwd='1111',
        db ='wikizhjwpl',
        )
cur = conn.cursor()

# print cur.execute("select id, group_concat(inlinks) from category_inlinks GROUP BY id")
print cur.execute("select * from redirects LIMIT 0, 999999")
a = list(cur)

cur.close()
conn.commit()
conn.close()

outF = open("../../data/redi", 'w')
for i in a:
    outF.write(i[0].decode())
    outF.write('\t')
    outF.write(i[1].decode())
    outF.write('\n')

