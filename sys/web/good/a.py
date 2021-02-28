# -*- coding: UTF-8 -*-
import MySQLdb as mdb
import time
import codecs

start=time.time()
ccccc = 1
def createTrain():
    #将con设定为全局连接
    con = mdb.connect('120.26.172.66', 'prj', '8878', 'prj', charset='utf8')
    #

    #获取连接的cursor，
    cur = con.cursor()
    #创建一个数据表 writers(id,name)
    # cur.execute("DROP TABLE IF EXISTS a")
    # cur.execute("CREATE TABLE a (\
    # uid varchar(255) NOT NULL,\
    # mid varchar(255) NOT NULL,\
    # time date NOT NULL,\
    # ) ENGINE=MyISAM DEFAULT CHARSET=utf8;")
    #cur.execute("set names 'utf8'")

    f = open("D:/aaa.txt", 'r', encoding='utf-8')
    line = f.readline()               # 调用文件的 readline()方法

    print(f)

    while line:
        print('start')
        print(line)
        print('the line is over')
        cur.execute("INSERT INTO accusation(accusationName) VALUES(%s)", [line])
        con.commit()
        line = f.readline()

createTrain()
print(time.time()-start)
print('done')