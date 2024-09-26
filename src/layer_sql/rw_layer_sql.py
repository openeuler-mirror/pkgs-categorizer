# SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import sqlite3 
import sys
from enum import Enum 
sys.path.append("..")
sys.path.append(".")
from config import fcfl_config

# 继承枚举类，存储分类的信息
class creat_tables(Enum):
    srcrpm_info = '''create table srcrpm_info (pkgID            INTEGER     PRIMARY KEY  NOT NULL,
                                         pkgName                TEXT                    NOT NULL,
                                         zero_description       TEXT ,
                                         zero_summary           TEXT ,
                                         description            TEXT ,
                                         summary                TEXT ,
                                         primary_rpm_name       TEXT ,
                                         primary_description    TEXT ,
                                         primary_summary        TEXT ,
                                         rpm_name               TEXT);'''
                                    

    dependence = '''CREATE TABLE dependence (pkgName TEXT NOT NULL,
                                 depName TEXT NOT NULL);'''

    rpm_info = '''CREATE TABLE rpm_info (pkgID              INTEGER     PRIMARY KEY     NOT NULL,
                                        pkgName             TEXT NOT NULL ,
                                        srcPkgName          TEXT NOT NULL ,
                                        srcPkgVersion       TEXT NOT NULL ,
                                        srcPkgRelease       TEXT NOT NULL ,
                                        zero_description    TEXT  ,
                                        zero_summary        TEXT  ,
                                        description         TEXT  ,
                                        summary             TEXT  );'''

    anchored = '''CREATE TABLE anchored (pkgName           TEXT PRIMARY KEY NOT NULL ,
                                      layer             TEXT             NOT NULL ,
                                      classification    TEXT             NOT NULL);'''

    layer_class_info = '''CREATE TABLE layer_class_info (pkgName           TEXT PRIMARY KEY    NOT NULL ,
                                              layer             TEXT                NOT NULL ,
                                              classification    TEXT                NOT NULL ,
                                              manimark          INT                 NOT NULL);'''


    domain = '''CREATE TABLE domain (pkgName           TEXT PRIMARY KEY NOT NULL ,
                                      domain_info           TEXT ,
                                      manimark              INT          NOT NULL);'''

    function_class = ''' CREATE TABLE function_class (pkgName      TEXT,
                                                      format       TEXT,
                                                      genaric      TEXT,
                                                      graphic      TEXT,
                                                      manager      TEXT,
                                                      network      TEXT,
                                                      other        TEXT,
                                                      virt         TEXT,
                                                      device       TEXT,
                                                      storage      TEXT,
                                                      media        TEXT);'''

    
# 继承枚举类，存储分类的信息
class insert_tables(Enum):
    srcrpm_info = "insert into srcrpm_info (pkgName,zero_description,zero_summary,description,summary,primary_rpm_name,primary_description,primary_summary,rpm_name) VALUES (?,?,?,?,?,?,?,?,?);"
    dependence = "insert into dependence values (?,?);" 
    rpm_info = "insert into rpm_info (pkgName,srcPkgName,srcPkgVersion,srcPkgRelease,zero_description,zero_summary,description,summary) values (?,?,?,?,?,?,?,?);"
    anchored = "insert into anchored values(?,?,?);" 
    layer_class_info = 'insert into layer_class_info values (?,?,?,?);'
    domain = 'insert into domain values (?,?,?);'
    function_class = 'insert into function_class values(?, ?, ? , ?, ?, ?, ?, ?, ?, ?, ?, ?);'

# 继承枚举类，存储分类的信息
class update_tables(Enum):
    layer_class_info = "update layer_class_info set layer = '%s', classification = '%s', manimark = 1 where pkgName = '%s'"
    anchored = "update anchored set layer = '%s', classification = '%s' where pkgName = '%s'"
    srcrpm_info = "update srcrpm_info set zero_description = '%s', zero_summary = '%s', description = '%s', summary = '%s' where pkgName = '%s'"
    domain = "update domain set domain_info = '%s', manimark = 1 where pkgName = '%s' "
    function_class = "update function_class set '%s' = '%s' where pkgName = '%s'"

class selete_tables(Enum):
    select_deps = "select * from dependence where pkgName = '%s' ;"
    select_dependence = "select * from  dependence where pkgName = '%s' "
    select_description = "select * from srcrpm_info where pkgName = '%s' "
    select_description_rpm = "select * from rpm_info where pkgName = '%s' "
    select_layer_class_info = "select * from  layer_class_info where pkgName = '%s' "
    select_domain = "select * from domain where domain_info = '%s' "
    select_function_class = "select * from function_class where package = '%s' "

class selete_all_tables(Enum):
    select_srcrpm_info = "select * from  srcrpm_info"
    select_dependence = "select * from  dependence"
    select_rpm_info = "select * from  rpm_info"
    select_layer_class_info =  "select * from  layer_class_info"  
    select_dep_srcName =  "select distinct pkgName from dependence "
    select_domain  = "select * from domain" 
    select_function_class = "select * from function_class"

class base_oprate_with_db:
    def __init__(self):
        fcfl_cfg= fcfl_config()
        print( fcfl_cfg.get_storage_path())
        self.layer_db_path = fcfl_cfg.get_storage_path()+'layerinfo.db'
        print("dbpath",self.layer_db_path)
        self.l_con = sqlite3.connect(self.layer_db_path)
        self.l_db = self.l_con.cursor()

    def open_layer_db(self):
        '''
        打开数据库
        '''
        self.l_con = sqlite3.connect(self.layer_db_path)
        self.l_db = self.l_con.cursor()

    def close_layer_db(self):
        '''
        关闭数据库
        '''
        self.l_con.close()

    def value_is_in_table(self,ntype,value,table):
        """
        判断某值是否在某表中
        """
        sql_cmd = "select count(*) from %s where %s = '%s'"  % (table,ntype,value)
        res = self.l_db.execute(sql_cmd)
        item = res.fetchone()
        return item[0]

    def get_maxPkgid(self,table):
        """
        获取表中存储数据的个数
        """
        sql_cmd = "select  count(*) from %s" %(table)
        res = self.l_db.execute(sql_cmd)
        item = res.fetchone()
        return item[0]

    def create_table(self,table_name):
        """
        create
        """
        select_str = "select count(*) from sqlite_master where name = '%s'" % table_name
        res = self.l_db.execute(select_str)
        item = res.fetchone()
        if item[0]:
            exist_table_cmd = "drop table %s" % table_name
            self.l_db.execute(exist_table_cmd)

        for name,member in creat_tables.__members__.items():
            if table_name == str(name):
                execute_cmd = member.value
                break

        self.l_db.execute(execute_cmd)
        self.l_con.commit()

    """
    insert
    """
    def insert_pkg_into_table(self,table_name,insert_info):

        for name,member in insert_tables.__members__.items():
            if table_name == str(name):
                sql_cmd = member.value 
                break
        try:
            self.l_db.execute(sql_cmd,insert_info)
            self.l_con.commit()
            return 0
        except:
            return 1

    def insert_huge_date_into_table(self,table_name,insert_info_list):
        #self.clear_table(table_name)
        print("in insert_huge_date_into_table: %s" %str(table_name))
        """
        向表中填入大量数据
        """
        for name,member in insert_tables.__members__.items():
            if table_name == str(name):
                sql_cmd = member.value 
                break
        try:
            print("sql:%s" %(sql_cmd))
            self.l_db.executemany(sql_cmd,insert_info_list)
            self.l_con.commit()
            print("instert huge ok!")
            return 0
        except:
            print("instert huge error!")
            self.l_con.rollback()
            return 1

    """
    update
    """
    def update_pkg_in_table(self,table_name,update_info):

        for name,member in update_tables.__members__.items():
            if table_name == str(name):
                sql_cmd = member.value % update_info
                break
        try:
            self.l_db.execute(sql_cmd)
            self.l_con.commit()
            return 0
        except:
            print("insert %s failed" %table_name)
            return 1

    def delete_pkgs(self,tablename,pkgname):
        """
        delete
        """
        #print(tablename,pkgname)
        del_cmd = "delete from '%s' where pkgName = '%s'" %(tablename,pkgname)
        try:    
            self.l_db.execute(del_cmd)
            self.l_con.commit()
            return 0
        except:
        #l_con.rollback()
            #print("glibc %s delete failed",tablename)
            return 1

    def delete_dep_pkgs(self,pkgname):
        """
        delete
        """
        #print(tablename,pkgname)
        del_cmd = "delete from dependence where depName = '%s'" %(pkgname)
        try:    
            self.l_db.execute(del_cmd)
            self.l_con.commit()
            return 0
        except:
        #l_con.rollback()
            #print("glibc %s delete failed",tablename)
            return 1
 
    def delete_src_pkgs(self,pkgname):
        """
        delete
        """
        #print(tablename,pkgname)
        del_cmd = "delete from rpm_info where srcPkgName = '%s'" %(pkgname)
        try:   
            self.l_db.execute(del_cmd)
            self.l_con.commit()
            return 0
        except:
        #l_con.rollback()
            #print("glibc %s delete failed",tablename)
            return 1

    def clear_table(self,tableName):
        sql_cmd = "delete from %s" %tableName
        try:
            self.l_db.execute(sql_cmd)
            self.l_con.commit()
            return 0
        except:
            #l_con.rollback()
            print("%s celan failed" %tableName) 
            return 1

    """
    select
    """
    def select_from_table(self,selete_opreate,selete_info):
        for name,member in selete_tables.__members__.items():
            if selete_opreate == str(name):
                sql_cmd = member.value % selete_info
                break
        try:
            res = self.l_db.execute(sql_cmd)
            item = res.fetchall()
            return item
        except:
            print("selete %s failed" % selete_opreate)
            return 1

    def select_all_from_table(self,selete_opreate):
        for name,member in selete_all_tables.__members__.items():
            if selete_opreate == str(name):
                sql_cmd = member.value 
                break
        try:
            #print(sql_cmd)
            res = self.l_db.execute(sql_cmd)
            item = res.fetchall()
            return item
        except:
            #l_con.rollback()
            print("selete %s failed" %selete_opreate)
            return 1


if __name__ == "__main__":
    db0 = base_oprate_with_db()
    for name,member in creat_tables.__members__.items():
        db0.create_table(name)
    db0.insert_pkg_into_table('dependence',('a', 'b'))
    db0.insert_pkg_into_table('dependence',('a', 'c'))
    db0.insert_pkg_into_table('dependence',('b', 'c'))
