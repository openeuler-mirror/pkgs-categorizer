# SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import srpm_dep as sDep
import read_sql as sql
import config 

class pkgInfo:
    def __init__(self,pkgName):
        self.name = pkgName
        self.dep = []
        self.zeroInNodeDes = ""
        self.zeroInNodeSummary = ""
        self.Des = ""
        self.summary = ""

    def getPkgName(self):
        return self.name
    
    def getDep(self):
        return self.dep

    def getZeroInNodeDes(self):
        return self.zeroInNodeDes
    
    def getZeroInNodeSummary(self):
        return self.zeroInNodeSummary

    def getDes(self):
        return self.Des
    
    def getSummary(self):
        return self.summary

    def setdep(self,deplist):
        self.dep = deplist

    def setZeroInNodeDes(self,description):
        self.zeroInNodeDes = self.zeroInNodeDes + description
    
    def setZeroInNodeSummary(self,summary):
        self.zeroInNodeSummary += summary

    def setDes(self,description):
        self.Des += description
    
    def setSummary(self,summary):
        self.summary += summary


class pkgs:
    def __init__(self):
        self.pkgslist = []

    def updataPkgslist(self,pkgInfollist):
        self.pkgslist.extend(pkgInfollist)


    def getSinglePKgInfo(self,srpmName):
        zero_des_str = ""
        zero_summary_str = ""
        des_list_str = ""
        summary_list_str = ""

        table_list = [] #存放requires表中的rpm名
        sql.open_db()
        reqtable_name = sql.getRequiresName()  #requires中的所有rpmName，不包含带有路径的
        reqtable_name = list(reqtable_name)
        for name in reqtable_name:
            table_list.append(name[0])

        rpm_list = sql.getRpmInfo(srpmName)  #源码包提供的二进制包列表
        sql.close_db()
        for rpm in rpm_list:
            rpmName = rpm[0]
            des = rpm[1]
            summary = rpm[2]

            if  rpmName.endswith("-doc") == True or \
                (rpmName.endswith("-devel") == True and rpmName.find("golang") == -1 )or \
                rpmName.find("glibc-langpack") != -1 or \
                rpmName.find("glibc-minimal") != -1 or \
                rpmName.find("glibc-all") != -1 or \
                rpmName.find("glibc-benchtests") != -1 or \
                rpmName.endswith("-tests") == True or \
                rpmName.endswith("-test") == True or \
                rpmName.endswith("-debuginfo") == True:
                
                continue

            if rpmName not in table_list:
                if des:     
                    zero_des_str = zero_des_str + "\n" + des 
                if summary:
                    zero_summary_str = zero_summary_str + "\n" + summary + '.'
                
            else:
                if des:
                    des_list_str = des_list_str + "\n" + des 
                if summary:
                    summary_list_str = summary_list_str + "\n" + summary + '.'
                    
        return zero_des_str,zero_summary_str,des_list_str,summary_list_str


    def setPkgInfo(self,srcRpmName):
        name = sDep.srpmName_analyze(srcRpmName[0])
        info = pkgInfo(name)
        info.dep = sDep.get_srpm_dep_without_V(srcRpmName)
        info.zeroInNodeDes,info.zeroInNodeSummary,info.Des,info.summary = self.getSinglePKgInfo(srcRpmName)
        return info


    def generatePkgsInfo(self):
        infoList = []
        sql.open_db()
        srpmNames = sql.getAllSrcrpm()
        sql.close_db()
        for srpmName in srpmNames:
            Info = self.setPkgInfo(srpmName)
            infoList.append(Info)
        return infoList



def getAllPkgsInfo(sql_name):
    pkgsInfo = pkgs()
    if sql_name == "ALL":
        sql_type = ["base","app","PT","plus","DDE","ext","HA","exp"]
        for type in sql_type:
            config.db_type = type 
            sql.select_openSql(config.db_type) 
            templist = pkgsInfo.generatePkgsInfo()
            pkgsInfo.updataPkgslist(templist)

    else:
        config.db_type = sql_name    
        #选择开启的数据库
        sql.select_openSql(config.db_type) 
        templist = pkgsInfo.generatePkgsInfo()
        pkgsInfo.updataPkgslist(templist)

    return pkgsInfo

pkgsinfo = getAllPkgsInfo("HA") 
print(pkgsinfo.pkgslist[1].name)
print(pkgsinfo.pkgslist[1].dep)
print(pkgsinfo.pkgslist[1].zeroInNodeDes)
print(pkgsinfo.pkgslist[1].zeroInNodeSummary)
print(pkgsinfo.pkgslist[1].Des)
print(pkgsinfo.pkgslist[1].summary)


