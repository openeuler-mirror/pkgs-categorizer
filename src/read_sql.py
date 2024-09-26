# SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import sqlite3
import get_dep
import os
from config import src_zero_in_list,rpm_zero_in_list,repo_type_list,fcfl_config
import pprint
import hawkey


class REPO_SQL:

    def __init__(self):
        self.sql_path = ""
        self.primary_con = ""


    def read_path(self, repoFile, repoDir, dbName):
        '''
        查询config文件下repoFile目录的*dbName* or *filelists*
        返回路径
        '''
        fcfl_config_obj = fcfl_config()
        path = repoDir +'/' + repoFile +'/'
        file_list = os.listdir(path)
        for file_name in file_list:
            if dbName == "primary":
                if "primary" in file_name:
                    return path + file_name
            elif dbName == "filelists":
                if "filelists" in file_name:
                    return path + file_name

        print("there is no file in path of %s" %path)
        return 1

    def open_sql(self,sql_path):
        self.primary_con = sqlite3.connect(sql_path)
        self.p_db = self.primary_con.cursor()


    def close_sql(self):
        """
        关闭数据库
        """
        self.primary_con.close()

    def get_rpm_primary_files_table(self):
        sql_cmd = "select pkgKey, name from files"
        res = self.p_db.execute(sql_cmd)
        items = res.fetchall()
        files_dict = {}
        for temp in items:
            if temp[0] not in files_dict.keys():
                files_dict[temp[0]]  = [(temp[1] ,"")] 
            else:
                files_dict[temp[0]].append((temp[1] ,""))
        return files_dict

    def get_rpm_files(self):
        sql_cmd = "select pkgKey ,dirname , filenames from filelist"
        res = self.p_db.execute(sql_cmd)
        items = res.fetchall()
        files_dict = {}
        for temp in items:
            if temp[0] not in files_dict.keys():
                files_dict[temp[0]]  = [(temp[1] ,temp[2])] 
            else:
                files_dict[temp[0]].append((temp[1] ,temp[2]))
        return files_dict
    
    def get_rpm_key_and_pkgname(self):
        sql_cmd = "select pkgKey , name from packages"
        res = self.p_db.execute(sql_cmd)
        items = res.fetchall()
        files_dict = {}
        for temp in items:
            files_dict[temp[0]] = temp[1]
        return files_dict

    def read_requires(self):
        """
        读取依赖列表
        返回值：依赖关系字典  key：pkgKey    value：req_list[]
        """
        flag = 0  #记录返回的名字有哪几位构成
        req_dict = {}
        sql_cmd = "select name, pkgkey, version, release, flags from requires"
        res = self.p_db.execute(sql_cmd)
        items = res.fetchall()

        for temp in items:
            flag = 0
            if temp[1] in req_dict.keys():
                req_dict[temp[1]].append((temp[0], temp[2], temp[3], temp[4]))
            else:
                req_dict[temp[1]]= [(temp[0], temp[2], temp[3], temp[4])]
        return req_dict

    def read_provides(self):
        """
        读取provides表
        获取提供列表字典    key：name    value：pkgkey
        """
        prov_dict = {}
        sql_cmd = "select name, flags, pkgkey, version, release from provides"
        res = self.p_db.execute(sql_cmd)
        items = res.fetchall()

        for temp in items:
            if temp[0] in prov_dict.keys():
                prov_dict[temp[0]].append((temp[1],temp[2]))
            else:
                prov_dict[temp[0]]=[(temp[1],temp[2])]

        return prov_dict

    def read_files(self):
        """
        读取files表
        获取files字典    key：name      value：pkgkey
        """
        files_dict = {}
        sql_cmd = "select name, pkgkey from files"
        res = self.p_db.execute(sql_cmd)
        items = res.fetchall()
        fileitem=[]

        for temp in items:
            fileitem.append(temp[1])
            files_dict[temp[0]] = fileitem

        return files_dict

    def getPkgkeyFromProvides(self,name,version,release):
        """
        获得提供该文件的包的pkgkey（源码包版本原因，会出现name相同pkgkey不同的现象，此时只取一个）
        查找文件：provides
        查找内容：二进制包名（name）
        查找依据：索引（pkgkey）
        """
        sql_cmd = "select pkgkey from provides where name = '%s' and version = '%s' and release = '%s' " % (name,version,release)
        res = self.p_db.execute(sql_cmd)
        items = res.fetchall()   
        return items

    def getPkgkeyFromProvides_name(self,name):
        """
        获得提供该文件的包的pkgkey（源码包版本原因，会出现name相同pkgkey不同的现象，此时只取一个）
        查找文件：provides
        查找内容：二进制包名（name）
        查找依据：索引（pkgkey）
        """
        sql_cmd = "select pkgkey from provides where name = '%s' " % (name)
        res = self.p_db.execute(sql_cmd)
        items = res.fetchall()
        return items

    def getPkgkeyFromProvides_like_name(self,name):
        """
        获得提供该文件的包的pkgkey（源码包版本原因，会出现name相同pkgkey不同的现象，此时只取一个）
        查找文件：provides
        查找内容：二进制包名（name）
        查找依据：索引（pkgkey）
        """
        sql_cmd = "select pkgkey from provides where name like '%%%s' " % (name)
        res = self.p_db.execute(sql_cmd)
        items = res.fetchall()
        return items

    def getPkgkeyFromProvides_version(self,name,version):
        """
        获得提供该文件的包的pkgkey（源码包版本原因，会出现name相同pkgkey不同的现象，此时只取一个）
        查找文件：provides
        查找内容：二进制包名（name）
        查找依据：索引（pkgkey）
        """
        sql_cmd = "select pkgkey from provides where name = '%s' and version = '%s' " % (name,version)
        res = self.p_db.execute(sql_cmd)
        items = res.fetchall()   #只找一个
        return items

    def getPkgkeyFromProvides_release(self,name,release):
            """
            获得提供该文件的包的pkgkey（源码包版本原因，会出现name相同pkgkey不同的现象，此时只取一个）
            查找文件：provides
            查找内容：二进制包名（name）
            查找依据：索引（pkgkey）
            """
            sql_cmd = "select pkgkey from provides where name = '%s' and release = '%s' " % (name,release)
            res = self.p_db.execute(sql_cmd)
            items = res.fetchall()
            return items

    def getPkgkeyFromFiles_name(self,name):
        """
        在files中根据路径查找pkgkey
        查找文件：files
        查找内容：二进制包名（name）
        查找依据：索引（pkgkey）
        """
        sql_cmd = "select pkgkey from files where name = '%s' " % (name)
        res = self.p_db.execute(sql_cmd)
        items = res.fetchall()
        return items


    def getPkgkeyFromFiles_like_name(self,name):
        """
        在files中根据路径查找pkgkey
        查找文件：files
        查找内容：二进制包名（name）
        查找依据：索引（pkgkey）
        """
        sql_cmd = "select pkgkey from files where name like '%%%s' " % (name)
        res = self.p_db.execute(sql_cmd)
        items = res.fetchall()   #取一个就行
        return items

    def getSNVRFromPkg(self,pkgkey):
        """
        获得源码包名
        查找文件：packages
        查找内容：源码包名
        查找依据：索引（pkgkey）
        """
        sql_cmd = "select rpm_sourcerpm,name,version, release  from packages where pkgkey = %d " % pkgkey  
        res = self.p_db.execute(sql_cmd)
        resultSet = res.fetchall()
        return resultSet

    def getSrpmName(self,pkgkey):
        """
        获得源码包名
        查找文件：packages
        查找内容：源码包名
        查找依据：索引（pkgkey）
        """
        sql_cmd = "select rpm_sourcerpm,name,version, release  from packages where pkgkey = %d " % pkgkey  
        res = self.p_db.execute(sql_cmd)
        items = res.fetchone()
        src_name = item[0]
        rpm_name = item[1] + '-' + item[2] + '-' + item[3]
        return src_name,rpm_name

    def get_depsFromRequires(self,pkgkey):
        """
        获得依赖列表
        requires
        查找内容：name
        查找依据：索引（pkgkey）
        """
        sql_cmd = "select name,version,release  from requires where pkgkey = %d " % pkgkey    
        res = self.p_db.execute(sql_cmd)
        items = res.fetchall()
        return items

    def getAllSrcrpm(self):
        """
        查找源中的所有源码包名
        """
        sql_cmd = "select distinct rpm_sourcerpm from packages"
        res = self.p_db.execute(sql_cmd)
        srpm_name = res.fetchall()
        return srpm_name

    def getRequiresName(self):
        """
        通过requires表获取rpm名，除去带有‘/’的，
        """
        sql_cmd = "select distinct name from requires where name not like '%/%' "
        res = self.p_db.execute(sql_cmd)
        item = res.fetchall()
        return item

    def getRpmInfoByRpmName(self,rpmName):
        """
        查找文件：packages
        查找内容：源码包提供的所有二进制包名、描述信息、概述
        查找依据：rpm包名（name）
        """
        sql_cmd = "select name,version,release,description,summary,pkgkey from packages where name = '%s' " % rpmName
        res = self.p_db.execute(sql_cmd)
        item = res.fetchall()
        return item

    def getRpmInfo(self,srpmName):
        """
        查找文件：packages
        查找内容：源码包提供的所有二进制包名、描述信息、概述
        查找依据：源码包名（rpm_sourcerpm）
        """
        sql_cmd = "select name,version,release,description,summary,pkgkey from packages where rpm_sourcerpm = '%s' " % srpmName
        res = self.p_db.execute(sql_cmd)
        item = res.fetchall()
        return item

    def read_all_rpm_info(self):
        """
        查找文件：packages
        查找内容：源码包提供的所有二进制包名、描述信息、概述
        查找依据：源码包名（rpm_sourcerpm）
        """
        sql_cmd = "select pkgkey, rpm_sourcerpm, name,version,release,description,summary from packages" 
        res = self.p_db.execute(sql_cmd)
        item = res.fetchall()
        return item


    def exec_repo_sql(self,cmd):
        res = self.p_db.execute(cmd)
        item = res.fetchall()
        return item


class REPO_DEP:
    """
    生成 dot 使用
    """
    def __init__(self):
        self.repo_obj = REPO_SQL()

    def get_all_dicks(self, repoType, repoDir):
        """
        调用get_all_dicts_by_db_path,获取4个表中的字典
        """
        path = self.repo_obj.read_path(repoType, repoDir, "primary")
        return self.get_all_dicts_by_db_path(path)

    def get_all_dicts_by_db_path(self,path):
        '''
        取得4个表中的字典
        传入参数为path
        '''
        self.repo_obj.open_sql(path)
        req_dict= self.repo_obj.read_requires()
        prov_dict = self.repo_obj.read_provides()
        files_dict = self.repo_obj.read_files()
        items = self.repo_obj.read_all_rpm_info()
        pack_dict = {}
        for temp in items:
            temp_list = []
            temp_list.append(temp[1])
            temp_list.append(temp[2])
            temp_list.append(temp[3])
            temp_list.append(temp[4])
            temp_list.append(temp[5])
            temp_list.append(temp[6])
            pack_dict[temp[0]] = temp_list
        self.repo_obj.close_sql()

        return req_dict,prov_dict,files_dict,pack_dict

    def get_srcPackage_directly_dep(self,srcName,sql_path):
        """
        源码包直接依赖
        在根据源码包查找依赖时使用
        """
        req_list = []
        src_list = []
        self.repo_obj.open_sql(sql_path)
        rpm_info_list= self.repo_obj.getRpmInfo(srcName)
        for rpminfo in rpm_info_list:
            if get_dep.rpm_is_need_remove(rpminfo[0]):
                continue
            temp_list = self.repo_obj.get_depsFromRequires(rpminfo[5])
            req_list.extend(temp_list)
        req_list = list(set(req_list))

        for rpmName in req_list:
            name = rpmName[0]
            version = rpmName[1]
            release = rpmName[2]

            if name and version and release:
                p_pkgkey = self.repo_obj.getPkgkeyFromProvides(name,version,release)
            elif name and version:
                p_pkgkey = self.repo_obj.getPkgkeyFromProvides_version(name,version)
            elif name and release:
                p_pkgkey = self.repo_obj.getPkgkeyFromProvides_release(name,release)
            else:
                p_pkgkey = self.repo_obj.getPkgkeyFromFiles_name(name)

            if p_pkgkey:
                src_name = self.repo_obj.getSrpmName(p_pkgkey)[0]
                if src_name != srcName:
                    src_list.append(src_name)
            else:
                f_pkgkey = self.repo_obj.getPkgkeyFromFiles_name(rpmName)
                if f_pkgkey:
                    src_name = self.repo_obj.getSrpmName(f_pkgkey)
                    if src_name != srcName:
                        src_list.append(src_name)
        src_list = list(set(src_list))
        return src_list

    def get_binPackage_directly_dep(self,name,sql_path):
        """
        源码包直接依赖
        在根据源码包查找依赖时使用
        """
        req_list = []
        src_list = []
        self.repo_obj.open_sql(sql_path)
        rpm_info_list= self.repo_obj.getRpmInfoByRpmName(name)
        for rpminfo in rpm_info_list:
            if get_dep.rpm_is_need_remove(rpminfo[0]):
                continue
            temp_list = self.repo_obj.get_depsFromRequires(rpminfo[5])
            req_list.extend(temp_list)
        req_list = list(set(req_list))

        for rpmName in req_list:
            name = rpmName[0]
            version = rpmName[1]
            release = rpmName[2]

            if name and version and release:
                p_pkgkey = self.repo_obj.getPkgkeyFromProvides(name,version,release)
            elif name and version:
                p_pkgkey = self.repo_obj.getPkgkeyFromProvides_version(name,version)
            elif name and release:
                p_pkgkey = self.repo_obj.getPkgkeyFromProvides_release(name,release)
            else:
                p_pkgkey = self.repo_obj.getPkgkeyFromFiles_name(name)

            if p_pkgkey:
                src_name = self.repo_obj.getSrpmName(p_pkgkey)[0]
                if src_name != srcName:
                    src_list.append(src_name)
            else:
                f_pkgkey = self.repo_obj.getPkgkeyFromFiles_name(rpmName)
                if f_pkgkey:
                    src_name = self.repo_obj.getSrpmName(f_pkgkey)
                    if src_name != srcName:
                        src_list.append(src_name)
        src_list = list(set(src_list))
        return src_list

    def search_src_name_files(self,sql_path,req_n, req_v, rpm_r, rpm_f, rpm_nvr, repoDir):
        """
        跨文件查找依赖
        输入参数：sql_path  数据库路径
                rpm_name  查找的二进制名
        """
        path = self.repo_obj.read_path(sql_path, repoDir, "primary")
        self.repo_obj.open_sql(path)
        dep_src_names = {}
        dep_rpm_names = {}

        if  req_n and req_v and rpm_r:
            p_pkgkeys = self.repo_obj.getPkgkeyFromProvides(req_n,req_v,rpm_r)
            f_pkgkeys = self.repo_obj.getPkgkeyFromFiles_name(req_n)
        elif req_n and req_v and len(str(rpm_r))==0:
            p_pkgkeys = self.repo_obj.getPkgkeyFromProvides_version(req_n,req_v)
            f_pkgkeys = self.repo_obj.getPkgkeyFromFiles_name(req_n)
        elif req_n and rpm_r and len(str(req_v))==0:
            p_pkgkeys = self.repo_obj.getPkgkeyFromProvides_release(req_n,rpm_r)
            f_pkgkeys = self.repo_obj.getPkgkeyFromFiles_name(req_n)
        else:
            p_pkgkeys = self.repo_obj.getPkgkeyFromProvides_name(req_n,req_v,rpm_r, rpm_f, rpm_nvr)
            f_pkgkeys = self.repo_obj.getPkgkeyFromFiles_name(rpm_nvr)


        if p_pkgkeys:
            for p_pkgkey in p_pkgkeys:
                dep_src_name,dep_rpm_name = self.repo_obj.getSrpmName(p_pkgkey)
                dep_src_names.add(dep_src_name)
                dep_rpm_names.add(dep_rpm_name)
            
        elif f_pkgkeys:
            for f_pkgkey in f_pkgkeys:
                dep_src_name,dep_rpm_name = self.repo_obj.getSrpmName(f_pkgkey)
                dep_src_names.add(dep_src_name)
                dep_rpm_names.add(dep_rpm_name)
        else:
            dep_src_name = 0
            dep_rpm_name = 0

        self.repo_obj.close_sql()
        return dep_src_names,dep_rpm_names

class REPO_CSV:
    """
    生成 csv 使用
    """
    def __init__(self):
        self.repo_obj = REPO_SQL()


    def rpm_is_nead_remove_for_csv(self,rpmName):
        if  rpmName.endswith("-doc") == True or \
            (rpmName.endswith("-devel") == True and rpmName.find("golang") == -1 )or \
            rpmName.find("glibc-langpack") != -1 or \
            rpmName.find("glibc-minimal") != -1 or \
            rpmName.find("glibc-all") != -1 or \
            rpmName.find("glibc-benchtests") != -1 or \
            rpmName.endswith("-tests") == True or \
            rpmName.endswith("-test") == True or \
            rpmName.endswith("-debuginfo") == True:
            return 1
        return 0

    def get_one_rpm_info(self,repo_type,src_name, repoDir):
        """
        获取一个源码包的所有二进制包描述信息
        """
        info_dict = {}
        path = self.repo_obj.read_path(repo_type,repoDir, "primary")
        self.repo_obj.open_sql(path)
        rpm_info_list = self.repo_obj.getRpmInfo(src_name) 
        rpm_dep_list = self.repo_obj.getRequiresName()
        self.repo_obj.close_sql()

        for rpm_info in rpm_info_list:
            name,version,release,description,summary, pkgid = rpm_info
            if self.rpm_is_nead_remove_for_csv(name) :
                continue
            if name in rpm_dep_list:
                in_degree = 1
            else:
                in_degree = 0
            name = name + '-' + version + '-' + release
            info_dict[name] = (description,summary,in_degree)
        return info_dict

    def get_one_repo_rpm_description(self,repo_type, repoDir):
        """
        获取一个repo中全部rpm信息
        """
        dict_list = []
        path = self.repo_obj.read_path(repo_type, repoDir, "primary")
        self.repo_obj.open_sql(path)
        rpm_info_list = self.repo_obj.read_all_rpm_info()
        self.repo_obj.close_sql()

        for rpm_info in rpm_info_list:

            pid ,rpm_sourcerpm,name,version,release,description,summary= rpm_info
            # if self.rpm_is_nead_remove_for_csv(name) :
            #     continue
            # else:
            info_dict = {}
            info_dict["rpm_name"] = ''
            info_dict["zero_description"] = ''
            info_dict["zero_summary"] = ''
            info_dict["description"] = ''
            info_dict["summary"] = ''
            info_dict["src_name"] = ''

            info_dict["rpm_name"] = name + '-'+ version + '-' + release
            info_dict["src_name"] = rpm_sourcerpm
            if name in rpm_zero_in_list:
                info_dict["zero_description"] = description
                info_dict["zero_summary"] = summary
            else:
                info_dict["description"] = description
                info_dict["summary"] = summary
            dict_list.append(info_dict)
        return dict_list


    def get_all_repo_dict(self, repoDir):
        '''
        return src_name:{bin_name,desc, summ)
        '''
        all_src_dict = {}
        for repo_type in repo_type_list:
            path = self.repo_obj.read_path(repo_type, repoDir, "primary")
            self.repo_obj.open_sql(path)
            cursor = self.repo_obj.getAllSrcrpm() #源码包列表

            for row in cursor:   #row in cursor.
                src_name = row[0]       #row[0] is colum 0
                if src_name in all_src_dict.keys():
                    temp_set = all_src_dict[src_name]
                else:
                    temp_set = set()

                rpm_info_list = self.repo_obj.getRpmInfo(src_name)  #源码包提供的二进制包列表信息
                for rpm_info in rpm_info_list:
                    name,version,release,description,summary,pkgid = rpm_info
                    temp_tuple = (name,description,summary)
                    temp_set.add(temp_tuple)
                tmp=hawkey.split_nevra(src_name).name    
                all_src_dict[tmp] = temp_set

        return all_src_dict

    def get_all_rpm_repo_dict_for_test(self, repoDir):

        all_src_dict = {}
        for repo_type in repo_type_list:
            path = self.repo_obj.read_path(repo_type, repoDir, "primary")
            self.repo_obj.open_sql(path)
            rpm_info_list = self.repo_obj.read_all_rpm_info() #源码包列表
            self.repo_obj.close_sql()

            for rpm_info in rpm_info_list:
                pid ,rpm_sourcerpm,name,version,release,description,summary= rpm_info
                if get_dep.rpm_is_need_remove(name):
                    continue
                temp_set = set() 
                #rpmname = name + '-'+ version + '-' + release
                rpmname = name
                srcrpms = rpm_sourcerpm.split("-")[-2]
                sName = rpm_sourcerpm.split("-%s" % srcrpms)[0]
                temp_set = (description,summary,sName,version,release)
                all_src_dict[rpmname] = temp_set

        return all_src_dict
       
    def get_all_rpm_repo_dict(self, repoDir):
        all_rpm_dict = {}
        '''
        读全部repo,返回{rpmname:(desc,summ,srpmname, version, release, pid)}
        '''
        for repo_type in repo_type_list:
            path = self.repo_obj.read_path(repo_type, repoDir, "primary")
            self.repo_obj.open_sql(path)
            rpm_info_list = self.repo_obj.read_all_rpm_info() #二进制包列表
            self.repo_obj.close_sql()

            for rpm_info in rpm_info_list:
                pid ,rpm_sourcerpm,name,version,release,description,summary= rpm_info
                if get_dep.rpm_is_need_remove(name):
                    continue
                temp_set = set() 
                rpmname = name
                temp_set = {(description,summary,rpm_sourcerpm, version, release, pid)}
                if rpmname not in all_rpm_dict:
                    all_rpm_dict[rpmname] = temp_set
                else:
                    all_rpm_dict[rpmname] = all_rpm_dict[rpmname]|temp_set
        return all_rpm_dict

    def get_one_repo_src_description(self,repo_type, repoDir):
        dict_list=[]
        path = self.repo_obj.read_path(repo_type, repoDir, "primary")
        self.repo_obj.open_sql(path)
        srpm_name_list = self.repo_obj.getAllSrcrpm() #源码包列表

        for src_name in srpm_name_list:
            src_name = src_name[0]
            info_dict = {}
            info_dict["name"] = src_name
            info_dict["description"] = ''
            info_dict["summary"] = ''
            info_dict["zero_description"]  = ''
            info_dict["zero_summary"] = ''

            rpm_info_list = self.repo_obj.getRpmInfo(src_name)  #源码包提供的二进制包列表信息

            if src_name in src_zero_in_list:
                for rpm_info in rpm_info_list:
                    name,version,release,description,summary,pkgid = rpm_info
                    if self.rpm_is_nead_remove_for_csv(name) :
                        continue 
                    if description and description not in info_dict["zero_description"]:     
                        info_dict["zero_description"] += "\n"+ description  
                    if summary and summary not in info_dict["zero_summary"]:
                        info_dict["zero_summary"] += "\n" +summary
            else:
                for rpm_info in rpm_info_list:
                    name,version,release,description,summary,pid = rpm_info
                    if self.rpm_is_nead_remove_for_csv(name) :
                        continue 
                    if description and description not in info_dict["description"]:     
                        info_dict["description"] += "\n"+ description  
                    if summary and summary not in info_dict["summary"]:
                        info_dict["summary"] += "\n" +summary
            dict_list.append(info_dict)

        self.repo_obj.close_sql()
        return dict_list

def test():
    a = REPO_SQL()
    a.read_path("aaa","repoDir", "bbb")
    print("111")
    return 1,2


if __name__ == "__main__":
    csv_obj = REPO_CSV()
    csv_obj.get_one_repo_rpm_description("HighAvailability")
