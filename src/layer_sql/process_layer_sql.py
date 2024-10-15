# SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import sys
sys.path.append(".")
sys.path.append("..")

from  layer_sql.rw_layer_sql import base_oprate_with_db
import pprint

table_name = ["srcrpm_info","dependence","rpm_info","anchored","layer_class_info","domain", "function_class",]
src_table_name = ["srcrpm_info", "dependence", "anchored", "layer_class_info", "domain" ,"src_function_class"]

class STORAGE_SQL:
    def __init__(self):
        self.l_sql = base_oprate_with_db()

    def open_Ldb(self):
        """
        打开数据库
        """
        self.l_sql.open_layer_db()

    def close_ldb(self):
        """
        关闭数据库
        """
        self.l_sql.close_layer_db()

    def create_table(self):
        """
        创建数据库表
        """
        self.open_Ldb()
        for name in table_name:
            self.l_sql.create_table(name)
        self.close_ldb()
    
    def create_one_table(self,name):
        """
        创建一个表    
        """
        self.open_Ldb()
        self.l_sql.create_table(name)
        self.close_ldb()

    '''
    insert information to domain table
    '''
    def importDomainInfo(self,domain_info_dict):
        info_list = []
        for name in domain_info_dict:
            cl_info = domain_info_dict[name]
            temp = (name,cl_info[0],cl_info[1])
            info_list.append(temp)

        self.open_Ldb()

        res = self.l_sql.insert_huge_date_into_table("domain",info_list)
        if res:
            self.close_ldb()
            return 1
        self.close_ldb()

    def importPkgLayerClassInfo(self,cl_info_dict):
        """
        存储软件包分层分类信息
        """
        info_list = []
        for name in cl_info_dict:
            cl_info = cl_info_dict[name]
            temp = (name,cl_info[0],cl_info[1],cl_info[2])
            info_list.append(temp)
            
        self.open_Ldb()
        res = self.l_sql.insert_huge_date_into_table("layer_class_info",info_list)
        if res:
            self.close_ldb()
            print("inster layer_class_info failed")
            return 1
        self.close_ldb()
        return 0

    def loadDBPkgLayerClassificationInfo(self):
        """
        读取软件包分层分类信息
        """
        cl_info_dict = {}
        self.open_Ldb()
        items = self.l_sql.select_all_from_table("select_layer_class_info")
        self.close_ldb()
        
        for item in items:
            cl_info_dict[item[0]] = (item[1],item[2],item[3])

        return cl_info_dict

    def updatePkgLayerClassification(self,pkg, layer, classification):
        """
        更新分层分类表
        返回值： 0 成功， 1 失败 
        """
        self.l_sql.open_layer_db()
        res = self.l_sql.value_is_in_table("pkgName",pkg,"layer_class_info")
        #软件包存在在数据库中 更新
        if res:
            print("%s package alreadly in db" % pkg) 
            res = self.l_sql.update_pkg_in_table("layer_class_info",(layer,classification,pkg))
            #res = self.l_sql.update_layer_class_info(pkg,layer,classification)
        #软件包不再数据库中  添加
        else:
            print("%s package  not in db " % pkg) 
            #res = self.l_sql.insert_layer_class_info(pkg,layer,classification,1)
            res = self.l_sql.insert_pkg_into_table("layer_class_info",(pkg,layer,classification,1))
        if res:
            print("update %s package fcfl info failed" % pkg) 
            self.close_ldb()
            return 1
        self.close_ldb()
        return 0

    # def deletePkgLayerClass(self,pkg):
    #     """
    #     删除软件包分层分类信息
    #     返回值： 0 成功， 1 失败 
    #     """
    #     self.open_Ldb()
    #     res = self.l_sql.value_is_in_table("pkgName",pkg,"layer_class_info")
    #     if res:
    #         res = self.l_sql.delete_pkgs("layer_class_info",pkg)
    #         self.close_ldb()
    #         return res
    #     else:
    #         print("No %s package layer classification information" %pkg) 
    #         self.close_ldb()
    #         return 1

    
    """
    依赖关系
    """
    def savePkgDepsInfo(self,dep_dict):
        """
        存入软件包依赖关系
        参数：dep_dict 依赖字典
        返回值：0 成功， 1 失败 
        """
        list_info = []
        for pkgName in dep_dict:
            if dep_dict[pkgName] == ():
                tmp = (pkgName,'')
                list_info.append(tmp)
            else :
                for depName in dep_dict[pkgName]:
                    temp = (pkgName,depName)
                    list_info.append(temp)

        self.open_Ldb()
        res = self.l_sql.insert_huge_date_into_table("dependence",list_info)
        if res == 1:
            self.close_ldb()
            return 1
        self.close_ldb()
        return 0

    def readPkgDepsInfo(self,pkg):
        """
        读取软件包 依赖关系
        参数：要查询的软件包名
        返回值：成功 软件包依赖，  失败 1
        """
        self.l_sql.open_layer_db()
        res = self.l_sql.value_is_in_table("pkgName",pkg,"dependence")

        if res:
            res = self.l_sql.select_from_table('select_deps',pkg)
            self.close_ldb()
            return res
        else:
            print("There is no dependence in the %s package" % pkg) 
            self.close_ldb()
            return 1

    def deletePkgDepsInfo(self,pkg):
        """
        删除软件包依赖信息
        参数：软件包名
        返回值：0 删除成功， 1  删除失败
        """
        self.open_Ldb()
        res = self.l_sql.value_is_in_table("pkgName",pkg,"dependence")
        
        if res:
            res = self.l_sql.delete_pkgs('dependence',pkg)
            if res:
                print("%s dependence delete failed" % pkg)
                self.close_ldb() 
                return 1
            else:
                self.close_ldb()
                return 0
        else:
            print("There is no dependence in the %s package" %(pkg))
            self.close_ldb() 
            return 1

    def deletePkgsDepsInfo(self,pkg_list):
        """
        删除软件包依赖信息
        参数：软件包名列表
        返回值：0 删除成功， 1  删除失败
        """
        self.open_Ldb()
        for pkg in pkg_list:
            res = self.deletePkgDepsInfo(pkg)
            if res:
                self.close_ldb()
                return 1
        self.close_ldb()
        return 0

    def updatePkgDepInfo(self,dep_dict):
        """
        更新软件包依赖
        参数：依赖字典
        返回值：0 成功， 1 失败
        """
        self.open_Ldb()
        for pkgName in dep_dict:
            res = self.l_sql.value_is_in_table("pkgName",pkgName,"dependence")
            if res:
                res = self.l_sql.delete_pkgs('dependence',pkgName)
                if res:
                    print("%s delete failed,package dependence update stoped" % pkgName) 
                    self.close_ldb()
                    return 1
            for depName in dep_dict[pkgName]:
                res = self.l_sql.insert_pkg_into_table("dependence",(pkgName, depName))
                if res:
                    print("%s -> %s insert failed,package dependence update stoped" %(pkgName, depName)) 
                    self.close_ldb()
                    return 1
        self.close_ldb()
        return 0

    def loadDepsFromDependeceTab(self):
        '''
        获取全部依赖关系
        pkg -> deps
        '''
        self.open_Ldb()
        deps = self.l_sql.select_all_from_table('select_dependence')
        depsDict = {}
        for item in deps:
            if item[0] not in depsDict:
                tmp = set()
                tmp.add(item[1])
                depsDict[item[0]] = tmp
            else:
                depsDict[item[0]].add(item[1])
        self.close_ldb()
        return depsDict

    def readDepsrcName(self):
        """
        读取依赖列表
        返回值：源码包列表
        """
        srcName_list = []
        self.open_Ldb()
        srcNames = self.l_sql.select_all_from_table('select_dependence')
        self.close_ldb()
        print(srcNames)
        for srcName in srcNames:
            srcName_list.append(srcName[0])
        return srcName_list


    """
    描述信息
    """
    def saveDescription(self,csv_dict_list):
        """
        存储软件包描述信息
        """
        info_list = []
        for csv_dict in csv_dict_list:
            name = csv_dict["src_name"]
            zero_des = csv_dict["zero_description"]
            zero_summary = csv_dict["zero_summary"]
            des = csv_dict["description"]
            summary = csv_dict["summary"]
            primary_rpm_name = csv_dict["primary_rpm_name"]
            primary_description = csv_dict["primary_description"]
            primary_summary = csv_dict["primary_summary"]
            rpm_name = csv_dict["rpm_name"]
            temp = (name,zero_des,zero_summary,des,summary,primary_rpm_name,primary_description,primary_summary)
            info_list.append(temp)

        self.open_Ldb()
        res = self.l_sql.insert_huge_date_into_table("srcrpm_info",info_list)
        if res:
            self.close_ldb()
            return 1
        self.close_ldb()
        return 0

    
    """
    rpm描述信息
    """
    def saveRpmDescription(self,csv_dict_list):
        """
        存储软件包描述信息
        """
        info_list = []
        for csv_dict in csv_dict_list:
            name = csv_dict["rpm_name"]
            src_version = csv_dict["srcPkgVersion"]
            src_release = csv_dict["srcPkgRelease"]
            zero_des = csv_dict["zero_description"]
            zero_summary = csv_dict["zero_summary"]
            des = csv_dict["description"]
            summary = csv_dict["summary"]
            src_name = csv_dict["src_name"]
            temp = (name,src_name,src_version,src_release, zero_des,zero_summary,des,summary)
            info_list.append(temp)

        self.open_Ldb()
        res = self.l_sql.insert_huge_date_into_table("rpm_info",info_list)
        if res:
            self.close_ldb()
            return 1
        self.close_ldb()
        return 0

    def update_pkg_description(self,pkg,desc):
        """
        更新描述信息
        参数：pkg 软件包名，  desc 描述信息列表 
        """
        self.open_Ldb()
        res = self.l_sql.value_is_in_table("pkgName",pkg,"srcrpm_info")
        if res == 0:
            res = self.l_sql.insert_pkg_into_table("srcrpm_info",(pkg,desc[0],desc[1],desc[2],desc[3]))
            if res != 0:
                self.close_ldb()
                return res
        else:
            res = self.l_sql.update_pkg_in_table("srcrpm_info",(desc[0],desc[1],desc[2],desc[3],pkg))
        if res != 0:
            self.close_ldb()
            return res
        return 0
        

    def get_pkg_description(self,pkg):
        """
        获取软件包描述信息
        参数：软件包名
        返回值：成功 描述信息， 失败 1
        """
        self.open_Ldb()
        res = self.l_sql.value_is_in_table("pkgName",pkg,"srcrpm_info")
        if res:
            res = self.l_sql.select_from_table('select_description',pkg)
            self.close_ldb()
            return res
        else:
            print("%s package not have description" %pkg) 
            self.close_ldb()
            return 1

    """
    update domain info
    """
    def update_domain_info(self,pkgName,domain):
        self.open_Ldb()
        res = self.l_sql.value_is_in_table("pkgName",pkgName,"domain")
        if res:
            res = self.l_sql.update_pkg_in_table("domain",(domain,pkgName))
            if res == 0:
                self.close_ldb()
                return 0
            else:
                print("%s domain point update failed" %pkgName)
                self.close_ldb() 
                return 1  
        else :
            res =  self.l_sql.insert_pkg_into_table("domain",(pkgName,domain,1))
            if res == 0:
                self.close_ldb()
                return 0
            else:
                print("%s domain point insert failed" %pkgName) 
                self.close_ldb()
                return 1
    """
    锚点
    """
    def update_anchored_info(self,pkgName,layer,classification):
        """
        更新锚点信息
        参数：pkgName 软件包名，    layer 层级，    classification 类别
        返回值：0 成功，    1 失败
        """
        self.open_Ldb()
        res = self.l_sql.value_is_in_table("pkgName",pkgName,"anchored")
        if res:
            res = self.l_sql.update_pkg_in_table("anchored",(layer,classification,pkgName))
            #res = self.l_sql.update_anchored(pkgName,layer,classification)
            if res == 0:
                self.close_ldb()
                return 0
            else:
                print("%s anchor point update failed" %pkgName)
                self.close_ldb() 
                return 1
        else:
            res = self.l_sql.insert_pkg_into_table("anchored",(pkgName,layer,classification))
            #res = self.l_sql.insert_anchored(pkgName,layer,classification)
            if res == 0:
                self.close_ldb()
                return 0
            else:
                print("%s anchor point insert failed" %pkgName) 
                self.close_ldb()
                return 1

    def delete_anchored(self,pkgName):
        """
        删除锚点
        参数：pkgName 软件包名
        返回值：0 成功，    1 失败
        """
        self.open_Ldb()
        res = self.l_sql.value_is_in_table("pkgName",pkgName,"anchored")
        if res:
            res = self.l_sql.delete_pkgs('anchored',pkgName)
            if res == 0:
                self.close_ldb()
                return 0
            else:
                print("%s anchor point delete failed" %pkgName)
                self.close_ldb()  
                return 1
        else:
            print("%s is not anchor point" %pkgName)
            self.close_ldb()
            return 1

    '''
    delete  domain info
    '''
    def delete_domain(self,pkgname):
        self.open_Ldb()
        res = self.l_sql.value_is_in_table("pkgName",pkgName,"domain")
        if res:
            res = self.l_sql.delete_pkgs('domain',pkgName)
            if res == 0:
                self.close_ldb()
                return 0
            else:
                print("%s domain point delete failed" %pkgName)
                self.close_ldb()  
                return 1
        else:
            print("%s is not domain point" %pkgName)
            self.close_ldb()
            return 1

    """
    清空所有列表
    """
    def clearAllTable(self):
        """
        清空所有表
        返回值：0 清除成功，    1 清除失败
        """
        self.open_Ldb()
        for table in table_name:
            res = self.l_sql.clear_table(table)
            if res != 0:
                print("%s table clear faile" %table)
                self.close_ldb() 
                return 1
        self.close_ldb()
        return 0
     
    def deletePkgLayerClass(self,pkg):
        self.open_Ldb()
        for table in table_name:
            self.l_sql.delete_pkgs(table,pkg)

        self.close_ldb()
    
    def deletesrcPkgLayerClass(self,pkg):
        self.open_Ldb()
        for table in src_table_name:
            self.l_sql.delete_pkgs(table,pkg)

        self.l_sql.delete_src_pkgs(pkg)
        self.close_ldb()

    def get_pkgs_csv_dict_from_table(self):
        """
        获取表中所有包的分层分类信息，转换成 csv字典
        """
        srcName_dict = {}
        self.open_Ldb()
        srcNames = self.l_sql.select_all_from_table('select_layer_class_info')
        self.close_ldb()
        #print(srcNames)
        for srcName in srcNames:
            srcName_dict[srcName[0]] = (srcName[1],srcName[2],srcName[3])

        pprint.pprint(srcName_dict)
        return srcName_dict

    def get_pkgs_dot_dict_from_table(self,pkg):
        """
        获取表中所有包的分层分类信息，转换成 dot字典
        """
        srcName_dict = {}
        self.open_Ldb()
        srcNames = self.l_sql.select_all_from_table('select_dependence')
        self.close_ldb()

        for srcName in srcNames:
            if srcName[0] not in srcName_dict.keys():
                srcName_dict[srcName[0]] = [srcName[1]]
            elif srcName[1] not in srcName_dict[srcName[0]]:
                srcName_dict[srcName[0]] += [srcName[1]] 
           
        pprint.pprint(srcName_dict)
        return srcName_dict

    def get_pkg_csv_dict_from_table(self,pkg):
        """
        获取表中pkg包的分层分类信息，转换成 csv字典
        """
        srcName_dict = {}
        self.open_Ldb()
        srcNames = self.l_sql.select_from_table('select_layer_class_info',pkg)
        self.close_ldb()
        #print(srcNames)
        for srcName in srcNames:
            srcName_dict[srcName[0]] = (srcName[1],srcName[2],srcName[3])

        pprint.pprint(srcName_dict)
        return srcName_dict

    def get_pkg_dot_dict_from_table(self,pkg):
        """
        获取表中所有包的分层分类信息，转换成 dot字典
        """
        srcName_dict = {}
        self.open_Ldb()
        srcNames = self.l_sql.select_from_table('select_dependence',pkg)
        self.close_ldb()
        srcName_dict[pkg] = []
        for srcName in srcNames:
            srcName_dict[srcName[0]].append(srcName[1])
        srcName_dict[pkg] = list(set(srcName_dict[pkg])) 
        pprint.pprint(srcName_dict)
        return srcName_dict

    def get_pkgs_desc_csv_dict_from_table(self):
        """
        获取表中pkg包的信息，转换成 csv字典
        """
        srcName_dict = {}
        self.open_Ldb()
        srcNames = self.l_sql.select_all_from_table('select_srcrpm_info')
        self.close_ldb()

        for srcName in srcNames:
            srcName_dict[srcName[1]] = (srcName[2],srcName[3],srcName[4],srcName[5],srcName[6],srcName[7],srcName[8],srcName[9])

        return srcName_dict

    def get_pkgs_rpm_desc_csv_dict_from_table(self):
        """
        获取表中pkg包的信息，转换成 csv字典
        """
        rpmName_dict = {}
        self.open_Ldb()
        rpmNames = self.l_sql.select_all_from_table('select_rpm_info')
        self.close_ldb()

        for rpmName in rpmNames:
            rpmName_dict[rpmName[1]] = (rpmName[5],rpmName[6],rpmName[7],rpmName[8],rpmName[2],rpmName[3],rpmName[4])

        return rpmName_dict

    def get_pkg_desc_csv_dict_from_table(self,pkg):
        srcName_dict = {}
        self.open_Ldb()
        srcNames = self.l_sql.select_from_table('select_description',pkg)
        self.close_ldb()
        for srcName in srcNames:
            srcName_dict[pkg] = (srcName[2],srcName[3],srcName[4],srcName[5],srcName[6],srcName[7],srcName[8],srcName[9])
        return srcName_dict

    def get_pkg_rpm_desc_csv_dict_from_table(self,pkg):
        srcName_dict = {}
        self.open_Ldb()
        srcNames = self.l_sql.select_from_table('select_description_rpm',pkg)
        self.close_ldb()
        for srcName in srcNames:
            srcName_dict[pkg] = (srcName[5],srcName[6],srcName[7],srcName[8],srcName[2],srcName[3],srcName[4])
        return srcName_dict

    def get_domain_csv_dict_from_table(self):
        srcName_dict = {}
        self.open_Ldb()
        srcNames = self.l_sql.select_all_from_table('select_domain')
        self.close_ldb()
        #print(srcNames)
        for srcName in srcNames:
            srcName_dict[srcName[0]] = (srcName[1],srcName[2])
        return srcName_dict

    def get_special_domain_csv_dict_from_table(self,special_domain):
        srcName_dict = {}
        self.open_Ldb()
        srcNames = self.l_sql.select_from_table('select_domain',special_domain)
        self.close_ldb()
        #print(srcNames)
        for srcName in srcNames: 
            srcName_dict[srcName[0]] = (srcName[1],srcName[2])
        return srcName_dict


if __name__ == "__main__":
    db0 = STORAGE_SQL()
    #for name,member in creat_tables.__members__.items():
    #    db0.create_table(name)
    db0.create_table()
    print("after create tables")
    db0.l_sql.insert_pkg_into_table('dependence',('a', 'b'))
    db0.l_sql.insert_pkg_into_table('dependence',('a', 'c'))
    db0.l_sql.insert_pkg_into_table('dependence',('b', 'c'))
    d = db0.loadDepsFromDependeceTab()
    for i in d.keys():
        print(d[i])

