# SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import read_sql as sql
import get_dep as Dep
import config 

def srpmName_analyze(srpmName):
    """
    功能：源码包名字解析
    思路：源码包名用“-”分开后，最有两部分对应的是version和release
        用“-veesion”将源码包分为两部分，取前面的部分就是源码包名
    """
    srpmName_2 = srpmName.split("-")[-2]
    sName = srpmName.split("-%s" % srpmName_2)[0]
    return sName


def get_srpm_dep_without_V(srpm_name):
    """
    功能：获取srcrpm直接依赖的源码包 不带版本号
    name:rpm_spurcerpm
    返回值:依赖的源码包名列表
    """
    srpm_list = []  #源码包直接依赖列表

    sql.select_openSql(config.db_type) 
    sql.open_db()

    names = sql.getAllRpmName(srpm_name)    #源码包提供所有二进制包名
    sql.close_db()

    for name in names:
        #print("rrrrrr",name)
        #去除二进制包中的干扰项
        if name[0].endswith("-doc") == True or \
            name[0].endswith("-devel") == True or \
            name[0].find("glibc-langpack") != -1 or \
            name[0].find("glibc-minimal") != -1 or \
            name[0].find("glibc-all") != -1 or \
            name[0].find("glibc-benchtests") != -1 or \
            name[0].endswith("-tests") == True or \
            name[0].endswith("-test") == True or \
            name[0].endswith("-debug") == True or\
            name[0].endswith("-demo") == True or\
            name[0].find("-debuginfo") != -1:   
            continue

        srpms = Dep.get_rpm_depdens(name) #获取rpm包依赖的源码包的名称列表

        src_rpm = srpmName_analyze(srpm_name[0])
        for srpm in srpms:
            #去除源码包中依赖的干扰项
            if srpm == src_rpm or \
               srpm.startswith("filesystem") == True or \
               srpm.startswith("basesystem") == True or \
               srpm.startswith("system-release") == True or \
               srpm.startswith("setup") == True:
                continue
        
            #sName = srpmName_analyze(srpm[0])
            if srpm not in srpm_list:
                srpm_list.append(srpm)
        
    return srpm_list


def get_all_srpm_dep_without_V():
    """
    获取源中的所有源码包直接依赖，不带版本号
    """
    counter = 0
    dep_dict = {}  #依赖字典 key：不带版本浩的源码包名     value：依赖的源码包名列表（不带版本号）

    sql.open_db()
    srpm_name_list = sql.getAllSrcrpm()   #获取源中所有源码包名（此时源码包名带有版本号）
    sql.close_db()
    print("srpm_name_list len =",len(srpm_name_list))

    for srpm_name in srpm_name_list:
        sName = srpmName_analyze(srpm_name[0])  #去掉源码包名的版本号等，只留下源码包名)
        dep_list = get_srpm_dep_without_V(srpm_name)

        #防止键值相同覆盖
        if sName in dep_dict.keys():
            print("same name srpm_name",srpm_name)
            counter += 1
            dep_dict_temp = dep_dict[sName]
            
            for dep in dep_dict_temp:
                if dep not in dep_list:
                    print(dep)
                    dep_list.append(dep)
            
        dep_dict[sName] = dep_list

    print("counter = ",counter)
    print(len(dep_dict.keys()))
    return dep_dict
        
        
if __name__ == "__main__":
    get_all_srpm_dep_without_V()




