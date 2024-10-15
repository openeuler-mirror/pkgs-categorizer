# SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import read_sql as sql
import csv
import sys
import os
import config
import pprint
import get_dep
from  layer_sql.process_layer_sql import  STORAGE_SQL
from config import fcfl_config
import hawkey
csv.field_size_limit(500*1024*1024)
class CSV:
    def write_csv(self,csv_dict_list,csv_name,table_type):
        """
        写csv文件
        输入参数：csv_dict_list 字典列表，字典键值为csv的表头
                csv_name  csv文件名,内部根据config添加路径和.csv
        返回值：生成的文件路径
        """
        count=0  #统计csv行数
        data = []
        src_name_list = []  #存储源码包名列表
         
        file_name = csv_name + '.csv'

        header = config.header_dict[table_type]

        fcfl_configs = fcfl_config()
        csv_path = fcfl_configs.get_csv_path()
        
        if file_name in os.listdir(csv_path):
            os.remove(csv_path+file_name)

        """
        源码包去重操作
        """
        for i in csv_dict_list:
            src_name = i[config.header_dict[table_type][0]] #获取包名
            if src_name not in src_name_list:
                data.append(i)
                count +=1
                src_name_list.append(src_name)
        #写csv
        file_name = csv_path + file_name

        with open(file_name,'w') as f:
            writer = csv.DictWriter(f,header)
            writer.writeheader()
            writer.writerows(data) 

        return file_name

    def write_csvfile(self,csv_dict_list,filePathName,table_type):
        """
        写csv文件
        输入参数：csv_dict_list 字典列表，字典键值为csv的表头
                csv_name  csv文件名,内部根据config添加路径和.csv
        返回值：生成的文件路径
        """
        data = []
        src_name_list = []  #存储源码包名列表
   
        file_name = filePathName

        header = config.header_dict[table_type]

        csv_path = os.path.dirname(file_name)
        
        if file_name in os.listdir(csv_path):
            os.remove(file_name)

        """
        源码包去重操作
        """
        for i in csv_dict_list:
            src_name = i[config.header_dict[table_type][0]] #获取包名
            if src_name not in src_name_list:
                data.append(i)
                src_name_list.append(src_name)    
        #写csv
        with open(file_name,'w') as f:
            writer = csv.DictWriter(f,header)
            writer.writeheader()
            writer.writerows(data) 

        return file_name


    def str2_is_in_str1(self,str1,str2):
        if str2 and str2 not in str1:
            return str1 + '\n' + str2
        else :
            return str1


    def generate_src_csv_files(self, repoDir):
        # 从repo中获得
        def primary_pkg_strategy(pkg, src_pkg):
            if pkg == get_dep.srpmName_analyze(src_pkg):
                return True
            else:
                return False
        '''
        根据字典生成csv文件
        '''
        # 创建列表字符串
        csv_list =[]
        csv_obj = sql.REPO_CSV()
        all_src_dict = {}
         # 获取字典
        all_src_dict = csv_obj.get_all_repo_dict(repoDir)
        for src_name in all_src_dict.keys():
            csv_dict = {}
            zero_description = ''
            zero_summary = ''
            description = ''
            summary = ''
            primary_rpm_name =''
            primary_description = ''
            primary_summary = ''
            rpm_name = ''
            have_not_primary_pkg = []
            if src_name in config.src_zero_in_list:
                for rpm_info in all_src_dict[src_name]:
                    #if primary_pkg_strategy(rpm_info[0],src_name):
                    #判断主包
                    if rpm_info[0]==src_name:
                        primary_rpm_name = rpm_info[0]
                        primary_description = rpm_info[1]
                        primary_summary = rpm_info[2]
                        zero_description = self.str2_is_in_str1(zero_description,rpm_info[1])
                        zero_summary =  self.str2_is_in_str1(zero_summary,rpm_info[2])
                        rpm_name = self.str2_is_in_str1(rpm_name,rpm_info[0])
                    else :
                        if csv_obj.rpm_is_nead_remove_for_csv(rpm_info[0]) == 0:
                            have_not_primary_pkg.append(rpm_info[0])
                            zero_description = self.str2_is_in_str1(zero_description,rpm_info[1])
                            zero_summary = self.str2_is_in_str1(zero_summary,rpm_info[2])
                            rpm_name = self.str2_is_in_str1(rpm_name,rpm_info[0])
                if primary_rpm_name == '':
                    for pkg in have_not_primary_pkg : 
                        primary_rpm_name +=  pkg + '\n'
                    primary_description = zero_description
                    primary_summary = zero_summary
                
                csv_dict["src_name"] = src_name 
                csv_dict["description"] = description
                csv_dict["summary"] = summary
                csv_dict["zero_description"] = zero_description
                csv_dict["zero_summary"] = zero_summary   
                csv_dict["primary_rpm_name"] = primary_rpm_name
                csv_dict["primary_description"] = primary_description
                csv_dict["primary_summary"] = primary_summary
                csv_dict["rpm_name"] = rpm_name
            else:
                #不在０list
                for rpm_info in all_src_dict[src_name]:
                    #if primary_pkg_strategy(rpm_info[0],src_name):
                    if rpm_info[0]==src_name:
                        primary_rpm_name = rpm_info[0]
                        primary_description = rpm_info[1]
                        primary_summary = rpm_info[2]  
                        description =  self.str2_is_in_str1(description,rpm_info[1])
                        summary = self.str2_is_in_str1(summary,rpm_info[2])
                        rpm_name = self.str2_is_in_str1(rpm_name,rpm_info[0])
                    else :
                        if csv_obj.rpm_is_nead_remove_for_csv(rpm_info[0]) == 0:
                            have_not_primary_pkg.append(rpm_info[0])
                            description = self.str2_is_in_str1( description,rpm_info[1])
                            summary = self.str2_is_in_str1(summary,rpm_info[2])
                            rpm_name = self.str2_is_in_str1(rpm_name,rpm_info[0])
                if primary_rpm_name == '':
                    for pkg in have_not_primary_pkg : 
                        primary_rpm_name +=  pkg + '\n'
                    primary_description = description
                    primary_summary = summary

                csv_dict["src_name"] = src_name 
                #csv_dict["src_name"] = hawkey.split_nevra(src_name).name
                csv_dict["description"] = description
                csv_dict["summary"] = summary
                csv_dict["zero_description"] = zero_description
                csv_dict["zero_summary"] = zero_summary   
                csv_dict["primary_rpm_name"] = primary_rpm_name
                csv_dict["primary_description"] = primary_description
                csv_dict["primary_summary"] = primary_summary
                csv_dict["rpm_name"] = rpm_name
            csv_list.append(csv_dict)
        return self.write_csv(csv_list,"src_all","src_header")

    def generate_rpm_all_csv_files(self, repoDir):
        """
        生成所有csv文件
        """
        csv_obj = sql.REPO_CSV()
        all_csv_list = []
        csv_dict_dict = csv_obj.get_all_rpm_repo_dict(repoDir)
        for rpmpkg in csv_dict_dict.keys():
            csv_dict = {}
            zero_description = ''
            zero_summary = ''
            description = ''
            summary = ''
            rpm_name = rpmpkg
            src_name = ''
            version = ''
            release = ''
            sets = csv_dict_dict[rpmpkg]
            for i in sets:
                src_name = i[2]
                version = i[3]
                release = i[4]
                if rpmpkg in config.rpm_zero_in_list:
                    rpm_name = rpmpkg
                    zero_description = i[0]
                    zero_summary = i[1]
                    description = ''
                    summary = ''
                else :
                    zero_description = '' 
                    zero_summary = ''
                    description = i[0]
                    summary = i[1]
                csv_dict['rpm_name'] = rpm_name
                csv_dict['zero_description'] = zero_description
                csv_dict['zero_summary'] = zero_summary
                csv_dict['description'] = description
                csv_dict['summary'] = summary
                csv_dict['src_name'] = src_name
                csv_dict['srcPkgVersion'] = version
                csv_dict['srcPkgRelease'] = release
                all_csv_list.append(csv_dict)
        csv_path = self.write_csv(all_csv_list,"rpm_all","rpm_header")
        return csv_path

    def generate_rpm_csv_files_for_test(self):
        """
        生成所有csv文件
        """
        def primary_vs_strategy(vs1, rls1,vs2, rls2):

            if vs2 in vs1 and rls2 in rls1:
                return vs1, rls1
            else:
                return vs1 + '\n' + vs2, rls1 + '\n' + rls2

        csv_obj = sql.REPO_CSV()
        all_csv_list = []
        csv_dict_dict = csv_obj.get_all_rpm_repo_dict_for_test()
        for rpmpkg in csv_dict_dict.keys():
            csv_dict = {}
            zero_description = ''
            zero_summary = ''
            description = ''
            summary = ''
            rpm_name = rpmpkg
            src_name = csv_dict_dict[rpmpkg][2]
            version = ''
            release = ''
            if rpmpkg in config.rpm_zero_in_list:
                rpm_name = rpmpkg
                zero_description = csv_dict_dict[rpmpkg][0]
                zero_summary = csv_dict_dict[rpmpkg][1]
                description = ''
                summary = ''
            else :
                zero_description = '' 
                zero_summary = ''
                description = csv_dict_dict[rpmpkg][0]
                summary = csv_dict_dict[rpmpkg][1]
            version,release = primary_vs_strategy (version,release,csv_dict_dict[rpmpkg][3],csv_dict_dict[rpmpkg][4])
            csv_dict['rpm_name'] = rpm_name
            csv_dict['zero_description'] = zero_description
            csv_dict['zero_summary'] = zero_summary
            csv_dict['description'] = description
            csv_dict['summary'] = summary
            csv_dict['src_name'] = src_name
            csv_dict['srcPkgVersion'] = version
            csv_dict['srcPkgRelease'] = release
            all_csv_list.append(csv_dict)
        csv_path = self.write_csv(all_csv_list,"rpm_all_test","rpm_header")
        return csv_path

    def generate_fcfl_pkg_description_csv_file(self,pkgname):
        csv_list = []
        new_db = STORAGE_SQL()
        csv_dict = new_db.get_pkg_desc_csv_dict_from_table(pkgname)

        if csv_dict != {}:
            new_csv_dict = {}
            new_csv_dict["src_name"] = pkgname
            new_csv_dict["zero_description"] = csv_dict[pkgname][0]
            new_csv_dict["zero_summary"] =  csv_dict[pkgname][1]
            new_csv_dict["description"] = csv_dict[pkgname][2]
            new_csv_dict["summary"] = csv_dict[pkgname][3]
            new_csv_dict["primary_rpm_name"] = csv_dict[pkgname][4]
            new_csv_dict["primary_description"] = csv_dict[pkgname][5]
            new_csv_dict["primary_summary"] = csv_dict[pkgname][6]
            new_csv_dict["rpm_name"] = csv_dict[pkgname][7]
            csv_list.append(new_csv_dict)
            return self.write_csv(csv_list,"fcfl_"+pkgname+"_dec","src_header")
        else :
            print("%s not in table" %pkgname)
            return ''

    def generate_fcfl_pkg_rpm_description_csv_file(self,pkgname):
        csv_list = []   
        new_db = STORAGE_SQL()
        csv_dict = new_db.get_pkg_rpm_desc_csv_dict_from_table(pkgname)
        if csv_dict != {}:
            new_csv_dict = {}
            new_csv_dict['rpm_name'] = pkgname
            new_csv_dict['zero_description'] = csv_dict[pkgname][0]
            new_csv_dict['zero_summary'] = csv_dict[pkgname][1]
            new_csv_dict['description'] = csv_dict[pkgname][2]
            new_csv_dict['summary'] = csv_dict[pkgname][3]
            new_csv_dict['src_name'] = csv_dict[pkgname][4]
            new_csv_dict['srcPkgVersion'] = csv_dict[pkgname][5]
            new_csv_dict['srcPkgRelease'] = csv_dict[pkgname][6]
            csv_list.append(new_csv_dict)
            return self.write_csv(csv_list,"fcfl_rpm"+pkgname+"_dec","rpm_header")
        else :
            print("%s not in table" %pkgname)
            return ''

    def generate_fcfl_pkgs_description_csv_file(self):
        csv_list = []
        new_db = STORAGE_SQL()
        csv_dict = new_db.get_pkgs_desc_csv_dict_from_table()
        for pkg in csv_dict.keys():
            new_csv_dict = {}
            new_csv_dict["src_name"] = pkg
            new_csv_dict["zero_description"] = csv_dict[pkg][0]
            new_csv_dict["zero_summary"] =  csv_dict[pkg][1]
            new_csv_dict["description"] = csv_dict[pkg][2]
            new_csv_dict["summary"] = csv_dict[pkg][3]
            new_csv_dict["primary_rpm_name"] = csv_dict[pkg][4]
            new_csv_dict["primary_description"] = csv_dict[pkg][5]
            new_csv_dict["primary_summary"] = csv_dict[pkg][6]
            new_csv_dict["rpm_name"] = csv_dict[pkg][7]
            csv_list.append(new_csv_dict)
        return self.write_csv(csv_list,"fcfl_src_all","src_header")

    def generate_fcfl_pkgs_rpm_description_csv_file(self):  
        csv_list = []   
        new_db = STORAGE_SQL()
        csv_dict = new_db.get_pkgs_rpm_desc_csv_dict_from_table()
        for pkg in csv_dict.keys():
            new_csv_dict = {}
            new_csv_dict['rpm_name'] = pkg
            new_csv_dict['zero_description'] = csv_dict[pkg][0]
            new_csv_dict['zero_summary'] = csv_dict[pkg][1]
            new_csv_dict['description'] = csv_dict[pkg][2]
            new_csv_dict['summary'] =  csv_dict[pkg][3]
            new_csv_dict['src_name'] =  csv_dict[pkg][4]
            new_csv_dict['srcPkgVersion'] =  csv_dict[pkg][5]
            new_csv_dict['srcPkgRelease'] = csv_dict[pkg][6]
            csv_list.append(new_csv_dict)
        return self.write_csv(csv_list,"fcfl_rpm_all","rpm_header")

    def generate_fcfl_files_csv_file(self,filename):
        """
        生成所有分层分类包 csv文件
        """
        all_csv_list = []
        new_db = STORAGE_SQL()
        csv_dict = new_db.get_pkgs_csv_dict_from_table()
        if csv_dict != {}:
            for pkg in csv_dict.keys():
                new_csv_dict = {}
                new_csv_dict['src_name'] = pkg
                new_csv_dict['layer'] = csv_dict[pkg][0]
                new_csv_dict['classification'] = csv_dict[pkg][1]
                new_csv_dict['manimark'] = csv_dict[pkg][2]
                all_csv_list.append(new_csv_dict)
            csv_path = self.write_csv(all_csv_list,"allpkgs_"+filename,"fcfl_header")
            return csv_path
        else :
            print("%s not in table" %filename)
            return ''

    def generate_fcfl_file_csv_file(self,pkgname):
        """
        生成单个分层分类包 csv文件
        """
        all_csv_list = []   
        new_db = STORAGE_SQL()
        csv_dict = new_db.get_pkg_csv_dict_from_table(pkgname)
        if csv_dict != {}:
            for pkg in csv_dict.keys():
                new_csv_dict = {}
                new_csv_dict['src_name'] = pkg
                new_csv_dict['layer'] = csv_dict[pkg][0]
                new_csv_dict['classification'] = csv_dict[pkg][1]
                new_csv_dict['manimark'] = csv_dict[pkg][2]
                all_csv_list.append(new_csv_dict)
            csv_path = self.write_csv(all_csv_list,'fcfl'+pkgname,"fcfl_header")
            return csv_path
        else :
            print("%s not in table" %pkgname)
            return ''

    def generate_fcfl_domain_all_csv_file(self):
        """
        生成 domain csv文件
        """
        all_csv_list = []   
        new_db = STORAGE_SQL()
        csv_dict = new_db.get_domain_csv_dict_from_table()
        for pkg in csv_dict.keys():
            new_csv_dict = {}
            new_csv_dict['src_name'] = pkg
            new_csv_dict['domain_info'] = csv_dict[pkg][0]
            new_csv_dict['manimark'] = csv_dict[pkg][1]
            all_csv_list.append(new_csv_dict)
        csv_path = self.write_csv(all_csv_list,'fcfl_all_domain',"domain_header")
        return csv_path

    def generate_fcfl_domain_special_csv_file(self,speclal_domain):
        """
        生成special domain csv文件
        """
        all_csv_list = []   
        new_db = STORAGE_SQL()
        csv_dict = new_db.get_special_domain_csv_dict_from_table(speclal_domain)
        if csv_dict != {}:
            for pkg in csv_dict.keys():
                new_csv_dict = {}
                new_csv_dict['src_name'] = pkg
                new_csv_dict['domain_info'] = csv_dict[pkg][0]
                new_csv_dict['manimark'] = csv_dict[pkg][1]
                all_csv_list.append(new_csv_dict)
            csv_path = self.write_csv(all_csv_list,'fcfl_'+speclal_domain+'_domain','domain_header')
            return csv_path
        else :
            print("%s not in table" %speclal_domain)
            return ''

    def generate_rpm_csv_file(self,repo_type):
        """
        生成单个rpm csv文件
        """
        csv_obj = sql.REPO_CSV()
        csv_dict_list = csv_obj.get_one_repo_rpm_description(repo_type)
        path = self.write_csv(csv_dict_list,file_name,"rpm_header")
        return path

    def generate_src_csv_file(self,repo_type):
        """
        生成单个src csv文件
        """
        csv_obj = sql.REPO_CSV()
        csv_dict_list = csv_obj.get_one_repo_src_description(repo_type)
        path = self.write_csv(csv_dict_list,file_name,"src_header")
        return path

    def read_src_csv_file(self,csv_path):
        """
        读取src csv文件
        """
        scv_dict_list = []
        with open(csv_path) as csvfile:             # 使用csv.reader读取csvfile中的文件
            csv_reader = csv.reader(csvfile)        # 将csv 文件中的数据保存到data中 
            for row in csv_reader:   
                if  "src_name" == row[0]:
                    continue                 
                scv_dict={}          
                scv_dict["src_name"] = ''
                scv_dict["zero_description"] = ''
                scv_dict["zero_summary"] = ''
                scv_dict["description"] = ''
                scv_dict["summary"] = ''
                scv_dict["primary_rpm_name"] = ""
                scv_dict["primary_description"] = ""
                scv_dict["primary_summary"] = ""
                scv_dict["rpm_name"] = ""         
                scv_dict["src_name"] = row[0] 
                scv_dict["zero_description"] = row[1] 
                scv_dict["zero_summary"] = row[2] 
                scv_dict["description"] = row[3] 
                scv_dict["summary"] = row[4]
                scv_dict["primary_rpm_name"] = row[5]
                scv_dict["primary_description"] = row[6]
                scv_dict["primary_summary"] = row[7]
                scv_dict["rpm_name"] = row[8]
                scv_dict_list.append(scv_dict)
        return scv_dict_list

    def read_rpm_csv_file(self,csv_path):
        """
        读取rpm csv文件
        """
        scv_dict_list = []
        with open(csv_path) as csvfile:             # 使用csv.reader读取csvfile中的文件
            csv_reader = csv.reader(csvfile)        # 将csv 文件中的数据保存到data中 
            for row in csv_reader:   
                if  "rpm_name" == row[0]:
                    continue                 
                scv_dict={}          
                scv_dict["rpm_name"] = ''
                scv_dict["zero_description"] = ''
                scv_dict["zero_summary"] = ''
                scv_dict["description"] = ''
                scv_dict["summary"] = ''         
                scv_dict["rpm_name"] = row[0] 
                scv_dict["zero_description"] = row[1] 
                scv_dict["zero_summary"] = row[2] 
                scv_dict["description"] = row[3] 
                scv_dict["summary"] = row[4]
                scv_dict_list.append(scv_dict)
        return scv_dict_list

    def read_csv_file(self,csv_path,csv_type):
        """
        读取src csv文件
        """
        scv_dict_list = []
        header = config.header_dict[csv_type]
        with open(csv_path) as csvfile:             # 使用csv.reader读取csvfile中的文件
            csv_reader = csv.reader(csvfile)        # 将csv 文件中的数据保存到data中
            for row in csv_reader:
                if  header[0] == row[0]:
                    continue
                scv_dict={}
                for ii in range(len(header)):
                    if header[ii] == None:
                        continue
                    scv_dict[header[ii]] = row[ii]
                scv_dict_list.append(scv_dict)
        return scv_dict_list

    #'rpm_header' : ["rpm_name","zero_description","zero_summary","description","summary","src_name","srcPkgVersion","srcPkgRelease"] ,
    #'src_header' : ["src_name","zero_description","zero_summary","description","summary","primary_rpm_name","primary_description","primary_summary","rpm_name"] , 
    def read_csv_file2(self,csv_path,csv_type):
        csv_dict = {}
        header = config.header_dict[csv_type]
        with open(csv_path) as csvfile:             # 使用csv.reader读取csvfile中的文件
            csv_reader = csv.reader(csvfile)        # 将csv 文件中的数据保存到data中 
            for row in csv_reader:
                print("aaaaaaaaaaaaaaaaa", row)
                if  header[0] == row[0]:
                    continue
                row_dict={}
                for ii in range(len(header)):
                    if header[ii] == None:
                        continue
                    row_dict[header[ii]] = row[ii]
                if csv_type == 'rpm_header':
                    csv_dict[row_dict['rpm_name']]= (row_dict['zero_description'], row_dict['zero_summary'], row_dict['description'], row_dict['summary'], row_dict['src_name'], row_dict['srcPkgVersion'], row_dict['srcPkgRelease'])
                if csv_type == 'src_header':
                    csv_dict[row_dict['src_name']] = (row_dict['zero_description'], row_dict['zero_summary'], row_dict['description'], row_dict['summary'], row_dict['primary_rpm_name'], row_dict['primary_description'], row_dict['primary_summary'], row_dict['rpm_name'])
        return csv_dict

    def mergeFunc(ii, jj, key):
        new_dict={}
        if key == "zero_summary" or key== "zero_description" or key=="description" or key=="summary":
            # 第一个文件中的入度非0，
            if ii["zero_summary"] == '' and ii["zero_description"] == '':
                new_dict["zero_summary"] = ii["zero_summary"]
                new_dict["zero_description"] = ii["zero_description"]
                # 两个文件中包的入度都是非 0
                if jj["zero_summary"] == ''and jj["zero_description"] == '':
                    new_dict["summary"] = self.str2_is_in_str1(ii["summary"],jj["summary"])
                    new_dict["description"] = self.str2_is_in_str1(ii["description"],jj["description"])
                    # 第一个文件中 包的入度非0，第二个文件中包的入度为0 
                else:
                    new_dict["summary"] =  self.str2_is_in_str1( ii["summary"],jj["zero_summary"]) 
                    new_dict["description"] = self.str2_is_in_str1( ii["description"],jj["zero_description"])
                # 第一个文件中包的入度为0
            elif ii["zero_summary"] != '' or ii["zero_description"] != '':
                # 第二个包的入度非0
                if jj["zero_summary"] == '' and jj["zero_description"] == '':
                    #repo中该包入度为0,（即无其他包依赖它），但新引入包入度不为0,说明它是被依赖的，则需把zero清空，信息移动到desc,summ.
                    new_dict["zero_summary"] = ''
                    new_dict["zero_description"] = ''

                    new_dict["summary"] = self.str2_is_in_str1(ii["zero_summary"],ii["summary"])
                    new_dict["description"] = self.str2_is_in_str1(ii["zero_description"],ii["description"])
                    new_dict["summary"] = self.str2_is_in_str1(new_dict["zero_summary"],jj["summary"])
                    new_dict["description"] = self.str2_is_in_str1(new_dict["zero_description"] , jj["description"])

                # 两个文件中包的入度都为0
                else :
                    #repo中该包入度为0,（即它被），新引入时它可能有人使用它。
                    new_dict["zero_summary"] = self.str2_is_in_str1(ii["zero_summary"], jj["zero_summary"])
                    new_dict["zero_description"] = self.str2_is_in_str1(ii["zero_description"], jj["zero_description"])
                    new_dict["summary"] = self.str2_is_in_str1(ii["summary"],jj["summary"])
                    new_dict["description"] = self.str2_is_in_str1(ii["description"] , jj["description"])
        else:
            #无需特殊处理
            ii[key]= jj[key]

    def mergeDict(destdict, srcdict, key, keylist):
        '''
        destdict, 
        srcdict, 从srcdict往destdict合并
        key, 用于判断是否合并，比较的key
        keylist, 要merged字典
        mergeIfNeddedFunc 合并函数
        '''
        new_dict={}
        if destdict[key]==srcdict[key]:
            # 第一个文件中的入度非0，
            if ii["zero_summary"] == '' and ii["zero_description"] == '':
                new_dict["zero_summary"] = ""
                new_dict["zero_description"] = ""
                # 两个文件中包的入度都是非 0, 描述和摘要都放到desc,summ
                if jj["zero_summary"] == ''and jj["zero_description"] == '':
                    new_dict["summary"] = self.str2_is_in_str1(ii["summary"],jj["summary"])
                    new_dict["description"] = self.str2_is_in_str1(ii["description"],jj["description"])

                # 第一个文件中 包的入度非0，第二个文件中包的入度为0,说明第二个包是依赖的最顶部的包。描述和摘要放到desc,summ
                else:
                    new_dict["summary"] = self.str2_is_in_str1( ii["summary"],jj["zero_summary"]) 
                    new_dict["description"] = self.str2_is_in_str1( ii["description"],jj["zero_description"])

            # 第一个文件中包的入度为0
            elif ii["zero_summary"] != '' or ii["zero_description"] != '':
                # 第二个包的入度非0
                if jj["zero_summary"] == '' and jj["zero_description"] == '':
                    #repo中该包入度为0,（即它是工具包），但新引入包入度不为0,说明它是被依赖的，则需把zero清空，信息移动到desc,summ.
                    new_dict["zero_summary"] = ''
                    new_dict["zero_description"] = ''

                    new_dict["summary"] = self.str2_is_in_str1(ii["zero_summary"], ii["summary"])
                    new_dict["description"] = self.str2_is_in_str1(ii["zero_description"], ii["description"])
                    new_dict["summary"] = self.str2_is_in_str1(new_dict["summary"], jj["summary"])
                    new_dict["description"] = self.str2_is_in_str1(new_dict["description"] , jj["description"])

                # 两个文件中包的入度都为0
                else :
                    #repo中该包入度为0,（即它是工具包），新引入时它可能有人使用它。
                    new_dict["zero_summary"] = self.str2_is_in_str1(ii["zero_summary"], jj["zero_summary"])
                    new_dict["zero_description"] = self.str2_is_in_str1(ii["zero_description"], jj["zero_description"])
                    new_dict["summary"] = self.str2_is_in_str1(ii["summary"], jj["summary"])
                    new_dict["description"] = self.str2_is_in_str1(ii["description"], jj["description"])
            #无需特殊处理
            keylist.remove("zero_summary")
            keylist.remove("zero_description")
            keylist.remove("summary")
            keylist.remove("description")
            for k in keylist:
                new_dict[key]= jj[key]
                if ii[key] and ii[key] !='':
                    new_dict[key]= ii[key]
        else:
            #neednt merge
            pass
        return new_dict

    def merge_rpm_dict(self, dict1, dict2, csvKey):
        for itemKey in dict2:
            if itemKey in dict1 and dict2[itemKey][3] == dict1[itemKey][3]:
                allitem_zd = ''
                allitem_zs = ''
                allitem_d = ''
                allitem_s = ''
                allitem_name = itemKey;
                if dict1[itemKey][0] == '' and dict2[itemKey][0] == '': #全表该包有入度，单表该包有入度->该字段为空
                    allitem_zd = ''
                elif dict1[itemKey][0] == '' and dict2[itemKey][0] !='': #全表该包有入度，单表该包无入度->认为有入度，将所有description放入description
                    allitem_zd = ''
                    allitem_d = dict2[itemKey][0]
                elif dict1[itemKey][0] !='' and dict2[itemKey][0] == '': #全表该包无入度，单表该包有入度->认为有入度，只取全表的放入description
                    allitem_zd = ''
                    allitem_d = dict1[itemKey][0]
                else:                                                    #全表该包有入度，单表该包有入度->将两者的zero_description合并
                    allitem_zd = dict2[itemKey][0]

                if dict1[itemKey][1] == '' and dict2[itemKey][1] == '': #全表该包有入度，单表该包有入度->该字段为空
                    allitem_zs = ''
                elif dict1[itemKey][1] == '' and dict2[itemKey][1] !='': #全表该包有入度，单表该包无入度->认为有入度，将所有description放入description
                    allitem_zs = ''
                    allitem_s = dict2[itemKey][1]
                elif dict1[itemKey][1] !='' and dict2[itemKey][1] == '': #全表该包无入度，单表该包有入度->认为有入度，只取全表的放入description
                    allitem_zs = ''
                    allitem_s = dict1[itemKey][1]
                else:                                                    #全表该包有入度，单表该包有入度->将两者的zero_description合并
                    allitem_zs = dict2[itemKey][1]

                if len(allitem_d) == 0 :
                    allitem_d = dict2[itemKey][2]
                if len(allitem_zs) != 0:
                    allitem_s = dict2[itemKey][3]

                dict1[itemKey]=(allitem_zd, allitem_zs, allitem_d, allitem_s, dict1[itemKey][4], dict1[itemKey][5],dict1[itemKey][6])
            else:
                dict1[itemKey]= dict2[itemKey]
            list1 = []
            for k,v in dict1.items():
                dictitem ={}
                dictitem['rpm_name'] = k
                dictitem['zero_description'] = v[0]
                dictitem['zero_summary'] = v[1]
                dictitem['description'] = v[2]
                dictitem['summary'] = v[3]
                list1.append(dictitem)

        return dict1;
    def merge_src_dict(self, dict1, dict2, csvKey):
        for itemKey in dict2:
            if itemKey in dict1 and dict2[itemKey][3] == dict1[itemKey][3]:
                allitem_zd = ''
                allitem_zs = ''
                allitem_d = ''
                allitem_s = ''
                allitem_name = itemKey;
                if dict1[itemKey][0] == '' and dict2[itemKey][0] == '': #全表该包有入度，单表该包有入度->该字段为空
                    allitem_zd = ''
                elif dict1[itemKey][0] == '' and dict2[itemKey][0] !='': #全表该包有入度，单表该包无入度->认为有入度，将所有description放入description
                    allitem_zd = ''
                    allitem_d = dict2[itemKey][0]
                elif dict1[itemKey][0] !='' and dict2[itemKey][0] == '': #全表该包无入度，单表该包有入度->认为有入度，只取全表的放入description
                    allitem_zd = ''
                    allitem_d = dict1[itemKey][0]
                else:                                                    #全表该包有入度，单表该包有入度->将两者的zero_description合并
                    allitem_zd = dict2[itemKey][0]

                if dict1[itemKey][1] == '' and dict2[itemKey][1] == '': #全表该包有入度，单表该包有入度->该字段为空
                    allitem_zs = ''
                elif dict1[itemKey][1] == '' and dict2[itemKey][1] !='': #全表该包有入度，单表该包无入度->认为有入度，将所有description放入description
                    allitem_zs = ''
                    allitem_s = dict2[itemKey][1]
                elif dict1[itemKey][1] !='' and dict2[itemKey][1] == '': #全表该包无入度，单表该包有入度->认为有入度，只取全表的放入description
                    allitem_zs = ''
                    allitem_s = dict1[itemKey][1]
                else:                                                    #全表该包有入度，单表该包有入度->将两者的zero_description合并
                    allitem_zs = dict2[itemKey][1]

                if len(allitem_d) == 0 :
                    allitem_d = dict2[itemKey][2]
                if len(allitem_zs) != 0:
                    allitem_s = dict2[itemKey][3]

                dict1[itemKey]=(allitem_zd, allitem_zs, allitem_d, allitem_s, dict1[itemKey][4], dict1[itemKey][5],dict1[itemKey][6], dict1[itemKey][7])
            else:
                dict1[itemKey]= dict2[itemKey]
            list1 = []
            for k,v in dict1.items():
                dictitem ={}
                dictitem['src_name'] = k
                dictitem['zero_description'] = v[0]
                dictitem['zero_summary'] = v[1]
                dictitem['description'] = v[2]
                dictitem['summary'] = v[3]
                dictitem['primary_rpm_name'] = v[4]
                dictitem['primary_description'] = v[5]
                dictitem['primary_summary'] = v[6]
                dictitem['rpm_name'] = v[7]
                list1.append(dictitem)
        return list1;

    def merge_csv_dict_list(self,dict_list1,dict_list2, csvKey):
        """
        合并csv文件中的 list,每一项是个dict
        """
        common_keys_merge_lists = []
        # sa
        common_a_keys_lists = []
        common_b_keys_lists = []
        not_common_keys_lists = []
        for ii in dict_list1:
            for jj in dict_list2:
                if ii[csvKey] == jj[csvKey]:
                    common_a_keys_lists.append(ii)
                    common_b_keys_lists.append(jj)
        not_common_keys_lists = [x for x in dict_list1 if x not in common_a_keys_lists] + [x for x in dict_list2 if x not in common_b_keys_lists]
        if len(common_a_keys_lists) == 0:
            return  not_common_keys_lists

        for i in range(len(common_a_keys_lists)):
            ii = common_a_keys_lists[i]
            jj = common_b_keys_lists[i]
            new_dict = {}
            if ii == jj:
                new_dict = ii
                common_keys_merge_lists.append(new_dict)
                continue
            else:
                new_dict = mergeDict(ii, jj, csvKey, config.header_dict['src_header'])
                common_keys_merge_lists.append(new_dict)

        all_list = common_keys_merge_lists + not_common_keys_lists
        return all_list

    def merge_csv_file(self,csv_path1,csv_path2, csvType, newCsvPath=""):
        """
        根据csv文件, 进行文件的合并
        """
        csvHeader = ''
        key = 'name'
        if csvType == "srpm":
            csvHeader = 'src_header'
            key = 'src_name'
        else:
            csvHeader = 'rpm_header'
            key = 'rpm_name'
        merge_scv_dict_list  = []
        scv_dict_list1 =  self.read_csv_file(csv_path1, csvHeader)
        scv_dict_list2 =  self.read_csv_file(csv_path2, csvHeader)
        merge_scv_dict_list = self.merge_csv_dict_list(scv_dict_list1,scv_dict_list2, key)
        if newCsvPath=="":
            newcsvfile = '_new_%s'%((csv_path1.split('/')[-1]).split('.')[0])
        else:
            newcsvfile= newCsvPath
        self.write_csv(merge_scv_dict_list, newcsvfile, csvHeader)
        return newcsvfile

    def merge_csv_file_as_dict_list(self,csv_path1,csv_path2):
        merge_scv_dict_list  = []
        scv_dict_list1 =  self.read_csv_file(csv_path1)
        scv_dict_list2 =  self.read_csv_file(csv_path2)
        merge_scv_dict_list = self.merge_csv_dict_list(scv_dict_list1,scv_dict_list2,"rpm_name")
        return  merge_scv_dict_list

"""
base:baseOs     app:appSteram       DDE:DDE     exp:experimental        ext:extras
HA:HighAvailability     plus:plus       PT:powerTools       SM:ShangMi
"""
def main(name):
    csv_obj = CSV()
    path = csv_obj.generate_csv_file(name)
    print("csv path = ",path)

if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
    #sys.exit()
    
