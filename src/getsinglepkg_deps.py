# SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import sys
from hawkey import Subject
import hawkey
import os
import rw_csv
import dot
import pandas as pd
import global_variable as gb

# 输入新的repo 路径，创建新repo
class Newrepo:
    def __init__(self,repoinfo):
        self.set_repo_name(self.urls_local_path(repoinfo))
        self.repo = hawkey.Repo(self.repo_name)

    #  设置repo 名字
    def set_repo_name(self,repo_name):
        self.repo_name = repo_name

    #  根据路径获取repo 名字
    def urls_local_path(self,url_path):
        return  url_path.split('/')[-1]
    
class Get_rpm_Dep:
    def __init__(self, reponame):
        self.repo = hawkey.Repo(reponame)
        self.sack = hawkey.Sack(make_cache_dir=True)
        self.repoName = reponame
         # 存储二进制关系
        self.rpm_dict = {}

    # 删除 repo
    def del_repo(self):
        del self.repo
        self.repo.repomd_fn = ""
        self.repo.filelists_fn = "" 
        self.repo.primary_fn = ""

    # 配置 repo 
    def conf_repofile_file_withpath(self,repopath):
        localpath = '%s/repodata/'%(repopath)
        items = os.listdir(localpath)

        for item in items:
            strr = os.path.basename(item) 
            if 'repomd.xml' in strr:
               repomdFile = '%s%s' %(localpath,strr)
            elif 'filelists.xml' in strr:
                filelistsFile = '%s%s' %(localpath,strr)
            elif 'primary.xml' in strr :
                primaryFile = '%s%s' %(localpath,strr)
        self.set_repo_file_to_hawkey(primaryFile, filelistsFile, repomdFile)

    def set_repo_file_to_hawkey(self,primaryFile, fileListsFile, repomdFile):
        self.repo.repomd_fn = repomdFile
        self.repo.filelists_fn = fileListsFile
        self.repo.primary_fn = primaryFile
        self.sack.load_repo(self.repo,load_filelists=True)

    # 获取单个包的直接依赖，返回set(),包含依赖的包hawkey.Package
    def get_deps(self,pkg):
        binfac = set()
        binfac.clear
        rpmfac = set()
        rpmfac.clear
        for s in pkg.requires:
            # s 为依赖的库文件
            binname = s.__str__()  #like this: libc.so***   libxslt.so***
            if binname.startswith('rpmlib'):
                continue
            binfac.add(binname)
        for req in sorted([str(req) for req in binfac]):
            # subject 类实现了一种解析此类输入的通用方法，并生成一个Query，
            # 列出与输入匹配的所有包，或一个Selector，选择与给定事务操作的输入最匹配的单个包。
            subject = hawkey.Subject(req)
            query = subject.get_best_query(self.sack)
            if query is None:
                return None
            else :
                if pkg not in self.rpm_dict.keys(): 
                    # query 为依赖的库文件对应的hawkey.Package  
                    for provider in query.run():
                        if provider.name != pkg.name:
                            if provider not in rpmfac:
                                rpmfac.add(provider)
        if pkg not in self.rpm_dict.keys():
            self.rpm_dict[pkg] = rpmfac
        return rpmfac

    def get_src_from_rpm(self, rpmName):
        pass

    # 以层级进行查询依赖包
    # get indirect depdences pkgs as layer type
    def get_indirect_deps(self,pkgs):
        indirectrpm = set()
        indirectrpm.clear
        getindirectrpm = set()
        getindirectrpm.clear
        if pkgs is None:
            print("pkgs is None")
        else :
            for hawkey_rpm in pkgs:
                getindirectrpm = self.get_deps(hawkey_rpm)
                for pp in getindirectrpm:
                    indirectrpm.add(pp) 
        return indirectrpm

    def getSepecificPkgDeps2(self,reponame,pkgName):
        pass

    def getSepecificPkgDeps(self,reponame,pkgname):
        self.rpm_dict.clear()
        self.conf_repofile_file_withpath(reponame)
        newrepo = Newrepo(self.repoName)
        name = reponame.split('/')[-1] + '-' + pkgname 
        pkgnames = hawkey.Query(self.sack).filter(reponame = self.repoName, name__glob = pkgname)

        if len(pkgnames) == 0:
            self.name = None
        else:
            for pkg in pkgnames:
                #pkg  =  pkgnames[0]
                pkgs = self.get_deps(pkg)
                if len(pkgs) != 0:
                    indirectrpm = self.get_indirect_deps(pkgs)
                    # i , j count allrpmnode
                    i = 0
                    j = 0
                    while indirectrpm is not None:
                        i = len(self.rpm_dict)
                        #sss =  indirectrpm
                        indirectrpm = self.get_indirect_deps(indirectrpm)
                        j = len(self.rpm_dict) 
                        if i == j:
                            break
            dots = dot.DOT()
            dots.generate_dot(self.get_rpm_dot_dict(),name)
            self.get_rpm_csv_dict_list(pkg)

        return name

    #获取源码包依赖
    def getSepecificSrcPkgDeps(self,primaryFile, filelistsFile, repomdFile,pkgname):
        self.repoName =  primaryFile.split('/')[-1]
        newrepo = Newrepo(self.repoName)
        newrepo.set_repo_file_to_hawkey(primaryFile, filelistsFile, repomdFile)
        name = newrepo.repo_name + '-' + pkgname 
        pkgnames = hawkey.Query(self.sack).filter(reponame=self.repoName, sourcerpm=pkgname)
        if len(pkgnames) == 0:
            self.name = None
        else:
            for pkg in pkgnames:
                pkgs = self.get_deps(pkg)
                if len(pkgs) != 0:
                    indirectrpm = self.get_indirect_deps(pkgs)
                    i = 0
                    j = 0
                    while indirectrpm is not None:
                        i = len(self.rpm_dict)
                        indirectrpm = self.get_indirect_deps(indirectrpm)
                        j = len(self.rpm_dict) 
                        if i == j:
                            break
            dots = dot.DOT()
            dots.generate_dot(self.get_dot_dict(),'rpm_'+name)
            self.get_rpm_csv_dict_list(pkg)
        newrepo.del_repo()   
        return name

    #生成 源码与二进制对应字典
    def get_generate_src_dict(self):
        dcit_src = {}
        for pkg in self.rpm_dict.keys():
            same_src = set()
            for pkgs in self.rpm_dict.keys():
                if  pkg.sourcerpm == pkgs.sourcerpm:
                    same_src.add(pkgs)
                else :
                    continue
                dcit_src[pkg.sourcerpm] = same_src
        return dcit_src

    # 生成 csv 字典列表
    def get_src_scv_dict_list(self,pkgnames,dot_dict):
        src_dict = {}
        scv_dict_list = []
        src_dict = self.get_generate_src_dict()

        # 设置入度为0 的标志位
        flag_zero = 0        
        for jj in dot_dict.keys():
            if pkgnames.sourcerpm in dot_dict[jj]:
                flag_zero = 1
                break  

        for pkg in src_dict.keys(): 
            scv_dict={}          
            zero_description = ''
            zero_summary = ''
            description = ''
            summary = ''
            for pkgs in src_dict[pkg]:
                if pkgs in src_dict[pkgnames.sourcerpm] and flag_zero == 0 :
                        if pkgs.description != None and pkgs.description not in zero_description:
                            zero_description +=  pkgs.description + ''
                        if pkgs.summary  not in zero_summary :
                            zero_summary +=  pkgs.summary +''
                        description = ''
                        summary = ''
                else:
                    zero_description = ''
                    zero_summary = ''
                    if pkgs.description != None and pkgs.description not in description :
                        description += pkgs.description
                    if pkgs.summary  not in  summary :
                        summary += pkgs.summary + ''
            scv_dict["zero_description"] = zero_description
            scv_dict["zero_summary"] =  zero_summary
            scv_dict["summary"] =   summary  
            scv_dict["description"] = description
            scv_dict["name"] = pkg
            scv_dict_list.append(scv_dict)
        return  scv_dict_list
    
    # 生成 dot 字典
    def get_dot_dict(self):
        dot_dict = {}
        i = 0
        for pkg in self.rpm_dict:
            if pkg.sourcerpm not in dot_dict.keys():
                 dep_pkg_set = set() 
            else :
                dep_pkg_set = dot_dict[pkg.sourcerpm] 
            for dep_pkg in self.rpm_dict[pkg]:
                if dep_pkg.sourcerpm not in dep_pkg_set and dep_pkg.sourcerpm != pkg.sourcerpm:
                    dep_pkg_set.add(dep_pkg.sourcerpm)
                else :
                    continue
            dot_dict[pkg.sourcerpm] = dep_pkg_set
            i += len(dep_pkg_set)
            print(len(dep_pkg_set),i)
            dep_pkg_set.clear
        print(len(dot_dict))
        return  dot_dict
        # dot.generate_dot(self.rpm_dict,self.name)

    def srpmName_analyze(self,srpmName):
        # 解析源码包名
        srpmName_2 = srpmName.split("-")[-2]
        sName = srpmName.split("-%s" % srpmName_2)[0] 
        return sName    

    # 带版本的dot字典，一个repo源里可能有多个不同版本的pkg，写入dot时版本后，会有重复的包依赖关系
    def get_rpm_dot_dict(self):
        rpm_dot_dict = {}
        for pkg in self.rpm_dict:
            rpm_dep_set = set()
            for pkgs in self.rpm_dict[pkg]:
                rpm_dep_set.add(str(pkgs))
            print(len(rpm_dep_set))
            rpm_dot_dict[str(pkg)] = rpm_dep_set
        print(rpm_dot_dict.keys())   
        return rpm_dot_dict
    
    # 不带版本二进制dot
    def get_rpm_dot_dict_without_v(self):
        rpm_dot_dict = {}
        
        for pkg in self.rpm_dict:
            rpm_dep_set = set()
            for pkgs in self.rpm_dict[pkg]:
                rpm_dep_set.add(self.srpmName_analyze(str(pkgs)))
            print(len(rpm_dep_set))
            rpm_dot_dict[self.srpmName_analyze(str(pkg))] = rpm_dep_set
        print(rpm_dot_dict.keys())   
        return rpm_dot_dict
    
    # 不带版本二进制csv 
    def get_rpm_csv_dict_list(self,pkgnames):
        scv_dict_list = []
            # 设置入度为0 的标志位
        flag_zero = 0 
        if pkgnames in self.rpm_dict.values():
            flag_zero = 1

        df = pd.DataFrame(columns=('rpm_name',
                                   'zero_description',
                                   'zero_summary',
                                   'description',
                                   'summary',
                                   'src_name'))

        for pkg in self.rpm_dict :
            df = df.append(pd.DataFrame({'rpm_name' : [self.srpmName_analyze(str(pkg))],
                        'zero_description' : [pkg.description if pkg == pkgnames and flag_zero == 0 else ''],
                        'zero_summary' : [pkg.summary if pkg == pkgnames and flag_zero == 0 else '' ],
                        'description' : ['' if pkg == pkgnames and flag_zero == 0 else pkg.description ],
                        'summary' : [''  if pkg == pkgnames and flag_zero == 0 else pkg.summary ],
                        'src_name' : [self.srpmName_analyze(pkg.sourcerpm)]}),ignore_index=True)
        df.to_csv('{}rpm_{}.csv'.format(gb.csv_path,pkgnames),index=False)

def main(args):
    if args:
        ss = Get_rpm_Dep("reponame")
        ss.getSepecificPkgDeps(reponame = args[0],pkgname = args[1])

if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except  KeyboardInterrupt:
        sys.exit(1)

