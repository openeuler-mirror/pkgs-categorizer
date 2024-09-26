# SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import read_sql as sql
import config 
import pprint
import hawkey 
test_name = 'cups'
def srpmName_analyze(srpmName):
    """
    功能：源码包名字解析
    思路：源码包名用“-”分开后，最有两部分对应的是version和release
        用“-veesion”将源码包分为两部分，取前面的部分就是源码包名
    """
    srpmName_2 = srpmName.split("-")[-2]
    sName = srpmName.split("-%s" % srpmName_2)[0]
    return sName

def pkg_is_need_remove(pkgName):
    """
    软件包是否需要去除
    返回值：1 需要去除，0 不需要去除
    """
    if pkgName.endswith("-doc") == True or \
        pkgName.endswith("-devel") == True or \
        pkgName.find("glibc-langpack") != -1 or \
        pkgName.find("glibc-minimal") != -1 or \
        pkgName.find("glibc-all") != -1 or \
        pkgName.find("glibc-benchtests") != -1 or \
        pkgName.endswith("-tests") == True or \
        pkgName.endswith("-test") == True or \
        pkgName.endswith("-debug") == True or\
        pkgName.endswith("-demo") == True or\
        pkgName.find("-debuginfo") != -1 or\
        pkgName.startswith("filesystem") == True or\
        pkgName.startswith("basesystem") == True or\
        pkgName.startswith("system-release") == True or\
        pkgName == 'setup':
        return True
    return False

def rpm_is_need_remove(pkgName):
    """
    软件包是否需要去除
    返回值：1 需要去除，0 不需要去除
    """
    if pkgName.endswith("-doc") == True or \
        pkgName.endswith("-devel") == True or \
        pkgName.find("glibc-langpack") != -1 or \
        pkgName.find("glibc-minimal") != -1 or \
        pkgName.find("glibc-all") != -1 or \
        pkgName.find("glibc-benchtests") != -1 or \
        pkgName.endswith("-tests") == True or \
        pkgName.endswith("-test") == True or \
        pkgName.endswith("-debug") == True or\
        pkgName.endswith("-demo") == True or\
        pkgName.find("-debuginfo") != -1 :
        return True
    return False
  
def src_is_need_remove(pkgName):
    """
    软件包是否需要去除
    返回值：1 需要去除，0 不需要去除
    """
    if pkgName.startswith("filesystem") == True or\
        pkgName.startswith("basesystem") == True or\
        pkgName.startswith("system-release") == True or\
        pkgName == 'setup':
        return True
    return False


"""
获取直接依赖有两种方法，用方法一生成所有软件包的dot文件大约是40s，方法二大约15s
基本思路：
    方法一：1）获得所有源码包
           2）获取每个源码包提供的二进制包
           3）查找二进制包的依赖，对依赖查找提供者，将提供者还原为源码包

    方法二：1）读取packages（pkgkey，rpm_sourcerpm），provides（name，pkgkey），requires（name，pkgkey），files（name，pkgkey）
              四个表中的所有软件包括号中的信息，输出四个字典 
           2）将requires对应的字典还原为源码包字典
方法二只用到了 源码包名解析接口
"""

"""
方法二
"""
class DEP:
    def __init__(self):
        #self.sql_obj = sql.REPO_SQL()
        self.dep_obj = sql.REPO_DEP()
        self.src_dict = {}
        self.rpm_dict = {}

        self.src_name = ""
        self.rpm_name = ""
        self.version  = ""
        self.release  = ""
        self.rpm_nvr = ""
        self.src_nvr = ""


    def get_four_dict(self,repoType, repoDir):
        """
        获取4个字典
        """
        self.req_dict,self.prov_dict,self.files_dict,self.pack_dict = self.dep_obj.get_all_dicks(repoType, repoDir)

    def compare_nvr(self, nvr1, nvr2, condition):
        sack = hawkey.Sack()
        re = sack.evr_cmp(nvr1, nvr2)

        if condition == 'EQ':
            if re == 0:
                return True
        # prov > req . so req(nvr1) <= prov(nvr2)    
        elif condition == 'GE' or condition == 'GT':
            if re <= 0:
                return True
        elif condition =="LT" or condition == 'LE':
            if re >= 0:
                return True
        else:
            return False
        return False


    def findfromprovs(self, req_name,  req_version, req_release, req_flags):
        for prov_flag, prov_pid in self.prov_dict[req_name]:
            if self.getDepFromPkgDict(prov_pid, req_name,  req_version, req_release, req_flags) == True:
                return True
        return False

    def findfromfiles(self, req_name,  req_version, req_release, req_flags):
        for file_pkg_key in self.files_dict[req_name]:
            if self.getDepFromPkgDict(file_pkg_key, req_name,  req_version, req_release, req_flags) == True:
                return True
        return False

    def saveToDict(self, prov_src_nvr, prov_nvr, pkg_dep_name):
        # 存rpm
        if pkg_dep_name == self.rpm_name or\
                rpm_is_need_remove(pkg_dep_name) == True :
            return True
        try:
            #self.rpm_dict[self.rpm_name].add(hawkey.split_nevra(prov_nvr).name)
            self.rpm_dict[self.rpm_name].add(prov_nvr.rsplit('-',2)[-3])
        except:
            print("error prov_nvr:",prov_nvr)
            quit()

        # 存srpm
        if prov_src_nvr.rsplit('-',2)[-3] == self.src_name or\
                src_is_need_remove(prov_src_nvr) == True:
            return True
        try:
            self.src_dict[self.src_name].add(prov_src_nvr.rsplit('-',2)[-3])
        except:
            print("error prov_src_nvr:",prov_src_nvr)
            quit()
        return True

    def getDepFromPkgDict(self, dep_key, req_name,  req_version, req_release, req_flags):
        pkg_dep_src,pkg_dep_name, pkg_dep_ver, pkg_dep_rel = (self.pack_dict[dep_key][0],self.pack_dict[dep_key][1],self.pack_dict[dep_key][2], self.pack_dict[dep_key][3])
        #因为req_name不一定是包名，有可能是so或符号，但是req_name可以索引到prov，然后从pkg查到的一定是包名
        #因此req_nvr通过pkg_dep_name, req_version,req_release组合
        req_nvr = self.build_nvr(pkg_dep_name, req_version, req_release)
        prov_nvr = self.build_nvr(pkg_dep_name, pkg_dep_ver, pkg_dep_rel)
        prov_src_nvr = pkg_dep_src.replace(".src.rpm","")
        # 如果req_flags 为空
        if req_flags is None:
            # 传进来的依赖等于查到的依赖
            # 如果源码包名就是自己 或者 根据依赖规则删除
            return self.saveToDict(prov_src_nvr,prov_nvr,pkg_dep_name)
        else:
            if self.compare_nvr(req_nvr, prov_nvr, req_flags) == True:
                return self.saveToDict(prov_src_nvr,prov_nvr,pkg_dep_name)
        return False

    def build_nvr(self, name, version, release):
        nvr=''+name
        if version is not None :
            nvr=''+name+'-'+str(version)
            if release is not None: 
                nvr=''+str(name)+'-'+str(version)+'-'+str(release)
        return nvr

    def get_one_repo_rpm_form_dep(self,repoType, repoDir):
        self.get_pkg_dep_dict(repoType, repoDir)
        return self.rpm_dict

    def get_one_repo_src_form_dep(self, repoType, repoDir):
        self.get_pkg_dep_dict(repoType, repoDir)
        return self.src_dict

    def get_pkg_dep_dict(self, repoType, repoDir):
        #获取当前数据库名称
        self.db_type = repoType 
        self.rpm_dict = {}
        self.src_dict = {}
        self.get_four_dict(repoType, repoDir)
        for pkgid, pack_value in self.pack_dict.items():
            #查找依赖时，去除源码包提供的一些二进制包的依赖
            srcFileName = pack_value[0]
            self.rpm_name = pack_value[1]
            self.version  = pack_value[2]
            self.release  = pack_value[3]

            if rpm_is_need_remove(self.rpm_name):
                continue
            if src_is_need_remove(srcFileName):
                continue

            self.src_nvr = srcFileName.replace(".src.rpm","")
            self.rpm_nvr = self.build_nvr(self.rpm_name, self.version, self.release)
            #key还原成源码包名
            self.src_name = hawkey.split_nevra(srcFileName).name

            #同一个源码包提供的rpm包整合
            #print("########################")
            #print("src_nvr:", self.src_nvr, "    rpm_nvr", self.rpm_nvr)

            # 源码包用nvr
            #if self.src_nvr not in self.src_dict.keys():
            #    self.src_dict[self.src_nvr] = set()
            # 源码包用name
            if self.src_name not in self.src_dict.keys():
                self.src_dict[self.src_name] = set()
            # 用nvr
            #if self.rpm_nvr not in self.rpm_dict.keys():
            #    self.rpm_dict[self.rpm_nvr] =set()
            # 用name
            if self.rpm_name not in self.rpm_dict.keys():
                if self.rpm_name == "aspnetcore":
                    print("1111111111")
                self.rpm_dict[self.rpm_name] =set()

            if pkgid not in self.req_dict:
                continue
            self.findPkgReq(pkgid, repoDir)

    def findPkgReq(self, pkgid, repoDir):
        #从req字典中遍历二进制包依赖
        for req_name, req_version, req_release, req_flags in self.req_dict[pkgid]:
            #在provides中查找提供者,找打会置1.
            # 如果req_name 在prov中
            if req_name in self.prov_dict.keys():
                #print("   find", req_name, " from prov...")
                if True == self.findfromprovs(req_name,  req_version, req_release, req_flags):
                    continue
            tmp_name = req_name.split(r"/")[-1]
            if tmp_name in self.prov_dict.keys():
                #print("    find ", req_name, " not in prov,try tmp_name:", tmp_name)
                if True == self.findfromprovs(tmp_name,  req_version, req_release, req_flags):
                    continue
                #print(" prov_dict not found", self.rpm_name, "deps:", req_name)
            if req_name in self.files_dict.keys():
                #print("   find", req_name, " from file...")
                if True == self.findfromfiles(req_name, req_version, req_release, req_flags):
                    continue
            if tmp_name in self.files_dict.keys():
                #print("    find ", req_name, " not in files,try tmp_name:", tmp_name)
                if True == self.findfromfiles(tmp_name, req_version, req_release, req_flags):
                    continue
                #print("files_dict not found", self.rpm_name, "deps:", req_name)
            #跨文件查找提供者
            #else:
            findout = False
            for sql_type in config.repo_type_list:
                if sql_type == self.db_type:
                    continue
                #print("   find ", req_name,"from DB:", sql_type, "...")
                findout = self.search_src_name_files(sql_type, req_name, req_version, req_release, req_flags, repoDir) #跨文件查找
                if True == findout:
                    break 
            if findout == True:
                continue
            #else:
                #print("   not found!,pkg=", self.rpm_name, "deps:", req_name)


    def search_src_name_files(self,sql_path, req_n, req_v, req_r, req_f, repoDir):
        """
        跨文件查找依赖
        输入参数：sql_path  数据库路径
                rpm_name  查找的二进制名
        """
        path = self.dep_obj.repo_obj.read_path(sql_path, repoDir, "primary")
        self.dep_obj.repo_obj.open_sql(path)

        if self.findFromProvsTab(req_n, req_v, req_r, req_f) == True:
            self.dep_obj.repo_obj.close_sql()
            return True

        if self.findFromFilesTab(req_n, req_v, req_r, req_f) == True:
            self.dep_obj.repo_obj.close_sql()
            return True

        self.dep_obj.repo_obj.close_sql()
        return False

    def findFromProvsTab(self, req_name,  req_v, req_r, req_flags):
        if req_flags is None:
            p_pkgkeys = self.dep_obj.repo_obj.getPkgkeyFromProvides_name(req_name)
            if len(p_pkgkeys) == 0:
                tmp_name = req_name.split(r"/")[-1]
                p_pkgkeys = self.dep_obj.repo_obj.getPkgkeyFromProvides_like_name(tmp_name)
            for p_pkgid in p_pkgkeys:
                depsInfo = self.dep_obj.repo_obj.getSNVRFromPkg(p_pkgid)
                for item in depsInfo:
                    #print("   find from PorvTab if req_name", req_name, "req_src:", item[0],"N:",  item[1],"V:",  item[2],"R:", item[3])
                    src_dep_nvr = item[0].replace(".src.rpm","")
                    pkg_dep_nvr = self.build_nvr(item[1], item[2], item[3])
                    return self.saveToDict(src_dep_nvr,pkg_dep_nvr,item[1])
            return False
        else:
            p_pkgkeys = self.dep_obj.repo_obj.getPkgkeyFromProvides_name(req_name)
            if len(p_pkgkeys) == 0:
                tmp_name = req_name.split(r"/")[-1]
                p_pkgkeys = self.dep_obj.repo_obj.getPkgkeyFromProvides_like_name(tmp_name)
            for p_pkgid in p_pkgkeys:
                depsInfo = self.dep_obj.repo_obj.getSNVRFromPkg(p_pkgid)
                for item in depsInfo:
                    #print("   find from PorvTab else req_name", req_name, "req_src:", item[0],"N:",  item[1],"V:",  item[2],"R:", item[3])
                    req_nvr = self.build_nvr(item[1],  req_v ,req_r)
                    src_dep_nvr = item[0].replace(".src.rpm","")
                    pkg_dep_nvr = self.build_nvr(item[1], item[2], item[3])
                    if self.compare_nvr(req_nvr, pkg_dep_nvr, req_flags) == True:
                        return self.saveToDict(src_dep_nvr,pkg_dep_nvr,item[1])
            return False


    def findFromFilesTab(self, req_name,  req_v, req_r, req_flags):
        if req_flags is None:
            p_pkgkeys = self.dep_obj.repo_obj.getPkgkeyFromFiles_name(req_name)
            if len(p_pkgkeys) == 0:
                tmp_name = req_name.split(r"/")[-1]
                p_pkgkeys = self.dep_obj.repo_obj.getPkgkeyFromFiles_like_name(tmp_name)
            for p_pkgid in p_pkgkeys:
                depsInfo = self.dep_obj.repo_obj.getSNVRFromPkg(p_pkgid)
                for item in depsInfo:
                    #print("   find from FilesTab if req_name", req_name, "req_src:", item[0],"N:",  item[1],"V:",  item[2],"R:", item[3])
                    src_dep_nvr = item[0].replace(".src.rpm","")
                    pkg_dep_nvr = self.build_nvr(item[1], item[2], item[3])
                    return self.saveToDict(src_dep_nvr,pkg_dep_nvr,item[1])
            return False
        else:
            p_pkgkeys = self.dep_obj.repo_obj.getPkgkeyFromFiles_name(req_name)
            if len(p_pkgkeys) == 0:
                tmp_name = req_name.split(r"/")[-1]
                p_pkgkeys = self.dep_obj.repo_obj.getPkgkeyFromFiles_like_name(tmp_name)
            for p_pkgid in p_pkgkeys:
                depsInfo = self.dep_obj.repo_obj.getSNVRFromPkg(p_pkgid)
                for item in depsInfo:
                    #print("   find from FilesTab else req_name", req_name, "req_src:", item[0],"N:",  item[1],"V:",  item[2],"R:", item[3])
                    req_nvr = self.build_nvr(item[1],  req_v, req_r)
                    pkg_dep_nvr = self.build_nvr(item[1], item[2], item[3])
                    src_dep_nvr = item[0].replace(".src.rpm","")
                    return self.saveToDict(src_dep_nvr,pkg_dep_nvr,item[1])
            return False


#repo_type_list = ["baseOs","appStream","DDE","Experimental","Extras","HighAvailability","Plus","PowerTools"]
if __name__ == "__main__":
    dep = DEP()
    dep.get_one_repo_rpm_form_dep("appStream")
    dep.get_one_repo_src_form_dep("appStream")

