# SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from get_dep import DEP
from graphviz import Digraph  
import sys
import hawkey
import os
import config 
# from public import update_zero_in_and_out
from layer_sql.process_layer_sql import STORAGE_SQL
from config import fcfl_config
import pprint

class DOT:

    #传入dot不带扩展名的文件名，通过config获取路径。入参字典为：{pkg:set(dep1,dep2)}
    def generate_dot(self,dep_dict,name):
        """
        生成dot文件
        dep_dict：依赖字典
        name：dot名称
        """
        #判断文件是否存在  存在先删除
        name = name + '.dot'
        if name in os.listdir(config.get_dot_path):
            os.remove(config.dot_path + name)
        path = config.dot_path + name 
        self.generate_dotfile(dep_dict, path)
        return path

    def generate_dotfile(self, dep_dict,dot_path):
        dot = Digraph(name = "dep_Picture",format="png")
        dot.graph_attr['rankdir'] = 'LR'
        countNode=0
        countEdge=0
        if os.path.exists(dot_path):
            os.remove(dot_path)

        #在dot文件中填入节点和边, 字典是pkg:[dep1,dep2}
        for key in dep_dict:
            dotkey=""
            try:
                dotkey = hawkey.split_nevra(str(key)).name
                dot.node(dotkey)
            except:
                dot.node(key)
            countNode+=1
            for edge in dep_dict[key]:
                try:
                    dotedge = hawkey.split_nevra(str(edge)).name
                    dot.edge(dotkey,dotedge)
                except:
                    dot.edge(key,edge)
                countEdge+=1

        with open(dot_path,'w') as fdot:
            print(dot.source,file=fdot)

        #返回dot文件路径
        return dot_path

    # debian 平台    
    #传入dot不带扩展名的文件名，通过config获取路径。入参字典为：{pkg:set(dep1,dep2)}
    def generate_dot_u(self,dep_dict,name):
        """
        生成dot文件
        dep_dict：依赖字典
        name：dot名称
        """
        #判断文件是否存在  存在先删除
        name = name + '.dot'
        if name in os.listdir(config.dot_path):
            os.remove(config.dot_path + name)
        path = config.dot_path + name 
        self.generate_dotfile_u(dep_dict, path)
        return path

    def generate_dotfile_u(self, dep_dict,dot_path):
        dot = Digraph(name = "dep_Picture",format="png")
        dot.graph_attr['rankdir'] = 'LR'
        countNode=0
        countEdge=0
        if os.path.exists(dot_path):
            os.remove(dot_path)

        #在dot文件中填入节点和边, 字典是pkg:[dep1,dep2}
        for key in dep_dict.keys():
            dot.node(key)
            countEdge+=1
            for edge in dep_dict[key]:
                dot.edge(key,edge)
                countEdge+=1
                
        with open(dot_path,'w') as fdot:
            print(dot.source,file=fdot)
        print("dot len keys=%d" %(len(dep_dict.keys())))

        #返回dot文件路径
        return dot_path
    
    # 生成总dot文件 
    def generate_src_all_dotFiles(self, repoDir):
        """
        生成总的src dot时更新一下入度和出度为0的列表
        """
        dep_obj = DEP()
        all_dep_dict = {}
        for type in config.repo_type_list:
            dep_dict = {}
            dep_dict = dep_obj.get_one_repo_src_form_dep(type, repoDir)
            all_dep_dict = self.merge_dot_dict(dep_dict,all_dep_dict)
            
        path = self.generate_dot(all_dep_dict,"src_all")
        return path

    # 生成总dot文件 
    def generate_fcfl_pkgs_dotFiles(self):
        """
        生成总的fcfl dot
        """
        new_db = STORAGE_SQL()
        dot_dict = new_db.loadDepsFromDependeceTab() 
        path = self.generate_dot(dot_dict,"fcfl_all_dep")
        return path

    # 生成总dot文件 
    def generate_fcfl_pkg_dotFiles(self,pkg):
        """
        生成pkg的fcfl dot
        """
        new_db = STORAGE_SQL()
        dot_dict = new_db.get_pkg_dot_dict_from_table(pkg) 
        path = self.generate_dot(dot_dict,"fcfl_"+pkg+"_dep")
        return path

    def generate_rpm_all_dotFiles(self, repoDir):
        """
        生成总的rpm dot时更新一下入度和出度为0的列表
        """
        dep_obj = DEP()
        all_dep_dict = {}
        all_dep_dict = self.generate_rpm_all_dot_dict(repoDir);
        path = self.generate_dot(all_dep_dict,"rpm_all")
        return path

    def generate_rpm_all_dot_dict(self, repoDir):
        """
        生成总的rpm dot时更新一下入度和出度为0的列表
        """
        dep_obj = DEP()
        all_dep_dict = {}
        for type in config.repo_type_list:
            dep_dict = {}
            dep_dict = dep_obj.get_one_repo_rpm_form_dep(type, repoDir)
            all_dep_dict = self.merge_dot_dict(dep_dict,all_dep_dict)

        return all_dep_dict
    
    def get_NodeDep_from_str(self,s):
        """
        处理读取的字符串，返回节点和其依赖
        """
        s = s.replace('"','').replace('\n','')
        index = s.split("->")
        node = index[0].strip()
        dep = index[1].strip()
        return node, dep

    def read_dot_file(self,dot_path):
        """
        读取dot文件，返回依赖字典
        """
        dep_dict = {}
        with open(dot_path,'r') as f:
            line = f.readline()
            while line:
                if "->" in line:
                    pkg,dep = self.get_NodeDep_from_str(line)
                    if pkg in dep_dict:
                        dep_dict[pkg].add(dep)
                    else:
                        tmp = set()
                        tmp.add(dep)
                        dep_dict[pkg] = tmp
                line = f.readline()
        return dep_dict

    def read_dot_file_for_test(self,dot_path):
        """
        读取dot文件，返回依赖字典
        参数：dot文件路径
        返回值：所有节点和依赖的字典 {nodeName：[depList]}
        """
        dep_list = []
        with open(dot_path,'r') as f:
            line = f.readline()
            while line:     
                if "->" not in line:
                    if 'digraph dep_Picture {' not in  line and  'graph [rankdir=LR]' not in  line and '}' not in  line :
                        dep_list.append(line)
                line = f.readline()
        return dep_list

    def merge_dot_dict(self,dict1,dict2):

        dot_dict = {}
        d2_keys_not_in_d1 = set(dict2.keys()) - set(dict1.keys())
        d1_keys_not_in_d2 = set(dict1.keys()) - set(dict2.keys())
        common_keys = dict2.keys() & dict1.keys()
        dict1count =0
        for i in common_keys:
            dot_dict[i]= list(set(list(dict1[i]) +list( dict2[i])))
            dict1count+=len(set(list(dict1[i])))
        for i in d1_keys_not_in_d2:
            dict1count+=len(set(list(dict1[i])))
            dot_dict[i]=dict1[i]
        for i in d2_keys_not_in_d1:
            dot_dict[i]=dict2[i]
        return dot_dict

    def merge_dot_file(self,dot_path1,dot_path2):
        merge_dot_dict = {}
        dict1 = self.read_dot_file(dot_path1) 
        dict2 =  self.read_dot_file(dot_path2)  
        merge_dot_dict = self.merge_dot_dict(dict1,dict2)
        newdotfilename = '_new_%s'%((dot_path1.split('/')[-1]).split('.')[0])
        self.generate_dot(merge_dot_dict,newdotfilename)
        return newdotfilename
        
if __name__ == "__main__":
    dot = DOT()
    sys.exit(dot.generate_all_dotFiles())
