# SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import configparser
import os

CONFIGPATH='./fcfl.ini'

#记录最开始打开的数据库类型，后面跨数据库查找时，在其他数据库查找完成后要回到最先打开的数据库
db_type = ""   
class fcfl_config:
    
    def __init__(self):
        self.cfg = configparser.ConfigParser()
        self.cfg.read(CONFIGPATH)
        pass

    def get_storage_path(self):
        v_storage_path = self.cfg.get('storage', 'path')
        return v_storage_path

    def get_DB_path(self):
        v_db_path = self.cfg.get('DB', 'path')
        return v_db_path
        
    def get_repo_path(self):
        v_repo_path = self.cfg.get('repo', 'path')
        return v_repo_path

    def get_csv_path(self):
        v_csv_path = self.cfg.get('csv', 'path')
        print(v_csv_path)
        return v_csv_path

    def get_dot_path(self):
        v_dot_path = self.cfg.get('dot', 'path')
        print(v_dot_path)
        return v_dot_path

    def get_layer_config_file(self):
        layer_config_file = self.cfg.get('layer',"path")
        return 

# global_info

#文件保存路径
dot_path = "/var/fcfl/dot_files/"
repo_type_list = ["baseOs","appStream","DDE","Experimental","Extras","HighAvailability","Plus","PowerTools"]

# table header
header_dict = {
    'src_header' : ["src_name","zero_description","zero_summary","description","summary","primary_rpm_name","primary_description","primary_summary","rpm_name"] , 
    'rpm_header' : ["rpm_name","zero_description","zero_summary","description","summary","src_name","srcPkgVersion","srcPkgRelease"] ,
    'fcfl_header' : ["src_name","layer","classification","manimark"],
    'domain_header' : ["src_name","domain_info","manimark"] ,
    'classification_test' : ["rpm_name","description","summary","library_value","tool_value","service_value","font_value","classification","class_by_files"],
    'human_bin_classification':['rpm_name','description','zero_summary','zero_description','summary','src_name','classification'],
    'human_rpm_classification':['rpm_name','zero_description','zero_summary','description','summary','src_name','分层分类'],
    'human_src_classification':['src_name','zero_description','zero_summary','description','summary','primary_rpm_name','primary_description','primary_summary','rpm_name','分层分类',None,None,'源码包名'],
    'unsuited_classification': ["rpm_name","description","summary","src_name","library_value","tool_value","service_value","font_value","classification","human_soure_class","human_classification"],
    'file_classification' : ["rpm_name","dirs","filenames","classification_info","classification"],
    'u_header' : ["pkgName","zero_description","zero_summary","description","summary","src_name",'sections','tag','proprity'],
    'u_src_header' : ["src_name","pkgName","description","zero_description","summary","zero_summary",'sections','tag','proprity'],
    'file_classification2' : ["rpm_name","dirs","filenames","classification_info","classification","classhuman"],
    'duibi': ['rpm_name','description','zero_summary','zero_description','summary','src_name','classification', 'ori', 'mechinclass', 'filenames', 'classinfo'],
    'diff': ['name', 'label', 'mc'],
    'diff2': ['name', 'label', 'mc','stan'],
    'res': ['rpm_name', 'classification']
}

#源码包入度和出度为零列表
src_zero_in_list = []    # src 入度为0列表
src_zero_out_list = []   # src 出度为0列表
rpm_zero_in_list = []    # rpm 入度为0列表
rpm_zero_out_list = []  # rpm 出度为0列表

