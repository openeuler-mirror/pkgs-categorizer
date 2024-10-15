# SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from layer_sql.process_layer_sql import STORAGE_SQL 
import csv
from rw_csv import CSV
from dot import DOT
from algorithm.fcfl_result import LAYER
from log import FCFLLog
from algorithm.Layer.bert_pkg_emb import do_layer
from algorithm.Layer.tools import csv_and_dot_to_json 

#import algorithm.func_classer.func_classifier as c_fc
import pandas as pd
import sys

from algorithm.classer.class_algorithm  import class_algorithm 
import algorithm.data_config as data_config

def readRepoPkgInfo(repoDir):
    dot_obj = DOT()
    dot_file = dot_obj.generate_rpm_all_dotFiles(repoDir)
    csv_obj = CSV()
    csv_file = csv_obj.generate_rpm_all_csv_files(repoDir)
    log = FCFLLog()
    log.info("export csvfile:%s" %csv_file)
    log.info("export dotfile:%s" %dot_file)
    return (dot_file,csv_file)

def updatePkg(dotfile, csvpath, reOrg):
    log = FCFLLog()
    log.info("execute updatePkg, csvPath=%s, csvPath=%s reOrg=%d" %(dotfile, csvpath, reOrg))
    layer_obj = LAYER()            #??/
    
    #step 1: 获取单一包和全部包的依赖关系。
    #读单一包dot文集，获取单一包依赖字典name:set(deps)
    dot_obj = DOT()
    single_dep_dict = dot_obj.read_dot_file(dotfile)
    log.info("load single dotfile, len=%d" %(len(single_dep_dict.keys())))

    stor_sql = STORAGE_SQL()
    #获取已经存库全部包依赖关系是name:set(deps)
    all_dep_dict= stor_sql.loadDepsFromDependeceTab()
    log.info("load all dotfile, len=%d" %(len(all_dep_dict.keys())))

    #合并依赖字典
    for i in single_dep_dict:
        if i not in all_dep_dict:
            all_dep_dict[i] = single_dep_dict[i]
        else:
            all_dep_dict[i]= all_dep_dict[i] | single_dep_dict[i]
    dot_path = dot_obj.generate_dot(all_dep_dict,"merged_all")
    log.info("=merge dot len=%d" %(len(all_dep_dict.keys())))

    #读取软件包描述信息
    #读单独csv文件, 返回一个字典
    csv_obj = CSV()
    single_csv_dict = csv_obj.read_csv_file2(csvpath,'rpm_header')
    log.info("single csv len=%d" %(len(single_csv_dict)))

    c_al = class_algorithm()
    #合并依赖字典
    #获取库内csv
    all_csv_dict = stor_sql.get_pkgs_rpm_desc_csv_dict_from_table()
    log.info("all csv len=%d" %(len(all_csv_dict)))

    final_dist = csv_obj.merge_rpm_dict(all_csv_dict,single_csv_dict, 'rpm_name')
    log.info("final csv len=%d" %(len(final_dist)))
    final_dict_list = []
    for k,v in final_dist.items():
        dict_item = dict()
        dict_item['src_name'] = v[4]
        dict_item['zero_description'] = v[0]
        dict_item['zero_summary'] = v[1]
        dict_item['description'] = v[2]
        dict_item['summary'] = v[3]
        dict_item['srcPkgVersion']=v[5]
        dict_item['srcPkgRelease']=v[6]
        final_dict_list.append(dict_item)
    layer_dict = do_layer('./data/1228_bin_all_test_set2.json')
    print("计算分层分类中............")
    '''
    # 计算分层分类
    print("222222计算分层分类中............")
    try:
        layer_dict = do_layer('./data/1228_bin_all_test_set2.json')
        print("calc d size %d" %len(layer_dict))
        if layer_dict:
            flag = 0
        else :
            flag = 1
        pass
    except:
        flag = 1
    if flag == 1:
        print("exec_layer failed")
        return 1

        class_dict = c_al.getALLPkgclassfication(dot_file,csv_file,False) 
        if class_dict:
            flag = 0
        else :
            flag = 1  
    try:
        pass
    except:
        flag = 1
    if flag == 1:
        print("exec_calssification failed")
        return 1
    '''
    """
    存储数据库
    在存储前需要新建数据库表
    """
    #存储分层分类信息
    class_dict = c_al.getALLPkgclassfication(dot_file,csv_file, False) 
    cl_dict = generate_LC_info(layer_dict=layer_dict, class_dict=class_dict)
    res = stor_sql.importPkgLayerClassInfo(cl_dict)
    if res:
        print("insert layer_class_info failed")
        return 1
    print("insert layer_class_info success")

    #存储依赖信息
    res = stor_sql.savePkgDepsInfo(all_dep_dict)
    if res:
        print("insert dependence failed")
        return 1
    print("insert dependence success")
    
    #存储csv信息
    print("start insert description")
    res = stor_sql.saveRpmDescription(final_dict_list)
    if res:
        print("insert description failed")
        return 1
    print("insert description success")
    return 0

def specificPkgLayerClass(pkg, layer, classification):
    new_sql = STORAGE_SQL()
    new_sql.updatePkgLayerClassification(pkg,layer,classification) 

def deletePkgLayerClass(pkg):
    new_sql = STORAGE_SQL()
    new_sql.deletePkgLayerClass(pkg)

def exportPkgClassificationToCSV(pkgname):
    csv_obj = CSV()
    csv_file = csv_obj.generate_fcfl_file_csv_file(pkgname)
    return csv_file

def exportAllPkgClassificationToCSV(filename):
    csv_obj = CSV()
    csv_file = csv_obj.generate_fcfl_files_csv_file(filename)
    return csv_file

def exportPkgDepsToDot(pkg):
    dot_obj = DOT()
    dot_file = dot_obj.generate_fcfl_pkg_dotFiles(pkg)
    return dot_file

def exportAllPkgDepsToDot():
    dot_obj = DOT()
    dot_file = dot_obj.generate_fcfl_pkgs_dotFiles()
    return dot_file

def exportAllPkgDescriptionsToCSV():
    csv_obj = CSV()
    csv_file = csv_obj.generate_fcfl_pkgs_rpm_description_csv_file()
    return csv_file
    
def exportPkgDescriptionsToCSV(pkgname):
    csv_obj = CSV()
    csv_file = csv_obj.generate_fcfl_pkg_rpm_description_csv_file(pkgname)
    return csv_file

def exportSpecificDomainLayer(special_domain):
    csv_obj = CSV()
    csv_file = csv_obj.generate_fcfl_domain_special_csv_file(special_domain)
    return csv_file

def exportAllDomainLayer():
    csv_obj = CSV()
    csv_file = csv_obj.generate_fcfl_domain_all_csv_file()
    return csv_file

'''
从repo 中导出csv和dot。
'''
def generate_csv_dot_all():
    """
    生成总的csv和dot文件
    """
    dot_obj = DOT()
    csv_obj = CSV()

    dot_path = dot_obj.generate_src_all_dotFiles()
    print("generate dot success")
    csv_path = csv_obj.generate_src_csv_files()
    print("generate csv success")
    return dot_path,csv_path

def generate_LC_info(layer_dict,class_dict):
    """
    整合分层分类字典
    """
    cl_dict = {}
    name_list = list(set(list(layer_dict.keys()) + list(class_dict.keys()) ) )
    print("==========in generate LC len=%d", len(name_list))

    for name in name_list:
        try:
            l_info = str(layer_dict[name])
        except:
            l_info = "Null"
        try:
            c_info = str(class_dict[name])
        except:
            c_info = "Null"
        cl_dict[name] = (l_info,c_info,0)
        '''
        try:
            func_info = str(func_df)
        except:
            cl_dict[name] = (l_info,c_info,0)
        '''
        print("==============len cl_dict============%d" %len(cl_dict))
    return cl_dict

def calcAllPkgsLayerClassification(dot_path, csv_path):
    """
    计算分层分类，存储数据库
    """
    dot_obj = DOT()
    csv_obj = CSV()
    layer_obj = LAYER()
    # class_obj = CLASSIFICATION()
    stor_sql = STORAGE_SQL()

    #读取dot、csv文件
    dep_dict = dot_obj.read_dot_file(dot_path)
    print("read dot success")
    csv_dict_list = csv_obj.read_csv_file(csv_path,'rpm_header')
    csv_dict_list2 = pd.read_csv(csv_path)
    print("read csv success")

    print("计算分层中............")
    tmp_json_file = data_config.tmp_json_file
    csv_and_dot_to_json(csv_file, dot_file, tmp_json_file)
    layer_dict = do_layer(tmp_json_file)
    if layer_dict:
        flag = 0
    else :
        flag = 1
    if flag == 1:
        print("exec_layer failed")
        return 1
  
    print("=========计算分类中............")
    c_al = class_algorithm()
    class_dict = c_al.getALLPkgclassfication(dot_file, csv_file, False, data_config.classer_model)
    if class_dict:
        flag = 0
    else :
        flag = 1
    if flag == 1:
        print("exec_calssification failed")
        return 1

    #存储临时文件
    field1 = 'rpm_name'
    field2 = 'classification'
    field3 = 'layer'

    csvlist = []
    tmpcsvkey = set(class_dict.keys())
    tmpcsvkey = tmpcsvkey &layer_dict.keys()
    for k in tmpcsvkey:
        d = dict()
        d[field1] = k
        d[field2] = class_dict[k]
        d[field3] = layer_dict[k]
        csvlist.append(d)
    with open(data_config.tmp_class_layer,'w') as f:
        writer = csv.DictWriter(f,['rpm_name', 'classification', 'layer'])
        writer.writeheader()
        writer.writerows(csvlist) 

    d_cls_dict={}
    d_lay_dict={}
    with open(data_config.tmp_class_layer, 'r') as f2:
        reader = csv.reader(f2)
        for row in reader:   
            if  "rpm_name" == row[0]:
                continue                 
            d_cls_dict[row[0]] = row[1]
            d_lay_dict[row[0]] = row[2]

    #存储数据库
    #在存储前需要新建数据库表
    stor_sql.create_table()
    #存储分层分类信息
    cl_dict = generate_LC_info(layer_dict=d_lay_dict, class_dict=d_cls_dict)
    #cl_dict = generate_LC_info(layer_dict=layer_dict, class_dict=class_dict)
    res = stor_sql.importPkgLayerClassInfo(cl_dict)
    if res:
        print("insert layer_class_info failed")
        return 1
    print("insert layer_class_info success")


    #计算功能分类


    #存储依赖信息
    res = stor_sql.savePkgDepsInfo(dep_dict)
    if res:
        print("insert dependence failed")
        return 1
    print("insert dependence success")
    
    #存储csv信息
    print("start insert description")
    res = stor_sql.saveRpmDescription(csv_dict_list)
    if res:
        print("insert description failed")
        return 1
    print("insert description success")
    return 0

def reOrgDBPkgslayerClassification():
    """
    读取依赖关系表和源码包信息表,重新计算分层分类，将分层结果存储数据库
    """
    stor_sql = STORAGE_SQL()
    layer_obj = LAYER()

    all_dep_dict= stor_sql.loadDepsFromDependeceTab()
    all_csv_dict = stor_sql.get_pkgs_rpm_desc_csv_dict_from_table()

    c_al = class_algorithm()
    #计算分层分类
    try:
        layer_dict = do_layer(data_config.all_test_dir)
        if layer_dict:
            flag = 0
        else :
            flag = 1  
    except:
        flag = 1
    if flag == 1:
        print("layer failed")
        return 1

    class_dict = c_al.getALLPkgclassfication(dot_file,csv_file,False) 
    if class_dict:
        flag = 0
    else :
        flag = 1

def test_func_class():
    # 计算功能分类
    func_dict = {'network':"network_best_model_bert.h5",
                 'format':"fmt_best_model_bert.h5",
                 'generic':"gene_best_model_bert.h5",
                 'graphic-management':"graphic_best_model_bert.h5",
                 'management':"management_best_model_bert.h5",
                 'other':"other_best_model_bert.h5",
                 'virt':'virt_best_model_bert.h5',
                 'storage':'storage_best_model_bert.h5',
                 'media': 'media_best_model_bert.h5',
                 'dev' : 'dev_best_model_bert.h5'
                 }
    try:
        for cls_key,h5file in func_dict.items():
            tst = c_fc.subCalssifer(cls_key)
            func_class_df = tst.calcClass(csv_file, h5file)

        if func_class_df:
            filename=csv_file+'.csv'
            func_class_df.to_csv(filename)
            flag = 0
        else :
            flag = 1
    except:
        flag = 1
    if flag == 1:
        print("class failed")
        return 1


    #存储分层分类信息
    cl_dict = generate_LC_info(layer_dict=layer_dict, class_dict=class_dict)
    res = stor_sql.importPkgLayerClassInfo(cl_dict)
    if res:
        print("insert layer_class_info failed")
        return 1


if __name__ == "__main__":

    dot_file = '/var/fcfl/data/915_rpm_all.dot'
    csv_file = '/var/fcfl/data/915_rpm_all.csv'
    calcAllPkgsLayerClassification(dot_file, csv_file)

