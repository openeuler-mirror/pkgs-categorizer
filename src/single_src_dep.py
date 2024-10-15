# SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import get_dep
from layer_sql import process_layer_sql
from read_sql import REPO_SQL
import queue


def get_src_dep(srcName,sql_path):
    """
    获取源码包依赖连（如果依赖的源码包已经存储在数据库中了，则不再寻找该源码包的依赖）
    输入参数：srcName 带有版本号的源码包名
            sql_path repo路径
    返回值：源码包依赖字典  key：源码包名   value：依赖列表
    """
    src_dict = {}

    sdb_obj = process_layer_sql.STORAGE_SQL()
    sql_obj = REPO_SQL()
    src_queue = queue.Queue()
    serach_list = []

    src_queue.put(srcName)                      #队列添加源码包
    srcName_list = sdb_obj.readDepsrcName() #读取数据库中的源码包列表

    while src_queue.qsize():     #队列不为空
        temp_list = []
        src_name = src_queue.get()  #取队列中的一个源码包名
        src_dep_list = sql_obj.get_srcPackage_directly_dep(src_name,sql_path) #获取源码包直接依赖  带有版本号

        if len(src_dep_list) == 0:
            src_name = get_dep.srpmName_analyze(src_name)
            src_dict[src_name] = temp_list
            continue

        for src_dep_name in src_dep_list:   #遍历依赖
            dep_name = get_dep.srpmName_analyze(src_dep_name) #去除依赖版本号
            temp_list.append(dep_name)  #添加列表 不带版本号

            if dep_name in srcName_list or \
                dep_name in serach_list:        #依赖包不在数据库和查询列表中
                continue

            src_queue.put(src_dep_name)     #依赖包添加到队列中
            name = get_dep.srpmName_analyze(src_dep_name)
            serach_list.append(name)    #依赖包添加到查询列表中

        src_name = get_dep.srpmName_analyze(src_name)
        src_dict[src_name] = temp_list
    return src_dict

