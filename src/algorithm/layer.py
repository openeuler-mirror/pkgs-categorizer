# SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
#                              Peking University
# SPDX-License-Identifier: Apache-2.0

import pprint
import algorithm.dfs_and_topo as dt
def get_NodeDep_from_str(s):
    """
    处理读取的字符串，返回节点和其依赖
    """
    s = s.replace('"','').replace('\n','')
    index = s.split("->")
    node = index[0].strip()
    dep = index[1].strip()

    return node, dep

def read_dot_file(dot_path):
    """
    读取dot文件，返回依赖字典
    """
    dep_dict = {}
    with open(dot_path,'r') as f:
        line = f.readline()
        while line:
            if "->" in line:
                pkg,dep = get_NodeDep_from_str(line)
                if pkg in dep_dict.keys():
                    dep_dict[pkg].add(dep)
                else:
                    tmp = set()
                    tmp.add(item[1])
                    dep_dict[pkg] = tmp
            line = f.readline()
    return dep_dict


dot_path = "dot_files/app.dot"
#dot_path = "test.dot"
nodeSet = dt.NodeSet()

def nodeSet_Init():
    """
    节点集合初始化
    """
    dep_dict = read_dot_file(dot_path)

    for key in dep_dict.keys():
        for dep in dep_dict[key]:
            nodeSet.setInit(key,dep)

    #print(nodeSet.packages)
    #print(nodeSet.nodeSet["A"].packages)
    pprint.pprint(nodeSet.nodeSet)

def originAlgorithm():
    """
    原始分层算法
    """
    firstLayer = []
    secondLyer = []
    thriedLayer = []
    fourLayer = []
    nodeSet_Init()         #节点集合的初始化
    nodeSet.dfs()           #dfs算法找环
    topoList = nodeSet.generateTopoList()   #生成拓扑序列
    #print("topList",topoList)
    layer = nodeSet.generateLayer(topoList) #分层
    #print(layer)
    nodeSet.get_packages_layer(layer)       #计算包属于哪层

    for key in nodeSet.packageLayer.keys():
        if nodeSet.packageLayer[key] == 1:
            if len(nodeSet.nodeSet[key].outNode) == 0:
                secondLyer.append(key)
            else:
                firstLayer.append(key)
        elif nodeSet.packageLayer[key] == 2:
            if len(nodeSet.nodeSet[key].inNode) == 0:
                secondLyer.append(key)
            else:
                fourLayer.append(key)
        else :
            thriedLayer.append(key)
    pprint.pprint(nodeSet.packageLayer)
    print("firstLayer len = ",len(firstLayer))
    print("firstLayer=",firstLayer)
    print("secondLyer len = ",len(secondLyer))
    print("secondLyer=",secondLyer)
    print("thriedLayer len = ",len(thriedLayer))
    print("thriedLayer=",thriedLayer)
    print("fourLayer len = ",len(fourLayer))
    print("fourLayer=",fourLayer)
    

originAlgorithm()




