# SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
#                              Peking University
# SPDX-License-Identifier: Apache-2.0

#-*-coding:utf-8-*-


from set import package
from set import node
from set import node_set
from set import load_data

#from asyncio.windows_events import NULL
from re import A, L
from mpl_toolkits.mplot3d import Axes3D
import collections
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import functools
import os
import math
import random
import json

from scipy.integrate import quad
import math as m
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# 分层层数
LAYER_NUMBER = 4


# 使用：package2layer_expert = load_expert_origin('./input/expert.txt')
# 作用：导入专家分层结果
# 返回值：dict    key为包名，str    value为专家分层结果，int
# './input/expert.txt'为去重、手动调整过冲突的结果
# './input/expert.txt'是原始的专家结果
def load_expert_origin(filename):
    with open(filename, 'r') as f:
        raw_data = f.read()
        lines = raw_data.split('\n')
        lines = [line.split('\t') for line in lines]
        package2layer_expert = {}
        for line in lines:
            if line[0] in package2layer_expert:
                if int(line[1]) != package2layer_expert[line[0]] and line[1] != '-1':
                    print('该包被分在不同的层', line[0], line[1], package2layer_expert[line[0]])
            else:
                package2layer_expert[line[0]] = int(line[1])
                if line[1] == '-1':
                    print('该包没有专家结果: ', line[0])
        #package2layer_expert = {line[0]:int(line[1]) for line in lines}
        return package2layer_expert


# 作用：保存上一个专家分层结果到文件中，以后使用直接读取
def save_expert(package2layer_expert, filename):
    with open(filename, 'w') as f:
        lines = [f'{key}\t{package2layer_expert[key]}\n' for key in package2layer_expert]
        f.writelines(lines)
        f.close()


# 作用：检测专家结果的分层在[1, LAYER_NUMBER]中
def check_package2layer_expert(package2layer_expert, debug=False):
    debug_info = []
    for key in package2layer_expert:
        if package2layer_expert[key] < 1 or package2layer_expert[key] > LAYER_NUMBER:
            if debug:
                debug_info.append(key)
                continue
            return False
    if debug and len(debug_info) != 0:
        print('[debug]检测专家结果的分层在[1, LAYER_NUMBER]中: ', debug_info)
    return len(debug_info) == 0


# 作用：检测node_set中的所有点都是单独的一个点，而且都有专家结果
def check_node(node_set, package2layer_expert, debug=False):
    debug_info = []
    for node in node_set.nodes:
        if len(node.packages) != 1 or node.packages[0] not in package2layer_expert:
            if debug:
                debug_info.append([node.packages, node.packages[0] in package2layer_expert])
                continue
            return False
    if debug and len(debug_info) != 0:
        print('[debug]检测node_set中的所有点都是单独的一个点,而且都有专家结果: ', debug_info)
    return len(debug_info) == 0


# 作用：统计list中的层次分布
def print_layer_static(nodes, package2layer_expert):
    static_info = f'节点分布:\n'
    static = collections.defaultdict(int)
    for node in nodes:
        static[package2layer_expert[node.packages[0]]] += 1
        # node.packages 是否是节点如果一些包合并了，取第一个作为这个合并的层级?

    for i in range(1, LAYER_NUMBER+1):
        if i in static:
            static_info += f'第{i}层的节点数量为: {static[i]}, 占{static[i] * 100/len(nodes)}% \n'
    print(static_info)

# 作用：统计node_set的点中,每一层有多少节点、有哪些节点有对上层的依赖、有哪些节点有跨层依赖...
def print_static(node_set, package2layer_expert):
    static_info = f'节点分布:\n'
    static = collections.defaultdict(int)
    for node in node_set.nodes:
        static[package2layer_expert[node.packages[0]]] += 1

    for i in range(1, LAYER_NUMBER+1):
        if i in static:
            static_info += f'第{i}层的节点数量为: {static[i]}, 占{static[i] * 100/len(node_set.nodes)}% \n'
    print(static_info)



    # depend_on_upper_info = f'对上层依赖:\n'
    # for node in node_set.nodes:
    #     for node_out in node.out_node:
    #         if package2layer_expert[node.packages[0]] < package2layer_expert[node_out.packages[0]]:
    #             depend_on_upper_info += f'{node.packages[0]} (第{package2layer_expert[node.packages[0]]}层)依赖{node_out.packages[0]}(第{package2layer_expert[node_out.packages[0]]}层)\n'

    # depend_on_lower_info = f'跨层依赖:\n'       
    # for node in node_set.nodes:
    #     for node_out in node.out_node:
    #         if package2layer_expert[node.packages[0]] > package2layer_expert[node_out.packages[0]]+1:
    #             depend_on_lower_info += f'{node.packages[0]} (第{package2layer_expert[node.packages[0]]}层)依赖{node_out.packages[0]}(第{package2layer_expert[node_out.packages[0]]}层)\n'

    '''
    print('打印第四层的相关依赖关系')
    for node in node_set.nodes:
        if package2layer_expert[node.packages[0]] == 4 :
            print(f'{node.packages[0]}位于第四层')
            print('该包依赖：')
            for node_out in node.out_node:
                print(f'{node_out.packages[0]}(第{package2layer_expert[node_out.packages[0]]}层)')
            print('该包被依赖：')
            for node_in in node.in_node:
                print(f'{node_in.packages[0]}(第{package2layer_expert[node_in.packages[0]]}层)')
            print('\n')
    '''
    #print(depend_on_upper_info)

    #print(depend_on_lower_info)

    depended_static_info = f'节点被依赖关系:\n'
    for node in node_set.nodes:
        depended_static_info += f'{node.packages[0]} 被依赖占比: '
        if len(node.in_node) != 0:
            single_node_static = [0 for i in range(0, LAYER_NUMBER)]
            for node_in in node.in_node:
                single_node_static[package2layer_expert[node_in.packages[0]] - 1] += 1
            for i in range(0, LAYER_NUMBER):
                depended_static_info += f'第{i+1}层有{single_node_static[i]}({single_node_static[i]*100/len(node.in_node)}%) |'
        depended_static_info += f'\n'
    #print(depended_static_info)

    # 统计各层对不同层级的依赖
    # layers_dependency = [collections.defaultdict(int), collections.defaultdict(int), collections.defaultdict(int), collections.defaultdict(int)]

    # for n in node_set.nodes:
    #     for out_n in n.out_node:
    #         layers_dependency[package2layer_expert[n.packages[0]] - 1][package2layer_expert[out_n.packages[0]]] += 1
    # print(f'软件层级 | 对第1层平均出度 | 对第2层平均出度 | 对第3层平均出度 | 对第4层平均出度 ')
    # for i in range(1, 5):
    #     print(f'第{i}层 | {layers_dependency[i-1][1]/static[i]} | {layers_dependency[i-1][2]/static[i]} | {layers_dependency[i-1][3]/static[i]} | {layers_dependency[i-1][4]/static[i]} \n')




# 生成每个节点的被不同层依赖的占比
def generate_depended_percent(node_set, package2layer_expert):
    depended_percent = {}
    for node in node_set.nodes:
        single_node_static = [0 for i in range(0, LAYER_NUMBER)]
        if len(node.in_node) != 0:
            for node_in in node.in_node:
                single_node_static[package2layer_expert[node_in.packages[0]] - 1] += 1
            for i in range(0, LAYER_NUMBER):
                single_node_static[i] /= len(node.in_node) #* len(node.in_node)  # 除平方是为了让被依赖越少的点，他的占比效果应该越好，可以考虑依赖这个点的分布和所有点的分布之间的差异来作为权重，即差异越大，权重越大
        depended_percent[node.packages[0]] = single_node_static
    return depended_percent


# 生成每个节点被依赖占比的权重
def generate_depended_percent_power(node_set, package2layer_expert):
    depended_percent_power = {}

    total_static = [0 for i in range(0, LAYER_NUMBER)]
    for node in node_set.nodes:
        total_static[package2layer_expert[node.packages[0]]-1] += 1
    for i in range(0, LAYER_NUMBER):
        total_static[i] /= len(node_set.nodes)

    for node in node_set.nodes:
        single_node_static = [0 for i in range(0, LAYER_NUMBER)]
        if len(node.in_node) != 0:
            for node_in in node.in_node:
                single_node_static[package2layer_expert[node_in.packages[0]] - 1] += 1
            for i in range(0, LAYER_NUMBER):
                single_node_static[i] /= len(node.in_node)
        depended_percent_power[node.packages[0]] = calculate_vector_distance(single_node_static, total_static)

    return depended_percent_power


# 计算两个向量之间的距离作为权重
def calculate_vector_distance(vector1, vector2):
    diff = 0
    for i in range(0, len(vector1)):
        diff += (vector1[i] - vector2[i]) * (vector1[i] - vector2[i])
    diff = math.sqrt(diff)
    return diff




# 对于每个节点，根据依赖的节点的被依赖占比，直接累加，来分层
def layer_by_depended_percent(node_set, depended_power, depended_percent_power):
    res = {}
    no_out_node_list = []
    for node in node_set.nodes:
        probability = [0 for i in range(0, LAYER_NUMBER)]
        if len(node.out_node) == 0:
            no_out_node_list.append(node.packages[0])
        else:
            for node_out in node.out_node:
                for i in range(0, LAYER_NUMBER):
                    probability[i] += depended_power[node_out.packages[0]][i] * depended_percent_power[node_out.packages[0]]
            single_res = 0
            for i in range(0, LAYER_NUMBER):
                if probability[i] > probability[single_res]:
                    single_res = i
            res[node.packages[0]] = single_res + 1

    print(f'不依赖任何包的包：', no_out_node_list)
    return res


# 对比分层结果
def compare_layer_res(layer_res, expert_res):
    true_num = 0
    false_num = 0
    total_num = 0
    diff_package = []
    compare_info = {}
    no_depend_packages = []
    for package in layer_res:
        if package not in expert_res:
            continue
        if layer_res[package] == -1:
            no_depend_packages.append(package)
            #print('该包没有依赖:', package)
            #continue
        if layer_res[package] == expert_res[package]:
            true_num += 1
            total_num += 1
        else:
            false_num += 1
            total_num += 1
            diff_package.append(package)

    print('没有依赖的包:', package)
    print(f'正确率:{true_num * 100/total_num}%, 正确数量:{true_num}, 错误数量{false_num}, 总数:{total_num}')
    # print('错误的节点：', diff_package)
    compare_info['diff_package'] = diff_package
    compare_info['no_depend_packages'] = no_depend_packages
    compare_info['true_num'] = true_num
    compare_info['false_num'] = false_num
    compare_info['total_num'] = total_num

    return compare_info


# 获取正确率
def get_layer_res_accuracy(layer_res, expert_res):
    true_num = 0
    false_num = 0
    total_num = 0
    diff_package = []
    no_depend_packages = []
    for package in layer_res:
        if layer_res[package] == -1:
            no_depend_packages.append(package)
            # print('该包没有依赖:', package)
            # continue
        if layer_res[package] == expert_res[package]:
            true_num += 1
            total_num += 1
        else:
            false_num += 1
            total_num += 1
            diff_package.append(package)
        
    print('没有依赖的包:', package)
    print(f'正确率:{true_num * 100/total_num}%, 正确数量:{true_num}, 错误数量{false_num}, 总数:{total_num}')
    #print('错误的节点：', diff_package)
    return true_num / total_num

# 错误节点的相关数据
def diff_node_status(diff_package, node_set, depended_power, expert_res):
    diff_info = f'-------------------------------\n'
    for package_name in diff_package:
        node = node_set.name2node[package_name]
        probability = [0 for i in range(0, LAYER_NUMBER)]
        if len(node.out_node) != 0:
            for node_out in node.out_node:
                for i in range(0, LAYER_NUMBER):
                    probability[i] += depended_power[node_out.packages[0]][i]
            single_res = 0
            for i in range(0, LAYER_NUMBER):
                if probability[i] > probability[single_res]:
                    single_res = i
        diff_info += node.self_print() + f'{probability}\n 专家结果:{expert_res[package_name]}\n-------------------------------\n'
    print(diff_info)



# 作用：将node_set中的点集合根据专家结果输出一个包名到不同层的依赖向量dict, 
# 依赖向量：比如[4,3,2,1]表示该包对1层的4个包有依赖,对2层的3个包有依赖,对3层的2个包有依赖,对4层的1个包有依赖
def node_set2dependency_vector(node_set, package2layer_expert):
    # 先检测点
    dependency_vector = {}
    if not check_node(node_set, package2layer_expert, True):
        return dependency_vector

    for node in node_set.nodes:
        dependency_vector[node.packages[0]] = [0 for i in range(0, LAYER_NUMBER)]
        for out_node in node.out_node:
            dependency_vector[node.packages[0]][package2layer_expert[out_node.packages[0]]-1] += 1

    return dependency_vector

# 作用:使依赖向量归一化
def dependency_vector_normalize(dependency_vector):
    max_list = [1 for i in range(0, LAYER_NUMBER)]
    for package in dependency_vector:
        for i in range(0, LAYER_NUMBER):
            if dependency_vector[package][i] > max_list[i]:
                max_list[i] = dependency_vector[package][i]
    for package in dependency_vector:
        for i in range(0, LAYER_NUMBER):
            dependency_vector[package][i] /= max_list[i]


# 作用：使依赖向量正则化
def dependency_vector_regularize(dependency_vector):
    for package in dependency_vector:
        total = 0
        for i in range(0, LAYER_NUMBER):
            total += dependency_vector[package][i] * dependency_vector[package][i]
        if total == 0:
            total = 1
        total = math.sqrt(total)
        for i in range(0, LAYER_NUMBER):
            dependency_vector[package][i] /= total

# 作用：随机选取 x %的节点，作为已知节点
def get_init_nodes_random(node_set, x):
    init_nodes = []
    if x <= 0 or x > 1 :
        return init_nodes
    total_nodes = node_set.nodes.copy()
    times = int(len(total_nodes) * x)
    for time in range(0, times):
        index = random.randint(0, len(total_nodes)-1)
        init_nodes.append(total_nodes[index])
        del total_nodes[index]
    return init_nodes

# 作用：随机选取 x %的节点，加上一个固定的列表，作为已知节点
def get_init_nodes_random_with_list(node_set, x, package_list):
    init_nodes = []
    if x <= 0 or x > 1 :
        init_nodes += get_init_nodes_from_list(node_set, package_list)
        return init_nodes
    total_nodes = node_set.nodes.copy()
    times = int(len(total_nodes) * x)
    for time in range(0, times):
        index = -1
        while index == -1 or total_nodes[index].packages[0] in package_list:
            index = random.randint(0, len(total_nodes)-1)
        init_nodes.append(total_nodes[index])
        del total_nodes[index]
    
    init_nodes += get_init_nodes_from_list(node_set, package_list)
    return init_nodes

# 作用：获取入度为0的节点加入到一个列表里
def get_package_list_no_indegree(node_set):
    package_list = []
    for node in node_set.nodes:
        if len(node.in_node) == 0:
            package_list.append(node.packages[0])
    return package_list

def get_init_nodes_ppt_case(node_set):
    init_nodes = []
    total_nodes = node_set.nodes.copy()
    for n in total_nodes:
        if n.packages[0] != 'p12':
            init_nodes.append(n)
    return init_nodes

# 作用：选取在三个包中重复的节点，作为已知节点
def get_init_nodes_repeat_in_three(node_set):
    ps = []
    with open('./input/repeat3.txt', 'r') as f:
        raw_data = f.read()
        ps += raw_data.split('\n')
    total_nodes = node_set.nodes.copy()
    init_nodes = []
    for n in total_nodes:
        if n.packages[0] in ps:
            init_nodes.append(n)
    return init_nodes

# 作用：选取在两个个包中重复的节点，作为已知节点
def get_init_nodes_repeat_in_two(node_set):
    ps = []
    with open('./input/repeat3.txt', 'r') as f:
        raw_data = f.read()
        ps += raw_data.split('\n')
    with open('./input/repeat2.txt', 'r') as f:
        raw_data = f.read()
        ps += raw_data.split('\n')
    total_nodes = node_set.nodes.copy()
    init_nodes = []
    for n in total_nodes:
        if n.packages[0] in ps:
            init_nodes.append(n)
    return init_nodes


# 作用：获取指定包名的节点作为已知节点
def get_init_nodes_from_list(node_set, packages):
    init_nodes = []

    for name in packages:
        init_nodes.append(node_set.name2node[name])
    return init_nodes

# 作用：获取指定包名有效的节点作为已知节点
def get_init_nodes_from_list_valid(node_set, package2layer_expert, packages):
    #packages是 src_name 列表, bert训练集合作已知节点 , package2layer_expert 是人工标记 , node_set 是dot 文件中的节点。
    init_nodes = []

    for name in packages:
        if name in package2layer_expert and name in node_set.name2node:
            init_nodes.append(node_set.name2node[name])
    return init_nodes
# 返回nodes列表

# 作用：读取排序后的包信息
def get_package_score():
    package_info = {}
    with open('./20220715/init_test_result.txt', 'r') as f:
        data = f.read()
        lines = data.split('\n')
        for line in lines:
            if line == '':
                continue
            pair = line.split(' : ')
            packages = pair[0].split(',')
            for package in packages:
                if package not in package_info:
                    package_info[package] = {
                        'score': 0,
                        'count': 0,
                    }
                package_info[package]['score'] += float(pair[1])
                package_info[package]['count'] += 1

    package_score = []
    for package in package_info:
        package_score.append([package , package_info[package]['score'] / package_info[package]['count']])

    package_score.sort(reverse=True,key=lambda x: x[1])
    return [p[0] for p in package_score]

#作用：读取bert的train_set包信息
def get_bert_train_set():
    package_list = []
    with open('./input/1228_bin_train_set.json', 'r') as f:
        features = json.load(f)
        for feature in features:
            package_list.append(feature['src_name'])
    return package_list


#作用：获取前n个包的name
pre_n = 0
def get_pre_n_packages(package_score, n):
    return package_score[:n]


#作用：打印n个node的出入度信息
def print_degree_infos(nodes):
    for node in nodes:
        print(f'包名: {node.packages[0]}, 度：{len(node.out_node) + len(node.in_node)}, 出度：{len(node.out_node)}, 入度：{len(node.in_node)}')


#作用：获取根据度排序后的前n个包作为已知节点
def get_sorted_packages_by_degree(node_set, n):
    init_nodes = []
    total_nodes = node_set.nodes.copy()
    total_nodes.sort(reverse=True, key=lambda node:(len(node.out_node) + len(node.in_node)))
    init_nodes = total_nodes[:n]
    return init_nodes

# 作用：根据点集和已知节点获取未知节点
def get_rest_nodes(node_set, init_nodes):
    rest_nodes = []
    for n in node_set.nodes:
        if n not in init_nodes:
            rest_nodes.append(n)
    return rest_nodes

# 作用：将初始点集分层
def layer_init_nodes(init_nodes, expert_res):
    layer_res = {}
    for n in init_nodes:
        layer_res[n.packages[0]] = expert_res[n.packages[0]] #手工标注的层级
    return layer_res

# 作用: 将一个点分层
def layer_single_node(layered_nodes, layer_res, single_node):
    # 计算当前已分类点的不同层比例
    static_layer_partition = [0 for i in range(0, LAYER_NUMBER)]
    total = 0
    for n in layered_nodes:
        if layer_res[n.packages[0]] != -1:
            static_layer_partition[layer_res[n.packages[0]]-1] += 1
            total += 1
    for i in range(0, LAYER_NUMBER):
        static_layer_partition[i] /= total

    # print(f'【已知节点分布向量】为{static_layer_partition}')

    # 计算single_node的out_node的【被依赖向量】和权重
    depended_percent = {}
    depended_percent_power = {}
    for node in single_node.out_node:
        single_node_static = [0 for i in range(0, LAYER_NUMBER)]
        total_count = 0
        if len(node.in_node) != 0:
            for node_in in node.in_node:
                if node_in.packages[0] in layer_res:
                    single_node_static[layer_res[node_in.packages[0]] - 1] += 1
                    total_count += 1
            if total_count != 0:
                for i in range(0, LAYER_NUMBER):
                    # 不使用total_count，因为有多少占比是已知的这个信息应该也影响到这个的权重
                    single_node_static[i] /= total_count #len(node.in_node)
        depended_percent[node.packages[0]] = single_node_static
        depended_percent_power[node.packages[0]] = calculate_vector_distance(single_node_static, static_layer_partition)

    # 根据【被依赖向量】计算每个依赖节点的【被依赖分布】加权和（加权在上面）
    probability = [0 for i in range(0, LAYER_NUMBER)]
    if len(single_node.out_node) == 0:
        layer_res[single_node.packages[0]] = -1
        layered_nodes.append(single_node)
    else:
        for node_out in single_node.out_node:
            # print(f'{node_out.packages[0]} 的【被依赖分布向量】为{depended_percent[node_out.packages[0]]}， 其权重为{depended_percent_power[node_out.packages[0]]}')
            for i in range(0, LAYER_NUMBER):
                probability[i] += depended_percent[node_out.packages[0]][i] * depended_percent_power[node_out.packages[0]]
        single_res = 0
        # print(f'{single_node.packages[0]}的各层权重为{probability}')
        for i in range(0, LAYER_NUMBER):
            if probability[i] > probability[single_res]:
                single_res = i
        layer_res[single_node.packages[0]] = single_res + 1
        layered_nodes.append(single_node)
    return [single_node.packages[0], probability]


# 作用：迭代分层
def layer_all_node(layered_nodes, layer_res):
    # 计算当前已分类点的不同层比例
    static_layer_partition = [0 for i in range(0, LAYER_NUMBER)]
    total = 0
    for n in layered_nodes:
        if layer_res[n.packages[0]] != -1:
            static_layer_partition[layer_res[n.packages[0]]-1] += 1
            total += 1
    for i in range(0, LAYER_NUMBER):
        static_layer_partition[i] /= total

    # print(f'【已知节点分布向量】为{static_layer_partition}')

    next_layer_res = {}
    for single_node in layered_nodes:

        # 计算single_node的out_node的【被依赖向量】和权重
        depended_percent = {}
        depended_percent_power = {}
        for node in single_node.out_node:
            single_node_static = [0 for i in range(0, LAYER_NUMBER)]
            total_count = 0
            if len(node.in_node) != 0:
                for node_in in node.in_node:
                    if node_in.packages[0] in layer_res:
                        single_node_static[layer_res[node_in.packages[0]] - 1] += 1
                        total_count += 1
                if total_count != 0:
                    for i in range(0, LAYER_NUMBER):
                        # 不使用total_count，因为有多少占比是已知的这个信息应该也影响到这个的权重
                        single_node_static[i] /= len(node.in_node)
            depended_percent[node.packages[0]] = single_node_static
            depended_percent_power[node.packages[0]] = calculate_vector_distance(single_node_static, static_layer_partition)

        # 根据【被依赖向量】计算每个依赖节点的【被依赖分布】加权和（加权在上面）
        probability = [0 for i in range(0, LAYER_NUMBER)]
        if len(single_node.out_node) == 0:
            next_layer_res[single_node.packages[0]] = -1
        else:
            for node_out in single_node.out_node:
                #print(f'{node_out.packages[0]} 的【被依赖分布向量】为{depended_percent[node_out.packages[0]]}， 其权重为{depended_percent_power[node_out.packages[0]]}')
                for i in range(0, LAYER_NUMBER):
                    probability[i] += depended_percent[node_out.packages[0]][i] * depended_percent_power[node_out.packages[0]]
            single_res = 0
            #print(f'{single_node.packages[0]}的各层权重为{probability}')
            for i in range(0, LAYER_NUMBER):
                if probability[i] > probability[single_res]:
                    single_res = i
            next_layer_res[single_node.packages[0]] = single_res + 1
    print(len(next_layer_res))

    is_changed = False
    changed_list = []
    for package in next_layer_res:
        if next_layer_res[package] != layer_res[package]:
            layer_res[package] = next_layer_res[package]
            is_changed = True
            changed_list.append({package: f' {layer_res[package]} -> {next_layer_res[package]}'})

    print(f'is_changed = {is_changed}, changed_list = {changed_list}')
    return is_changed



# 作用：按一定的规则，将每个未分类节点分类
def layer_rest_nodes(init_nodes, rest_nodes, layer_res):
    while len(rest_nodes) != 0:

        waited_degree = {}
        for n in rest_nodes:
            if len(n.out_node) == 0:
                waited_degree[n] = 0
                continue
            count = 0
            for out in n.out_node:
                if out.packages[0] in layer_res:
                    count += 1
            waited_degree[n] = count/len(n.out_node)

        rest_nodes.sort(key=lambda x: waited_degree[x])
        # print(waited_degree[rest_nodes[-1]])

        if isValid(init_nodes, layer_res):
            layer_single_node(init_nodes, layer_res, rest_nodes.pop())
        else:
            return


# 作用：按一定的规则，将每个未分类节点分类，返回分类的向量
def layer_rest_node_with_vector(init_nodes, rest_nodes, layer_res):
    packages_vector = {}

    while len(rest_nodes) != 0:
        waited_degree = {}
        for n in rest_nodes:
            if len(n.out_node) == 0:
                waited_degree[n] = 0
                continue
            count = 0
            for out in n.out_node:
                if out.packages[0] in layer_res:
                    count += 1
            waited_degree[n] = count/len(n.out_node)

        rest_nodes.sort(key=lambda x: waited_degree[x])
        # print(waited_degree[rest_nodes[-1]])

        if isValid(init_nodes, layer_res):
            package_vector = layer_single_node(init_nodes, layer_res, rest_nodes.pop())
        else:
            return packages_vector

        packages_vector[package_vector[0]] = package_vector[1]

    return packages_vector


# 作用，判断dict的key是不是都在list中
def isValid(init_nodes, layer_res):
    for n in init_nodes:
        if n.packages[0] not in layer_res:
            print('invalid:',n.packages[0])
            return False
    return True

# 作用， 补全package_vector
def padding_package_vector(package_vector):
    with open('./input/1228_bin_train_set.json', 'r') as f:
        features = json.load(f)
        for feature in features:
            if feature['src_name'] not in package_vector:
                package_vector[feature['src_name']] = [0, 0, 0, 0]
    with open('./input/1228_bin_test_set.json', 'r') as f:
        features = json.load(f)
        for feature in features:
            if feature['src_name'] not in package_vector:
                package_vector[feature['src_name']] = [0, 0, 0, 0]

# 作用， 补全new_package_vector
def padding_new_package_vector(package_vector):
    with open('./input/1228_bin_new_package_set.json', 'r') as f:
        features = json.load(f)
        for feature in features:
            if feature['src_name'] not in package_vector:
                package_vector[feature['src_name']] = [0, 0, 0, 0]

# 作用：导入点的依赖向量，画图
# 点的形状list长度必须大于等于LAYER_NUMBER markers = ['o', 'd', 's', 'h', 'p']
# 目前因为维度限制，LAYER_NUMBER只能等于4
def draw(dependency_vector, package2layer_expert):
    markers = ['o', 'd', 's', 'h', 'p']
    if LAYER_NUMBER > len(markers):
        print('点的形状数量少于分层结果')
        return

    fig = plt.figure()  #创建一个图
    ax = fig.add_subplot(111, projection='3d')
    cm = plt.cm.get_cmap('jet')  #颜色映射，为jet型映射规则

    for i in range(2, 3):
        ls_x = []  
        ls_y = []
        ls_z = []
        ls_C0 = []
        for package in dependency_vector:
            if package2layer_expert[package] == i+1:
                ls_x.append(dependency_vector[package][0])
                ls_y.append(dependency_vector[package][1])
                ls_z.append(dependency_vector[package][2])
                #ls_C0.append(dependency_vector[package][3])
                ls_C0.append(i+1)

        X, Y, Z = np.array(ls_x), np.array(ls_y), np.array(ls_z)  #给X,Y,Z赋值ndarray数组, 分别为C0*, w, y
        C0 = np.array(ls_C0)  #给C0赋值ndarray数组

        fig = ax.scatter3D(X, Y, Z, c = C0, cmap=cm, marker = markers[i], s = 40)

    cb = plt.colorbar(fig)  #设置坐标轴
    ax.set_xlabel('1 layer')
    ax.set_ylabel('2 layer')
    ax.set_zlabel('3 layer')
    cb.ax.tick_params(labelsize=12)
    cb.set_label('4 layer', size = 16)

    plt.show()


# 原始画图的参考函数
def draw_origin():
    D_0 = 5 * m.pow(10, -5)  #常数
    v = 1.01 * m.pow(10, -6)

    ls_C1 = []  #C0*列表
    ls_C0 = []
    ls_w  = []
    ls_y  = []
    marker = []
    now = True

    for C_1 in np.arange(0.0,100.1, 10):
        for w in range(1, 10001, 1000):
            for y in np.arange(m.pow(10,-5), m.pow(10,-3), 10 * m.pow(10,-5)):
                
                C_1 = float(C_1)
                y = float(y)

                Y = y/(m.pow(3, 1/3) * m.pow(D_0, 1/3) * m.pow(w, -1/2) * m.pow(v, 1/6))  #计算Y
                C_0 = (C_1/0.8934) * quad(lambda u:m.exp(-m.pow(u, 3)),0, Y)[0]  #计算C0
                
                ls_C1.append(C_1)
                ls_w.append(w)
                ls_y.append(y)
                ls_C0.append(C_0)
                if now :
                    marker.append('d')
                else:
                    marker.append('o')
                now = not now

    fig = plt.figure()  #创建一个图
    ax = fig.add_subplot(111, projection='3d')

    X, Y, Z = np.array(ls_C1), np.array(ls_w), np.array(ls_y)  #给X,Y,Z赋值ndarray数组, 分别为C0*, w, y
    C0 = np.array(ls_C0)  #给C0赋值ndarray数组

    cm = plt.cm.get_cmap('jet')  #颜色映射，为jet型映射规则
    # marker是形状，已知的有o，d，s，h，p
    # s是大小
    #fig = ax.scatter3D(X,Y,Z, c = C0, cmap=cm, marker = 'o')
    fig = ax.scatter3D(X[:500],Y[:500],Z[:500], c = C0[:500], cmap=cm, marker = 's', s = 40)
    fig = ax.scatter3D(X[500:],Y[500:],Z[500:], c = C0[500:], cmap=cm, marker = 'h', s = 40)

    cb = plt.colorbar(fig)  #设置坐标轴
    ax.set_xlabel('C0*')
    ax.set_ylabel('w')
    ax.set_zlabel('y')
    cb.ax.tick_params(labelsize=12)
    cb.set_label('C0', size = 16)

    plt.show()



init_nodes_res = set()
def save_init_nodes_res():
    f = open('./20220715/init_test_result.txt', 'w')
    for line in init_nodes_res:
        f.write(f'{line}\n')

write_infos = []
def save_infos(filename):
    f = open(filename, 'w')
    for line in write_infos:
        f.write(f'{line}\n')




def pivot():
    print(os.getcwd())
    #1228_bin_single_label.txt记录的是各个包的层级
    package2layer_expert = load_expert_origin('./input/1228/1228_bin_single_label.txt')
    if not check_package2layer_expert(package2layer_expert, True):
        return

    print('从文件中读入依赖关系对list')
    dependency_relationship = load_data('./input/1228/rpm_all.dot')
    s = node_set()
    print('将依赖关系对list添加到点集中')
    s.add_pairs(dependency_relationship)


    print('使用bert训练集作为已知节点')
    packages_list = get_bert_train_set()
    #packages_list 是 src_name 列表, package2layer_expert 是人工标记 , s 是dot 文件中的节点。
    init_nodes = get_init_nodes_from_list_valid(s, package2layer_expert, packages_list)

    print('初始已知节点数量:',len(init_nodes))

    print('统计初始节点的层次分布')
    print_layer_static(init_nodes, package2layer_expert)

    print('获取未知节点')
    rest_nodes = get_rest_nodes(s, init_nodes) # 返回一个其余包的列表

    print('初始点集分层')
    layer_res = layer_init_nodes(init_nodes, package2layer_expert)

    print('分层')
    package_vector = layer_rest_node_with_vector(init_nodes, rest_nodes, layer_res)
    padding_package_vector(package_vector)
    
    print('保存加权和向量')
    with open(f'./package_vector.json', 'w') as f:
        json.dump(package_vector, f)

    print('统计最终节点的层次分布')
    print_layer_static(init_nodes, layer_res)

    print('对比分层结果')
    compare_info = compare_layer_res(layer_res, package2layer_expert)
    return compare_info

    #init_nodes_res.add(f'{",".join(init_nodes_copy)} : {get_layer_res_accuracy(layer_res, package2layer_expert)}')

if __name__ == "__main__":
    pivot()
