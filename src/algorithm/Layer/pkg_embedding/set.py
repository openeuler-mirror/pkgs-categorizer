# SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
#                              Peking University
# SPDX-License-Identifier: Apache-2.0

#-*-coding:utf-8-*-

#from asyncio.windows_events import NULL
import collections
from re import A
import networkx as nx
import matplotlib.pyplot as plt
import functools
import os

from numpy import number

# 软件包，目前没有用到
class package:
    def __init__(self, name):
        self.name = name
    
    def __str__(self):
        return self.name


# 点，可能包括一个或者多个软件包，to_string()不等于包名
class node:
    def __init__(self, package_name):
        self.packages = [package_name]  # 节点的包括的软件包名
        self.in_node = []               # 节点的入节点
        self.out_node = []              # 节点的出节点

    def self_print(self):
        return f'{self.to_string()} \n in_node :{node.list_to_string(self.in_node)} \n out_node :{node.list_to_string(self.out_node)}\n'
    
    # node的list转成string的list
    def list_to_string(nodes):
        return [i.to_string() for i in nodes]

    def to_string(self):
        return f'({",".join(self.packages)})'

    # 添加入节点
    def add_in_node(self, node):
        if node not in self.in_node:
            self.in_node.append(node)
    
    # 删除入节点
    def del_in_node(self, node):
        if node in self.in_node:
            self.in_node.remove(node)

    # 添加出节点
    def add_out_node(self, node):
        if node not in self.out_node:
            self.out_node.append(node)
    
    # 删除出节点
    def del_out_node(self, node):
        if node in self.out_node:
            self.out_node.remove(node)

    # 节点的度
    def get_degree(self):
        return len(self.in_node) + len(self.out_node)


# 点集
class node_set:
    def __init__(self):
        self.nodes = []             # 该点集中的点集
        self.del_nodes = []         # 缩点后被删除的点集
        self.name2node = {}         # 包名到点的映射（key不会随着缩点而变化，value会随着缩点而变化）
        self.name2del_node = {}    # 缩点后被删除的映射（key包括所有缩点的包名，value的范围就是del_nodes）

    # 在点集中添加一个包
    def add_package(self, package_name):
        if package_name not in self.name2node:
            new_node = node(package_name)
            self.name2node[package_name] = new_node
            self.nodes.append(new_node)
        return self.name2node[package_name]  

    # 在点集将一个子集合并
    def merge(self, nodes):
        #print('merge : ', node.list_to_string(nodes))
        new_node = node('')
        new_node.packages = []
        self.nodes.append(new_node)
        for old_node in nodes:
            new_node.in_node += old_node.in_node
            new_node.out_node += old_node.out_node
            new_node.packages += old_node.packages
            self.nodes.remove(old_node)
            self.del_nodes.append(old_node)
            for name in old_node.packages:
                self.name2node[name] = new_node
                self.name2del_node[name] = old_node
        new_node.in_node = list(set(new_node.in_node) - set(nodes))
        new_node.out_node = list(set(new_node.out_node) - set(nodes))
        for one_in_node in new_node.in_node:
            one_in_node.out_node = list(set(one_in_node.out_node) - set(nodes))
            one_in_node.out_node.append(new_node)
        for one_out_node in new_node.out_node:
            one_out_node.in_node = list(set(one_out_node.in_node) - set(nodes))
            one_out_node.in_node.append(new_node)
    
    # 在点集中将一个子集删除
    def del_node_list(self, nodes):
        new_node = node('')
        new_node.packages = []
        for old_node in nodes:
            new_node.in_node += old_node.in_node
            new_node.out_node += old_node.out_node
            new_node.packages += old_node.packages
            self.nodes.remove(old_node)
            self.del_nodes.append(old_node)
            for name in old_node.packages:
                self.name2del_node[name] = old_node
        new_node.in_node = list(set(new_node.in_node) - set(nodes))
        new_node.out_node = list(set(new_node.out_node) - set(nodes))
        for one_in_node in new_node.in_node:
            one_in_node.out_node = list(set(one_in_node.out_node) - set(nodes))
        for one_out_node in new_node.out_node:
            one_out_node.in_node = list(set(one_out_node.in_node) - set(nodes))
        
    
    # 在点集将一个环中，如[node1, node2, node3] 依赖关系为node1 <- node2 <- node3 <- node1，假设边的权为节点的度之和，把权最大/小的边去掉，破环
    def del_edge_break_circle(self, nodes):
        edge_power = []
        for i in range(len(nodes)):
            edge_power.append(nodes[i].get_degree() + nodes[(i+1)%len(nodes)].get_degree())
        edge_power = [ [ edge_power[i], nodes[(i+1)%len(nodes)], nodes[i] ] for i in range(len(edge_power))]
        del_edge = max(edge_power, key = lambda x:x[0])
        del_edge[1].del_out_node(del_edge[2])
        del_edge[2].del_in_node(del_edge[1])
    

    # 在点集将一个环中，如[node1, node2, node3] 依赖关系为node1 <- node2 <- node3 <- node1，选择介数中心性最高的点，将它和它前面的点的边去掉，破环
    def del_edge_break_circle_by_betweenness_centrality(self, nodes, node2betweenness_centrality):
        edge_power = []
        for i in range(len(nodes)):
            edge_power.append([node2betweenness_centrality[nodes[i]], nodes[(i+1)%len(nodes)], nodes[i]])
        del_edge = max(edge_power, key = lambda x:x[0])
        del_edge[1].del_out_node(del_edge[2])
        del_edge[2].del_in_node(del_edge[1])


    # 在点集中添加一对包依赖关系
    def add_pair(self, pre_name, post_name):
        pre_node = self.add_package(pre_name)
        post_node = self.add_package(post_name)
        pre_node.add_out_node(post_node)
        post_node.add_in_node(pre_node)
    
    # 在点集中添加批量包依赖关系
    def add_pairs(self, pairs):
        for pair in pairs:
            self.add_pair(pair[0], pair[1])
    
    # 画图
    def to_graph(self):
        G = nx.DiGraph()
        node_list = node.list_to_string(self.nodes)
        for one_node in self.nodes:
            G.add_node(one_node.to_string())
            for to_node in one_node.out_node:
                G.add_edge(one_node.to_string(), to_node.to_string())

        #print(G.nodes())
        #print(G.edges())

        pos = nx.spring_layout(G)

        nx.draw_networkx_nodes(G, pos, node_size=100)
        nx.draw_networkx_labels(G, pos, font_size=7)
        nx.draw_networkx_edges(G, pos, edge_color='r', arrows = True, node_size=100)

        plt.show()

    # 给所有节点染色
    def dye(self, color):
        for node in self.nodes:
            node.color = color

    # def找环路，存在self.circle中
    def dfs_circle_find(self):
        self.dye('white')
        self.circle = []
        for one_node in self.nodes:
            if one_node.color == 'white':
                self.dfs_circle_recur(one_node)
    
    # def找环路的递归函数
    def dfs_circle_recur(self, u):
        u.color = 'gray'
        for v in u.out_node:
            if v.color == 'white':
                v.pre_node = u
                self.dfs_circle_recur(v)
            if v.color == 'gray':
                if u in v.out_node:
                    self.circle.append([v,u])
                self.circle.append([v])
                now = u
                while now != v:
                    self.circle[-1].append(now)
                    now = now.pre_node
            if v.color == 'black':
                continue
        u.color = 'black'
    
    # 根据self.circle中的环路贪心缩点
    def circle_to_merge(self):
        #0.self.circle.sort(key = lambda x: len(x), reverse = True)
        for one_circle in self.circle:
            if set(one_circle) == set(one_circle) & set(self.nodes):
                self.merge(one_circle)
    
    # 根据破边来消除环路
    def circle_to_break(self):
        self.gen_adj_dij()
        node2betweenness_centrality = self.get_betweenness_centrality()
        for one_circle in self.circle:
            for n in one_circle:
                if n not in node2betweenness_centrality:
                    self.gen_adj_dij()
                    node2betweenness_centrality = self.get_betweenness_centrality()
                    break
            if set(one_circle) == set(one_circle) & set(self.nodes):
                self.del_edge_break_circle_by_betweenness_centrality(one_circle, node2betweenness_centrality)

    # DCG -> DAG
    def dcg_to_dag(self):
        self.dfs_circle_find()
        while len(self.circle) != 0:
            #print('num of circles : ', len(self.circle))
            self.circle_to_merge()
            self.dfs_circle_find()
    
    # DAG -> 线性序列
    def dag_to_linear_sequence_by_indegree(self):
        power = {}
        indegree = {}
        sequence_list = []
        sequence_waited = self.nodes.copy()
        for one_node in sequence_waited:
            power[one_node.to_string()] = 1
            indegree[one_node.to_string()] = len(one_node.in_node)
        def sort_func(node_a, node_b):
            if indegree[node_a.to_string()] == indegree[node_b.to_string()]:
                return power[node_b.to_string()] - power[node_a.to_string()]
            else:
                return indegree[node_b.to_string()] - indegree[node_a.to_string()]
        
        while len(sequence_waited) != 0:
            sequence_waited.sort(key = functools.cmp_to_key(sort_func))
            selected_node = sequence_waited.pop()
            sequence_list.append(selected_node)
            #权重迭代
            if len(selected_node.out_node) != 0:
                one_shared = power[selected_node.to_string()]/len(selected_node.out_node)
                for to_node in selected_node.out_node:
                    indegree[to_node.to_string()] -= 1
                    power[to_node.to_string()] += one_shared
        
        return sequence_list

    # DAG -> 线性序列 自底向上
    def dag_to_linear_sequence_by_outdegree(self):
        power = {}
        outdegree = {}
        sequence_list = []
        sequence_waited = self.nodes.copy()
        for one_node in sequence_waited:
            power[one_node.to_string()] = 1
            outdegree[one_node.to_string()] = len(one_node.out_node)
        def sort_func(node_a, node_b):
            if outdegree[node_a.to_string()] == outdegree[node_b.to_string()]:
                return power[node_b.to_string()] - power[node_a.to_string()]
            else:
                return outdegree[node_b.to_string()] - outdegree[node_a.to_string()]
        sequence_waited.sort(key = lambda x:len(x.out_node), reverse = True)
        while len(sequence_waited) != 0:
            #sequence_waited.sort(key = functools.cmp_to_key(sort_func))
            selected_node = sequence_waited.pop()
            sequence_list.append(selected_node)
            #权重迭代
            if len(selected_node.in_node) != 0:
                one_shared = power[selected_node.to_string()]/len(selected_node.in_node)
                for to_node in selected_node.in_node:
                    outdegree[to_node.to_string()] -= 1
                    power[to_node.to_string()] += one_shared
        return sequence_list

    
    # DAG -> 线性序列 自底向上
    def dag_to_linear_sequence_only_by_outdegree(self):
        outdegree = {}
        sequence_list = []
        sequence_waited = self.nodes.copy()
        for one_node in sequence_waited:
            outdegree[one_node.to_string()] = len(one_node.out_node)
        def sort_func(node_a, node_b):
                return outdegree[node_b.to_string()] - outdegree[node_a.to_string()]
        sequence_waited.sort(key = lambda x:len(x.out_node), reverse = True)
        while len(sequence_waited) != 0:
            selected_node = sequence_waited.pop()
            sequence_list.append(selected_node)
        return sequence_list
    
    
    # DAG -> 线性序列 自底向上, 带初始出度
    def dag_to_linear_sequence_by_outdegree_with_init_power(self, package2power):
        power = {}
        outdegree = {}
        sequence_list = []
        sequence_waited = self.nodes.copy()
        for one_node in sequence_waited:
            power[one_node.to_string()] = 1
            outdegree[one_node.to_string()] = len(one_node.out_node)
            init_power = 0
            for pack in one_node.packages:
                if pack in package2power:
                    init_power += package2power[pack]
                else:
                    print(pack + ' not in package2power')
            outdegree[one_node.to_string()] += init_power/len(one_node.packages)

        def sort_func(node_a, node_b):
            if outdegree[node_a.to_string()] == outdegree[node_b.to_string()]:
                return power[node_b.to_string()] - power[node_a.to_string()]
            else:
                return outdegree[node_b.to_string()] - outdegree[node_a.to_string()]
        #sequence_waited.sort(key = lambda x:len(x.out_node), reverse = True)
        while len(sequence_waited) != 0:
            sequence_waited.sort(key = functools.cmp_to_key(sort_func))
            selected_node = sequence_waited.pop()
            sequence_list.append(selected_node)
            #权重迭代
            if len(selected_node.in_node) != 0:
                one_shared = power[selected_node.to_string()]/len(selected_node.in_node)
                for to_node in selected_node.in_node:
                    outdegree[to_node.to_string()] -= 1
                    power[to_node.to_string()] += one_shared
        return sequence_list
    
    # 关节点计算，Tarjan算法
    def get_biconnected_component(self,seq):
        for one_node in self.nodes:
            one_node.visited = False
            one_node.deep = 0
            one_node.low = 0
            one_node.pre_node = None
        self.biconnected_component_list = []
        for one_node in seq:
            if not one_node.visited:
                self.get_biconnected_component_recur(one_node, 0)
        # 去掉入度为0的点
        self.biconnected_component_list = [n for n in self.biconnected_component_list if len(n.in_node) != 0]

        #print(node.list_to_string(self.biconnected_component_list))
        return

    # 关节点计算递归函数
    def get_biconnected_component_recur(self, now_node, deep):
        now_node.visited = True
        now_node.deep = deep
        now_node.low = deep
        child_count = 0
        is_articulation = False
        for next_node in now_node.out_node:
            if next_node.visited != True:
                next_node.pre_node = now_node
                self.get_biconnected_component_recur(next_node, deep+1)
                child_count += 1
                if next_node.low >= now_node.deep:
                    is_articulation = True
                now_node.low = min(now_node.low, next_node.low)
            elif now_node.pre_node != None and next_node != now_node.pre_node:
                now_node.low = min(now_node.low, next_node.deep)
        if (now_node.pre_node != None and is_articulation) or (now_node.pre_node == None and child_count > 1):
            self.biconnected_component_list.append(now_node)

    # 生成邻接矩阵，并用弗洛伊德算法生成最短路径
    def gen_adj_dij(self):
        self.adj_matrix = [[0 for j in range(len(self.nodes)) ] for i in range(len(self.nodes))]
        self.node2index = {}
        for i in range(len(self.nodes)):
            self.node2index[self.nodes[i]] = i
        for from_node in self.nodes:
            for to_node in from_node.in_node:
                self.adj_matrix[self.node2index[from_node]][self.node2index[to_node]] = 1
                self.adj_matrix[self.node2index[to_node]][self.node2index[from_node]] = 1
        self.dij_matrix = self.adj_matrix.copy()
        for k in range(len(self.nodes)):
            for i in range(len(self.nodes)):
                if self.dij_matrix[i][k] == 0:
                    continue
                for j in range(len(self.nodes)):
                    if self.dij_matrix[k][i] == 0:
                        continue
                    if self.dij_matrix[i][j] == 0 or self.dij_matrix[i][j] > self.dij_matrix[i][k] + self.dij_matrix[k][i]:
                        self.dij_matrix[i][j] = self.dij_matrix[i][k] + self.dij_matrix[k][i]
    
    # gen_adj_dij()后，获取两个点之间的距离
    def get_dist(self, from_node, to_node):
        return self.adj_matrix[self.node2index[from_node]][self.node2index[to_node]]

    
    # 根据一个metric对node进行排序
    def sort_by_metric(self, metric_map):
        n = self.nodes.copy()
        n.sort(key = lambda x: metric_map[x], reverse = True)
        return n
    
    # 计算度中心性
    def get_degree_centrality(self):
        res = {}
        for n in self.nodes:
            res[n] = n.get_degree() / (len(self.nodes) - 1)
        return res
    
    # 计算接近中心性
    def get_closeness_centrality(self):
        res = {}
        def bfs_get_dist(u):
            total_dist = 0
            dist = 1
            visited = set([u])
            next = set(u.in_node + u.out_node) - visited
            while len(next) != 0:
                total_dist += dist * len(next)
                visited |= next
                next_waited = set()
                for v in next:
                    next_waited |= set(v.in_node) | set(v.out_node)
                next_waited -= visited
                next = next_waited
                dist += 1
            return total_dist
        
        for n in self.nodes:
            total_dist = bfs_get_dist(n)
            if total_dist != 0:
                res[n] = (len(self.nodes) - 1) / total_dist
            else:
                res[n] = 0
        return res
    

    # 计算介数中心性
    def get_betweenness_centrality(self):
        res = collections.defaultdict(int)
        for i in range(0,len(self.nodes)):
            for j in range(i+1, len(self.nodes)):
                min_dist = self.get_dist(self.nodes[i], self.nodes[j])
                in_road = []
                for k in range(0,len(self.nodes)):
                    if i == k or j == k:
                        continue
                    if self.get_dist(self.nodes[i], self.nodes[k]) + self.get_dist(self.nodes[k], self.nodes[j]) == min_dist:
                        in_road.append(k)
                for k in in_road:
                    res[self.nodes[k]] += 1/len(in_road)
        return res


    # 原始分层算法
    def origin_divide_layer(self, sequence_list):
        last = [n for n in sequence_list if len(n.in_node) == 0]
        layers = [last]
        last_dependency = set()
        for one_node in last:
            last_dependency |= set(one_node.out_node)
        layer = []
        layer_dependency = set()
        res_list = [n for n in sequence_list if len(n.in_node) != 0]
        waited_list = set(res_list)
        
        for now_node in res_list:
            if len(last_dependency & waited_list) == 0:
                last = layer
                last_dependency = layer_dependency
                layers.append(layer)
                layer = [now_node]
                layer_dependency = set(now_node.out_node)
            else:
                layer.append(now_node)
                layer_dependency |= set(now_node.out_node)
            waited_list.remove(now_node)
        layers.append(layer)
        return layers
    
    # 原始分层算法 自底向上分层
    def origin_divide_layer_reverse(self, sequence_list):
        last = [n for n in sequence_list if len(n.out_node) == 0]
        layers = [last]
        last_dependency = set()
        for one_node in last:
            last_dependency |= set(one_node.in_node)
        layer = []
        layer_dependency = set()
        res_list = [n for n in sequence_list if len(n.out_node) != 0]
        waited_list = set(res_list)
        
        for now_node in res_list:
            if len(last_dependency & waited_list) == 0:
            #if len(last_dependency & waited_list) / len(last_dependency) < 0.1:
                last = layer
                last_dependency = layer_dependency
                layers.append(layer)
                layer = [now_node]
                layer_dependency = set(now_node.in_node)
            else:
                layer.append(now_node)
                layer_dependency |= set(now_node.in_node)
            waited_list.remove(now_node)
        layers.append(layer)
        return layers
    
    # 原始分层算法 自底向上分层 带初始权重
    def origin_divide_layer_reverse_with_init_power(self, sequence_list, package2power):
        is_addition_list = True
        addition_list = []
        node2power = collections.defaultdict(int)
        for n in sequence_list:
            for pack in n.packages:
                if pack in package2power:
                    node2power[n] += package2power[pack]
            node2power[n] /= len(n.packages)
        last = [n for n in sequence_list if len(n.out_node) + node2power[n] == 0]
        layers = [last]
        last_dependency = set()
        for one_node in last:
            last_dependency |= set(one_node.in_node)
        layer = []
        layer_dependency = set()
        res_list = [n for n in sequence_list if len(n.out_node) + node2power[n] != 0]
        print(len(res_list))
        waited_list = set(res_list)
        for now_node in res_list:
            if len(last_dependency & waited_list) == 0:
            #if len(last_dependency & waited_list) / len(last_dependency) < 0.1:
                last = layer
                last_dependency = layer_dependency
                layers.append(layer)
                layer = [now_node]
                layer_dependency = set(now_node.in_node)
            else:
                if is_addition_list:
                    if len(now_node.in_node) + len(now_node.out_node) < 3:
                        addition_list.append(now_node)
                    else:
                        layer.append(now_node)
                        layer_dependency |= set(now_node.in_node)
                else:
                    layer.append(now_node)
                    layer_dependency |= set(now_node.in_node)
            waited_list.remove(now_node)
        layers.append(layer)
        if is_addition_list:
            layers.append(addition_list)
        return layers

    

        




def build_test_set1():
    s = node_set()
    s.add_pair('A','B')
    s.add_pair('A','C')
    s.add_pair('B','D0')
    s.add_pair('B','D1')
    s.add_pair('C','F')
    s.add_pair('D0','F')
    s.add_pair('D1','F')
    s.add_pair('D0','E')
    s.add_pair('E','B')
    return s

def build_test_set2():
    s = node_set()
    s.add_pair('A','B')
    s.add_pair('A','C')
    s.add_pair('B','D0')
    s.add_pair('B','D1')
    s.add_pair('C','F')
    s.add_pair('D0','F')
    s.add_pair('D1','F')
    s.add_pair('D0','E')
    s.add_pair('E','B')
    s.add_pair('C','A')
    return s

def build_test_set3():
    s = node_set()
    s.add_pair('A','B')
    s.add_pair('B','A')
    s.add_pair('B','C')
    s.add_pair('C','D')
    s.add_pair('D','C')
    s.add_pair('D','E')
    s.add_pair('D','A')
    s.add_pair('F','A')
    return s


def node_set_print(s):
    names = []
    for name in s.name2node:
        names.append(name)
    print(f' num of node : {len(s.nodes)} \n num of node map : {len(s.name2node)} \n num of del_node : {len(s.del_nodes)} \n num of del_node map : {len(s.name2del_node)}\n node name list : {node.list_to_string(s.nodes)} \n package name list : {names}\n')
    for n in s.nodes:
        print(n.self_print())
    print('\n')


def layers_print(layers):
    for layer in layers:
        package_num = 0
        for one_node in layer:
            package_num += len(one_node.packages)
        print(len(layer), package_num, node.list_to_string(layer))


def load_data(filename):
    with open(filename, 'r') as f:
        raw_data = f.read()
        lines = raw_data.split('\n')
        dependency_relationship = [line for line in lines if line.find('->') != -1]
        dependency_relationship = [ [i.strip(';').strip().strip('"') for i in pair.split('->')] for pair in dependency_relationship]
        return dependency_relationship


def load_expert(filename):
    with open(filename, 'r') as f:
        raw_data = f.read()
        lines = raw_data.split('\n')
        lines = [line.split('\t') for line in lines]
        package2layer_expert = {line[0]:int(line[1]) for line in lines}
        return package2layer_expert


def load_label(filename):
    with open(filename, 'r') as f:
        raw_data = f.read()
        lines = raw_data.split('\n')
        lines = [line.split('\t') for line in lines]
        label2level = {
            'ProgramLanguage':2,
            'DebugTools':2,
            'BuildTools':0,
            'DevelopTool':1,
            'Kernel':0,
            'SystemService':2,
            'SystemLibrary':2,
            'SystemTools':2,
            'InformationLibrary':2,
            'ApplicationService':3,
            'ApplicationLibrary':3,
            'ApplicationTools':3
        }
        level2power = {
            0: 0,
            1: 2,
            2: 4,
            3: 6,
            4: 8,
            5: 10
        }
        package2power = {line[0]:level2power[label2level[line[1]]] for line in lines}
        return package2power


def expert_test(filename, s, layers):
    expert_res = load_expert(filename)
    m = [0,0,0, [[],[],[]], []]                 # 分类正确数，分类错误数，未命中数, [分类正确，分类错误，未命中 的包名], [[包名，专家分层，算法分层]]
    for diff in range(-4, 3):
        package2layer = layers_to_map(layers,diff)
        t = [0,0,0, [[],[],[]], [], package2layer]            
        for one in expert_res:
            if one in package2layer:
                t[4].append([one, expert_res[one], package2layer[one]])
                if expert_res[one] == package2layer[one]:
                    t[0] += 1
                    t[3][0].append(one)
                else:
                    t[1] += 1
                    t[3][1].append(one)
            else:
                t[2] += 1
                t[3][2].append(one)
        m = max(t,m, key=lambda x: x[0])
    print(m[:3], m[0]/(m[0]+m[1]))


    analyze_expert_res(filename, s)
    return m

def analyze_expert_res(filename, s):
    expert_res = load_expert(filename)
    layers_expert = [[],[],[]]
    for pack in expert_res:
        if pack in s.name2node:
            layers_expert[3 - expert_res[pack]].append(s.name2node[pack])
    for layer in layers_expert:
            depend_layer = collections.defaultdict(int)
            for n in layer:
                for out_n in n.out_node:
                    if out_n.packages[0] in expert_res:
                        depend_layer[expert_res[out_n.packages[0]]] += 1
            for d in depend_layer:
                depend_layer[d] /= len(layer)
            depend_layer['length'] = len(layer)
            print(depend_layer)


# 将layers转成包名到层的映射
def layers_to_map(layers, diff = 1):
    m = {}
    for i in range(len(layers)):
        for one_node in layers[i]:
            for package in one_node.packages:
                m[package] = len(layers) - i + diff
    return m


def expert_test_reverse(s, layers):
    expert_res = load_expert('./20220715/input/mysqlexpert.txt')
    t = 0
    in_circle = 0
    for one in expert_res:
        if one[0] in s.name2node:
            if s.name2node[one[0]] in layers[0]:
                one.append('3')
            elif s.name2node[one[0]] in layers[1]:
                one.append('2')
            elif s.name2node[one[0]] in layers[2]:
                one.append('0')
        if one[1] == one[2]:
            t += 1
        else:
            if one[0] in s.name2del_node:
                in_circle += 1
    print(t, t/len(expert_res), in_circle)


def test():
    s = build_test_set3()
    #node_set_print(s)
    s.dcg_to_dag()
    #node_set_print(s)
    seq = s.dag_to_linear_sequence_by_indegree()
    print(node.list_to_string(seq))
    layers = s.origin_divide_layer(seq)
    for layer in layers:
        print(node.list_to_string(layer))
    

if __name__ == "__main__":
    for select_num in range(0, 1):
    
        # 从文件中读入依赖关系对list
        dependency_relationship = load_data('./20220715/input/mysqlanolis.dot')
        #dependency_relationship = load_data('./20220715/input/podman.dot')
        s = node_set()
        # 将依赖关系对list添加到点集中
        s.add_pairs(dependency_relationship)
        #analyze_expert_res('./20220715/input/mysqlexpert.txt', s)
        #s.to_graph()
        degree_center_1 = s.get_degree_centrality()
        degree_center_1_res = s.sort_by_metric(degree_center_1)
        s.del_node_list(degree_center_1_res[:select_num])
        degree_center_1_res = [[degree_center_1[n]] for n in degree_center_1_res]
        #print(degree_center_1_res)
        # 缩点、生成DAG
        print(s.name2node['glibc'].self_print())
        s.dcg_to_dag()
        print(s.name2node['glibc'].self_print())

        '''
        degree_center_2 = s.get_degree_centrality()
        degree_center_2_res = s.sort_by_metric(degree_center_2)
        print( [[n.to_string(), degree_center_2[n]] for n in degree_center_2_res])
        #s.del_node_list(degree_center_2_res[:1])
        degree_center_2_res = [[degree_center_2[n]] for n in degree_center_2_res]
        #print(degree_center_2_res)

        #big_node = s.name2node['perl']
        #print(big_node.to_string())
        #s.del_node_list([big_node])

        # 根据入度拓扑排序生成线性序列
        seq = s.dag_to_linear_sequence_by_indegree()
        print(node.list_to_string(seq))
        #s.get_biconnected_component(seq)

        layers = s.origin_divide_layer(seq)
        layers_print(layers)
        comp = expert_test('./20220715/input/mysqlexpert.txt', s, layers)
        #for one in comp[4]:
            #print(f'{one[0]},{one[1]},{one[2]}')
        package2layer = comp[5]
        #找出各层对不同层级的依赖
        print('-------------------------')
        for layer in layers:
            depend_layer = collections.defaultdict(int)
            for n in layer:
                for out_n in n.out_node:
                    depend_layer[package2layer[out_n.packages[0]]] += 1
            for d in depend_layer:
                depend_layer[d] /= len(layer)
            print(depend_layer)




        '''
        print('------------------reverse---------------------')
        # 根据出度拓扑排序生成线性序列
        #package2power = load_label('./20220715/input/label_map.txt')
        #seq_r = s.dag_to_linear_sequence_by_outdegree_with_init_power(package2power)
        #print(node.list_to_string(seq_r))
        #layers_r = s.origin_divide_layer_reverse_with_init_power(seq_r, package2power)

        seq_r = s.dag_to_linear_sequence_only_by_outdegree()
        print(node.list_to_string(seq_r))
        layers_r = s.origin_divide_layer_reverse(seq_r)
        print(len(s.nodes))
        layers_print(layers_r)
        #expert_test_reverse(s, layers_r)
        for layer in layers_r:
            print('---------------layer---------------')
            for n in layer:
                in_cycle = len(n.packages) != 1
                for pack in n.packages:
                    print(f'{pack}\t{len(n.out_node)}\t{len(n.in_node)}\t{in_cycle}')
        