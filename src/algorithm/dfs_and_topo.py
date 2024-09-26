# SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
#                              Peking University
# SPDX-License-Identifier: Apache-2.0

from email.errors import FirstHeaderLineIsContinuationDefect
import queue

class Package:
    
    def __init__(self,name):
        self.name = name

    def get_package_name(self):
        return self.name

class Node():
    def __init__(self):
        self.color = "white"
        self.dependency = []        #节点的依赖
        self.packages = set()          #节点包含的包  因为有环的存在，将形环的包当成一个节点进行分层

        self.inNode = set()         #入度节点
        self.outNode = set()        #出度节点
        self.indirectInNode = []    #间接入度节点


    def nodeAddPackage(self,packageName):
        self.packages.add(packageName)

    def addInNode(self,node):
        """
        添加入度节点
        """
        self.inNode.add(node)

    def addOutNode(self,node):
        """
        添加出度节点
        """
        self.outNode.add(node)
        

    def add_forward(self,node):
        self.forward = node
    
    def addDependency(self,node):
        self.dependency.append(node)

    def get_inNode_counter(self):
        return len(self.inNode)

    def get_indirectInNode_counter(self):
        """
        计算间接节点个数
        """
        return len(self.indirectInNode)

    def get_outNode_counter(self):
        """
        计算节点出度
        """
        return len(self.outNode)

    def union(self,circle):
        """
        环节点
        """
        self.dependency = []        #节点的依赖
        self.packages = []          #节点包含的包  因为有环的存在，将形环的包当成一个节点进行分层

        for node in circle:
            self.packages.extend(node.packages)
            self.dependency.extend(node.dependency)

        #删除依赖中的包本身
        for package in self.packages:

            if package in self.dependency:
                self.dependency.remove(package)

        self.packages = list(set(self.packages))
        self.dependency = list(set(self.dependency))
        return self



class NodeSet:

    def __init__(self):
        self.nodeSet = {}        #节点集合
        self.packages = set()    #软件包集合
        self.packageLayer = {}      #分层

    def addNode(self,name):
        """
        添加节点
        """
        if name not in self.packages:
            self.packages.add(name)     #nodeset中的packages添加包
            node = Node()               #新建节点
            self.nodeSet[name] = node   #节点填入字典
        return self.nodeSet.get(name)   #获取字典值

    def setInit(self,nodeName,dependencyName):
        """
        节点集合初始化
        各个节点添加依赖信息，入度信息，出度信息和节点所包含的软件包信息
        """
        node = self.addNode(nodeName)   
        dependencyNode = self.addNode(dependencyName)
        node.addDependency(dependencyName)  #添加依赖
        node.nodeAddPackage(nodeName)                   #节点添加包  
        dependencyNode.nodeAddPackage(dependencyName)   #依赖节点添加包

    def getNode(self,name):
        """
        获取节点
        """
        return self.nodeSet[name]

    def addDependency(self,nodeName,dependencyName):
        """
        添加依赖
        """
        node = self.addNode(nodeName)
        dependencyNode = self.addNode(dependencyName)
        node.addDependency(dependencyNode)
        

    def nodeUpdata(self,node):
        """
        更新节点信息
        """
        for package in node.packages:
            self.nodeSet[package] = node

    def dfs(self):
        """
        dfs算法
        """
        for node in self.nodeSet.values():
            if node.color == "white":
                self.dfs_search(node)

    def dfs_search(self,node):
        """
        查找环，并更新环中节点信息
        """
        node.color = "gray"
        for nextNode in node.dependency:
            nextNode = self.getNode(nextNode)
            if nextNode.color == "white":
                nextNode.add_forward(node)
                self.dfs_search(nextNode)

            elif nextNode.color == "gray":
                circle = []
                while node != nextNode:
                    circle.append(node)
                    node = node.forward
                circle.append(node)
                #print("circle = ",circle)

                newNode = Node()
                newNode.union(circle)
                newNode.add_forward(nextNode)
                self.nodeUpdata(newNode)
                #print("newNode = ",newNode.packages)
                self.dfs_search(newNode)

            else:
                continue

        node.color = "black"

    def add_inAndOut_node(self):
        """
        节点添加入度和出度
        """
        for node in self.nodeSet.values():          #所有节点
            for package in node.dependency:         #节点依赖名
                depNode = self.getNode(package)     #获得依赖节点
                node.addOutNode(depNode)            #节点添加出度
                depNode.addInNode(node)             #依赖节点添加入度 

    def part_add_inAndOut_node(self,part):
        """
        添加入度和出度节点，
        """
        #清除原有节点中的出度，入度、间接入度信息
        for node in part:
            node.inNode.clear()
            node.outNode.clear()
            node.indirectInNode.clear()

        for node in part:
            for package in node.dependency: 
                depNode = self.getNode(package)
                if depNode in part:             #被依赖的节点在part中
                    node.addOutNode(depNode)
                    depNode.addInNode(node)

    def add_indirectInNode(self):
        """
        添加间接入度节点
        """
        nodes = list(set(self.nodeSet.values()))
        zero_list = []
        finished = []

        #找出间接入度为0的节点
        for node in nodes:
            node.indirectInNode.extend(node.inNode)
            if node.get_indirectInNode_counter() == 0:
                zero_list.append(node)

        finished.extend(zero_list)
        for node in zero_list:
            for dependency in node.outNode:
                dependency.indirectInNode.extend(node.indirectInNode)
                if dependency in finished:
                    continue
                zero_list.append(dependency)
                finished.append(dependency)

            

    def generateTopoList(self):
        """
        拓扑排列
        """
        self.add_inAndOut_node()
        self.add_indirectInNode()

        Topo_list = []
        nodes = list(set(self.nodeSet.values()))

        while len(nodes) > 0:
            temp = []

            for node in nodes:
                if node.get_outNode_counter() == 0:
                    temp.append(node)
            
            temp.sort(key=lambda x:(x.get_inNode_counter(),x.get_indirectInNode_counter()))

            for node in temp:
                nodes.remove(node)
                Topo_list.append(node)

                for dep in node.inNode:
                    dep.outNode.remove(node)
        
        return Topo_list

    def isLayerDepend(self,remain,lastlayer):
        """
        remain是否依赖lastlayer
        """
        for node in remain:
            for dependency in node.outNode:
                if dependency in lastlayer:
                    return True
        return False


    def generateLayer(self,topoList):
        """
        原始分层
        """
        remain = list(set(topoList))
        layers = []
        lastlayer = []
        self.part_add_inAndOut_node(remain)
        
        i = 0
        for node in topoList:
            if node.get_outNode_counter() == 0:
                remain.remove(node)
                lastlayer.append(node)
                i += 1
            else:
                break
        #print("lastlayer = ",lastlayer)
        layers.append(lastlayer)

        layer = []
        while i<len(topoList):
            layer.append(topoList[i])
            remain.remove(topoList[i])

            if self.isLayerDepend(remain,lastlayer) == False:
                layers.append(layer)
                lastlayer = layer
                layer = []

            i += 1

        return layers

    def get_packages_layer(self,layers):
        """
        计算软件包所属层
        """
        for layer in layers:
            index = layers.index(layer)
            for node in layer:
                for package in node.packages:
                    self.packageLayer[package] = index+1

