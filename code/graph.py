__author__ = "chaitanya"

from collections import defaultdict


class Graph:

    def __init__(self, directed=True):
        self.relations = defaultdict()  #  defaultdict()是一个字典，当你访问一个不存在的键时，它会返回一个默认值，而不是抛出异常
        self.nodes = defaultdict()  # defaultdict()是一个字典，当你访问一个不存在的键时，它会返回一个默认值
        self.node2id = {}  # 一个空字典，用来存储节点的名字和id
        self.relation2id = {}  # 一个空字典，用来存储关系的名字和id
        self.edges = {}
        self.edgeCount = 0
        self.directed = directed  # 是否有环
        #self.add_node("UNK-NODE")
        #self.add_relation("UNK-REL")

    def add_edge(self, node1, node2, rel, label, weight, uri=None):
        """  # uri是什么

        :param node1: source node
        :param node2: target node
        :param rel: relation
        :param label: relation
        :param weight: weight of edge from [0.0, 1.0]
        :param uri: uri of edge
        :return: Edge object
        """
        new_edge = Edge(node1, node2, rel, label, weight, uri)

        if node2 in self.edges[node1]:
            self.edges[node1][node2].append(new_edge)
        else: 
            self.edges[node1][node2] = [new_edge]

        # node1.neighbors.add(node2)
        node2.neighbors.add(node1)
        self.edgeCount += 1  # 边的数量

        if (self.edgeCount) % 100000 == 0:
            print("Number of edges: %d" % self.edgeCount, end="\r")
          
        return new_edge      
        
    def add_node(self, name):
        """

        :param name:
        :return:
        """
        new_node = Node(name, len(self.nodes))  # 一个节点的名字和id, Node#0的意思是第一个节点
        self.nodes[len(self.nodes)] = new_node  #
        self.node2id[new_node.name] = len(self.nodes) - 1  #
        self.edges[new_node] = {}
        return self.node2id[new_node.name]

    def add_relation(self, name):
        """
        :param name
        :return:
        """
        new_relation = Relation(name, len(self.relations))
        self.relations[len(self.relations)] = new_relation
        self.relation2id[new_relation.name] = len(self.relations) - 1
        return self.relation2id[new_relation.name]

    def find_node(self, name):
        """
        :param name:
        :return:
        """
        if name in self.node2id:  # 如果name在node2id中,返回node2id[name]
            return self.node2id[name]
        else:
            return -1

    def find_relation(self, name):
        """
        :param name:
        :return:
        """
        if name in self.relation2id:
            return self.relation2id[name]
        else:
            return -1

    def is_connected(self, node1, node2):
        """

        :param node1:
        :param node2:
        :return:
        """

        if node1 in self.edges:
            if node2 in self.edges[node1]:
                return True
        return False

    def node_exists(self, node):
        """

        :param node: node to check
        :return: Boolean value
        """
        if node in self.nodes.values():
            return True
        return False

    def find_all_connections(self, relation):
        """
        :param relation:
        :return: list of all edges representing this relation
        """
        relevant_edges = []
        for edge in self.edges:
            if edge.relation == relation:
                relevant_edges.append(edge)

        return relevant_edges

    def iter_nodes(self):  # 迭代器
        return list(self.nodes.values())

    def iter_relations(self):
        return list(self.relations.values())

    def iter_edges(self):
        for node in self.edges:
            for edge_list in self.edges[node].values():
                for edge in edge_list:
                    yield edge  # yield 语句用于定义生成器，生成器是一个返回迭代器的函数，只能用于迭代操作，生成器函数与普通函数不同，生成器函数只能用于迭代操作

    def __str__(self):
        for node in self.nodes:
            print(node)


class Node:

    def __init__(self, name, id, lang='en'):
        self.name = name
        self.id = id
        self.lang = lang
        self.neighbors = set([])

    def get_neighbors(self):
        """
        :param node:
        :return:
        """
        return self.neighbors

    def get_degree(self):
        """

        :param node:
        :return:
        """
        return len(self.neighbors)

    def __str__(self):
        out = ("Node #%d : %s" % (self.id, self.name))
        return out


class Relation:

    def __init__(self, name, id):
        self.name = name
        self.id = id


class Edge:

    def __init__(self, node1, node2, relation, label, weight, uri):
        self.src = node1  # source node，源节点
        self.tgt = node2  # target node，目标节点
        self.relation = relation  # relation
        self.label = label  # relation
        self.weight = weight
        self.uri = uri

    def __str__(self):
        out = ("%s: %s --> %s" % (self.relation.name, self.src.name, self.tgt.name))
        return out
