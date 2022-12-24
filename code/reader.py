__author__ = "chaitanya"

import logging as logger

from tqdm import tqdm

from graph import Graph

import csv
import json
import os
import pandas as pd
import random
from sklearn.metrics.pairwise import cosine_similarity
from bert_feature_extractor import BertLayer
import numpy as np


class Reader:

    def print_summary(self):

        print("\n\nGraph Summary")  # 打印图的信息 有多少个节点，多少个边，多少个关系
        print("\nNodes: %d" % len(self.graph.nodes))  # 节点数7962
        print("Edges: %d" % self.graph.edgeCount)   # 边数  10000
        print("Relations: %d" % len(self.graph.relation2id))  # 关系数  28
        density = self.graph.edgeCount / (len(self.graph.nodes) * (len(self.graph.nodes) - 1))  # density是边的数量除以节点的数量,这里是0.000158
        print("Density: %f" % density)

        print("\n******************* Sample Edges *******************")

        for i, edge in enumerate(self.graph.iter_edges()):
            print(edge)
            if (i+1) % 10 == 0:
                break

        print("***************** ***************** *****************\n")

    def gen_negative_examples(self, tgt_size=None, sampling_type="random"):

        print("Generating negative examples..")

        existing_edges = list(self.graph.iter_edges())
        existing_nodes = list(self.graph.iter_nodes())
        existing_relations = list(self.graph.iter_relations())

        if tgt_size:
            selected_edges = random.sample(existing_edges, tgt_size)
        else:
            selected_edges = existing_edges

        # Generate 3 negative samples per example
        idx = 0

        for i, edge in enumerate(selected_edges):
            src, rel, tgt = edge.src, edge.relation, edge.tgt

            rand_nodes = []
            while len(rand_nodes) != 2:
                sample = random.sample(existing_nodes, 1)
                if sample not in [src, tgt]:
                    rand_nodes.append(sample[0])

            found = False
            while not found:
                sample = random.sample(existing_relations, 1)
                if sample != rel:
                    rand_rel = sample
                    found = True

            self.add_example(src.name, rand_nodes[0].name, rel.name, 1.0, 0)
            self.add_example(rand_nodes[1].name, tgt.name, rel.name, 1.0, 0)
            self.add_example(src.name, tgt.name, rand_rel[0].name, 1.0, 0)
            idx += 3

        print("Added %d negative examples using %s sampling" %(idx, sampling_type))

    def add_sim_edges_bert(self, bert_model):

        sim_counter = 0  # 用来计数的
        threshold = 0.999  # 阈值,大于这个阈值的就是相似的 语义相似度大于阈值 τ，我们就在它们之间添加一条辅助边
        sim_list = []  # 用来存储相似的边的
        node_list = self.graph.iter_nodes()  # 迭代器,返回所有的节点

        # 所有节点的向量表示，用于计算相似度，这里用的是bert的向量表示
        vecs = bert_model.forward(node_list)  # bert_model.forward()是bert的前向传播,返回所有节点的向量表示
        vecs = vecs.cpu().numpy()  # 转换成numpy数组 【8442,1024】 1024是bert的向量维度，8442是节点的数量

        # vecs = np.vstack(vecs)
        print("Computed embeddings.")  # 打印信息，已经计算好了向量表示

        batch_size = 1000  # 每次计算相似度的时候，每个batch的大小
        out_sims = []  # 用来存储相似度的，这里是一个二维数组，每个元素是一个一维数组，里面存储的是每个节点与其他节点的相似度
        # 用tqdm来显示进度条, 这里很慢,所以用了tqdm，可以看到进度条，知道大概需要多久
        for row_i in tqdm(range(0, int(vecs.shape[0] / batch_size) + 1)):
            start = row_i * batch_size  # 开始的位置
            end = min([(row_i + 1) * batch_size, vecs.shape[0]])
            if end <= start:
                break
            rows = vecs[start: end]  # 一次取出batch_size个向量
            sim = cosine_similarity(rows, vecs)  # rows is O(1) size, cosinesimilarity 是计算余弦相似度的函数
            # 2 nodes with unknown text can have perfect similarity
            sim[sim == 1.0] = 0  # 把相似度为1的都变成0
            sim[sim < threshold] = 0  # 把相似度小于阈值的都变成0

            for i in tqdm(range(rows.shape[0])):
                indices = np.nonzero(sim[i])[0]  # 返回非零元素的索引

                for index in indices:
                    if index!=i+start:
                        self.add_example(node_list[i+start].name, node_list[index].name, "sim", 1.0)
                        out_sims.append((node_list[i+start].name, node_list[index].name))
                        #self.add_example(node_list[index], node_list[i+start], "sim", 1.0)
                        sim_counter += 1  # 相似的边的数量加1
                        #if sim_counter > 150000:
                        #    break


        # with open("bert_atomic_sims.txt", 'w') as f:
        #     f.writelines([s[0] + "\t" + s[1] + "\n" for s in out_sims])

        print("Added %d sim edges" % sim_counter)  # 添加了多少个相似的边， 942 【0.999】


class ConceptNetTSVReader(Reader):  # 读取conceptnet的数据集

    def __init__(self, dataset):
        logger.info("Reading ConceptNet")  # 读取conceptnet, 生成图，读取概念网络，logger.info是打印日志的，打印到了log文件中，log文件的路径在config.py中
        self.dataset = dataset  # conceptnet
        self.graph = Graph()  # graph object, 图对象
        self.rel2id = {}  # relation to id
        self.unique_entities = set()  # 是一个集合，存储所有的实体，不重复

    def read_network(self, data_dir, split="train", train_network=None):

        if split == "train":
            data_path = os.path.join(data_dir, "train.txt")  # 训练集的路径'../data/ConceptNet/small_data/train.txt'
        elif split == "valid":  # 验证集的作用是用来调参的，调参的时候用的是验证集，验证集的数据是不参与训练的，是吗？
            data_path = os.path.join(data_dir, "valid.txt")  # 验证集的路径'../data/ConceptNet/small_data/valid.txt'
        elif split == "test":
            data_path = os.path.join(data_dir, "test.txt")

        with open(data_path) as f:
            data = f.readlines()  # 读取所有的行,返回一个列表,第一行数据：ReceivesAction	hockey	play on ice

        if split == "test":
            data = data[:1200]  # 只取前1200个

        for inst in data:  # inst是一行数据， 例如：'ReceivesAction\thockey\tplay on ice'
            inst = inst.strip()  # 去掉首尾的空格
            if inst:  # 如果不是空的，就进行处理
                inst = inst.split('\t')  # 以\t为分隔符，分割成一个列表,['ReceivesAction', 'hockey', 'play on ice']
                rel, src, tgt = inst  # rel是关系，src是源实体，tgt是目标实体 'ReceivesAction', 'hockey', 'play on ice'
                weight = 1.0  # 权重,这里都是1.0
                src = src.lower()  # 小写
                tgt = tgt.lower()
                self.unique_entities.add(src)  # 添加实体,集合中不会重复 add() 方法用于给集合添加元素，如果添加的元素在集合中已存在，则不执行任何操作。
                self.unique_entities.add(tgt)
                if split != "train":
                    self.add_example(src, tgt, rel, float(weight), int(weight), train_network)
                else:
                    self.add_example(src, tgt, rel, float(weight))  # 添加实例，添加到图中

        self.rel2id = self.graph.relation2id  #


    def add_example(self, src, tgt, relation, weight, label=1, train_network=None):
        # add_example()函数是添加实例，添加到图中
        src_id = self.graph.find_node(src)  # 找到src的id
        if src_id == -1:
            src_id = self.graph.add_node(src)

        tgt_id = self.graph.find_node(tgt)
        if tgt_id == -1:
            tgt_id = self.graph.add_node(tgt)

        relation_id = self.graph.find_relation(relation)
        if relation_id == -1:
            relation_id = self.graph.add_relation(relation)

        edge = self.graph.add_edge(self.graph.nodes[src_id],  # Node #0 : hockey
                                   self.graph.nodes[tgt_id],  # Node #1 : play on ice
                                   self.graph.relations[relation_id],  # Relation #0 : ReceivesAction
                                   label,  # 1
                                   weight)  # 1.0

        # add nodes/relations from evaluation graphs to training graph too, 添加节点和关系到训练图中
        if train_network is not None and label == 1:
            src_id = train_network.graph.find_node(src)
            if src_id == -1:
                src_id = train_network.graph.add_node(src)

            tgt_id = train_network.graph.find_node(tgt)
            if tgt_id == -1:
                tgt_id = train_network.graph.add_node(tgt)

            relation_id = train_network.graph.find_relation(relation)
            if relation_id == -1:
                relation_id = train_network.graph.add_relation(relation)

        return edge  # 返回边，是一个元组


class AtomicReader(Reader):

    def __init__(self):

        logger.info("Reading ATOMIC corpus")
        self.graph = Graph()
        self.rel2id = {}

    def read_network(self, data_path, split="trn"):

        df = pd.read_csv(os.path.join(data_path, "atomic/v4_atomic_all.csv"), index_col=0)
        df.iloc[:, :9] = df.iloc[:, :9].apply(lambda col: col.apply(json.loads))

        for rel in df.columns[:9]:
            self.rel2id[rel] = len(self.rel2id)

        for index, row in df[df['split'] == split].iterrows():
            event = row.name
            for rel in self.rel2id:
                if row[rel] or row[rel] == ["none"]:
                    for inst in row[rel]:
                        self.add_example(event, inst, rel)

    def add_example(self, src, tgt, rel, label=1):

        start_id = self.graph.find_node(src)
        if start_id == -1:
            start_id = self.graph.add_node(src)

        end_id = self.graph.find_node(tgt)
        if end_id == -1:
            end_id = self.graph.add_node(tgt)

        rel_id = self.graph.find_relation(rel)
        if rel_id == -1:
            rel_id = self.graph.add_relation(rel)

        self.graph.add_edge(self.graph.nodes[start_id],
                            self.graph.nodes[end_id],
                            self.graph.relations[rel_id],
                            label,
                            1.0)


class AtomicTSVReader(Reader):

    def __init__(self, dataset):
        logger.info("Reading ATOMIC corpus in TSV format")
        self.dataset = dataset
        self.graph = Graph()
        self.rel2id = {}

    def read_network(self, data_dir, split="train", train_network=None):

        data_path = data_dir
        filename = split + ".preprocessed.txt"
        #filename = split + ".txt"

        with open(os.path.join(data_path, filename)) as f:
            data = f.readlines()

        for inst in data:
            inst = inst.strip()
            if inst:
                inst = inst.split('\t')
                if len(inst) == 3:
                    src, rel, tgt = inst
                    #src = reader_utils.preprocess_atomic_sentence(src).replace("-", " ")
                    #tgt = reader_utils.preprocess_atomic_sentence(tgt).replace("-", " ")
                    if split != "train":
                        self.add_example(src, tgt, rel, train_network=train_network)
                    else:
                        self.add_example(src, tgt, rel)

    def add_example(self, src, tgt, relation, weight=1.0, label=1, train_network=None):

        src_id = self.graph.find_node(src)
        if src_id == -1:
            src_id = self.graph.add_node(src)

        tgt_id = self.graph.find_node(tgt)
        if tgt_id == -1:
            tgt_id = self.graph.add_node(tgt)

        relation_id = self.graph.find_relation(relation)
        if relation_id == -1:
            relation_id = self.graph.add_relation(relation)

        edge = self.graph.add_edge(self.graph.nodes[src_id],
                                   self.graph.nodes[tgt_id],
                                   self.graph.relations[relation_id],
                                   label,
                                   weight)

        # add nodes/relations from evaluation graphs to training graph too
        if train_network is not None and label == 1:
            src_id = train_network.graph.find_node(src)
            if src_id == -1:
                src_id = train_network.graph.add_node(src)

            tgt_id = train_network.graph.find_node(tgt)
            if tgt_id == -1:
                tgt_id = train_network.graph.add_node(tgt)

            relation_id = train_network.graph.find_relation(relation)
            if relation_id == -1:
                relation_id = train_network.graph.add_relation(relation)

        return edge


class FB15kReader(Reader):

    def __init__(self, dataset):
        logger.info("Reading FB15K-237..")
        self.graph = Graph()
        self.rel2id = {}

    def read_network(self, data_dir, keep_fraction=100, split="train", train_network=None):

        data_path = data_dir
        if split == "train":
            filename = split + str(keep_fraction) + "p.txt"
        else:
            filename = split + ".txt"

        with open(os.path.join(data_path, filename)) as f:
            data = f.readlines()

        for inst in data:
            inst = inst.strip()
            if inst:
                inst = inst.split('\t')
                src, rel, tgt = inst
                src = src.lower()
                tgt = tgt.lower()
                if split != "train":
                    self.add_example(src, tgt, rel, train_network=train_network)
                else:
                    self.add_example(src, tgt, rel)

    def add_example(self, src, tgt, relation, weight=1.0, label=1, train_network=None):

        src_id = self.graph.find_node(src)
        if src_id == -1:
            src_id = self.graph.add_node(src)

        tgt_id = self.graph.find_node(tgt)
        if tgt_id == -1:
            tgt_id = self.graph.add_node(tgt)

        relation_id = self.graph.find_relation(relation)
        if relation_id == -1:
            relation_id = self.graph.add_relation(relation)

        edge = self.graph.add_edge(self.graph.nodes[src_id],
                            self.graph.nodes[tgt_id],
                            self.graph.relations[relation_id],
                            label,
                            weight)

        # add nodes/relations from evaluation graphs to training graph too
        if train_network is not None and label == 1:
            src_id = train_network.graph.find_node(src)
            if src_id == -1:
                src_id = train_network.graph.add_node(src)

            tgt_id = train_network.graph.find_node(tgt)
            if tgt_id == -1:
                tgt_id = train_network.graph.add_node(tgt)

            relation_id = train_network.graph.find_relation(relation)
            if relation_id == -1:
                relation_id = train_network.graph.add_relation(relation)

        return edge
