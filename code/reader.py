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
        print("\nNodes: %d" % len(self.graph.nodes))
        print("Edges: %d" % self.graph.edgeCount)
        print("Relations: %d" % len(self.graph.relation2id))
        density = self.graph.edgeCount / (len(self.graph.nodes) * (len(self.graph.nodes) - 1))  # density是边的数量除以节点的数量
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
        vecs = vecs.cpu().numpy()

        # vecs = np.vstack(vecs)
        print("Computed embeddings.")

        batch_size = 1000
        out_sims = []  # 用来存储相似度的
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

        print("Added %d sim edges" % sim_counter)


class ConceptNetTSVReader(Reader):  # 读取conceptnet的数据集

    def __init__(self, dataset):
        logger.info("Reading ConceptNet")  # 读取conceptnet, 生成图，读取概念网络
        self.dataset = dataset  # dataset name
        self.graph = Graph()  # graph object
        self.rel2id = {}  # relation to id
        self.unique_entities = set()  # 是一个集合，存储所有的实体，不重复

    def read_network(self, data_dir, split="train", train_network=None):

        if split == "train":
            data_path = os.path.join(data_dir, "train.txt")
        elif split == "valid":
            data_path = os.path.join(data_dir, "valid.txt")
        elif split == "test":
            data_path = os.path.join(data_dir, "test.txt")

        with open(data_path) as f:
            data = f.readlines()

        if split == "test":
            data = data[:1200]  # 只取前1200个

        for inst in data:
            inst = inst.strip()  # 去掉首尾的空格
            if inst:  # 如果不是空的，就进行处理
                inst = inst.split('\t')
                rel, src, tgt = inst
                weight = 1.0
                src = src.lower()  # 小写
                tgt = tgt.lower()
                self.unique_entities.add(src)  # 添加实体
                self.unique_entities.add(tgt)
                if split != "train":
                    self.add_example(src, tgt, rel, float(weight), int(weight), train_network)
                else:
                    self.add_example(src, tgt, rel, float(weight))  # 添加实例，添加到图中

        self.rel2id = self.graph.relation2id

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

        edge = self.graph.add_edge(self.graph.nodes[src_id],
                                   self.graph.nodes[tgt_id],
                                   self.graph.relations[relation_id],
                                   label,
                                   weight)

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
