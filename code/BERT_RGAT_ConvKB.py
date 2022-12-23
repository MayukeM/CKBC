import torch
# from models import SpKBGATModified, SpKBGATConvOnly
# import '/tmp/pycharm_project_14/model_bert'
from model_bert import SpKBGATModified, SpKBGATConvOnly
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
from copy import deepcopy
from create_batch import Corpus
import utils

import random
import argparse
from collections import Counter
from reader import ConceptNetTSVReader
import reader_utils
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "0"  # 设置cuda
import time  #
import sys
import logging
import pickle


# %%
# %%from torchviz import make_dot, make_dot_from_trace

def save_model(model, name, epoch, folder_name):
    print("Saving Model")
    torch.save(model.state_dict(),
               (folder_name + "trained_{}.pth").format(epoch))
    print("Done saving Model")


def parse_args():
    args = argparse.ArgumentParser()  # argparse是python自带的一个命令行参数解析模块,用于解析命令行参数
    # network arguments
    args.add_argument("-d", "--dataset", type=str, default="conceptnet",  # 数据集，这里是conceptnet
                      help="dataset to use")  # help是帮助信息
    args.add_argument("-data", "--data",
                      default="../data/ConceptNet/", help="data directory")  # 数据集的路径，这里是../data/ConceptNet/
    args.add_argument("-e_g", "--epochs_gat", type=int,
                      default=3000, help="Number of epochs")  # gat训练的轮数，这里是3000
    args.add_argument("-e_c", "--epochs_conv", type=int,
                      default=200, help="Number of epochs")  # conv训练的轮数，这里是200
    args.add_argument("-w_gat", "--weight_decay_gat", type=float,
                      default=0.00001, help="L2 reglarization for gat")  # gat的L2正则化系数，这里是0.00001
    args.add_argument("-w_conv", "--weight_decay_conv", type=float,
                      default=0.000, help="L2 reglarization for conv")  # 0.000001
    args.add_argument("-pre_emb", "--pretrained_emb", type=bool,
                      default=False, help="Use pretrained embeddings")
    args.add_argument("-emb_size", "--embedding_size", type=int,
                      default=200, help="Size of embeddings (if pretrained not used)")
    args.add_argument("-l", "--lr", type=float, default=1e-3)  # 学习率，这里是1e-3
    args.add_argument("-g2hop", "--get_2hop", type=bool, default=True)  # 是否使用2hop,这里是True,即使用,g2hop是get_2hop的缩写
    args.add_argument("-u2hop", "--use_2hop", type=bool, default=True)  # 是否使用2hop,这里是True,即使用,u2hop是use_2hop的缩写
    args.add_argument("-p2hop", "--partial_2hop", type=bool, default=True)  # 是否使用2hop,这里是True,即使用,p2hop是partial_2hop的缩写
    args.add_argument("-outfolder", "--output_folder",
                      default="../checkpoints/cn/out/", help="Folder name to save the models.")

    # arguments for GAT,GAT是Graph Attention Network的缩写，是一种图神经网络
    args.add_argument("-b_gat", "--batch_size_gat", type=int,
                      default=100000, help="Batch size for GAT")  # gat的batch_size，这里是100000
    args.add_argument("-neg_s_gat", "--valid_invalid_ratio_gat", type=int,
                      # 意思是valid和invalid的比例，这里是1:1，valid是正样本，invalid是负样本
                      default=2,
                      help="Ratio of valid to invalid triples for GAT training")  # gat的valid_invalid_ratio，这里是2
    args.add_argument("-drop_GAT", "--drop_GAT", type=float,  # gat的dropout，这里是0.3
                      default=0.3, help="Dropout probability for SpGAT layer")
    args.add_argument("-alpha", "--alpha", type=float,
                      default=0.2, help="LeakyRelu alphs for SpGAT layer")  # gat的alpha，这里是0.2
    args.add_argument("-out_dim", "--entity_out_dim", type=int, nargs='+',  # gat的entity_out_dim，这里是[200, 200]
                      default=[100, 200], help="Entity output embedding dimensions")
    args.add_argument("-h_gat", "--nheads_GAT", type=int, nargs='+',  # gat的nheads_GAT，这里是[1, 1]
                      default=[2, 2], help="Multihead attention SpGAT")
    args.add_argument("-margin", "--margin", type=float,  # gat的margin，这里是1
                      default=1, help="Margin used in hinge loss")

    # arguments for convolution network，conv是convolution的缩写，是一种卷积神经网络
    args.add_argument("-b_conv", "--batch_size_conv", type=int,
                      default=128, help="Batch size for conv")  # conv的batch_size，这里是128
    args.add_argument("-alpha_conv", "--alpha_conv", type=float,
                      default=0.2, help="LeakyRelu alphas for conv layer")  # conv的alpha，这里是0.2
    args.add_argument("-neg_s_conv", "--valid_invalid_ratio_conv", type=int, default=40,
                      help="Ratio of valid to invalid triples for convolution training")
    args.add_argument("-o", "--out_channels", type=int, default=5,
                      help="Number of output channels in conv layer")
    args.add_argument("-drop_conv", "--drop_conv", type=float,
                      default=0.3, help="Dropout probability for convolution layer")  # conv的dropout，这里是0.3

    args = args.parse_args()  # 解析参数,这里的args是一个对象，里面包含了上面的参数
    return args


args = parse_args()
print(args)


# %%

def build_data(dataset, reader_cls, data_dir, sim_relations):
    # 构建关于训练集验证集测试集的知识图谱
    train_network = reader_cls(dataset)  # 读取训练集
    dev_network = reader_cls(dataset)  # 读取验证集
    test_network = reader_cls(dataset)  # 读取测试集

    train_network.read_network(data_dir=data_dir, split="train")  # 读取训练集，这里的split是train，split是分割的意思，这里是分割训练集
    # 输出训练图谱的信息
    train_network.print_summary()
    node_list = train_network.graph.iter_nodes()
    node_degrees = [node.get_degree() for node in node_list]
    degree_counter = Counter(node_degrees)
    avg_degree = sum([k * v for k, v in degree_counter.items()]) / sum([v for k, v in degree_counter.items()])
    print("Average Degree: ", avg_degree)

    # 将测试图和验证图上的节点边加入到训练图中；只是将节点和关系加入到了其中，并没有增加新的边
    dev_network.read_network(data_dir=data_dir, split="valid", train_network=train_network)
    test_network.read_network(data_dir=data_dir, split="test", train_network=train_network)

    # 所有的节点ID
    entity2id = train_network.graph.node2id
    # 所有的关系ID
    relation2id = train_network.graph.relation2id
    unique_entities_train = train_network.unique_entities

    # Add sim nodes，添加相似节点
    if sim_relations:
        print("Adding sim edges..")
        train_network.add_sim_edges_bert()

    train_network.print_summary()
    # 输出图谱
    train_triples = reader_utils.get_triple(entity2id, train_network, train_network)
    test_triples = reader_utils.get_triple(entity2id, test_network, train_network)
    valid_triples = reader_utils.get_triple(entity2id, dev_network, train_network)

    # build adj list and calculate degrees for sampling，构建邻接表和计算度数用于采样
    train_adjacency_mat = utils.get_adj(train_triples)
    valid_adjacency_mat = utils.get_adj(test_triples)
    test_adjacency_mat = utils.get_adj(valid_triples)

    train_data = (train_triples, train_adjacency_mat)  # 训练集的数据，包括三元组和邻接矩阵
    valid_data = (valid_triples, valid_adjacency_mat)
    test_data = (test_triples, test_adjacency_mat)

    return train_data, valid_data, test_data, entity2id, relation2id, train_network, unique_entities_train


def load_data(args):  # 加载数据
    train_data, validation_data, test_data, entity2id, relation2id, train_network, unique_entities_train = build_data(
        "conceptnet", ConceptNetTSVReader, "../data/ConceptNet/", False)

    entity_embeddings = np.random.randn(
        len(entity2id), args.embedding_size)
    relation_embeddings = np.random.randn(
        len(relation2id), args.embedding_size)
    print("Initialised relations and entities randomly")

    corpus = Corpus(args, train_data, validation_data, test_data, entity2id, relation2id,
                    args.batch_size_gat, args.valid_invalid_ratio_gat, unique_entities_train, args.get_2hop)
    # corpus是一个类，里面包含了训练集，验证集，测试集，实体ID，关系ID，batch_size，valid_invalid_ratio，unique_entities_train，get_2hop
    return corpus, torch.FloatTensor(entity_embeddings), torch.FloatTensor(relation_embeddings), train_network


# 得到语料库，实体嵌入矩阵，关系嵌入矩阵，cor
Corpus_, entity_embeddings, relation_embeddings, train_network = load_data(args)
print('Initialised Successfully')

# 存储2跳邻居的信息
# if(args.get_2hop):
#    file = args.data + "/2hop.pickle"
#    with open(file, 'wb') as handle:
#        pickle.dump(Corpus_.node_neighbors_2hop, handle,
#                    protocol=pickle.HIGHEST_PROTOCOL)


# if(args.use_2hop):
#    print("Opening node_neighbors pickle object")
#    file = args.data + "/2hop.pickle"
#    with open(file, 'rb') as handle:
#        node_neighbors_2hop = pickle.load(handle) # 2跳邻居的信息

entity_embeddings_copied = deepcopy(entity_embeddings)
relation_embeddings_copied = deepcopy(relation_embeddings)

print("Initial entity dimensions {} , relation dimensions {}".format(
    entity_embeddings.size(), relation_embeddings.size()))  # FB：【14541，100】，【237，100】
# %%

CUDA = torch.cuda.is_available()


def batch_gat_loss(gat_loss_func, train_indices, entity_embed, relation_embed):
    len_pos_triples = int(
        train_indices.shape[0] / (int(args.valid_invalid_ratio_gat) + 1))

    pos_triples = train_indices[:len_pos_triples]
    neg_triples = train_indices[len_pos_triples:]

    pos_triples = pos_triples.repeat(int(args.valid_invalid_ratio_gat), 1)

    source_embeds = entity_embed[pos_triples[:, 0]]
    relation_embeds = relation_embed[pos_triples[:, 1]]
    tail_embeds = entity_embed[pos_triples[:, 2]]

    x = source_embeds + relation_embeds - tail_embeds
    pos_norm = torch.norm(x, p=1, dim=1)

    source_embeds = entity_embed[neg_triples[:, 0]]
    relation_embeds = relation_embed[neg_triples[:, 1]]
    tail_embeds = entity_embed[neg_triples[:, 2]]

    x = source_embeds + relation_embeds - tail_embeds
    neg_norm = torch.norm(x, p=1, dim=1)

    y = -torch.ones(int(args.valid_invalid_ratio_gat) * len_pos_triples).cuda()

    loss = gat_loss_func(pos_norm, neg_norm, y)
    return loss


def train_gat(args):
    # Creating the gat model here.
    ####################################

    print("Defining model")

    print(
        "\nModel type -> GAT layer with {} heads used , Initital Embeddings training".format(args.nheads_GAT[0]))
    model_gat = SpKBGATModified(entity_embeddings, relation_embeddings, args.entity_out_dim, args.entity_out_dim,
                                args.drop_GAT, args.alpha, args.nheads_GAT)

    if CUDA:
        model_gat.cuda()

    optimizer = torch.optim.Adam(
        model_gat.parameters(), lr=args.lr, weight_decay=args.weight_decay_gat)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=500, gamma=0.5, last_epoch=-1)

    # 评价相似度的损失
    gat_loss_func = nn.MarginRankingLoss(margin=args.margin)

    # 当前batch的2跳邻居的切片
    current_batch_2hop_indices = torch.tensor([])
    if (args.use_2hop):
        current_batch_2hop_indices = Corpus_.get_batch_nhop_neighbors_all(args,
                                                                          Corpus_.unique_entities_train,
                                                                          node_neighbors_2hop)

    if CUDA:
        current_batch_2hop_indices = Variable(
            torch.LongTensor(current_batch_2hop_indices)).cuda()
    else:
        current_batch_2hop_indices = Variable(
            torch.LongTensor(current_batch_2hop_indices))

    epoch_losses = []  # losses of all epochs
    print("Number of epochs {}".format(args.epochs_gat))

    for epoch in range(args.epochs_gat):
        print("\nepoch-> ", epoch)
        random.shuffle(Corpus_.train_triples)
        Corpus_.train_indices = np.array(
            list(Corpus_.train_triples)).astype(np.int32)

        model_gat.train()  # getting in training mode
        start_time = time.time()
        epoch_loss = []

        # 获取一个epoch里的迭代次数
        if len(Corpus_.train_indices) % args.batch_size_gat == 0:
            num_iters_per_epoch = len(
                Corpus_.train_indices) // args.batch_size_gat
        else:
            num_iters_per_epoch = (
                                          len(Corpus_.train_indices) // args.batch_size_gat) + 1

        for iters in range(num_iters_per_epoch):
            start_time_iter = time.time()
            train_indices, train_values = Corpus_.get_iteration_batch(iters)

            if CUDA:
                train_indices = Variable(
                    torch.LongTensor(train_indices)).cuda()
                train_values = Variable(torch.FloatTensor(train_values)).cuda()

            else:
                train_indices = Variable(torch.LongTensor(train_indices))
                train_values = Variable(torch.FloatTensor(train_values))

            # forward pass
            entity_embed, relation_embed = model_gat(
                Corpus_, Corpus_.train_adj_matrix, train_indices, current_batch_2hop_indices)

            optimizer.zero_grad()

            loss = batch_gat_loss(
                gat_loss_func, train_indices, entity_embed, relation_embed)

            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.data.item())

            end_time_iter = time.time()

            print("Iteration-> {0}  , Iteration_time-> {1:.4f} , Iteration_loss {2:.4f}".format(
                iters, end_time_iter - start_time_iter, loss.data.item()))

        scheduler.step()
        print("Epoch {} , average loss {} , epoch_time {}".format(
            epoch, sum(epoch_loss) / len(epoch_loss), time.time() - start_time))
        epoch_losses.append(sum(epoch_loss) / len(epoch_loss))

        if (epoch + 1) % 300 == 0:
            save_model(model_gat, args.data, epoch,
                       args.output_folder)


def train_conv(args):
    # Creating convolution model here.
    ####################################

    print("Defining model")
    model_gat = SpKBGATModified(entity_embeddings, relation_embeddings, args.entity_out_dim, args.entity_out_dim,
                                args.drop_GAT, args.alpha, args.nheads_GAT)
    print("Only Conv model trained")
    model_conv = SpKBGATConvOnly(entity_embeddings, relation_embeddings, args.entity_out_dim, args.entity_out_dim,
                                 args.drop_GAT, args.drop_conv, args.alpha, args.alpha_conv,
                                 args.nheads_GAT, args.out_channels)

    if CUDA:
        model_conv.cuda()
        model_gat.cuda()

    model_gat.load_state_dict(torch.load(
        '{}/trained_{}.pth'.format(args.output_folder, args.epochs_gat - 1)), strict=False)
    model_conv.final_entity_embeddings = model_gat.final_entity_embeddings
    model_conv.final_relation_embeddings = model_gat.final_relation_embeddings

    Corpus_.batch_size = args.batch_size_conv
    Corpus_.invalid_valid_ratio = int(args.valid_invalid_ratio_conv)

    optimizer = torch.optim.Adam(
        model_conv.parameters(), lr=args.lr, weight_decay=args.weight_decay_conv)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=25, gamma=0.5, last_epoch=-1)

    margin_loss = torch.nn.SoftMarginLoss()

    epoch_losses = []  # losses of all epochs
    print("Number of epochs {}".format(args.epochs_conv))

    for epoch in range(args.epochs_conv):
        print("\nepoch-> ", epoch)
        random.shuffle(Corpus_.train_triples)
        Corpus_.train_indices = np.array(
            list(Corpus_.train_triples)).astype(np.int32)

        model_conv.train()  # getting in training mode
        start_time = time.time()
        epoch_loss = []

        if len(Corpus_.train_indices) % args.batch_size_conv == 0:
            num_iters_per_epoch = len(
                Corpus_.train_indices) // args.batch_size_conv
        else:
            num_iters_per_epoch = (
                                          len(Corpus_.train_indices) // args.batch_size_conv) + 1

        for iters in range(num_iters_per_epoch):
            start_time_iter = time.time()
            train_indices, train_values = Corpus_.get_iteration_batch(iters)

            if CUDA:
                train_indices = Variable(
                    torch.LongTensor(train_indices)).cuda()
                train_values = Variable(torch.FloatTensor(train_values)).cuda()

            else:
                train_indices = Variable(torch.LongTensor(train_indices))
                train_values = Variable(torch.FloatTensor(train_values))

            preds = model_conv(
                Corpus_, Corpus_.train_adj_matrix, train_indices)

            optimizer.zero_grad()

            # print(train_values) # 1和-1
            loss = margin_loss(preds.view(-1), train_values.view(-1))

            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.data.item())

            end_time_iter = time.time()

            print("Iteration-> {0}  , Iteration_time-> {1:.4f} , Iteration_loss {2:.4f}".format(
                iters, end_time_iter - start_time_iter, loss.data.item()))

        scheduler.step()
        print("Epoch {} , average loss {} , epoch_time {}".format(
            epoch, sum(epoch_loss) / len(epoch_loss), time.time() - start_time))
        epoch_losses.append(sum(epoch_loss) / len(epoch_loss))

        if (epoch + 1) % 20 == 0:
            save_model(model_conv, args.data, epoch,
                       args.output_folder + "conv/")


def evaluate_conv(args, unique_entities, train_network):
    model_conv = SpKBGATConvOnly(entity_embeddings, relation_embeddings, args.entity_out_dim, args.entity_out_dim,
                                 args.drop_GAT, args.drop_conv, args.alpha, args.alpha_conv,
                                 args.nheads_GAT, args.out_channels)
    model_conv.load_state_dict(torch.load(
        '{0}conv/trained_{1}.pth'.format(args.output_folder, args.epochs_conv - 1)), strict=False)

    model_conv.cuda()
    model_conv.eval()
    with torch.no_grad():
        Corpus_.get_validation_pred_new(args, model_conv, unique_entities, train_network)


# train_gat(args)
# train_conv(args)
evaluate_conv(args, Corpus_.unique_entities_train, train_network)
