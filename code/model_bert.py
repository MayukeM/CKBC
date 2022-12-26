import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time
from torch.nn.init import xavier_normal_
from layers import SpGraphAttentionLayer, ConvKB_Bert
from bert_feature_extractor import BertLayer

CUDA = torch.cuda.is_available()  # checking cuda availability

class SpGAT(nn.Module):  # SpGAT class, 稀疏图注意力网络，继承自nn.Module
    def __init__(self, num_nodes, nfeat, nhid, relation_dim, dropout, alpha, nheads):
        """
            Sparse version of GAT，GAT的稀疏版本
            nfeat -> Entity Input Embedding dimensions，实体输入嵌入维度
            nhid  -> Entity Output Embedding dimensions，实体输出嵌入维度
            relation_dim -> Relation Embedding dimensions，关系嵌入维度
            num_nodes -> number of nodes in the Graph，图中节点的数量
            nheads -> Used for Multihead attention，用于多头注意力

        """
        super(SpGAT, self).__init__()  # 调用父类的构造函数
        self.dropout = dropout  # dropout，丢弃率，防止过拟合，一般在0.5-0.8之间
        self.dropout_layer = nn.Dropout(self.dropout)  # dropout层，防止过拟合
        self.attentions = [SpGraphAttentionLayer(num_nodes, nfeat,  # num_nodes:节点数，nfeat:输入维度
                                                 nhid,  # nhid:输出维度100
                                                 relation_dim,  # relation_dim:关系维度200
                                                 dropout=dropout,  # dropout:丢弃率0.3
                                                 alpha=alpha,  # alpha:缩放因子0.2
                                                 concat=True)  # concat:是否拼接
                           for _ in range(nheads)]  # nheads:多头注意力的头数 2

        for i, attention in enumerate(self.attentions):  #[SpGraphAttentionLayer (200 -> 100), SpGraphAttentionLayer (200 -> 100)]
            self.add_module('attention_{}'.format(i), attention)  # 添加attention层, 用于多头注意力

        # W matrix to convert h_input to h_output dimension
        self.W = nn.Parameter(torch.zeros(size=(relation_dim, nheads * nhid)))  # W矩阵[200,200]，用于将h_input转换为h_output维度
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  # 初始化W矩阵，均匀分布，gain=1.414

        self.out_att = SpGraphAttentionLayer(num_nodes, nhid * nheads,  #8442, 200*2
                                             nheads * nhid, nheads * nhid, #2*100, 2*100
                                             dropout=dropout,
                                             alpha=alpha,
                                             concat=False
                                             )

    def forward(self, Corpus_, batch_inputs, entity_embeddings, relation_embed,
                edge_list, edge_type, edge_embed, edge_list_nhop, edge_type_nhop):
        x = entity_embeddings

        edge_embed_nhop = relation_embed[
            edge_type_nhop[:, 0]] + relation_embed[edge_type_nhop[:, 1]]

        x = torch.cat([att(x, edge_list, edge_embed, edge_list_nhop, edge_embed_nhop)
                       for att in self.attentions], dim=1)
        x = self.dropout_layer(x)

        out_relation_1 = relation_embed.mm(self.W)

        edge_embed = out_relation_1[edge_type]
        edge_embed_nhop = out_relation_1[
            edge_type_nhop[:, 0]] + out_relation_1[edge_type_nhop[:, 1]]

        x = F.elu(self.out_att(x, edge_list, edge_embed,
                               edge_list_nhop, edge_embed_nhop))
        return x, out_relation_1


class SpKBGATModified(nn.Module):  # SpKBGATModified class, 稀疏图注意力网络，继承自nn.Module
    def __init__(self, initial_entity_emb, initial_relation_emb, entity_out_dim, relation_out_dim,
                 drop_GAT, alpha, nheads_GAT):
        '''Sparse version of KBGAT
        entity_in_dim -> Entity Input Embedding dimensions，实体的输入维度
        entity_out_dim  -> Entity Output Embedding dimensions, passed as a list，实体的输出维度
        num_relation -> number of unique relations，关系的数量
        relation_dim -> Relation Embedding dimensions，关系的维度
        num_nodes -> number of nodes in the Graph，图中的节点数量
        nheads_GAT -> Used for Multihead attention, passed as a list，用于多头注意力的头数
        '''

        super().__init__()  # 调用父类的构造函数

        self.num_nodes = initial_entity_emb.shape[0]  # number of nodes in the Graph，图中的节点数量8442
        self.entity_in_dim = initial_entity_emb.shape[1]  # Entity Input Embedding dimensions，实体的输入维度200
        self.entity_out_dim_1 = entity_out_dim[0]  # Entity Output Embedding dimensions，实体的输出维度100
        self.nheads_GAT_1 = nheads_GAT[0]  # Used for Multihead attention, passed as a list，用于多头注意力的头数2
        self.entity_out_dim_2 = entity_out_dim[1]  # Entity Output Embedding dimensions，实体的输出维度200
        self.nheads_GAT_2 = nheads_GAT[1]  # Used for Multihead attention, passed as a list，用于多头注意力的头数2

        # Properties of Relations  关系的属性
        self.num_relation = initial_relation_emb.shape[0]  # number of unique relations，关系的数量29
        self.relation_dim = initial_relation_emb.shape[1]  # Relation Embedding dimensions，关系的维度200
        self.relation_out_dim_1 = relation_out_dim[0]  # Relation Output Embedding dimensions，关系的输出维度100

        self.drop_GAT = drop_GAT  # dropout rate for GAT layers，GAT层的dropout率0.3
        self.alpha = alpha      # For leaky relu，用于leaky relu，leaky relu是一种激活函数，用于解决relu激活函数的一些问题

        self.final_entity_embeddings = nn.Parameter(  # Final Entity Embeddings，最终的实体嵌入
            torch.randn(self.num_nodes, self.entity_out_dim_1 * self.nheads_GAT_1))  # 用于存储最终的实体嵌入【8442，100*2】

        self.final_relation_embeddings = nn.Parameter(  # Final Relation Embeddings，最终的关系嵌入，nn.Parameter()是一个tensor，但是会被自动添加到模型的参数列表中
            torch.randn(self.num_relation, self.entity_out_dim_1 * self.nheads_GAT_1))  # 用于存储最终的关系嵌入【29，100*2】

        self.entity_embeddings = nn.Parameter(initial_entity_emb)  # Entity Embeddings，实体嵌入
        self.relation_embeddings = nn.Parameter(initial_relation_emb)  # Relation Embeddings，关系嵌入

        self.sparse_gat_1 = SpGAT(self.num_nodes, self.entity_in_dim, self.entity_out_dim_1, self.relation_dim,  # GAT Layer 1，GAT层1
                                  self.drop_GAT, self.alpha, self.nheads_GAT_1)  # GAT Layer 1，GAT层1

        self.W_entities = nn.Parameter(torch.zeros(
            size=(self.entity_in_dim, self.entity_out_dim_1 * self.nheads_GAT_1)))
        nn.init.xavier_uniform_(self.W_entities.data, gain=1.414)  # 初始化实体嵌入矩阵xavier_uniform_()是xavier初始化的一种，gain是缩放因子

    def forward(self, Corpus_, adj, batch_inputs, train_indices_nhop):
        # getting edge list
        edge_list = adj[0]
        edge_type = adj[1]

        edge_list_nhop = torch.cat(
            (train_indices_nhop[:, 3].unsqueeze(-1), train_indices_nhop[:, 0].unsqueeze(-1)), dim=1).t()
        edge_type_nhop = torch.cat(
            [train_indices_nhop[:, 1].unsqueeze(-1), train_indices_nhop[:, 2].unsqueeze(-1)], dim=1)

        if(CUDA):
            edge_list = edge_list.cuda()
            edge_type = edge_type.cuda()
            edge_list_nhop = edge_list_nhop.cuda()
            edge_type_nhop = edge_type_nhop.cuda()

        edge_embed = self.relation_embeddings[edge_type]

        start = time.time()

        self.entity_embeddings.data = F.normalize(
            self.entity_embeddings.data, p=2, dim=1).detach()

        # self.relation_embeddings.data = F.normalize(
        #     self.relation_embeddings.data, p=2, dim=1)

        out_entity_1, out_relation_1 = self.sparse_gat_1(
            Corpus_, batch_inputs, self.entity_embeddings, self.relation_embeddings,
            edge_list, edge_type, edge_embed, edge_list_nhop, edge_type_nhop)

        mask_indices = torch.unique(batch_inputs[:, 2]).cuda()
        mask = torch.zeros(self.entity_embeddings.shape[0]).cuda()
        mask[mask_indices] = 1.0

        entities_upgraded = self.entity_embeddings.mm(self.W_entities)
        out_entity_1 = entities_upgraded + \
            mask.unsqueeze(-1).expand_as(out_entity_1) * out_entity_1

        out_entity_1 = F.normalize(out_entity_1, p=2, dim=1)

        self.final_entity_embeddings.data = out_entity_1.data
        self.final_relation_embeddings.data = out_relation_1.data

        return out_entity_1, out_relation_1


class SpKBGATConvOnly(nn.Module):  # 稀疏的KBGATConvOnly模型
    def __init__(self, bert_model, initial_entity_emb, initial_relation_emb, entity_out_dim, relation_out_dim,
                 drop_GAT, drop_conv, alpha, alpha_conv, nheads_GAT, conv_out_channels):
        '''Sparse version of KBGAT
        entity_in_dim -> Entity Input Embedding dimensions
        entity_out_dim  -> Entity Output Embedding dimensions, passed as a list
        num_relation -> number of unique relations
        relation_dim -> Relation Embedding dimensions
        num_nodes -> number of nodes in the Graph
        nheads_GAT -> Used for Multihead attention, passed as a list '''

        super().__init__()

        self.bert_model = bert_model
        self.num_nodes = initial_entity_emb.shape[0]
        self.entity_in_dim = initial_entity_emb.shape[1]
        self.entity_out_dim_1 = entity_out_dim[0]
        self.nheads_GAT_1 = nheads_GAT[0]
        self.entity_out_dim_2 = entity_out_dim[1]
        self.nheads_GAT_2 = nheads_GAT[1]

        # Properties of Relations
        self.num_relation = initial_relation_emb.shape[0]
        self.relation_dim = initial_relation_emb.shape[1]
        self.relation_out_dim_1 = relation_out_dim[0]

        self.drop_GAT = drop_GAT
        self.drop_conv = drop_conv
        self.alpha = alpha      # For leaky relu
        self.alpha_conv = alpha_conv
        self.conv_out_channels = conv_out_channels

        self.bert_dim = 1024

        self.final_entity_embeddings = nn.Parameter(
            torch.randn(self.num_nodes, self.entity_out_dim_1 * self.nheads_GAT_1))

        self.final_relation_embeddings = nn.Parameter(
            torch.randn(self.num_relation, self.entity_out_dim_1 * self.nheads_GAT_1))

        self.convKB = ConvKB_Bert(self.entity_out_dim_1 * self.nheads_GAT_1 + self.bert_dim, 3, 1,
                             self.conv_out_channels, self.drop_conv, self.alpha_conv, self.num_relation)

        self.bert_concat_layer = EmbeddingLayer(self.bert_model, self.num_nodes, self.bert_dim, "conceptnet", init_bert=True)


    def forward(self, Corpus_, adj, batch_inputs):
        head_bert_embs = self.bert_concat_layer.embedding(batch_inputs[:, 0])
        tail_bert_embs = self.bert_concat_layer.embedding(batch_inputs[:, 2])

        #if epoch is not None:
        #    head_bert_embs = self.mask_by_schedule(head_bert_embs, epoch)
        #    tail_bert_embs = self.mask_by_schedule(tail_bert_embs, epoch)

        head_entity_embs = torch.cat([self.final_entity_embeddings[batch_inputs[:, 0], :], head_bert_embs], dim=1)
        tail_entity_embs = torch.cat([self.final_entity_embeddings[batch_inputs[:, 2], :], tail_bert_embs], dim=1)
        out_conv = self.convKB(head_entity_embs, tail_entity_embs, self.final_relation_embeddings, batch_inputs)
        return out_conv

    def batch_test(self, batch_inputs):
        batch_inputs = torch.LongTensor(batch_inputs).cuda()
        head_bert = self.bert_concat_layer.embedding(batch_inputs[:, 0])
        tail_bert = self.bert_concat_layer.embedding(batch_inputs[:, 2])

        head_entity_embs = torch.cat([self.final_entity_embeddings[batch_inputs[:, 0], :], head_bert], dim=1)
        tail_entity_embs = torch.cat([self.final_entity_embeddings[batch_inputs[:, 2], :], tail_bert], dim=1)
        #del head_bert
        #del tail_bert
        out_conv = self.convKB(head_entity_embs, tail_entity_embs, self.final_relation_embeddings, batch_inputs)
        return out_conv

    def mask_by_schedule(self, tensor, epoch, epoch_cutoff=100):
        if epoch < epoch_cutoff:
            cuda_check = tensor.is_cuda

            if cuda_check:
                mask = torch.zeros((tensor.size(0), tensor.size(1)), device='cuda')
            else:
                mask = torch.zeros((tensor.size(0), tensor.size(1)))

            k = int((epoch / epoch_cutoff) * tensor.size(1))
            perm = torch.randperm(tensor.size(1))
            indices = perm[:k]
            mask[:, indices] = 1
            return tensor * mask
        else:
            return tensor


class EmbeddingLayer(nn.Module):
    def __init__(self, bert_model, num_nodes, h_dim, dataset=None, init_bert=None):
        super(EmbeddingLayer, self).__init__()
        self.embedding = torch.nn.Embedding(num_nodes, h_dim, padding_idx=0)
        if not init_bert:
            xavier_normal_(self.embedding.weight.data)
        else:
            self.init_with_bert(bert_model, num_nodes)

    def forward(self):
        pass

    def init_with_bert(self, bert_model, num_nodes):
        bert_weights = bert_model.forward_as_init(num_nodes)
        self.embedding.load_state_dict({'weight': bert_weights})
        # self.embedding.weight.requires_grad = False