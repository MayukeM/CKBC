from tqdm import tqdm
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertForPreTraining, BertConfig, \
    BertForSequenceClassification

import os
import re

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import numpy as np

torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

"""
Feature Extractor for BERT
"""


class InputExample(object):
    """A single training/test example for simple sequence classification with BERT."""

    def __init__(self, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.text_a = text_a  # 'play on ice' 这是三元组中的一个实体
        self.text_b = text_b  # None
        self.label = label  # None


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id=None):
        self.input_ids = input_ids  # 输入的id, 也就是输入的句子
        self.input_mask = input_mask  # 输入的mask, 也就是输入的句子的mask, 用来区分padding的部分, 1表示不是padding, 0表示padding
        self.segment_ids = segment_ids  # 输入的segment id, 也就是输入的句子的segment id, 用来区分两个句子, 0表示第一个句子, 1表示第二个句子
        self.label_id = label_id


def convert_examples_to_features(examples, max_seq_length, tokenizer, label_list=None):
    """Loads a data file into a list of `InputBatch`s."""  # 将数据转换为InputBatch的形式，也就是将数据转换为id的形式

    if label_list:
        label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)  # 编码节点名称，tokenizer.tokenize()方法会将节点名称分割成一个个token
        # tokens_a = ['play', 'on', 'ice']
        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)  #
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)] # 如果节点名称的token数量大于max_seq_length-2，则截断

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]  # 输入序列 ['[CLS]', 'play', 'on', 'ice', '[SEP]']
        segment_ids = [0] * len(tokens)  # 输入序列的segment id, [0, 0, 0, 0, 0]

        if tokens_b:  # 如果有第二个句子
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)  # 转化成ID, [101, 2377, 2006, 3256, 102]
        # convert_tokens_to_ids()方法会将token转化成ID，这里的ID是BERT模型中的ID，不是节点ID，Bert模型中的ID是从0开始的连续整数,这里的ID是用来索引BERT模型中的词向量的
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)  # 输入序列的mask，用来区分padding的部分，1表示不是padding，0表示padding
        # [1, 1, 1, 1, 1]
        # Zero-pad up to the sequence length.  意思是将序列长度补齐到max_seq_length
        padding = [0] * (max_seq_length - len(input_ids))  # 长度25 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        input_ids += padding    # [101, 2377, 2006, 3256, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        input_mask += padding   # [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        segment_ids += padding  # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        assert len(input_ids) == max_seq_length  # 确保序列长度为max_seq_length，否则报错
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if example.label:
            label_id = label_map[example.label]
        else:
            label_id = None

        features.append(  # 将每个样本转化成一个InputFeatures对象
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features


def camel_case_split(identifier):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return (" ".join([m.group(0) for m in matches])).lower()


def convert_edges_to_examples(edges, labels, network):
    examples = []

    for i, edge in enumerate(edges):
        edge = edge.cpu().numpy()
        text_a = network.graph.nodes[edge[0]].name + " " + camel_case_split(
            network.graph.relations[edge[1]].name) + " " + network.graph.nodes[edge[2]].name
        label = labels[i].cpu().item()
        examples.append(
            InputExample(text_a=text_a, text_b=None, label=label))

    return examples


def convert_nodes_to_examples(node_list):
    examples = []  # 8442个 label text_a text_b

    for node in node_list:  # node：Node #1 : play on ice
        text_a = node.name  # 'play on ice'
        examples.append(
            InputExample(text_a=text_a))  # 节点的名称

    return examples


class BertLayer(nn.Module):  # BERT模型，用于节点的表示，输入节点的名称，输出节点的表示
    def __init__(self, dataset):
        super(BertLayer, self).__init__()
        self.dataset = dataset  # 数据集
        output_dir = "../data/ConceptNet/nodes-lm-conceptnet/small"  # BERT模型的路径，这里是预训练好的BERT模型

        self.filename = os.path.join(output_dir, self.dataset + "_bert_embeddings.pt")  # BERT模型的输出文件'../data/ConceptNet/nodes-lm-conceptnet/small/conceptnet_bert_embeddings.pt'
        print(self.filename)

        if os.path.isfile(self.filename):  # 如果已经存在预训练的节点表示，则直接加载
            self.exists = True
            return

        self.exists = False
        self.max_seq_length = 30
        self.eval_batch_size = 64  # 评估时的batch_size，这里是64，即每次评估64个节点
        self.tokenizer = BertTokenizer.from_pretrained('../data_bert/vocab.txt', do_lower_case=True)  # 之后做更改
        # output_model_file = os.path.join(output_dir, "lm_pytorch_model.bin")
        # output_model_file = os.path.join("bert_model_embeddings/nodes-lm-conceptnet/", "lm_pytorch_model.bin")
        output_model_file = '../data_bert/'
        print("Loading model from %s" % output_dir)  # '../data/ConceptNet/nodes-lm-conceptnet/small'
        # config = BertConfig.from_pretrained('data_bert/config.json')
        # self.bert_net = torch.load(output_model_file, map_location='cpu')
        self.bert_model = BertForSequenceClassification.from_pretrained(output_model_file, num_labels=2)
        # self.bert_model.load_state_dict(self.bert_net)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert_model.to(self.device)

        # Make BERT parameters non-trainable
        # bert_params = list(self.bert_model.parameters())
        # for param in bert_params:
        #     param.requires_grad = False

    def forward(self, node_list):
        #
        if self.exists:  # 如果已经存在，直接读取
            print("Loading BERT embeddings from disk..")
            # 先判断一下gpu是否可用
            if torch.cuda.is_available():  # 如果gpu可用
                return torch.load(self.filename)
            else:
                return torch.load(self.filename, map_location='cpu')
        # pycharm中把一句代码变成try except的方法是：选中代码，按ctrl+alt+t
        print("Computing BERT embeddings..")  # 否则，计算，然后保存
        self.bert_model.eval()  # 设置为评估模式，不进行梯度更新

        eval_examples = convert_nodes_to_examples(node_list)  # 列表中每个元素是一个节点Node类
        eval_features = convert_examples_to_features(  # 将节点转换为特征，即将节点的名称转换为BERT模型的输入，即将节点的名称转换为token
            eval_examples, max_seq_length=self.max_seq_length, tokenizer=self.tokenizer)  # 8442个节点，每个节点的名称转换为token

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)  # 【8442，30】
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)  # 【8442，30】
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)  # 【8442，30】
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)  # 将数据转换为TensorDataset

        # Run prediction for full data，使用DataLoader进行数据的批处理
        eval_sampler = SequentialSampler(eval_data)  # 顺序采样,按顺序采样,不打乱顺序
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=self.eval_batch_size)
        sequence_outputs = []
        idx = 0
        for input_ids, input_mask, segment_ids in tqdm(eval_dataloader):
            # print(idx)
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)

            with torch.no_grad():  # 不进行梯度更新, 意思是不进行反向传播，不更新参数，只是前向传播，得到结果，因为是评估模式，不需要更新参数
                # sequence_output, _ = self.bert_model.bert(input_ids, segment_ids, input_mask, output_all_encoded_layers=False)
                # print(input_ids)
                torch.cuda.empty_cache()  # 清空显存，防止显存溢出，这里的显存是指GPU的显存
                sequence_output = self.bert_model.bert(input_ids, segment_ids, input_mask)[0]  # [0]表示取第一个元素，即取最后一层的输出
                torch.cuda.empty_cache()  # 清空显存，防止显存溢出，这里的显存是指GPU的显存，这里的显存是指GPU的显存
            # print(sequence_output)
            sequence_outputs.append(sequence_output[:, 0])  # CLS的嵌入保存起来

            # if len(sequence_outputs) == 800:
            #    self.save_to_disk(torch.cat(sequence_outputs, dim=0), idx)
            #    sequence_outputs = []
            #    idx += 1

        self.save_to_disk(torch.cat(sequence_outputs, dim=0), idx)

        return torch.cat(sequence_outputs, dim=0)

    def forward_as_init(self, num_nodes, network=None):

        if self.exists:
            print("Loading BERT embeddings from disk..")
            return torch.load(self.filename)

        node_ids = np.arange(num_nodes)
        node_list = [network.graph.nodes[idx] for idx in node_ids]  # 节点的列表

        print("Computing BERT embeddings..")
        self.bert_model.eval()

        eval_examples = convert_nodes_to_examples(node_list)
        eval_features = convert_examples_to_features(
            eval_examples, max_seq_length=self.max_seq_length, tokenizer=self.tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)

        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=self.eval_batch_size)
        sequence_outputs = []

        for input_ids, input_mask, segment_ids in eval_dataloader:
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)

            with torch.no_grad():
                sequence_output, _ = self.bert_model.bert(input_ids, segment_ids, input_mask,
                                                          output_all_encoded_layers=False)
            sequence_outputs.append(sequence_output[:, 0])

        return torch.cat(sequence_outputs, dim=0)

    def save_to_disk(self, tensor, idx):
        # torch.save(tensor, self.dataset + str(idx) + "_bert_embeddings.pt")
        torch.save(tensor, self.dataset + "_bert_embeddingss.pt")

