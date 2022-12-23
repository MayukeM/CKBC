# 我的尝试
模型权重链接：
https://pan.baidu.com/s/1Hjn0qmLnN_NH41KVf46SKg%EF%BC%8C%E6%8F%90%E5%8F%96%E7%A0%81%EF%BC%9AGt63#list/path=%2Fnodes-lm-conceptnet
nodes-lm-conceptnet/lm_pytorch_model.bin and conceptnet_bert_embeddings.pt

data_bert finetune的模型是（BERT-Large, Uncased）
https://huggingface.co/bert-large-uncased/tree/main

添加致密化后，用余弦相似度计算两个词向量的相似度，相似度大于一定阈值的会添加一条边，关系类型为sim
但是可以看到下面打印的一条结果，他把图中已有的边也更改了，这样是不是不太好
而且我哪怕用了更小的数据集（就是把数据只留10000一万条测试集，测试和验证都是800）并且把致密化的阈值提高到
0.99，仍然添加了977779九十七万条边呀，这计算相似度的方法是不是有问题

把阈值改成0.999，尽量少的添加边，第一次跑通结果：
（autodl服务器主机端口53308）
```
Current iteration time 22.607972145080566
Stats for replacing head are -> 
Current iteration Hits@100 are 0.6107491856677525
Current iteration Hits@10 are 0.3306188925081433
Current iteration Hits@3 are 0.21172638436482086
Current iteration Hits@1 are 0.11074918566775244
Current iteration Mean rank 375.64657980456025
Current iteration Mean Reciprocal Rank 0.1845679635936395

Stats for replacing tail are -> 
Current iteration Hits@100 are 0.6758957654723127
Current iteration Hits@10 are 0.36807817589576547
Current iteration Hits@3 are 0.20521172638436483
Current iteration Hits@1 are 0.09446254071661238
Current iteration Mean rank 388.442996742671
Current iteration Mean Reciprocal Rank 0.1820161572740297

Averaged stats for replacing head are -> 
Hits@100 are 0.6107491856677525
Hits@10 are 0.3306188925081433
Hits@3 are 0.21172638436482086
Hits@1 are 0.11074918566775244
Mean rank 375.64657980456025
Mean Reciprocal Rank 0.1845679635936395

Averaged stats for replacing tail are -> 
Hits@100 are 0.6758957654723127
Hits@10 are 0.36807817589576547
Hits@3 are 0.20521172638436483
Hits@1 are 0.09446254071661238
Mean rank 388.442996742671
Mean Reciprocal Rank 0.1820161572740297

Cumulative stats are -> 
Hits@100 are 0.6433224755700326
Hits@10 are 0.3493485342019544
Hits@3 are 0.20846905537459284
Hits@1 are 0.1026058631921824
Mean rank 382.0447882736156
Mean Reciprocal Rank 0.1832920604338346
```
IsA: hockey --> game

sim: hockey --> game

# CKBC_Model

### Dataset
The ConceptNet datasets are stored in the data folder, and the training, test, and validation sets are train.txt, test.txt, and dev.txt, respectively. the fine-tuned trained BERT model weights from the paper are stored in the [link](https://pan.baidu.com/s/19hYHzU3J336DHCdlvZ8QUQ)(password: bs45), and the folder in which the link is downloaded should be placed in the ConceptNet folder.

### Training

**Parameters:**

`--epochs_gat`: Number of epochs for gat training.

`--epochs_conv`: Number of epochs for convolution training.

`--lr`: Initial learning rate.

`--weight_decay_gat`: L2 reglarization for gat.

`--weight_decay_conv`: L2 reglarization for conv.

`--get_2hop`: Get a pickle object of 2 hop neighbors.

`--use_2hop`: Use 2 hop neighbors for training.  

`--partial_2hop`: Use only 1 2-hop neighbor per node for training.

`--output_folder`: Path of output folder for saving models.

`--batch_size_gat`: Batch size for gat model.

`--valid_invalid_ratio_gat`: Ratio of valid to invalid triples for GAT training.

`--drop_gat`: Dropout probability for attention layer.

`--alpha`: LeakyRelu alphas for attention layer.

`--nhead_GAT`: Number of heads for multihead attention.

`--margin`: Margin used in hinge loss.

`--batch_size_conv`: Batch size for convolution model.

`--alpha_conv`: LeakyRelu alphas for conv layer.

`--valid_invalid_ratio_conv`: Ratio of valid to invalid triples for conv training.

`--out_channels`: Number of output channels in conv layer.

`--drop_conv`: Dropout probability for conv layer.


The specific value settings for all parameters are included in the code

### Reproducing results

To reproduce the results published in the paper:      

        $ python code/SIM_BERT_RGAT_ConvKB.py

# 报错
Traceback (most recent call last):
  File "/home/CKBC/code/BERT_RGAT_ConvKB.py", line 431, in <module>
    evaluate_conv(args, Corpus_.unique_entities_train, train_network)
  File "/home/CKBC/code/BERT_RGAT_ConvKB.py", line 417, in evaluate_conv
    model_conv = SpKBGATConvOnly(entity_embeddings, relation_embeddings, args.entity_out_dim, args.entity_out_dim,
TypeError: __init__() missing 1 required positional argument: 'conv_out_channels'
