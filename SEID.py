# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn.inits import glorot, zeros
from torch.nn import Parameter, Linear, Sequential, BatchNorm1d, ReLU
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split, Subset, Dataset
import time
import math
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler


import sys
from datetime import datetime
import os
import argparse
from pathlib import Path
import json
import random


class OutLayer(nn.Module):
    def __init__(self, in_num, node_num, layer_num, inter_num=512, input_dim=10):
        super(OutLayer, self).__init__()

        modules = []

        for i in range(layer_num):
            # last layer, output shape:1
            if i == layer_num - 1:
                modules.append(nn.Linear(in_num if layer_num == 1 else inter_num, input_dim))
            else:
                layer_in_num = in_num if i == 0 else inter_num
                modules.append(nn.Linear(layer_in_num, inter_num))
                modules.append(nn.BatchNorm1d(inter_num))
                modules.append(nn.ReLU())

        self.mlp = nn.ModuleList(modules)

    def forward(self, x):
        out = x
        # [batch_size, node_num, feature_dim=embed_dim]
        for mod in self.mlp:
            if isinstance(mod, nn.BatchNorm1d):
                out = out.permute(0, 2, 1)
                out = mod(out)
                out = out.permute(0, 2, 1)
            else:
                out = mod(out)

        return out


class GraphLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1, concat=True,
                 negative_slope=0.2, dropout=0, bias=True, inter_dim=-1, **kwargs):
        super(GraphLayer, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat  # +
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.__alpha__ = None  # +

        self.lin = Linear(in_channels, heads * out_channels, bias=False)

        self.att_i = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_j = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_em_i = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_em_j = Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))   #  √
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin.weight)
        glorot(self.att_i)
        glorot(self.att_j)

        zeros(self.att_em_i)
        zeros(self.att_em_j)

        zeros(self.bias)

    def forward(self, x, edge_index, embedding, return_attention_weights=False):
        """"""
        # [bsz*num_nodes, inchannels=window_size]👉[bsz*num_nodes, heads * out_channels]
        if torch.is_tensor(x):
            x = self.lin(x)
            x = (x, x)
        else:
            x = (self.lin(x[0]), self.lin(x[1]))

        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index,
                                       num_nodes=x[1].size(self.node_dim))

        out = self.propagate(edge_index, x=x, embedding=embedding, edges=edge_index,
                             return_attention_weights=return_attention_weights,
                             )
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)

        else:
            out = out.mean(dim=1) 

        if self.bias is not None:
            out = out + self.bias

        if return_attention_weights:
            alpha, self.__alpha__ = self.__alpha__, None
            return out, (edge_index, alpha)  
        else:
            return out  

    def message(self, x_i, x_j, edge_index_i, size_i,
                embedding,
                edges,
                return_attention_weights):

        x_i = x_i.view(-1, self.heads, self.out_channels)
        x_j = x_j.view(-1, self.heads, self.out_channels)

        if embedding is not None: 
            # [E, embedding_dim]👉[E, heads, embed_dim]
            embedding_i, embedding_j = embedding[edge_index_i], embedding[edges[0]]
            embedding_i = embedding_i.unsqueeze(1).repeat(1, self.heads, 1)
            embedding_j = embedding_j.unsqueeze(1).repeat(1, self.heads, 1)

            # [E, heads, embed_dim]+[E, heads, out_channels]=[E, heads, 2*out_channels]
            key_i = torch.cat((x_i, embedding_i), dim=-1)
            key_j = torch.cat((x_j, embedding_j), dim=-1)

        # [1, heads, out_channels]👉[1, heads, 2 * out_channels]
        cat_att_i = torch.cat((self.att_i, self.att_em_i), dim=-1)
        cat_att_j = torch.cat((self.att_j, self.att_em_j), dim=-1)

        # [E, heads, 2 * out_channels]*[1, heads, 2 * out_channels]👉[E, heads]
        alpha = (key_i * cat_att_i).sum(-1) + (key_j * cat_att_j).sum(-1)
        # [E, heads]👉[E, heads, 1]
        alpha = alpha.view(-1, self.heads, 1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, size_i)

        if return_attention_weights:
            self.__alpha__ = alpha

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        # [E, heads, out_channels]*[E, heads, 1]👉[E, heads, out_channels]
        return x_j * alpha.view(-1, self.heads, 1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


class GNNLayer(nn.Module):
    def __init__(self,
                 in_channel,   # 10
                 out_channel,  # dim = embed_dim = 64
                 inter_dim=0,  # 64+64
                 heads=1,   # 1
                 ):
        super(GNNLayer, self).__init__()
        self.gnn = GraphLayer(in_channel, out_channel, inter_dim=inter_dim, heads=heads, concat=False)
        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x, edge_index, embedding=None, node_num=0):
        out, (new_edge_index, att_weight) = self.gnn(x, edge_index, embedding, return_attention_weights=True)
        self.att_weight_1 = att_weight
        self.edge_index_1 = new_edge_index

        out = self.bn(out)

        return self.relu(out)


def get_batch_edge_index(org_edge_index, batch_num, node_num):
    # org_edge_index:(2, edge_num)
    edge_index = org_edge_index.clone().detach()
    edge_num = org_edge_index.shape[1]
    batch_edge_index = edge_index.repeat(1, batch_num).contiguous()

    for i in range(batch_num):
        batch_edge_index[:, i * edge_num:(i + 1) * edge_num] += i * node_num

    return batch_edge_index.long()


class GDN(nn.Module):
    def __init__(self,
                 edge_index_sets,
                 node_num,
                 dim=64,  
                 input_dim=10,  
                 out_layer_num=1,
                 out_layer_inter_dim=256,
                 topk=20,
                 transformer_head=4,
                 transformer_encoder_layers_num=6,
                 classes_num=5,
                 alpha=0.1,
                 batch_size=64,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(GDN, self).__init__()
        self.edge_index_sets = edge_index_sets
        embed_dim = dim
        self.embedding = nn.Embedding(node_num, embed_dim)
        self.bn_outlayer_in = nn.BatchNorm1d(embed_dim)  # 批归一化
        edge_set_num = len(edge_index_sets)
        self.gnn_layers = nn.ModuleList([
            GNNLayer(input_dim, dim, inter_dim=dim + embed_dim, heads=2) for j in range(edge_set_num)
        ])  # [E, heads, out_channels]
        self.node_embedding = None
        self.topk = topk
        self.learned_graph = None
        self.out_layer = OutLayer(dim * edge_set_num, node_num, out_layer_num, inter_num=out_layer_inter_dim,
                                  input_dim=input_dim)

        self.cache_edge_index_sets = [None] * edge_set_num
        self.cache_embed_index = None
        self.batch_size=batch_size
        self.dp = nn.Dropout(0.2)
        # 时间编码
        self.pos=nn.Parameter(torch.arange(input_dim).unsqueeze(1).expand(-1, self.batch_size).
                              permute(1, 0).unsqueeze(2), requires_grad=False)



        print('node_num:',node_num)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=node_num,
                                                        nhead=transformer_head
                                                        ).to(device)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=transformer_encoder_layers_num).to(device)
        self.fc = nn.Linear(node_num, classes_num).to(device)
        self.classes_num = classes_num
        self.init_params()
        self.alpha = alpha
        self.d_model = node_num
        self.batch_size = batch_size
    def init_params(self):
        nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))

    def forward(self, time, data,
                org_edge_index,
                ):
        time = time
        x = data.clone().detach()
        edge_index_sets = self.edge_index_sets
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        batch_num, node_num, all_feature = x.shape  
 
        x = x.view(-1, all_feature).contiguous()    
        gcn_outs = []
        for i, edge_index in enumerate(edge_index_sets):
            edge_num = edge_index.shape[1]
            cache_edge_index = self.cache_edge_index_sets[i]
            if cache_edge_index is None or cache_edge_index.shape[1] != edge_num * batch_num:
                self.cache_edge_index_sets[i] = get_batch_edge_index(edge_index, batch_num, node_num).to(device)
            batch_edge_index = self.cache_edge_index_sets[i]
            all_embeddings = self.embedding(torch.arange(node_num).to(device))  
            weights_arr = all_embeddings.detach().clone()  
            all_embeddings = all_embeddings.repeat(batch_num, 1)  
            weights = weights_arr.view(node_num, -1) 
            cos_ji_mat = torch.matmul(weights, weights.T)
            normed_mat = torch.matmul(weights.norm(dim=-1).view(-1, 1), weights.norm(dim=-1).view(1, -1))

            cos_ji_mat = cos_ji_mat / normed_mat

            dim = weights.shape[-1]
            topk_num = self.topk

            topk_indices_ji = torch.topk(cos_ji_mat, topk_num, dim=-1)[1]
            self.learned_graph = topk_indices_ji

            gated_i = torch.arange(0, node_num).T.unsqueeze(1).repeat(1, topk_num).flatten().to(device).unsqueeze(0)
            gated_j = topk_indices_ji.flatten().unsqueeze(0)
            gated_edge_index = torch.cat((gated_j, gated_i), dim=0)

            batch_gated_edge_index = get_batch_edge_index(gated_edge_index, batch_num, node_num).to(device)
            gcn_out = self.gnn_layers[i](x, batch_gated_edge_index, node_num=node_num * batch_num,
                                         embedding=all_embeddings)
            gcn_outs.append(gcn_out)
        x = torch.cat(gcn_outs, dim=1)  
        x = x.view(batch_num, node_num, -1)
        indexes = torch.arange(0, node_num).to(device)
        out = torch.mul(x, self.embedding(indexes))
        out = out.permute(0, 2, 1)
        out = F.relu(self.bn_outlayer_in(out))
        out = out.permute(0, 2, 1)  

        out = self.dp(out)
        out = self.out_layer(out)  # [batch_size, node_num, feature_dim =embed_dim]

        out = out.permute(0, 2, 1) # [batch_size, feature_dim * N=embed_dim, node_num]

        time = torch.log2(1 + torch.true_divide(time, self.alpha)).unsqueeze(2).to(device)

        pos_time = self.pos + time
        # angle_rates = 1 / torch.pow(10000, (2 * (torch.arange(self.d_model) // 2)) / self.d_model).to(device)
        angle_rates = 1 / torch.pow(10000, torch.true_divide(2 * (torch.arange(self.d_model) // 2), self.d_model)).to(
            device)

        angle_rates = angle_rates.unsqueeze(0).unsqueeze(0)  # 形状 [1, 1, d_model]
        pos_encoding = pos_time * angle_rates
        pos_encoding[:, :, 0::2] = torch.sin(pos_encoding[:, :, 0::2])
        pos_encoding[:, :, 1::2] = torch.cos(pos_encoding[:, :, 1::2])
        out = out+pos_encoding
 
        out = out.permute(1, 0, 2)
        out = self.transformer_encoder(out)
        out = out.permute(1, 0, 2)
        out = self.fc(out)  

        out = out.view(-1, self.classes_num)  

        return out


class TimeDataset(Dataset):
    def __init__(self, raw_data, edge_index, mode='train', config=None):
        self.raw_data = raw_data
        self.config = config
        self.edge_index = edge_index
        self.mode = mode
        timestamp = raw_data[0]
        timestamp = torch.tensor(timestamp.values).double()
        print('timestamp.shape', timestamp.shape) # timestamp.shape torch.Size([1354684])
        data = raw_data[1:-5]  
        data = torch.tensor(data).double()
        print('data.shape', data.shape) # data.shape torch.Size([36, 1354684])
        labels = raw_data[-5:]  
        print('type(labels):', type(labels))
        labels = torch.tensor(np.array(labels)).long()
        labels = torch.argmax(labels, dim=0) # labels.shape torch.Size([1354684])
        print('type(labels):', type(labels))
        print('labels.shape', labels.shape) # labels.shape torch.Size([5, 1354684])


        self.timestamps, self.x, self.labels = self.process(timestamp, data, labels)

    def __len__(self):
        return len(self.x)

    def process(self, timestamp, data, labels):
        x_arr = []
        timestamp_arr = []
        labels_arr = []

        slide_win, slide_stride = [self.config[k] for k
                                   in ['slide_win', 'slide_stride']
                                   ]
        is_train = self.mode == 'train'

        node_num, total_time_len = data.shape

        rang = range(slide_win, total_time_len, slide_stride) if is_train else range(slide_win, total_time_len)

        for i in rang:
            # [node_num, total_time_len]
            ft = data[:, i - slide_win:i]
            x_arr.append(ft)
            timestamp_arr.append(timestamp[i-slide_win:i])
            # tar = data[:, i]
            # y_arr.append(tar)
            labels_arr.append(labels[i-slide_win:i]) 

        x = torch.stack(x_arr).contiguous() 
   
        timestamps = torch.stack(timestamp_arr).contiguous()
        labels = torch.stack(labels_arr).contiguous()

        return timestamps, x, labels,
            # y, \

    def __getitem__(self, idx):
        time = self.timestamps[idx].double()
        feature = self.x[idx].double()
        label = self.labels[idx].double()

        edge_index = self.edge_index.long()

        return time, feature, label, edge_index

def loss_func(y_pred, y_true):
    # [batch_size*window_size, config.classes_num]
    criterion = nn.CrossEntropyLoss()

    # criterion = nn.BCELoss(reduction='mean')  
    return criterion(y_pred, y_true.long())       


def train0(model=None, save_path='', config={}, train_dataloader=None, val_dataloader=None,
          # feature_map={},
          # test_dataloader=None,
          # test_dataset=None,
          # dataset_name='swat',
          # train_dataset=None
          ):
    seed = config['seed']
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['decay'])

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    train_loss_list = []
    val_loss_list = []
    # cmp_loss_list = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataloader = train_dataloader
    min_loss = 1e+8

    epoch = config['epoch']
    early_stop_win = 10
    total_train_time = 0
    total_train_samples = 0
    model.train()
    # train
    train_start_time = time.time()
    for i_epoch in range(epoch):
        acu_loss = 0
        model.train()
        for timestamp,\
                x, \
                labels,\
                edge_index in dataloader:  
            batch_start = time.time()
            timestamp, x, labels, edge_index = [item.float().to(device) for item in [timestamp, x, labels, edge_index]]
            optimizer.zero_grad()
            out = model(timestamp, x, edge_index).float().to(device)
            labels=labels.view(-1)
            loss = loss_func(out, labels) 
            loss.backward()
            optimizer.step()
            batch_end = time.time()
            batch_time = batch_end - batch_start
            batch_size = timestamp.shape[0]  
            total_train_time += batch_time
            total_train_samples += batch_size
            acu_loss += loss.item()

        print('epoch ({} / {}) (Loss:{:.8f})'.format(
            i_epoch,
            epoch,
            acu_loss / len(dataloader)
            ), flush=True)
        train_loss_list.append(acu_loss / len(dataloader))
        # val
        val_loss = vali(model, val_dataloader)
        val_loss_list.append(val_loss)
        print('epoch ({} / {}) (Val_Loss:{:.8f})'.format(
            i_epoch,
            epoch,
            val_loss
        ), flush=True)
        model_save_path = os.path.join(save_path, f"model_epoch_{i_epoch}.pt")
        torch.save(model.state_dict(), model_save_path)
        if val_loss < min_loss:
            min_loss = val_loss
            stop_improve_count = 0
        else:
            stop_improve_count += 1
        if stop_improve_count >= early_stop_win:
            break
    avg_train_time_per_sample = total_train_time / total_train_samples
    print(f"train sample time: {avg_train_time_per_sample * 1000:.4f} ms")
    return train_loss_list, val_loss_list, model_save_path


def vali(model, dataloader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    acu_loss = 0
    for timestamp, x, labels, edge_index in dataloader:
        timestamp, x, labels, edge_index = [item.to(device).float() for item in [timestamp, x, labels, edge_index]]
        with torch.no_grad():
            out = model(timestamp, x, edge_index).float().to(device)
            # labels = labels.view(-1, 5)  
            # labels = torch.argmax(labels, dim=1)  
            labels = labels.view(-1)
            loss = loss_func(out, labels)
            acu_loss += loss.item()

    avg_loss = acu_loss / len(dataloader)

    return avg_loss


def test(model, dataloader, batch_size, window_size, row_num):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    y_true,y_pred = [],[]
    # out1_list = []
    acu_loss = 0
    total_test_time = 0
    total_test_samples = 0
    for timestamp, x, labels, edge_index in dataloader:
        timestamp, x, labels, edge_index = [item.to(device).float() for item in [timestamp, x, labels, edge_index]]
        with torch.no_grad():
            torch.cuda.synchronize()
            batch_start = time.time()
            out = model(timestamp, x, edge_index).float().to(device)  
            torch.cuda.synchronize()
            batch_end = time.time()
            batch_time = batch_end - batch_start
            batch_size = timestamp.shape[0]  
            total_test_time += batch_time
            total_test_samples += batch_size
            # labels = labels.view(-1, 5) 
            # labels = torch.argmax(labels, dim=1) 
            labels1 = labels.view(-1)
            loss = loss_func(out, labels1)
            acu_loss += loss.item()
            
            preds = torch.argmax(out, dim=1)  
            preds = preds.view(batch_size, window_size)
            preds = preds[:, row_num]  
            labels = labels.view(batch_size, window_size)
            labels = labels[:, row_num]  
            y_true.append(labels)
            y_pred.append(preds)

    avg_train_time_per_sample = total_test_time / total_test_samples
    print(f"test sample time: {avg_train_time_per_sample * 1000:.4f} ms")

    y_true = torch.cat(y_true).cpu().numpy()
    print('line 583 y_true.shape:', y_true.shape)
    y_pred = torch.cat(y_pred).cpu().numpy()
    print('line 585 y_pred.shape:', y_pred.shape)
    # out1_array = torch.cat(out1_list, dim=0).cpu().numpy()
    # out_array = torch.cat(y_pred, dim=0).cpu().numpy()
    output = pd.DataFrame(y_pred)
    output.to_csv('output.csv', index=False)
    true_label = pd.DataFrame(y_true)
    true_label.to_csv('labels.csv', index=False)
    conf_matrix = confusion_matrix(y_true, y_pred)
    print('confusion_matrix:', conf_matrix)
    tp = np.diag(conf_matrix)
    fp = conf_matrix.sum(axis=0) - tp
    fn = conf_matrix.sum(axis=1) - tp
    tn = conf_matrix.sum() - (tp + fp + fn)
    for i in range(5):
        # 避免除以 0
        precision = tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) > 0 else 0
        accuracy = (tp[i] + tn[i]) / (tp[i] + tn[i] + fp[i] + fn[i])
        recall = tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) > 0 else 0
        specificity = tn[i] / (tn[i] + fp[i]) if (tn[i] + fp[i]) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print(f"Class {i}:")
        print(f"  TP={tp[i]}, FP={fp[i]}, FN={fn[i]}, TN={tn[i]}")
        print(f"  Precision={precision:.4f}")
        print(f"  Accuracy={accuracy:.4f}")
        print(f"  Recall={recall:.4f}")
        print(f"  Specificity={specificity:.4f}")
        print(f"  F1 Score={f1:.4f}\n")

    avg_loss = acu_loss / len(dataloader)
    print('test loss:', avg_loss)

    return fn


def normalize_datasets(train_df, val_df, test_df):
    normalizer = MinMaxScaler(feature_range=(0, 1))
    train_cols = train_df.iloc[:, 7:]
    train_df.iloc[:, 7:] = normalizer.fit_transform(train_cols)
    val_df.iloc[:, 7:] = normalizer.transform(val_df.iloc[:, 7:])
    test_df.iloc[:, 7:] = normalizer.transform(test_df.iloc[:, 7:])
    return train_df, val_df, test_df



class Main():
    def __init__(self, train_config, env_config, debug=False):
        self.train_config = train_config
        self.env_config = env_config
        self.pic_save_path = env_config['pic_save_path']
        if not os.path.exists(self.pic_save_path):
            os.makedirs(self.pic_save_path)
        dataset = self.env_config['dataset']
        train = pd.read_csv(f'./data/{dataset}/train.csv')  
        val = pd.read_csv(f'./data/{dataset}/val.csv')
        test = pd.read_csv(f'./data/{dataset}/test.csv')   
        print('len(train):', len(train))            
        print('len(val):', len(val))
        print('len(test):', len(test))            
 
        train, val, test = normalize_datasets(train, val, test)
        self.batch_size = self.train_config['batch']
        self.window_size = self.train_config['slide_win']
        self.row_num = self.train_config['row_num']


        def get_fc_graph_struc(dataset):
            feature_file = open(f'./data/{dataset}/list.txt', 'r')

            feature_list = []
            for ft in feature_file:
                feature_list.append(ft.strip())

            struc_map = {} 
            for ft in feature_list:
                if ft not in struc_map:
                    struc_map[ft] = []

                for other_ft in feature_list:
                    if other_ft is not ft:
                        struc_map[ft].append(other_ft)
            return struc_map, feature_list

        fc_struc, feature_map = get_fc_graph_struc(dataset)
        self.device = env_config['device']

        def build_loc_net(struc, all_features, feature_map=[]):  
            index_feature_map = feature_map
            edge_indexes = [
                [],
                []
            ]
            for node_name, node_list in struc.items():  
                if node_name not in all_features:
                    continue
                if node_name not in index_feature_map:
                    index_feature_map.append(node_name)
                p_index = index_feature_map.index(node_name)
                for child in node_list:
                    if child not in all_features:
                        continue

                    if child not in index_feature_map:
                        print(f'error: {child} not in index_feature_map')

                    c_index = index_feature_map.index(child)
                    # edge_indexes[0].append(p_index)
                    # edge_indexes[1].append(c_index)
                    edge_indexes[0].append(c_index)
                    edge_indexes[1].append(p_index)
            return edge_indexes

        fc_edge_index = build_loc_net(fc_struc, feature_map, feature_map=feature_map)
        fc_edge_index = torch.tensor(fc_edge_index, dtype=torch.long)  


        def construct_data(data, feature_map):
            res = []
            res.append(data.loc[:, 'timestamp'])
            for feature in feature_map:
                if feature in data.columns:
                    res.append(data.loc[:, feature].values.tolist())
                else:
                    print(feature, 'not exist in data')
            for i in range(5):
                res.append(data.loc[:, f'label_{i}']) 
            return res  

        train_dataset_indata = construct_data(train, feature_map)
        val_dataset_indata = construct_data(val, feature_map)
        test_dataset_indata = construct_data(test, feature_map)
        cfg = {
            'slide_win': train_config['slide_win'],
            'slide_stride': train_config['slide_stride'],
        }
        train_dataset = TimeDataset(train_dataset_indata, fc_edge_index, mode='train', config=cfg)
        val_dataset = TimeDataset(val_dataset_indata, fc_edge_index, mode='test', config=cfg)
        test_dataset = TimeDataset(test_dataset_indata, fc_edge_index, mode='test', config=cfg)
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        # train_dataloader, val_dataloader = self.get_loaders(train_dataset, train_config['seed'], train_config['batch'],
        #                                                     val_ratio=train_config['val_ratio'])
        self.train_dataloader = DataLoader(train_dataset, batch_size=train_config['batch'], drop_last=True, shuffle=True)

        self.val_dataloader = DataLoader(val_dataset, batch_size=train_config['batch'], drop_last=True,
                                    shuffle=False)
        self.test_dataloader = DataLoader(test_dataset, batch_size=train_config['batch'],
                                          shuffle=False, num_workers=0, drop_last=True)
        edge_index_sets = []
        edge_index_sets.append(fc_edge_index)
        print('len(edge_index_sets):', len(edge_index_sets))  # 1
        print('len(feature_map):', len(feature_map))  # 36
        self.model = GDN(edge_index_sets=edge_index_sets,
                         node_num=len(feature_map),  
                         dim=train_config['dim'],  
                         input_dim=train_config['slide_win'],
                         out_layer_num=train_config['out_layer_num'],
                         out_layer_inter_dim=train_config['out_layer_inter_dim'],
                         topk=train_config['topk'],
                         transformer_head=train_config['transformer_head'],
                         transformer_encoder_layers_num=train_config['transformer_encoder_layers_num'],
                         classes_num=train_config['class_num'],
                         alpha=train_config['alpha'],
                         batch_size=train_config['batch'],
                         device=env_config['device']
                         ).to(self.device)

    def plot_loss(self, train_log, val_log, pic_save_path):
        plt.figure(figsize=(10, 5))
        plt.plot(train_log, label='Train Loss')
        plt.plot(val_log, label='Validation Loss')
        plt.title('Training/Validation Loss per Epoch')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()
        complete_path = os.path.join(pic_save_path, 'loss_plot.png')
        plt.savefig(complete_path, format='png', dpi=900)  # 保存为 PNG 格式，高分辨率
        print(f"Plot saved to {pic_save_path}")

    def run(self):

        train_log, val_log, BestModelSavePath = train0(model=self.model,
                               save_path=env_config['load_model_path'],
                               config=train_config,
                               train_dataloader=self.train_dataloader,
                               val_dataloader=self.val_dataloader,
                               )
        self.plot_loss(train_log, val_log, self.pic_save_path)

        # test
        self.model.load_state_dict(torch.load(BestModelSavePath))
        best_model = self.model.to(self.device)
        print('test start')

        test(best_model, self.test_dataloader, self.batch_size, self.window_size, self.row_num)
        print('test finished!')

 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-batch', help='batch size', type = int, default=128)
    parser.add_argument('-epoch', help='train epoch', type = int, default=100)
    parser.add_argument('-slide_win', help='slide_win', type = int, default=7)
    parser.add_argument('-dim', help='dimension', type = int, default=64)   
    parser.add_argument('-slide_stride', help='slide_stride', type = int, default=1)
    parser.add_argument('-save_path_pattern', help='save path pattern', type = str, default='')
    parser.add_argument('-dataset', help='dataset file document下面有两个文件，一个train一个test', type = str, default='dataset')
    parser.add_argument('-device', help='cuda / cpu', type = str,
                        default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('-random_seed', help='random seed', type = int, default=0)
    # parser.add_argument('-comment', help='experiment comment', type = str, default='')
    parser.add_argument('-out_layer_num', help='outlayer num', type = int, default=1)
    parser.add_argument('-out_layer_inter_dim', help='out_layer_inter_dim', type = int, default=256)
    parser.add_argument('-decay', help='decay', type = float, default=0)  
    parser.add_argument('-val_ratio', help='val ratio', type = float, default=0.2)
    parser.add_argument('-topk', help='topk num', type = int, default=15)
    parser.add_argument('-report', help='best / val', type = str, default='best')  
    parser.add_argument('-transformer_head', help='transformer_attention_head', type = int, default='4')
    parser.add_argument('-transformer_encoder_layers_num', help='transformer encoder number', type = int, default='6')
    parser.add_argument('-class_num', help='final classes of anomaly+1', type = int, default='5')
    parser.add_argument('-alpha', help='α', type = float, default='0.01')
    parser.add_argument('-load_model_path', help='trained model path', type = str, default='model_checkpoint')
    parser.add_argument('-loss_pic_save_path', help='pic_save_path', type = str, default='./loss_save')
    parser.add_argument('-lr', help='learning rate', type = float, default='0.000001')
    parser.add_argument('-row_num', help='0<=chosen line<=slide win-1', type = int, default='4')

    args = parser.parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(args.random_seed)


    train_config = {
        'batch': args.batch,
        'epoch': args.epoch,
        'slide_win': args.slide_win,
        'dim': args.dim,   
        'slide_stride': args.slide_stride,
        'seed': args.random_seed,
        'out_layer_num': args.out_layer_num, 
        'out_layer_inter_dim': args.out_layer_inter_dim,  
        'decay': args.decay,  
        'val_ratio': args.val_ratio,  
        'topk': args.topk,  
        'transformer_head': args.transformer_head, 
        'transformer_encoder_layers_num': args.transformer_encoder_layers_num,
        'class_num': args.class_num, 
        'alpha': args.alpha,   
        'lr': args.lr,  
        'row_num': args.row_num 
    }

    env_config={
        'save_path': args.save_path_pattern,  # no
        'dataset': args.dataset,    
        'report': args.report,  # ?
        'device': args.device,
        'load_model_path': args.load_model_path,
        'pic_save_path': args.loss_pic_save_path
    }

    main = Main(train_config, env_config, debug=False)
    from thop import profile



    batch_size = train_config['batch']
    slide_win = train_config['slide_win']
    node_num = len(main.model.embedding.weight)  
    edge_num = main.model.edge_index_sets[0].shape[1]


    dummy_timestamp = torch.zeros((batch_size, slide_win)).float().to(main.device)
    dummy_x = torch.zeros((batch_size, node_num, slide_win)).float().to(main.device)
    dummy_edge_index = torch.zeros((2, edge_num)).long().to(main.device)


    flops, params = profile(main.model, inputs=(dummy_timestamp, dummy_x, dummy_edge_index), verbose=False)

    print(f'FLOPs: {flops:,}')
    print(f'Params: {params:,}')

    main.run()
