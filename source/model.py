import torch
from torch import nn
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import NNConv, Set2Set, global_mean_pool,global_max_pool,GATConv
import torch.nn.functional as F
import math,sys
import numpy as np
from torch_geometric.data import Data,Batch
from layers import *
from baseline import *

class MPNNConv(torch.nn.Module):
    def __init__(self,
                 node_in_dim,
                 node_out_dim=64,
                 edge_in_dim=3,
                 edge_out_dim=64,
                 num_step_message_passing=3
                 ):
        super(MPNNConv, self).__init__()
        self.mlp = nn.Linear(node_in_dim,node_out_dim)
        self.bn = nn.BatchNorm1d(node_out_dim)
        self.num_step_message_passing = num_step_message_passing
        self.edge_network = nn.Sequential(
                    nn.Linear(edge_in_dim, edge_out_dim),
                    nn.BatchNorm1d(edge_out_dim),
                    nn.ReLU(),
                    nn.Linear(edge_out_dim, node_out_dim * node_out_dim),
                    nn.ReLU()
                )
        self.conv = NNConv(node_out_dim, node_out_dim, self.edge_network, aggr='mean',root_weight=True)
        self.gru = nn.GRU(node_out_dim, node_out_dim)
        self.linear = nn.Linear(node_in_dim,node_out_dim)

    def forward(self, x,edge_index,edge_attr):
        #self.gru.flatten_parameters()
        out = F.relu(self.bn(self.mlp(x)))
        # out = F.relu(self.mlp(x))
        # out = F.dropout(out, p=0.5, training=self.training)
        h = out.unsqueeze(0)
        for i in range(self.num_step_message_passing):
            m = self.conv.forward(out, edge_index, edge_attr)
            m = F.relu(m)
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)
        out = self.linear(x)+out
        return out

class Murat_MPNNConv(torch.nn.Module):
    def __init__(self,
                 node_in_dim,
                 node_out_dim=64,
                 edge_in_dim=3,
                 edge_out_dim=64,
                 num_step_message_passing=3
                 ):
        super(Murat_MPNNConv, self).__init__()
        self.fe = nn.Linear(node_in_dim,node_out_dim)
        self.num_step_message_passing = num_step_message_passing
        self.conv = Murat_NNConv(node_out_dim, node_out_dim, edge_in_dim,edge_out_dim)
        self.fu = nn.Linear(2*node_out_dim,node_out_dim)
        self.linear = nn.Linear(node_in_dim,node_out_dim)

    def forward(self, x,edge_index,edge_attr):
        #self.gru.flatten_parameters()
        out = F.relu(self.fe(x))
        for i in range(self.num_step_message_passing):
            m = self.conv.forward(out, edge_index, edge_attr)
            out = self.fu(torch.cat([out,m],dim=-1))
        return out


class Murat_MPNN(torch.nn.Module):
    def __init__(self,option):
        super(Murat_MPNN, self).__init__()
        jet_in_dim, jet_edge_in_dim, particle_in_dim, particle_edge_in_dim = \
            option.jet_features,option.jet_edge_features,option.particle_features, option.particle_edge_features
        hid1,hid2,  edge_out_dim, num_step_message_passing,self.dropout = \
             option.hid1, option.hid2, option.hid1, option.num_step_message_passing, option.dropout

        self.conv = Murat_MPNNConv(jet_in_dim, hid2, jet_edge_in_dim,edge_out_dim)
        self.fv = nn.Linear(hid2,4)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, high_data,low_data=None):
        x, edge_index, edge_attr = high_data.x, high_data.edge_index, high_data.edge_attr
        out = self.conv.forward(x,edge_index,edge_attr)
        out = global_mean_pool(out, high_data.batch)
        out = self.logsoftmax(self.fv(out))
        return out


class Henrion_MPNNConv(torch.nn.Module):
    def __init__(self,
                 node_in_dim,
                 node_out_dim=64,
                 num_step_message_passing=2,
                 dropout=0.1
                 ):
        super(Henrion_MPNNConv, self).__init__()
        self.mlp = nn.Linear(node_in_dim,node_out_dim)
        self.num_step_message_passing = num_step_message_passing
        self.conv = HenrionConv(node_out_dim, node_out_dim, dropout=dropout)
        self.gru = nn.GRU(node_out_dim+node_in_dim, node_out_dim)
    def forward(self, x,edge_index):
        #self.gru.flatten_parameters()
        out = self.mlp(x)
        h = out.unsqueeze(0)
        for i in range(self.num_step_message_passing):
            m = self.conv.forward(out, edge_index)
            out, h = self.gru(torch.cat([m.unsqueeze(0),x.unsqueeze(0)],dim=-1), h)
            out = out.squeeze(0)

        return out

class EGatConv(torch.nn.Module):
    def __init__(self,
                 node_in_dim,
                 node_out_dim=64,
                 edge_in_dim=3,
                 heads=1,
                 num_step_message_passing=3,
                 dropout=0.1
                 ):
        super(EGatConv, self).__init__()
        self.mlp = nn.Linear(node_in_dim,node_out_dim)
        self.bn = nn.BatchNorm1d(node_out_dim)
        self.num_step_message_passing = num_step_message_passing
        self.conv = EGATConv(node_out_dim, node_out_dim,
                             edge_in_channels=edge_in_dim,heads=heads,dropout=dropout)
        self.gru = nn.GRU(node_out_dim*heads, node_out_dim)
        self.linear = nn.Linear(node_in_dim,node_out_dim)

    def forward(self, x,edge_index,edge_attr=None):
        #self.gru.flatten_parameters()
        out = F.relu(self.bn(self.mlp(x)))
        h = out.unsqueeze(0)
        for i in range(self.num_step_message_passing):
            m = self.conv.forward(out, edge_index,edge_attr)
            m = F.relu(m)
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)
        out = self.linear(x)+out
        return out

class MPNN(torch.nn.Module):
    def __init__(self,option):
        super(MPNN, self).__init__()
        jet_in_dim, jet_edge_in_dim, particle_in_dim, particle_edge_in_dim = \
            option.jet_features,option.jet_edge_features,option.particle_features, option.particle_edge_features
        hid1,hid2,  edge_out_dim, num_step_message_passing,self.dropout = \
             option.hid1, option.hid2, option.hid1, option.num_step_message_passing, option.dropout
        self.conv1 = MPNNConv(jet_in_dim, hid2, jet_edge_in_dim,edge_out_dim,num_step_message_passing)
        self.set2set = Set2Set(hid2, processing_steps=3)
        self.mlp1 = nn.Linear(2*hid2, hid2)
        self.mlp2 = nn.Linear(hid2,4)
        self.logsoftmax=nn.LogSoftmax(dim=-1)
    def forward(self, high_data,low_data=None):
        x, edge_index, edge_attr = high_data.x, high_data.edge_index, high_data.edge_attr
        x = self.conv1.forward(x,edge_index,edge_attr)
        x = self.set2set(x, high_data.batch)
        out = F.relu(self.mlp1(x))
        out = F.dropout(out,p=self.dropout,training=self.training)
        out = self.logsoftmax(self.mlp2(out))
        # out = self.logsoftmax(self.mlp1(out))
        return out

class Henrion_MPNN(torch.nn.Module):
    """According to 'Neural Message Passing for Jet Physics' """
    def __init__(self,option):
        super(Henrion_MPNN, self).__init__()
        jet_in_dim, jet_edge_in_dim, particle_in_dim, particle_edge_in_dim = \
            option.jet_features,option.jet_edge_features,option.particle_features, option.particle_edge_features
        hid1,hid2,  edge_out_dim, num_step_message_passing,self.dropout = \
             option.hid1, option.hid2, option.hid1, option.num_step_message_passing, option.dropout
        self.conv1 = Henrion_MPNNConv(jet_in_dim, hid2,dropout=self.dropout)
        self.set2set = Set2Set(hid2, processing_steps=3)
        self.mlp1 = nn.Linear(2*hid2, hid2)
        self.mlp2 = nn.Linear(hid2,4)
        self.logsoftmax=nn.LogSoftmax(dim=-1)
    def forward(self, high_data,low_data=None):
        x, edge_index, edge_attr = high_data.x, high_data.edge_index, high_data.edge_attr
        x = self.conv1.forward(x,edge_index)
        x = self.set2set(x, high_data.batch)
        out = F.relu(self.mlp1(x))
        out = F.dropout(out,p=self.dropout,training=self.training)
        out = self.logsoftmax(self.mlp2(out))
        # out = self.logsoftmax(self.mlp1(out))
        return out

class HierMPNN_attention_set(torch.nn.Module):
    def __init__(self,option):
        super(HierMPNN_attention_set, self).__init__()
        jet_in_dim, node_in_dim, hid1,hid2, edge_in_dim, edge_out_dim, num_step_message_passing,self.dropout = \
            option.jet_features, option.particle_features, option.hid1, option.hid2, option.particle_edge_features,\
            option.hid1, option.num_step_message_passing, option.dropout

        self.conv1 = MPNNConv(node_in_dim, hid1, edge_in_dim,edge_out_dim,num_step_message_passing)
        self.mlp = nn.Linear(jet_in_dim, hid1)

        self.ln = nn.Linear(jet_in_dim,hid2)
        self.conv2 = MPNNConv(hid1+hid2, hid2, edge_in_dim, edge_out_dim, num_step_message_passing)

        self.set2set = Set2Set(hid2, processing_steps=3)
        self.att = MultiHeadedAttention(4, hid1, dropout=self.dropout)
        self.mlp1 = nn.Linear(2*hid2, hid2)
        self.mlp2 = nn.Linear(hid2,4)
        self.logsoftmax=nn.LogSoftmax(dim=-1)
    def forward(self, high_data, low_data):
        x, edge_index, edge_attr = high_data.x, high_data.edge_index, high_data.edge_attr
        low_x, low_edge_index, low_edge_attr = low_data.x, low_data.edge_index, low_data.edge_attr

        low_x = self.conv1.forward(low_x,low_edge_index,low_edge_attr)
        low_x = F.dropout(low_x, p=self.dropout, training=self.training)

        low_x,_ = to_dense_batch(low_x,low_data.batch)  # (batch,len,dim)

        x_q = F.relu(self.mlp(x))
        low_x = self.att(low_x,low_x,x_q.unsqueeze(1)).squeeze()

        x = torch.cat([F.relu((self.ln(x))),low_x],dim=-1)

        x = self.conv2.forward(x, edge_index, edge_attr)
        # x = torch.cat([global_mean_pool(x, high_data.batch), global_max_pool(x, high_data.batch)], dim=-1)
        x = self.set2set(x, high_data.batch)
        out = F.relu(self.mlp1(x))
        out = F.dropout(out,p=self.dropout,training=self.training)
        out = self.logsoftmax(self.mlp2(out))
        # out = self.logsoftmax(self.mlp1(out))
        return out

class HierEGAT_attention_set(torch.nn.Module):
    def __init__(self,option):
        super(HierEGAT_attention_set, self).__init__()
        jet_in_dim, node_in_dim, hid1,hid2, edge_in_dim, edge_out_dim, num_step_message_passing,self.dropout = \
            option.jet_features, option.particle_features, option.hid1, option.hid2, option.particle_edge_features,\
            option.hid1, option.num_step_message_passing, option.dropout
        heads = option.heads
        self.conv1 = EGatConv(node_in_dim, hid1, edge_in_dim, heads, num_step_message_passing,dropout=self.dropout)
        self.mlp = nn.Linear(jet_in_dim, hid1)

        self.ln = nn.Linear(jet_in_dim,hid2)
        self.conv2 = EGatConv(hid1+hid2, hid2, edge_in_dim, heads, num_step_message_passing,dropout=self.dropout)

        self.set2set = Set2Set(hid2, processing_steps=3)
        self.att = MultiHeadedAttention(4, hid1, dropout=self.dropout)
        self.mlp1 = nn.Linear(2*hid2, hid2)
        self.mlp2 = nn.Linear(hid2,4)
        self.logsoftmax=nn.LogSoftmax(dim=-1)
    def forward(self, high_data, low_data):
        x, edge_index, edge_attr = high_data.x, high_data.edge_index, high_data.edge_attr
        low_x, low_edge_index, low_edge_attr = low_data.x, low_data.edge_index, low_data.edge_attr

        low_x = self.conv1.forward(low_x,low_edge_index,low_edge_attr)
        low_x = F.dropout(low_x, p=self.dropout, training=self.training)

        low_x,_ = to_dense_batch(low_x,low_data.batch)  # (batch,len,dim)

        x_q = F.relu(self.mlp(x))
        low_x = self.att(low_x,low_x,x_q.unsqueeze(1)).squeeze()

        x = torch.cat([F.relu((self.ln(x))),low_x],dim=-1)

        x = self.conv2.forward(x, edge_index, edge_attr)
        # x = torch.cat([global_mean_pool(x, high_data.batch), global_max_pool(x, high_data.batch)], dim=-1)
        x = self.set2set(x, high_data.batch)
        out = F.relu(self.mlp1(x))
        out = F.dropout(out,p=self.dropout,training=self.training)
        out = self.logsoftmax(self.mlp2(out))
        # out = self.logsoftmax(self.mlp1(out))
        return out


class Baselines(torch.nn.Module):
    def __init__(self, option):
        super(Baselines, self).__init__()
        jet_in_dim, node_in_dim, hid1, hid2, edge_in_dim, edge_out_dim, num_step_message_passing, self.dropout = \
            option.jet_features, option.particle_features, option.hid1, option.hid2, option.particle_edge_features, \
            option.hid1, option.num_step_message_passing, option.dropout
        heads = option.heads
        gnn_name = option.gnn

        self.conv1 = getattr(sys.modules[__name__], gnn_name)(node_in_dim, hid1, heads, self.dropout)
        self.mlp = nn.Linear(jet_in_dim, hid1)

        self.ln = nn.Linear(jet_in_dim, hid2)
        self.conv2 = getattr(sys.modules[__name__], gnn_name)(hid1 + hid2, hid2, heads, self.dropout)

        self.set2set = Set2Set(hid2, processing_steps=3)
        self.att = MultiHeadedAttention(4, hid1, dropout=self.dropout)
        self.mlp1 = nn.Linear(2 * hid2, hid2)
        self.mlp2 = nn.Linear(hid2, 4)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, high_data, low_data):
        x, edge_index, edge_attr = high_data.x, high_data.edge_index, high_data.edge_attr
        low_x, low_edge_index, low_edge_attr = low_data.x, low_data.edge_index, low_data.edge_attr

        low_x = self.conv1.forward(low_x, low_edge_index)
        low_x = F.dropout(low_x, p=self.dropout, training=self.training)

        low_x, _ = to_dense_batch(low_x, low_data.batch)  # (batch,len,dim)

        x_q = F.relu(self.mlp(x))
        low_x = self.att(low_x, low_x, x_q.unsqueeze(1)).squeeze()

        x = torch.cat([F.relu((self.ln(x))), low_x], dim=-1)

        x = self.conv2.forward(x, edge_index)

        x = self.set2set(x, high_data.batch)
        out = F.relu(self.mlp1(x))
        out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.logsoftmax(self.mlp2(out))

        return out

class EGAT(torch.nn.Module):
    def __init__(self,option):
        super(EGAT, self).__init__()
        jet_in_dim, node_in_dim, hid1,hid2, edge_in_dim, edge_out_dim, num_step_message_passing,self.dropout = \
            option.jet_features, option.particle_features, option.hid1, option.hid2, option.particle_edge_features,\
            option.hid1, option.num_step_message_passing, option.dropout
        heads = option.heads
        self.conv1 = EGatConv(jet_in_dim, hid2, edge_in_dim, heads, num_step_message_passing,dropout=self.dropout)

        self.set2set = Set2Set(hid2, processing_steps=3)

        self.mlp1 = nn.Linear(2*hid2, hid2)
        self.mlp2 = nn.Linear(hid2,4)
        self.logsoftmax=nn.LogSoftmax(dim=-1)
    def forward(self, high_data, low_data):
        x, edge_index, edge_attr = high_data.x, high_data.edge_index, high_data.edge_attr
        x = self.conv1.forward(x,edge_index,edge_attr)
        # x = torch.cat([global_mean_pool(x, high_data.batch), global_max_pool(x, high_data.batch)], dim=-1)
        x = self.set2set(x, high_data.batch)
        out = F.relu(self.mlp1(x))
        out = F.dropout(out,p=self.dropout,training=self.training)
        out = self.logsoftmax(self.mlp2(out))
        # out = self.logsoftmax(self.mlp1(out))
        return out


class Baselines_withouthier(torch.nn.Module):
    def __init__(self, option):
        super(Baselines_withouthier, self).__init__()
        jet_in_dim, node_in_dim, hid1, hid2, edge_in_dim, edge_out_dim, num_step_message_passing, self.dropout = \
            option.jet_features, option.particle_features, option.hid1, option.hid2, option.particle_edge_features, \
            option.hid1, option.num_step_message_passing, option.dropout
        heads = option.heads
        gnn_name = option.gnn
        if gnn_name!='EGAT':
            self.conv1 = getattr(sys.modules[__name__], gnn_name)(jet_in_dim, hid2, heads, self.dropout)
        else:
            self.conv1 = EGAT(jet_in_dim,hid2,edge_in_dim,heads,self.dropout)
        self.set2set = Set2Set(hid2, processing_steps=3)
        self.mlp1 = nn.Linear(2 * hid2, hid2)
        self.mlp2 = nn.Linear(hid2, 4)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, high_data, low_data=None):
        x, edge_index, edge_attr = high_data.x, high_data.edge_index, high_data.edge_attr
        x = self.conv1.forward(x, edge_index,edge_attr)
        x = self.set2set(x, high_data.batch)
        out = F.relu(self.mlp1(x))
        out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.logsoftmax(self.mlp2(out))
        return out

class MultiHeadedAttention(nn.Module):
    """
    Args:
       head_count (int): number of parallel heads
       model_dim (int): the dimension of keys/values/queries,
           must be divisible by head_count
       dropout (float): dropout parameter
    """

    def __init__(self, head_count, model_dim, dropout=0.1):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim
        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count

        self.linear_keys = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_values = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_query = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(model_dim, model_dim)

    def forward(self, key, value, query):
        """
        Compute the context vector and the attention vectors.

        Args:
           key (`FloatTensor`): set of `key_len`
                key vectors `[batch, key_len, dim]`
           value (`FloatTensor`): set of `key_len`
                value vectors `[batch, key_len, dim]`
           query (`FloatTensor`): set of `query_len`
                 query vectors  `[batch, query_len, dim]`

        Returns:
           * output context vectors `[batch, query_len, dim]`
        """

        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count

        def shape(x):
            """  projection """
            # (B,L,h) >> (B,L,heads, h/heads) >> (B,heads,L,h/heads)
            return x.view(batch_size, -1, head_count, dim_per_head) \
                .transpose(1, 2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous() \
                .view(batch_size, -1, head_count * dim_per_head)

        # 1) Project key, value, and query.

        key = self.linear_keys(key)
        value = self.linear_values(value)
        query = self.linear_query(query)
        key = shape(key)
        value = shape(value)
        query = shape(query)

        # 2) Calculate and scale scores.
        query = query / math.sqrt(dim_per_head)
        scores = torch.matmul(query, key.transpose(2, 3))  # (B,heads,L,h/heads) (B,heads,h/heads,1)
        # 3) Apply attention dropout and compute context vectors.
        attn = self.softmax(scores)

        drop_attn = self.dropout(attn)
        context = unshape(torch.matmul(drop_attn, value))
        output = self.final_linear(context)

        return output

