from torch import nn
from torch.nn import Parameter,Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
import torch
import torch.nn.functional as F
import math


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


class HenrionConv(MessagePassing):
    def __init__(self, in_channels, out_channels, dropout=0, **kwargs):
        super(HenrionConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        self.att = nn.Linear(out_channels,1) #Parameter(torch.Tensor(1, heads, 2 * out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)

    def forward(self, x, edge_index, size=None):
        """"""
        if size is None and torch.is_tensor(x):
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        if torch.is_tensor(x):
            x = torch.matmul(x, self.weight)
        else:
            x = (None if x[0] is None else torch.matmul(x[0], self.weight),
                 None if x[1] is None else torch.matmul(x[1], self.weight))

        return self.propagate(edge_index, size=size, x=x)


    def message(self, edge_index_i, x_i, x_j, size_i):
        # Compute attention coefficients.
        x_j = x_j.view(-1, self.out_channels)
        x_i = x_i.view(-1, self.out_channels)

        alpha = self.att(x_j+x_i)
        alpha = softmax(alpha, edge_index_i, size_i)

        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.view(-1, 1)

    def update(self, aggr_out):
        return torch.tanh(aggr_out)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


class Murat_NNConv(MessagePassing):
    def __init__(self, in_node_channels, out_node_channels, edge_channels, edge_out,**kwargs):
        super(Murat_NNConv, self).__init__(aggr='add', **kwargs)
        self.rbf_d = Linear(edge_channels, edge_out)
        m_channels = in_node_channels + edge_out  # fe_x + rbf_d
        self.fm_sd = Linear(m_channels, out_node_channels)  # assert m_channels = sd_channels

    def forward(self, x, edge_index, edge_attr, size=None):
        rbf_d = F.relu(self.rbf_d(edge_attr))
        # Formula 2.6
        x = self.propagate(edge_index, size=size, x=x, rbf_d=rbf_d)
        # Formula 2.9
        return x

    def message(self, edge_index_i, x_i, x_j, rbf_d, size_i):

        return F.relu(self.fm_sd(torch.cat([x_j, rbf_d], dim=-1)))

    def update(self, aggr_out):
        # just a placeholder to let the original update function invalidate
        return aggr_out

class EGATConv(MessagePassing):

    def __init__(self, in_channels, out_channels, edge_in_channels, heads=1,concat=True,
                 negative_slope=0.2, dropout=0, bias=True,**kwargs):
        super(EGATConv, self).__init__(aggr='add', **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.weight = Parameter(torch.Tensor(in_channels,heads * out_channels))
        self.weight2 = Parameter(torch.Tensor(edge_in_channels,heads*out_channels))
        self.att = Parameter(torch.Tensor(1, heads, 3 * out_channels))
        # self.mlp = nn.Linear(3*heads*out_channels,edge_in_channels)
        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.weight2)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr, size=None):
        # if size is None and torch.is_tensor(x):
        #     edge_index, _ = remove_self_loops(edge_index)
        #     edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(self.node_dim))
        x = F.relu(torch.matmul(x, self.weight))
        edge_attr=torch.matmul(edge_attr,self.weight2)
        pseudo = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        return self.propagate(edge_index, size=size, x=x, pseudo=pseudo)

    def message(self, edge_index_i, x_i, x_j, pseudo, size_i):
        # Compute attention coefficients.
        x_j = x_j.view(-1, self.heads, self.out_channels)
        x_i = x_i.view(-1, self.heads, self.out_channels)
        e_ij = pseudo.view(-1, self.heads, self.out_channels)

        triplet = torch.cat([x_i, e_ij, x_j], dim=-1)
        alpha = (triplet * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, size_i)

        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        aggr_out = aggr_out + self.bias
        return aggr_out

class EGATConv2(MessagePassing):

    def __init__(self, in_channels, out_channels, edge_in_channels, heads=1,concat=True,
                 negative_slope=0.2, dropout=0, bias=True,**kwargs):
        super(EGATConv2, self).__init__(aggr='add', **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.weight = Parameter(torch.Tensor(in_channels,heads * out_channels))
        self.weight2 = Parameter(torch.Tensor(edge_in_channels,heads*out_channels))

        self.linear = nn.Linear(3*out_channels,out_channels)

        self.mlp = nn.Linear(3*heads*out_channels,edge_in_channels)
        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.weight2)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr, size=None):
        x = F.relu(torch.matmul(x, self.weight))
        edge_attr=torch.matmul(edge_attr,self.weight2)
        pseudo = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        return self.propagate(edge_index, size=size, x=x, pseudo=pseudo)

    def message(self, x_i, x_j, pseudo):
        # Compute attention coefficients.
        x_j = x_j.view(-1, self.heads, self.out_channels)
        x_i = x_i.view(-1, self.heads, self.out_channels)
        e_ij = pseudo.view(-1, self.heads, self.out_channels)

        triplet = torch.cat([x_i, e_ij, x_j], dim=-1)
        alpha = self.linear(triplet)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = F.softmax(alpha, dim=0)
        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.view(-1, self.heads, self.out_channels)

    def update(self, aggr_out):
        aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        aggr_out = aggr_out + self.bias
        return aggr_out
