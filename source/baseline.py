import torch
from torch import nn
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import GCNConv,GATConv,SAGEConv,Set2Set
import torch.nn.functional as F
import sys
from layers import *



class GCN(nn.Module):
    def __init__(self, node_in_dim,node_out_dim=64,
                 heads=1,
                 dropout=0.1
                 ):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(node_in_dim, node_out_dim)
        self.conv2 = GCNConv(node_out_dim, node_out_dim)

    def forward(self, x, edge_index, edge_attr=None):
        x = self.conv1.forward(x, edge_index)
        x = self.conv2.forward(x, edge_index)
        return x

class GraphSAGE(nn.Module):
    def __init__(self, node_in_dim,node_out_dim=64,
                 heads=1,
                 dropout=0.1
                 ):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(node_in_dim, node_out_dim)
        self.conv2 = SAGEConv(node_out_dim, node_out_dim)

    def forward(self, x, edge_index, edge_attr=None):
        x = self.conv1.forward(x, edge_index)
        x = self.conv2.forward(x, edge_index)
        return x

class GAT(nn.Module):
    def __init__(self,node_in_dim,
                 node_out_dim=64,
                 heads=1,
                 dropout=0.1):
        super(GAT,self).__init__()
        self.conv1 = GATConv(node_in_dim,node_out_dim,heads,dropout=dropout)
        self.conv2 = GATConv(node_out_dim*heads,node_out_dim,heads,concat=False,dropout=dropout)

    def forward(self, x,edge_index,edge_attr=None):
        x = self.conv1.forward(x,edge_index)
        x = self.conv2.forward(x,edge_index)
        return x

class EGAT(nn.Module):
    def __init__(self,node_in_dim,
                 node_out_dim=64,
                 edge_in_dim=4,
                 heads=1,
                 dropout=0.1):
        super(EGAT,self).__init__()
        self.conv1 = EGATConv(node_in_dim,node_out_dim,edge_in_dim,heads,dropout=dropout)
        self.conv2 = EGATConv(node_out_dim*heads,node_out_dim,edge_in_dim,heads,concat=False,dropout=dropout)

    def forward(self, x,edge_index,edge_attr=None):
        x = self.conv1.forward(x,edge_index,edge_attr)

        x = self.conv2.forward(x,edge_index,edge_attr)
        return x
