import torch
import numpy as np
from torch.nn import Linear, BatchNorm1d, LayerNorm, Dropout, ReLU
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.graphgym.register import *

@register_layer('origin_gt')
class GraphTransformer(MessagePassing):
    def __init__(self, in_dim, out_dim, num_heads, dropout=0.0, layer_norm=False, batch_norm=True, residual=True,
                 use_bias=False):
        super().__init__(aggr='add')  # "Add" aggregation.
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = Dropout(dropout)
        self.residual = residual
        self.layer_norm = LayerNorm(out_dim) if layer_norm else None
        self.batch_norm = BatchNorm1d(out_dim) if batch_norm else None

        # Define the layers
        self.Q = Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.K = Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.V = Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.proj_e = Linear(in_dim, out_dim * num_heads, bias=use_bias)

        self.O_h = Linear(out_dim, out_dim)
        self.O_e = Linear(out_dim, out_dim)

        self.FFN_h_layer1 = Linear(out_dim, out_dim * 2)
        self.FFN_h_layer2 = Linear(out_dim * 2, out_dim)
        self.FFN_e_layer1 = Linear(out_dim, out_dim * 2)
        self.FFN_e_layer2 = Linear(out_dim * 2, out_dim)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        # Multi-head attention mechanism
        # Splitting the linearly transformed x into multiple heads
        Q = self.Q(x).view(-1, self.num_heads, self.out_channels)
        K = self.K(x).view(-1, self.num_heads, self.out_channels)
        V = self.V(x).view(-1, self.num_heads, self.out_channels)

        # Edge feature transformation
        proj_e = self.proj_e(edge_attr).view(-1, self.num_heads, self.out_channels)

        # Propagate method calls the message and update functions under the hood
        x, edge_attr = self.propagate(edge_index, x=(Q, K, V), edge_attr=proj_e, size=None)

        # Apply batch/layer norm and residual connections as needed
        if self.residual:
            x += self.O_h(x)
            edge_attr += self.O_e(edge_attr)

        if self.layer_norm is not None:
            x = self.layer_norm(x)
            edge_attr = self.layer_norm(edge_attr)

        if self.batch_norm is not None:
            x = self.batch_norm(x)
            edge_attr = self.batch_norm(edge_attr)

        # FFN for node features
        x = self.FFN_h_layer1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.FFN_h_layer2(x)

        # FFN for edge features
        edge_attr = self.FFN_e_layer1(edge_attr)
        edge_attr = F.relu(edge_attr)
        edge_attr = self.dropout(edge_attr)
        edge_attr = self.FFN_e_layer2(edge_attr)

        data.x = x
        data.edge_attr = edge_attr

        return data

    def message(self, x_i, x_j, edge_attr):
        # Compute attention scores
        score = (x_i * x_j).sum(-1) / np.sqrt(self.out_channels)
        score = score.view(-1, 1, self.num_heads)  # Reshape for broadcasting
        score = F.softmax(score, dim=0)  # Softmax over the source node dimension

        # Update edge features and node features
        edge_attr = score * edge_attr
        return edge_attr * x_j

    def update(self, aggr_out, x):
        # Update node features
        return aggr_out, x