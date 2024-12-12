import math
import numpy as np
import torch
from tqdm import tqdm
from torch.nn import (ModuleList, Linear, Conv1d, MaxPool1d, Embedding, ReLU,
                      Sequential, LayerNorm, BatchNorm1d)
import torch.nn.functional as F
from torch_geometric.nn import (GCNConv, SAGEConv, GINConv,
                                global_sort_pool, global_add_pool, global_mean_pool)
from torch_geometric.data import Data
from utilities.graphormer.model import Graphormer
from utilities.graph_transformer_layer import GraphTransformerLayer
from utilities.pyg_to_dgl import pyg_to_dgl
import pdb

def abstract_pair_data(data, z_emb_pair=None):
    if z_emb_pair is None:
        pair_data = Data(x=data.pair_x, z=data.pair_z, edge_index=data.pair_edge_idx)
    else:  # 传入z_emb，就用z_emb替代feature
        pair_data = Data(x=z_emb_pair, z=data.pair_z, edge_index=data.pair_edge_idx)
    for key in data.keys():
        if key.startswith('pair_') and key not in ['pair_x', 'pair_z', 'pair_edge_idx']:
            pair_data[key[5:]] = data[key]
    return pair_data

class SANGraphormer(torch.nn.Module):
    def __init__(self, args, hidden_channels, num_layers, max_z, num_features,
                 use_feature=False, use_feature_GT=True, use_time_feature=False, node_embedding=None, dropout = 0.5,
                 GT_n_heads=4, full_graph=False, layer_norm=False, gamma=1e-5):
        super(SANGraphormer, self).__init__()

        # Original params
        self.use_feature = use_feature
        self.use_feature_GT = use_feature_GT
        self.use_time_feature = use_time_feature
        self.node_embedding = node_embedding
        self.max_z = max_z
        self.use_rpe = args.use_rpe
        self.num_step = args.num_step
        self.rpe_hidden_dim = args.rpe_hidden_dim

        # SAN params
        self.GT_n_heads = GT_n_heads
        self.full_graph = full_graph
        self.layer_norm = layer_norm
        self.gamma = gamma
        self.dropout = dropout

        # Calculate initial SAN dimension including timestamp
        initial_channels = hidden_channels # For z_emb
        if self.use_feature:
            initial_channels += hidden_channels # For first entry in feature vector
        if self.use_time_feature:
            initial_channels += num_features - 1 # For timestamps, i.e. rest of feature vector
        if self.node_embedding is not None:
            initial_channels += node_embedding.embedding_dim
        if self.use_rpe:
            initial_channels += self.rpe_hidden_dim * 2
            self.trainable_embedding = Sequential(Linear(in_features=self.num_step, out_features=self.rpe_hidden_dim), ReLU(), Linear(in_features=self.rpe_hidden_dim, out_features=self.rpe_hidden_dim))

        #Embeddings and normalizer
        self.z_embedding = Embedding(self.max_z, hidden_channels)
        self.x_embedding = Embedding(371, hidden_channels) # Hard coded the max number for court features since annoying
        self.e_embedding = Embedding(2, hidden_channels)  # For edge features
        self.h_embedding = Linear(initial_channels, hidden_channels) # To reshape h to hidden channels
        if self.layer_norm == True:
            self.initial_layer_norm = LayerNorm(hidden_channels)
       
        # SAN Layers
        self.layers = ModuleList([
            GraphTransformerLayer(
                self.gamma, hidden_channels, hidden_channels, 
                GT_n_heads, self.full_graph, dropout,
                layer_norm=self.layer_norm, batch_norm=True
            ) for _ in range(num_layers)
        ])

        # Graphormer layer
        input_dim = num_features if use_feature_GT else hidden_channels
        self.graphormer = Graphormer(n_layers=3,
                                     input_dim=input_dim,
                                     num_heads=args.num_heads,
                                     hidden_dim=hidden_channels,
                                     ffn_dim=hidden_channels,
                                     grpe_cross=args.grpe_cross,
                                     use_len_spd=args.use_len_spd,
                                     use_num_spd=args.use_num_spd,
                                     use_cnb_jac=args.use_cnb_jac,
                                     use_cnb_aa=args.use_cnb_aa,
                                     use_cnb_ra=args.use_cnb_ra,
                                     use_degree=args.use_degree,
                                     mul_bias=args.mul_bias,
                                     gravity_type=args.gravity_type)

        # Output layers
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)
        self.lin2.final_layer = True # Marker for final layer

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        self.graphormer.reset_parameters()

    def print_gradient_norms(self):
        tqdm.write("\nLarge gradients:")
        for name, param in self.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                if grad_norm > 10:  # Only print large gradients
                    tqdm.write(f"{name}: grad norm = {grad_norm:.4f}")

    def forward(self, data):

        # Validation
        device = next(self.parameters()).device
        if device != data.x.device:
             raise ValueError('Data and model on different devices!')

        if data.x.dim() < 2 or data.x.size(1) < 2:
            raise ValueError(f"Input features must have at least 2 dimensions with shape [N, 2+], got {data.x.shape}")

        #only use first entry of features as x
        x = data.x[:,0].long()
        #use rest as timestamp
        t = data.x[:,1:].long()
        z = data.z
        edge_index = data.edge_index
        batch = data.batch
        edge_weight = data.edge_weight
        node_id = data.node_id

        z_emb = self.z_embedding(z)
        x_emb = self.x_embedding(x)
        if z_emb.ndim == 3:  # in case z has multiple integer labels
            z_emb = z_emb.sum(dim=1)
        if self.use_feature and x is not None:
            h = torch.cat([z_emb, x_emb], 1)
        else:
            h = z_emb
        if self.use_time_feature:
            h = torch.cat([h, t], 1)
        if self.node_embedding is not None and node_id is not None:
            n_emb = self.node_embedding(node_id)
            h = torch.cat([h, n_emb], 1)
        if self.use_rpe:
            x_rpe = data.x_rpe
            x_rpe_emb = self.trainable_embedding(x_rpe).view(x_rpe.shape[0], -1)
            h = torch.cat([h, x_rpe_emb], 1)

        try:
            # DGL conversion
            g = pyg_to_dgl(data, self.full_graph, device)
        except:
            raise RuntimeError(f"Error during DGL conversion: {str(e)}")

        e = g.edata['feat'].flatten().long().to(device) # See SAN train_SBMs_node_classification.py
        e = self.e_embedding(e).to(device)
        h = self.h_embedding(h)

        #tqdm.write(f"Initial e stats: mean={e.abs().mean():.4f}, max={e.abs().max():.4f}")
        #tqdm.write(f"Initial h stats: mean={h.abs().mean():.4f}, max={h.abs().max():.4f}")
        if self.layer_norm:
            e = self.initial_layer_norm(e)
            h = self.initial_layer_norm(h)
        #tqdm.write(f"After initial norm h stats: mean={h.abs().mean():.4f}, max={h.abs().max():.4f}")

        for i, layer in enumerate(self.layers):
            h, e = layer(g, h, e)
            #tqdm.write(f"After layer {i} h stats: mean={h.abs().mean():.4f}, max={h.abs().max():.4f}")
        
        if True:  # center pooling
            _, center_indices = np.unique(batch.cpu().numpy(), return_index=True)
            h_src = h[center_indices]
            h_dst = h[center_indices + 1]
            h = h_src * h_dst
            if self.use_feature_GT:
                pair_data = abstract_pair_data(data)
            else:
                z_emb_src = z_emb[center_indices]
                z_emb_dst = z_emb[center_indices + 1]
                z_emb_pair = torch.cat([z_emb_src.unsqueeze(1), z_emb_dst.unsqueeze(1)], dim=1)
                pair_data = abstract_pair_data(data, z_emb_pair)  # 不用feature，就用z_emb替代
            h_graphormer = self.graphormer(pair_data)
            h_src_graphormer = h_graphormer[:, 0, :]
            h_dst_graphormer = h_graphormer[:, 1, :]
            h_graphormer = h_src_graphormer * h_dst_graphormer
        else:  # sum pooling
            h = global_add_pool(h, batch)
        h = torch.cat((h, h_graphormer), dim=-1)
        h = F.relu(self.lin1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.lin2(h)

        return h
    
class GCNGraphormer(torch.nn.Module):
    def __init__(self, args, hidden_channels, num_layers, max_z, num_features,
                 use_feature=False, use_feature_GT=True, node_embedding=None, dropout=0.5):
        super(GCNGraphormer, self).__init__()
        self.use_feature = use_feature
        self.use_feature_GT = use_feature_GT
        self.node_embedding = node_embedding
        self.max_z = max_z
        self.z_embedding = Embedding(self.max_z, hidden_channels)
        self.use_rpe = args.use_rpe
        self.num_step = args.num_step
        self.rpe_hidden_dim = args.rpe_hidden_dim

        self.convs = ModuleList()
        initial_channels = hidden_channels
        if self.use_feature:
            initial_channels += num_features
        if self.node_embedding is not None:
            initial_channels += node_embedding.embedding_dim
        if self.use_rpe:
            initial_channels += self.rpe_hidden_dim * 2
            self.trainable_embedding = Sequential(Linear(in_features=self.num_step, out_features=self.rpe_hidden_dim), ReLU(), Linear(in_features=self.rpe_hidden_dim, out_features=self.rpe_hidden_dim))

        self.input_normalizer = BatchNorm1d(initial_channels)
        self.convs.append(GCNConv(initial_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.dropout = dropout
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

        # 不用feature，就用z_emb代替，维度就是hidden_channels
        input_dim = num_features if use_feature_GT else hidden_channels
        self.graphormer = Graphormer(n_layers=3,
                                     input_dim=input_dim,
                                     num_heads=args.num_heads,
                                     hidden_dim=hidden_channels,
                                     ffn_dim=hidden_channels,
                                     grpe_cross=args.grpe_cross,
                                     use_len_spd=args.use_len_spd,
                                     use_num_spd=args.use_num_spd,
                                     use_cnb_jac=args.use_cnb_jac,
                                     use_cnb_aa=args.use_cnb_aa,
                                     use_cnb_ra=args.use_cnb_ra,
                                     use_degree=args.use_degree,
                                     mul_bias=args.mul_bias,
                                     gravity_type=args.gravity_type)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.graphormer.reset_parameters()

    def forward(self, data):
        x = data.x
        z = data.z
        edge_index = data.edge_index
        batch = data.batch
        edge_weight = data.edge_weight
        node_id = data.node_id

        z_emb = self.z_embedding(z)
        if z_emb.ndim == 3:  # in case z has multiple integer labels
            z_emb = z_emb.sum(dim=1)
        if self.use_feature and x is not None:
            h = torch.cat([z_emb, x.to(torch.float)], 1)
        else:
            h = z_emb
        if self.node_embedding is not None and node_id is not None:
            n_emb = self.node_embedding(node_id)
            h = torch.cat([h, n_emb], 1)
        if self.use_rpe:
            x_rpe = data.x_rpe
            x_rpe_emb = self.trainable_embedding(x_rpe).view(x_rpe.shape[0], -1)
            h = torch.cat([h, x_rpe_emb], 1)

        h = self.input_normalizer(h)
        for conv in self.convs[:-1]:
            h = conv(h, edge_index, edge_weight)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.convs[-1](h, edge_index, edge_weight)

        if True:  # center pooling
            _, center_indices = np.unique(batch.cpu().numpy(), return_index=True)
            h_src = h[center_indices]
            h_dst = h[center_indices + 1]
            h = h_src * h_dst
            if self.use_feature_GT:
                pair_data = abstract_pair_data(data)
            else:
                z_emb_src = z_emb[center_indices]
                z_emb_dst = z_emb[center_indices + 1]
                z_emb_pair = torch.cat([z_emb_src.unsqueeze(1), z_emb_dst.unsqueeze(1)], dim=1)
                pair_data = abstract_pair_data(data, z_emb_pair)  # 不用feature，就用z_emb替代
            h_graphormer = self.graphormer(pair_data)
            h_src_graphormer = h_graphormer[:, 0, :]
            h_dst_graphormer = h_graphormer[:, 1, :]
            h_graphormer = h_src_graphormer * h_dst_graphormer
        else:  # sum pooling
            h = global_add_pool(h, batch)
        h = torch.cat((h, h_graphormer), dim=-1)
        h = F.relu(self.lin1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.lin2(h)

        return h