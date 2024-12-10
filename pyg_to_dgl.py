import dgl
import networkx as nx
import torch

def pyg_to_dgl(data, full_graph, device):

    # Input validation
    if not hasattr(data, 'num_nodes') or not hasattr(data, 'edge_index') or not hasattr(data, 'batch'):
        raise ValueError("Input PyG data must have num_nodes, edge_index, and batch attributes")
    
    if data.edge_index.shape[0] != 2:
        raise ValueError(f"Edge index must have shape [2, num_edges], got {data.edge_index.shape}")

    num_nodes = data.num_nodes
    edge_index = data.edge_index.to(device)
    batch_size = data.batch.max().item() + 1

    if full_graph:

        node_counts = torch.bincount(data.batch, minlength=batch_size)
        node_offsets = torch.zeros(batch_size, device=device)
        node_offsets[1:] = torch.cumsum(node_counts[:-1], dim=0)
        num_edges = (node_counts * node_counts).sum().item()
        
        src_all = torch.empty(num_edges, dtype=torch.long, device=device)
        dst_all = torch.empty(num_edges, dtype=torch.long, device=device)

        current_idx = 0
        for i in range(batch_size):
            n = node_counts[i].item()
            if n == 0:  # Skip empty graphs
                continue
                
            start = node_offsets[i].item()
            nodes = torch.arange(start, start + n, device=device)
            
            # Create indices for this batch's edges
            end_idx = current_idx + n * n
            
            # Vectorized edge creation
            src_all[current_idx:end_idx] = nodes.repeat_interleave(n)
            dst_all[current_idx:end_idx] = nodes.repeat(n)
            
            current_idx = end_idx

        # Remove self loops
        no_self_loops_mask = src_all != dst_all
        src_all = src_all[no_self_loops_mask]
        dst_all = dst_all[no_self_loops_mask]
        num_edges = num_edges - num_nodes

        # Create the DGL graph
        g = dgl.graph((src_all, dst_all), num_nodes=num_nodes, device=device).to(device)

        # Fast edge feature assignment using simple hashing
        edge_hash = src_all * num_nodes + dst_all
        orig_edge_hash = edge_index[0] * num_nodes + edge_index[1]

        edge_features = torch.zeros(num_edges, dtype=torch.long, device=device)
        edge_features.index_put_(
            (torch.bucketize(orig_edge_hash, edge_hash),),
            torch.ones(len(orig_edge_hash), dtype=torch.long, device=device)
        )

        g.edata['feat'] = edge_features
        g.edata['real'] = edge_features

    else:
        num_edges = data.num_edges
        g = dgl.graph((edge_index[0], edge_index[1]), num_nodes=num_nodes).to(device)
        g.edata['feat'] = torch.ones(num_edges, 1, device=device)
    
    return g


