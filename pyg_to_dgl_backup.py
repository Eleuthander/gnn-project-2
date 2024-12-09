import dgl
import networkx as nx
import torch

def pyg_to_dgl(data, full_graph, device):

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

        # Create the DGL graph
        g = dgl.graph((src_all, dst_all), num_nodes=num_nodes, device=device)


        #Populate added edge features w/ 0s and original edge features w/ 1s
        edge_ids = g.edge_ids(edge_index[0], edge_index[1])

        mask = torch.zeros(num_edges, dtype=torch.bool, device=device)
        mask[edge_ids] = True

        g.edata['feat'] = mask.long()
        g.edata['real'] = mask.long()

    else:
        num_edges = data.num_edges
        g = dgl.graph((edge_index[0], edge_index[1]), num_nodes=data.num_nodes)
        g.edata['feat'] = torch.ones(num_edges, 1, device=device)
    
    return g


