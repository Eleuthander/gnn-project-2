import dgl
import networkx as nx
import torch

def pyg_to_dgl(data, full_graph):

    num_nodes = data.num_nodes
    num_edges = data.num_edges
    edge_index = data.edge_index

    if full_graph:
        
        g = dgl.from_networkx(nx.complete_graph(num_nodes))

        #Populate edge features w/ 0s
        g.edata['feat']=torch.zeros(g.number_of_edges(), dtype=torch.long)
        g.edata['real']=torch.zeros(g.number_of_edges(), dtype=torch.long)
        
        #Copy real edge data over
        g.edges[edge_index[0], edge_index[1]].data['feat'] = torch.ones(g.edata['feat'].shape[0], dtype=torch.long) 
        g.edges[edge_index[0], edge_index[1]].data['real'] = torch.ones(g.edata['feat'].shape[0], dtype=torch.long) 
    
    else:

        g = dgl.graph((edge_index[0], edge_index[1]), num_nodes=data.num_nodes)
        g.edata['feat'] = torch.ones(num_edges, 1)
    
    return g


