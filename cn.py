import numpy as np
import scipy.sparse as ssp
import torch
from torch_geometric.loader import DataLoader as PygDataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
from utilities.utils import do_edge_split

def common_neighbors_large(A, edge_index, batch_size=1000, cn_types=['in', 'out', 'undirected']):
    """Same implementation as before"""
    num_nodes = A.shape[0]
    A = A.tocsr()
    A_t = A.transpose().tocsr()

    if 'undirected' in cn_types:
        x_ind, y_ind = A.nonzero()
        weights = np.array(A[x_ind, y_ind]).flatten()
        if not (A.todense() == A_t.todense()).all():
            weights_concat = np.concatenate([weights, weights])
            indices_concat = np.concatenate([x_ind, y_ind])
            indices_concat_reverse = np.concatenate([y_ind, x_ind])
            A_undirected = ssp.csr_matrix(
                (weights_concat, (indices_concat, indices_concat_reverse)),
                shape=(num_nodes, num_nodes))
        else:
            A_undirected = A

    link_loader = PygDataLoader(range(edge_index.size(1)), batch_size)
    multi_type_scores = []
    
    for cn_type in cn_types:
        scores = []
        for ind in tqdm(link_loader, desc=f'Processing {cn_type} neighbors'):
            src, dst = edge_index[0, ind], edge_index[1, ind]
            
            if cn_type == 'undirected':
                cur_scores = np.array(np.sum(A_undirected[src].multiply(A_undirected[dst]), 1)).flatten()
            elif cn_type == 'in':
                cur_scores = np.array(np.sum(A[src].multiply(A[dst]), 1)).flatten()
            elif cn_type == 'out':
                cur_scores = np.array(np.sum(A_t[src].multiply(A_t[dst]), 1)).flatten()
            
            scores.append(cur_scores)
            
        multi_type_scores.append(torch.FloatTensor(np.concatenate(scores, 0)))
    
    return torch.stack(multi_type_scores), edge_index

def evaluate_common_neighbors_large(train_edges, pos_test_edges, neg_test_edges, num_nodes):
    """Updated evaluation function with both AUC and AP scores"""
    # Create sparse adjacency matrix from training edges
    train_adj = ssp.csr_matrix(
        (np.ones(train_edges.size(1)),
         (train_edges[0].numpy(), train_edges[1].numpy())),
        shape=(num_nodes, num_nodes),
        dtype=np.float32)

    # Combine positive and negative edges for processing
    test_edges = torch.cat([pos_test_edges, neg_test_edges], dim=1)

    # Calculate CN scores for all types
    scores, _ = common_neighbors_large(
        train_adj,
        test_edges,
        batch_size=1000,
        cn_types=['in', 'out', 'undirected']
    )

    # Create labels array (1 for positive edges, 0 for negative edges)
    labels = torch.zeros(pos_test_edges.size(1) + neg_test_edges.size(1))
    labels[:pos_test_edges.size(1)] = 1
    
    # Calculate both AUC and AP scores for each type
    results = {}
    for idx, cn_type in enumerate(['in', 'out', 'undirected']):
        scores_np = scores[idx].numpy()
        labels_np = labels.numpy()
        results[cn_type] = {
            'auc': roc_auc_score(labels_np, scores_np),
            'ap': average_precision_score(labels_np, scores_np)
        }

    return results

# Main execution
if __name__ == "__main__":
    print("Loading data...")
    data = torch.load("appellate_graph_final.pt")
    print(data)
    
    print("\nPerforming edge split...")
    dataset = [data]
    split_edge = do_edge_split(dataset, val_ratio=0.1, test_ratio=0.2, neg_ratio=1)
    print(f"Train edges: {split_edge['train']['edge'].size(0)}")

    print("\nEvaluating Common Neighbors...")
    results = evaluate_common_neighbors_large(
        train_edges=split_edge["train"]["edge"].t(),
        pos_test_edges=split_edge["test"]["edge"].t(),
        neg_test_edges=split_edge["test"]["edge_neg"].t(),
        num_nodes=data.num_nodes
    )

    print("\nCommon Neighbors scores:")
    for cn_type in ['in', 'out', 'undirected']:
        print(f"\n{cn_type.capitalize()} neighbors:")
        print(f"  AUC: {results[cn_type]['auc']:.4f}")
        print(f"  AP:  {results[cn_type]['ap']:.4f}")