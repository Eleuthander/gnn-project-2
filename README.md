# CPSC 486 Final Project

## Summary

This is the code repository for my CPSC486 final project substituting replacing a GCN module with Kreuzer et al.'s[^1] SAN in Shi et al.'s[^2] SIEG architecture. 

Run `run_program.sh` to reproduce my results.


The remaining folders and files in the repository are:
- `/checkpoints`: Stores model checkpoints from `train_test.py`.
- `/logs`: Logs console output from `train_test.py`.
- `/metrics`: Stores performance metrics for models from `train_test.py`.
- `/utilities/graphormer.py`: Taken from Shi et al. Creates the Graphormer module for the SIEG architecture.
- `utilities/collection.py`: A utility file for treating tensors like sets.
- `graph_transformer_layer.py`: Taken from Kreuzer et al. Creates transformer_layers for the SAN.
- `'preprocess.py`: Taken from Shi et al. Extracts statistics from sugraphs.
- `pyg_to_dgl.py`: Turns PyG graphs into DGL graphs for processing by the SAN.
- `utils.py`: Taken from Shi et al and modified. Contains miscellaneous processing functions.
- `cn.py`: Predicts edges using the common neighbor heuristic.
- `dataset.py`: Taken from Shi et al. Creates dataloaders for loading subgraphs around target edges.
- `models.py`: Contains the new SANGraphormer and takes GCNGraphormer from Shi et al.
- `train_test.py`: Contains the train-test loop. 

In the rest of this README I summarize the sieg architecture and then go through the key files in more detail.

## SIEG Architecture

The SIEG architecture is designed for link prediction. It calculates the k-hop neighborhoods of the two nodes and then passes them to two components. The first is a standard GNN. The second is a modified version of Graphormer that takes in various statistics about the notes, such as their Jacquard indices, the length of the shortest path between them, etc.

The baseline SIEG architecture uses a GCN for its GNN and is called "GCNGraphormer." I modify it to use a SAN instead, and so call it "SANGraphormer."

## File Descriptions

### pyg_to_dgl.py

Kreuzer et al.'s code for SAN uses DGL graphs to take advantage of their efficient edge processing. SIEG, on the other hand, uses very efficient code on PyG graphs. To retain the efficiencies of both, I created a file that converts a batch of PyG graphs to on large DGL graph for processing by SAN.

SAN has an option to fully connect the graphs in the batch. To make this as efficient as possible, I use vectorization to create a new edge_index for the fully conencted subgraphs in the batch. Then I assign edge features to the edges in teh original graph, using hashing to optimize the retrieval process for the original edges.

### utils.py and preprocess.py

These contains miscellaneous utilities for graph processing. For example:
- SIEG uses DRNL node labelling to label nodes before passing them to the Graphormer, so `utils.py` contains a DRNL labeler. 
- `utils.py` contains optimized utilities for retrieving the k-hop neighborhood of two target nodes for link prediction.
- `utils.py` contains basic functions for splitting edges into training, validation, and test edges for use by the dataset classes in dataset.py.
- `preprocess.py` takes in a subgraph and calculates various statistics between the target nodes to be fed to the Graphormer module.

### dataset.py

SIEG functions by retrieving the subgraph around the target nodes for link prediction. The classes in this file handle that retrieval. 

I use SEALDynamicDataset, which processes the entire graph initially and creates efficient representations for calculating the k-hop neighborhoods, such as sparse adjacency matrices. When a dataloader calls get it then uses the utilities in utils.py and preprocess.py to create and prepare these subgraphs before passing them to the model for prediction.

### models.py

This file contains the model architectures. GCNGraphormer is taken from Shi et al., and processes the initial embeddings based on various initliaization arguments before passing them to the GCN layers, while the graphormer does its own predictions, and them the two results are combined.

SNAGraphormer is similar but replaces the GCN layers with transformer layers from graph_transformer_layer.py. 

### train_test.py



## References

[^1]: D. Kreuzer, D. Beaini, W. L. Hamilton, V. Létourneau, and P. Tossou. *Rethinking graph transformers with spectral attention.* In Proceedings of the 35th International Conference on Neural Information Processing Systems, NIPS ’21, Red Hook, NY, USA, 2024. Curran Associates Inc. ISBN 9781713845393.

[^2]: See L. Shi, B. Hu, D. Zhao, J. He, Z. Zhang, and J. Zhou. *Structural information enhanced graph
representation for link prediction.* Proceedings of the AAAI Conference on Artificial Intelligence,
38(13):14964–14972, Mar. 2024. doi: 10.1609/aaai.v38i13.29417. URL https://ojs.aaai.org/index.php/AAAI/article/view/29417.