import torch
from torch_geometric.loader import DataLoader as PygDataLoader
from dataset import SEALDynamicDataset, SEALIterableDataset
from utils import *
from models import GCNGraphormer, SANGraphormer
import argparse
from types import SimpleNamespace
from preprocess import preprocess
from usage_monitoring import ResourceMonitor
import time
from functools import partial
from tqdm import tqdm
import sys
from torch.cuda.amp import autocast # for preprocessing
import gc

#Suppress annoying torch.load warning
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Load data
data = torch.load('fed_cites_graph_processed.pt')
print(data)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create default args for GCNGraphormer
args = SimpleNamespace(

    # Model args
    model = 'GCNGraphormer',
    hidden_channels=64, # transformer block dims (64-128 reasonable) \ embedding dim in GCN (32 reasonable)
    num_layers=3,
    dropout=0.2,
    full_graph=True, # whether to add fake edges to SAN
    gamma=1e-6, # between 0 and 1:  0 is fully sparse, 1 fully (10e-7 through 1-05 reasonable)
    GT_n_heads = 4, # Num heads for SAN module (4-8 reasonable)
    num_heads=4, # Num heads for graphormer module (4 used in SIEG)

    # Subgraph args
    num_hops = 2,
    max_nodes_per_hop = 10,
    max_z=1000,  # Max value for structural encoding

    #Batching args
    batch_size=128,
    num_workers=12,
    pin_memory=True,
    prefetch_factor=4,
    persistent_workers=True,
    sample_type = 2, # 0 is standard; 2 is "k_hop_subgraph_sampler_tensor"

    # Features used
    use_feature=True, # whether to use features in the GNN
    use_time_feature=True, # whether to use timestamps as positional encodings
    use_feature_GT=True, # whether to use features in the graphormer module
    use_edge_weight=False,
    grpe_cross=False,
    use_len_spd=True,
    use_num_spd=False,
    use_cnb_jac=True,
    use_cnb_aa=True,
    use_cnb_ra=False,
    use_degree=True,
    mul_bias=True,
    gravity_type=0,
    use_rpe = False,
    rpe_hidden_dim = 1,
    num_step = 1
)

# Create splits
print("Creating splits...")
dataset = [data]  # Wrap in list for do_edge_split
split_edge = do_edge_split(dataset, fast_split=True, val_ratio=0.01, test_ratio=0.01, neg_ratio=1)
data.edge_index = split_edge['train']['edge'].t()
print("Created splits")

#Determine preprocessing function
preprocess_fn = partial(preprocess,
                        grpe_cross=args.grpe_cross,
                        use_len_spd=args.use_len_spd,
                        use_num_spd=args.use_num_spd,
                        use_cnb_jac=args.use_cnb_jac,
                        use_cnb_aa=args.use_cnb_aa,
                        use_cnb_ra=args.use_cnb_ra,
                        use_degree=args.use_degree,
                        gravity_type=args.gravity_type,
                )

print("Creating test dataset...")
test_dataset = SEALIterableDataset(
    root='./temp_seal_data',
    data=data,
    split_edge=split_edge,
    num_hops=args.num_hops,
    split='test',
    directed=True,
    use_coalesce=False,
    node_label='drnl',
    ratio_per_hop=1.0,
    max_nodes_per_hop=args.max_nodes_per_hop,
    preprocess_fn=preprocess_fn,
    sample_type=args.sample_type,
    shuffle=False,
    slice_type=0
)
print("Created test dataset...")

print("Creating dataloader...")
if args.num_workers > 0:
    test_loader = PygDataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers
    )
else:
    test_loader = PygDataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=args.pin_memory,
        prefetch_factor=None  # Must be None when num_workers=0
    )
print("Created dataloader...")

# # Create test dataset
# print("Creating test dataset...")
# test_dataset = SEALDynamicDataset(
#     root='./temp_seal_data',
#     data=data,
#     split_edge=split_edge,
#     num_hops=args.num_hops,
#     split='test',
#     directed=True,
#     use_coalesce=False,
#     node_label='drnl',
#     ratio_per_hop=1.0,
#     max_nodes_per_hop=args.max_nodes_per_hop,
#     preprocess_fn=preprocess_fn,
#     sample_type=args.sample_type
# )
# print("Created test dataset")

# # Create dataloader
# print("Creating dataloader...")
# test_loader = PygDataLoader(
#     test_dataset,
#     batch_size=args.batch_size,
#     shuffle=False,
#     num_workers=args.num_workers,
#     pin_memory=True,
#     prefetch_factor=args.prefetch_factor
# )
# print("Created dataloader")

# Initialize model
print("Creating Model")
if args.model == 'GCNGraphormer':
    model = GCNGraphormer(
        args=args,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        max_z=args.max_z,
        num_features=data.x.size(1),
        use_feature=args.use_feature,
        use_feature_GT=args.use_feature_GT,
        node_embedding=None,
        dropout=args.dropout
    ).to(device)
elif args.model == 'SANGraphormer':
    model = SANGraphormer(
        args=args,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        max_z=args.max_z,
        num_features=data.x.size(1),
        use_feature=args.use_feature,
        use_feature_GT=args.use_feature_GT,
        use_time_feature=args.use_time_feature,
        node_embedding=None,
        dropout=args.dropout,
        GT_n_heads=args.GT_n_heads,
        full_graph=args.full_graph,
        gamma=args.gamma
    ).to(device)
else:
    print("Error: Model not recognized.")
    exit()
print("Created Model")

# Evaluate
def evaluate_model(model, test_loader, device, monitor_interval=5):
    """
    Evaluate model with resource monitoring
    """
    model.eval()
    all_preds = []

    # Create progress bar
    pbar = tqdm(test_loader, desc="Evaluating", ncols=120)
    
    # Initialize and start resource monitor with progress bar
    monitor = ResourceMonitor(pbar, interval=monitor_interval)
    monitor.start()
    
    # Calculate next batch on CPU as GPU processes this batch
    # try:
    #     with autocast(), torch.no_grad():
    #         for batch in pbar:
    #             batch = batch.to(device, non_blocking=True)  # Enable async transfer
    #             pred = model(batch)
    #             all_preds.append(pred.cpu())

    #             # Clear GPU memory
    #             del batch
    #             del pred
    #             torch.cuda.empty_cache()

    try:
        with torch.no_grad():
            for batch in pbar:
                batch = batch.to(device)
                pred = model(batch)
                all_preds.append(pred.cpu())

                # Clear GPU memory
                del batch
                del pred
                torch.cuda.empty_cache()  # Clear any remaining GPU memory
                
                # Optional: force garbage collection; can slow things down
                # gc.collect()

    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
    finally:
        # Always stop monitoring
        monitor.stop()
        pbar.close()

        # Final cleanup
        torch.cuda.empty_cache()
        gc.collect()
    
    if all_preds:
        all_preds = torch.cat(all_preds, dim=0)
    return all_preds


print("Evaluating Model")
all_preds = evaluate_model(model, test_loader, device, monitor_interval=5)

print(f"Generated predictions shape: {all_preds.shape}")