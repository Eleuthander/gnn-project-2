import torch
from torch_geometric.loader import DataLoader as PygDataLoader
from dataset import SEALDynamicDataset, SEALIterableDataset
from utils import *
from models import GCNGraphormer, SANGraphormer
import argparse
from types import SimpleNamespace
from preprocess import preprocess
import time
from functools import partial
from tqdm import tqdm
import sys

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
    hidden_channels=32, # transformer block dims (64-128 reasonable) \ embedding dim in GCN (32 reasonable)
    num_layers=3,
    dropout=0.2,
    full_graph=True, # whether to add fake edges to SAN
    gamma=1e-1, # between 0 and 1:  0 is fully sparse, 1 fully (10e-7 through 1-05 reasonable)
    GT_n_heads = 4, # Num heads for SAN module (4-8 reasonable)
    num_heads=4, # Num heads for graphormer module (4 used in SIEG)

    # Subgraph args
    num_hops = 2,
    max_nodes_per_hop = 10,
    max_z=1000,  # Max value for structural encoding

    #Batching args
    batch_size=256,
    num_workers=6,
    pin_memory=True,
    prefetch_factor=2,
    sample_type = 2, # 0 is standard; 2 is "k_hop_subgraph_sampler_tensor"

    # Features used
    use_feature=True, # whether to use features in the GNN
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
        prefetch_factor=args.prefetch_factor
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
    model = GCNGraphormer(
        args=args,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        max_z=args.max_z,
        num_features=data.x.size(1),
        use_feature=args.use_feature,
        use_feature_GT=args.use_feature_GT,
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

import psutil
import torch
from tqdm import tqdm
import time
from threading import Thread
import GPUtil

class ResourceMonitor:
    def __init__(self, progress_bar, interval=5):
        """
        Initialize resource monitor
        Args:
            progress_bar (tqdm): tqdm progress bar to update
            interval (int): Monitoring interval in seconds
        """
        self.interval = interval
        self.running = False
        self.thread = None
        self.progress_bar = progress_bar
        
    def get_gpu_usage(self):
        """Get GPU utilization if available"""
        try:
            gpu = GPUtil.getGPUs()[0]  # Assuming first GPU
            return f"GPU: {gpu.load*100:.1f}% | Mem: {gpu.memoryUtil*100:.1f}%"
        except:
            return "GPU stats N/A"
    
    def get_cpu_usage(self):
        """Get CPU utilization"""
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        return f"CPU: {cpu_percent:.1f}% | RAM: {memory.percent:.1f}%"
    
    def monitor(self):
        """Main monitoring loop"""
        while self.running:
            gpu_stats = self.get_gpu_usage()
            cpu_stats = self.get_cpu_usage()
            self.progress_bar.set_postfix_str(f"{gpu_stats} | {cpu_stats}")
            time.sleep(self.interval)
    
    def start(self):
        """Start monitoring in background thread"""
        self.running = True
        self.thread = Thread(target=self.monitor, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop monitoring"""
        self.running = False
        if self.thread:
            self.thread.join()

def evaluate_model(model, test_loader, device, monitor_interval=5):
    """
    Evaluate model with resource monitoring
    """
    model.eval()
    all_preds = []

    # Create progress bar first
    pbar = tqdm(test_loader, desc="Evaluating", ncols=120)  # Increased ncols to accommodate stats
    
    # Initialize and start resource monitor with progress bar
    monitor = ResourceMonitor(pbar, interval=monitor_interval)
    monitor.start()
    
    try:
        with torch.no_grad():
            for batch in pbar:
                batch = batch.to(device)
                pred = model(batch)
                all_preds.append(pred.cpu())
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
    finally:
        # Always stop monitoring
        monitor.stop()
        pbar.close()
    
    if all_preds:
        all_preds = torch.cat(all_preds, dim=0)
    return all_preds


print("Evaluating Model")
all_preds = evaluate_model(model, test_loader, device, monitor_interval=5)

print(f"Generated predictions shape: {all_preds.shape}")