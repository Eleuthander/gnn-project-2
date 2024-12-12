import torch
from torch_geometric.loader import DataLoader as PygDataLoader
from dataset import SEALDynamicDataset, SEALIterableDataset
from utilities.utils import *
from models import GCNGraphormer, SANGraphormer
import argparse
from types import SimpleNamespace
from utilities.preprocess import preprocess
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
    
    #Training args
    runs = 1,
    epochs = 2,

    # Model args
    model = 'GCNGraphormer',
    hidden_channels=64, # transformer block dims (64-128 reasonable) \ embedding dim in GCN (32 reasonable)
    num_layers=6, # number of GNN module layers (6 reasonable for SAN, 3 for GCN)
    dropout=0.2,
    full_graph=True, # whether to add fake edges to SAN
    gamma=1e-6, # between 0 and 1:  0 is fully sparse, 1 fully (1e-7 through 1e-5 reasonable)
    GT_n_heads = 4, # Num heads for SAN module (4-8 reasonable)
    num_heads=4, # Num heads for graphormer module (4 used in SIEG)

    # Subgraph args
    num_hops = 2,
    max_nodes_per_hop = 32,
    max_z=1000,  # Max value for structural encoding

    #Batching args
    batch_size=512,
    num_workers=8,
    pin_memory=True,
    prefetch_factor=2,
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
)

# Create test dataset
#print("Creating test dataset...")
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
#     sample_type=args.sample_type,
#     shuffle=True,
# )
# print("Created test dataset")

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

print(f"Total trainable parameters in model: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

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

def benchmark_configuration(model, data, split_edge, args, device, run_time=60, log_file=None):
    """
    Benchmark a specific configuration for run_time seconds.
    Logs results to specified file.
    """
    log_file.write(f"\nBenchmarking configuration:\n")
    log_file.write(f"Batch size: {args.batch_size}\n")
    log_file.write(f"Num workers: {args.num_workers}\n")
    log_file.write(f"Prefetch factor: {args.prefetch_factor}\n")
    log_file.write(f"Sample type: {args.sample_type}\n")

    # Create dataset with current config
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

    # Create dataloader with current config
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
            prefetch_factor=None
        )

    model.eval()
    start_time = time.time()
    iterations = 0
    
    try:
        with torch.no_grad():
            for batch in test_loader:
                if time.time() - start_time > run_time:
                    break
                    
                batch = batch.to(device)
                _ = model(batch)
                iterations += 1

                del batch
                torch.cuda.empty_cache()

    except Exception as e:
        log_file.write(f"Error during benchmarking: {str(e)}\n")
        return 0, 0

    total_time = time.time() - start_time
    iterations_per_second = iterations / total_time

    log_file.write(f"Iterations completed: {iterations}\n")
    log_file.write(f"Iterations per second: {iterations_per_second:.2f}\n")
    log_file.flush()  # Ensure writes are flushed to file

    return iterations, iterations_per_second

def find_best_configuration(model, data, split_edge, base_args, device, log_path="benchmark_results.txt"):
    """
    Test different configurations and return the best one.
    Logs all results to specified file.
    """
    configs = {
        'batch_size': [64, 128, 256, 512],
        'num_workers': [4, 8, 12],
        'prefetch_factor': [2, 4, 8],
        'sample_type': [0, 1, 2]  # 0 for standard, 2 for k_hop_subgraph_sampler_tensor
    }

    results = []
    best_perf = 0
    best_config = None

    with open(log_path, 'w') as log_file:
        log_file.write("Starting configuration search...\n")
        log_file.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for batch_size in configs['batch_size']:
            for num_workers in configs['num_workers']:
                # Skip prefetch_factor when num_workers = 0
                prefetch_factors = configs['prefetch_factor'] if num_workers > 0 else [None]
                for prefetch_factor in prefetch_factors:
                    for sample_type in configs['sample_type']:
                        # Create args copy with current config
                        current_args = SimpleNamespace(**vars(base_args))
                        current_args.batch_size = batch_size
                        current_args.num_workers = num_workers
                        current_args.prefetch_factor = prefetch_factor
                        current_args.sample_type = sample_type

                        iterations, iter_per_sec = benchmark_configuration(
                            model, data, split_edge, current_args, device, log_file=log_file
                        )

                        result = {
                            'batch_size': batch_size,
                            'num_workers': num_workers,
                            'prefetch_factor': prefetch_factor,
                            'sample_type': sample_type,
                            'iterations': iterations,
                            'iterations_per_second': iter_per_sec
                        }
                        results.append(result)

                        if iter_per_sec > best_perf:
                            best_perf = iter_per_sec
                            best_config = result

                        # Clear GPU memory between runs
                        torch.cuda.empty_cache()
                        gc.collect()

        log_file.write("\n" + "="*50 + "\n")
        log_file.write("Best configuration found:\n")
        log_file.write(f"Batch size: {best_config['batch_size']}\n")
        log_file.write(f"Num workers: {best_config['num_workers']}\n")
        log_file.write(f"Prefetch factor: {best_config['prefetch_factor']}\n")
        log_file.write(f"Sample type: {best_config['sample_type']}\n")
        log_file.write(f"Iterations per second: {best_config['iterations_per_second']:.2f}\n")
        log_file.write("="*50 + "\n")

    return best_config, results

# Execution
# print("Finding optimal configuration...")
# best_config, all_results = find_best_configuration(model, data, split_edge, args, device)

predictions = evaluate_model(model, test_loader, device, monitor_interval=3)