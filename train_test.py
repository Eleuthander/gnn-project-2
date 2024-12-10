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
from torch.cuda.amp import autocast, GradScaler  # for mixed precision training
import gc
import os
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import random
import math
import torch_sparse as ssp
from threading import Thread
import GPUtil
import psutil
import logging
from datetime import datetime

# Suppress annoying torch.load warning
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------
# Logging Configuration
# ---------------------------

def setup_logging(log_dir='./logs'):
    """
    Set up logging to file and console.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create a unique log file name with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join(log_dir, f'training_{timestamp}.log')
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(f"Logging initialized. Log file: {log_filename}")

# ---------------------------
# Set Random Seeds
# ---------------------------

def set_seed(seed=234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    logging.info(f"Random seed set to {seed}")

# ---------------------------
# Evaluation Metrics
# ---------------------------

def evaluate_metrics(y_true, y_pred):
    """
    Compute AUC and Average Precision (AP) scores.
    """
    try:
        auc = roc_auc_score(y_true, y_pred)
    except ValueError:
        auc = 0.0
    try:
        ap = average_precision_score(y_true, y_pred)
    except ValueError:
        ap = 0.0
    return auc, ap

# ---------------------------
# Parameter Count
# ---------------------------

def count_parameters(model):
    """
    Count the number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ---------------------------
# Training Function
# ---------------------------

def train(model, loader, optimizer, scheduler, criterion, device, scaler=None):
    """
    Train the model for one epoch.
    """
    model.train()
    total_loss = 0
    all_labels = []
    all_preds = []

    pbar = tqdm(loader, desc="Training", ncols=120)
    monitor = ResourceMonitor(pbar, interval=5)
    monitor.start()

    try:
        for batch in pbar:
            batch = batch.to(device)

            optimizer.zero_grad()
            
            if scaler is not None:
                with autocast():
                    out = model(batch)
                    loss = criterion(out, batch.y.float())
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                with autocast():
                    out = model(batch)
                    loss = criterion(out, batch.y.float())
                loss.backward()
                optimizer.step()
            
            # Step the scheduler per batch
            scheduler.step()

            total_loss += loss.item()
            
            preds = torch.sigmoid(out).detach().cpu()
            labels = batch.y.detach().cpu()
            all_preds.append(preds)
            all_labels.append(labels)
            
            # Update progress bar postfix with loss and resource usage
            current_loss = total_loss / (len(all_preds))
            pbar.set_postfix({'loss': f"{current_loss:.4f}"})

            # Clear GPU memory
            del batch
            del out
            torch.cuda.empty_cache()

    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
    finally:
        # Always stop monitoring
        monitor.stop()
        pbar.close()

        # Final cleanup
        torch.cuda.empty_cache()
        gc.collect()

    if all_preds:
        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()
        auc, ap = evaluate_metrics(all_labels, all_preds)
    else:
        auc, ap = 0, 0

    avg_loss = total_loss / len(loader)
    return avg_loss, auc, ap

# ---------------------------
# Validation Function
# ---------------------------

def validate(model, loader, criterion, device):
    """
    Validate the model on the validation set.
    """
    model.eval()
    total_loss = 0
    all_labels = []
    all_preds = []

    pbar = tqdm(loader, desc="Validation", ncols=120)
    monitor = ResourceMonitor(pbar, interval=5)
    monitor.start()

    try:
        with torch.no_grad():
            for batch in pbar:
                batch = batch.to(device)
                with autocast():
                    out = model(batch)
                    loss = criterion(out, batch.y.float())
                total_loss += loss.item()
                
                preds = torch.sigmoid(out).detach().cpu()
                labels = batch.y.detach().cpu()
                all_preds.append(preds)
                all_labels.append(labels)
                
                # Update progress bar postfix with loss and resource usage
                current_loss = total_loss / (len(all_preds))
                pbar.set_postfix({'loss': f"{current_loss:.4f}"})

                # Clear GPU memory
                del batch
                del out
                torch.cuda.empty_cache()

    except KeyboardInterrupt:
        logging.info("Validation interrupted by user")
    finally:
        # Always stop monitoring
        monitor.stop()
        pbar.close()

        # Final cleanup
        torch.cuda.empty_cache()
        gc.collect()

    if all_preds:
        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()
        auc, ap = evaluate_metrics(all_labels, all_preds)
    else:
        auc, ap = 0, 0

    avg_loss = total_loss / len(loader)
    return avg_loss, auc, ap

# ---------------------------
# Test Function
# ---------------------------

def test(model, loader, criterion, device):
    """
    Test the model on the test set.
    """
    model.eval()
    total_loss = 0
    all_labels = []
    all_preds = []

    pbar = tqdm(loader, desc="Testing", ncols=120)
    monitor = ResourceMonitor(pbar, interval=5)
    monitor.start()

    try:
        with torch.no_grad():
            for batch in pbar:
                batch = batch.to(device)
                with autocast():
                    out = model(batch)
                    loss = criterion(out, batch.y.float())
                total_loss += loss.item()
                
                preds = torch.sigmoid(out).detach().cpu()
                labels = batch.y.detach().cpu()
                all_preds.append(preds)
                all_labels.append(labels)
                
                # Update progress bar postfix with loss and resource usage
                current_loss = total_loss / (len(all_preds))
                pbar.set_postfix({'loss': f"{current_loss:.4f}"})

                # Clear GPU memory
                del batch
                del out
                torch.cuda.empty_cache()

    except KeyboardInterrupt:
        logging.info("Testing interrupted by user")
    finally:
        # Always stop monitoring
        monitor.stop()
        pbar.close()

        # Final cleanup
        torch.cuda.empty_cache()
        gc.collect()

    if all_preds:
        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()
        auc, ap = evaluate_metrics(all_labels, all_preds)
        logging.info(f"Test Loss: {total_loss / len(loader):.4f}, AUC: {auc:.4f}, AP: {ap:.4f}")
    else:
        auc, ap = 0, 0

    return total_loss / len(loader), auc, ap

# ---------------------------
# Main Function
# ---------------------------

def main():
    # Setup logging
    setup_logging()

    # Set random seed
    set_seed(234)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Load data
    data = torch.load('fed_cites_graph_processed.pt')
    logging.info(f"Loaded data: {data}")

    # Create default args for GCNGraphormer
    args = SimpleNamespace(
        # Model args
        model = 'GCNGraphormer',  # or 'SANGraphormer'
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

        # Batching args
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
        num_step = 1,

        # Training args
        learning_rate=0.001,
        weight_decay=5e-4,
        num_epochs=100,
        early_stopping_patience=10,  # Number of epochs to wait for improvement
    )

    # Create splits
    logging.info("Creating splits...")
    dataset = [data]  # Wrap in list for do_edge_split
    split_edge = do_edge_split(dataset, fast_split=True, val_ratio=0.01, test_ratio=0.01, neg_ratio=1)
    data.edge_index = split_edge['train']['edge'].t()
    logging.info("Created splits")

    # Adjust edge_index and edge_weight for testing by including validation edges
    logging.info("Including validation edges into training edges for testing...")
    val_edge_index = split_edge['valid']['edge'].t()
    data.edge_index = torch.cat([data.edge_index, val_edge_index], dim=-1)
    
    if hasattr(data, 'edge_weight'):
        val_edge_weight = torch.ones([val_edge_index.size(1), 1], dtype=data.edge_weight.dtype, device=data.edge_weight.device)
        data.edge_weight = torch.cat([data.edge_weight, val_edge_weight], dim=0)
    else:
        val_edge_weight = torch.ones([val_edge_index.size(1), 1], dtype=torch.float)
        data.edge_weight = val_edge_weight
    logging.info("Included validation edges into training edges.")

    # Determine preprocessing function
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
    logging.info("Preprocessing function configured.")

    # Create datasets
    logging.info("Creating datasets...")
    train_dataset = SEALIterableDataset(
        root='./temp_seal_data/train',
        data=data,
        split_edge=split_edge,
        num_hops=args.num_hops,
        split='train',
        directed=True,
        use_coalesce=False,
        node_label='drnl',
        ratio_per_hop=1.0,
        max_nodes_per_hop=args.max_nodes_per_hop,
        preprocess_fn=preprocess_fn,
        sample_type=args.sample_type,
        shuffle=True,
        slice_type=0
    )
    val_dataset = SEALIterableDataset(
        root='./temp_seal_data/val',
        data=data,
        split_edge=split_edge,
        num_hops=args.num_hops,
        split='valid',
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
    test_dataset = SEALIterableDataset(
        root='./temp_seal_data/test',
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
    logging.info("Created datasets.")

    # Create DataLoaders
    logging.info("Creating dataloaders...")
    if args.num_workers > 0:
        train_loader = PygDataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            prefetch_factor=args.prefetch_factor,
            persistent_workers=args.persistent_workers
        )
        val_loader = PygDataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            prefetch_factor=args.prefetch_factor,
            persistent_workers=args.persistent_workers
        )
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
        train_loader = PygDataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=args.pin_memory,
            prefetch_factor=None
        )
        val_loader = PygDataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=args.pin_memory,
            prefetch_factor=None
        )
        test_loader = PygDataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=args.pin_memory,
            prefetch_factor=None
        )
    logging.info("Created dataloaders.")

    # Initialize model
    logging.info("Creating Model")
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
        logging.error("Error: Model not recognized.")
        exit()
    logging.info("Created Model")

    # Count and log model parameters
    num_params = count_parameters(model)
    logging.info(f"Total trainable parameters in the model: {num_params}")

    # Define loss and optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = GradScaler() if device.type == 'cuda' else None

    # Initialize Learning Rate Scheduler (Per Batch)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2)
    logging.info("Initialized CosineAnnealingWarmRestarts scheduler with T_0=2")

    # Training loop with validation and early stopping
    best_val_auc = 0
    best_epoch = -1
    patience_counter = 0

    for epoch in range(1, args.num_epochs + 1):
        logging.info(f"\nEpoch {epoch}/{args.num_epochs}")

        # Training
        train_loss, train_auc, train_ap = train(model, train_loader, optimizer, scheduler, criterion, device, scaler)
        logging.info(f"Train Loss: {train_loss:.4f}, AUC: {train_auc:.4f}, AP: {train_ap:.4f}")

        # Validation
        val_loss, val_auc, val_ap = validate(model, val_loader, criterion, device)
        logging.info(f"Validation Loss: {val_loss:.4f}, AUC: {val_auc:.4f}, AP: {val_ap:.4f}")

        # Check for improvement
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), 'best_model.pth')
            logging.info(f"New best model saved at epoch {epoch} with AUC: {val_auc:.4f}")
        else:
            patience_counter += 1
            logging.info(f"No improvement in AUC for {patience_counter} epoch(s)")

        # Early stopping
        if patience_counter >= args.early_stopping_patience:
            logging.info(f"Early stopping triggered. Best epoch: {best_epoch} with AUC: {best_val_auc:.4f}")
            break

    # Load the best model
    logging.info("Loading the best model for testing...")
    model.load_state_dict(torch.load('best_model.pth'))

    # Evaluate on test set
    logging.info("Evaluating on test set...")
    test_loss, test_auc, test_ap = test(model, test_loader, criterion, device)

    logging.info("Training and evaluation complete.")

if __name__ == "__main__":
    main()
