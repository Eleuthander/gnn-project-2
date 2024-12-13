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
from torch.cuda.amp import autocast, GradScaler  # for mixed precision training
import gc
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
import random
import math
import torch_sparse as ssp
from threading import Thread
import GPUtil
import psutil
import logging
from datetime import datetime
from torch.nn import Linear

CHECKPOINT_DIR = './checkpoints'
METRICS_DIR = './metrics'

# Suppress annoying warnings; the deprecation warnings are ignorable since you are using an old version of torch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning) # This is for a specific warning about optimizer and scheduler that was a false positive

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

def save_metrics(metrics_dict, filename):
    """
    Save metrics to a CSV file.
    """
    df = pd.DataFrame([metrics_dict])
    
    # Create directory if it doesn't exist
    os.makedirs(METRICS_DIR, exist_ok=True)
    
    # If file exists, append without header. If not, create with header
    if os.path.exists(filename):
        df.to_csv(filename, mode='a', header=False, index=False)
    else:
        df.to_csv(filename, mode='w', header=True, index=False)

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

    try:
        for i, batch in enumerate(pbar):
            batch = batch.to(device)
            optimizer.zero_grad()
            
            if scaler is not None:
                with autocast():
                    out = model(batch).squeeze().view(-1)
                    loss = criterion(out, batch.y.float().view(-1))
                scaler.scale(loss).backward()
                
                #model.print_gradient_norms()
                # Unscale before measuring gradients
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(batch).view(-1)
                loss = criterion(out, batch.y.float().view(-1))
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            scheduler.step()

            # Calculate loss and preds
            total_loss += loss.item()
            preds = torch.sigmoid(out).detach().cpu().view(-1)
            labels = batch.y.detach().cpu().view(-1)
            all_preds.append(preds)
            all_labels.append(labels)

            # Update progress bar
            current_loss = total_loss / (len(all_preds))
            pbar.set_postfix({'loss': f"{current_loss:.4f}"})

            # Print gradients, loss, and usage periodically
            if i > 0 and i % 100 == 0:  # max(1, len(pbar) // 10)
            
                # Calculate metrics on accumulated predictions
                batch_preds = torch.cat(all_preds[-100:], dim=0).numpy()  # Last 100 batches
                batch_labels = torch.cat(all_labels[-100:], dim=0).numpy()
                batch_auc, batch_ap = evaluate_metrics(batch_labels, batch_preds)
                
                gpu = GPUtil.getGPUs()[0]
                cpu_percent = psutil.cpu_percent()
                memory = psutil.virtual_memory()
                tqdm.write(f"Batch {i} | Gradient norm: {grad_norm:.4f} | Loss: {current_loss:.4f} | AUC: {batch_auc:.4f} | AP: {batch_ap:.4f} | GPU: {gpu.load*100:.1f}% | Mem: {gpu.memoryUtil*100:.1f}% | CPU: {cpu_percent:.1f}% | RAM: {memory.percent:.1f}%")

            # Memory cleanup
            del batch
            del out
            torch.cuda.empty_cache()

    except Exception as e:
        logging.info(f"Training interrupted: {e}")
        torch.save(model.state_dict(), 'interrupted_model.pth')

        checkpoint_path = os.path.join(CHECKPOINT_DIR, f'interruption_time_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, checkpoint_path)
    finally:
        pbar.close()
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

    try:
        with torch.no_grad():
            for i, batch in enumerate(pbar):
                batch = batch.to(device)
                with autocast():
                    out = model(batch).view(-1) # logits are returned as [batch_size, 1] so need squeeze
                    loss = criterion(out, batch.y.float().view(-1))
                
                # Calculate loss and preds
                total_loss += loss.item()
                preds = torch.sigmoid(out).detach().cpu().view(-1)
                labels = batch.y.detach().cpu().view(-1)
                all_preds.append(preds)
                all_labels.append(labels)

                # Update progress bar
                current_loss = total_loss / (len(all_preds))
                pbar.set_postfix({'loss': f"{current_loss:.4f}"})

                # Prin loss and usage periodically
                if i > 0 and i % 100 == 0:  # max(1, len(pbar) // 10)
                    gpu = GPUtil.getGPUs()[0]
                    cpu_percent = psutil.cpu_percent()
                    memory = psutil.virtual_memory()
                    tqdm.write(f"Batch {i} | Loss: {current_loss:.4f} | GPU: {gpu.load*100:.1f}% | Mem: {gpu.memoryUtil*100:.1f}% | CPU: {cpu_percent:.1f}% | RAM: {memory.percent:.1f}%")

                # Clear GPU memory
                del batch
                del out
                torch.cuda.empty_cache()

    except Exception as e:
        logging.info(f"Validation interrupted: {e}")
    finally:
        pbar.close()
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

    try:
        with torch.no_grad():
            for i, batch in enumerate(pbar):
                batch = batch.to(device)
                with autocast():
                    out = model(batch).view(-1)
                    loss = criterion(out, batch.y.float().view(-1))

                # Calculate loss and preds
                total_loss += loss.item()
                preds = torch.sigmoid(out).detach().cpu().view(-1)
                labels = batch.y.detach().cpu().view(-1)
                all_preds.append(preds)
                all_labels.append(labels)

                # Update progress bar
                current_loss = total_loss / (len(all_preds))
                pbar.set_postfix({'loss': f"{current_loss:.4f}"})

                # Print loss and usage periodically
                if i > 0 and i % 100 == 0:  # max(1, len(pbar) // 10)
                    gpu = GPUtil.getGPUs()[0]
                    cpu_percent = psutil.cpu_percent()
                    memory = psutil.virtual_memory()
                    tqdm.write(f"Batch {i} | Loss: {current_loss:.4f} | GPU: {gpu.load*100:.1f}% | Mem: {gpu.memoryUtil*100:.1f}% | CPU: {cpu_percent:.1f}% | RAM: {memory.percent:.1f}%")

                # Clear GPU memory
                del batch
                del out
                torch.cuda.empty_cache()

    except Exception as e:
        logging.info(f"Testing interrupted: {e}")
    finally:
        pbar.close()
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
# Weight Initialization Function
# ---------------------------

def init_weights(m, model, full_graph):
    if model=='GCNGraphormer':
        if isinstance(m, Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    elif model=='SANGraphormer' and full_graph:
        if isinstance(m, Linear):
            if hasattr(m, 'final_layer') and m.final_layer:
                # Xavier/Glorot initialization with larger gain for final layer
                torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    torch.nn.init.uniform_(m.bias, -bound, bound)
            else:
                # Standard Xavier/Glorot for other layers
                torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
    elif model=='SANGraphormer' and not full_graph:
        if isinstance(m, Linear):
            if hasattr(m, 'final_layer') and m.final_layer:
                # Xavier/Glorot initialization with larger gain for final layer
                torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    torch.nn.init.uniform_(m.bias, -bound, bound)
            else:
                # Standard Xavier/Glorot for other layers
                torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
    else:
        pass

# ---------------------------
# Main Function
# ---------------------------

def main():
    # Setup logging and checkpoint savespace
    setup_logging()
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Create default args
    args = SimpleNamespace(
        seed = 234,

        # Model args
        model = 'SANGraphormer',  # either 'SANGraphormer' or 'GCNGraphormer'
        hidden_channels=64, # transformer block dims (32-64 reasonable) \ embedding dim in GCN (32-64 reasonable)
        num_layers=3, # 4 for SAN, 3 for GCN
        dropout=0.2, # 0.7 for SAN, 0.3 for GCN
        full_graph=False, # whether to add fake edges to SAN
        layer_norm=True, # whether to implement layer norms in the SAN; batch norm always implemented
        gamma=1e-7, # between 0 and 1:  0 is fully sparse, 1 fully connected (1e- through 1e- reasonable for this impl)
        GT_n_heads = 6, # Num heads for SAN module (3-6 reasonable)
        num_heads=4, # Num heads for graphormer module (4 used in SIEG)

        # Subgraph args
        num_hops = 1,
        max_nodes_per_hop = 500,
        max_z=1000,  # Max value for structural encoding

        # Batching args
        batch_size=256,
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
        num_step = 1,

        # Training args
        initial_lr=1e-4,
        min_lr=1e-6,
        warmup_proportion=1.0, # what proportion of an epoch you want used for warmup
        T_0=1.0, # what proportion of an epoch you want for cosine period in scheduler
        weight_decay=1e-4,
        num_epochs=5,
        early_stopping_patience=6,  # Number of epochs to wait for improvement
    )

    # Take command line args
    parser = argparse.ArgumentParser(description="Override default arguments.")
    parser.add_argument("--model", type=str, default=args.model, help="Name of the model")
    parser.add_argument("--full_graph", action="store_true", default=args.full_graph, help="Adds false edges to SAN")
    parser.add_argument("--hidden_channels", type=int, default=args.hidden_channels, help="Hidden dimmension")
    parser.add_argument("--num_layers", type=int, default=args.num_layers, help="Depth of the model")
    parser.add_argument("--dropout", type=float, default=args.dropout, help="Dropout for layers")
    parser.add_argument("--seed", type=int, default=args.dropout, help="Randomization replicator")
    parser.add_argument("--use_time_feature", action="store_false", default=args.use_time_feature, help="Use time_feature as PE in SAN")
    parser.add_argument("--num_epochs", type=int, default=args.num_epochs, help="Epochs")

    parsed_args = parser.parse_args()
    for key, value in vars(parsed_args).items():
        setattr(args, key, value)
        logging.info(f"Argument {key} is {value}")

    # Set random seed
    set_seed(args.seed)

    # Load data
    data = torch.load('appellate_graph_final.pt')
    logging.info(f"Loaded data: {data}")

    # Create splits
    logging.info("Creating splits...")
    dataset = [data]  # Wrap in list for do_edge_split
    split_edge = do_edge_split(dataset, val_ratio=0.1, test_ratio=0.2, neg_ratio=1)
    data.edge_index = split_edge['train']['edge'].t()
    logging.info("Created splits")

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
    train_dataset = SEALDynamicDataset(
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
        internal_shuffle=True,
        slice_type=0
    )
    val_dataset = SEALDynamicDataset(
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
        internal_shuffle=False,
        slice_type=0
    )
    test_dataset = SEALDynamicDataset(
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
        internal_shuffle=False,
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
        logging.info("Model: GCNGraphormer")
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
            layer_norm=args.layer_norm,
            gamma=args.gamma
        ).to(device)
        logging.info("Model: SANGraphormer")
    else:
        logging.error("Error: Model not recognized.")
        exit()

    init_weights_with_args = partial(init_weights, model=args.model, full_graph=args.full_graph)
    model.apply(init_weights_with_args)
    logging.info("Created Model")

    # Count and log model parameters
    num_params = count_parameters(model)
    logging.info(f"Total trainable parameters in the model: {num_params}")

    # Define loss and optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.initial_lr, weight_decay=args.weight_decay)
    scaler = GradScaler() if device.type == 'cuda' else None

    # Initialize Learning Rate Scheduler (Per Batch)
    iterations_per_epoch = len(train_loader)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, 
        start_factor=0.3,
        end_factor=1.0, 
        total_iters=int(args.warmup_proportion * iterations_per_epoch),
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=int(args.T_0 * iterations_per_epoch),
        eta_min=args.min_lr
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[int(args.warmup_proportion * iterations_per_epoch)]
    )
    logging.info("Initialized sequential scheduler with warmup and cosine annealing")

    # Create metrics filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    metrics_filename = os.path.join(METRICS_DIR, f'training_metrics_{timestamp}.csv')

    # Training loop with validation and early stopping
    best_val_metric = 0
    best_val_auc = 0
    best_val_ap = 0
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

        # Evaluate on test set
        test_loss, test_auc, test_ap = test(model, test_loader, criterion, device)
        logging.info(f"Test Loss: {test_loss:.4f}, AUC: {test_auc:.4f}, AP: {test_ap:.4f}")

        # Save metrics
        metrics = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'epoch': epoch,
            'train_loss': train_loss,
            'train_auc': train_auc,
            'train_ap': train_ap,
            'val_loss': val_loss,
            'val_auc': val_auc,
            'val_ap': val_ap,
            'test_loss': test_loss,
            'test_auc': test_auc,
            'test_ap': test_ap,
            'learning_rate': optimizer.param_groups[0]['lr']
        }
        save_metrics(metrics, metrics_filename)

        # Check for improvement
        current_metric = val_auc * 0.5 + val_ap * 0.5
        if current_metric > best_val_metric:
            best_val_metric = current_metric
            best_epoch = epoch
            patience_counter = 0

            # Save best model if improvement
            torch.save(model.state_dict(), 'best_model.pth')
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_epoch_{epoch}_time_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_metric': best_val_metric,
                'best_val_auc': best_val_auc,
                'best_val_ap': best_val_ap,
                'patience_counter': patience_counter,
            }, checkpoint_path)
            logging.info(f"New best model saved at epoch {epoch} with combined metric: {current_metric:.4f} (AUC: {val_auc:.4f}, AP: {val_ap:.4f})")
        else:
            patience_counter += 1
            logging.info(f"No improvement for {patience_counter} epoch(s)")

        # Early stopping
        if patience_counter >= args.early_stopping_patience:
            logging.info(f"Early stopping triggered. Best epoch: {best_epoch} with combined metric: {best_val_metric:.4f} (AUC: {best_val_auc:.4f}, AP: {best_val_ap:.4f})")
            break

    # Final evaluation with best model
    model.load_state_dict(torch.load('best_model.pth'))
    final_test_loss, final_test_auc, final_test_ap = test(model, test_loader, criterion, device)
    
    # Save final metrics
    final_metrics = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'epoch': 'final',
        'train_loss': None,
        'train_auc': None,
        'train_ap': None,
        'val_loss': None,
        'val_auc': None,
        'val_ap': None,
        'test_loss': final_test_loss,
        'test_auc': final_test_auc,
        'test_ap': final_test_ap,
        'learning_rate': optimizer.param_groups[0]['lr']
    }
    save_metrics(final_metrics, metrics_filename)
    
    logging.info("Training and evaluation complete. Metrics saved to: " + metrics_filename)

if __name__ == "__main__":
    main()
