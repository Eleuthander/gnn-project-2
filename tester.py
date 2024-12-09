import torch
from torch_geometric.loader import DataLoader as PygDataLoader
from dataset import SEALDynamicDataset
from utils import *
from models import GCNGraphormer
import argparse
from types import SimpleNamespace
from preprocess import preprocess

# Load data
data = torch.load('fed_cites_graph.pt')
print(data)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create default args for GCNGraphormer
args = SimpleNamespace(
    hidden_channels=32,
    num_layers=3,
    max_z=1000,  # Max value for structural encoding
    use_feature=True,
    use_feature_GT=True,
    use_edge_weight=False,
    num_heads=4,
    grpe_cross=False,
    use_len_spd=True,
    use_num_spd=True,
    use_cnb_jac=True,
    use_cnb_aa=True,
    use_cnb_ra=True,
    use_degree=True,
    mul_bias=True,
    gravity_type=0
)

# Create splits
dataset = [data]  # Wrap in list for do_edge_split
split_edge = do_edge_split(dataset)
data.edge_index = split_edge['train']['edge'].t()

# Create test dataset
test_dataset = SEALDynamicDataset(
    path='./temp_seal_data',
    data=data,
    split_edge=split_edge,
    num_hops=2,
    split='test',
    directed=True,
    use_coalesce=False,
    node_label='drnl',
    ratio_per_hop=1.0,
    max_nodes_per_hop=100,
    preprocess_fn=preprocess
)

# Create dataloader
test_loader = PygDataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=1
)

# Initialize model
model = GCNGraphormer(
    args=args,
    hidden_channels=args.hidden_channels,
    num_layers=args.num_layers,
    max_z=args.max_z,
    train_dataset=test_dataset,  # Used for getting dataset properties
    use_feature=args.use_feature,
    use_feature_GT=args.use_feature_GT,
    node_embedding=None
).to(device)

# Evaluate
model.eval()
all_preds = []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        pred = model(batch)
        all_preds.append(pred.cpu())
    all_preds = torch.cat(all_preds, dim=0)

print(f"Generated predictions shape: {all_preds.shape}")