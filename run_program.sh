#!/bin/bash

pip install -r requirements.txt

# First do CN calculation

ptrhon3 

# First do 3 GCNGraphormer runs; results will be recorded in /logs, /checkpoints, and /metrics

python3 train_test.py --model GCNGraphormer --hidden_channels 48 --num_layers 3 --dropout 0.3 --seed 234

python3 train_test.py --model GCNGraphormer --hidden_channels 48 --num_layers 3 --dropout 0.3 --seed 345

python3 train_test.py --model GCNGraphormer --hidden_channels 48 --num_layers 3 --dropout 0.3 --seed 456

# Then do 3 SANGraphormer runs *with* time_features; results will be recorded in /logs, /checkpoints, and /metrics

python3 train_test.py --model SANGraphormer  --full_graph --hidden_channels 36 --num_layers 4 --dropout 0.7 --seed 234

python3 train_test.py --model SANGraphormer  --full_graph --hidden_channels 36 --num_layers 4 --dropout 0.7 --seed 345

python3 train_test.py --model SANGraphormer  --full_graph --hidden_channels 36 --num_layers 4 --dropout 0.7 --seed 456

# Then do 3 SANGraphormer runs *without* time_features; results will be recorded in /logs, /checkpoints, and /metrics

python3 train_test.py --model SANGraphormer  --full_graph --hidden_channels 36 --num_layers 4 --dropout 0.7 --seed 234 --use_time_feature

python3 train_test.py --model SANGraphormer  --full_graph --hidden_channels 36 --num_layers 4 --dropout 0.7 --seed 345 --use_time_feature

python3 train_test.py --model SANGraphormer  --full_graph --hidden_channels 36 --num_layers 4 --dropout 0.7 --seed 456 --use_time_feature