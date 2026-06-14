import time
import os
import random
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score

from utils import *
from model import *
from layers import *
from graphsage import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

"""
    Training CR-PGNN
    Paper: Camouflage-Resistant Graph Neural Networks for Power Grid Anomaly Detection

    Core modules:
    1. Multi-head Physics Consistency (MPC) Module
    2. RL-based Adaptive Neighbor Selector
    3. Relation-aware Gated Aggregator
"""

parser = argparse.ArgumentParser()

# dataset and model dependent args
parser.add_argument('--data', type=str, default='ieee118', help='The dataset name. [ieee14, ieee57, ieee118]')
parser.add_argument('--model', type=str, default='CR_PGNN', help='The model name. [CR_PGNN, SAGE, GCN, GAT]')
parser.add_argument('--inter', type=str, default='Gated',
                    help='The inter-relation aggregator type. [Gated, Att, Weight, Mean]')
parser.add_argument('--batch-size', type=int, default=1024, help='Batch size for training.')

# hyper-parameters
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--lambda_phys', type=float, default=0.5, help='Physics consistency loss weight.')
parser.add_argument('--lambda_rl', type=float, default=0.1, help='RL reward loss weight.')
parser.add_argument('--lambda_sparse', type=float, default=0.01, help='Sparsity penalty weight.')
parser.add_argument('--weight-decay', type=float, default=1e-3, help='Weight decay (L2 loss weight).')
parser.add_argument('--emb-size', type=int, default=64, help='Node embedding size at the last layer.')
parser.add_argument('--n-heads', type=int, default=4, help='Number of attention heads in MPC module.')
parser.add_argument('--n-layers', type=int, default=2, help='Number of GNN layers (optimal: 2 or 3).')
parser.add_argument('--num-epochs', type=int, default=50, help='Number of epochs.')
parser.add_argument('--test-epochs', type=int, default=5, help='Epoch interval to run test set.')
parser.add_argument('--step-size', type=float, default=0.05, help='RL action step size')
parser.add_argument('--alpha-camouflage', type=float, default=0.7, help='Camouflage strength for feature manipulation')

# other args
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
print(f'Run on {args.data} with {args.model}')

# load graph, feature, and label
# Returns: [physical_adj, geographic_adj, logical_adj, homo_adj], features, labels, physics_residuals
adj_lists, feat_data, labels, physics_residuals = load_power_grid_data(args.data)

# train_test split (70/30 as in Section 4.1)
np.random.seed(args.seed)
random.seed(args.seed)
index = list(range(len(labels)))
idx_train, idx_test, y_train, y_test = train_test_split(
    index, labels, stratify=labels, test_size=0.30, random_state=args.seed, shuffle=True
)

print(f'Training samples: {len(idx_train)}, Test samples: {len(idx_test)}')
print(f'Anomaly ratio in training: {np.mean(y_train):.4f}')
print(f'Anomaly ratio in test: {np.mean(y_test):.4f}')

# split pos neg sets for under-sampling to handle class imbalance
train_pos, train_neg = pos_neg_split(idx_train, y_train)

# initialize model input
features = nn.Embedding(feat_data.shape[0], feat_data.shape[1])
feat_data = normalize(feat_data)
features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
if args.cuda:
    features.cuda()

# set input graph (multi-relational)
if args.model == 'SAGE' or args.model == 'GCN':
    adj_lists = adj_lists[3]  # use homogeneous graph for baselines
else:
    adj_lists = adj_lists[:3]  # use multi-relational graphs for CR-PGNN

print(f'Model: {args.model}, Inter-AGG: {args.inter}, emb_size: {args.emb_size}, n_layers: {args.n_layers}')

# build CR-PGNN models
if args.model == 'CR_PGNN':
    # Build intra-relation aggregators for each relation
    intra_aggs = nn.ModuleList([
        IntraRelationAggregator(features, feat_data.shape[1], cuda=args.cuda)
        for _ in range(3)
    ])

    # Build inter-relation aggregator
    inter_agg = InterAgg(
        features, feat_data.shape[1], args.emb_size, adj_lists, intra_aggs,
        inter=args.inter, n_heads=args.n_heads, step_size=args.step_size, cuda=args.cuda
    )

    # Build RL neighbor selector
    rl_selector = RLNeighborSelector(step_size=args.step_size, cuda=args.cuda)

    # Build multi-layer CR-PGNN model
    if args.n_layers == 1:
        gnn_model = OneLayerCR_PGNN(2, inter_agg, args.lambda_phys, args.lambda_rl)
    else:
        # Build multiple layers for deeper architecture
        inter_aggs = [inter_agg]
        for _ in range(args.n_layers - 1):
            # For deeper layers, use the same structure but with previous embeddings
            inter_aggs.append(InterAgg(
                features, args.emb_size, args.emb_size, adj_lists, intra_aggs,
                inter=args.inter, n_heads=args.n_heads, step_size=args.step_size, cuda=args.cuda
            ))
        gnn_model = MultiLayerCR_PGNN(2, inter_aggs, args.lambda_phys, args.lambda_rl, args.n_layers)

elif args.model == 'SAGE':
    agg1 = MeanAggregator(features, cuda=args.cuda)
    enc1 = Encoder(features, feat_data.shape[1], args.emb_size, adj_lists, agg1, gcn=True, cuda=args.cuda)
    enc1.num_samples = 5
    gnn_model = GraphSage(2, enc1)

elif args.model == 'GCN':
    # Simple GCN baseline
    from torch_geometric.nn import GCNConv

    gnn_model = SimpleGCN(feat_data.shape[1], args.emb_size, 2)

elif args.model == 'GAT':
    # GAT baseline
    from torch_geometric.nn import GATConv

    gnn_model = SimpleGAT(feat_data.shape[1], args.emb_size, 2)

if args.cuda:
    gnn_model.cuda()

# Optimizer with weight decay
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, gnn_model.parameters()),
    lr=args.lr, weight_decay=args.weight_decay
)

times = []
performance_log = []

# Training loop
print('\nStarting training...')
for epoch in range(args.num_epochs):
    # Randomly under-sample negative nodes for each epoch to handle class imbalance
    sampled_idx_train = undersample(train_pos, train_neg,
                                    scale=args.under_sample if hasattr(args, 'under_sample') else 1)
    random.shuffle(sampled_idx_train)

    # Send number of batches to model for RL module to know training progress
    num_batches = int(len(sampled_idx_train) / args.batch_size) + 1
    if args.model == 'CR_PGNN':
        inter_agg.batch_num = num_batches

    loss = 0.0
    epoch_time = 0

    # Mini-batch training
    for batch in range(num_batches):
        start_time = time.time()
        i_start = batch * args.batch_size
        i_end = min((batch + 1) * args.batch_size, len(sampled_idx_train))
        batch_nodes = sampled_idx_train[i_start:i_end]
        batch_label = labels[np.array(batch_nodes)]

        optimizer.zero_grad()

        if args.cuda:
            loss = gnn_model.loss(
                batch_nodes,
                Variable(torch.cuda.LongTensor(batch_label)),
                train_flag=True
            )
        else:
            loss = gnn_model.loss(
                batch_nodes,
                Variable(torch.LongTensor(batch_label)),
                train_flag=True
            )

        loss.backward()
        optimizer.step()

        end_time = time.time()
        epoch_time += end_time - start_time
        loss += loss.item()

    print(f'Epoch: {epoch}, loss: {loss.item() / num_batches:.4f}, time: {epoch_time:.2f}s')

    # Test the model every test_epoch epochs
    if epoch % args.test_epochs == 0:
        if args.model == 'SAGE' or args.model == 'GCN' or args.model == 'GAT':
            test_results = test_baseline(idx_test, y_test, gnn_model, args.batch_size, args.cuda)
            gnn_auc, gnn_f1, gnn_precision, gnn_recall = test_results
            performance_log.append([gnn_auc, gnn_f1, gnn_precision, gnn_recall])
            print(f'  Test - AUC: {gnn_auc:.4f}, F1: {gnn_f1:.4f}, '
                  f'Precision: {gnn_precision:.4f}, Recall: {gnn_recall:.4f}')
        else:
            test_results = test_cr_pgnn(idx_test, y_test, gnn_model, args.batch_size, args.cuda, physics_residuals)
            gnn_auc, gnn_f1, gnn_precision, gnn_recall, phys_auc = test_results
            performance_log.append([gnn_auc, gnn_f1, gnn_precision, gnn_recall, phys_auc])
            print(f'  Test - GNN AUC: {gnn_auc:.4f}, F1: {gnn_f1:.4f}, '
                  f'Precision: {gnn_precision:.4f}, Recall: {gnn_recall:.4f}, '
                  f'Physics AUC: {phys_auc:.4f}')

# Final evaluation
print('\n' + '=' * 60)
print('Final Results:')
print('=' * 60)

best_epoch = np.argmax([p[1] for p in performance_log])  # best F1
best_results = performance_log[best_epoch]

if args.model == 'CR_PGNN':
    print(f'Best Epoch: {best_epoch * args.test_epochs}')
    print(f'Best GNN AUC: {best_results[0]:.4f}')
    print(f'Best F1-score: {best_results[1]:.4f}')
    print(f'Best Precision: {best_results[2]:.4f}')
    print(f'Best Recall: {best_results[3]:.4f}')
    print(f'Best Physics AUC: {best_results[4]:.4f}')
else:
    print(f'Best Epoch: {best_epoch * args.test_epochs}')
    print(f'Best AUC: {best_results[0]:.4f}')
    print(f'Best F1-score: {best_results[1]:.4f}')
    print(f'Best Precision: {best_results[2]:.4f}')
    print(f'Best Recall: {best_results[3]:.4f}')