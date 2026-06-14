import gzip
import pickle
import numpy as np
import scipy.sparse as sp
import random as rd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score

rd.seed(42)

"""
    Data preprocessing for IEEE power system datasets
    Build multi-relational graphs (physical, geographic, logical)
    Generate camouflaged anomalies for evaluation
"""


def build_multi_relational_graph(adj_physical, adj_geographic, adj_logical):
    """
    Build multi-relational graph adjacency matrices for CR-PGNN
    """
    n_nodes = adj_physical.shape[0]

    # Convert to sparse lil matrix format
    phys_adj = sp.lil_matrix(adj_physical)
    geog_adj = sp.lil_matrix(adj_geographic)
    logi_adj = sp.lil_matrix(adj_logical)

    return phys_adj, geog_adj, logi_adj


def add_feature_camouflage(features, labels, anomaly_ratio=0.15, alpha=0.7):
    """
    Add feature camouflage to anomalous nodes
    Eq. (14) in the paper: x_cam = (1-α)*x + α*mean(x_neighbor) + ε
    """
    n_nodes = features.shape[0]
    anomaly_indices = np.where(labels == 1)[0]
    n_anomalies = int(n_nodes * anomaly_ratio)

    if len(anomaly_indices) > n_anomalies:
        anomaly_indices = np.random.choice(anomaly_indices, n_anomalies, replace=False)

    features_cam = features.copy()

    for idx in anomaly_indices:
        # Find neighbor indices (simplified: use random neighbors)
        neighbor_indices = np.random.choice(n_nodes, 5, replace=False)
        neighbor_mean = np.mean(features[neighbor_indices], axis=0)
        noise = np.random.normal(0, 0.01, features.shape[1])
        features_cam[idx] = (1 - alpha) * features[idx] + alpha * neighbor_mean + noise

    return features_cam, anomaly_indices


def add_relation_camouflage(adj_physical, anomaly_indices, n_healthy_connections=3):
    """
    Add relation camouflage by connecting anomalous nodes to healthy neighbors
    """
    adj_cam = adj_physical.copy()
    n_nodes = adj_cam.shape[0]
    healthy_indices = list(set(range(n_nodes)) - set(anomaly_indices))

    for idx in anomaly_indices:
        # Select random healthy neighbors
        healthy_neighbors = np.random.choice(healthy_indices, n_healthy_connections, replace=False)
        for neighbor in healthy_neighbors:
            adj_cam[idx, neighbor] = 1
            adj_cam[neighbor, idx] = 1

    return adj_cam


def split_train_test(labels, train_ratio=0.7):
    """
    Split dataset into train and test sets
    """
    indices = np.arange(len(labels))
    train_idx, test_idx = train_test_split(
        indices, test_size=1 - train_ratio, stratify=labels, random_state=42
    )
    return train_idx, test_idx


if __name__ == "__main__":
    # ========== Load IEEE 14-bus system ==========
    print("Loading IEEE 14-bus system...")
    ieee14 = loadmat('data/IEEE14.mat')

    # Extract multi-relational graphs
    phys_adj_14 = ieee14['net_physical']  # Physical transmission lines
    geog_adj_14 = ieee14['net_geographic']  # Geographic proximity
    logi_adj_14 = ieee14['net_logical']  # Logical control dependencies
    features_14 = ieee14['features']  # PMU measurements
    labels_14 = ieee14['labels'].flatten()  # Ground truth labels

    # Convert to sparse format and save
    phys_adj_14_sp, geog_adj_14_sp, logi_adj_14_sp = build_multi_relational_graph(
        phys_adj_14, geog_adj_14, logi_adj_14
    )

    sp.save_npz('data/ieee14_physical_adj.npz', phys_adj_14_sp.tocsr())
    sp.save_npz('data/ieee14_geographic_adj.npz', geog_adj_14_sp.tocsr())
    sp.save_npz('data/ieee14_logical_adj.npz', logi_adj_14_sp.tocsr())
    np.save('data/ieee14_features.npy', features_14)
    np.save('data/ieee14_labels.npy', labels_14)

    # ========== Load IEEE 57-bus system ==========
    print("Loading IEEE 57-bus system...")
    ieee57 = loadmat('data/IEEE57.mat')

    phys_adj_57 = ieee57['net_physical']
    geog_adj_57 = ieee57['net_geographic']
    logi_adj_57 = ieee57['net_logical']
    features_57 = ieee57['features']
    labels_57 = ieee57['labels'].flatten()

    phys_adj_57_sp, geog_adj_57_sp, logi_adj_57_sp = build_multi_relational_graph(
        phys_adj_57, geog_adj_57, logi_adj_57
    )

    sp.save_npz('data/ieee57_physical_adj.npz', phys_adj_57_sp.tocsr())
    sp.save_npz('data/ieee57_geographic_adj.npz', geog_adj_57_sp.tocsr())
    sp.save_npz('data/ieee57_logical_adj.npz', logi_adj_57_sp.tocsr())
    np.save('data/ieee57_features.npy', features_57)
    np.save('data/ieee57_labels.npy', labels_57)

    # ========== Load IEEE 118-bus system ==========
    print("Loading IEEE 118-bus system...")
    ieee118 = loadmat('data/IEEE118.mat')

    phys_adj_118 = ieee118['net_physical']
    geog_adj_118 = ieee118['net_geographic']
    logi_adj_118 = ieee118['net_logical']
    features_118 = ieee118['features']
    labels_118 = ieee118['labels'].flatten()

    phys_adj_118_sp, geog_adj_118_sp, logi_adj_118_sp = build_multi_relational_graph(
        phys_adj_118, geog_adj_118, logi_adj_118
    )

    sp.save_npz('data/ieee118_physical_adj.npz', phys_adj_118_sp.tocsr())
    sp.save_npz('data/ieee118_geographic_adj.npz', geog_adj_118_sp.tocsr())
    sp.save_npz('data/ieee118_logical_adj.npz', logi_adj_118_sp.tocsr())
    np.save('data/ieee118_features.npy', features_118)
    np.save('data/ieee118_labels.npy', labels_118)

    # ========== Generate camouflaged anomalies for evaluation ==========
    print("Generating camouflaged anomalies...")

    # For IEEE 118-bus system
    features_cam, anomaly_indices = add_feature_camouflage(
        features_118, labels_118, anomaly_ratio=0.15, alpha=0.7
    )
    adj_cam = add_relation_camouflage(phys_adj_118, anomaly_indices, n_healthy_connections=3)

    np.save('data/ieee118_features_camouflaged.npy', features_cam)
    sp.save_npz('data/ieee118_adj_camouflaged.npz', adj_cam.tocsr())
    np.save('data/ieee118_anomaly_indices.npy', anomaly_indices)

    # ========== Train/test split ==========
    print("Creating train/test splits...")

    train_idx, test_idx = split_train_test(labels_118, train_ratio=0.7)
    np.save('data/ieee118_train_idx.npy', train_idx)
    np.save('data/ieee118_test_idx.npy', test_idx)

    print("Data preprocessing completed!")
    print(f"  - IEEE 14-bus: {features_14.shape[0]} nodes, {features_14.shape[1]} features")
    print(f"  - IEEE 57-bus: {features_57.shape[0]} nodes, {features_57.shape[1]} features")
    print(f"  - IEEE 118-bus: {features_118.shape[0]} nodes, {features_118.shape[1]} features")
    print(f"  - Anomaly ratio: {np.mean(labels_118):.2%}")
    print(f"  - Camouflage added: {len(anomaly_indices)} nodes")