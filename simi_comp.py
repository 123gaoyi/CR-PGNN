from scipy.io import loadmat
import numpy as np
import scipy.sparse as sp
import torch


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1)) + 0.01
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def compute_physics_residual(V_i, V_j, theta_i, theta_j, X_ij, P_ij_reported):
    """
    Compute AC power flow mismatch Gamma_{i,j}
    Eq. (4) in the paper
    """
    V = V_i * V_j
    sin_theta = np.sin(theta_i - theta_j)
    expected_flow = (V * sin_theta) / (X_ij + 1e-8)
    residual = np.abs(expected_flow - P_ij_reported)
    return residual


def compute_physics_similarity(features, residuals, gamma=0.1):
    """
    Compute physics-aware similarity scores
    Eq. (5) in the paper
    """
    n_nodes = features.shape[0]
    similarity = np.zeros((n_nodes, n_nodes))

    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                dist = np.sum((features[i] - features[j]) ** 2)
                denominator = gamma * residuals[i, j] + 1e-8
                similarity[i, j] = np.exp(-dist / denominator)

    return similarity


def compute_node_physics_loss(residuals, adj_list, nodes):
    """
    Compute node-level physics consistency loss
    Eq. (6): L_phys(u_i) = (1/|N_i|) * sum(Gamma_{i,j})
    """
    phys_losses = []
    for node in nodes:
        neighbors = adj_list[int(node)]
        if len(neighbors) > 0:
            node_residuals = [residuals[node, n] for n in neighbors]
            phys_loss = np.mean(node_residuals)
        else:
            phys_loss = 0.0
        phys_losses.append(phys_loss)
    return np.array(phys_losses)


# ========== Load IEEE Power System Data ==========
data_name = 'IEEE118.mat'  # 'IEEE14.mat', 'IEEE57.mat', or 'IEEE118.mat'
mode = 'pos'  # if set to 'pos', only compute metrics for anomalous nodes

data = loadmat(data_name)

# Multi-relational graph adjacency lists
# Physical transmission lines, geographic proximity, logical control dependencies
if data_name == 'IEEE14.mat':
    net_list = [
        data['net_physical'].nonzero(),  # physical transmission lines
        data['net_geographic'].nonzero(),  # geographic proximity (kNN)
        data['net_logical'].nonzero(),  # logical control dependencies
        data['homo'].nonzero()  # homogeneous graph
    ]
elif data_name == 'IEEE57.mat':
    net_list = [
        data['net_physical'].nonzero(),
        data['net_geographic'].nonzero(),
        data['net_logical'].nonzero(),
        data['homo'].nonzero()
    ]
else:  # IEEE118.mat
    net_list = [
        data['net_physical'].nonzero(),
        data['net_geographic'].nonzero(),
        data['net_logical'].nonzero(),
        data['homo'].nonzero()
    ]

# Node features: PMU measurements (voltage magnitude, phase angle, active/reactive power)
feature = normalize(data['features']).toarray()
label = data['label'][0]

# PMU measurement components
V_mag = feature[:, 0]  # voltage magnitude
theta = feature[:, 1]  # phase angle
P_inj = feature[:, 2]  # active power injection
Q_inj = feature[:, 3]  # reactive power injection

# Line parameters (reactance X_ij for each edge)
# This would be loaded from a separate line parameter file
# Here we use a placeholder
X_ij = 0.1  # typical reactance value for transmission lines

# Compute AC power flow residuals for all edges
print('Computing AC power flow residuals...')
residuals = np.zeros((feature.shape[0], feature.shape[0]))
for net in net_list:
    for u, v in zip(net[0].tolist(), net[1].tolist()):
        # Compute power flow mismatch using Eq. (4)
        P_reported = P_inj[u] - P_inj[v]  # simplified power flow
        residual = compute_physics_residual(
            V_mag[u], V_mag[v], theta[u], theta[v], X_ij, P_reported
        )
        residuals[u, v] = residual
        residuals[v, u] = residual

# ========== Compute Similarity Metrics for Camouflage Analysis ==========
# Extract anomalous nodes (positive nodes)
pos_nodes = set(label.nonzero()[0].tolist())
node_list = [set(net[0].tolist()) for net in net_list]
pos_node_list = [list(net_nodes.intersection(pos_nodes)) for net_nodes in node_list]
pos_idx_list = []
for net, pos_node in zip(net_list, pos_node_list):
    pos_idx_list.append(np.in1d(net[0], np.array(pos_node)).nonzero()[0])

# Compute physics similarity and feature similarity for comparison
feature_simi_list = []
physics_simi_list = []
label_simi_list = []

print('Computing similarity metrics...')
for net, pos_idx in zip(net_list, pos_idx_list):
    feature_simi = 0
    physics_simi = 0
    label_simi = 0

    if mode == 'pos':  # compute metrics only for anomalous nodes
        for idx in pos_idx:
            u, v = net[0][idx], net[1][idx]
            # Feature similarity (RBF kernel)
            feature_simi += np.exp(-1 * np.square(np.linalg.norm(feature[u] - feature[v])))
            # Physics-aware similarity (Eq. 5)
            physics_simi += np.exp(
                -1 * np.square(np.linalg.norm(feature[u] - feature[v])) / (0.1 * residuals[u, v] + 1e-8))
            # Label similarity
            label_simi += label[u] == label[v]

        feature_simi = feature_simi / pos_idx.size
        physics_simi = physics_simi / pos_idx.size
        label_simi = label_simi / pos_idx.size

    else:  # compute metrics for all nodes
        for u, v in zip(net[0].tolist(), net[1].tolist()):
            feature_simi += np.exp(-1 * np.square(np.linalg.norm(feature[u] - feature[v])))
            physics_simi += np.exp(
                -1 * np.square(np.linalg.norm(feature[u] - feature[v])) / (0.1 * residuals[u, v] + 1e-8))
            label_simi += label[u] == label[v]

        feature_simi = feature_simi / net[0].size
        physics_simi = physics_simi / net[0].size
        label_simi = label_simi / net[0].size

    feature_simi_list.append(feature_simi)
    physics_simi_list.append(physics_simi)
    label_simi_list.append(label_simi)

print(f'Feature similarity (camouflaged nodes): {feature_simi_list}')
print(f'Physics-aware similarity (camouflaged nodes): {physics_simi_list}')
print(f'Label similarity: {label_simi_list}')

# ========== Compute Physics Consistency Loss Distribution ==========
# For Fig. 4 in the paper
print('\nComputing physics consistency loss distribution...')
normal_nodes = np.where(label == 0)[0]
anomaly_nodes = np.where(label == 1)[0]

normal_phys_loss = compute_node_physics_loss(residuals, net_list[0], normal_nodes)
anomaly_phys_loss = compute_node_physics_loss(residuals, net_list[0], anomaly_nodes)

print(f'Normal nodes - mean physics loss: {np.mean(normal_phys_loss):.4f}, std: {np.std(normal_phys_loss):.4f}')
print(f'Anomaly nodes - mean physics loss: {np.mean(anomaly_phys_loss):.4f}, std: {np.std(anomaly_phys_loss):.4f}')

# ========== Generate Camouflaged Anomalies ==========
# Eq. (14) in the paper
print('\nGenerating camouflaged anomalies...')
alpha = 0.7  # camouflage strength
noise_std = 0.01

features_cam = feature.copy()
for idx in anomaly_nodes:
    # Find neighboring normal nodes
    neighbors = net_list[0][1][net_list[0][0] == idx].tolist()
    if len(neighbors) > 0:
        neighbor_mean = np.mean(feature[neighbors], axis=0)
        noise = np.random.normal(0, noise_std, feature.shape[1])
        features_cam[idx] = (1 - alpha) * feature[idx] + alpha * neighbor_mean + noise

print(f'Camouflaged features generated for {len(anomaly_nodes)} anomalous nodes')

# ========== Save Processed Data ==========
print('\nSaving processed data...')
np.save(f'data/{data_name[:-4]}_features_processed.npy', feature)
np.save(f'data/{data_name[:-4]}_features_camouflaged.npy', features_cam)
np.save(f'data/{data_name[:-4]}_labels.npy', label)
np.save(f'data/{data_name[:-4]}_physics_residuals.npy', residuals)
np.save(f'data/{data_name[:-4]}_normal_phys_loss.npy', normal_phys_loss)
np.save(f'data/{data_name[:-4]}_anomaly_phys_loss.npy', anomaly_phys_loss)

print('Data processing completed!')