import pickle
import random as rd
import numpy as np
import scipy.sparse as sp
from scipy.io import loadmat
import copy as cp
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, roc_auc_score, \
    average_precision_score
from collections import defaultdict

"""
    Utility functions for CR-PGNN
    Paper: Camouflage-Resistant Graph Neural Networks for Power Grid Anomaly Detection

    Functions:
    - Load IEEE power grid data
    - Normalize features
    - Build adjacency lists
    - Split positive/negative nodes
    - Under-sampling for class imbalance
    - Test CR-PGNN and baseline models
"""


def load_power_grid_data(data):
    """
    Load graph, feature, and label for IEEE power system datasets
    :param data: dataset name [ieee14, ieee57, ieee118]
    :returns: [physical_adj, geographic_adj, logical_adj, homo_adj], feature, label, physics_residuals
    """
    prefix = 'data/'

    if data == 'ieee14':
        data_file = loadmat(prefix + 'IEEE14.mat')
        labels = data_file['label'].flatten()
        feat_data = data_file['features'].todense().A
        physics_residuals = data_file['physics_residuals'].todense().A

        # Load preprocessed adjacency lists
        with open(prefix + 'ieee14_physical_adjlists.pickle', 'rb') as file:
            physical_adj = pickle.load(file)
        with open(prefix + 'ieee14_geographic_adjlists.pickle', 'rb') as file:
            geographic_adj = pickle.load(file)
        with open(prefix + 'ieee14_logical_adjlists.pickle', 'rb') as file:
            logical_adj = pickle.load(file)
        with open(prefix + 'ieee14_homo_adjlists.pickle', 'rb') as file:
            homo_adj = pickle.load(file)

    elif data == 'ieee57':
        data_file = loadmat(prefix + 'IEEE57.mat')
        labels = data_file['label'].flatten()
        feat_data = data_file['features'].todense().A
        physics_residuals = data_file['physics_residuals'].todense().A

        with open(prefix + 'ieee57_physical_adjlists.pickle', 'rb') as file:
            physical_adj = pickle.load(file)
        with open(prefix + 'ieee57_geographic_adjlists.pickle', 'rb') as file:
            geographic_adj = pickle.load(file)
        with open(prefix + 'ieee57_logical_adjlists.pickle', 'rb') as file:
            logical_adj = pickle.load(file)
        with open(prefix + 'ieee57_homo_adjlists.pickle', 'rb') as file:
            homo_adj = pickle.load(file)

    elif data == 'ieee118':
        data_file = loadmat(prefix + 'IEEE118.mat')
        labels = data_file['label'].flatten()
        feat_data = data_file['features'].todense().A
        physics_residuals = data_file['physics_residuals'].todense().A

        with open(prefix + 'ieee118_physical_adjlists.pickle', 'rb') as file:
            physical_adj = pickle.load(file)
        with open(prefix + 'ieee118_geographic_adjlists.pickle', 'rb') as file:
            geographic_adj = pickle.load(file)
        with open(prefix + 'ieee118_logical_adjlists.pickle', 'rb') as file:
            logical_adj = pickle.load(file)
        with open(prefix + 'ieee118_homo_adjlists.pickle', 'rb') as file:
            homo_adj = pickle.load(file)
    else:
        raise ValueError(f'Unknown dataset: {data}')

    return [physical_adj, geographic_adj, logical_adj, homo_adj], feat_data, labels, physics_residuals


def normalize(mx):
    """
    Row-normalize sparse matrix
    Code from https://github.com/williamleif/graphsage-simple/
    """
    rowsum = np.array(mx.sum(1)) + 0.01
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_to_adjlist(sp_matrix, filename):
    """
    Transfer sparse matrix to adjacency list
    :param sp_matrix: the sparse matrix
    :param filename: the filename of adjlist
    """
    # Add self loop
    homo_adj = sp_matrix + sp.eye(sp_matrix.shape[0])
    # Create adj_list
    adj_lists = defaultdict(set)
    edges = homo_adj.nonzero()
    for index, node in enumerate(edges[0]):
        adj_lists[node].add(edges[1][index])
        adj_lists[edges[1][index]].add(node)
    with open(filename, 'wb') as file:
        pickle.dump(adj_lists, file)


def pos_neg_split(nodes, labels):
    """
    Find positive and negative nodes given a list of nodes and their labels
    :param nodes: a list of nodes
    :param labels: a list of node labels
    :returns: the split positive and negative nodes
    """
    pos_nodes = []
    neg_nodes = cp.deepcopy(nodes)
    aux_nodes = cp.deepcopy(nodes)
    for idx, label in enumerate(labels):
        if label == 1:
            pos_nodes.append(aux_nodes[idx])
            neg_nodes.remove(aux_nodes[idx])
    return pos_nodes, neg_nodes


def undersample(pos_nodes, neg_nodes, scale=1):
    """
    Under-sample the negative nodes to handle class imbalance
    :param pos_nodes: a list of positive nodes
    :param neg_nodes: a list negative nodes
    :param scale: the under-sampling scale
    :return: a list of under-sampled batch nodes
    """
    aux_nodes = cp.deepcopy(neg_nodes)
    aux_nodes = rd.sample(aux_nodes, k=int(len(pos_nodes) * scale))
    batch_nodes = pos_nodes + aux_nodes
    return batch_nodes


def test_baseline(test_cases, labels, model, batch_size, cuda):
    """
    Test the performance of baseline models (GCN, GAT, GraphSAGE)
    :param test_cases: a list of testing nodes
    :param labels: a list of testing node labels
    :param model: the GNN model
    :param batch_size: number of nodes in a batch
    :param cuda: whether to use GPU
    :returns: AUC, F1, Precision, Recall
    """
    test_batch_num = int(len(test_cases) / batch_size) + 1
    f1 = 0.0
    acc = 0.0
    recall = 0.0
    precision = 0.0
    pred_list = []

    for iteration in range(test_batch_num):
        i_start = iteration * batch_size
        i_end = min((iteration + 1) * batch_size, len(test_cases))
        batch_nodes = test_cases[i_start:i_end]
        batch_label = labels[i_start:i_end]

        if cuda:
            prob = model.to_prob(batch_nodes)
            pred = prob.data.cpu().numpy().argmax(axis=1)
        else:
            prob = model.to_prob(batch_nodes)
            pred = prob.data.numpy().argmax(axis=1)

        f1 += f1_score(batch_label, pred, average="macro")
        acc += accuracy_score(batch_label, pred)
        recall += recall_score(batch_label, pred, average="macro")
        precision += precision_score(batch_label, pred, average="macro", zero_division=0)
        pred_list.extend(prob.data.cpu().numpy()[:, 1].tolist())

    auc = roc_auc_score(labels, np.array(pred_list))
    ap = average_precision_score(labels, np.array(pred_list))

    print(f"GNN F1: {f1 / test_batch_num:.4f}")
    print(f"GNN Accuracy: {acc / test_batch_num:.4f}")
    print(f"GNN Precision: {precision / test_batch_num:.4f}")
    print(f"GNN Recall: {recall / test_batch_num:.4f}")
    print(f"GNN AUC: {auc:.4f}")
    print(f"GNN AP: {ap:.4f}")

    return auc, f1 / test_batch_num, precision / test_batch_num, recall / test_batch_num


def test_cr_pgnn(test_cases, labels, model, batch_size, cuda, physics_residuals=None):
    """
    Test the performance of CR-PGNN
    :param test_cases: a list of testing nodes
    :param labels: a list of testing node labels
    :param model: the CR-PGNN model
    :param batch_size: number of nodes in a batch
    :param cuda: whether to use GPU
    :param physics_residuals: AC power flow residuals for physics consistency evaluation
    :returns: GNN AUC, GNN F1, GNN Precision, GNN Recall, Physics AUC
    """
    test_batch_num = int(len(test_cases) / batch_size) + 1
    f1_gnn = 0.0
    acc_gnn = 0.0
    recall_gnn = 0.0
    precision_gnn = 0.0
    f1_phys = 0.0
    acc_phys = 0.0
    recall_phys = 0.0
    precision_phys = 0.0
    gnn_list = []
    phys_list = []

    for iteration in range(test_batch_num):
        i_start = iteration * batch_size
        i_end = min((iteration + 1) * batch_size, len(test_cases))
        batch_nodes = test_cases[i_start:i_end]
        batch_label = labels[i_start:i_end]

        if cuda:
            gnn_prob, phys_prob = model.to_prob(batch_nodes, batch_label, train_flag=False)
            gnn_pred = gnn_prob.data.cpu().numpy().argmax(axis=1)
            phys_pred = phys_prob.data.cpu().numpy().argmax(axis=1)
        else:
            gnn_prob, phys_prob = model.to_prob(batch_nodes, batch_label, train_flag=False)
            gnn_pred = gnn_prob.data.numpy().argmax(axis=1)
            phys_pred = phys_prob.data.numpy().argmax(axis=1)

        # GNN evaluation metrics
        f1_gnn += f1_score(batch_label, gnn_pred, average="macro")
        acc_gnn += accuracy_score(batch_label, gnn_pred)
        recall_gnn += recall_score(batch_label, gnn_pred, average="macro")
        precision_gnn += precision_score(batch_label, gnn_pred, average="macro", zero_division=0)
        gnn_list.extend(gnn_prob.data.cpu().numpy()[:, 1].tolist())

        # Physics consistency evaluation metrics
        f1_phys += f1_score(batch_label, phys_pred, average="macro")
        acc_phys += accuracy_score(batch_label, phys_pred)
        recall_phys += recall_score(batch_label, phys_pred, average="macro")
        precision_phys += precision_score(batch_label, phys_pred, average="macro", zero_division=0)
        phys_list.extend(phys_prob.data.cpu().numpy()[:, 1].tolist())

    # AUC scores
    auc_gnn = roc_auc_score(labels, np.array(gnn_list))
    ap_gnn = average_precision_score(labels, np.array(gnn_list))
    auc_phys = roc_auc_score(labels, np.array(phys_list))
    ap_phys = average_precision_score(labels, np.array(phys_list))

    print(f"GNN F1: {f1_gnn / test_batch_num:.4f}")
    print(f"GNN Accuracy: {acc_gnn / test_batch_num:.4f}")
    print(f"GNN Precision: {precision_gnn / test_batch_num:.4f}")
    print(f"GNN Recall: {recall_gnn / test_batch_num:.4f}")
    print(f"GNN AUC: {auc_gnn:.4f}")
    print(f"GNN AP: {ap_gnn:.4f}")
    print(f"Physics F1: {f1_phys / test_batch_num:.4f}")
    print(f"Physics Accuracy: {acc_phys / test_batch_num:.4f}")
    print(f"Physics Precision: {precision_phys / test_batch_num:.4f}")
    print(f"Physics Recall: {recall_phys / test_batch_num:.4f}")
    print(f"Physics AUC: {auc_phys:.4f}")
    print(f"Physics AP: {ap_phys:.4f}")

    return auc_gnn, f1_gnn / test_batch_num, precision_gnn / test_batch_num, recall_gnn / test_batch_num, auc_phys


def compute_camouflage_metrics(features, labels, adj_lists, physics_residuals, mode='pos'):
    """
    Compute feature similarity, physics similarity, and label similarity for camouflage analysis
    :param features: node feature matrix
    :param labels: node labels
    :param adj_lists: list of adjacency lists for each relation
    :param physics_residuals: AC power flow residuals
    :param mode: 'pos' for positive nodes only, 'all' for all nodes
    :returns: feature_simi_list, physics_simi_list, label_simi_list
    """
    pos_nodes = set(labels.nonzero()[0].tolist())
    node_list = [set(adj_list.keys()) for adj_list in adj_lists[:3]]
    pos_node_list = [list(net_nodes.intersection(pos_nodes)) for net_nodes in node_list]

    feature_simi_list = []
    physics_simi_list = []
    label_simi_list = []

    for r_idx, adj_list in enumerate(adj_lists[:3]):
        feature_simi = 0
        physics_simi = 0
        label_simi = 0
        edge_count = 0

        if mode == 'pos':
            for node in pos_node_list[r_idx]:
                for neighbor in adj_list[node]:
                    if neighbor in pos_nodes:
                        feature_simi += np.exp(-1 * np.square(np.linalg.norm(features[node] - features[neighbor])))
                        physics_simi += np.exp(-1 * np.square(np.linalg.norm(features[node] - features[neighbor])) /
                                               (0.1 * physics_residuals[node, neighbor] + 1e-8))
                        label_simi += labels[node] == labels[neighbor]
                        edge_count += 1
        else:
            for node in adj_list:
                for neighbor in adj_list[node]:
                    if node < neighbor:  # avoid double counting
                        feature_simi += np.exp(-1 * np.square(np.linalg.norm(features[node] - features[neighbor])))
                        physics_simi += np.exp(-1 * np.square(np.linalg.norm(features[node] - features[neighbor])) /
                                               (0.1 * physics_residuals[node, neighbor] + 1e-8))
                        label_simi += labels[node] == labels[neighbor]
                        edge_count += 1

        feature_simi_list.append(feature_simi / edge_count if edge_count > 0 else 0)
        physics_simi_list.append(physics_simi / edge_count if edge_count > 0 else 0)
        label_simi_list.append(label_simi / edge_count if edge_count > 0 else 0)

    return feature_simi_list, physics_simi_list, label_simi_list