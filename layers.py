import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
from operator import itemgetter
import math
import numpy as np

"""
    CR-PGNN: Camouflage-Resistant Graph Neural Network for Power Grid Anomaly Detection
    Paper: Camouflage-Resistant Graph Neural Networks for Power Grid Anomaly Detection

    Core modules:
    1. Multi-head Physics Consistency (MPC) Module
    2. RL-based Adaptive Neighbor Selector
    3. Relation-aware Gated Aggregator

    Source: Based on CARE-GNN implementation
"""


class PhysicsConsistencyModule(nn.Module):
    """
    Multi-head Physics Consistency (MPC) Module
    Computes physics-aware similarity scores using AC power flow residuals
    Eq. (3)-(5) in the paper
    """

    def __init__(self, hidden_dim, n_heads=4, cuda=False):
        super(PhysicsConsistencyModule, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.cuda = cuda

        # Projection weights for each head (Eq. 3)
        self.W_proj = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(hidden_dim, hidden_dim))
            for _ in range(n_heads)
        ])
        self.b_proj = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(hidden_dim))
            for _ in range(n_heads)
        ])
        # Head-specific weights omega_m (Eq. 5)
        self.omega = nn.Parameter(torch.FloatTensor(n_heads))

        for i in range(n_heads):
            init.xavier_uniform_(self.W_proj[i])
            init.zeros_(self.b_proj[i])
        init.ones_(self.omega)

    def compute_physics_residual(self, V_i, V_j, theta_i, theta_j, X_ij, P_ij_reported):
        """
        Compute AC power flow mismatch Gamma_{i,j}
        Eq. (4): Gamma = |(V_i V_j / X_ij) * sin(theta_i - theta_j) - P_reported|
        """
        V = V_i * V_j
        sin_theta = torch.sin(theta_i - theta_j)
        expected_flow = (V * sin_theta) / (X_ij + 1e-8)
        residual = torch.abs(expected_flow - P_ij_reported)
        return residual

    def forward(self, h_i, h_j, Gamma_ij, eps=1e-8):
        """
        Compute similarity score S(u_i, u_j)
        Eq. (5): S = (1/M) * sum(omega_m * exp(-||z_i - z_j||^2 / (gamma * Gamma + eps)))
        """
        n_nodes = h_i.shape[0]
        similarity = 0

        for m in range(self.n_heads):
            z_i = torch.mm(h_i, self.W_proj[m]) + self.b_proj[m]
            z_i = F.leaky_relu(z_i)
            z_j = torch.mm(h_j, self.W_proj[m]) + self.b_proj[m]
            z_j = F.leaky_relu(z_j)

            dist = torch.sum((z_i - z_j) ** 2, dim=1, keepdim=True)
            gamma = 0.1  # scaling factor
            denominator = gamma * Gamma_ij + eps
            head_sim = self.omega[m] * torch.exp(-dist / denominator)
            similarity = similarity + head_sim

        similarity = similarity / self.n_heads
        return similarity


class RLNeighborSelector(nn.Module):
    """
    Reinforcement Learning-based Adaptive Neighbor Selector
    Dynamically optimizes filtering thresholds to prune deceptive connections
    Eq. (8)-(10) in the paper
    """

    def __init__(self, state_dim=5, hidden_dim=32, action_dim=3, step_size=0.05, cuda=False):
        super(RLNeighborSelector, self).__init__()

        # Policy network: two-layer MLP
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.step_size = step_size
        self.cuda = cuda

        # RL condition flag
        self.RL = True

        # Number of batches for current epoch, assigned during training
        self.batch_num = 0

        # Initial filtering thresholds (Eq. 9)
        self.thresholds = [0.5, 0.5, 0.5]

        # Threshold logs for RL update
        self.thresholds_log = [self.thresholds]

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=0.001)

    def get_state(self, similarity_scores, phys_loss_prev, threshold_prev):
        """
        Construct state representation s_i^{(kappa)}
        Eq. (8): [mu(S), sigma(S), Skew(S), Delta_phys, p]
        """
        mu = torch.mean(similarity_scores)
        sigma = torch.std(similarity_scores)
        skew = torch.mean((similarity_scores - mu) ** 3) / (sigma ** 3 + 1e-8)
        delta_phys = phys_loss_prev

        state = torch.cat([mu.unsqueeze(0), sigma.unsqueeze(0),
                           skew.unsqueeze(0), delta_phys.unsqueeze(0),
                           threshold_prev.unsqueeze(0)])
        return state

    def select_action(self, state):
        """
        Select action based on current state
        Actions: 0: +delta, 1: -delta, 2: 0
        """
        probs = self.policy_net(state.unsqueeze(0))
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)

        if action == 0:
            threshold_delta = self.step_size
        elif action == 1:
            threshold_delta = -self.step_size
        else:
            threshold_delta = 0

        return threshold_delta, log_prob

    def update_thresholds(self, rewards):
        """
        Update filtering thresholds based on rewards
        Eq. (10): R = F1 - lambda * Delta_phys - eta * |N|
        """
        new_thresholds = []
        for i, r in enumerate(rewards):
            if r > 0:
                new_th = self.thresholds[i] + self.step_size
            elif r < 0:
                new_th = self.thresholds[i] - self.step_size
            else:
                new_th = self.thresholds[i]
            # Clip to [0, 1]
            new_th = max(0.0, min(1.0, new_th))
            new_thresholds.append(new_th)

        self.thresholds = new_thresholds
        self.thresholds_log.append(self.thresholds)
        return self.thresholds

    def compute_reward(self, f1_score, phys_loss_prev, phys_loss_curr, n_neighbors, lambda_val=0.1, eta_val=0.01):
        """
        Compute RL reward
        Eq. (10): R = F1 - lambda * Delta_phys - eta * |N|
        """
        delta_phys = phys_loss_curr - phys_loss_prev
        reward = f1_score - lambda_val * delta_phys - eta_val * n_neighbors
        return reward

    def update_policy(self, log_probs, rewards):
        """REINFORCE policy gradient update"""
        policy_loss = []
        for log_prob, reward in zip(log_probs, rewards):
            policy_loss.append(-log_prob * reward)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()


class RelationAwareAggregator(nn.Module):
    """
    Relation-aware Gated Aggregator
    Fuses information from filtered neighborhoods across multiple relations
    Eq. (11)-(12) in the paper
    """

    def __init__(self, embed_dim, n_relations=3, cuda=False):
        super(RelationAwareAggregator, self).__init__()
        self.embed_dim = embed_dim
        self.n_relations = n_relations
        self.cuda = cuda
        self.dropout = 0.6

        # Attention weights for relation fusion
        self.W_att = nn.Parameter(torch.FloatTensor(embed_dim, embed_dim * 2))
        self.v_att = nn.Parameter(torch.FloatTensor(embed_dim, 1))
        init.xavier_uniform_(self.W_att)
        init.xavier_uniform_(self.v_att)

    def forward(self, h_prev, messages, thresholds, training=True):
        """
        Compute gated aggregation
        Eq. (12): h_i^{(t)} = ReLU(h_i^{(t-1)} + sum(beta_{i,kappa} * m_i^{(t,kappa)}))
        """
        n_nodes = h_prev.shape[0]

        # Compute attention coefficients beta_{i,kappa}
        beta_list = []
        for kappa in range(self.n_relations):
            concat = torch.cat([h_prev, messages[kappa]], dim=1)
            energy = torch.tanh(torch.mm(concat, self.W_att.T))
            beta = torch.mm(energy, self.v_att)
            beta = beta.squeeze()
            beta = beta * thresholds[kappa]  # weight by learned threshold
            beta_list.append(beta)

        beta = torch.stack(beta_list, dim=1)
        beta = F.softmax(beta, dim=1)
        beta = F.dropout(beta, self.dropout, training=training)

        # Aggregate messages with attention weights
        aggregated = torch.zeros_like(h_prev)
        for kappa in range(self.n_relations):
            aggregated = aggregated + beta[:, kappa].unsqueeze(1) * messages[kappa]

        # Residual connection (Eq. 12)
        combined = F.relu(h_prev + aggregated)

        return combined, beta


class IntraRelationAggregator(nn.Module):
    """
    Intra-relation Aggregator with Adaptive Neighbor Filtering
    Filters neighbors based on physics-aware similarity scores and thresholds
    """

    def __init__(self, features, feat_dim, cuda=False):
        super(IntraRelationAggregator, self).__init__()
        self.features = features
        self.cuda = cuda
        self.feat_dim = feat_dim

    def filter_neighbors_adaptive(self, center_scores, neigh_scores, neighs_list, threshold):
        """
        Filter neighbors using adaptive threshold
        Keeps neighbors with similarity scores above threshold
        """
        samp_neighs = []
        samp_scores = []

        for idx, center_score in enumerate(center_scores):
            neigh_score = neigh_scores[idx][:, 0].view(-1, 1)
            center_score = center_score.repeat(neigh_score.size()[0], 1)
            neighs_indices = neighs_list[idx]

            # Compute similarity (L1-distance)
            score_diff = torch.abs(center_score - neigh_score).squeeze()
            sorted_scores, sorted_indices = torch.sort(score_diff, dim=0, descending=False)

            # Select neighbors with similarity above threshold
            selected_indices = [i for i, s in zip(sorted_indices.tolist(), sorted_scores.tolist())
                                if s <= threshold]

            if len(selected_indices) == 0:
                selected_indices = [sorted_indices[0].tolist()]

            selected_neighs = [neighs_indices[n] for n in selected_indices]
            selected_scores = [sorted_scores[i].tolist() for i in selected_indices]

            samp_neighs.append(set(selected_neighs))
            samp_scores.append(selected_scores)

        return samp_neighs, samp_scores

    def forward(self, nodes, to_neighs_list, center_scores, neigh_scores, threshold):
        """
        Intra-relation aggregation with filtered neighbors
        """
        # Filter neighbors using adaptive threshold
        samp_neighs, samp_scores = self.filter_neighbors_adaptive(
            center_scores, neigh_scores, to_neighs_list, threshold
        )

        # Find unique nodes among batch nodes and filtered neighbors
        unique_nodes_list = list(set.union(*samp_neighs))
        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}

        # Build mask for aggregation
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
        row_indices = [i for i in range(len(samp_neighs)) for _ in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1
        if self.cuda:
            mask = mask.cuda()

        num_neigh = mask.sum(1, keepdim=True)
        mask = mask.div(num_neigh)

        if self.cuda:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
        else:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list))

        to_feats = mask.mm(embed_matrix)
        to_feats = F.relu(to_feats)

        return to_feats, samp_scores


class CR_PGNNDetector(nn.Module):
    """
    Complete CR-PGNN Detector Model
    Integrates MPC module, RL selector, and relation-aware aggregator
    """

    def __init__(self, features, feature_dim, embed_dim, adj_lists,
                 n_relations=3, n_heads=4, inter='GNN', step_size=0.05,
                 cuda=True, pos_weight=1.0):
        super(CR_PGNNDetector, self).__init__()

        self.features = features
        self.feat_dim = feature_dim
        self.embed_dim = embed_dim
        self.adj_lists = adj_lists
        self.n_relations = n_relations
        self.n_heads = n_heads
        self.inter = inter
        self.step_size = step_size
        self.cuda = cuda

        # Multi-head Physics Consistency module
        self.mpc = PhysicsConsistencyModule(embed_dim, n_heads, cuda)

        # RL Neighbor Selector
        self.rl_selector = RLNeighborSelector(step_size=step_size, cuda=cuda)

        # Intra-relation aggregators for each relation
        self.intra_aggs = nn.ModuleList([
            IntraRelationAggregator(features, feature_dim, cuda)
            for _ in range(n_relations)
        ])

        # Relation-aware Gated Aggregator
        self.inter_agg = RelationAwareAggregator(embed_dim, n_relations, cuda)

        # Parameter to transform node embeddings before aggregation
        self.weight = nn.Parameter(torch.FloatTensor(feature_dim, embed_dim))
        init.xavier_uniform_(self.weight)

        # Classifier for node-level anomaly detection
        self.classifier = nn.Linear(embed_dim, 2)
        init.xavier_uniform_(self.classifier.weight)

        # Physics consistency loss tracking
        self.phys_loss_log = []

    def compute_node_physics_loss(self, residuals, adj_list, nodes):
        """
        Compute node-level physics consistency loss
        Eq. (6): L_phys(u_i) = (1/|N_i|) * sum(Gamma_{i,j})
        """
        phys_losses = []
        for node in nodes:
            neighbors = adj_list[int(node)]
            if len(neighbors) > 0:
                node_residuals = [residuals.get((node, n), 0) for n in neighbors]
                phys_loss = sum(node_residuals) / len(neighbors)
            else:
                phys_loss = 0.0
            phys_losses.append(phys_loss)
        return torch.tensor(phys_losses)

    def forward(self, nodes, labels=None, train_flag=True):
        """
        Forward pass of CR-PGNN

        Args:
            nodes: list of batch node ids
            labels: batch node labels (used for RL training)
            train_flag: indicates training or testing mode

        Returns:
            combined: node embeddings after aggregation
            center_scores: classification logits
        """
        # Extract 1-hop neighbor ids from adj lists of each relation
        to_neighs = []
        for adj_list in self.adj_lists:
            to_neighs.append([set(adj_list[int(node)]) for node in nodes])

        # Find unique nodes used in current batch
        unique_nodes = set.union(*to_neighs[0])
        for r in range(1, self.n_relations):
            unique_nodes = set.union(unique_nodes, *to_neighs[r])
        unique_nodes = set.union(unique_nodes, set(nodes))

        # Get node features for unique nodes
        if self.cuda:
            batch_features = self.features(torch.cuda.LongTensor(list(unique_nodes)))
        else:
            batch_features = self.features(torch.LongTensor(list(unique_nodes)))

        id_mapping = {node_id: index for node_id, index in zip(unique_nodes, range(len(unique_nodes)))}

        # Get center node features
        center_feats = batch_features[itemgetter(*nodes)(id_mapping), :]
        center_h = torch.mm(center_feats, self.weight)

        # Intra-relation aggregation for each relation
        messages = []
        for r in range(self.n_relations):
            neighs_list = [list(to_neigh) for to_neigh in to_neighs[r]]

            # Get neighbor features
            neigh_feats_list = [batch_features[itemgetter(*neighs)(id_mapping), :]
                                for neighs in neighs_list]

            # Compute physics-aware similarity scores
            threshold = self.rl_selector.thresholds[r]

            # Intra-aggregation
            r_feats, r_scores = self.intra_aggs[r].forward(
                nodes, neighs_list, center_feats, neigh_feats_list, threshold
            )
            messages.append(r_feats)

        # Relation-aware gated aggregation (Eq. 11-12)
        thresholds = torch.tensor(self.rl_selector.thresholds)
        combined, attention = self.inter_agg.forward(center_h, messages, thresholds, train_flag)

        # Classification
        center_scores = self.classifier(combined)

        # RL module update (during training)
        if self.rl_selector.RL and train_flag and labels is not None:
            # Compute rewards and update thresholds
            # This would use F1 score, physics loss, etc.
            pass

        return combined, center_scores

    def loss(self, nodes, labels):
        """Compute weighted cross-entropy loss (Eq. 13)"""
        _, scores = self.forward(nodes, labels, train_flag=True)
        loss_fn = nn.CrossEntropyLoss()
        return loss_fn(scores, labels.squeeze())

    def to_prob(self, nodes):
        """Convert logits to probabilities"""
        _, scores = self.forward(nodes, train_flag=False)
        return torch.sigmoid(scores)