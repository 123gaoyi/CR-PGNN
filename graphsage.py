import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
import random
import numpy as np

"""
    CR-PGNN: Camouflage-Resistant Graph Neural Network for Power Grid Anomaly Detection
    Paper: Camouflage-Resistant Graph Neural Networks for Power Grid Anomaly Detection

    Core modules:
    1. Multi-head Physics Consistency (MPC) module
    2. RL-based Adaptive Neighbor Selector
    3. Relation-aware Gated Aggregator
"""


class PhysicsConsistencyModule(nn.Module):
    """
    Multi-head Physics Consistency (MPC) Module
    Computes physics-aware similarity scores using AC power flow residuals
    """

    def __init__(self, hidden_dim, n_heads=4, cuda=False):
        super(PhysicsConsistencyModule, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.cuda = cuda

        # Projection weights for each head
        self.W_proj = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(hidden_dim, hidden_dim))
            for _ in range(n_heads)
        ])
        self.b_proj = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(hidden_dim))
            for _ in range(n_heads)
        ])
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
    """

    def __init__(self, state_dim=5, hidden_dim=32, action_dim=3, lr=0.001):
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
        self.lr = lr

        # Action mapping: 0: +delta, 1: -delta, 2: 0
        self.delta = 0.05

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)

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
        Actions: 0: increase threshold (+delta), 1: decrease threshold (-delta), 2: maintain
        """
        probs = self.policy_net(state.unsqueeze(0))
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)

        if action == 0:
            threshold_delta = self.delta
        elif action == 1:
            threshold_delta = -self.delta
        else:
            threshold_delta = 0

        return threshold_delta, log_prob

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
    """

    def __init__(self, embed_dim, n_relations=3, cuda=False):
        super(RelationAwareAggregator, self).__init__()
        self.embed_dim = embed_dim
        self.n_relations = n_relations
        self.cuda = cuda

        # Attention weights for relation fusion
        self.W_att = nn.Parameter(torch.FloatTensor(embed_dim, embed_dim * 2))
        self.v_att = nn.Parameter(torch.FloatTensor(embed_dim, 1))
        init.xavier_uniform_(self.W_att)
        init.xavier_uniform_(self.v_att)

    def forward(self, h_prev, messages, thresholds):
        """
        Compute gated aggregation
        Eq. (11-12): h_i^{(t)} = ReLU(h_i^{(t-1)} + sum(beta_{i,kappa} * m_i^{(t,kappa)}))
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

        # Aggregate messages
        combined = torch.zeros_like(h_prev)
        for kappa in range(self.n_relations):
            combined = combined + beta[:, kappa].unsqueeze(1) * messages[kappa]

        # Residual connection
        h_new = F.relu(h_prev + combined)

        return h_new


class CR_PGNN(nn.Module):
    """
    CR-PGNN: Complete Model Architecture
    Consists of:
    - Two-layer GNN with hidden dimension 64
    - MPC module with M=4 heads
    - RL-based adaptive neighbor selector
    - Relation-aware gated aggregation
    """

    def __init__(self, n_nodes, n_features, hidden_dim=64, n_classes=2,
                 n_relations=3, n_heads=4, cuda=False):
        super(CR_PGNN, self).__init__()

        self.n_nodes = n_nodes
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.n_relations = n_relations
        self.n_heads = n_heads
        self.cuda = cuda

        # Input projection
        self.W_init = nn.Parameter(torch.FloatTensor(hidden_dim, n_features))
        self.b_init = nn.Parameter(torch.FloatTensor(hidden_dim))
        init.xavier_uniform_(self.W_init)
        init.zeros_(self.b_init)

        # Batch normalization
        self.bn = nn.BatchNorm1d(hidden_dim)

        # Multi-head Physics Consistency module
        self.mpc = PhysicsConsistencyModule(hidden_dim, n_heads, cuda)

        # RL Neighbor Selector
        self.rl_selector = RLNeighborSelector()

        # Relation-aware Aggregator (for layer 2)
        self.aggregator = RelationAwareAggregator(hidden_dim, n_relations, cuda)

        # Output layer
        self.classifier = nn.Linear(hidden_dim, n_classes)
        init.xavier_uniform_(self.classifier.weight)
        init.zeros_(self.classifier.bias)

        # Initialize filtering thresholds
        self.thresholds = nn.ParameterList([
            nn.Parameter(torch.FloatTensor([0.5])) for _ in range(n_relations)
        ])
        for p in self.thresholds:
            p.data.fill_(0.5)

    def forward(self, features, adj_lists, physics_residuals):
        """
        Forward pass of CR-PGNN

        Args:
            features: Node feature matrix [n_nodes x n_features]
            adj_lists: List of adjacency lists for each relation
            physics_residuals: AC power flow residuals [n_edges]

        Returns:
            logits: Classification logits [n_nodes x n_classes]
        """
        # Step 1: Feature alignment (Eq. 2)
        h = F.leaky_relu(self.bn(torch.mm(self.W_init, features.t()).t() + self.b_init))

        # Step 2: Filter neighbors using physics consistency
        filtered_adj = []
        for kappa in range(self.n_relations):
            # Compute similarity scores and filter
            threshold = self.thresholds[kappa]
            # Apply RL-based filtering (simplified)
            filtered_neighbors = self.apply_filtering(h, adj_lists[kappa],
                                                      physics_residuals, threshold)
            filtered_adj.append(filtered_neighbors)

        # Step 3: Message passing (Eq. 11)
        messages = []
        for kappa in range(self.n_relations):
            # Aggregate neighbor features
            m = self.aggregate_neighbors(h, filtered_adj[kappa])
            messages.append(m)

        # Step 4: Relation-aware gated aggregation (Eq. 12)
        thresholds_vals = torch.tensor([p.item() for p in self.thresholds])
        h = self.aggregator(h, messages, thresholds_vals)

        # Step 5: Classification
        logits = self.classifier(h)

        return logits

    def apply_filtering(self, h, adj, residuals, threshold):
        """Filter neighbors based on similarity scores"""
        # Compute similarity scores and filter
        # Simplified: keep neighbors with similarity >= threshold
        filtered = []
        # Implementation depends on graph structure
        return filtered

    def aggregate_neighbors(self, h, adj):
        """Aggregate neighbor features (mean aggregation)"""
        # Simplified mean aggregation
        # Full implementation depends on graph structure
        return h

    def compute_physics_loss(self, residuals):
        """Compute node-level physics consistency loss (Eq. 6)"""
        phys_loss = torch.mean(residuals)
        return phys_loss


class CR_PGNNDetector(nn.Module):
    """
    Complete CR-PGNN Detector with Training Logic
    Uses dual-objective loss: Weighted Cross-Entropy + RL Reward
    """

    def __init__(self, n_nodes, n_features, hidden_dim=64,
                 n_classes=2, n_relations=3, n_heads=4,
                 pos_weight=1.0, cuda=False):
        super(CR_PGNNDetector, self).__init__()

        self.model = CR_PGNN(n_nodes, n_features, hidden_dim,
                             n_classes, n_relations, n_heads, cuda)
        self.pos_weight = pos_weight

        # Class weights for imbalanced data
        self.class_weights = torch.FloatTensor([1.0, pos_weight])
        if cuda:
            self.class_weights = self.class_weights.cuda()
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)

    def forward(self, features, adj_lists, physics_residuals):
        return self.model(features, adj_lists, physics_residuals)

    def compute_task_loss(self, logits, labels):
        """Weighted Cross-Entropy Loss (Eq. 13)"""
        return self.criterion(logits, labels)

    def compute_reward(self, f1_score, phys_loss_prev, phys_loss_curr, n_neighbors):
        """Compute RL reward (Eq. 10): R = F1 - lambda * Delta_phys - eta * |N|"""
        lambda_val = 0.1
        eta_val = 0.01
        delta_phys = phys_loss_curr - phys_loss_prev
        reward = f1_score - lambda_val * delta_phys - eta_val * n_neighbors
        return reward

    def train_step(self, features, adj_lists, physics_residuals, labels, optimizer):
        """Single training step with joint optimization"""
        optimizer.zero_grad()

        logits = self.model(features, adj_lists, physics_residuals)
        task_loss = self.compute_task_loss(logits, labels)

        task_loss.backward()
        optimizer.step()

        return task_loss.item()