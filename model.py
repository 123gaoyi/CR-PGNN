import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

"""
    CR-PGNN: Camouflage-Resistant Graph Neural Network for Power Grid Anomaly Detection
    Paper: Camouflage-Resistant Graph Neural Networks for Power Grid Anomaly Detection

    Core modules:
    1. Multi-head Physics Consistency (MPC) Module
    2. RL-based Adaptive Neighbor Selector
    3. Relation-aware Gated Aggregator

    Source: Based on CARE-GNN implementation
"""


class OneLayerCR_PGNN(nn.Module):
    """
    The CR-PGNN model in one layer
    """

    def __init__(self, num_classes, inter_agg, lambda_phys, lambda_rl):
        """
        Initialize the CR-PGNN model
        :param num_classes: number of classes (2 in our paper)
        :param inter_agg: the inter-relation aggregator that outputs the final embedding
        :param lambda_phys: weight for physics consistency loss
        :param lambda_rl: weight for RL reward loss
        """
        super(OneLayerCR_PGNN, self).__init__()
        self.inter_agg = inter_agg
        self.xent = nn.CrossEntropyLoss()

        # Parameter to transform the final embedding to classification scores
        self.weight = nn.Parameter(torch.FloatTensor(inter_agg.embed_dim, num_classes))
        init.xavier_uniform_(self.weight)
        self.lambda_phys = lambda_phys
        self.lambda_rl = lambda_rl

    def forward(self, nodes, labels=None, train_flag=True):
        """
        Forward pass of CR-PGNN
        :param nodes: list of batch node ids
        :param labels: batch node labels (used for RL training)
        :param train_flag: indicates training or testing mode
        :return scores: classification logits
        :return phys_scores: physics consistency scores
        """
        embeds, phys_scores, thresholds = self.inter_agg(nodes, labels, train_flag)
        scores = torch.mm(embeds, self.weight)
        return scores, phys_scores, thresholds

    def to_prob(self, nodes, labels=None, train_flag=True):
        """
        Convert logits to probabilities
        :param nodes: list of batch node ids
        :param labels: batch node labels
        :param train_flag: indicates training or testing mode
        :return gnn_prob: classification probabilities
        :return phys_prob: physics consistency probabilities
        """
        gnn_scores, phys_scores, _ = self.forward(nodes, labels, train_flag)
        gnn_prob = F.softmax(gnn_scores, dim=1)
        phys_prob = F.softmax(phys_scores, dim=1)
        return gnn_prob, phys_prob

    def loss(self, nodes, labels, train_flag=True):
        """
        Compute the joint loss function of CR-PGNN
        :param nodes: list of batch node ids
        :param labels: batch node labels
        :param train_flag: indicates training or testing mode
        :return final_loss: combined loss for backpropagation
        """
        gnn_scores, phys_scores, _ = self.forward(nodes, labels, train_flag)

        # Physics consistency loss (Eq. 6 in the paper)
        phys_loss = self.xent(phys_scores, labels.squeeze())

        # GNN classification loss (Weighted Cross-Entropy, Eq. 13)
        # Apply class weights to handle imbalanced data
        class_weights = torch.FloatTensor([1.0, self.lambda_rl]).to(gnn_scores.device)
        weighted_xent = nn.CrossEntropyLoss(weight=class_weights)
        gnn_loss = weighted_xent(gnn_scores, labels.squeeze())

        # Total loss function of CR-PGNN
        # Combines task loss and physics consistency regularization
        final_loss = gnn_loss + self.lambda_phys * phys_loss

        return final_loss


class MultiLayerCR_PGNN(nn.Module):
    """
    Multi-layer CR-PGNN model for deeper architecture
    Supports 2 or 3 layers as optimal depth from sensitivity analysis
    """

    def __init__(self, num_classes, inter_agg_layers, lambda_phys, lambda_rl, n_layers=2):
        """
        Initialize the multi-layer CR-PGNN model
        :param num_classes: number of classes (2 in our paper)
        :param inter_agg_layers: list of inter-relation aggregators for each layer
        :param lambda_phys: weight for physics consistency loss
        :param lambda_rl: weight for RL reward loss
        :param n_layers: number of GNN layers (optimal: 2 or 3)
        """
        super(MultiLayerCR_PGNN, self).__init__()

        self.n_layers = n_layers
        self.inter_aggs = nn.ModuleList(inter_agg_layers)
        self.xent = nn.CrossEntropyLoss()

        # Final classification layer
        self.weight = nn.Parameter(torch.FloatTensor(
            inter_agg_layers[-1].embed_dim, num_classes
        ))
        init.xavier_uniform_(self.weight)
        self.lambda_phys = lambda_phys
        self.lambda_rl = lambda_rl

        # Layer-wise threshold logs for analysis
        self.thresholds_log = []

    def forward(self, nodes, labels=None, train_flag=True):
        """
        Forward pass through multiple GNN layers
        :param nodes: list of batch node ids
        :param labels: batch node labels
        :param train_flag: indicates training or testing mode
        :return scores: classification logits
        :return phys_scores: physics consistency scores from final layer
        :return all_thresholds: filtering thresholds from all layers
        """
        embeds = None
        phys_scores = None
        all_thresholds = []

        for layer_idx, inter_agg in enumerate(self.inter_aggs):
            embeds, phys_scores, thresholds = inter_agg(nodes, labels, train_flag)
            all_thresholds.append(thresholds)

            # Update nodes for next layer (using current embeddings as new features)
            # This enables hierarchical representation learning
            if layer_idx < self.n_layers - 1:
                # Store thresholds for analysis
                if train_flag:
                    self.thresholds_log.append(thresholds)

        scores = torch.mm(embeds, self.weight)
        return scores, phys_scores, all_thresholds

    def to_prob(self, nodes, labels=None, train_flag=True):
        """
        Convert logits to probabilities
        """
        gnn_scores, phys_scores, _ = self.forward(nodes, labels, train_flag)
        gnn_prob = F.softmax(gnn_scores, dim=1)
        phys_prob = F.softmax(phys_scores, dim=1)
        return gnn_prob, phys_prob

    def loss(self, nodes, labels, train_flag=True):
        """
        Compute the joint loss function for multi-layer CR-PGNN
        Includes layer-wise physics consistency regularization
        """
        gnn_scores, phys_scores, all_thresholds = self.forward(nodes, labels, train_flag)

        # Physics consistency loss (Eq. 6)
        phys_loss = self.xent(phys_scores, labels.squeeze())

        # Weighted classification loss for imbalanced data (Eq. 13)
        class_weights = torch.FloatTensor([1.0, self.lambda_rl]).to(gnn_scores.device)
        weighted_xent = nn.CrossEntropyLoss(weight=class_weights)
        gnn_loss = weighted_xent(gnn_scores, labels.squeeze())

        # RL threshold regularization
        # Penalizes extreme threshold values to maintain stable filtering
        rl_reg = 0
        for thresholds in all_thresholds:
            for th in thresholds:
                rl_reg = rl_reg + (th - 0.5) ** 2
        rl_reg = rl_reg / len(all_thresholds) if len(all_thresholds) > 0 else 0

        # Total loss function
        final_loss = gnn_loss + self.lambda_phys * phys_loss + 0.01 * rl_reg

        return final_loss

    def get_thresholds_log(self):
        """
        Return the history of filtering thresholds for analysis
        Used for interpretability (Fig. 5 in the paper)
        """
        return self.thresholds_log


class CR_PGNNWithRL(nn.Module):
    """
    CR-PGNN model with integrated RL-based threshold optimization
    Implements the complete camouflage-resistant detection framework
    """

    def __init__(self, num_classes, inter_agg, rl_selector, lambda_phys, lambda_rl, lambda_sparse):
        """
        Initialize CR-PGNN with RL-based adaptive threshold selection
        :param num_classes: number of classes (2 in our paper)
        :param inter_agg: the inter-relation aggregator
        :param rl_selector: RL neighbor selector module
        :param lambda_phys: weight for physics consistency loss
        :param lambda_rl: weight for RL reward loss
        :param lambda_sparse: weight for sparsity penalty in reward (Eq. 10)
        """
        super(CR_PGNNWithRL, self).__init__()

        self.inter_agg = inter_agg
        self.rl_selector = rl_selector
        self.xent = nn.CrossEntropyLoss()

        # Classification layer
        self.weight = nn.Parameter(torch.FloatTensor(inter_agg.embed_dim, num_classes))
        init.xavier_uniform_(self.weight)

        self.lambda_phys = lambda_phys
        self.lambda_rl = lambda_rl
        self.lambda_sparse = lambda_sparse

        # Tracking metrics for RL reward
        self.prev_phys_loss = None
        self.reward_history = []

    def forward(self, nodes, labels=None, train_flag=True):
        """
        Forward pass with RL-guided threshold adaptation
        """
        embeds, phys_scores, thresholds = self.inter_agg(nodes, labels, train_flag)
        scores = torch.mm(embeds, self.weight)
        return scores, phys_scores, thresholds

    def compute_rl_reward(self, f1_score, phys_loss_curr, n_neighbors):
        """
        Compute RL reward for threshold update
        Eq. (10): R = F1 - lambda * Delta_phys - eta * |N|
        """
        if self.prev_phys_loss is None:
            self.prev_phys_loss = phys_loss_curr
            return 0

        delta_phys = phys_loss_curr - self.prev_phys_loss
        reward = f1_score - self.lambda_rl * delta_phys - self.lambda_sparse * n_neighbors

        self.reward_history.append(reward)
        self.prev_phys_loss = phys_loss_curr

        return reward

    def to_prob(self, nodes, labels=None, train_flag=True):
        """
        Convert logits to probabilities
        """
        gnn_scores, phys_scores, _ = self.forward(nodes, labels, train_flag)
        gnn_prob = F.softmax(gnn_scores, dim=1)
        phys_prob = F.softmax(phys_scores, dim=1)
        return gnn_prob, phys_prob

    def loss(self, nodes, labels, train_flag=True):
        """
        Compute the joint loss function including RL reward optimization
        """
        gnn_scores, phys_scores, thresholds = self.forward(nodes, labels, train_flag)

        # Physics consistency loss (Eq. 6)
        phys_loss = self.xent(phys_scores, labels.squeeze())

        # Weighted classification loss for imbalanced data (Eq. 13)
        class_weights = torch.FloatTensor([1.0, self.lambda_rl]).to(gnn_scores.device)
        weighted_xent = nn.CrossEntropyLoss(weight=class_weights)
        gnn_loss = weighted_xent(gnn_scores, labels.squeeze())

        # RL-based adaptive loss
        # The RL loss is computed externally via policy gradient
        # This is the task loss component

        # Total loss function of CR-PGNN with RL
        final_loss = gnn_loss + self.lambda_phys * phys_loss

        return final_loss

    def update_thresholds_with_reward(self, rewards):
        """
        Update filtering thresholds based on computed rewards
        Implements the RL update rule from Eq. (8)-(10)
        """
        self.rl_selector.update_thresholds(rewards)

    def get_current_thresholds(self):
        """
        Return current filtering thresholds for each relation
        """
        return self.rl_selector.thresholds