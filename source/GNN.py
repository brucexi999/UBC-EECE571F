import torch
import torch.nn as nn


class MPNN(nn.Module):
    def __init__(self, node_feature_dimension: int, action_space_dimension: int, nn_type: str):
        super(MPNN, self).__init__()

        # Message-passing network
        # TODO: how to make sure the architecture can generalize and does not depend on the input graph's topology?
        self.nn_type = nn_type

        self.message = nn.Sequential(
            nn.Linear(node_feature_dimension * 2 + 1, node_feature_dimension),
            nn.ReLU()
        )

        # Update network
        self.update = nn.GRUCell(node_feature_dimension, node_feature_dimension)

        # Readout network
        self.readout_policy = nn.Linear(node_feature_dimension, action_space_dimension)
        self.readout_value = nn.Linear(node_feature_dimension, 1)

    def forward(self, node_features: torch.Tensor, edge_features: torch.Tensor, adjacency_matrix: torch.Tensor):
        for _ in range(3):
            # Identify the source and target nodes for each edge
            edge_indices = adjacency_matrix.nonzero(as_tuple=True)
            source_indices, target_indices = edge_indices

            # Gather the source and target node features for all edges
            source_features = node_features[source_indices]
            target_features = node_features[target_indices]

            # Gather the edge features
            edge_feats = edge_features[source_indices, target_indices].unsqueeze(-1)

            # Concatenate the source node features, target node features, and edge features
            combined_features = torch.cat([source_features, target_features, edge_feats], dim=1)

            # Compute messages in parallel for all edges
            messages = self.message(combined_features)

            # Aggregate messages by summing them for each target node
            aggregated_messages = torch.zeros_like(node_features).index_add_(0, target_indices, messages)

            # Update node features with a GRU
            updated_node_features = self.update(aggregated_messages, node_features)

        # Readout
        node_feature_sum = updated_node_features.sum(dim=0)
        if (self.nn_type == "policy"):
            policy_logits = self.readout_policy(node_feature_sum)
            return policy_logits
        elif (self.nn_type == "value"):
            v_value = self.readout_value(node_feature_sum)
            return v_value
