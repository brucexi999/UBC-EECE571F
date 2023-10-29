import torch
import torch.nn as nn


class MPNN(nn.Module):
    def __init__(self, node_feature_dimension: int, action_space_dimension: int):
        super(MPNN, self).__init__()
        
        # Message-passing network
        # TODO: how to make sure the architecture can generalize and does not depend on the input graph's topology?
        self.message = nn.Sequential(
            nn.Linear(node_feature_dimension * 2 + 1, node_feature_dimension),
            nn.ReLU()
        )

        # Update network
        self.update = nn.GRUCell(node_feature_dimension, node_feature_dimension)

        # Readout networks
        # There are two separate readout networks; one for policy, one for value
        self.readout_policy = nn.Linear(node_feature_dimension, action_space_dimension)
        self.readout_value = nn.Linear(node_feature_dimension, 1)

    def forward(self, node_features: torch.Tensor, edge_features: torch.Tensor, adjacency_matrix: torch.Tensor):
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
        # TODO: For training, define a proper loss function that takes both the loss from policy and value
        #  networks into account (only back-propagation from final layers would be tricky)
        node_feature_sum = updated_node_features.sum(dim=0)
        q_values = self.readout_policy(node_feature_sum)
        v_value = self.readout_value(node_feature_sum)

        return q_values, v_value


'''# Example usage
node_features = torch.tensor([[1, 0], [0, 0], [0, 1], [0, 0]], dtype=torch.float32)  # 4 nodes, each with 2 features
adjacency_matrix = torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]])  # 4x4 edges, each with a scalar feature
edge_features = adjacency_matrix


# Create an instance of the MPNN model
mpnn = MPNN(node_feature_dimension=2, action_space_dimension=4)

# Forward pass
action_probabilities = mpnn(node_features, edge_features, adjacency_matrix)

print(action_probabilities)'''
