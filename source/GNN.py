import torch
import torch.nn as nn
import logging


class MPNN(nn.Module):
    def __init__(self, node_feature_dimension: int, action_space_dimension: int, nn_type: str):
        super(MPNN, self).__init__()

        # Message-passing network
        # TODO: how to make sure the architecture can generalize and does not depend on the input graph's topology?
        self.nn_type = nn_type
        self.node_feature_dimension = node_feature_dimension

        self.message = nn.Sequential(
            nn.Linear(self.node_feature_dimension * 2 + 1, self.node_feature_dimension),
            nn.ReLU()
        )

        # Update network
        self.update = nn.GRUCell(self.node_feature_dimension, self.node_feature_dimension)

        # Readout network
        self.readout_policy = nn.Linear(self.node_feature_dimension, action_space_dimension)
        self.readout_value = nn.Linear(self.node_feature_dimension, 1)

    def forward(self, node_features: torch.Tensor, edge_features: torch.Tensor, adjacency_matrix: torch.Tensor):
        T = 3
        self.num_batches = adjacency_matrix.shape[0]
        self.num_nodes_per_batch = adjacency_matrix.shape[1]
        batch_indices, source_indices, target_indices = adjacency_matrix.nonzero(as_tuple=True)
        for _ in range(T):
            # Identify the source and target nodes for each edge
            # This is the indices for individual batches, since all batches have the same adj mat,
            # the computation is done on batch #0.

            #print("batch_indices: ", batch_indices)
            #print("source_indices: ", source_indices)
            #print("target_indices: ", target_indices)
            # Gather the source and target node features for all edges
            source_features = node_features[batch_indices, source_indices]
            #print("source_features: ", source_features)
            target_features = node_features[batch_indices, target_indices]
            #print("target_features: ", target_features)
            # Gather the edge features
            edge_feats = edge_features[batch_indices, source_indices, target_indices].unsqueeze(-1)
            #print("edge_feats: ", edge_feats)
            # Concatenate the source node features, target node features, and edge features
            combined_features = torch.cat([source_features, target_features, edge_feats], dim=1)
            #print("combined_features: ", combined_features)
            # Compute messages in parallel for all edges
            messages = self.message(combined_features)
            # Separate the batches out
            batched_messages = messages.reshape(self.num_batches, self.num_nodes_per_batch*2, self.node_feature_dimension)
            logging.critical("batched_messages:\n", batched_messages)
            #print("messages: ", messages)
            #print("batched messages: ", batched_messages)

            # Aggregate messages by summing them for each target node
            aggregated_messages = self.aggregation(target_indices, batched_messages)
            logging.critical("aggregated message:\n", aggregated_messages)

            # Update node features with a GRU, loop through each batch
            for batch in range(node_features.shape[0]):
                node_features[batch] = self.update(aggregated_messages[batch], node_features[batch])
            logging.critical("updated node features:\n", node_features)

        # Readout
        node_feature_sum = node_features.sum(dim=1)
        print(node_feature_sum)
        if (self.nn_type == "policy"):
            policy_logits = self.readout_policy(node_feature_sum)
            return policy_logits
        elif (self.nn_type == "value"):
            v_value = self.readout_value(node_feature_sum)
            return v_value

    def aggregation(self, target_indices, batched_messages):
        # Reshape the target indices, separate batches
        reshaped_indices = target_indices.reshape(self.num_batches, self.num_nodes_per_batch*2)
        # All nodes receive messages from neighbors
        node_number = torch.arange(self.num_nodes_per_batch).unsqueeze(1)
        batch_index = torch.arange(self.num_batches)

        # NOTE: Assumed all adjacency matrices in the batch are identical
        # Just use the first batch [0] for computing
        # Look into the reshaped target indices, extract the positions of messages to node 0, 1, ... N
        comparison_mat = reshaped_indices[0] == node_number
        # Get the non-zero, i.e., pinpointed indices for node 0, 1, ... N
        pinpointed_indices = comparison_mat.nonzero(as_tuple=False)
        # Sort the indices, such that node 0 comes first, then node 1, and so on
        # Then repeat the indices for 1 batch to all batches
        sorted_indices = pinpointed_indices[:, 1].view(-1, 2)
        sorted_indices = sorted_indices.unsqueeze(0).repeat(self.num_batches, 1, 1)
        # For each batch, put messages for node 0 first, then 1, ...N
        sorted_messages = batched_messages[batch_index, sorted_indices]

        # Sum up the messages for each node
        aggregated_messages = torch.sum(sorted_messages, dim=2)

        return aggregated_messages
