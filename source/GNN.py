import torch
import torch.nn as nn
import logging

logger = logging.getLogger('ray_task_logger')
logger.setLevel(logging.INFO)
fh = logging.FileHandler('ray_task.log')
fh.setLevel(logging.INFO)
logger.addHandler(fh)


class MPNN(nn.Module):
    def __init__(self, node_feature_dimension: int, action_space_dimension: int, nn_type: str, diameter: int):
        super().__init__()

        # Message-passing network
        self.nn_type = nn_type
        self.diameter = diameter
        self.node_feature_dimension = node_feature_dimension
        self.action_space_dimension = action_space_dimension

        self.message = nn.Sequential(
            nn.Linear(self.node_feature_dimension * 2 + 1, self.node_feature_dimension),
            nn.SELU()
        )

        # Update network
        self.update = nn.GRUCell(self.node_feature_dimension, self.node_feature_dimension)

        # Readout network
        self.readout = nn.Sequential(
            nn.Linear(self.node_feature_dimension, self.node_feature_dimension),
            nn.SELU(),
            nn.Linear(self.node_feature_dimension, self.node_feature_dimension),
            nn.SELU()
        )
        self.readout_policy = nn.Linear(self.node_feature_dimension, self.action_space_dimension)
        self.readout_value = nn.Linear(self.node_feature_dimension, 1)

    def forward(self, node_features: torch.Tensor, edge_features: torch.Tensor, adjacency_matrix: torch.Tensor):
        #logger.info("node_features\n", node_features)
        #logger.info("edge_features\n", edge_features)
        #logger.info("adjacency_matrix\n", adjacency_matrix)
        node_features = node_features.clone()

        self.num_batches = adjacency_matrix.shape[0]
        self.num_nodes_per_batch = adjacency_matrix.shape[1]
        batch_indices, source_indices, target_indices = adjacency_matrix.nonzero(as_tuple=True)
        # NOTE: assume adjacency matrices are the same across all batches
        single_batch_src_idx, single_batch_tgt_idx = adjacency_matrix[0].nonzero(as_tuple=True)
        num_msg_per_batch = len(single_batch_src_idx)
        '''print("batch_indices: ", batch_indices)
        print("source_indices: ", source_indices)
        print("target_indices: ", target_indices)
        print("single_batch_src_idx: ", single_batch_src_idx)
        print("single_batch_tgt_idx: ", single_batch_tgt_idx)
        print("num_msg_per_batch: ", num_msg_per_batch)'''

        is_empty = source_indices.numel() == 0
        if (is_empty):
            if (self.nn_type == "policy"):
                policy_logits = torch.zeros(self.num_batches, self.action_space_dimension)
                return policy_logits
            elif (self.nn_type == "value"):
                v_value = torch.zeros(self.num_batches)
                return v_value

        for _ in range(self.diameter):
            # Identify the source and target nodes for each edge
            # This is the indices for individual batches, since all batches have the same adj mat,
            # the computation is done on batch #0.

            # Gather the source and target node features for all edges
            #print("old")
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
            #print("messages: ", messages)
            # Separate the batches out
            batched_messages = messages.reshape(self.num_batches, num_msg_per_batch, self.node_feature_dimension)
            #logger.critical("batched_messages:\n", batched_messages)
            #print("messages: ", messages)
            #print("batched messages: ", batched_messages)

            # Aggregate messages by summing them for each target node
            aggregated_messages = torch.zeros(self.num_batches, self.num_nodes_per_batch, self.node_feature_dimension, device=node_features.device)

            # Aggregate messages for each node in each batch
            for batch in range(self.num_batches):
                for node in range(self.num_nodes_per_batch):
                    # Find all messages targeting this node (n) in batch (b)
                    target_mask = single_batch_tgt_idx == node
                    aggregated_messages[batch, node] = batched_messages[batch][target_mask].sum(dim=0)
            #logger.critical("aggregated message:\n", aggregated_messages)

            #print("aggregated_messages", aggregated_messages)
            # Update node features with a GRU, loop through each batch
            updated_node_features = torch.zeros_like(node_features)
            for batch in range(node_features.shape[0]):
                updated_node_features[batch] = self.update(aggregated_messages[batch], node_features[batch])
            #logger.critical("updated node features:\n", node_features)
            node_features = updated_node_features
            
            #print(node_features)

        # Readout
        node_feature_sum = node_features.sum(dim=1)
        #print(node_feature_sum)
        if (self.nn_type == "policy"):
            policy_logits = self.readout_policy(self.readout(node_feature_sum))
            return policy_logits
        elif (self.nn_type == "value"):
            v_value = self.readout_value(self.readout(node_feature_sum))
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
        logger.critical("batched_msg: ", batched_messages.shape)
        logger.critical('batch idx: ', batch_index.shape)
        logger.critical("sorted idx: ", sorted_indices.shape)
        sorted_messages = batched_messages[batch_index, sorted_indices]

        # Sum up the messages for each node
        aggregated_messages = torch.sum(sorted_messages, dim=2)

        return aggregated_messages


class new_MPNN(nn.Module):
    def __init__(self, node_feature_dimension: int, action_space_dimension: int, nn_type: str, diameter: int):
        super().__init__()

        # Message-passing network
        self.nn_type = nn_type
        self.diameter = diameter
        self.node_feature_dimension = node_feature_dimension
        self.action_space_dimension = action_space_dimension

        self.message = nn.Sequential(
            nn.Linear(self.node_feature_dimension * 2 + 1, self.node_feature_dimension),
            nn.SELU()
        )

        # Update network
        self.update = nn.GRUCell(self.node_feature_dimension, self.node_feature_dimension)

        # Readout network
        self.readout = nn.Sequential(
            nn.Linear(self.node_feature_dimension, self.node_feature_dimension),
            nn.SELU(),
            nn.Linear(self.node_feature_dimension, self.node_feature_dimension),
            nn.SELU()
        )
        self.readout_policy = nn.Linear(self.node_feature_dimension, self.action_space_dimension)
        self.readout_value = nn.Linear(self.node_feature_dimension, 1)

    def forward(self, node_features: torch.Tensor, edge_features: torch.Tensor, adjacency_matrix: torch.Tensor):
        #logger.info("node_features\n", node_features)
        #logger.info("edge_features\n", edge_features)
        #logger.info("adjacency_matrix\n", adjacency_matrix)
        node_features = node_features.clone()

        self.num_batches = adjacency_matrix.shape[0]
        self.num_nodes_per_batch = adjacency_matrix.shape[1]
        batch_indices, source_indices, target_indices = adjacency_matrix.nonzero(as_tuple=True)
        # NOTE: assume adjacency matrices are the same across all batches
        single_batch_src_idx, single_batch_tgt_idx = adjacency_matrix[0].nonzero(as_tuple=True)
        num_msg_per_batch = len(single_batch_src_idx)
        '''print("batch_indices: ", batch_indices)
        print("source_indices: ", source_indices)
        print("target_indices: ", target_indices)
        print("single_batch_src_idx: ", single_batch_src_idx)
        print("single_batch_tgt_idx: ", single_batch_tgt_idx)
        print("num_msg_per_batch: ", num_msg_per_batch)'''

        is_empty = source_indices.numel() == 0
        if (is_empty):
            if (self.nn_type == "policy"):
                policy_logits = torch.zeros(self.num_batches, self.action_space_dimension)
                return policy_logits
            elif (self.nn_type == "value"):
                v_value = torch.zeros(self.num_batches)
                return v_value

        for _ in range(self.diameter):
            # Identify the source and target nodes for each edge
            # This is the indices for individual batches, since all batches have the same adj mat,
            # the computation is done on batch #0.

            # Gather the source and target node features for all edges
            #print("new")
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
            #print("messages: ", messages)
            # Separate the batches out
            batched_messages = messages.reshape(self.num_batches, num_msg_per_batch, self.node_feature_dimension)
            #logger.critical("batched_messages:\n", batched_messages)
            #print("messages: ", messages)
            #print("batched messages: ", batched_messages)

            # Vectorized message aggregation
            aggregated_messages = torch.zeros(self.num_batches, self.num_nodes_per_batch, self.node_feature_dimension, device=node_features.device)

            # Modify target_indices dimensions to match batched_messages for scatter_add_
            reshaped_target_indices = target_indices.view(self.num_batches, -1, 1).expand(-1, -1, self.node_feature_dimension)

            # Now use scatter_add_
            aggregated_messages.scatter_add_(1, reshaped_target_indices, batched_messages)

            # Vectorized node update
            updated_node_features = self.update(aggregated_messages.view(-1, self.node_feature_dimension), node_features.view(-1, self.node_feature_dimension))
            node_features = updated_node_features.view(self.num_batches, self.num_nodes_per_batch, self.node_feature_dimension)
            
            #print(node_features)

        # Readout
        node_feature_sum = node_features.sum(dim=1)
        #print(node_feature_sum)
        if (self.nn_type == "policy"):
            policy_logits = self.readout_policy(self.readout(node_feature_sum))
            return policy_logits
        elif (self.nn_type == "value"):
            v_value = self.readout_value(self.readout(node_feature_sum))
            return v_value
