import torch
from GNN import MPNN, new_MPNN

# Define the batch size
num_batches = 4

# Original node_features and adjacency_matrix for one batch
node_features_single = torch.tensor([[[1, 0], [0, 0], [0, 0], [0, 1]]], dtype=torch.float32)  # 4 nodes, each with 2 features
adjacency_matrix_single = torch.tensor([[[0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0]]], dtype=torch.float32)  # 4x4 edges

# Repeat these for num_batches times
node_features = node_features_single.repeat(num_batches, 1, 1)
adjacency_matrix = adjacency_matrix_single.repeat(num_batches, 1, 1)

# Check shapes
print("node_features shape:", node_features.shape)  # Should be [128, 4, 2]
print("adjacency_matrix shape:", adjacency_matrix.shape)  # Should be [128, 4, 4]
edge_features = adjacency_matrix

# Create an instance of the MPNN model as policy NN
policyNN = MPNN(node_feature_dimension=2, action_space_dimension=4, nn_type="policy", diameter=5)
new_policyNN = new_MPNN(node_feature_dimension=2, action_space_dimension=4, nn_type="policy", diameter=5)
policyNN.eval()
new_policyNN.eval()
new_policyNN.load_state_dict(policyNN.state_dict())


# Forward pass
with torch.no_grad():
    policy_logits = policyNN(node_features, edge_features, adjacency_matrix)
    new_policy_logits = new_policyNN(node_features, edge_features, adjacency_matrix)

tolerance = 1e-5

# Check if the outputs are almost equal
assert torch.allclose(policy_logits, new_policy_logits, atol=tolerance), "The outputs are not matching!"
print(policy_logits)
print(new_policy_logits)

'''# Create an instance of the MPNN model as policy NN
valueNN = MPNN(node_feature_dimension=2, action_space_dimension=4, nn_type="value")

# Forward pass
v_value = valueNN(node_features, edge_features, adjacency_matrix)

print(v_value)'''
