import torch
from GNN import MPNN

node_features = torch.tensor([[1, 0], [0, 0], [0, 1], [0, 0]], dtype=torch.float32)  # 4 nodes, each with 2 features
adjacency_matrix = torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]])  # 4x4 edges, each with a scalar feature
edge_features = adjacency_matrix

# Create an instance of the MPNN model as policy NN
policyNN = MPNN(node_feature_dimension=2, action_space_dimension=4, nn_type="policy")

# Forward pass
action_probabilities = policyNN(node_features, edge_features, adjacency_matrix)

print(action_probabilities)

# Create an instance of the MPNN model as policy NN
valueNN = MPNN(node_feature_dimension=2, action_space_dimension=4, nn_type="value")

# Forward pass
v_value = valueNN(node_features, edge_features, adjacency_matrix)

print(v_value)
