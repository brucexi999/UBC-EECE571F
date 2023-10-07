import torch
import torch.nn as nn
import torch.nn.functional as F


class GNN(torch.nn.Module):
    """Q-network implemented by message-passing GNN (learn rich information
    about the graph) + FC (predict Q values for all actions)"""
    def __init__(self, state_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        hidden_dim = state_dim

        # Message-passing layer
        self.message_passing = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SELU()
        )

        # Update node feature using RNN
        self.update = torch.nn.GRUCell(input_size=hidden_dim)

        # Fully connected layer to predict Q-values from the state feature vector
        self.readout = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        T = 3
        node_feature = data
        # Execute T times
        for _ in range(T):
            messages = self.message_passing(node_feature)
        # TODO: Figure out how to concat two nodes' features and do message passing


        # Use the resulting state feature vector to predict Q-values for each action
        q_values = self.fc1(x)

        return q_values