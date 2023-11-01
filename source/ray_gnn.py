# ===== RLLib-compatible custom PPO model =====
# Imports
import torch
import torch.nn as nn
import gymnasium as gym
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.utils.annotations import override
from GNN import MPNN


# Custom ray-compatible GNN-based model
class MPNNModel(TorchModelV2, nn.Module):
    def __init(self, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, num_outputs: int, model_config: dict, name: str):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # General specs
        self.n_agents = 1
        self.node_features_dim = model_config['node_features_dim']
        self.edge_features_dim = model_config['edge_features_dim']
        self.action_space_dim = model_config['action_space_dim']
        self._cur_values = None
        # TODO: Optimize above code & add more!

        # Policy & value networks
        self.policy_nn = MPNN(
            node_feature_dimension=self.node_features_dim,
            action_space_dimension=self.action_space_dim,
            nn_type="policy"
        )
        self.value_nn = MPNN(
            node_feature_dimension=self.node_features_dim,
            action_space_dimension=self.action_space_dim,
            nn_type="value"
        )

    @override(ModelV2)
    def forward(self, input_dict: dict, state: list[torch.Tensor], seq_lens: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        # Input observation (must be a dictionary containing node & edge features, and adj. matrix)
        obs = input_dict["obs"]

        # Decompose the observation into its different matrices
        node_features = obs["node_features"]
        edge_features = obs["edge_features"]
        adjacency_matrix = obs["adjacency_matrix"]

        # Calculate new policy & values
        action_probabilities = self.policy_nn(node_features, edge_features, adjacency_matrix)
        self._cur_values = self.value_nn(node_features, edge_features, adjacency_matrix)

        return action_probabilities, state

    @override(ModelV2)
    def value_function(self):
        assert self._cur_values is not None, "ERROR: Must call `forward()` first"
        return self._cur_values
