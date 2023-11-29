# ===== RLLib-compatible custom PPO model =====
# Imports
import torch
import torch.nn as nn
import gym
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.annotations import override
from GNN import MPNN, new_MPNN
from ray.rllib.models import ModelCatalog
import logging


# Custom ray-compatible GNN-based model
class MPNNModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, num_outputs: int, model_config: dict, name: str):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # General specs
        self.node_features_dim = model_config["custom_model_config"]['node_features_dim']
        self.action_space_dim = model_config["custom_model_config"]['action_space_dim']
        diameter = model_config["custom_model_config"]['diameter']
        self._cur_values = None

        # Policy & value networks
        self.policy_nn = new_MPNN(
            node_feature_dimension=self.node_features_dim,
            action_space_dimension=self.action_space_dim,
            nn_type="policy",
            diameter=diameter
        )
        self.value_nn = new_MPNN(
            node_feature_dimension=self.node_features_dim,
            action_space_dimension=self.action_space_dim,
            nn_type="value",
            diameter=diameter
        )

        self.fcnet = TorchFC(
            obs_space, action_space, num_outputs, model_config, name="fcnet"
        )


    @override(ModelV2)
    def forward(self, input_dict: dict, state: list[torch.Tensor], seq_lens: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        # Input observation (must be a dictionary containing node & edge features, and adj. matrix)
        obs = input_dict["obs"]
        #logging.critical("YOYOYOYOYOYOYO\n", obs["node_feature_mat"])
        # Decompose the observation into its different matrices
        node_features = obs["node_feature_mat"]
        edge_features = obs["edge_feature_mat"]
        adjacency_matrix = obs["adj_max"]

        # Calculate new policy & values
        policy_logits = self.policy_nn(node_features, edge_features, adjacency_matrix)
        #_ = self.fcnet.forward(input_dict, state, seq_lens)
        self._cur_values = self.value_nn(node_features, edge_features, adjacency_matrix)

        return policy_logits, state

    @override(ModelV2)
    def value_function(self):
        assert self._cur_values is not None, "ERROR: Must call `forward()` first"
        #return self.fcnet.value_function()
        return torch.reshape(self._cur_values, [-1])
