import torch
import torch.nn as nn
import gym
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.typing import ModelConfigDict, TensorType


class BaselineMLP(nn.Module):
    def __init__(self, node_feature_dimension: int, action_space_dimension: int, nn_type: str) -> None:
        super().__init__()
        self.nn_type = nn_type
        self.node_feature_dimension = node_feature_dimension
        self.action_space_dimension = action_space_dimension

        self.readout = nn.Sequential(
            nn.Linear(self.node_feature_dimension, self.node_feature_dimension),
            nn.SELU(),
            nn.Linear(self.node_feature_dimension, self.node_feature_dimension),
            nn.SELU()
        )
        self.readout_policy = nn.Linear(self.node_feature_dimension, self.action_space_dimension)
        self.readout_value = nn.Linear(self.node_feature_dimension, 1)

    def forward(self, node_features: torch.Tensor, edge_features: torch.Tensor, adjacency_matrix: torch.Tensor):
        node_features = node_features.clone()

        node_feature_sum = node_features.sum(dim=1)

        if (self.nn_type == "policy"):
            policy_logits = self.readout_policy(self.readout(node_feature_sum))
            return policy_logits
        elif (self.nn_type == "value"):
            v_value = self.readout_value(self.readout(node_feature_sum))
            return v_value


class BaselineMLPModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, num_outputs: int, model_config: dict, name: str):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.node_features_dim = model_config["custom_model_config"]['node_features_dim']
        self.action_space_dim = model_config["custom_model_config"]['action_space_dim']
        self._cur_values = None

        # Policy & value networks
        self.policy_nn = BaselineMLP(
            node_feature_dimension=self.node_features_dim,
            action_space_dimension=self.action_space_dim,
            nn_type="policy"
        )
        self.value_nn = BaselineMLP(
            node_feature_dimension=self.node_features_dim,
            action_space_dimension=self.action_space_dim,
            nn_type="value"
        )

        #self.fcnet = TorchFC(
        #    obs_space, action_space, num_outputs, model_config, name="fcnet"
        #)

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
        self._cur_values = self.value_nn(node_features, edge_features, adjacency_matrix)
        #policy_logits = self.fcnet.forward(input_dict, state, seq_lens)

        return policy_logits, state

    def value_function(self):
        #assert self._cur_values is not None, "ERROR: Must call `forward()` first"
        #return self.fcnet.value_function()
        return torch.reshape(self._cur_values, [-1])
