import torch
import numpy as np
from environment import SimpleEnv
from ray_gnn import MPNNModel
from unittest import TestCase
from gymnasium.spaces import Discrete, Dict, Box
from ray.rllib.models.torch.torch_action_dist import TorchCategorical


class TestRayGNN(TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.num_node = 4
        self.node_feature_dimension = 2
        self.edge_features_dimension = 1
        self.max_capacity = 2
        action_space = Discrete(4)
        observation_space = Dict({
            'node_feature_mat': Box(low=0, high=1, shape=(self.num_node, self.node_feature_dimension), dtype=np.float32),
            'edge_feature_mat': Box(low=0, high=self.max_capacity, shape=(self.num_node, self.num_node), dtype=np.float32),
            'adj_max': Box(low=0, high=1, shape=(self.num_node, self.num_node), dtype=np.uint8)
        })
        num_outputs = 4
        model_config = {
            "node_features_dim": self.node_feature_dimension,
            "edge_features_dim": self.edge_features_dimension,
            "action_space_dim": 4
        }
        name = "test_model"

        node_feature_mat = torch.tensor([[1, 0], [0, 0], [0, 0], [0, 1]], dtype=torch.float32)  # 4 nodes, each with 2 features
        adj_mat = torch.tensor([[0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0]])  # 4x4 edges, each with a scalar feature
        edge_feature_mat = torch.tensor([[0, 2, 2, 0], [2, 0, 0, 2], [2, 0, 0, 2], [0, 2, 2, 0]])
        observation = {'node_feature_mat': node_feature_mat,
                       'edge_feature_mat': edge_feature_mat,
                       'adj_max': adj_mat}
        self.input_dict = {"obs": observation}

        self.model = MPNNModel(observation_space, action_space, num_outputs, model_config, name)

    def test_initialization(self):
        assert self.model.parameters() is not None, "Model parameters should not be None"
        assert self.model.node_features_dim == self.node_feature_dimension
        assert self.model.action_space_dim == 4

    def test_forward(self):
        policy_logits, _ = self.model.forward(self.input_dict, [], torch.tensor([]))

        assert policy_logits.shape == torch.Size([4]), "Policy logits shape {}".format(policy_logits.shape)

    def test_value(self):
        policy_logits, _ = self.model.forward(self.input_dict, [], torch.tensor([]))
        value = self.model.value_function()
        assert value.shape == torch.Size([1])

    def test_action_distribution(self):
        policy_logits, _ = self.model.forward(self.input_dict, [], torch.tensor([]))
        action_dis = TorchCategorical(policy_logits, self.model)
        sampled_action = action_dis.sample()
        prob_dis = action_dis.logp(sampled_action)
        assert sampled_action is not None
        assert prob_dis.shape == torch.Size([]), "Action probability distribution shape {}".format(prob_dis.shape)

    def test_integration(self):
        length, width, capacity = 2, 2, 2
        edge_capacity = np.full((length, width, 4), capacity)
        macros = []
        env = SimpleEnv(length, width, macros, edge_capacity)
        observation = env.reset()
        input_dict = {"obs": observation}
        policy_logits, _ = self.model.forward(input_dict, [], torch.tensor([]))
        action_dis = TorchCategorical(policy_logits, self.model)
        sampled_action = action_dis.sample().numpy()
        print(sampled_action)
        new_obs, reward, done, info = env.step(sampled_action)
        print(new_obs)
        assert new_obs is not None
