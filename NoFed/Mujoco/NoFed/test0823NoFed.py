import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.distributions.kl import kl_divergence as kl_loss_fn
from torch.distributions.normal import Normal
import gym
import ast
from typing import Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

plt.rcParams["figure.figsize"] = (10, 5)


class Policy_Network(nn.Module):
    """Parametrized Policy Network with customizable hidden layers and activation functions."""

    def __init__(self, obs_space_dims: int, action_space_dims: int,
                 hidden_layers: list, activations: list):
        super().__init__()
        layers = []
        input_dim = obs_space_dims

        # Create hidden layers with specified activations
        for hidden_dim, activation in zip(hidden_layers, activations):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(self.get_activation(activation))
            input_dim = hidden_dim

        self.shared_net = nn.Sequential(*layers)

        self.policy_mean_net = nn.Sequential(
            nn.Linear(input_dim, action_space_dims)
        )

        self.policy_stddev_net = nn.Sequential(
            nn.Linear(input_dim, action_space_dims)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shared_features = self.shared_net(x.float())
        action_means = self.policy_mean_net(shared_features)
        action_stddevs = torch.log(
            1 + torch.exp(self.policy_stddev_net(shared_features))
        )
        return action_means, action_stddevs

    @staticmethod
    def get_activation(name: str):
        if name == "relu":
            return nn.ReLU()
        elif name == "tanh":
            return nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation function: {name}")


class REINFORCE:
    """REINFORCE algorithm with customizable learning rates."""

    def __init__(self, obs_space_dims: int, action_space_dims: int,
                 hidden_layers: list, activations: list, learning_rate: float):
        self.learning_rate = learning_rate
        self.gamma = 0.99
        self.eps = 1e-6

        self.probs = []
        self.rewards = []

        self.net = Policy_Network(obs_space_dims, action_space_dims, hidden_layers, activations)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.99)

    def sample_action(self, state: np.ndarray) -> float:
        state = torch.tensor(np.array([state]))
        action_means, action_stddevs = self.net(state)

        distrib = Normal(action_means[0] + self.eps, action_stddevs[0] + self.eps)
        action = distrib.sample()
        prob = distrib.log_prob(action)

        action = action.numpy()

        self.probs.append(prob)

        return action

    def update(self):
        running_g = 0
        gs = []

        for R in self.rewards[::-1]:
            running_g = R + self.gamma * running_g
            gs.insert(0, running_g)

        deltas = torch.tensor(gs)

        loss = 0
        for log_prob, delta in zip(self.probs, deltas):
            loss += log_prob.mean() * delta * (-1)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        self.probs = []
        self.rewards = []


# Create and wrap the environments for multiple agents
num_agents = 10  # Number of agents
envs = [gym.make("InvertedPendulum-v2") for _ in range(num_agents)]
wrapped_envs = [gym.wrappers.RecordEpisodeStatistics(env, deque_size=50) for env in envs]

total_num_episodes = int(6e3)
obs_space_dims = envs[0].observation_space.shape[0]
action_space_dims = envs[0].action_space.shape[0]

# Define different configurations for agents
agent_configs = [
    {"hidden_layers": [16, 32], "activations": ["tanh", "tanh"], "learning_rate": 1e-4},
    {"hidden_layers": [32, 32], "activations": ["relu", "relu"], "learning_rate": 1e-4},
    {"hidden_layers": [64, 128], "activations": ["tanh", "relu"], "learning_rate": 6e-5},
    {"hidden_layers": [128, 256], "activations": ["relu", "relu"], "learning_rate": 1e-5},
    {"hidden_layers": [32, 64], "activations": ["relu", "tanh"], "learning_rate": 1e-4},
    {"hidden_layers": [64, 64], "activations": ["tanh", "tanh"], "learning_rate": 8e-5},
    {"hidden_layers": [128, 128], "activations": ["relu", "relu"], "learning_rate": 4e-5},
    {"hidden_layers": [64, 32], "activations": ["tanh", "relu"], "learning_rate": 7e-5},
    {"hidden_layers": [256, 128], "activations": ["tanh", "tanh"], "learning_rate": 2e-5},
    {"hidden_layers": [32, 128], "activations": ["relu", "relu"], "learning_rate": 5e-5},
]


# Store rewards for each agent across episodes
rewards_over_seeds = {i: [] for i in range(num_agents)}

# 将各个智能体的训练步骤顺序化处理
for i, config in enumerate(agent_configs):
    print(f"Agent{i} || Agentconfig: {config}")
    for seed in [20, 25, 30, 35, 40]:
        print(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        agent = REINFORCE(obs_space_dims, action_space_dims, **config)
        reward_history = []

        # Store rewards in a DataFrame for this seed
        seed_rewards = pd.DataFrame(columns=["Episode", f"Agent {i + 1}"])

        for episode in range(total_num_episodes):
            obs = wrapped_envs[i].reset(seed=seed)
            done = False
            total_reward = 0  # Initialize the total reward for the agent

            while not done:
                action = agent.sample_action(obs)
                obs, reward, done, info = wrapped_envs[i].step(action)

                if not done:  # Only update if the agent is not done
                    agent.rewards.append(reward)
                    total_reward += reward  # Accumulate the reward for the current episode

            # Append the rewards for the current episode to the DataFrame
            seed_rewards = seed_rewards.append({
                "Episode": episode + 1,
                f"Agent {i + 1}": total_reward
            }, ignore_index=True)

            reward_history.append(total_reward)
            if len(reward_history) > 100:  # 保持列表长度最多为100
                reward_history.pop(0)

            # 更新智能体
            agent.update()

            # 打印当前 episode 的奖励和过去 100 个 episode 的平均奖励
            if (episode + 1) % 5 == 0 or episode == total_num_episodes - 1:
                avg_reward = int(np.mean(reward_history[-100:]))  # 计算最近100个episode的平均奖励
                print(f"Episode: {episode + 1}, Agent {i + 1}: Reward: {total_reward}, Average Reward (last 100 episodes): {avg_reward}")

        # 保存每个随机<font color="red">`**`</font>下的奖励为 CSV 文件
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        seed_rewards.to_csv(f'rewards_Agent_{i + 1}_seed_{seed}_{timestamp}.csv', index=False)

        print(f"Rewards for Agent {i + 1}, seed {seed} to 'rewards_Agent_{i + 1}_seed_{seed}.csv'.")

