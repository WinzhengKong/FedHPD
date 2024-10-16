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

        self.probs = []
        self.rewards = []


class Server:
    """Server class for federated learning."""

    def __init__(self):
        self.global_means = None
        self.global_stddevs = None
        self.local_means = []
        self.local_stddevs = []

    def aggregate(self, local_means: list, local_stddevs: list):
        """Averaging the local means and stddevs from different agents."""
        self.global_means = torch.mean(torch.stack(local_means), dim=0)
        self.global_stddevs = torch.mean(torch.stack(local_stddevs), dim=0)
        self.local_means = local_means
        self.local_stddevs = local_stddevs

    def broadcast(self):
        """Broadcast global averages and local outputs back to the agents."""
        return self.global_means, self.global_stddevs, self.local_means, self.local_stddevs


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


for i, config in enumerate(agent_configs):
    print(f"Agent Config {i + 1}:")
    print(f"  Hidden Layers : {config['hidden_layers']}")
    print(f"  Activations   : {config['activations']}")
    print(f"  Learning Rate : {config['learning_rate']}\n")

# 将各个智能体的训练步骤并行化处理
for seed in [20, 25, 30, 35, 40]:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    agents = [REINFORCE(obs_space_dims, action_space_dims, **config) for config in agent_configs]
    reward_histories = [[] for _ in range(num_agents)]
    server = Server()  # Initialize the server for federated learning

    # Store rewards in a DataFrame for this seed
    seed_rewards = pd.DataFrame(columns=["Episode"] + [f"Agent {i + 1}" for i in range(num_agents)])

    for episode in range(total_num_episodes):
        obs = [wrapped_env.reset(seed=seed) for wrapped_env in wrapped_envs]
        done = [False] * num_agents
        total_rewards = [0] * num_agents  # Initialize the total reward for each agent

        while not all(done):
            for i, agent in enumerate(agents):
                if not done[i]:
                    action = agent.sample_action(obs[i])
                    next_obs, reward, done[i], info = wrapped_envs[i].step(action)
                    agent.rewards.append(reward)
                    total_rewards[i] += reward
                    obs[i] = next_obs

        # Append the rewards for the current episode to the DataFrame
        seed_rewards = seed_rewards.append({
            "Episode": episode + 1,
            **{f"Agent {i + 1}": total_rewards[i] for i in range(num_agents)}
        }, ignore_index=True)

        for i in range(num_agents):
            reward_histories[i].append(total_rewards[i])
            if len(reward_histories[i]) > 100:
                reward_histories[i].pop(0)

        # Update the agents
        for agent in agents:
            agent.update()

        # Perform federated learning every 10 episodes
        if (episode + 1) % 5 == 0:
            sampled_states = pd.read_csv('sampled_state_transitions_20240815_213813.csv')
            sampled_states['State'] = sampled_states['State'].apply(ast.literal_eval)
            sampled_states_tensor = torch.tensor(sampled_states['State'].values.tolist(), dtype=torch.float32)

            half_size = len(sampled_states_tensor) // 2
            sampled_states_tensor_part1 = sampled_states_tensor[:half_size]
            sampled_states_tensor_part2 = sampled_states_tensor[half_size:]

            distill_loop = 5
            for distillation_round in range(distill_loop):
                print(f"\nKnowledge Distillation Round {distillation_round + 1}:")

                # 进行两次更新，分别使用数据集的两部分
                for part_idx, sampled_states_part in enumerate(
                        [sampled_states_tensor_part1, sampled_states_tensor_part2], start=1):
                    local_means, local_stddevs = [], []
                    for agent in agents:
                        means, stddevs = agent.net(sampled_states_part)
                        local_means.append(means)
                        local_stddevs.append(stddevs)

                    server.aggregate(local_means, local_stddevs)
                    global_means, global_stddevs, all_means, all_stddevs = server.broadcast()

                    for idx, (agent, local_mean, local_stddev) in enumerate(zip(agents, local_means, local_stddevs)):
                        output_means, output_stddevs = agent.net(sampled_states_part)
                        output_distribution = Normal(output_means, output_stddevs)

                        global_distribution = Normal(global_means.detach(), global_stddevs.detach())
                        kl_loss_global = kl_loss_fn(output_distribution, global_distribution).mean()

                        kl_loss = kl_loss_global

                        # Perform backpropagation and optimization
                        agent.optimizer.zero_grad()
                        kl_loss.backward(retain_graph=True)  # Ensure kl_loss_inter requires gradient
                        agent.optimizer.step()

                        print(f"Updated Agent {idx + 1} on Part {part_idx}, KD_Loss: {kl_loss}")
                print()

        # 打印当前 episode 的奖励和过去 100 个 episode 的平均奖励
        if (episode + 1) % 5 == 0 or episode == total_num_episodes - 1:
            print(f"Episode: {episode + 1}")
            for i, total_reward in enumerate(total_rewards):
                avg_reward = int(np.mean(reward_histories[i][-100:]))  # 计算每个智能体最近100个episode的平均奖励
                print(f"  Agent {i + 1}: Reward: {total_reward}, Average Reward (last 100 episodes): {avg_reward}")
            print()  # 添加额外的行以便于阅读

    # 保存每个随机种子下的奖励为 CSV 文件
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    seed_rewards.to_csv(f'rewards_Fed_SL_seed_{seed}_{timestamp}.csv', index=False)

    print(f"Saved rewards for seed {seed} to 'rewards_seed_{seed}.csv'.")


