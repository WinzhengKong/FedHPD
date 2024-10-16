import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.distributions.kl import kl_divergence
from torch.distributions.normal import Normal
import gym
import ast
from typing import Tuple
from datetime import datetime

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
        elif name == "sigmoid":
            return nn.Sigmoid()
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

    def update_with_kl(self, global_state, local_means, local_stddevs, global_means, global_stddevs):
        """通过KL散度更新网络。"""

        output_means, output_stddevs = self.net(global_state)
        output_dis = Normal(output_means, output_stddevs)

        kl_divergences = []
        for other_means, other_stddevs in zip(local_means, local_stddevs):
            if torch.equal(other_means, output_means):
                continue
            other_dis = Normal(other_means, other_stddevs)
            # 使用KL散度计算
            kl_divergence = torch.distributions.kl_divergence(output_dis, other_dis).mean().detach()
            kl_divergences.append(kl_divergence)

        kl_loss_inter = (sum(kl_divergences) / (len(local_means) - 1))

        global_dis = Normal(global_means, global_stddevs)
        kl_loss_global = torch.distributions.kl_divergence(output_dis, global_dis).mean()

        kl_loss = kl_loss_inter + kl_loss_global

        self.optimizer.zero_grad()
        kl_loss.backward()
        self.optimizer.step()

        return kl_loss.item()


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
num_agents = 4  # Number of agents
envs = [gym.make("InvertedPendulum-v2") for _ in range(num_agents)]
wrapped_envs = [gym.wrappers.RecordEpisodeStatistics(env, deque_size=50) for env in envs]

total_num_episodes = int(1e4)
obs_space_dims = envs[0].observation_space.shape[0]
action_space_dims = envs[0].action_space.shape[0]

# Define different configurations for agents
agent_configs = [
    {"hidden_layers": [16, 32], "activations": ["tanh", "tanh"], "learning_rate": 1e-4},
    {"hidden_layers": [32, 32], "activations": ["relu", "relu"], "learning_rate": 1e-4},
    {"hidden_layers": [64, 128], "activations": ["tanh", "relu"], "learning_rate": 6e-5},
    {"hidden_layers": [128, 256], "activations": ["relu", "sigmoid"], "learning_rate": 1e-5},
]

# Store rewards for each agent across episodes
rewards_over_seeds = {i: [] for i in range(num_agents)}

for seed in [20]:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    agents = [REINFORCE(obs_space_dims, action_space_dims, **config) for config in agent_configs]
    reward_histories = [[] for _ in range(num_agents)]
    server = Server()  # Initialize the server for federated learning

    # Store rewards in a DataFrame for this seed
    seed_rewards = pd.DataFrame(columns=["Episode", "Agent 1", "Agent 2", "Agent 3", "Agent 4"])

    for episode in range(total_num_episodes):
        obs = [wrapped_env.reset(seed=seed) for wrapped_env in wrapped_envs]

        done = [False] * num_agents
        total_rewards = [0] * num_agents  # Initialize the total reward for each agent

        while not all(done):
            actions = [agent.sample_action(ob) for agent, ob in zip(agents, obs)]
            obs, rewards, done, infos = zip(
                *[wrapped_env.step(action) for wrapped_env, action in zip(wrapped_envs, actions)])
            obs = list(obs)
            done = list(done)
            rewards = list(rewards)

            for i, agent in enumerate(agents):
                if not done[i]:  # Only update if the agent is not done
                    agent.rewards.append(rewards[i])
                    total_rewards[i] += rewards[i]  # Accumulate the reward for the current episode

        # Append the rewards for the current episode to the DataFrame
        seed_rewards = seed_rewards.append({
            "Episode": episode + 1,
            "Agent 1": total_rewards[0],
            "Agent 2": total_rewards[1],
            "Agent 3": total_rewards[2],
            "Agent 4": total_rewards[3]
        }, ignore_index=True)

        for i in range(num_agents):
            reward_histories[i].append(total_rewards[i])
            if len(reward_histories[i]) > 100:  # 保持列表长度最多为100
                reward_histories[i].pop(0)

        for agent in agents:
            agent.update()

        # Perform federated learning every 10 episodes
        if (episode + 1) % 5 == 0:
            # Load states from sampled_state_transitions.csv
            sampled_states = pd.read_csv('sampled_state_transitions_20240815_213813.csv')
            sampled_states['State'] = sampled_states['State'].apply(ast.literal_eval)

            # 将 DataFrame 转换为 numpy 数组，然后转换为 torch.tensor
            sampled_states_tensor = torch.tensor(sampled_states['State'].values.tolist(), dtype=torch.float32)

            # Collect local outputs from all agents
            local_means = []
            local_stddevs = []

            # Calculate local outputs for each agent
            for agent in agents:
                sampled_state_outputs = agent.net(sampled_states_tensor)
                local_means.append(sampled_state_outputs[0])
                local_stddevs.append(sampled_state_outputs[1])

            # Server aggregates the local outputs
            server.aggregate(local_means, local_stddevs)

            # Broadcast global averages and local outputs back to agents
            global_means, global_stddevs, local_means, local_stddevs = server.broadcast()

            # Each agent updates with KL divergence and global knowledge
            kl_losses = []
            for i, agent in enumerate(agents):
                kl_loss = agent.update_with_kl(sampled_states_tensor, local_means, local_stddevs, global_means, global_stddevs)
                kl_losses.append(kl_loss)

        # 打印当前 episode 的奖励和过去 100 个 episode 的平均奖励
        if (episode + 1) % 5 == 0 or episode == total_num_episodes - 1:
            print(f"Episode: {episode + 1}")
            for i, total_reward in enumerate(total_rewards):
                avg_reward = int(np.mean(reward_histories[i]))
                print(f"  Agent {i + 1}: Reward: {total_reward}, Average Reward: {avg_reward}")
            print()  # 添加额外的行以便于阅读

    # 保存每个随机种子下的奖励为 CSV 文件
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    seed_rewards.to_csv(f'rewards_Fed_seed_{seed}_{timestamp}.csv', index=False)

    print(f"Saved rewards for seed {seed} to 'rewards_seed_{seed}.csv'.")

# 准备数据以便绘制学习曲线
plot_data = []
for seed in [20]:
    seed_data = pd.read_csv(f'rewards_seed_{seed}.csv')
    for agent_id in range(num_agents):
        plot_data.append(pd.DataFrame({
            "episode": seed_data["Episode"],
            "reward": seed_data[f"Agent {agent_id + 1}"],
            "agent": f"Agent {agent_id + 1}",
            "seed": f"Seed {seed}"
        }))

df_plot = pd.concat(plot_data)

# 分别绘制不同种子下的学习曲线
sns.set(style="darkgrid", context="talk", palette="rainbow")
seeds = df_plot['seed'].unique()

for seed in seeds:
    plt.figure()
    sns.lineplot(x="episode", y="reward", hue="agent", data=df_plot[df_plot['seed'] == seed]).set(
        title=f"REINFORCE Rewards Over Episodes - {seed}"
    )
    plt.show()




