import os
import torch
import numpy as np
import pandas as pd
import gym
from torch.distributions.normal import Normal
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from typing import Tuple
from datetime import datetime

class Policy_Network(nn.Module):
    """Parametrized Policy Network."""

    def __init__(self, obs_space_dims: int, action_space_dims: int):
        super().__init__()

        hidden_space1 = 16
        hidden_space2 = 32

        self.shared_net = nn.Sequential(
            nn.Linear(obs_space_dims, hidden_space1),
            nn.Tanh(),
            nn.Linear(hidden_space1, hidden_space2),
            nn.Tanh(),
        )

        self.policy_mean_net = nn.Sequential(
            nn.Linear(hidden_space2, action_space_dims)
        )

        self.policy_stddev_net = nn.Sequential(
            nn.Linear(hidden_space2, action_space_dims)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shared_features = self.shared_net(x.float())
        action_means = self.policy_mean_net(shared_features)
        action_stddevs = torch.log(
            1 + torch.exp(self.policy_stddev_net(shared_features))
        )
        return action_means, action_stddevs


class REINFORCE:
    """REINFORCE algorithm."""

    def __init__(self, obs_space_dims: int, action_space_dims: int):
        self.learning_rate = 1e-4
        self.gamma = 0.99
        self.eps = 1e-6

        self.probs = []
        self.rewards = []

        self.net = Policy_Network(obs_space_dims, action_space_dims)
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



# Function to load a saved model
def load_model(agent, model_path):
    agent.net.load_state_dict(torch.load(model_path))
    agent.net.eval()  # Set the model to evaluation mode

# Function to test the agent
def test_agent(agent, env, num_episodes, seed):
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    total_rewards = []
    state_history = []

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.sample_action(state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            # Record the state, action, reward, and next state
            state_history.append({
                "Episode": episode + 1,
                "State": state.tolist(),
                "Action": action.tolist(),
                "Reward": reward,
                "Next_State": next_state.tolist(),
                "Done": done
            })

            state = next_state

        total_rewards.append(total_reward)

    return total_rewards, state_history

# Directory where models are saved
model_dir = "models"

# Test settings
num_episodes = 1  # Number of test episodes per seed
seeds = range(1, 11)  # Random seeds from 1 to 10
env_name = "InvertedPendulum-v2"

# Initialize environment
env = gym.make(env_name)
obs_space_dims = env.observation_space.shape[0]
action_space_dims = env.action_space.shape[0]

# DataFrame to store all state transitions
all_state_transitions_df = pd.DataFrame()

# Iterate over all saved models
for model_file in os.listdir(model_dir):
    if model_file.endswith(".pth"):
        # Extract agent number, seed, and episode from the filename
        parts = model_file.replace(".pth", "").split("_")
        agent_num = int(parts[1])
        model_seed = int(parts[3])
        model_episode = int(parts[5])

        # Initialize agent and load the model
        agent = REINFORCE(obs_space_dims, action_space_dims)
        model_path = os.path.join(model_dir, model_file)
        load_model(agent, model_path)

        print(f"Testing model: {model_file}")

        # Test the agent with random seeds from 1 to 10
        for seed in seeds:
            total_rewards, state_history = test_agent(agent, env, num_episodes, seed)
            avg_reward = np.mean(total_rewards)
            print(f"  Seed {seed}: Average Reward: {avg_reward}")

            # Append the state history to the DataFrame
            state_df = pd.DataFrame(state_history)
            state_df["Model"] = model_file
            state_df["Test_Seed"] = seed

            all_state_transitions_df = pd.concat([all_state_transitions_df, state_df], ignore_index=True)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Save all state transitions to a CSV file
all_state_transitions_df.to_csv(f"all_state_transitions_{timestamp}.csv", index=False)
print("State transitions saved!")
