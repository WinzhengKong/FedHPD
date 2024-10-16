import gym
import numpy as np
import torch
from datetime import datetime
import pandas as pd
from policy.policy import PolicyNetwork

device = torch.device("cpu")


def load_model(model_path, state_size, action_size, hidden_sizes, activation_fn):
    model = PolicyNetwork(state_size, action_size, hidden_sizes, activation_fn).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model


def test_model(env, model, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.seed(seed)

    states = []
    state = torch.FloatTensor(env.reset()).unsqueeze(0).to(device)
    done = False
    total_reward = 0

    while not done:
        with torch.no_grad():
            probs = model(state)
            action = probs.argmax(dim=1).item()
        next_state, reward, done, _ = env.step(action)
        total_reward += reward

        # Record state
        states.append(state.cpu().numpy().flatten().tolist())

        state = torch.FloatTensor(next_state).unsqueeze(0).to(device)

    return states, total_reward


def main():
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    hidden_sizes = [128]
    activation_fn = 'relu'

    model_paths = [f'agent_model_{i}.pt' for i in range(1000, 2001, 100)]
    all_states = []

    for model_path in model_paths:
        model = load_model(model_path, state_size, action_size, hidden_sizes, activation_fn)
        for seed in range(21, 31):
            states, total_reward = test_model(env, model, seed)
            print(f"Model: {model_path}, Seed: {seed}, Total Reward: {total_reward}")
            all_states.extend(states)

    # Convert to DataFrame
    all_states_df = pd.DataFrame({
        'State': [','.join(map(str, x)) for x in all_states]
    })

    # Save to CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'states_{timestamp}.csv'
    all_states_df.to_csv(filename, index=False)
    print(f"All states saved to {filename}")


if __name__ == "__main__":
    main()
