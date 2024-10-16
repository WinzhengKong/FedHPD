import csv
import numpy as np
from datetime import datetime


# Define the function to read data
def read_data(filename):
    with open(filename, newline='') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',')
        y = []
        for row in datareader:
            # Assuming each row has only one value as y
            y.append(float(row[0]))
        return y

# Define moving average function
def moving_average(data, window_size):
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[window_size:] - cumsum[:-window_size]) / window_size

# Define function to compute standard deviation
def compute_std(data, window_size):
    std_devs = []
    for i in range(len(data)):
        std_dev = np.std(data[max(0, i - window_size + 1):i + 1])
        std_devs.append(std_dev)
    return std_devs

# Define function to compute weighted average across agents for each category
def compute_weighted_average(agent_rewards_list, window_size):
    weighted_avg = np.mean(agent_rewards_list, axis=0)
    y_smooth = moving_average(weighted_avg, window_size)
    y_std = compute_std(weighted_avg, window_size)

    # Ensure that y_smooth and y_std have the same length
    min_length = min(len(y_smooth), len(y_std))
    y_smooth = y_smooth[:min_length]
    y_std = y_std[:min_length]

    return y_smooth, y_std


# Define function to calculate and print the comparison of four categories
def calculate_comparison(num_agents, window_size=50):
    filenames_category_1 = [f'KD_NoFed_Agent{i}.csv' for i in range(1, num_agents + 1)]
    filenames_category_2 = [f'KD_Black_Box_SL_reward_Agent{i}.csv' for i in range(1, num_agents + 1)]
    filenames_category_3 = [f'KD_Black_Box_SL_10_KDper10_Agent{i}.csv' for i in range(1, num_agents + 1)]
    filenames_category_4 = [f'KD_Black_Box_SL_10_KDper20_Agent{i}.csv' for i in range(1, num_agents + 1)]

    # Read data for all categories
    rewards_category_1 = [read_data(filename) for filename in filenames_category_1]
    rewards_category_2 = [read_data(filename) for filename in filenames_category_2]
    rewards_category_3 = [read_data(filename) for filename in filenames_category_3]
    rewards_category_4 = [read_data(filename) for filename in filenames_category_4]

    # Compute weighted averages for all categories
    y_smooth_1, y_std_1 = compute_weighted_average(rewards_category_1, window_size)
    y_smooth_2, y_std_2 = compute_weighted_average(rewards_category_2, window_size)
    y_smooth_3, y_std_3 = compute_weighted_average(rewards_category_3, window_size)
    y_smooth_4, y_std_4 = compute_weighted_average(rewards_category_4, window_size)

    # Calculate the overall averages and standard deviations for each category
    avg_reward_1, std_reward_1 = np.mean(y_smooth_1), np.mean(y_std_1)
    avg_reward_2, std_reward_2 = np.mean(y_smooth_2), np.mean(y_std_2)
    avg_reward_3, std_reward_3 = np.mean(y_smooth_3), np.mean(y_std_3)
    avg_reward_4, std_reward_4 = np.mean(y_smooth_4), np.mean(y_std_4)

    # Print the results in tabular format
    print(f"{'Category':<20}{'Average Reward':<20}{'Standard Deviation':<20}")
    print('-' * 60)
    print(f"{'NoFed':<20}{avg_reward_1:<20.2f}{std_reward_1:<20.2f}")
    print(f"{'FedHPK(d=5)':<20}{avg_reward_2:<20.2f}{std_reward_2:<20.2f}")
    print(f"{'FedHPK(d=10)':<20}{avg_reward_3:<20.2f}{std_reward_3:<20.2f}")
    print(f"{'FedHPK(d=20)':<20}{avg_reward_4:<20.2f}{std_reward_4:<20.2f}")


# Call the function to calculate and print the comparison
calculate_comparison(10)
