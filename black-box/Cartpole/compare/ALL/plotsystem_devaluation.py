import matplotlib.pyplot as plt
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


# Define function to plot the comparison of seven categories
def plot_comparison(num_agents, window_size=50):
    filenames_category_1 = [f'KD_Black_Box_WoFed_0827_Agent{i}.csv' for i in range(1, num_agents + 1)]
    filenames_category_2 = [f'KD_Black_Box_SL_10_KDper2_Agent{i}.csv' for i in range(1, num_agents + 1)]
    filenames_category_3 = [f'KD_Black_Box_SL_10_0827_Agent{i}.csv' for i in range(1, num_agents + 1)]
    filenames_category_4 = [f'KD_Black_Box_SL_10_KDper10_Agent{i}.csv' for i in range(1, num_agents + 1)]
    filenames_category_5 = [f'KD_Black_Box_SL_10_KDper20_Agent{i}.csv' for i in range(1, num_agents + 1)]
    filenames_category_6 = [f'KD_Black_Box_SL_10_KDper40_Agent{i}.csv' for i in range(1, num_agents + 1)]
    filenames_category_7 = [f'KD_Black_Box_SL_10_KDper80_Agent{i}.csv' for i in range(1, num_agents + 1)]

    # Read data for all categories
    rewards_category_1 = [read_data(filename) for filename in filenames_category_1]
    rewards_category_2 = [read_data(filename) for filename in filenames_category_2]
    rewards_category_3 = [read_data(filename) for filename in filenames_category_3]
    rewards_category_4 = [read_data(filename) for filename in filenames_category_4]
    rewards_category_5 = [read_data(filename) for filename in filenames_category_5]
    rewards_category_6 = [read_data(filename) for filename in filenames_category_6]
    rewards_category_7 = [read_data(filename) for filename in filenames_category_7]

    # Compute weighted averages for all categories
    y_smooth_1, y_std_1 = compute_weighted_average(rewards_category_1, window_size)
    y_smooth_2, y_std_2 = compute_weighted_average(rewards_category_2, window_size)
    y_smooth_3, y_std_3 = compute_weighted_average(rewards_category_3, window_size)
    y_smooth_4, y_std_4 = compute_weighted_average(rewards_category_4, window_size)
    y_smooth_5, y_std_5 = compute_weighted_average(rewards_category_5, window_size)
    y_smooth_6, y_std_6 = compute_weighted_average(rewards_category_6, window_size)
    y_smooth_7, y_std_7 = compute_weighted_average(rewards_category_7, window_size)

    # Plot the comparison
    plt.figure(figsize=(4, 3))
    labels = ['NoFed', 'FedHPK(d=2)', 'FedHPK(d=5)', 'FedHPK(d=10)',
              'FedHPK(d=20)', 'FedHPK(d=40)', 'FedHPK(d=80)']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2']

    # Ensure all y_smooth and y_std have the same length
    min_length = min(
        len(y_smooth_1), len(y_smooth_2), len(y_smooth_3), len(y_smooth_4), len(y_smooth_5), len(y_smooth_6),
        len(y_smooth_7)
    )
    y_smooth_list = [y_smooth_1[:min_length], y_smooth_2[:min_length], y_smooth_3[:min_length], y_smooth_4[:min_length],
                     y_smooth_5[:min_length], y_smooth_6[:min_length], y_smooth_7[:min_length]]
    y_std_list = [y_std_1[:min_length], y_std_2[:min_length], y_std_3[:min_length], y_std_4[:min_length],
                  y_std_5[:min_length], y_std_6[:min_length], y_std_7[:min_length]]

    x = range(1, min_length + 1)

    # Plot each category
    for i in range(7):
        plt.plot(x, y_smooth_list[i], label=labels[i], color=colors[i], linewidth=2)
        plt.fill_between(x, y_smooth_list[i] - y_std_list[i], y_smooth_list[i] + y_std_list[i], alpha=0.2,
                         color=colors[i])

    plt.title('Cartpole')
    plt.xlabel('Rounds')
    plt.ylabel('Average Rewards')
    plt.xlim(0, 2000)
    plt.ylim(0, 500)  # Set y-axis range
    plt.legend(fontsize=8)
    plt.tight_layout()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_filename = f'Rewards_Cartpole_system_d_explore.png'
    plt.savefig(output_filename, dpi=600, bbox_inches='tight')
    plt.show()


# Call the plotting function to compare the overall performance of the seven categories
plot_comparison(10)
