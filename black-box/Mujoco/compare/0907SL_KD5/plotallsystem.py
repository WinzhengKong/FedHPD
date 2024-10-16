import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import csv
import numpy as np
from datetime import datetime

# Define the function to read data
def read_data(filename):
    with open(filename, newline='') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',')
        data = []
        for row in datareader:
            # Skip the header row
            if row[0] != 'Episode':
                row_data = [float(x) for x in row[1:]]
                data.append(row_data)
        return np.array(data).T  # Transpose the data to get agent-wise rewards

# Define moving average function
def moving_average(data, window_size):
    cum_sums = np.cumsum(np.insert(data, 0, 0, axis=1), axis=1)
    return (cum_sums[:, window_size:] - cum_sums[:, :-window_size]) / window_size

# Define function to compute standard deviation, keeping the length consistent with moving average
def compute_std(data, window_size):
    std_devs = []
    for i in range(data.shape[1] - window_size + 1):
        std_dev = np.std(data[:, i:i + window_size], axis=1)
        std_devs.append(std_dev)
    return np.array(std_devs).T  # Transpose the result to match the data shape

# Define function to plot agent rewards
def plot_agent_rewards(filenames, window_size=50):
    # Create lists for smoothed data and standard deviation
    y_smooth_list = []
    y_std_list = []
    for filename in filenames:
        data = read_data(filename)
        y_smooth = moving_average(data, window_size)  # Smooth the data
        y_std = compute_std(data, window_size)  # Compute standard deviation

        y_smooth_list.append((y_smooth, y_std))

    return y_smooth_list

# Define function to plot multiple subplots
def plot_all_agents(filenames, window_size=50, save_to_file=False):
    # Set figure size
    fig_width, fig_height = 5, 5
    fig = plt.figure(figsize=(fig_width, fig_height))

    labels = ['WoFed', 'FedBPG']
    colors = ['#1f77b4', '#ff7f0e']

    y_smooth_list = plot_agent_rewards(filenames, window_size)

    # Compute the average of smoothed rewards across all agents
    y_smooth_avg_list = []
    y_std_avg_list = []
    for y_smooth, y_std in y_smooth_list:
        y_smooth_avg = np.mean(y_smooth, axis=0)
        y_std_avg = np.mean(y_std, axis=0)
        y_smooth_avg_list.append(y_smooth_avg)
        y_std_avg_list.append(y_std_avg)

    # Plot the average rewards
    for j, (y_smooth_avg, y_std_avg) in enumerate(zip(y_smooth_avg_list, y_std_avg_list)):
        x = range(1, len(y_smooth_avg) + 1)
        plt.plot(x, y_smooth_avg, label=labels[j], color=colors[j])
        # Ensure standard deviation aligns with smoothed data
        std_fill_lower = y_smooth_avg - y_std_avg
        std_fill_upper = y_smooth_avg + y_std_avg
        plt.fill_between(x, std_fill_lower, std_fill_upper, color=colors[j], alpha=0.2)

    plt.legend(fontsize=12)
    plt.xlabel('Episodes', fontsize=12)
    plt.ylabel('Average Reward', fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=10)

    if save_to_file:
        plt.savefig('Rewards_Mujoco_System.png', dpi=300, bbox_inches='tight')
    else:
        plt.show()



filenames = ['rewards_WoFed_0831.csv', 'rewards_Fed_0901.csv']
plot_all_agents(filenames, window_size=50, save_to_file=True)
