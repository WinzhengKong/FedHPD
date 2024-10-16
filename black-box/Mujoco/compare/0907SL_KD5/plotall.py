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
def plot_all_agents(filenames, num_agents, window_size=50, save_to_file=False):
    # Set figure size
    fig_width, fig_height = 15, 5  # Adjusted figure size
    fig = plt.figure(figsize=(fig_width, fig_height))

    # Use GridSpec to control subplot sizes
    gs = gridspec.GridSpec(2, 5, wspace=0.35, hspace=0.35, width_ratios=[1] * 5, height_ratios=[1] * 2)

    labels = ['WoFed', 'FedBPG']  # Update labels for two categories
    colors = ['#1f77b4', '#ff7f0e']  # Update colors for two categories

    y_smooth_list = plot_agent_rewards(filenames, window_size)

    handles = []  # To collect handles for legend
    for i in range(num_agents):
        row, col = i // 5, i % 5
        ax = plt.subplot(gs[row, col])
        ax.set_title(f'Agent {i+1}', fontsize=10)  # Adjusted font size for titles
        for j, (y_smooth, y_std) in enumerate(y_smooth_list):
            x = range(1, len(y_smooth[i]) + 1)
            line, = ax.plot(x, y_smooth[i], label=labels[j], color=colors[j])
            # Collect handles for legend
            if i == 0:
                handles.append(line)
            # Ensure standard deviation aligns with smoothed data
            std_fill_lower = y_smooth[i] - y_std[i]
            std_fill_upper = y_smooth[i] + y_std[i]
            ax.fill_between(x, std_fill_lower, std_fill_upper, color=colors[j], alpha=0.2)
        ax.set_xlabel('Episodes', fontsize=8)  # Adjusted font size for labels
        ax.set_ylabel('Reward', fontsize=8)  # Adjusted font size for labels
        ax.tick_params(axis='both', which='major', labelsize=8)  # Adjusted font size for tick labels

    # Adjust the spacing between subplots and create space for the legend
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.25, top=0.85, wspace=0.35, hspace=0.38)

    # Add global legend at the bottom of the figure
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), fontsize=10, bbox_to_anchor=(0.5, 0.05))

    if save_to_file:
        plt.savefig('Rewards_Mujoco_10.png', dpi=300, bbox_inches='tight')
    else:
        plt.show()

# Example usage
filenames = ['rewards_WoFed_0831.csv', 'rewards_Fed_0907.csv']
plot_all_agents(filenames, num_agents=10, window_size=50, save_to_file=False)

