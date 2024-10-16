import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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


# Define function to compute standard deviation, keeping the length consistent with moving average
def compute_std(data, window_size):
    std_devs = []
    for i in range(len(data) - window_size + 1):
        std_dev = np.std(data[i:i + window_size])
        std_devs.append(std_dev)
    return std_devs


# Define function to plot agent rewards
def plot_agent_rewards(agent_number, window_size=50):
    # Define filenames for seven categories of files
    filenames = [
        f'KD_Black_Box_WoFed_10_0828_Agent{agent_number}.csv',
        f'KD_Black_Box_10_KDper2_Agent{agent_number}.csv',
        f'KD_Black_Box_SL_10_0827_Agent{agent_number}.csv',
        f'KD_Black_Box_10_KDper10_Agent{agent_number}.csv',
        f'KD_Black_Box_10_KDper20_Agent{agent_number}.csv',  # New fifth category
        f'KD_Black_Box_10_KDper40_Agent{agent_number}.csv',  # New sixth category
        f'KD_Black_Box_10_KDper80_Agent{agent_number}.csv'  # New seventh category
    ]

    # Create lists for smoothed data and standard deviation
    y_smooth_list = []
    y_std_list = []
    for filename in filenames:
        y = read_data(filename)
        y_smooth = moving_average(y, window_size)  # Smooth the data
        y_std = compute_std(y, window_size)  # Compute standard deviation
        # Ensure standard deviation length matches smoothed data length
        y_std = y_std[:len(y_smooth)]
        y_smooth_list.append((y_smooth, y_std))
    return y_smooth_list


# Define function to plot multiple subplots
def plot_all_agents(num_agents, window_size=50, save_to_file=False):
    # Set figure size
    fig_width, fig_height = 24, 7  # Adjusted figure size
    fig = plt.figure(figsize=(fig_width, fig_height))

    # Use GridSpec to control subplot sizes with smaller height ratio
    gs = gridspec.GridSpec(2, 5, wspace=0.24, hspace=0.32, width_ratios=[1] * 5, height_ratios=[1] * 2)

    labels = ['NoFed', 'FedHPK(d=2)', 'FedHPK(d=5)', 'FedHPK(d=10)',
              'FedHPK(d=20)', 'FedHPK(d=40)', 'FedHPK(d=80)']  # Updated labels for seven categories
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2']  # Updated colors for seven categories

    handles = []  # To collect handles for legend
    for i in range(1, num_agents + 1):
        y_smooth_list = plot_agent_rewards(i, window_size)
        row, col = (i - 1) // 5, (i - 1) % 5
        ax = plt.subplot(gs[row, col])
        ax.set_title(f'Agent {i}', fontsize=10)  # Adjusted font size for titles
        for j, (y_smooth, y_std) in enumerate(y_smooth_list):
            x = range(1, len(y_smooth) + 1)
            line, = ax.plot(x, y_smooth, label=labels[j], color=colors[j], linewidth=1.5)
            # Ensure standard deviation aligns with smoothed data
            std_fill_lower = np.array(y_smooth) - np.array(y_std)
            std_fill_upper = np.array(y_smooth) + np.array(y_std)
            ax.fill_between(x, std_fill_lower, std_fill_upper, color=colors[j], alpha=0.2)
            if i == 1:  # Collect handles and labels only once
                handles.append(line)
        ax.set_xlabel('Rounds', fontsize=10)  # Adjusted font size for labels
        ax.set_ylabel('Rewards', fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=10)  # Adjusted tick label size

    # Adjust layout to make space for the legend
    plt.tight_layout(pad=2.0)
    # Add global legend at the bottom of the figure
    fig.subplots_adjust(bottom=0.13)  # Create space at the bottom for the legend
    fig.legend(handles, labels, loc='lower center', ncol=7, fontsize=10)

    if save_to_file:
        # Save the figure with high resolution and remove extra white space
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f'Rewards_LunarLander_dexplore.png'
        plt.savefig(output_filename, dpi=600, bbox_inches='tight')
    else:
        plt.show()


# Example usage: save the image to file instead of displaying it
plot_all_agents(10, save_to_file=True)
