import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import csv
import seaborn as sns
import numpy as np

# Use Seaborn style
sns.set(style="whitegrid")

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
    # Define filenames for three categories of files
    filenames = [
        f'KD_NoFed_Agent{agent_number}.csv',
        f'KD_Black_Box_SL_reward_Agent{agent_number}.csv',
        f'KD_Black_Box_MM_reward_Agent{agent_number}.csv'  # Add new category
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
def plot_all_agents(num_agents, window_size=50):
    # Set figure size
    fig_width, fig_height = 20, 8
    fig = plt.figure(figsize=(fig_width, fig_height))

    # Use GridSpec to control subplot sizes
    gs = gridspec.GridSpec(2, 5, wspace=0.3, hspace=0.3, width_ratios=[1] * 5, height_ratios=[1] * 2)

    labels = ['WoFed', 'FedBPG', 'FedBPG']  # Update labels for three categories
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Update colors for three categories

    for i in range(1, num_agents + 1):
        y_smooth_list = plot_agent_rewards(i, window_size)
        row, col = (i - 1) // 5, (i - 1) % 5
        ax = plt.subplot(gs[row, col])
        ax.set_title(f'Agent {i}')
        for j, (y_smooth, y_std) in enumerate(y_smooth_list):
            x = range(1, len(y_smooth) + 1)
            ax.plot(x, y_smooth, label=labels[j], color=colors[j])
            # Ensure standard deviation aligns with smoothed data
            std_fill_lower = np.array(y_smooth) - np.array(y_std)
            std_fill_upper = np.array(y_smooth) + np.array(y_std)
            ax.fill_between(x, std_fill_lower, std_fill_upper, color=colors[j], alpha=0.2)
        ax.legend()
        ax.set_xlabel('Rounds')
        ax.set_ylabel('Trajectories')
        ax.set_ylim(0, 500)  # Set y-axis range

    plt.tight_layout()
    plt.show()

# Call the plotting function to plot graphs for three categories of files
plot_all_agents(10)
