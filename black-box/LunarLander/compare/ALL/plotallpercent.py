import matplotlib.pyplot as plt
import csv
import numpy as np
import seaborn as sns

# Define the function to read data
def read_data(filename):
    with open(filename, newline='') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',')
        y = []
        for row in datareader:
            # Assuming each row has only one value as y
            y.append(float(row[0]))
        return y

# Define the function to plot the graph
def plot_agent_rewards():
    # Define the filename list, containing two categories of files
    filenames = [f'Lunar_NoFed_Agent{i + 1}.csv' for i in range(10)] + \
                [f'KD_Black_Box_Agent{i + 1}.csv' for i in range(10)]  # Only two categories

    # Define algorithm labels, containing two algorithms
    algorithm_labels = ['WoFed', 'Fed']  # Only two algorithm labels

    # Read the data
    y_data = [read_data(filename) for filename in filenames]

    # Aggregate reward values of all agents for each algorithm
    aggregated_data = [np.concatenate([y_data[i + j * 10] for i in range(10)]) for j in range(2)]  # Only two categories

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the figure size

    means = []

    for j, data in enumerate(aggregated_data):
        # Calculate median, quartiles, and mean
        median = np.median(data)
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        mean = np.mean(data)

        means.append(mean)

        # Calculate the top and bottom 25% of data points
        top_25 = np.percentile(data, 75)
        bottom_25 = np.percentile(data, 25)

        # Filter data points within the interquartile range
        filtered_data = [x for x in data if q1 <= x <= q3]
        filtered_count = len(filtered_data)

        # Plot the box plot
        sns.boxplot(data=filtered_data, ax=ax, positions=[j], width=0.5, showfliers=False)

        # Print and label the median, quartiles, and mean
        print(f'{algorithm_labels[j]}: Median = {median:.2f}, Mean = {mean:.2f}, Q1 = {q1:.2f}, Q3 = {q3:.2f}, Points in Q1-Q3 = {filtered_count}')
        ax.text(j, median, f'{median:.2f}', ha='center', va='bottom', color='black')
        ax.text(j, q1, f'{q1:.2f}', ha='center', va='top', color='black')
        ax.text(j, q3, f'{q3:.2f}', ha='center', va='bottom', color='black')
        ax.text(j, mean, f'{mean:.2f}', ha='center', va='bottom', color='blue')

        # Mark the top and bottom 25% points
        ax.plot(j, top_25, 'g^', markersize=10)
        ax.plot(j, bottom_25, 'rv', markersize=10)

    # Set labels
    ax.set_title('Reward Distribution of Algorithms')
    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Reward')
    ax.set_xticks(range(2))  # Only two categories
    ax.set_xticklabels(algorithm_labels)

    # Set y-axis range
    # ax.set_ylim(0, 500)

    # Set margins to ensure all plots are fully displayed
    ax.margins(x=0.05)  # Adjust x-axis margins

    # Find the best algorithm
    best_idx = np.argmax(means)
    print(f'Best Algorithm is {algorithm_labels[best_idx]} with Mean Reward = {means[best_idx]:.2f}')

    plt.tight_layout()
    plt.show()

# Call the plot function, aggregating all agents' rewards in two categories, and plot the points within the interquartile range
plot_agent_rewards()
