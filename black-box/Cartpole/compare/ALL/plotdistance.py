import csv
import numpy as np

# Define the function to read data
def read_data(filename):
    with open(filename, newline='') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',')
        y = []
        for row in datareader:
            # Assuming each row has only one value as y
            y.append(float(row[0]))
        return y

# Calculate the maximum difference, corresponding average values, and overall improvement percentage
def calculate_difference(filename1, filename2, num_agents):
    max_diff = 0
    max_diff_y1_avg = 0
    max_diff_y2_avg = 0
    total_y1 = 0
    total_y2 = 0
    total_rounds = 0

    # Find the maximum number of rounds
    max_rounds = 0
    for agent_number in range(1, num_agents + 1):
        file1 = f'{filename1}_Agent{agent_number}.csv'
        file2 = f'{filename2}_Agent{agent_number}.csv'
        y1 = read_data(file1)
        y2 = read_data(file2)
        if len(y1) > max_rounds:
            max_rounds = len(y1)
        if len(y2) > max_rounds:
            max_rounds = len(y2)

    # Iterate through the rounds and calculate the maximum difference, corresponding averages, and overall improvement
    for round_number in range(max_rounds):
        y1_sum = 0
        y2_sum = 0
        y1_count = 0
        y2_count = 0
        for agent_number in range(1, num_agents + 1):
            file1 = f'{filename1}_Agent{agent_number}.csv'
            file2 = f'{filename2}_Agent{agent_number}.csv'
            y1 = read_data(file1)
            y2 = read_data(file2)
            if round_number < len(y1):
                y1_sum += y1[round_number]
                y1_count += 1
            if round_number < len(y2):
                y2_sum += y2[round_number]
                y2_count += 1
        y1_avg = y1_sum / y1_count
        y2_avg = y2_sum / y2_count
        diff = abs(y1_avg - y2_avg)
        if diff > max_diff:
            max_diff = diff
            max_diff_y1_avg = y1_avg
            max_diff_y2_avg = y2_avg
        total_y1 += y1_avg
        total_y2 += y2_avg
        total_rounds += 1

    overall_improvement = (total_y2 - total_y1) / total_y1 * 100

    return max_diff, max_diff_y1_avg, max_diff_y2_avg, overall_improvement

# Example usage
filename1 = 'KD_Black_Box_WoFed_0827'
filename2 = 'KD_Black_Box_SL_10_0827'
num_agents = 10

max_diff, max_diff_y1_avg, max_diff_y2_avg, overall_improvement = calculate_difference(filename1, filename2, num_agents)
print(f"Maximum difference: {max_diff:.4f}")
print(f"Average for {filename1} at maximum difference: {max_diff_y1_avg:.4f}")
print(f"Average for {filename2} at maximum difference: {max_diff_y2_avg:.4f}")
print(f"Overall improvement percentage: {overall_improvement:.2f}%")
