import pandas as pd
import numpy as np
from datetime import datetime


def load_states(filename):
    df = pd.read_csv(filename)
    states = df['State'].apply(lambda x: list(map(float, x.split(','))))
    return states.tolist(), df


def select_random_states(df, n_samples=5000):
    # Randomly select n_samples states from the DataFrame
    selected_indices = np.random.choice(df.index, n_samples, replace=False)
    selected_states = df.loc[selected_indices]
    return selected_states


def main():
    filename = 'states_20240919_094500.csv'  # Replace with your actual filename
    states, df = load_states(filename)

    # Randomly select 5000 states
    selected_states = select_random_states(df, n_samples=5000)

    # Save the selected states to a new CSV file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    selected_states.to_csv(f'KD_selected_states_{timestamp}.csv', index=False)


if __name__ == "__main__":
    main()
