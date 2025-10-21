"""
Data generation script for weather HMM project
This script generates synthetic weather data for training and testing
"""

import numpy as np
import pandas as pd
from hmm_implementation import generate_weather_data

def generate_and_save_data(n_days=300, seed=42):
    """
    Generate synthetic weather data and save to CSV files.
    
    Args:
        n_days: Number of days to generate
        seed: Random seed for reproducibility
    """
    print(f"Generating {n_days} days of synthetic weather data...")
    
    # Generate data
    true_states, observations = generate_weather_data(n_days=n_days, seed=seed)
    
    # Create DataFrame
    weather_df = pd.DataFrame({
        'day': range(len(true_states)),
        'true_state': true_states,
        'temperature': observations[:, 0],
        'humidity': observations[:, 1],
        'umbrella_usage': observations[:, 2]
    })
    
    # Map states to names
    state_names = ['Sunny', 'Cloudy', 'Rainy']
    weather_df['true_state_name'] = weather_df['true_state'].map({0: 'Sunny', 1: 'Cloudy', 2: 'Rainy'})
    
    # Save to CSV
    weather_df.to_csv('weather_data.csv', index=False)
    
    print(f"Data saved to weather_data.csv")
    print(f"Shape: {weather_df.shape}")
    print(f"Columns: {list(weather_df.columns)}")
    
    # Display summary
    print("\nData Summary:")
    print(weather_df.describe())
    
    print("\nState Distribution:")
    print(weather_df['true_state_name'].value_counts())
    
    return weather_df

if __name__ == "__main__":
    # Generate data
    data = generate_and_save_data(n_days=300, seed=42)
    
    print("\nData generation completed successfully!")

