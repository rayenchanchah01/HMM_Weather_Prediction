"""
Final dataset comparison and model training script
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from hmm_implementation import HiddenMarkovModel, generate_weather_data

def load_seattle_data():
    """Load and preprocess Seattle weather data."""
    print("Loading Seattle Weather Dataset...")
    
    # Load data
    df = pd.read_csv('data/seattle-weather.csv')
    
    # Map weather types to numerical states
    weather_mapping = {
        'sun': 0,   # Sunny
        'drizzle': 1, 'fog': 1,  # Cloudy
        'rain': 2, 'snow': 2     # Rainy
    }
    
    df['state'] = df['weather'].map(weather_mapping)
    
    # Create observations: [temp_max, temp_min, precipitation]
    observations = df[['temp_max', 'temp_min', 'precipitation']].values
    states = df['state'].values
    
    # Normalize observations
    scaler = StandardScaler()
    observations_scaled = scaler.fit_transform(observations)
    
    print(f"Dataset size: {len(states)} days")
    print(f"State distribution: {np.unique(states, return_counts=True)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    return observations_scaled, states, df, scaler

def load_synthetic_data():
    """Load synthetic weather data."""
    print("\nGenerating Synthetic Weather Dataset...")
    
    # Generate synthetic data
    true_states, observations = generate_weather_data(n_days=300, seed=42)
    
    print(f"Dataset size: {len(true_states)} days")
    print(f"State distribution: {np.unique(true_states, return_counts=True)}")
    
    return observations, true_states

def train_model(observations, states, dataset_name):
    """Train HMM model on given dataset."""
    print(f"\nTraining HMM on {dataset_name}...")
    
    # Split data
    train_size = int(0.8 * len(observations))
    train_obs = observations[:train_size]
    test_obs = observations[train_size:]
    train_states = states[:train_size]
    test_states = states[train_size:]
    
    print(f"Training set: {len(train_obs)} days")
    print(f"Test set: {len(test_obs)} days")
    
    # Train HMM
    hmm = HiddenMarkovModel(n_states=3, n_observations=3)
    log_likelihoods = hmm.baum_welch(train_obs, max_iterations=50, tolerance=1e-6)
    
    # Test
    viterbi_states, _ = hmm.viterbi_algorithm(test_obs)
    accuracy = accuracy_score(test_states, viterbi_states)
    
    print(f"Training iterations: {len(log_likelihoods)}")
    print(f"Final log-likelihood: {log_likelihoods[-1]:.2f}")
    print(f"Viterbi accuracy: {accuracy:.3f}")
    
    return accuracy, hmm, test_states, viterbi_states

def create_combined_dataset(seattle_obs, seattle_states, synthetic_obs, synthetic_states):
    """Create a combined dataset from both sources."""
    print("\nCreating Combined Dataset...")
    
    # Combine observations and states
    combined_obs = np.vstack([seattle_obs, synthetic_obs])
    combined_states = np.concatenate([seattle_states, synthetic_states])
    
    print(f"Combined dataset size: {len(combined_states)} days")
    print(f"Combined state distribution: {np.unique(combined_states, return_counts=True)}")
    
    return combined_obs, combined_states

if __name__ == "__main__":
    print("Dataset Comparison and Model Training")
    print("=" * 50)
    
    # Load datasets
    seattle_obs, seattle_states, seattle_df, scaler = load_seattle_data()
    synthetic_obs, synthetic_states = load_synthetic_data()
    
    # Train models on individual datasets
    seattle_acc, seattle_hmm, seattle_test, seattle_pred = train_model(
        seattle_obs, seattle_states, "Seattle Weather"
    )
    
    synthetic_acc, synthetic_hmm, synthetic_test, synthetic_pred = train_model(
        synthetic_obs, synthetic_states, "Synthetic Weather"
    )
    
    # Create and train on combined dataset
    combined_obs, combined_states = create_combined_dataset(
        seattle_obs, seattle_states, synthetic_obs, synthetic_states
    )
    
    combined_acc, combined_hmm, combined_test, combined_pred = train_model(
        combined_obs, combined_states, "Combined Dataset"
    )
    
    # Results comparison
    print("\n" + "=" * 50)
    print("RESULTS COMPARISON")
    print("=" * 50)
    
    results = [
        ("Seattle Weather", seattle_acc),
        ("Synthetic Weather", synthetic_acc),
        ("Combined Dataset", combined_acc)
    ]
    
    for name, acc in results:
        print(f"{name:20}: {acc:.3f} accuracy")
    
    # Find best model
    best_name, best_acc = max(results, key=lambda x: x[1])
    
    print(f"\nBEST PERFORMING DATASET: {best_name}")
    print(f"Accuracy: {best_acc:.3f}")
    
    # Recommendations
    print(f"\nRECOMMENDATIONS:")
    if best_name == "Seattle Weather":
        print("- Use Seattle Weather Dataset")
        print("- Real-world data with authentic weather patterns")
        print("- Larger dataset provides better generalization")
        print("- More diverse weather conditions")
    elif best_name == "Combined Dataset":
        print("- Use Combined Dataset")
        print("- Benefits from both real and synthetic data")
        print("- Largest training dataset")
        print("- Best overall performance")
    else:
        print("- Use Synthetic Weather Dataset")
        print("- Clean, controlled data for educational purposes")
        print("- Reproducible results")
    
    print(f"\nModel training completed successfully!")
    print(f"Final recommendation: {best_name} with {best_acc:.3f} accuracy")

