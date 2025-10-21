"""
Clean final analysis without Unicode issues
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from hmm_implementation import HiddenMarkovModel, generate_weather_data

def load_combined_dataset():
    """Load the best performing combined dataset."""
    print("Loading Combined Dataset (Seattle + Synthetic)...")
    
    # Load Seattle data
    seattle_df = pd.read_csv('data/seattle-weather.csv')
    weather_mapping = {
        'sun': 0,   # Sunny
        'drizzle': 1, 'fog': 1,  # Cloudy
        'rain': 2, 'snow': 2     # Rainy
    }
    seattle_df['state'] = seattle_df['weather'].map(weather_mapping)
    seattle_obs = seattle_df[['temp_max', 'temp_min', 'precipitation']].values
    seattle_states = seattle_df['state'].values
    
    # Generate synthetic data
    synthetic_states, synthetic_obs = generate_weather_data(n_days=300, seed=42)
    
    # Normalize Seattle data
    scaler = StandardScaler()
    seattle_obs_scaled = scaler.fit_transform(seattle_obs)
    
    # Combine datasets
    combined_obs = np.vstack([seattle_obs_scaled, synthetic_obs])
    combined_states = np.concatenate([seattle_states, synthetic_states])
    
    print(f"Combined dataset: {len(combined_states)} days")
    print(f"Seattle data: {len(seattle_states)} days")
    print(f"Synthetic data: {len(synthetic_states)} days")
    print(f"State distribution: {np.unique(combined_states, return_counts=True)}")
    
    return combined_obs, combined_states, seattle_df, scaler

def train_and_evaluate_model():
    """Train and evaluate the best HMM model."""
    print("\nTraining Best HMM Model...")
    
    # Load combined dataset
    observations, states, seattle_df, scaler = load_combined_dataset()
    
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
    
    # Test inference
    viterbi_states, viterbi_log_prob = hmm.viterbi_algorithm(test_obs)
    test_log_likelihood = hmm.log_likelihood(test_obs)
    
    # Calculate accuracy
    viterbi_accuracy = accuracy_score(test_states, viterbi_states)
    
    print(f"Training completed in {len(log_likelihoods)} iterations")
    print(f"Final training log-likelihood: {log_likelihoods[-1]:.2f}")
    print(f"Test log-likelihood: {test_log_likelihood:.2f}")
    print(f"Viterbi accuracy: {viterbi_accuracy:.3f}")
    
    return {
        'hmm': hmm,
        'log_likelihoods': log_likelihoods,
        'test_log_likelihood': test_log_likelihood,
        'viterbi_accuracy': viterbi_accuracy,
        'test_states': test_states,
        'viterbi_states': viterbi_states,
        'test_obs': test_obs,
        'seattle_df': seattle_df
    }

def print_results_summary(results):
    """Print comprehensive results summary."""
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    
    hmm = results['hmm']
    test_states = results['test_states']
    viterbi_states = results['viterbi_states']
    
    # Overall performance
    print(f"Overall Performance:")
    print(f"  Viterbi Accuracy: {results['viterbi_accuracy']:.3f}")
    print(f"  Test Log-Likelihood: {results['test_log_likelihood']:.2f}")
    print(f"  Training Iterations: {len(results['log_likelihoods'])}")
    
    # Per-state analysis
    print(f"\nPer-State Analysis:")
    state_names = ['Sunny', 'Cloudy', 'Rainy']
    for i, state_name in enumerate(state_names):
        true_count = np.sum(test_states == i)
        correct_count = np.sum((test_states == i) & (viterbi_states == i))
        if true_count > 0:
            accuracy = correct_count / true_count
            print(f"  {state_name:8}: {accuracy:.3f} accuracy ({correct_count}/{true_count})")
    
    # Learned parameters
    print(f"\nLearned Parameters:")
    print(f"Initial probabilities: {hmm.pi}")
    print(f"Transition matrix:")
    for i, state in enumerate(state_names):
        print(f"  {state} -> [Sunny: {hmm.A[i,0]:.3f}, Cloudy: {hmm.A[i,1]:.3f}, Rainy: {hmm.A[i,2]:.3f}]")
    
    print(f"Emission means:")
    obs_names = ['Temperature', 'Humidity', 'Precipitation']
    for i, state in enumerate(state_names):
        print(f"  {state}: ", end="")
        for j, obs_name in enumerate(obs_names):
            print(f"{obs_name}={hmm.mu[i,j]:.2f}", end="")
            if j < len(obs_names) - 1:
                print(", ", end="")
        print()

def demonstrate_forecasting(results):
    """Demonstrate weather forecasting capabilities."""
    print(f"\n" + "="*60)
    print("WEATHER FORECASTING DEMONSTRATION")
    print("="*60)
    
    hmm = results['hmm']
    test_obs = results['test_obs']
    
    # Use last 30 days as context
    context_obs = test_obs[-30:]
    
    # Forecast next 7 days
    forecast_steps = 7
    pred_states, pred_obs = hmm.forecast(context_obs, steps=forecast_steps)
    
    print(f"Forecasting next {forecast_steps} days based on last 30 days:")
    print(f"(Note: Values are normalized)")
    
    state_names = ['Sunny', 'Cloudy', 'Rainy']
    obs_names = ['Temperature', 'Humidity', 'Precipitation']
    
    for i, state in enumerate(pred_states):
        state_name = state_names[state]
        print(f"Day {i+1}: {state_name}")
        for j, obs_name in enumerate(obs_names):
            print(f"         {obs_name}: {pred_obs[i, j]:.3f}")

if __name__ == "__main__":
    print("Final Best Model Analysis")
    print("Using Combined Dataset (Seattle Weather + Synthetic)")
    print("="*60)
    
    # Train and evaluate model
    results = train_and_evaluate_model()
    
    # Print detailed results
    print_results_summary(results)
    
    # Demonstrate forecasting
    demonstrate_forecasting(results)
    
    print(f"\n" + "="*60)
    print("ANALYSIS COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"Best model achieved {results['viterbi_accuracy']:.3f} accuracy")
    print(f"Using combined dataset with {len(results['test_states'])} test samples")
    print("Model is ready for weather prediction!")

