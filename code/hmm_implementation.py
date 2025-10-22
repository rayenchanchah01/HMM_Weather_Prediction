"""
Hidden Markov Model Implementation
Weather Prediction Case Study

This module implements a complete HMM from scratch including:
- Forward-Backward algorithm
- Viterbi algorithm  
- Baum-Welch (EM) training algorithm
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import logsumexp
from typing import Tuple, List, Optional
import warnings

class HiddenMarkovModel:
    """
    Hidden Markov Model implementation for weather prediction.
    
    States: [Sunny, Cloudy, Rainy]
    Observations: [temperature, humidity, umbrella_count]
    """
    
    def __init__(self, n_states: int = 3, n_observations: int = 3):
        """
        Initialize HMM with random parameters.
        
        Args:
            n_states: Number of hidden states (default: 3 for weather)
            n_observations: Number of observation dimensions (default: 3)
        """
        self.n_states = n_states
        self.n_observations = n_observations
        
        # Initialize parameters randomly
        self.pi = np.random.dirichlet(np.ones(n_states))  # Initial state probabilities
        self.A = np.random.dirichlet(np.ones(n_states), size=n_states)  # Transition matrix
        self.mu = np.random.normal(0, 1, (n_states, n_observations))  # Emission means
        self.sigma = np.array([np.eye(n_observations) for _ in range(n_states)])  # Emission covariances
        
        # State names for weather
        self.state_names = ['Sunny', 'Cloudy', 'Rainy']
        
    def log_likelihood(self, observations: np.ndarray) -> float:
        """
        Compute log-likelihood of observations using forward algorithm.
        
        Args:
            observations: Array of shape (T, n_observations)
            
        Returns:
            Log-likelihood value
        """
        T = observations.shape[0]
        
        # Forward algorithm
        alpha = np.zeros((T, self.n_states))
        
        # Initialization
        for i in range(self.n_states):
            alpha[0, i] = np.log(self.pi[i]) + self._log_emission_prob(observations[0], i)
        
        # Recursion
        for t in range(1, T):
            for j in range(self.n_states):
                log_sum = logsumexp([
                    alpha[t-1, i] + np.log(self.A[i, j])
                    for i in range(self.n_states)
                ])
                alpha[t, j] = log_sum + self._log_emission_prob(observations[t], j)
        
        # Final log-likelihood
        return logsumexp(alpha[-1, :])
    
    def forward_algorithm(self, observations: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Forward algorithm for computing state probabilities.
        
        Args:
            observations: Array of shape (T, n_observations)
            
        Returns:
            Tuple of (alpha matrix, log-likelihood)
        """
        T = observations.shape[0]
        alpha = np.zeros((T, self.n_states))
        
        # Initialization
        for i in range(self.n_states):
            alpha[0, i] = np.log(self.pi[i]) + self._log_emission_prob(observations[0], i)
        
        # Recursion
        for t in range(1, T):
            for j in range(self.n_states):
                log_sum = logsumexp([
                    alpha[t-1, i] + np.log(self.A[i, j])
                    for i in range(self.n_states)
                ])
                alpha[t, j] = log_sum + self._log_emission_prob(observations[t], j)
        
        log_likelihood = logsumexp(alpha[-1, :])
        return alpha, log_likelihood
    
    def backward_algorithm(self, observations: np.ndarray) -> np.ndarray:
        """
        Backward algorithm for computing state probabilities.
        
        Args:
            observations: Array of shape (T, n_observations)
            
        Returns:
            Beta matrix
        """
        T = observations.shape[0]
        beta = np.zeros((T, self.n_states))
        
        # Initialization
        beta[-1, :] = 0  # log(1) = 0
        
        # Recursion
        for t in range(T-2, -1, -1):
            for i in range(self.n_states):
                log_sum = logsumexp([
                    np.log(self.A[i, j]) + self._log_emission_prob(observations[t+1], j) + beta[t+1, j]
                    for j in range(self.n_states)
                ])
                beta[t, i] = log_sum
        
        return beta
    
    def viterbi_algorithm(self, observations: np.ndarray) -> Tuple[List[int], float]:
        """
        Viterbi algorithm for finding most likely state sequence.
        
        Args:
            observations: Array of shape (T, n_observations)
            
        Returns:
            Tuple of (state sequence, log-probability)
        """
        T = observations.shape[0]
        delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)
        
        # Initialization
        for i in range(self.n_states):
            delta[0, i] = np.log(self.pi[i]) + self._log_emission_prob(observations[0], i)
        
        # Recursion
        for t in range(1, T):
            for j in range(self.n_states):
                max_log_prob = -np.inf
                best_state = 0
                for i in range(self.n_states):
                    log_prob = delta[t-1, i] + np.log(self.A[i, j])
                    if log_prob > max_log_prob:
                        max_log_prob = log_prob
                        best_state = i
                delta[t, j] = max_log_prob + self._log_emission_prob(observations[t], j)
                psi[t, j] = best_state
        
        # Backtracking
        states = [0] * T
        states[-1] = np.argmax(delta[-1, :])
        log_prob = delta[-1, states[-1]]
        
        for t in range(T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]
        
        return states, log_prob
    
    def baum_welch(self, observations: np.ndarray, max_iterations: int = 100, 
                   tolerance: float = 1e-6) -> List[float]:
        """
        Baum-Welch algorithm for training HMM parameters.
        
        Args:
            observations: Array of shape (T, n_observations)
            max_iterations: Maximum number of EM iterations
            tolerance: Convergence tolerance
            
        Returns:
            List of log-likelihood values per iteration
        """
        log_likelihoods = []
        
        for iteration in range(max_iterations):
            # E-step: Compute forward and backward probabilities
            alpha, log_likelihood = self.forward_algorithm(observations)
            beta = self.backward_algorithm(observations)
            
            log_likelihoods.append(log_likelihood)
            
            # Compute gamma and xi
            gamma = self._compute_gamma(alpha, beta, observations)
            xi = self._compute_xi(alpha, beta, observations)
            
            # M-step: Update parameters
            self._update_parameters(gamma, xi, observations)
            
            # Check convergence
            if iteration > 0 and abs(log_likelihood - log_likelihoods[-2]) < tolerance:
                break
        
        return log_likelihoods
    
    def _log_emission_prob(self, observation: np.ndarray, state: int) -> float:
        """Compute log emission probability for Gaussian emissions."""
        try:
            # Multivariate Gaussian log probability
            diff = observation - self.mu[state]
            inv_sigma = np.linalg.inv(self.sigma[state])
            
            log_prob = -0.5 * (
                self.n_observations * np.log(2 * np.pi) +
                np.log(np.linalg.det(self.sigma[state])) +
                diff.T @ inv_sigma @ diff
            )
            return log_prob
        except np.linalg.LinAlgError:
            # Handle singular covariance matrices
            return -np.inf
    
    def _compute_gamma(self, alpha: np.ndarray, beta: np.ndarray, 
                      observations: np.ndarray) -> np.ndarray:
        """Compute gamma probabilities (state posteriors)."""
        T = observations.shape[0]
        gamma = np.zeros((T, self.n_states))
        
        for t in range(T):
            for i in range(self.n_states):
                gamma[t, i] = alpha[t, i] + beta[t, i]
            
            # Normalize
            gamma[t, :] = gamma[t, :] - logsumexp(gamma[t, :])
        
        return np.exp(gamma)
    
    def _compute_xi(self, alpha: np.ndarray, beta: np.ndarray, 
                   observations: np.ndarray) -> np.ndarray:
        """Compute xi probabilities (transition posteriors)."""
        T = observations.shape[0]
        xi = np.zeros((T-1, self.n_states, self.n_states))
        
        for t in range(T-1):
            for i in range(self.n_states):
                for j in range(self.n_states):
                    xi[t, i, j] = (
                        alpha[t, i] + 
                        np.log(self.A[i, j]) + 
                        self._log_emission_prob(observations[t+1], j) + 
                        beta[t+1, j]
                    )
            
            # Normalize
            xi[t, :, :] = xi[t, :, :] - logsumexp(xi[t, :, :])
        
        return np.exp(xi)
    
    def _update_parameters(self, gamma: np.ndarray, xi: np.ndarray, 
                          observations: np.ndarray):
        """Update HMM parameters using EM estimates."""
        T = observations.shape[0]
        
        # Update initial probabilities
        self.pi = gamma[0, :]
        
        # Update transition matrix
        for i in range(self.n_states):
            denominator = np.sum(gamma[:-1, i])
            if denominator > 0:
                for j in range(self.n_states):
                    self.A[i, j] = np.sum(xi[:, i, j]) / denominator
        
        # Update emission parameters (Gaussian)
        for i in range(self.n_states):
            gamma_i = gamma[:, i:i+1]  # Keep dimensions
            
            # Update means
            self.mu[i] = np.sum(gamma_i * observations, axis=0) / np.sum(gamma_i)
            
            # Update covariances
            diff = observations - self.mu[i]
            weighted_diff = gamma_i * diff
            self.sigma[i] = (
                weighted_diff.T @ diff / np.sum(gamma_i) + 
                1e-6 * np.eye(self.n_observations)  # Regularization
            )
    
    def forecast(self, observations: np.ndarray, steps: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forecast future observations and states.
        
        Args:
            observations: Past observations
            steps: Number of steps to forecast
            
        Returns:
            Tuple of (predicted_states, predicted_observations)
        """
        # Get most recent state probabilities
        alpha, _ = self.forward_algorithm(observations)
        current_state_probs = np.exp(alpha[-1, :] - logsumexp(alpha[-1, :]))
        
        predicted_states = []
        predicted_observations = []
        
        for step in range(steps):
            # Predict next state probabilities
            next_state_probs = current_state_probs @ self.A
            
            # Sample or predict most likely state
            predicted_state = np.argmax(next_state_probs)
            predicted_states.append(predicted_state)
            
            # Predict observation
            predicted_obs = self.mu[predicted_state]
            predicted_observations.append(predicted_obs)
            
            # Update for next step
            current_state_probs = next_state_probs
        
        return np.array(predicted_states), np.array(predicted_observations)
    
    def plot_transition_matrix(self, figsize=(8, 6)):
        """Plot the transition matrix as a heatmap."""
        plt.figure(figsize=figsize)
        sns.heatmap(self.A, annot=True, fmt='.3f', cmap='Blues',
                   xticklabels=self.state_names, yticklabels=self.state_names)
        plt.title('HMM Transition Matrix')
        plt.xlabel('Next State')
        plt.ylabel('Current State')
        plt.tight_layout()
        plt.show()
    
    def plot_emission_parameters(self, figsize=(12, 4)):
        """Plot emission parameters for each state."""
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        for i, state in enumerate(self.state_names):
            ax = axes[i]
            
            # Plot means
            ax.bar(range(self.n_observations), self.mu[i], 
                  color=['red', 'blue', 'green'][i], alpha=0.7)
            ax.set_title(f'{state} State Emissions')
            ax.set_xlabel('Observation Dimension')
            ax.set_ylabel('Mean Value')
            ax.set_xticks(range(self.n_observations))
            ax.set_xticklabels(['Temperature', 'Humidity', 'Umbrella'])
        
        plt.tight_layout()
        plt.show()
if __name__ == "__main__":
    # This module provides the HMM implementation only.
    # Please use scripts in the `code/` folder (e.g., `final_comparison.py`)
    # or the notebook `code/weather_hmm_analysis.ipynb` to train using
    # the dataset in `data/seattle-weather.csv`.
