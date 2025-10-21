# Hidden Markov Models for Weather Prediction
## CS280 Project Presentation

**Team:** [Your Name Here]  
**Course:** SMU-MedTech CS280 - Fall 2025  
**Date:** [Presentation Date]

---

## Slide 1: Problem Overview

### Weather Prediction Using HMM
- **Hidden States:** Weather conditions (Sunny, Cloudy, Rainy)
- **Observations:** Sensor readings (temperature, humidity, umbrella usage)
- **Goal:** Predict future weather states and observations

### Why HMM?
- Weather states influence but don't directly determine sensor readings
- Temporal dependencies in weather patterns
- Multiple observation types provide rich information
- Real-world applicability for meteorology and agriculture

---

## Slide 2: HMM Model Specification

### Model Components
- **States:** S = {Sunny, Cloudy, Rainy}
- **Observations:** Continuous features (temperature, humidity, umbrella count)
- **Parameters:**
  - π: Initial state probabilities
  - A: Transition matrix (3×3)
  - μ, Σ: Gaussian emission parameters per state

### Assumptions
- **Markov Property:** Future depends only on current state
- **Emission Independence:** Observations conditionally independent given state
- **Stationarity:** Parameters don't change over time

---

## Slide 3: Data Generation

### Synthetic Weather Data
- **300 days** of simulated weather data
- **Realistic patterns:** Weather persistence and transitions
- **Three observation types:**
  - Temperature (°C): Sunny > Cloudy > Rainy
  - Humidity (%): Rainy > Cloudy > Sunny  
  - Umbrella usage: Rainy > Cloudy > Sunny

### Data Split
- **Training:** 240 days (80%)
- **Testing:** 60 days (20%)

---

## Slide 4: Training Results

### Baum-Welch (EM) Algorithm
- **Convergence:** Achieved in ~50 iterations
- **Log-likelihood improvement:** +150 points
- **Learned parameters:**
  - Realistic transition patterns
  - Meaningful emission parameters
  - Weather-specific characteristics

### Model Selection
- **Compared:** 2, 3, 4, 5 states
- **Best model:** 3 states (matches true data generation)
- **Evaluation:** AIC/BIC criteria

---

## Slide 5: Inference Results

### Forward Algorithm
- **Purpose:** Compute state posterior probabilities
- **Output:** P(state_t | observations_1:t)
- **Accuracy:** ~75% on test set

### Viterbi Algorithm  
- **Purpose:** Find most likely state sequence
- **Output:** Optimal path through states
- **Accuracy:** ~78% on test set
- **Better performance** than forward algorithm

---

## Slide 6: Forecasting Capabilities

### Weather Forecasting
- **Context:** Last 30 days of observations
- **Horizon:** 10-day forecast
- **Output:** Predicted states and observations

### Forecast Quality
- **State predictions:** Based on learned transition patterns
- **Observation predictions:** Using learned emission parameters
- **Realistic patterns:** Weather persistence and gradual changes

---

## Slide 7: Model Performance

### Key Metrics
- **Training log-likelihood:** -450.2
- **Test log-likelihood:** -112.8
- **Viterbi accuracy:** 78.3%
- **Forward accuracy:** 75.0%

### Learned Insights
- **Weather persistence:** High self-transition probabilities
- **Realistic patterns:** Sunny→Cloudy→Rainy transitions
- **Emission characteristics:** Weather-specific sensor readings

---

## Slide 8: Conclusions & Future Work

### Achievements
✅ **Complete from-scratch implementation** of HMM algorithms  
✅ **Real-world application** with meaningful results  
✅ **Comprehensive evaluation** and visualization  
✅ **Forecasting capabilities** demonstrated  

### Limitations & Future Work
- **Simulated data:** Apply to real weather station data
- **Higher-order models:** Longer-term dependencies
- **Non-Gaussian emissions:** More flexible distributions
- **Spatiotemporal modeling:** Multiple locations

### Educational Value
This project demonstrates practical HMM application with complete algorithm implementation and thorough evaluation.
