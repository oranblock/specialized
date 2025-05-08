# Forex Spike Predictor 📈

A multi-component, ML-enhanced system for forex pattern detection and price movement prediction.

## System Architecture

This system consists of five specialized components:

1. **Data Collector**: Reliably gathers forex price data from external APIs
2. **Data Analyzer**: Processes raw data and applies technical analysis
3. **Prediction Engine**: Generates price movement forecasts using multiple methods including ML
4. **Dashboard**: Visualizes data, patterns, and predictions with ML indicators
5. **Continuous Learner**: Silently improves ML models based on performance feedback

## Key Features

### 🧠 Hybrid Prediction Methodology

Predictions are generated using a weighted ensemble of:

- **Technical Analysis**: MACD, RSI, Bollinger Bands, Moving Averages, etc.
- **Pattern Recognition**: Candlestick patterns (doji, engulfing, hammers, etc.)
- **Machine Learning**: Random Forest and XGBoost models trained on historical data

### 🤖 Machine Learning with Visual Indicators

ML-detected patterns are highlighted in the dashboard:

- **Robot Icons** (🤖) appear next to patterns identified by ML models
- **Blue Triangle Markers** indicate points of high-confidence ML predictions
- **Special Highlighting** for patterns that coincide with ML signals

### 📊 Interactive Real-Time Dashboard

Streamlit-based dashboard shows:

- **Price Charts** with multiple timeframes and indicators
- **Pattern Detection** with visual annotations
- **Prediction Cards** showing expected price movements
- **Performance Metrics** tracking prediction accuracy
- **Real-Time Updates** with anti-flickering technology

### 🔄 Continuous Learning

The system includes a background process that:

- Evaluates prediction performance
- Retrains ML models with successful patterns
- Improves over time without manual intervention
- Updates models without disrupting the main system

## Directory Structure

```
specialized/
├── data_collector/         # Component for reliable API data collection
├── data_analyzer/          # Component for technical analysis and pattern detection
├── prediction_engine/      # Component for generating forecasts with ML
├── dashboard/              # Streamlit interface with ML visualization
├── models/                 # Machine learning models and metadata
├── scripts/                # Scripts for starting, stopping, and monitoring
│   └── continuous_learner.py  # ML model improvement system
└── system_design.json      # Comprehensive system design document
```

## Getting Started

### Prerequisites

- Python 3.8+
- TA-Lib
- Pandas, Numpy, Scikit-learn, XGBoost
- Streamlit, Plotly
- Polygon.io API key

### Running the System

1. Make all scripts executable:
   ```
   chmod +x specialized/scripts/*.sh
   ```

2. Start all components:
   ```
   ./specialized/scripts/start_all.sh
   ```

3. Start the continuous learner (optional but recommended):
   ```
   ./specialized/scripts/start_learner.sh
   ```

4. Monitor system status:
   ```
   ./specialized/scripts/monitor.sh
   ```

5. Stop all components:
   ```
   ./specialized/scripts/stop_all.sh
   ```

### Accessing the Dashboard

When the system is running, the dashboard is available at:
http://localhost:8501

## Analysis Depths

The system supports three analysis depths:

1. **Minimum** (2 hours): Basic pattern detection with limited indicators
2. **Recommended** (4 hours): Comprehensive technical analysis with moderate history
3. **Optimal** (8 hours): Advanced analysis with extended history for best predictions

## Component Details

### Data Collector
- Fetches real-time forex data from Polygon.io
- Implements robust error handling and reconnection
- Stores data in CSV format for persistence
- Supports multiple currency pairs

### Data Analyzer
- Processes raw data into technical indicators
- Detects common candlestick patterns using TA-Lib
- Calculates support and resistance levels
- Analyzes at three different time depths

### Prediction Engine
- Generates price movement forecasts using ensemble methods
- Combines technical analysis, pattern recognition, and ML predictions
- Tracks prediction accuracy and performance metrics
- Calculates pip targets and risk/reward ratios
- Features ML model adaptation for various input formats

### Dashboard
- Provides real-time data visualization
- Shows detected patterns and predictions with ML indicators
- Displays performance metrics for all prediction methods
- Supports comparison between different analysis depths
- Highlights ML-detected patterns with robot icons

### Continuous Learner
- Runs in the background to evaluate prediction performance
- Retrains ML models based on successful predictions
- Adapts to changing market conditions automatically
- Silently improves the system's prediction accuracy over time

## ML Model Details

### Available Models

- **Random Forest**: Classification model for binary (UP/DOWN) prediction
- **XGBoost**: Gradient boosting model with feature importance analysis

Both models are continuously improved through a feedback loop that evaluates prediction success and adjusts feature weights accordingly.

### ML Visual Indicators

The dashboard includes special visual indicators for ML-based patterns:

- **Robot Icons** (🤖) appear next to patterns identified by ML 
- **Blue Triangle Markers** show points of high-confidence ML predictions
- **Special Highlighting** for patterns that coincide with ML signals