# Specialized Forex Spike Predictor

A modular, component-based system for forex pattern detection and prediction.

## System Architecture

This system consists of four specialized components:

1. **Data Collector**: Reliably gathers forex price data from external APIs
2. **Data Analyzer**: Processes raw data and applies technical analysis
3. **Prediction Engine**: Generates price movement forecasts using multiple methods
4. **Dashboard**: Visualizes data, patterns, and predictions

## Directory Structure

```
specialized/
├── data_collector/         # Component for reliable API data collection
├── data_analyzer/          # Component for technical analysis and pattern detection
├── prediction_engine/      # Component for generating forecasts
├── dashboard/              # Streamlit interface for visualization
├── shared_data/            # Shared data directory for component communication
├── scripts/                # Scripts for starting, stopping, and monitoring the system
└── system_design.json      # Comprehensive system design document
```

## Getting Started

### Prerequisites

- Python 3.8+
- TA-Lib
- Pandas
- Numpy
- Streamlit
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

3. Monitor system status:
   ```
   ./specialized/scripts/monitor.sh
   ```

4. Stop all components:
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
- Generates price movement forecasts
- Implements multiple prediction methods
- Tracks prediction accuracy
- Calculates pip targets and risk/reward ratios

### Dashboard
- Provides real-time data visualization
- Shows detected patterns and predictions
- Displays performance metrics
- Supports comparison between different analysis depths