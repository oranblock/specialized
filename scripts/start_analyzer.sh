#!/bin/bash
# Start the Data Analyzer component

# Navigate to the component directory
cd /home/clouduser/spike/forex_spike_predictor/specialized/data_analyzer

# Create logs directory if it doesn't exist
mkdir -p logs

# Start the analyzer with python
echo "Starting Forex Data Analyzer..."
python -m src.analyzer --continuous > logs/analyzer_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Save the PID
echo $! > .analyzer.pid
echo "Data Analyzer started with PID: $!"