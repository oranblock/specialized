#!/bin/bash
# Start the Prediction Engine component

# Navigate to the component directory
cd /home/clouduser/spike/forex_spike_predictor/specialized/prediction_engine

# Create logs directory if it doesn't exist
mkdir -p logs

# Start the predictor with python
echo "Starting Forex Prediction Engine..."
python -m src.predictor --continuous > logs/predictor_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Save the PID
echo $! > .predictor.pid
echo "Prediction Engine started with PID: $!"