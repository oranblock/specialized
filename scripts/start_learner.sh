#!/bin/bash
# Start the Continuous Learner component

# Navigate to the component directory
cd /home/clouduser/spike/forex_spike_predictor/specialized

# Create logs directory if it doesn't exist
mkdir -p logs

# Start the learner with python
echo "Starting Continuous ML Learner..."
nohup python -m scripts.continuous_learner --interval 12 > logs/learner_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Save the PID
echo $! > .learner.pid
echo "Continuous Learner started with PID: $!"