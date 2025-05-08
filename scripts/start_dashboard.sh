#!/bin/bash
# Start the Dashboard component

# Navigate to the component directory
cd /home/clouduser/spike/forex_spike_predictor/specialized/dashboard

# Create logs directory if it doesn't exist
mkdir -p logs

# Start the Streamlit dashboard
echo "Starting Forex Dashboard..."
streamlit run src/dashboard.py > logs/dashboard_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Save the PID
echo $! > .dashboard.pid
echo "Dashboard started with PID: $!"
echo "Dashboard URL: http://localhost:8501"