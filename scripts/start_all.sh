#!/bin/bash
# Master script to start all components of the Forex Spike Predictor

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="/home/clouduser/spike/forex_spike_predictor/specialized"

# Create shared data directory if it doesn't exist
mkdir -p "$BASE_DIR/shared_data"

echo "=========================================="
echo "Starting Forex Spike Predictor System"
echo "=========================================="

# Start components in the proper order
echo "Step 1/4: Starting Data Collector"
bash "$SCRIPTS_DIR/start_collector.sh"
sleep 2

echo "Step 2/4: Starting Data Analyzer"
bash "$SCRIPTS_DIR/start_analyzer.sh"
sleep 2

echo "Step 3/4: Starting Prediction Engine"
bash "$SCRIPTS_DIR/start_predictor.sh"
sleep 2

echo "Step 4/4: Starting Dashboard"
bash "$SCRIPTS_DIR/start_dashboard.sh"

echo ""
echo "=========================================="
echo "All components started successfully!"
echo "View the dashboard at: http://localhost:8501"
echo "Check individual logs in each component's logs directory"
echo "=========================================="