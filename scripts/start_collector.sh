#!/bin/bash
# Start the Data Collector component

# Navigate to the component directory
cd /home/clouduser/spike/forex_spike_predictor/specialized/data_collector

# Create logs directory if it doesn't exist
mkdir -p logs

# Create necessary data directories
mkdir -p ../data/raw/EUR_USD/5m

# Load API key from config
API_KEY=$(grep -o '"polygon_api_key": "[^"]*"' ../config/settings.json | cut -d'"' -f4)

if [ -z "$API_KEY" ] || [ "$API_KEY" = "YOUR_API_KEY_HERE" ]; then
    echo "ERROR: Polygon API key not found in config or using default value."
    echo "Please update your API key in /home/clouduser/spike/forex_spike_predictor/specialized/config/settings.json"
    exit 1
fi

# Start the collector with python
echo "Starting Forex Data Collector..."
python -m src.collector --api-key "$API_KEY" --symbol EUR_USD --timeframe 5m \
    --data-dir ../data/raw > logs/collector_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Save the PID
echo $! > .collector.pid
echo "Data Collector started with PID: $!"