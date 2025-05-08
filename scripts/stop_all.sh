#!/bin/bash
# Script to stop all components of the Forex Spike Predictor

BASE_DIR="/home/clouduser/spike/forex_spike_predictor/specialized"
COMPONENTS=("data_collector" "data_analyzer" "prediction_engine" "dashboard")

echo "=========================================="
echo "Stopping Forex Spike Predictor System"
echo "=========================================="

# Function to stop a component with a PID file
stop_component() {
    local component=$1
    local pid_file="$BASE_DIR/$component/.${component##*/}.pid"
    
    if [ -f "$pid_file" ]; then
        PID=$(cat "$pid_file")
        echo "Stopping $component (PID: $PID)..."
        
        if kill -15 "$PID" 2>/dev/null; then
            echo "$component stopped successfully"
            rm "$pid_file"
        else
            echo "Warning: $component process not found, may have already stopped"
            rm "$pid_file"
        fi
    else
        echo "No PID file found for $component, may not be running"
    fi
}

# Stop components in reverse order
for ((i=${#COMPONENTS[@]}-1; i>=0; i--)); do
    stop_component "${COMPONENTS[$i]}"
    sleep 1
done

echo ""
echo "=========================================="
echo "All components stopped successfully!"
echo "=========================================="