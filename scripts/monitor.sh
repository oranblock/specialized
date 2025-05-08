#!/bin/bash
# Monitor script to check the status of all components

BASE_DIR="/home/clouduser/spike/forex_spike_predictor/specialized"
COMPONENTS=("data_collector" "data_analyzer" "prediction_engine" "dashboard")

echo "=========================================="
echo "Forex Spike Predictor System Status"
echo "=========================================="
echo "Time: $(date)"
echo ""

# Check shared data directory
echo "Shared Data Status:"
DATA_FILES=$(find "$BASE_DIR/shared_data" -type f | wc -l)
echo "- Files in shared data directory: $DATA_FILES"
if [ -f "$BASE_DIR/shared_data/latest_raw_data.csv" ]; then
    LAST_UPDATE=$(stat -c %y "$BASE_DIR/shared_data/latest_raw_data.csv" | cut -d. -f1)
    echo "- Last data update: $LAST_UPDATE"
fi
echo ""

# Check component status
echo "Component Status:"
for component in "${COMPONENTS[@]}"; do
    # Correct PID file path based on component name
    case "$component" in
        "data_collector") pid_file="$BASE_DIR/$component/.collector.pid" ;;
        "data_analyzer") pid_file="$BASE_DIR/$component/.analyzer.pid" ;;
        "prediction_engine") pid_file="$BASE_DIR/$component/.predictor.pid" ;;
        "dashboard") pid_file="$BASE_DIR/$component/.dashboard.pid" ;;
        *) pid_file="$BASE_DIR/$component/.${component##*/}.pid" ;;
    esac
    
    echo -n "- $component: "
    if [ -f "$pid_file" ]; then
        PID=$(cat "$pid_file")
        if ps -p "$PID" > /dev/null; then
            echo "RUNNING (PID: $PID)"
            
            # Check log file for recent activity
            LOG_DIR="$BASE_DIR/$component/logs"
            if [ -d "$LOG_DIR" ]; then
                LATEST_LOG=$(ls -t "$LOG_DIR" | head -1)
                if [ -n "$LATEST_LOG" ]; then
                    LAST_ACTIVITY=$(stat -c %y "$LOG_DIR/$LATEST_LOG" | cut -d. -f1)
                    echo "  Last activity: $LAST_ACTIVITY"
                    
                    # Show recent errors if any
                    ERRORS=$(grep -i "error\|exception\|failed" "$LOG_DIR/$LATEST_LOG" | tail -3)
                    if [ -n "$ERRORS" ]; then
                        echo "  Recent errors:"
                        echo "$ERRORS" | sed 's/^/    /'
                    fi
                fi
            fi
        else
            echo "NOT RUNNING (stale PID file)"
        fi
    else
        echo "NOT RUNNING"
    fi
done

echo ""
echo "=========================================="
if [ -f "$BASE_DIR/dashboard/.dashboard.pid" ]; then
    PID=$(cat "$BASE_DIR/dashboard/.dashboard.pid")
    if ps -p "$PID" > /dev/null; then
        echo "Dashboard is available at: http://localhost:8501"
    fi
fi
echo "=========================================="