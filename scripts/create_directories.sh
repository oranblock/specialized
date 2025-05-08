#!/bin/bash
# Create all necessary directories for the Forex Spike Predictor system

BASE_DIR="/home/clouduser/spike/forex_spike_predictor/specialized"

# Create main directories
mkdir -p "$BASE_DIR/data/raw/EUR_USD/5m"
mkdir -p "$BASE_DIR/data/analyzed/EUR_USD/5m/minimum"
mkdir -p "$BASE_DIR/data/analyzed/EUR_USD/5m/recommended"
mkdir -p "$BASE_DIR/data/analyzed/EUR_USD/5m/optimal"
mkdir -p "$BASE_DIR/data/predictions/EUR_USD/5m"
mkdir -p "$BASE_DIR/data/performance/EUR_USD/5m"

# Create component-specific directories
mkdir -p "$BASE_DIR/data_collector/logs"
mkdir -p "$BASE_DIR/data_analyzer/logs"
mkdir -p "$BASE_DIR/prediction_engine/logs"
mkdir -p "$BASE_DIR/dashboard/logs"

echo "Directory structure created successfully."