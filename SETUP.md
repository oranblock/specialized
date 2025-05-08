# Forex Spike Predictor - Setup Guide

This document provides detailed setup instructions for the Specialized Forex Spike Predictor system.

## Installation Requirements

1. Install required system packages:
   ```bash
   sudo apt-get update
   sudo apt-get install -y build-essential python3-dev python3-pip
   ```

2. Install TA-Lib dependencies:
   ```bash
   sudo apt-get install -y wget
   wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
   tar -xzf ta-lib-0.4.0-src.tar.gz
   cd ta-lib/
   ./configure --prefix=/usr
   make
   sudo make install
   cd ..
   rm -rf ta-lib-0.4.0-src.tar.gz ta-lib/
   ```

3. Install Python packages:
   ```bash
   pip install numpy pandas TA-Lib polygon-api-client streamlit plotly scikit-learn joblib
   ```

## Configuration

1. Create a configuration file:
   ```bash
   mkdir -p /home/clouduser/spike/forex_spike_predictor/specialized/config
   ```

2. Add your Polygon.io API key to the configuration:
   ```bash
   echo '{
     "polygon_api_key": "YOUR_API_KEY_HERE",
     "currency_pairs": ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD"],
     "collection_interval_seconds": 60,
     "analysis_interval_seconds": 300,
     "prediction_interval_seconds": 600,
     "shared_data_path": "/home/clouduser/spike/forex_spike_predictor/specialized/shared_data",
     "log_level": "INFO"
   }' > /home/clouduser/spike/forex_spike_predictor/specialized/config/settings.json
   ```

3. Replace `YOUR_API_KEY_HERE` with your actual Polygon.io API key.

## Component Setup

Each component of the system needs to be properly initialized:

1. **Data Collector Setup**:
   ```bash
   mkdir -p /home/clouduser/spike/forex_spike_predictor/specialized/data_collector/logs
   mkdir -p /home/clouduser/spike/forex_spike_predictor/specialized/data_collector/cache
   ```

2. **Data Analyzer Setup**:
   ```bash
   mkdir -p /home/clouduser/spike/forex_spike_predictor/specialized/data_analyzer/logs
   mkdir -p /home/clouduser/spike/forex_spike_predictor/specialized/data_analyzer/outputs
   ```

3. **Prediction Engine Setup**:
   ```bash
   mkdir -p /home/clouduser/spike/forex_spike_predictor/specialized/prediction_engine/logs
   mkdir -p /home/clouduser/spike/forex_spike_predictor/specialized/prediction_engine/models
   ```

4. **Dashboard Setup**:
   ```bash
   mkdir -p /home/clouduser/spike/forex_spike_predictor/specialized/dashboard/logs
   mkdir -p /home/clouduser/spike/forex_spike_predictor/specialized/dashboard/assets
   ```

5. **Shared Data Directory**:
   ```bash
   mkdir -p /home/clouduser/spike/forex_spike_predictor/specialized/shared_data
   ```

## Service Setup (Optional)

For automatic startup as system services, you can create systemd service files:

```bash
sudo bash -c 'cat > /etc/systemd/system/forex-collector.service << EOL
[Unit]
Description=Forex Data Collector Service
After=network.target

[Service]
User=clouduser
WorkingDirectory=/home/clouduser/spike/forex_spike_predictor/specialized
ExecStart=/bin/bash /home/clouduser/spike/forex_spike_predictor/specialized/scripts/start_collector.sh
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOL'
```

Create similar service files for analyzer, predictor, and dashboard components.

Then enable and start the services:

```bash
sudo systemctl daemon-reload
sudo systemctl enable forex-collector.service
sudo systemctl start forex-collector.service
# Repeat for other components
```

## Testing the Setup

1. Run the monitor script to check if everything is configured correctly:
   ```bash
   ./specialized/scripts/monitor.sh
   ```

2. Start the system manually:
   ```bash
   ./specialized/scripts/start_all.sh
   ```

3. Visit the dashboard at http://localhost:8501

4. Check component logs:
   ```bash
   tail -f /home/clouduser/spike/forex_spike_predictor/specialized/*/logs/*.log
   ```

## Troubleshooting

1. If TA-Lib installation fails, try the pre-built wheels:
   ```bash
   pip install --no-cache-dir ta-lib
   ```

2. If a component fails to start, check its log file for specific errors.

3. Ensure your Polygon.io API key is valid and has access to Forex data.

4. If components can't communicate, ensure the shared_data directory has proper permissions.