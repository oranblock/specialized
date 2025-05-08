#!/usr/bin/env python
"""
Forex Dashboard for Multi-Component Pattern Prediction System

This module provides a Streamlit-based dashboard for monitoring patterns,
predictions, and performance metrics from the prediction system.
"""

import os
import sys
import argparse
import logging
import json
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import threading
import queue

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("dashboard.log")
    ]
)
logger = logging.getLogger("Dashboard")

# Data update queue for thread-safe communication
data_update_queue = queue.Queue()

class ForexDashboard:
    """Streamlit dashboard for forex pattern prediction system."""
    
    def __init__(self, symbol, timeframe='5m', data_dir=None):
        """
        Initialize the dashboard.
        
        Args:
            symbol: Currency pair (format: XXX_YYY, e.g., EUR_USD)
            timeframe: Data timeframe (default: 5m)
            data_dir: Root directory for data
        """
        self.symbol = symbol
        self.timeframe = timeframe
        
        # Set up data directory
        if data_dir is None:
            self.data_dir = os.path.abspath(os.path.join(
                os.path.dirname(__file__), '..', '..', 'data'
            ))
        else:
            self.data_dir = os.path.abspath(data_dir)
        
        # Data paths
        self.raw_data_dir = os.path.join(self.data_dir, 'raw', self.symbol, self.timeframe)
        self.analyzed_data_dirs = {
            'minimum': os.path.join(self.data_dir, 'analyzed', self.symbol, self.timeframe, 'minimum'),
            'recommended': os.path.join(self.data_dir, 'analyzed', self.symbol, self.timeframe, 'recommended'),
            'optimal': os.path.join(self.data_dir, 'analyzed', self.symbol, self.timeframe, 'optimal')
        }
        self.prediction_dir = os.path.join(self.data_dir, 'predictions', self.symbol, self.timeframe)
        self.performance_dir = os.path.join(self.data_dir, 'performance', self.symbol, self.timeframe)
        
        # Ensure directories exist
        for directory in [self.raw_data_dir, *self.analyzed_data_dirs.values(), 
                         self.prediction_dir, self.performance_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Cache data
        self.raw_data = None
        self.analyzed_data = {depth: None for depth in self.analyzed_data_dirs}
        self.pattern_data = {depth: None for depth in self.analyzed_data_dirs}
        self.predictions = None
        self.performance = None
        self.last_update_time = {
            'raw': None,
            'analyzed': {depth: None for depth in self.analyzed_data_dirs},
            'predictions': None,
            'performance': None
        }
        
        logger.info(f"Initialized dashboard for {symbol} with {timeframe} timeframe")
        logger.info(f"Data directory: {self.data_dir}")
    
    def find_latest_file(self, directory, pattern):
        """
        Find the latest file matching a pattern in a directory.
        
        Args:
            directory: Directory to search
            pattern: File pattern to match
            
        Returns:
            Path to the latest file or None if not found
        """
        if not os.path.exists(directory):
            logger.warning(f"Directory not found: {directory}")
            return None
            
        files = [f for f in os.listdir(directory) if pattern in f]
        if not files:
            logger.warning(f"No files matching '{pattern}' found in {directory}")
            return None
            
        # Sort files by name (which includes date)
        files.sort(reverse=True)
        latest_file = os.path.join(directory, files[0])
        return latest_file
    
    def load_raw_data(self, force=False):
        """
        Load latest raw price data.
        
        Args:
            force: Force reload even if recently loaded
            
        Returns:
            DataFrame with raw price data
        """
        # Anti-flickering: ALWAYS keep existing data available until new data is ready
        existing_data = None
        
        # First try to get from session state cache for immediate display
        if 'cached_data' in st.session_state and st.session_state.data_loaded and st.session_state.cached_data['raw'] is not None:
            existing_data = st.session_state.cached_data['raw']
            
            # Return cached data immediately to prevent flicker if not forcing reload
            if not force:
                self.raw_data = existing_data
                return self.raw_data
        # Otherwise use instance data if available
        elif self.raw_data is not None:
            existing_data = self.raw_data
            
            # Check freshness
            if not force and self.last_update_time['raw'] is not None:
                if (datetime.now() - self.last_update_time['raw']).total_seconds() < 60:
                    # Data is fresh enough
                    return self.raw_data
        
        try:
            # Find latest file
            latest_file = self.find_latest_file(self.raw_data_dir, '.csv')
            if latest_file is None:
                # Return existing data if available
                if existing_data is not None:
                    return existing_data
                return None
                
            # Check file modification time
            last_modified = datetime.fromtimestamp(os.path.getmtime(latest_file))
            if not force and self.last_update_time['raw'] is not None:
                if last_modified <= self.last_update_time['raw']:
                    # File hasn't changed - return existing data
                    if existing_data is not None:
                        return existing_data
                    return self.raw_data
            
            logger.info(f"Loading raw data from {latest_file}")
            df = pd.read_csv(latest_file)
            
            # Ensure timestamp is datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            logger.info(f"Loaded {len(df)} records from {latest_file}")
            
            # Update cache
            self.raw_data = df
            self.last_update_time['raw'] = datetime.now()
            
            # Update session state cache if possible
            if 'cached_data' in st.session_state:
                st.session_state.cached_data['raw'] = df
                st.session_state.data_loaded = True
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading raw data: {str(e)}")
            # Return existing data rather than None if possible
            if existing_data is not None:
                return existing_data
            return None
    
    def load_analyzed_data(self, depth, force=False):
        """
        Load latest analyzed data for a specific depth.
        
        Args:
            depth: Analysis depth (minimum, recommended, optimal)
            force: Force reload even if recently loaded
            
        Returns:
            DataFrame with analyzed data
        """
        # Anti-flickering: ALWAYS keep existing data available until new data is ready
        existing_data = None
        
        # First try to get from session state cache for immediate display
        if 'cached_data' in st.session_state and st.session_state.data_loaded:
            if depth in st.session_state.cached_data.get('analyzed', {}) and st.session_state.cached_data['analyzed'][depth] is not None:
                existing_data = st.session_state.cached_data['analyzed'][depth]
                
                # Return cached data immediately to prevent flicker if not forcing reload
                if not force:
                    self.analyzed_data[depth] = existing_data
                    return self.analyzed_data[depth]
        # Otherwise use instance data if available
        elif self.analyzed_data[depth] is not None:
            existing_data = self.analyzed_data[depth]
            
            # Check freshness
            if not force and self.last_update_time['analyzed'][depth] is not None:
                if (datetime.now() - self.last_update_time['analyzed'][depth]).total_seconds() < 60:
                    # Data is fresh enough
                    return self.analyzed_data[depth]
        
        try:
            # Find latest file
            latest_file = self.find_latest_file(self.analyzed_data_dirs[depth], '_analyzed.csv')
            if latest_file is None:
                # Return existing data if available
                if existing_data is not None:
                    return existing_data
                return None
                
            # Check file modification time
            last_modified = datetime.fromtimestamp(os.path.getmtime(latest_file))
            if not force and self.last_update_time['analyzed'][depth] is not None:
                if last_modified <= self.last_update_time['analyzed'][depth]:
                    # File hasn't changed - return existing data
                    if existing_data is not None:
                        return existing_data
                    return self.analyzed_data[depth]
            
            logger.info(f"Loading analyzed data for {depth} from {latest_file}")
            df = pd.read_csv(latest_file)
            
            # Ensure timestamp is datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            logger.info(f"Loaded {len(df)} records from {latest_file}")
            
            # Update cache
            self.analyzed_data[depth] = df
            self.last_update_time['analyzed'][depth] = datetime.now()
            
            # Update session state cache
            if 'cached_data' in st.session_state:
                if 'analyzed' not in st.session_state.cached_data:
                    st.session_state.cached_data['analyzed'] = {}
                st.session_state.cached_data['analyzed'][depth] = df
                st.session_state.data_loaded = True
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading analyzed data for {depth}: {str(e)}")
            # Return existing data rather than None if possible
            if existing_data is not None:
                return existing_data
            return None
    
    def load_pattern_data(self, depth, force=False):
        """
        Load latest pattern recognition data for a specific depth.
        
        Args:
            depth: Analysis depth (minimum, recommended, optimal)
            force: Force reload even if recently loaded
            
        Returns:
            Dictionary with pattern data
        """
        # First check session state cache for immediate display
        if not force and 'cached_data' in st.session_state and st.session_state.data_loaded:
            if depth in st.session_state.cached_data.get('patterns', {}) and st.session_state.cached_data['patterns'][depth] is not None:
                # Return cached data immediately to prevent flicker
                self.pattern_data[depth] = st.session_state.cached_data['patterns'][depth]
                return self.pattern_data[depth]
                
        # Check if we need to reload
        if not force and self.pattern_data[depth] is not None and self.last_update_time['analyzed'][depth] is not None:
            if (datetime.now() - self.last_update_time['analyzed'][depth]).total_seconds() < 60:
                # Data is fresh enough
                return self.pattern_data[depth]
        
        try:
            # Find latest file
            latest_file = self.find_latest_file(self.analyzed_data_dirs[depth], '_patterns.json')
            if latest_file is None:
                return None
                
            # Check file modification time
            last_modified = datetime.fromtimestamp(os.path.getmtime(latest_file))
            if not force and self.last_update_time['analyzed'][depth] is not None:
                if last_modified <= self.last_update_time['analyzed'][depth]:
                    # File hasn't changed
                    return self.pattern_data[depth]
            
            logger.info(f"Loading pattern data for {depth} from {latest_file}")
            with open(latest_file, 'r') as f:
                patterns_data = json.load(f)
            
            # Properly process the new pattern format
            # The top level object has pattern array, support and resistance levels
            if 'patterns' in patterns_data and isinstance(patterns_data['patterns'], list):
                # Convert to the format expected by the dashboard
                patterns = {}
                for pattern in patterns_data['patterns']:
                    pattern_name = pattern['pattern']
                    if pattern_name not in patterns:
                        patterns[pattern_name] = {
                            'instances': [],
                            'count': 0
                        }
                    
                    # Determine pattern type
                    if pattern_name.startswith('BULLISH') or pattern_name in ['HAMMER', 'MORNING_STAR', 'PIERCING']:
                        pattern_type = 'bullish'
                    else:
                        pattern_type = 'bearish'
                    
                    # Add instance
                    patterns[pattern_name]['instances'].append({
                        'timestamp': pattern['time'],
                        'type': pattern_type,
                        'strength': pattern['strength'],
                        'high': 0.0,  # Not provided in new format
                        'low': 0.0    # Not provided in new format
                    })
                    patterns[pattern_name]['count'] = len(patterns[pattern_name]['instances'])
            else:
                # Fallback to old format or empty dict
                patterns = patterns_data
            
            pattern_count = sum(p['count'] for p in patterns.values()) if patterns else 0
            logger.info(f"Loaded {len(patterns)} pattern types with {pattern_count} instances from {latest_file}")
            
            # Update cache
            self.pattern_data[depth] = patterns
            self.last_update_time['analyzed'][depth] = datetime.now()
            
            # Update session state cache
            if 'cached_data' in st.session_state:
                if 'patterns' not in st.session_state.cached_data:
                    st.session_state.cached_data['patterns'] = {}
                st.session_state.cached_data['patterns'][depth] = patterns
                st.session_state.data_loaded = True
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error loading pattern data for {depth}: {str(e)}")
            # If we have cached data, return it as fallback
            if 'cached_data' in st.session_state and 'patterns' in st.session_state.cached_data and depth in st.session_state.cached_data['patterns']:
                return st.session_state.cached_data['patterns'][depth]
            return None
    
    def load_predictions(self, force=False):
        """
        Load latest prediction data.
        
        Args:
            force: Force reload even if recently loaded
            
        Returns:
            Dictionary with prediction data
        """
        # First check session state cache for immediate display
        if not force and 'cached_data' in st.session_state and st.session_state.data_loaded:
            if st.session_state.cached_data['predictions'] is not None:
                # Return cached data immediately to prevent flicker
                self.predictions = st.session_state.cached_data['predictions']
                return self.predictions
                
        # Check if we need to reload
        if not force and self.predictions is not None and self.last_update_time['predictions'] is not None:
            if (datetime.now() - self.last_update_time['predictions']).total_seconds() < 60:
                # Data is fresh enough
                return self.predictions
        
        try:
            # Find latest file
            latest_file = self.find_latest_file(self.prediction_dir, '_predictions.json')
            if latest_file is None:
                return None
                
            # Check file modification time
            last_modified = datetime.fromtimestamp(os.path.getmtime(latest_file))
            if not force and self.last_update_time['predictions'] is not None:
                if last_modified <= self.last_update_time['predictions']:
                    # File hasn't changed
                    return self.predictions
            
            logger.info(f"Loading predictions from {latest_file}")
            with open(latest_file, 'r') as f:
                predictions = json.load(f)
            
            logger.info(f"Loaded {len(predictions.get('predictions', []))} predictions from {latest_file}")
            
            # Update cache
            self.predictions = predictions
            self.last_update_time['predictions'] = datetime.now()
            
            # Update session state cache
            if 'cached_data' in st.session_state:
                st.session_state.cached_data['predictions'] = predictions
                st.session_state.data_loaded = True
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error loading predictions: {str(e)}")
            # If we have cached data, return it as fallback
            if 'cached_data' in st.session_state and st.session_state.cached_data['predictions'] is not None:
                return st.session_state.cached_data['predictions']
            return None
    
    def load_performance(self, force=False):
        """
        Load latest performance metrics.
        
        Args:
            force: Force reload even if recently loaded
            
        Returns:
            Dictionary with performance metrics
        """
        # First check session state cache for immediate display
        if not force and 'cached_data' in st.session_state and st.session_state.data_loaded:
            if st.session_state.cached_data['performance'] is not None:
                # Return cached data immediately to prevent flicker
                self.performance = st.session_state.cached_data['performance']
                return self.performance
                
        # Check if we need to reload
        if not force and self.performance is not None and self.last_update_time['performance'] is not None:
            if (datetime.now() - self.last_update_time['performance']).total_seconds() < 60:
                # Data is fresh enough
                return self.performance
        
        try:
            # Find latest file - FIXED: Use performance_dir instead of prediction_dir
            latest_file = self.find_latest_file(self.performance_dir, '_performance.json')
            if latest_file is None:
                # Try for latest_performance.json format
                latest_file = self.find_latest_file(self.performance_dir, 'latest_performance.json')
                if latest_file is None:
                    return None
                
            # Check file modification time
            last_modified = datetime.fromtimestamp(os.path.getmtime(latest_file))
            if not force and self.last_update_time['performance'] is not None:
                if last_modified <= self.last_update_time['performance']:
                    # File hasn't changed
                    return self.performance
            
            logger.info(f"Loading performance metrics from {latest_file}")
            with open(latest_file, 'r') as f:
                performance = json.load(f)
            
            logger.info(f"Loaded performance metrics from {latest_file}")
            
            # Update cache
            self.performance = performance
            self.last_update_time['performance'] = datetime.now()
            
            # Update session state cache
            if 'cached_data' in st.session_state:
                st.session_state.cached_data['performance'] = performance
                st.session_state.data_loaded = True
            
            return performance
            
        except Exception as e:
            logger.error(f"Error loading performance metrics: {str(e)}")
            # If we have cached data, return it as fallback
            if 'cached_data' in st.session_state and st.session_state.cached_data['performance'] is not None:
                return st.session_state.cached_data['performance']
            return None
    
    def create_price_chart(self, df, depth=None, patterns=None, predictions=None):
        """
        Create price chart with indicators and patterns.
        
        Args:
            df: DataFrame with price data
            depth: Analysis depth (for title)
            patterns: Dictionary with pattern data
            predictions: Dictionary with prediction data
            
        Returns:
            Plotly figure object
        """
        try:
            if df is None or len(df) < 5:
                fig = go.Figure()
                fig.update_layout(title="No data available")
                return fig
            
            # Create figure
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                               row_heights=[0.7, 0.3], 
                               vertical_spacing=0.05,
                               subplot_titles=["Price", "Indicators"])
            
            # Limit to last 50 candles for clarity
            display_df = df.tail(50).copy()
            
            # Add candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=display_df['timestamp'],
                    open=display_df['open'],
                    high=display_df['high'],
                    low=display_df['low'],
                    close=display_df['close'],
                    name="Price",
                    increasing_line_color='green',
                    decreasing_line_color='red'
                ),
                row=1, col=1
            )
            
            # Add moving averages if available
            for ma_type, color in [('sma_20', 'blue'), ('sma_50', 'orange'), ('ema_20', 'purple')]:
                if ma_type in display_df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=display_df['timestamp'],
                            y=display_df[ma_type],
                            name=ma_type.upper(),
                            line=dict(color=color, width=1)
                        ),
                        row=1, col=1
                    )
            
            # Add Bollinger Bands if available
            if all(col in display_df.columns for col in ['bb_upper', 'bb_middle', 'bb_lower']):
                fig.add_trace(
                    go.Scatter(
                        x=display_df['timestamp'],
                        y=display_df['bb_upper'],
                        name='Upper BB',
                        line=dict(color='rgba(250, 0, 0, 0.3)', width=1)
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=display_df['timestamp'],
                        y=display_df['bb_lower'],
                        name='Lower BB',
                        line=dict(color='rgba(250, 0, 0, 0.3)', width=1),
                        fill='tonexty',
                        fillcolor='rgba(250, 0, 0, 0.05)'
                    ),
                    row=1, col=1
                )
            
            # Add MACD if available
            if all(col in display_df.columns for col in ['macd', 'macd_signal']):
                fig.add_trace(
                    go.Scatter(
                        x=display_df['timestamp'],
                        y=display_df['macd'],
                        name='MACD',
                        line=dict(color='blue', width=1)
                    ),
                    row=2, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=display_df['timestamp'],
                        y=display_df['macd_signal'],
                        name='Signal',
                        line=dict(color='red', width=1)
                    ),
                    row=2, col=1
                )
                
                # Add MACD histogram if available
                if 'macd_hist' in display_df.columns:
                    colors = ['green' if val > 0 else 'red' for val in display_df['macd_hist']]
                    fig.add_trace(
                        go.Bar(
                            x=display_df['timestamp'],
                            y=display_df['macd_hist'],
                            name='Histogram',
                            marker_color=colors,
                            opacity=0.5
                        ),
                        row=2, col=1
                    )
            
            # Add RSI if available and no MACD
            elif 'rsi_14' in display_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=display_df['timestamp'],
                        y=display_df['rsi_14'],
                        name='RSI',
                        line=dict(color='purple', width=1)
                    ),
                    row=2, col=1
                )
                
                # Add RSI levels
                fig.add_shape(
                    type="line",
                    x0=display_df['timestamp'].iloc[0],
                    y0=70,
                    x1=display_df['timestamp'].iloc[-1],
                    y1=70,
                    line=dict(color="red", dash="dash", width=1),
                    row=2, col=1
                )
                
                fig.add_shape(
                    type="line",
                    x0=display_df['timestamp'].iloc[0],
                    y0=30,
                    x1=display_df['timestamp'].iloc[-1],
                    y1=30,
                    line=dict(color="green", dash="dash", width=1),
                    row=2, col=1
                )
            
            # Track ML model predictions for pattern highlights
            ml_prediction_points = []
            
            # Add prediction markers if available - process predictions first to identify ML predictions
            if predictions and 'predictions' in predictions:
                for pred in predictions['predictions'][-5:]:  # Show last 5 predictions
                    try:
                        # Convert timestamp to datetime if it's a string
                        if isinstance(pred['timestamp'], str):
                            timestamp = datetime.fromisoformat(pred['timestamp'].replace('Z', '+00:00'))
                        else:
                            timestamp = pred['timestamp']
                        
                        # Skip if timestamp is not in displayed range
                        if timestamp not in display_df['timestamp'].values:
                            # Just use the latest timestamp with a small offset
                            timestamp = display_df['timestamp'].iloc[-1]
                        
                        # Get prediction details
                        direction = pred['direction']
                        confidence = pred['confidence']
                        
                        # Check if this was an ML prediction
                        if 'signals' in pred and 'ml' in pred['signals']:
                            # This is an ML-based prediction, add to our highlight list
                            ml_direction = pred['signals']['ml']['direction'] 
                            ml_confidence = pred['signals']['ml']['confidence']
                            
                            # Only highlight strong ML signals
                            if ml_confidence > 0.65:
                                # Find the closest price point
                                closest_idx = display_df['timestamp'].searchsorted(timestamp)
                                if closest_idx < len(display_df):
                                    price_point = {
                                        'timestamp': timestamp,
                                        'price': display_df.iloc[closest_idx]['close'],
                                        'direction': ml_direction,
                                        'confidence': ml_confidence
                                    }
                                    ml_prediction_points.append(price_point)
                        
                        # Set color based on direction
                        color = 'green' if direction == 'UP' else 'red' if direction == 'DOWN' else 'gray'
                        
                        # Add take profit and stop loss lines if available
                        if 'take_profit' in pred and 'stop_loss' in pred:
                            take_profit = pred['take_profit']
                            stop_loss = pred['stop_loss']
                            
                            # Add take profit line
                            fig.add_shape(
                                type="line",
                                x0=timestamp,
                                y0=take_profit,
                                x1=display_df['timestamp'].iloc[-1] + pd.Timedelta(minutes=10),
                                y1=take_profit,
                                line=dict(color="green", dash="dash", width=1),
                                row=1, col=1
                            )
                            
                            # Add stop loss line
                            fig.add_shape(
                                type="line",
                                x0=timestamp,
                                y0=stop_loss,
                                x1=display_df['timestamp'].iloc[-1] + pd.Timedelta(minutes=10),
                                y1=stop_loss,
                                line=dict(color="red", dash="dash", width=1),
                                row=1, col=1
                            )
                            
                            # Add annotations
                            fig.add_annotation(
                                x=display_df['timestamp'].iloc[-1] + pd.Timedelta(minutes=5),
                                y=take_profit,
                                text=f"TP: {take_profit:.5f}",
                                showarrow=False,
                                font=dict(size=10, color="green"),
                                row=1, col=1
                            )
                            
                            fig.add_annotation(
                                x=display_df['timestamp'].iloc[-1] + pd.Timedelta(minutes=5),
                                y=stop_loss,
                                text=f"SL: {stop_loss:.5f}",
                                showarrow=False,
                                font=dict(size=10, color="red"),
                                row=1, col=1
                            )
                    
                    except Exception as e:
                        logger.error(f"Error adding prediction marker: {str(e)}")
                        continue
            
            # Add pattern markers if available
            if patterns:
                for pattern_name, pattern_data in patterns.items():
                    for instance in pattern_data['instances']:
                        try:
                            # Convert timestamp to datetime if it's a string
                            if isinstance(instance['timestamp'], str):
                                timestamp = datetime.fromisoformat(instance['timestamp'].replace('Z', '+00:00'))
                            else:
                                timestamp = instance['timestamp']
                            
                            # Check if timestamp is in displayed range
                            if timestamp in display_df['timestamp'].values:
                                # Get pattern type
                                pattern_type = instance['type']
                                # Set color based on type
                                color = 'green' if pattern_type == 'bullish' else 'red'
                                
                                # Check if this pattern is at an ML prediction point
                                is_ml_pattern = False
                                for ml_point in ml_prediction_points:
                                    time_diff = abs((ml_point['timestamp'] - timestamp).total_seconds())
                                    # If the pattern is within 5 minutes of an ML prediction point
                                    if time_diff < 300:
                                        is_ml_pattern = True
                                        break
                                
                                # Add robot icon for ML-detected patterns
                                if is_ml_pattern:
                                    pattern_text = f"🤖 {pattern_name}"
                                    arrow_color = 'blue'  # Special color for ML-detected patterns
                                    bg_color = f"rgba{(*hex_to_rgb('blue'), 0.4)}"
                                else:
                                    pattern_text = pattern_name
                                    arrow_color = color
                                    bg_color = f"rgba{(*hex_to_rgb(color), 0.3)}"
                                
                                # Add annotation
                                fig.add_annotation(
                                    x=timestamp,
                                    y=instance['high'] * 1.001 if pattern_type == 'bullish' else instance['low'] * 0.999,
                                    text=pattern_text,
                                    showarrow=True,
                                    arrowhead=2,
                                    arrowsize=1,
                                    arrowwidth=1,
                                    arrowcolor=arrow_color,
                                    font=dict(size=8, color=arrow_color),
                                    bgcolor=bg_color,
                                    row=1, col=1
                                )
                        except Exception as e:
                            logger.error(f"Error adding pattern marker: {str(e)}")
                            continue
            
            # Add special ML prediction markers
            for ml_point in ml_prediction_points:
                try:
                    if ml_point['direction'] == 'UP':
                        marker_color = 'blue'
                        marker_symbol = 'triangle-up'
                        y_position = ml_point['price'] * 0.9995
                    else:
                        marker_color = 'purple'
                        marker_symbol = 'triangle-down'
                        y_position = ml_point['price'] * 1.0005
                        
                    # Add a special marker for ML predictions
                    fig.add_trace(
                        go.Scatter(
                            x=[ml_point['timestamp']],
                            y=[y_position],
                            mode='markers+text',
                            marker=dict(
                                symbol=marker_symbol,
                                size=12,
                                color=marker_color,
                                line=dict(width=2, color='white')
                            ),
                            text=['🤖'],
                            textposition='middle center',
                            name='ML Signal',
                            showlegend=False
                        ),
                        row=1, col=1
                    )
                    
                except Exception as e:
                    logger.error(f"Error adding ML marker: {str(e)}")
                    continue
            
            # Update layout
            title = f"Price Chart for {self.symbol}"
            if depth:
                title += f" ({depth} depth)"
                
            fig.update_layout(
                title=title,
                xaxis_title="Time",
                yaxis_title="Price",
                xaxis_rangeslider_visible=False,
                height=700,
                margin=dict(l=50, r=50, b=50, t=80),
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating price chart: {str(e)}")
            fig = go.Figure()
            fig.update_layout(title=f"Error creating chart: {str(e)}")
            return fig
    
    def create_pattern_card(self, patterns, depth):
        """
        Create HTML pattern summary card.
        
        Args:
            patterns: Dictionary with pattern data
            depth: Analysis depth
            
        Returns:
            HTML string for pattern card
        """
        if not patterns:
            return "<div class='stCard'><h3>No Patterns Detected</h3></div>"
        
        try:
            # Count patterns
            bullish_count = 0
            bearish_count = 0
            
            # Count recent patterns (last 3 candles)
            recent_bullish = []
            recent_bearish = []
            
            for pattern_name, pattern_data in patterns.items():
                for instance in pattern_data['instances']:
                    if instance['type'] == 'bullish':
                        bullish_count += 1
                        # Check if recent
                        timestamp = instance['timestamp']
                        if isinstance(timestamp, str):
                            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        
                        if (datetime.now() - timestamp).total_seconds() < 900:  # Last 15 minutes
                            recent_bullish.append(pattern_name)
                    else:
                        bearish_count += 1
                        # Check if recent
                        timestamp = instance['timestamp']
                        if isinstance(timestamp, str):
                            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        
                        if (datetime.now() - timestamp).total_seconds() < 900:  # Last 15 minutes
                            recent_bearish.append(pattern_name)
            
            # Determine dominant direction
            if bullish_count > bearish_count:
                dominant = "Bullish"
                color = "green"
            elif bearish_count > bullish_count:
                dominant = "Bearish"
                color = "red"
            else:
                dominant = "Neutral"
                color = "gray"
            
            # Create card
            html = f"""
            <div style="border: 1px solid {color}; border-radius: 10px; padding: 15px; margin-bottom: 15px;">
                <h3 style="text-align: center; color: {color};">{depth.capitalize()} Pattern Analysis</h3>
                <p style="font-size: 1.2em; text-align: center;">
                    Dominant Bias: <strong style="color: {color};">{dominant}</strong>
                </p>
                <hr style="border-top: 1px solid {color}50;">
                <div style="display: flex; justify-content: space-between;">
                    <div style="text-align: center; flex: 1;">
                        <p style="color: green; font-weight: bold;">Bullish: {bullish_count}</p>
                    </div>
                    <div style="text-align: center; flex: 1;">
                        <p style="color: red; font-weight: bold;">Bearish: {bearish_count}</p>
                    </div>
                </div>
            """
            
            # Add recent patterns
            if recent_bullish or recent_bearish:
                html += "<h4>Recent Patterns</h4><div style='display: flex; justify-content: space-between;'>"
                
                # Bullish patterns
                html += "<div style='flex: 1;'>"
                if recent_bullish:
                    html += "<p style='color: green;'>Bullish:</p><ul style='color: green;'>"
                    for pattern in recent_bullish[:3]:  # Show top 3
                        html += f"<li>{pattern}</li>"
                    html += "</ul>"
                html += "</div>"
                
                # Bearish patterns
                html += "<div style='flex: 1;'>"
                if recent_bearish:
                    html += "<p style='color: red;'>Bearish:</p><ul style='color: red;'>"
                    for pattern in recent_bearish[:3]:  # Show top 3
                        html += f"<li>{pattern}</li>"
                    html += "</ul>"
                html += "</div>"
                
                html += "</div>"
            
            html += "</div>"
            return html
            
        except Exception as e:
            logger.error(f"Error creating pattern card: {str(e)}")
            return f"<div class='stCard'><h3>Error Creating Pattern Card</h3><p>{str(e)}</p></div>"
    
    def create_prediction_card(self, predictions):
        """
        Create HTML prediction summary card.
        
        Args:
            predictions: Dictionary with prediction data
            
        Returns:
            HTML string for prediction card
        """
        if not predictions or 'predictions' not in predictions or not predictions['predictions']:
            return "<div class='stCard'><h3>No Predictions Available</h3></div>"
        
        try:
            # Get latest prediction
            pred = predictions['predictions'][-1]
            
            # Get prediction details
            direction = pred['direction']
            confidence = pred['confidence']
            pip_target = pred.get('pip_target', 0)
            take_profit = pred.get('take_profit', 0)
            stop_loss = pred.get('stop_loss', 0)
            method = pred.get('method', 'unknown')
            depth = pred.get('data_depth', 'unknown')
            
            # Handle timestamp safely
            age_str = "recently"
            if 'timestamp' in pred:
                timestamp = pred['timestamp']
                
                # Convert timestamp to datetime if it's a string
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                
                # Age of prediction
                age = datetime.now() - timestamp
                age_str = f"{int(age.total_seconds() // 60)} minutes ago" if age.total_seconds() >= 60 else f"{int(age.total_seconds())} seconds ago"
            
            # Set color based on direction
            if direction == 'UP':
                color = 'green'
                emoji = '🔼'
            elif direction == 'DOWN':
                color = 'red'
                emoji = '🔽'
            else:
                color = 'gray'
                emoji = '◾'
            
            # Create card
            html = f"""
            <div style="border: 1px solid {color}; border-radius: 10px; padding: 15px; margin-bottom: 15px;">
                <h3 style="text-align: center; color: {color};">Latest Prediction</h3>
                <p style="font-size: 1.5em; text-align: center;">
                    Direction: <strong style="color: {color};">{direction} {emoji}</strong><br>
                    Confidence: <strong>{confidence*100:.1f}%</strong>
                </p>
                <hr style="border-top: 1px solid {color}50;">
                <div style="display: flex; justify-content: space-between;">
                    <div style="text-align: center; flex: 1;">
                        <p>Target: <strong>{pip_target:.1f} pips</strong></p>
                    </div>
                    <div style="text-align: center; flex: 1;">
                        <p>Method: <strong>{method.replace('_', ' ').title()}</strong></p>
                    </div>
                </div>
                <div style="display: flex; justify-content: space-between; margin-top: 10px;">
                    <div style="text-align: center; flex: 1;">
                        <p style="color: green;">Take Profit: <strong>{take_profit:.5f}</strong></p>
                    </div>
                    <div style="text-align: center; flex: 1;">
                        <p style="color: red;">Stop Loss: <strong>{stop_loss:.5f}</strong></p>
                    </div>
                </div>
                <p style="text-align: right; font-style: italic; margin-top: 10px; font-size: 0.8em;">
                    Generated {age_str} using {depth} depth
                </p>
            </div>
            """
            
            return html
            
        except Exception as e:
            logger.error(f"Error creating prediction card: {str(e)}")
            return f"<div class='stCard'><h3>Error Creating Prediction Card</h3><p>{str(e)}</p></div>"
    
    def create_performance_card(self, performance):
        """
        Create HTML performance summary card.
        
        Args:
            performance: Dictionary with performance metrics
            
        Returns:
            HTML string for performance card
        """
        if not performance:
            return "<div class='stCard'><h3>No Performance Data Available</h3></div>"
        
        try:
            # Extract metrics - Handle different potential formats
            win_rate = performance.get('win_rate', 0) * 100
            
            # Get total predictions - use ensemble method count if total_predictions not present
            total_predictions = performance.get('total_predictions', 0)
            if not total_predictions and 'by_method' in performance and 'ensemble_method' in performance['by_method']:
                total_predictions = performance['by_method']['ensemble_method'].get('count', 0)
            
            # Get win/loss counts - calculate from by_method if not directly available
            win_count = performance.get('win_count', 0)
            loss_count = performance.get('loss_count', 0)
            
            if not win_count and 'by_method' in performance:
                # Sum wins across all methods
                win_count = sum(method.get('wins', 0) for method in performance['by_method'].values())
            
            if not loss_count and 'by_method' in performance:
                # Sum losses across all methods
                loss_count = sum(method.get('losses', 0) for method in performance['by_method'].values())
            
            profit_factor = performance.get('profit_factor', 0)
            avg_win = performance.get('average_win', 0)
            avg_loss = performance.get('average_loss', 0)
            
            # Set color based on win rate
            if win_rate >= 60:
                color = 'green'
            elif win_rate >= 50:
                color = 'orange'
            else:
                color = 'red'
            
            # Create card
            html = f"""
            <div style="border: 1px solid {color}; border-radius: 10px; padding: 15px; margin-bottom: 15px;">
                <h3 style="text-align: center; color: {color};">Performance Metrics</h3>
                <p style="font-size: 1.5em; text-align: center;">
                    Win Rate: <strong style="color: {color};">{win_rate:.1f}%</strong>
                </p>
                <hr style="border-top: 1px solid {color}50;">
                <div style="display: flex; justify-content: space-between;">
                    <div style="text-align: center; flex: 1;">
                        <p>Predictions: <strong>{total_predictions}</strong></p>
                    </div>
                    <div style="text-align: center; flex: 1;">
                        <p style="color: green;">Wins: <strong>{win_count}</strong></p>
                    </div>
                    <div style="text-align: center; flex: 1;">
                        <p style="color: red;">Losses: <strong>{loss_count}</strong></p>
                    </div>
                </div>
                <div style="display: flex; justify-content: space-between; margin-top: 10px;">
                    <div style="text-align: center; flex: 1;">
                        <p>Profit Factor: <strong>{profit_factor:.2f}</strong></p>
                    </div>
                    <div style="text-align: center; flex: 1;">
                        <p style="color: green;">Avg Win: <strong>{avg_win:.1f} pips</strong></p>
                    </div>
                    <div style="text-align: center; flex: 1;">
                        <p style="color: red;">Avg Loss: <strong>{avg_loss:.1f} pips</strong></p>
                    </div>
                </div>
            </div>
            """
            
            return html
            
        except Exception as e:
            logger.error(f"Error creating performance card: {str(e)}")
            return f"<div class='stCard'><h3>Error Creating Performance Card</h3><p>{str(e)}</p></div>"
    
    def create_prediction_history_table(self, predictions):
        """
        Create DataFrame for prediction history table.
        
        Args:
            predictions: Dictionary with prediction data
            
        Returns:
            DataFrame with prediction history
        """
        if not predictions or 'predictions' not in predictions or not predictions['predictions']:
            return None
        
        try:
            # Extract predictions
            preds = predictions['predictions']
            
            # Create DataFrame
            data = []
            for pred in preds[-10:]:  # Show last 10 predictions
                timestamp = pred['timestamp']
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                
                # Format timestamp
                time_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
                
                # Get prediction details
                direction = pred['direction']
                confidence = pred['confidence']
                pip_target = pred.get('pip_target', 0)
                method = pred.get('method', 'unknown')
                depth = pred.get('data_depth', 'unknown')
                
                # Get outcome if available
                outcome = pred.get('outcome', 'Pending')
                outcome_details = pred.get('outcome_details', '')
                pips_gained = pred.get('pips_gained', 0)
                pips_lost = pred.get('pips_lost', 0)
                
                # Format outcome
                if outcome == 'WIN':
                    outcome_color = 'green'
                    pips = f"+{pips_gained:.1f}"
                elif outcome == 'LOSS':
                    outcome_color = 'red'
                    pips = f"-{pips_lost:.1f}"
                else:
                    outcome_color = 'gray'
                    pips = '0.0'
                
                # Add to data
                data.append({
                    'Time': time_str,
                    'Direction': direction,
                    'Confidence': f"{confidence*100:.1f}%",
                    'Target': f"{pip_target:.1f} pips",
                    'Method': method.replace('_', ' ').title(),
                    'Depth': depth.capitalize(),
                    'Outcome': f"<span style='color: {outcome_color};'>{outcome}</span>",
                    'Pips': pips
                })
            
            # Create DataFrame
            df = pd.DataFrame(data)
            return df
            
        except Exception as e:
            logger.error(f"Error creating prediction history table: {str(e)}")
            return None
    
    def _thread_safe_load_raw_data(self, force=True):
        """Thread-safe version of load_raw_data that doesn't access session state."""
        try:
            # Find latest file
            latest_file = self.find_latest_file(self.raw_data_dir, '.csv')
            if latest_file is None:
                return None
                
            # Check file modification time
            last_modified = datetime.fromtimestamp(os.path.getmtime(latest_file))
            if not force and self.last_update_time['raw'] is not None:
                if last_modified <= self.last_update_time['raw']:
                    # File hasn't changed
                    return self.raw_data
            
            logger.info(f"Loading raw data from {latest_file}")
            df = pd.read_csv(latest_file)
            
            # Ensure timestamp is datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            logger.info(f"Loaded {len(df)} records from {latest_file}")
            
            # Update cache
            self.last_update_time['raw'] = datetime.now()
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading raw data: {str(e)}")
            return None
            
    def _thread_safe_load_analyzed_data(self, depth, force=True):
        """Thread-safe version of load_analyzed_data that doesn't access session state."""
        try:
            # Find latest file
            latest_file = self.find_latest_file(self.analyzed_data_dirs[depth], '_analyzed.csv')
            if latest_file is None:
                return None
                
            # Check file modification time
            last_modified = datetime.fromtimestamp(os.path.getmtime(latest_file))
            if not force and self.last_update_time['analyzed'][depth] is not None:
                if last_modified <= self.last_update_time['analyzed'][depth]:
                    # File hasn't changed
                    return self.analyzed_data[depth]
            
            logger.info(f"Loading analyzed data for {depth} from {latest_file}")
            df = pd.read_csv(latest_file)
            
            # Ensure timestamp is datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            logger.info(f"Loaded {len(df)} records from {latest_file}")
            
            # Update cache time
            self.last_update_time['analyzed'][depth] = datetime.now()
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading analyzed data for {depth}: {str(e)}")
            return None
            
    def _thread_safe_load_pattern_data(self, depth, force=True):
        """Thread-safe version of load_pattern_data that doesn't access session state."""
        try:
            # Find latest file
            latest_file = self.find_latest_file(self.analyzed_data_dirs[depth], '_patterns.json')
            if latest_file is None:
                return None
                
            # Check file modification time
            last_modified = datetime.fromtimestamp(os.path.getmtime(latest_file))
            if not force and self.last_update_time['analyzed'][depth] is not None:
                if last_modified <= self.last_update_time['analyzed'][depth]:
                    # File hasn't changed
                    return self.pattern_data[depth]
            
            logger.info(f"Loading pattern data for {depth} from {latest_file}")
            with open(latest_file, 'r') as f:
                patterns_data = json.load(f)
            
            # Properly process the new pattern format
            # The top level object has pattern array, support and resistance levels
            if 'patterns' in patterns_data and isinstance(patterns_data['patterns'], list):
                # Convert to the format expected by the dashboard
                patterns = {}
                for pattern in patterns_data['patterns']:
                    pattern_name = pattern['pattern']
                    if pattern_name not in patterns:
                        patterns[pattern_name] = {
                            'instances': [],
                            'count': 0
                        }
                    
                    # Determine pattern type
                    if pattern_name.startswith('BULLISH') or pattern_name in ['HAMMER', 'MORNING_STAR', 'PIERCING']:
                        pattern_type = 'bullish'
                    else:
                        pattern_type = 'bearish'
                    
                    # Add instance
                    patterns[pattern_name]['instances'].append({
                        'timestamp': pattern['time'],
                        'type': pattern_type,
                        'strength': pattern['strength'],
                        'high': 0.0,  # Not provided in new format
                        'low': 0.0    # Not provided in new format
                    })
                    patterns[pattern_name]['count'] = len(patterns[pattern_name]['instances'])
            else:
                # Fallback to old format or empty dict
                patterns = patterns_data
            
            pattern_count = sum(p['count'] for p in patterns.values()) if patterns else 0
            logger.info(f"Loaded {len(patterns)} pattern types with {pattern_count} instances from {latest_file}")
            
            # Update cache time
            self.last_update_time['analyzed'][depth] = datetime.now()
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error loading pattern data for {depth}: {str(e)}")
            return None
            
    def _thread_safe_load_predictions(self, force=True):
        """Thread-safe version of load_predictions that doesn't access session state."""
        try:
            # Find latest file
            latest_file = self.find_latest_file(self.prediction_dir, '_predictions.json')
            if latest_file is None:
                return None
                
            # Check file modification time
            last_modified = datetime.fromtimestamp(os.path.getmtime(latest_file))
            if not force and self.last_update_time['predictions'] is not None:
                if last_modified <= self.last_update_time['predictions']:
                    # File hasn't changed
                    return self.predictions
            
            logger.info(f"Loading predictions from {latest_file}")
            with open(latest_file, 'r') as f:
                predictions = json.load(f)
            
            logger.info(f"Loaded {len(predictions.get('predictions', []))} predictions from {latest_file}")
            
            # Update cache time
            self.last_update_time['predictions'] = datetime.now()
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error loading predictions: {str(e)}")
            return None
            
    def _thread_safe_load_performance(self, force=True):
        """Thread-safe version of load_performance that doesn't access session state."""
        try:
            # Find latest file - FIXED: Use performance_dir instead of prediction_dir
            latest_file = self.find_latest_file(self.performance_dir, '_performance.json')
            if latest_file is None:
                # Try for latest_performance.json format
                latest_file = self.find_latest_file(self.performance_dir, 'latest_performance.json')
                if latest_file is None:
                    return None
                
            # Check file modification time
            last_modified = datetime.fromtimestamp(os.path.getmtime(latest_file))
            if not force and self.last_update_time['performance'] is not None:
                if last_modified <= self.last_update_time['performance']:
                    # File hasn't changed
                    return self.performance
            
            logger.info(f"Loading performance metrics from {latest_file}")
            with open(latest_file, 'r') as f:
                performance = json.load(f)
            
            logger.info(f"Loaded performance metrics from {latest_file}")
            
            # Update cache time
            self.last_update_time['performance'] = datetime.now()
            
            return performance
            
        except Exception as e:
            logger.error(f"Error loading performance metrics: {str(e)}")
            return None
    
    def data_refresh_thread(self, interval_seconds=60):
        """
        Background thread to refresh data periodically.
        
        This thread loads data directly into session state without triggering
        a full UI refresh. This prevents flickering by ensuring data is always
        available without forcing reruns.
        """
        logger.info(f"Starting data refresh thread with {interval_seconds}s interval")
        
        # Initial backoff to allow main thread to initialize
        time.sleep(3)
        
        while True:
            try:
                # Track if any files have changed
                files_updated = False
                data_updated = False
                
                # Load raw data directly into instance variables using thread-safe methods
                raw_data = self._thread_safe_load_raw_data(force=True)
                if raw_data is not None:
                    # Update instance variable
                    self.raw_data = raw_data
                    files_updated = True
                    data_updated = True
                
                # Load analyzed data and patterns for each depth
                for depth in self.analyzed_data_dirs:
                    analyzed_data = self._thread_safe_load_analyzed_data(depth, force=True)
                    if analyzed_data is not None:
                        self.analyzed_data[depth] = analyzed_data
                        files_updated = True
                        data_updated = True
                    
                    pattern_data = self._thread_safe_load_pattern_data(depth, force=True)
                    if pattern_data is not None:
                        self.pattern_data[depth] = pattern_data
                        files_updated = True
                        data_updated = True
                
                # Load predictions
                predictions = self._thread_safe_load_predictions(force=True)
                if predictions is not None:
                    self.predictions = predictions
                    files_updated = True
                    data_updated = True
                
                # Load performance
                performance = self._thread_safe_load_performance(force=True)
                if performance is not None:
                    self.performance = performance
                    files_updated = True
                    data_updated = True
                
                # If files were updated, but we couldn't update the data, just log it
                if files_updated and not data_updated:
                    logger.warning("Files were updated but data couldn't be loaded properly")
                
                # Only signal UI refresh if new data was loaded
                if data_updated:
                    # Log successful update
                    logger.info("Data refresh thread updated data successfully")
                    
                    # Signal UI refresh 
                    data_update_queue.put(('refresh', True))
                else:
                    logger.info("No new data found in refresh cycle")
                
            except Exception as e:
                logger.error(f"Error in data refresh thread: {str(e)}")
                # Don't signal a refresh if there was an error
            
            # Wait for next refresh
            time.sleep(interval_seconds)
    
    def run_dashboard(self):
        """Run the Streamlit dashboard."""
        # Title and config
        st.set_page_config(
            page_title=f"Forex Dashboard - {self.symbol}",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Add a placeholder at the top for loading indicators
        load_placeholder = st.empty()
        
        # Initialize refresh control
        if 'last_refresh_time' not in st.session_state:
            st.session_state.last_refresh_time = datetime.now()
            st.session_state.should_refresh = False
            st.session_state.manual_refresh = False
        
        # Initialize session state for data persistence
        if 'initialized' not in st.session_state:
            with load_placeholder.container():
                st.info("Loading dashboard data for the first time...")
            
            st.session_state.initialized = True
            st.session_state.data_loaded = False
            st.session_state.cached_data = {
                'raw': None,
                'analyzed': {
                    'minimum': None,
                    'recommended': None,
                    'optimal': None
                },
                'patterns': {
                    'minimum': None,
                    'recommended': None,
                    'optimal': None
                },
                'predictions': None,
                'performance': None
            }
            
            # Perform initial data loading to ensure the dashboard has data on first display
            self.load_raw_data(force=True)
            for depth in self.analyzed_data_dirs:
                self.load_analyzed_data(depth, force=True)
                self.load_pattern_data(depth, force=True)
            self.load_predictions(force=True)
            self.load_performance(force=True)
            
            # Mark data as loaded
            st.session_state.data_loaded = True
            logger.info("Initial data loading complete")
        
        # Clear loading indicator
        load_placeholder.empty()
        
        # Anti-flickering: ensure we have data in cache before rendering UI
        # If we don't have data, we should try to load it but avoid rerunning
        if 'cached_data' not in st.session_state or not st.session_state.data_loaded:
            logger.warning("Session state data missing - using instance data as fallback")
            
            # Emergency fallback - use the instance data
            if not hasattr(st.session_state, 'cached_data'):
                st.session_state.cached_data = {}
                
            # Use whatever data we have in the instance
            if self.raw_data is not None and 'raw' not in st.session_state.cached_data:
                st.session_state.cached_data['raw'] = self.raw_data
                
            if 'analyzed' not in st.session_state.cached_data:
                st.session_state.cached_data['analyzed'] = {}
            for depth, data in self.analyzed_data.items():
                if data is not None:
                    st.session_state.cached_data['analyzed'][depth] = data
                    
            if 'patterns' not in st.session_state.cached_data:
                st.session_state.cached_data['patterns'] = {}
            for depth, data in self.pattern_data.items():
                if data is not None:
                    st.session_state.cached_data['patterns'][depth] = data
                    
            if self.predictions is not None:
                st.session_state.cached_data['predictions'] = self.predictions
                
            if self.performance is not None:
                st.session_state.cached_data['performance'] = self.performance
                
            st.session_state.data_loaded = True
            logger.info("Emergency fallback data loaded from instance")
        
        # Check if we need to copy instance data to session state - this avoids flickering
        # by ensuring we always show "something" even during refresh
        if self.raw_data is not None and st.session_state.cached_data['raw'] is None:
            st.session_state.cached_data['raw'] = self.raw_data
            
        for depth, data in self.analyzed_data.items():
            if data is not None and (depth not in st.session_state.cached_data['analyzed'] or st.session_state.cached_data['analyzed'][depth] is None):
                st.session_state.cached_data['analyzed'][depth] = data
                
        for depth, data in self.pattern_data.items():
            if data is not None and (depth not in st.session_state.cached_data['patterns'] or st.session_state.cached_data['patterns'][depth] is None):
                st.session_state.cached_data['patterns'][depth] = data
                
        if self.predictions is not None and st.session_state.cached_data['predictions'] is None:
            st.session_state.cached_data['predictions'] = self.predictions
            
        if self.performance is not None and st.session_state.cached_data['performance'] is None:
            st.session_state.cached_data['performance'] = self.performance
        
        # Sidebar
        with st.sidebar:
            st.title("Forex Pattern Dashboard")
            st.subheader(f"{self.symbol} - {self.timeframe}")
            
            # Refresh button and last update time
            col1, col2 = st.columns([2, 1])
            with col1:
                # Create a more reliable refresh button with a key and container width
                if st.button("↻ Refresh Dashboard", key="refresh_button_fixed", use_container_width=True):
                    # Force an immediate refresh of all data
                    self.load_raw_data(force=True)
                    for depth in self.analyzed_data_dirs:
                        self.load_analyzed_data(depth, force=True)
                        self.load_pattern_data(depth, force=True)
                    self.load_predictions(force=True)
                    self.load_performance(force=True)
                    
                    # Update refresh timestamp
                    st.session_state.last_refresh_time = datetime.now()
                    
                    # Update session state to indicate we've loaded data
                    st.session_state.data_loaded = True
                    
                    # Show visual confirmation
                    st.success("✓ Data refreshed!")
            
            # Show last refresh time and cache status
            with col2:
                if 'last_refresh_time' in st.session_state:
                    time_str = st.session_state.last_refresh_time.strftime("%H:%M:%S")
                    st.info(f"Last: {time_str}")
                    
            # Data status indicator
            if st.session_state.data_loaded:
                st.success("✓ Data cache active")
            
            # Data information
            st.subheader("Data Sources")
            
            # Raw data info
            if self.raw_data is not None:
                raw_count = len(self.raw_data)
                raw_update = self.last_update_time['raw']
                raw_update_str = raw_update.strftime('%H:%M:%S') if raw_update else "Never"
                st.info(f"Raw Data: {raw_count} candles (Last update: {raw_update_str})")
            else:
                st.error("Raw Data: Not loaded")
            
            # Analyzed data info
            for depth in self.analyzed_data_dirs:
                if self.analyzed_data[depth] is not None:
                    count = len(self.analyzed_data[depth])
                    update = self.last_update_time['analyzed'][depth]
                    update_str = update.strftime('%H:%M:%S') if update else "Never"
                    cols = len(self.analyzed_data[depth].columns)
                    st.info(f"{depth.capitalize()} Data: {count} candles with {cols} indicators (Last update: {update_str})")
                else:
                    st.warning(f"{depth.capitalize()} Data: Not loaded")
            
            # Prediction info
            if self.predictions is not None:
                count = len(self.predictions.get('predictions', []))
                update = self.last_update_time['predictions']
                update_str = update.strftime('%H:%M:%S') if update else "Never"
                st.info(f"Predictions: {count} entries (Last update: {update_str})")
            else:
                st.warning("Predictions: Not loaded")
            
            # Display directories
            st.subheader("Data Directories")
            st.text_input("Raw Data", value=self.raw_data_dir, disabled=True)
            st.text_input("Analyzed Data", value=self.analyzed_data_dirs['optimal'], disabled=True)
            st.text_input("Predictions", value=self.prediction_dir, disabled=True)
        
        # Main content
        st.title(f"📊 Forex Pattern Dashboard - {self.symbol}")
        
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "Current Analysis", 
            "Depth Comparison", 
            "Prediction History", 
            "Performance Metrics"
        ])
        
        # Tab 1: Current Analysis
        with tab1:
            # Status row
            status_col1, status_col2, status_col3 = st.columns(3)
            
            # Current price
            with status_col1:
                if self.raw_data is not None and len(self.raw_data) > 0:
                    current_price = self.raw_data['close'].iloc[-1]
                    update_time = self.raw_data['timestamp'].iloc[-1]
                    time_diff = (datetime.now() - update_time).total_seconds()
                    
                    if time_diff < 60:
                        st.success(f"Current Price: {current_price:.5f}")
                    else:
                        mins = int(time_diff / 60)
                        st.warning(f"Last Price: {current_price:.5f} ({mins}m ago)")
                else:
                    st.error("No price data available")
            
            # Latest pattern
            with status_col2:
                optimal_patterns = self.pattern_data.get('optimal')
                if optimal_patterns:
                    # Find most recent pattern
                    latest_time = datetime.min
                    latest_pattern = None
                    latest_type = None
                    
                    for pattern_name, pattern_data in optimal_patterns.items():
                        for instance in pattern_data['instances']:
                            timestamp = instance['timestamp']
                            if isinstance(timestamp, str):
                                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                            
                            if timestamp > latest_time:
                                latest_time = timestamp
                                latest_pattern = pattern_name
                                latest_type = instance['type']
                    
                    if latest_pattern:
                        time_diff = (datetime.now() - latest_time).total_seconds()
                        color = "green" if latest_type == "bullish" else "red"
                        
                        if time_diff < 300:
                            st.markdown(f"<div style='color: {color};'>Latest Pattern: <strong>{latest_pattern}</strong> ({latest_type})</div>", unsafe_allow_html=True)
                        else:
                            mins = int(time_diff / 60)
                            st.markdown(f"<div style='color: {color};'>Last Pattern: <strong>{latest_pattern}</strong> ({mins}m ago)</div>", unsafe_allow_html=True)
                    else:
                        st.info("No patterns detected")
                else:
                    st.info("No pattern data available")
            
            # Latest prediction
            with status_col3:
                if self.predictions and 'predictions' in self.predictions and self.predictions['predictions']:
                    try:
                        latest_pred = self.predictions['predictions'][-1]
                        direction = latest_pred['direction']
                        confidence = latest_pred['confidence']
                        
                        # Handle timestamp carefully
                        if 'timestamp' in latest_pred:
                            timestamp = latest_pred['timestamp']
                            if isinstance(timestamp, str):
                                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                            
                            time_diff = (datetime.now() - timestamp).total_seconds()
                            color = "green" if direction == "UP" else "red" if direction == "DOWN" else "gray"
                            
                            if time_diff < 300:
                                st.markdown(f"<div style='color: {color};'>Prediction: <strong>{direction}</strong> ({confidence*100:.1f}%)</div>", unsafe_allow_html=True)
                            else:
                                mins = int(time_diff / 60)
                                st.markdown(f"<div style='color: {color};'>Last Prediction: <strong>{direction}</strong> ({mins}m ago)</div>", unsafe_allow_html=True)
                        else:
                            # If timestamp is missing, just show direction and confidence
                            color = "green" if direction == "UP" else "red" if direction == "DOWN" else "gray"
                            st.markdown(f"<div style='color: {color};'>Prediction: <strong>{direction}</strong> ({confidence*100:.1f}%)</div>", unsafe_allow_html=True)
                    except Exception as e:
                        st.warning(f"Error displaying prediction: {str(e)}")
                else:
                    st.info("No prediction data available")
            
            # Charts and cards
            chart_col, card_col = st.columns([3, 1])
            
            with chart_col:
                # Price chart with optimal data
                if self.analyzed_data['optimal'] is not None:
                    df = self.analyzed_data['optimal']
                    patterns = self.pattern_data.get('optimal')
                    predictions = self.predictions
                    
                    chart = self.create_price_chart(df, 'optimal', patterns, predictions)
                    st.plotly_chart(chart, use_container_width=True, key="optimal_chart")
                else:
                    st.warning("Optimal data not available for chart")
                    
                    # Try other depths
                    for depth in ['recommended', 'minimum']:
                        if self.analyzed_data[depth] is not None:
                            df = self.analyzed_data[depth]
                            patterns = self.pattern_data.get(depth)
                            predictions = self.predictions
                            
                            chart = self.create_price_chart(df, depth, patterns, predictions)
                            st.plotly_chart(chart, use_container_width=True, key=f"{depth}_fallback_chart")
                            break
                    else:
                        # No data at all
                        st.error("No data available for chart")
            
            with card_col:
                # Pattern card (optimal depth)
                if self.pattern_data['optimal'] is not None:
                    pattern_card = self.create_pattern_card(self.pattern_data['optimal'], 'optimal')
                    st.markdown(pattern_card, unsafe_allow_html=True)
                else:
                    st.warning("Optimal pattern data not available")
                
                # Prediction card
                if self.predictions is not None:
                    prediction_card = self.create_prediction_card(self.predictions)
                    st.markdown(prediction_card, unsafe_allow_html=True)
                else:
                    st.warning("Prediction data not available")
                
                # Performance card
                if self.performance is not None:
                    performance_card = self.create_performance_card(self.performance)
                    st.markdown(performance_card, unsafe_allow_html=True)
                else:
                    st.warning("Performance data not available")
        
        # Tab 2: Depth Comparison
        with tab2:
            st.subheader("Analysis Depth Comparison")
            
            # Create columns for each depth
            depth_cols = st.columns(3)
            
            for i, depth in enumerate(['minimum', 'recommended', 'optimal']):
                with depth_cols[i]:
                    st.subheader(f"{depth.capitalize()} Depth")
                    
                    if self.analyzed_data[depth] is not None:
                        df = self.analyzed_data[depth]
                        patterns = self.pattern_data.get(depth)
                        
                        # Small chart
                        chart = self.create_price_chart(df, depth, patterns)
                        st.plotly_chart(chart, use_container_width=True, key=f"{depth}_depth_chart")
                        
                        # Pattern card
                        if patterns is not None:
                            pattern_card = self.create_pattern_card(patterns, depth)
                            st.markdown(pattern_card, unsafe_allow_html=True)
                        else:
                            st.warning("Pattern data not available")
                    else:
                        st.warning(f"No data available for {depth} depth")
        
        # Tab 3: Prediction History
        with tab3:
            st.subheader("Prediction History")
            
            # Prediction history table
            if self.predictions is not None:
                history_df = self.create_prediction_history_table(self.predictions)
                if history_df is not None:
                    st.dataframe(history_df, use_container_width=True, hide_index=True)
                else:
                    st.warning("No prediction history available")
            else:
                st.warning("Prediction data not available")
            
            # Prediction metrics
            if self.predictions and 'predictions' in self.predictions:
                preds = self.predictions['predictions']
                
                # Count predictions by direction
                directions = {'UP': 0, 'DOWN': 0, 'NEUTRAL': 0}
                outcomes = {'WIN': 0, 'LOSS': 0, 'Pending': 0}
                depths = {'minimum': 0, 'recommended': 0, 'optimal': 0}
                methods = {'technical_consensus': 0, 'pattern_recognition': 0, 
                          'machine_learning': 0, 'ensemble_method': 0}
                
                for pred in preds:
                    direction = pred['direction']
                    directions[direction] = directions.get(direction, 0) + 1
                    
                    outcome = pred.get('outcome', 'Pending')
                    outcomes[outcome] = outcomes.get(outcome, 0) + 1
                    
                    depth = pred.get('data_depth', 'unknown')
                    depths[depth] = depths.get(depth, 0) + 1
                    
                    method = pred.get('method', 'unknown')
                    methods[method] = methods.get(method, 0) + 1
                
                # Display metrics
                st.subheader("Prediction Metrics")
                
                metrics_cols = st.columns(4)
                
                # Directions
                with metrics_cols[0]:
                    st.subheader("By Direction")
                    for direction, count in directions.items():
                        if count > 0:
                            color = "green" if direction == "UP" else "red" if direction == "DOWN" else "gray"
                            st.markdown(f"<div style='color: {color};'>{direction}: <strong>{count}</strong></div>", unsafe_allow_html=True)
                
                # Outcomes
                with metrics_cols[1]:
                    st.subheader("By Outcome")
                    for outcome, count in outcomes.items():
                        if count > 0:
                            color = "green" if outcome == "WIN" else "red" if outcome == "LOSS" else "gray"
                            st.markdown(f"<div style='color: {color};'>{outcome}: <strong>{count}</strong></div>", unsafe_allow_html=True)
                
                # Depths
                with metrics_cols[2]:
                    st.subheader("By Depth")
                    for depth, count in depths.items():
                        if count > 0:
                            st.text(f"{depth.capitalize()}: {count}")
                
                # Methods
                with metrics_cols[3]:
                    st.subheader("By Method")
                    for method, count in methods.items():
                        if count > 0:
                            st.text(f"{method.replace('_', ' ').title()}: {count}")
            else:
                st.warning("No prediction data available for metrics")
        
        # Tab 4: Performance Metrics
        with tab4:
            st.subheader("Performance Metrics")
            
            if self.performance is not None:
                perf = self.performance
                
                # Overall metrics
                st.subheader("Overall Performance")
                
                metrics_cols = st.columns(4)
                
                with metrics_cols[0]:
                    win_rate = perf.get('win_rate', 0) * 100
                    color = "green" if win_rate >= 60 else "orange" if win_rate >= 50 else "red"
                    st.markdown(f"<div style='text-align: center;'><h1 style='color: {color};'>{win_rate:.1f}%</h1><p>Win Rate</p></div>", unsafe_allow_html=True)
                
                with metrics_cols[1]:
                    profit_factor = perf.get('profit_factor', 0)
                    color = "green" if profit_factor >= 1.5 else "orange" if profit_factor >= 1 else "red"
                    st.markdown(f"<div style='text-align: center;'><h1 style='color: {color};'>{profit_factor:.2f}</h1><p>Profit Factor</p></div>", unsafe_allow_html=True)
                
                with metrics_cols[2]:
                    avg_win = perf.get('average_win', 0)
                    st.markdown(f"<div style='text-align: center;'><h1 style='color: green;'>{avg_win:.1f}</h1><p>Avg Win (pips)</p></div>", unsafe_allow_html=True)
                
                with metrics_cols[3]:
                    avg_loss = perf.get('average_loss', 0)
                    st.markdown(f"<div style='text-align: center;'><h1 style='color: red;'>{avg_loss:.1f}</h1><p>Avg Loss (pips)</p></div>", unsafe_allow_html=True)
                
                # Detailed breakdowns
                st.subheader("Performance Breakdowns")
                
                tabs = st.tabs(["By Direction", "By Depth", "By Method"])
                
                # By Direction
                with tabs[0]:
                    if 'by_direction' in perf:
                        dir_cols = st.columns(len(perf['by_direction']))
                        
                        for i, (direction, stats) in enumerate(perf['by_direction'].items()):
                            with dir_cols[i]:
                                st.subheader(direction)
                                
                                # Handle both stats structures
                                # Some files have win_rate at the top level, others have wins/losses/count
                                win_rate = stats.get('win_rate', 0)
                                # If win_rate not directly in stats, calculate it
                                if win_rate == 0 and 'wins' in stats and 'count' in stats and stats['count'] > 0:
                                    win_rate = stats['wins'] / stats['count']
                                
                                win_rate *= 100  # Convert to percentage
                                
                                # Get wins and losses
                                wins = stats.get('wins', 0)
                                losses = stats.get('losses', 0)
                                
                                # Get count
                                count = stats.get('count', 0)
                                if count == 0 and wins > 0 and losses > 0:
                                    count = wins + losses
                                
                                # Display stats
                                if direction != 'NEUTRAL':
                                    color = "green" if win_rate >= 60 else "orange" if win_rate >= 50 else "red"
                                    st.markdown(f"<div><p>Win Rate: <strong style='color: {color};'>{win_rate:.1f}%</strong></p></div>", unsafe_allow_html=True)
                                    st.text(f"Wins: {wins}")
                                    st.text(f"Losses: {losses}")
                                
                                st.text(f"Total: {count}")
                
                # By Depth
                with tabs[1]:
                    if 'by_depth' in perf:
                        depth_cols = st.columns(len(perf['by_depth']))
                        
                        for i, (depth, stats) in enumerate(perf['by_depth'].items()):
                            with depth_cols[i]:
                                st.subheader(depth.capitalize())
                                
                                # Handle both stats structures
                                win_rate = stats.get('win_rate', 0)
                                # If win_rate not directly in stats, calculate it
                                if win_rate == 0 and 'wins' in stats and 'count' in stats and stats['count'] > 0:
                                    win_rate = stats['wins'] / stats['count']
                                
                                win_rate *= 100  # Convert to percentage
                                
                                # Get wins and losses
                                wins = stats.get('wins', 0)
                                losses = stats.get('losses', 0)
                                
                                # Get count
                                count = stats.get('count', 0)
                                if count == 0 and wins > 0 and losses > 0:
                                    count = wins + losses
                                
                                # Display stats
                                color = "green" if win_rate >= 60 else "orange" if win_rate >= 50 else "red"
                                st.markdown(f"<div><p>Win Rate: <strong style='color: {color};'>{win_rate:.1f}%</strong></p></div>", unsafe_allow_html=True)
                                st.text(f"Wins: {wins}")
                                st.text(f"Losses: {losses}")
                                st.text(f"Total: {count}")
                
                # By Method
                with tabs[2]:
                    if 'by_method' in perf:
                        method_cols = st.columns(len(perf['by_method']))
                        
                        for i, (method, stats) in enumerate(perf['by_method'].items()):
                            with method_cols[i]:
                                st.subheader(method.replace('_', ' ').title())
                                
                                # Handle both stats structures
                                win_rate = stats.get('win_rate', 0)
                                # If win_rate not directly in stats, calculate it
                                if win_rate == 0 and 'wins' in stats and 'count' in stats and stats['count'] > 0:
                                    win_rate = stats['wins'] / stats['count']
                                
                                win_rate *= 100  # Convert to percentage
                                
                                # Get wins and losses
                                wins = stats.get('wins', 0)
                                losses = stats.get('losses', 0)
                                
                                # Get count
                                count = stats.get('count', 0)
                                if count == 0 and wins > 0 and losses > 0:
                                    count = wins + losses
                                
                                # Display stats
                                color = "green" if win_rate >= 60 else "orange" if win_rate >= 50 else "red"
                                st.markdown(f"<div><p>Win Rate: <strong style='color: {color};'>{win_rate:.1f}%</strong></p></div>", unsafe_allow_html=True)
                                st.text(f"Wins: {wins}")
                                st.text(f"Losses: {losses}")
                                st.text(f"Total: {count}")
            else:
                st.warning("No performance data available")
        
        # Footer
        st.markdown("---")
        st.markdown("Multi-Component Forex Pattern Prediction System")
        
        # Check data updates from background thread
        refresh_signaled = False
        while not data_update_queue.empty():
            try:
                update_type, update_data = data_update_queue.get_nowait()
                if update_type == 'refresh':
                    # Just set a flag, don't refresh immediately
                    refresh_signaled = True
                    logger.info("Refresh signal received from background thread")
            except queue.Empty:
                break
        
        # Determine if we should actually refresh based on time since last refresh
        # and whether a manual refresh was requested
        current_time = datetime.now()
        time_since_refresh = (current_time - st.session_state.last_refresh_time).total_seconds()
        
        # Check if user pressed the refresh button
        if 'manual_refresh_clicked' in st.session_state and st.session_state.manual_refresh_clicked:
            st.session_state.manual_refresh = True
            st.session_state.manual_refresh_clicked = False  # Reset button state
            logger.info("Manual refresh requested by user")
        
        # Determine if we should refresh
        should_refresh = (
            # Manual refresh requested
            st.session_state.manual_refresh or
            # Background thread signaled AND it's been at least 5 seconds since last refresh
            (refresh_signaled and time_since_refresh > 5) or
            # Auto refresh every 60 seconds
            time_since_refresh > 60
        )
        
        # Handle refresh if needed
        if should_refresh and get_script_run_ctx() is not None:
            logger.info(f"Refreshing dashboard (manual={st.session_state.manual_refresh}, "
                       f"signaled={refresh_signaled}, elapsed={time_since_refresh:.1f}s)")
            
            # Update refresh timestamp
            st.session_state.last_refresh_time = current_time
            
            # Reset manual refresh flag
            st.session_state.manual_refresh = False
            
            # Quietly update session state with latest data (without rerun)
            # This avoids flickering by updating session state before rerun
            if self.raw_data is not None:
                st.session_state.cached_data['raw'] = self.raw_data
            
            for depth in self.analyzed_data_dirs:
                if self.analyzed_data[depth] is not None:
                    if 'analyzed' not in st.session_state.cached_data:
                        st.session_state.cached_data['analyzed'] = {}
                    st.session_state.cached_data['analyzed'][depth] = self.analyzed_data[depth]
                
                if self.pattern_data[depth] is not None:
                    if 'patterns' not in st.session_state.cached_data:
                        st.session_state.cached_data['patterns'] = {}
                    st.session_state.cached_data['patterns'][depth] = self.pattern_data[depth]
            
            if self.predictions is not None:
                st.session_state.cached_data['predictions'] = self.predictions
            
            if self.performance is not None:
                st.session_state.cached_data['performance'] = self.performance
            
            # Mark data as loaded
            st.session_state.data_loaded = True
            
            # Add a small visual indicator that refresh is happening
            refresh_indicator = st.empty()
            with refresh_indicator.container():
                with st.spinner("Refreshing..."):
                    # Small delay so the spinner is visible but not distracting
                    time.sleep(0.3)
                refresh_indicator.empty()
            
            # Use rerun to update the UI with the latest data
            st.rerun()

def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Forex Dashboard')
    parser.add_argument('--symbol', default='EUR_USD', help='Forex pair symbol (e.g., EUR_USD)')
    parser.add_argument('--timeframe', default='5m', help='Data timeframe')
    parser.add_argument('--data-dir', help='Root directory for data')
    parser.add_argument('--refresh-interval', type=int, default=60, 
                       help='Data refresh interval in seconds')
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Create dashboard
    dashboard = ForexDashboard(
        symbol=args.symbol,
        timeframe=args.timeframe,
        data_dir=args.data_dir
    )
    
    # Start data refresh thread
    refresh_thread = threading.Thread(
        target=dashboard.data_refresh_thread,
        args=(args.refresh_interval,),
        daemon=True
    )
    refresh_thread.start()
    
    # Run dashboard
    dashboard.run_dashboard()

if __name__ == "__main__":
    main()