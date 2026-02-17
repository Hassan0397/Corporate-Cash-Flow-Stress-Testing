"""
Time Series Forecasting Module for Corporate Cash Flow Stress Testing Platform
Professional Forecasting with Prophet, ARIMA, and LSTM Models
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Prophet
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly

# ARIMA
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

# LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

import warnings
warnings.filterwarnings('ignore')

def forecast_cash_flows(df):
    """
    Main forecasting function with multiple model options
    
    Args:
        df (pandas.DataFrame): Cleaned dataset with date and cash flow columns
    
    Returns:
        dict: Forecast results and model performance metrics
    """
    
    if df is None:
        st.error("❌ No data provided for forecasting")
        return None
    
    # Professional header
    st.markdown("""
    <div style="background: linear-gradient(135deg, #0891b215 0%, #0e749015 100%);
                padding: 2rem; border-radius: 12px; margin-bottom: 2rem;
                border: 1px solid #e2e8f0;">
        <h3 style="color: #1e293b; margin-bottom: 1rem; display: flex; align-items: center;">
            <span style="background: #0891b2; color: white; padding: 0.5rem 1rem;
                        border-radius: 8px; margin-right: 1rem; font-size: 1.2rem;">
                📈
            </span>
            Advanced Time Series Forecasting
        </h3>
        <p style="color: #475569; line-height: 1.6; margin-bottom: 0;">
            Leveraging state-of-the-art forecasting models (Prophet, ARIMA, LSTM) to predict 
            future cash flows with confidence intervals. Ensemble methods for robust predictions.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Prepare data for forecasting
    df_forecast = prepare_forecast_data(df)
    
    if df_forecast is None or len(df_forecast) < 12:
        st.error("Insufficient data for forecasting. Please provide at least 12 months of data.")
        return None
    
    # Model selection and configuration
    st.markdown("### 🎯 Model Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        forecast_horizon = st.slider(
            "Forecast Horizon (months)",
            min_value=3,
            max_value=24,
            value=12,
            step=3,
            help="Number of months to forecast into the future"
        )
    
    with col2:
        selected_models = st.multiselect(
            "Select Forecasting Models",
            ["Prophet", "ARIMA/SARIMA", "LSTM Neural Network"],
            default=["Prophet", "ARIMA/SARIMA", "LSTM Neural Network"],
            help="Choose one or more models for ensemble forecasting"
        )
    
    with col3:
        confidence_level = st.select_slider(
            "Confidence Level",
            options=[80, 85, 90, 95, 99],
            value=95,
            help="Confidence interval for predictions"
        )
    
    # Advanced options expander
    with st.expander("⚙️ Advanced Configuration", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            seasonality_mode = st.selectbox(
                "Seasonality Mode",
                ["additive", "multiplicative"],
                index=0,
                help="Seasonality component type"
            )
            
            changepoint_prior = st.slider(
                "Changepoint Prior Scale",
                min_value=0.001,
                max_value=0.5,
                value=0.05,
                step=0.01,
                format="%.3f",
                help="Flexibility of trend changes"
            )
        
        with col2:
            arima_order_p = st.slider("ARIMA Order - p (AR)", 0, 5, 1)
            arima_order_d = st.slider("ARIMA Order - d (Diff)", 0, 2, 1)
            arima_order_q = st.slider("ARIMA Order - q (MA)", 0, 5, 1)
        
        with col3:
            lstm_epochs = st.slider("LSTM Epochs", 50, 500, 100, 50)
            lstm_batch_size = st.selectbox("LSTM Batch Size", [16, 32, 64, 128], index=1)
            lstm_units = st.slider("LSTM Units", 32, 256, 64, 32)
    
    # Run forecasting
    if st.button("🚀 Run Forecast", type="primary", use_container_width=True):
        with st.spinner("Training forecasting models and generating predictions..."):
            
            results = {}
            predictions = []
            
            # Create forecast dates
            forecast_dates = pd.date_range(
                start=df_forecast['ds'].max() + pd.Timedelta(days=30),
                periods=forecast_horizon,
                freq='M'
            )
            
            # Create tabs for different views
            tab_forecast, tab_components, tab_performance, tab_ensemble = st.tabs([
                "📊 Forecast Visualization",
                "🔍 Model Components",
                "📋 Performance Metrics",
                "🎲 Ensemble Forecast"
            ])
            
            # Run selected models
            if "Prophet" in selected_models:
                with st.status("📈 Training Prophet model...", expanded=False) as status:
                    prophet_results = run_prophet_forecast(
                        df_forecast, 
                        forecast_horizon,
                        confidence_level,
                        seasonality_mode,
                        changepoint_prior
                    )
                    if 'error' not in prophet_results:
                        results['Prophet'] = prophet_results
                        forecast_future = prophet_results['forecast_df']
                        forecast_future = forecast_future[forecast_future['ds'] > df_forecast['ds'].max()]
                        predictions.append(forecast_future['yhat'])
                        status.update(label="✅ Prophet model completed", state="complete")
                    else:
                        status.update(label="❌ Prophet model failed", state="error")
            
            if "ARIMA/SARIMA" in selected_models:
                with st.status("📊 Training ARIMA model...", expanded=False) as status:
                    arima_results = run_arima_forecast(
                        df_forecast,
                        forecast_horizon,
                        (arima_order_p, arima_order_d, arima_order_q)
                    )
                    if 'error' not in arima_results:
                        results['ARIMA'] = arima_results
                        predictions.append(arima_results['forecast'])
                        status.update(label="✅ ARIMA model completed", state="complete")
                    else:
                        status.update(label="❌ ARIMA model failed", state="error")
            
            if "LSTM Neural Network" in selected_models:
                with st.status("🧠 Training LSTM neural network...", expanded=False) as status:
                    lstm_results = run_lstm_forecast(
                        df_forecast,
                        forecast_horizon,
                        lstm_epochs,
                        lstm_batch_size,
                        lstm_units
                    )
                    if 'error' not in lstm_results:
                        results['LSTM'] = lstm_results
                        predictions.append(lstm_results['forecast'])
                        status.update(label="✅ LSTM model completed", state="complete")
                    else:
                        status.update(label="❌ LSTM model failed", state="error")
            
            with tab_forecast:
                display_forecast_visualization(df_forecast, results, forecast_dates)
            
            with tab_components:
                display_model_components(results)
            
            with tab_performance:
                display_performance_metrics(results, df_forecast)
            
            with tab_ensemble:
                if len(predictions) > 1:
                    display_ensemble_forecast(df_forecast, predictions, results, forecast_dates)
                else:
                    st.info("Select multiple models to generate ensemble forecast")
            
            return results
    
    return None

def prepare_forecast_data(df):
    """Prepare data for time series forecasting"""
    
    df_fc = df.copy()
    
    # Ensure we have the required columns
    if 'Date' not in df_fc.columns or 'Free_Cash_Flow_USD_M' not in df_fc.columns:
        st.error("Required columns (Date, Free_Cash_Flow_USD_M) not found")
        return None
    
    # Create Prophet-compatible dataframe
    df_prophet = pd.DataFrame({
        'ds': pd.to_datetime(df_fc['Date']),
        'y': df_fc['Free_Cash_Flow_USD_M'].astype(float)
    })
    
    # Remove any NaN values
    df_prophet = df_prophet.dropna()
    
    # Sort by date
    df_prophet = df_prophet.sort_values('ds').reset_index(drop=True)
    
    # Ensure no duplicate dates
    df_prophet = df_prophet.groupby('ds')['y'].mean().reset_index()
    
    return df_prophet

def run_prophet_forecast(df, forecast_horizon, confidence_level, seasonality_mode, changepoint_prior):
    """Run Prophet forecasting model"""
    
    results = {
        'model': 'Prophet',
        'forecast_df': None,
        'model_object': None,
        'metrics': {},
        'components': {}
    }
    
    try:
        # Initialize and configure Prophet
        model = Prophet(
            seasonality_mode=seasonality_mode,
            changepoint_prior_scale=changepoint_prior,
            interval_width=confidence_level/100,
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False
        )
        
        # Add custom seasonalities
        model.add_seasonality(
            name='quarterly',
            period=91.25,
            fourier_order=5
        )
        
        # Add country holidays if needed
        model.add_country_holidays(country_name='US')
        
        # Fit the model
        model.fit(df)
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=forecast_horizon, freq='M')
        
        # Predict
        forecast = model.predict(future)
        
        # Ensure forecast columns are Series
        for col in ['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend', 'yearly', 'quarterly']:
            if col in forecast.columns:
                forecast[col] = pd.Series(forecast[col].values)
        
        # Calculate metrics on training data
        train_predictions = forecast.iloc[:len(df)]['yhat'].values
        train_actual = df['y'].values
        
        # Ensure same length
        min_len = min(len(train_actual), len(train_predictions))
        train_actual = train_actual[:min_len]
        train_predictions = train_predictions[:min_len]
        
        mae = mean_absolute_error(train_actual, train_predictions)
        rmse = np.sqrt(mean_squared_error(train_actual, train_predictions))
        mape = np.mean(np.abs((train_actual - train_predictions) / np.abs(train_actual))) * 100
        r2 = r2_score(train_actual, train_predictions)
        
        results['forecast_df'] = forecast
        results['model_object'] = model
        results['metrics'] = {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'R2': r2
        }
        
        # Extract components for visualization - ensure they are Series
        forecast_tail = forecast.tail(forecast_horizon).copy()
        results['components'] = {
            'trend': pd.DataFrame({
                'ds': pd.Series(forecast_tail['ds'].values),
                'trend': pd.Series(forecast_tail['trend'].values)
            }),
            'yearly': pd.DataFrame({
                'ds': pd.Series(forecast_tail['ds'].values),
                'yearly': pd.Series(forecast_tail['yearly'].values)
            }),
            'quarterly': pd.DataFrame({
                'ds': pd.Series(forecast_tail['ds'].values),
                'quarterly': pd.Series(forecast_tail['quarterly'].values)
            })
        }
        
    except Exception as e:
        results['error'] = str(e)
    
    return results

def run_arima_forecast(df, forecast_horizon, order):
    """Run ARIMA/SARIMA forecasting model"""
    
    results = {
        'model': 'ARIMA',
        'forecast': None,
        'conf_int': None,
        'model_object': None,
        'metrics': {},
        'stationarity_test': {}
    }
    
    try:
        # Extract time series
        ts = df.set_index('ds')['y']
        
        # Perform Augmented Dickey-Fuller test for stationarity
        adf_result = adfuller(ts.dropna(), autolag='AIC')
        results['stationarity_test'] = {
            'ADF Statistic': adf_result[0],
            'p-value': adf_result[1],
            'Critical Values': adf_result[4],
            'Stationary': adf_result[1] < 0.05
        }
        
        # Fit ARIMA model
        model = ARIMA(ts, order=order)
        fitted_model = model.fit()
        
        # Generate forecast
        forecast_result = fitted_model.get_forecast(steps=forecast_horizon)
        forecast = pd.Series(forecast_result.predicted_mean.values)
        conf_int = pd.DataFrame(forecast_result.conf_int().values)
        
        # Calculate metrics on training data
        train_predictions = pd.Series(fitted_model.fittedvalues.values)
        train_actual = pd.Series(ts.values[-len(train_predictions):])
        
        # Remove NaN values
        mask = ~(np.isnan(train_actual) | np.isnan(train_predictions))
        train_actual = train_actual[mask]
        train_predictions = train_predictions[mask]
        
        if len(train_actual) > 0:
            mae = mean_absolute_error(train_actual, train_predictions)
            rmse = np.sqrt(mean_squared_error(train_actual, train_predictions))
            mape = np.mean(np.abs((train_actual - train_predictions) / np.abs(train_actual))) * 100
            r2 = r2_score(train_actual, train_predictions)
            
            results['metrics'] = {
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape,
                'R2': r2,
                'AIC': fitted_model.aic,
                'BIC': fitted_model.bic
            }
        
        results['forecast'] = forecast
        results['conf_int'] = conf_int
        results['model_object'] = fitted_model
        
    except Exception as e:
        results['error'] = str(e)
    
    return results

def run_lstm_forecast(df, forecast_horizon, epochs, batch_size, units):
    """Run LSTM neural network forecasting model"""
    
    results = {
        'model': 'LSTM',
        'forecast': None,
        'model_object': None,
        'metrics': {},
        'training_history': None
    }
    
    try:
        # Prepare data for LSTM
        ts = df.set_index('ds')['y'].values.reshape(-1, 1)
        
        # Normalize data
        scaler = MinMaxScaler(feature_range=(0, 1))
        ts_scaled = scaler.fit_transform(ts)
        
        # Create sequences
        def create_sequences(data, seq_length=12):
            X, y = [], []
            for i in range(len(data) - seq_length):
                X.append(data[i:i + seq_length])
                y.append(data[i + seq_length])
            return np.array(X), np.array(y)
        
        seq_length = min(12, max(3, len(ts_scaled) // 3))
        X, y = create_sequences(ts_scaled, seq_length)
        
        if len(X) == 0:
            raise ValueError("Not enough data for LSTM sequences")
        
        # Split into train and validation
        train_size = max(1, int(len(X) * 0.8))
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        # Build LSTM model
        model = Sequential([
            LSTM(units, return_sequences=True, input_shape=(seq_length, 1)),
            Dropout(0.2),
            LSTM(units // 2, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if len(X_val) > 0 else (X_train, y_train),
            epochs=epochs,
            batch_size=min(batch_size, len(X_train)),
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Generate forecast
        last_sequence = ts_scaled[-seq_length:]
        forecast_scaled = []
        
        for _ in range(forecast_horizon):
            next_pred = model.predict(last_sequence.reshape(1, seq_length, 1), verbose=0)
            forecast_scaled.append(next_pred[0, 0])
            last_sequence = np.roll(last_sequence, -1)
            last_sequence[-1] = next_pred
        
        # Inverse transform forecast
        forecast = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1)).flatten()
        forecast = pd.Series(forecast)
        
        # Calculate metrics on validation set if available
        if len(X_val) > 0:
            val_predictions = model.predict(X_val, verbose=0)
            val_predictions = scaler.inverse_transform(val_predictions)
            y_val_actual = scaler.inverse_transform(y_val)
            
            mae = mean_absolute_error(y_val_actual, val_predictions)
            rmse = np.sqrt(mean_squared_error(y_val_actual, val_predictions))
            mape = np.mean(np.abs((y_val_actual - val_predictions) / np.abs(y_val_actual))) * 100
            r2 = r2_score(y_val_actual, val_predictions)
            
            results['metrics'] = {
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape,
                'R2': r2,
                'Training Loss': history.history['loss'][-1],
                'Validation Loss': history.history['val_loss'][-1]
            }
        
        results['forecast'] = forecast
        results['model_object'] = model
        results['scaler'] = scaler
        results['training_history'] = history.history
        
    except Exception as e:
        results['error'] = str(e)
    
    return results

def display_forecast_visualization(df, results, forecast_dates):
    """Display comprehensive forecast visualizations"""
    
    st.markdown("### 📈 Forecast Visualization")
    
    # Determine number of models to display
    valid_models = [m for m in results.keys() if 'error' not in results[m]]
    n_models = len(valid_models)
    
    if n_models == 0:
        st.warning("No successful models to display")
        return
    
    # Create subplot grid based on number of models
    if n_models <= 2:
        rows, cols = 1, 2
    else:
        rows, cols = 2, 2
    
    # Create subplot titles
    subplot_titles = []
    for model in valid_models[:4]:  # Max 4 subplots
        subplot_titles.append(f"{model} Forecast")
    
    # Add comparison subplot if we have space
    if n_models > 1 and len(subplot_titles) < rows * cols:
        subplot_titles.append("Model Comparison")
    
    fig = make_subplots(
        rows=rows, 
        cols=cols,
        subplot_titles=subplot_titles,
        vertical_spacing=0.15,
        horizontal_spacing=0.15
    )
    
    # Add historical data to all subplots
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            fig.add_trace(
                go.Scatter(
                    x=df['ds'],
                    y=df['y'],
                    name="Historical",
                    line=dict(color='#1e293b', width=2),
                    showlegend=(i == 1 and j == 1)
                ),
                row=i, col=j
            )
    
    # Add model forecasts to individual subplots
    current_row, current_col = 1, 1
    
    for model_name in valid_models:
        model_results = results[model_name]
        
        if model_name == 'Prophet':
            forecast = model_results['forecast_df']
            forecast_future = forecast[forecast['ds'] > df['ds'].max()]
            
            # Add forecast line
            fig.add_trace(
                go.Scatter(
                    x=forecast_future['ds'],
                    y=forecast_future['yhat'],
                    name=f"{model_name} Forecast",
                    line=dict(color='#3b82f6', width=2),
                    showlegend=True
                ),
                row=current_row, col=current_col
            )
            
            # Add confidence interval
            if len(forecast_future) > 0:
                # Create proper Series objects for concatenation
                x_upper = pd.Series(forecast_future['ds'].values)
                x_lower = pd.Series(forecast_future['ds'].values[::-1])
                y_upper = pd.Series(forecast_future['yhat_upper'].values)
                y_lower = pd.Series(forecast_future['yhat_lower'].values[::-1])
                
                fig.add_trace(
                    go.Scatter(
                        x=pd.concat([x_upper, x_lower]),
                        y=pd.concat([y_upper, y_lower]),
                        fill='toself',
                        fillcolor='rgba(59, 130, 246, 0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name=f"{model_name} CI",
                        showlegend=True
                    ),
                    row=current_row, col=current_col
                )
        
        elif model_name == 'ARIMA' and model_results['forecast'] is not None:
            # Add forecast line
            fig.add_trace(
                go.Scatter(
                    x=forecast_dates[:len(model_results['forecast'])],
                    y=model_results['forecast'],
                    name=f"{model_name} Forecast",
                    line=dict(color='#ef4444', width=2),
                    showlegend=True
                ),
                row=current_row, col=current_col
            )
            
            # Add confidence interval
            if model_results['conf_int'] is not None:
                conf_int = model_results['conf_int']
                dates = forecast_dates[:len(conf_int)]
                
                # Create proper Series objects
                x_upper = pd.Series(dates.values)
                x_lower = pd.Series(dates.values[::-1])
                y_upper = pd.Series(conf_int.iloc[:, 1].values)
                y_lower = pd.Series(conf_int.iloc[:, 0].values[::-1])
                
                fig.add_trace(
                    go.Scatter(
                        x=pd.concat([x_upper, x_lower]),
                        y=pd.concat([y_upper, y_lower]),
                        fill='toself',
                        fillcolor='rgba(239, 68, 68, 0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name=f"{model_name} CI",
                        showlegend=True
                    ),
                    row=current_row, col=current_col
                )
        
        elif model_name == 'LSTM' and model_results['forecast'] is not None:
            fig.add_trace(
                go.Scatter(
                    x=forecast_dates[:len(model_results['forecast'])],
                    y=model_results['forecast'],
                    name=f"{model_name} Forecast",
                    line=dict(color='#8b5cf6', width=2),
                    showlegend=True
                ),
                row=current_row, col=current_col
            )
        
        # Update subplot position
        current_col += 1
        if current_col > cols:
            current_col = 1
            current_row += 1
    
    # Add comparison subplot if we have multiple models and space
    if n_models > 1 and current_row <= rows and current_col <= cols:
        for model_name in valid_models:
            model_results = results[model_name]
            
            if model_name == 'Prophet':
                forecast = model_results['forecast_df']
                forecast_future = forecast[forecast['ds'] > df['ds'].max()]
                y_values = forecast_future['yhat']
                dates = forecast_future['ds']
                color = '#3b82f6'
            elif model_name == 'ARIMA' and model_results['forecast'] is not None:
                y_values = model_results['forecast']
                dates = forecast_dates[:len(y_values)]
                color = '#ef4444'
            elif model_name == 'LSTM' and model_results['forecast'] is not None:
                y_values = model_results['forecast']
                dates = forecast_dates[:len(y_values)]
                color = '#8b5cf6'
            else:
                continue
            
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=y_values,
                    name=model_name,
                    line=dict(color=color, width=2),
                    showlegend=True
                ),
                row=current_row, col=current_col
            )
    
    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    fig.update_xaxes(gridcolor='#e2e8f0', tickangle=45)
    fig.update_yaxes(gridcolor='#e2e8f0', title="Free Cash Flow (USD M)")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Individual model details
    if results:
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        if valid_results:
            st.markdown("### 📊 Model Details")
            
            tabs = st.tabs([f"📈 {name}" for name in valid_results.keys()])
            
            for tab, (model_name, model_results) in zip(tabs, valid_results.items()):
                with tab:
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.markdown("**Forecast Values**")
                        
                        if model_name == 'Prophet':
                            forecast = model_results['forecast_df']
                            forecast_future = forecast[forecast['ds'] > df['ds'].max()]
                            forecast_table = pd.DataFrame({
                                'Date': forecast_future['ds'].dt.strftime('%Y-%m'),
                                'Forecast (USD M)': forecast_future['yhat'].round(2),
                                'Lower CI': forecast_future['yhat_lower'].round(2),
                                'Upper CI': forecast_future['yhat_upper'].round(2)
                            })
                            st.dataframe(forecast_table, use_container_width=True, hide_index=True)
                        
                        elif model_name == 'ARIMA' and model_results['forecast'] is not None:
                            dates = forecast_dates[:len(model_results['forecast'])]
                            forecast_table = pd.DataFrame({
                                'Date': dates.strftime('%Y-%m'),
                                'Forecast (USD M)': model_results['forecast'].round(2),
                                'Lower CI': model_results['conf_int'].iloc[:, 0].round(2),
                                'Upper CI': model_results['conf_int'].iloc[:, 1].round(2)
                            })
                            st.dataframe(forecast_table, use_container_width=True, hide_index=True)
                        
                        elif model_name == 'LSTM' and model_results['forecast'] is not None:
                            dates = forecast_dates[:len(model_results['forecast'])]
                            forecast_table = pd.DataFrame({
                                'Date': dates.strftime('%Y-%m'),
                                'Forecast (USD M)': model_results['forecast'].round(2)
                            })
                            st.dataframe(forecast_table, use_container_width=True, hide_index=True)
                    
                    with col2:
                        st.markdown("**Model Performance**")
                        if model_results.get('metrics'):
                            metrics_df = pd.DataFrame(
                                list(model_results['metrics'].items()),
                                columns=['Metric', 'Value']
                            )
                            
                            # Format metrics
                            metrics_df['Value'] = metrics_df.apply(
                                lambda x: f"{x['Value']:.3f}" if isinstance(x['Value'], (int, float)) else x['Value'],
                                axis=1
                            )
                            
                            st.dataframe(metrics_df, use_container_width=True, hide_index=True)

def display_model_components(results):
    """Display model components and decomposition"""
    
    st.markdown("### 🔍 Model Components Analysis")
    
    for model_name, model_results in results.items():
        if 'error' not in model_results and model_name == 'Prophet':
            st.markdown(f"#### 📊 Prophet Model Components")
            
            try:
                # Plot components
                fig_components = plot_components_plotly(
                    model_results['model_object'], 
                    model_results['forecast_df']
                )
                
                fig_components.update_layout(
                    height=500,
                    plot_bgcolor='white',
                    paper_bgcolor='white'
                )
                
                st.plotly_chart(fig_components, use_container_width=True)
                
                # Component interpretation
                if 'components' in model_results and model_results['components']:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        trend_data = model_results['components']['trend']
                        trend_change = trend_data['trend'].diff().mean()
                        
                        st.markdown(f"""
                        <div style="background: white; padding: 1rem; border-radius: 8px;
                                    border: 1px solid #e2e8f0;">
                            <h5 style="color: #1e293b; margin-bottom: 0.5rem;">📈 Trend Interpretation</h5>
                            <p style="color: #475569; font-size: 0.95rem;">
                                The trend component shows the long-term direction of cash flows.
                                {'Positive' if trend_change > 0 else 'Negative'} trend detected 
                                with average monthly change of ${abs(trend_change):.2f}M.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        yearly_seasonality = model_results['components']['yearly']['yearly'].mean()
                        quarterly_seasonality = model_results['components']['quarterly']['quarterly'].mean()
                        
                        st.markdown(f"""
                        <div style="background: white; padding: 1rem; border-radius: 8px;
                                    border: 1px solid #e2e8f0;">
                            <h5 style="color: #1e293b; margin-bottom: 0.5rem;">📅 Seasonality Impact</h5>
                            <p style="color: #475569; font-size: 0.95rem;">
                                • Yearly seasonality: ±${abs(yearly_seasonality):.2f}M<br>
                                • Quarterly seasonality: ±${abs(quarterly_seasonality):.2f}M
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
            
            except Exception as e:
                st.warning(f"Could not display Prophet components: {str(e)}")
        
        elif model_name == 'ARIMA' and 'stationarity_test' in model_results:
            st.markdown(f"#### 📊 ARIMA Model Diagnostics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Stationarity Test Results**")
                stationarity_df = pd.DataFrame(
                    list(model_results['stationarity_test'].items()),
                    columns=['Test', 'Result']
                )
                
                st.dataframe(stationarity_df, use_container_width=True, hide_index=True)
            
            with col2:
                st.markdown("**Model Summary**")
                if model_results['model_object']:
                    st.text(str(model_results['model_object']).split('\n')[0][:200] + "...")

def display_performance_metrics(results, df):
    """Display model performance comparison"""
    
    st.markdown("### 📋 Model Performance Comparison")
    
    # Create comparison dataframe
    comparison_data = []
    
    for model_name, model_results in results.items():
        if 'error' not in model_results and model_results.get('metrics'):
            metrics = model_results['metrics']
            comparison_data.append({
                'Model': model_name,
                'MAE (USD M)': metrics.get('MAE', np.nan),
                'RMSE (USD M)': metrics.get('RMSE', np.nan),
                'MAPE (%)': metrics.get('MAPE', np.nan),
                'R²': metrics.get('R2', np.nan),
                'AIC': metrics.get('AIC', np.nan),
                'BIC': metrics.get('BIC', np.nan)
            })
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        
        # Style the dataframe
        styled_df = comparison_df.style.format({
            'MAE (USD M)': '${:.2f}M',
            'RMSE (USD M)': '${:.2f}M',
            'MAPE (%)': '{:.1f}%',
            'R²': '{:.3f}',
            'AIC': '{:.0f}',
            'BIC': '{:.0f}'
        }).background_gradient(subset=['R²'], cmap='RdYlGn')
        
        st.dataframe(styled_df, use_container_width=True)
        
        # Best model identification
        if 'R²' in comparison_df.columns and 'MAPE (%)' in comparison_df.columns:
            best_r2_idx = comparison_df['R²'].idxmax()
            best_r2 = comparison_df.loc[best_r2_idx]
            
            best_mape_idx = comparison_df['MAPE (%)'].idxmin()
            best_mape = comparison_df.loc[best_mape_idx]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div style="background: #dcfce7; padding: 1rem; border-radius: 8px;
                            border-left: 4px solid #22c55e;">
                    <h5 style="color: #166534; margin-bottom: 0.25rem;">🏆 Best Predictive Power</h5>
                    <p style="color: #166534; margin: 0;">
                        <strong>{best_r2['Model']}</strong> (R² = {best_r2['R²']:.3f})
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div style="background: #dcfce7; padding: 1rem; border-radius: 8px;
                            border-left: 4px solid #22c55e;">
                    <h5 style="color: #166534; margin-bottom: 0.25rem;">🎯 Lowest Error Rate</h5>
                    <p style="color: #166534; margin: 0;">
                        <strong>{best_mape['Model']}</strong> (MAPE = {best_mape['MAPE (%)']:.1f}%)
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        # Performance visualization
        st.markdown("### 📊 Performance Visualization")
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Model Comparison - Error Metrics", "Model Comparison - R² Score"),
            horizontal_spacing=0.2
        )
        
        # Error metrics bar chart
        metrics_to_plot = ['MAE (USD M)', 'RMSE (USD M)', 'MAPE (%)']
        colors = ['#3b82f6', '#8b5cf6', '#ef4444']
        
        for metric, color in zip(metrics_to_plot, colors):
            if metric in comparison_df.columns:
                fig.add_trace(
                    go.Bar(
                        name=metric.replace(' (USD M)', '').replace(' (%)', ''),
                        x=comparison_df['Model'],
                        y=comparison_df[metric],
                        marker_color=color,
                        text=comparison_df[metric].round(1),
                        textposition='auto',
                    ),
                    row=1, col=1
                )
        
        # R² score bar chart
        if 'R²' in comparison_df.columns:
            fig.add_trace(
                go.Bar(
                    name='R² Score',
                    x=comparison_df['Model'],
                    y=comparison_df['R²'],
                    marker_color='#10b981',
                    text=comparison_df['R²'].round(3),
                    textposition='auto',
                ),
                row=1, col=2
            )
        
        fig.update_layout(
            height=400,
            showlegend=True,
            plot_bgcolor='white',
            paper_bgcolor='white',
            barmode='group'
        )
        
        fig.update_xaxes(gridcolor='#e2e8f0')
        fig.update_yaxes(gridcolor='#e2e8f0')
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No performance metrics available for comparison")

def display_ensemble_forecast(df, predictions, results, forecast_dates):
    """Display ensemble forecast combining multiple models"""
    
    st.markdown("### 🎲 Ensemble Forecast (Weighted Average)")
    
    # Calculate ensemble weights based on model performance
    weights = {}
    total_weight = 0
    
    for model_name, model_results in results.items():
        if 'error' not in model_results and model_results.get('metrics'):
            r2 = model_results['metrics'].get('R2', 0)
            if not np.isnan(r2) and r2 > 0:
                weights[model_name] = max(r2, 0.1)  # Minimum weight of 0.1
                total_weight += weights[model_name]
    
    # Normalize weights
    if total_weight > 0:
        for model_name in weights:
            weights[model_name] /= total_weight
    else:
        # Equal weights if no performance metrics
        n_models = len([p for p in predictions if p is not None])
        if n_models > 0:
            weight = 1.0 / n_models
            for model_name in results.keys():
                if 'error' not in results[model_name]:
                    weights[model_name] = weight
    
    # Create ensemble forecast
    ensemble_forecast = None
    min_length = len(forecast_dates)
    
    for model_name, pred in zip(results.keys(), predictions):
        if model_name in weights and pred is not None:
            # Convert to numpy array and ensure proper length
            if isinstance(pred, pd.Series):
                pred_array = pred.iloc[:min_length].values
            else:
                pred_array = np.array(pred)[:min_length]
            
            if ensemble_forecast is None:
                ensemble_forecast = pred_array * weights[model_name]
            else:
                # Align lengths
                common_length = min(len(ensemble_forecast), len(pred_array))
                ensemble_forecast = ensemble_forecast[:common_length] + (pred_array[:common_length] * weights[model_name])
    
    if ensemble_forecast is not None:
        # Create visualization
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=df['ds'],
            y=df['y'],
            name="Historical",
            line=dict(color='#1e293b', width=3),
            mode='lines'
        ))
        
        # Individual model forecasts
        colors = ['#3b82f6', '#ef4444', '#8b5cf6']
        color_idx = 0
        
        for model_name, pred in zip(results.keys(), predictions):
            if model_name in weights and pred is not None and color_idx < len(colors):
                # Convert to numpy array and ensure proper length
                if isinstance(pred, pd.Series):
                    pred_values = pred.iloc[:len(forecast_dates)].values
                else:
                    pred_values = np.array(pred)[:len(forecast_dates)]
                
                dates = forecast_dates[:len(pred_values)]
                
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=pred_values,
                    name=f"{model_name} (Weight: {weights.get(model_name, 0):.1%})",
                    line=dict(color=colors[color_idx], width=1.5, dash='dash'),
                    opacity=0.7
                ))
                color_idx += 1
        
        # Ensemble forecast
        ensemble_dates = forecast_dates[:len(ensemble_forecast)]
        
        fig.add_trace(go.Scatter(
            x=ensemble_dates,
            y=ensemble_forecast,
            name="🎯 Ensemble Forecast",
            line=dict(color='#f59e0b', width=4),
            mode='lines+markers',
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title="Ensemble Forecast (Weighted by Model Performance)",
            xaxis_title="Date",
            yaxis_title="Free Cash Flow (USD Millions)",
            height=500,
            hovermode='x unified',
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        fig.update_xaxes(gridcolor='#e2e8f0')
        fig.update_yaxes(gridcolor='#e2e8f0')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Ensemble forecast table
        st.markdown("#### 📊 Ensemble Forecast Values")
        
        ensemble_table = pd.DataFrame({
            'Date': ensemble_dates.strftime('%Y-%m'),
            'Ensemble Forecast (USD M)': ensemble_forecast.round(2),
            'Lower Bound (80%)': (ensemble_forecast * 0.85).round(2),
            'Upper Bound (80%)': (ensemble_forecast * 1.15).round(2)
        })
        
        st.dataframe(ensemble_table, use_container_width=True, hide_index=True)
        
        # Download ensemble forecast
        csv = ensemble_table.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Ensemble Forecast",
            data=csv,
            file_name=f"ensemble_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True
        )
    else:
        st.warning("Could not generate ensemble forecast")

def calculate_forecast_accuracy(actual, predicted):
    """Calculate comprehensive forecast accuracy metrics"""
    
    metrics = {}
    
    # Remove NaN values
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    actual_clean = actual[mask]
    predicted_clean = predicted[mask]
    
    if len(actual_clean) > 0:
        # Mean Absolute Error
        metrics['MAE'] = mean_absolute_error(actual_clean, predicted_clean)
        
        # Root Mean Square Error
        metrics['RMSE'] = np.sqrt(mean_squared_error(actual_clean, predicted_clean))
        
        # Mean Absolute Percentage Error
        with np.errstate(divide='ignore', invalid='ignore'):
            mape = np.mean(np.abs((actual_clean - predicted_clean) / np.abs(actual_clean))) * 100
            metrics['MAPE'] = mape if not np.isnan(mape) else np.nan
        
        # R-squared
        metrics['R2'] = r2_score(actual_clean, predicted_clean)
        
        # Symmetric Mean Absolute Percentage Error
        with np.errstate(divide='ignore', invalid='ignore'):
            smape = np.mean(2 * np.abs(actual_clean - predicted_clean) / 
                           (np.abs(actual_clean) + np.abs(predicted_clean))) * 100
            metrics['sMAPE'] = smape if not np.isnan(smape) else np.nan
        
        # Mean Absolute Scaled Error
        if len(actual_clean) > 1:
            naive_forecast = actual_clean[:-1]
            naive_error = np.mean(np.abs(actual_clean[1:] - naive_forecast))
            if naive_error > 0:
                metrics['MASE'] = metrics['MAE'] / naive_error
            else:
                metrics['MASE'] = np.nan
    
    return metrics

def validate_forecast_assumptions(df):
    """Validate time series assumptions for forecasting"""
    
    validation_results = {}
    
    try:
        # Check for sufficient data
        validation_results['sufficient_data'] = len(df) >= 24
        
        # Check for seasonality
        if len(df) >= 24:
            ts = df.set_index('ds')['y']
            decomposition = seasonal_decompose(ts, model='additive', period=12, extrapolate_trend='freq')
            seasonal_strength = 1 - np.var(decomposition.resid) / np.var(decomposition.seasonal + decomposition.resid)
            validation_results['seasonal_strength'] = seasonal_strength
            validation_results['has_seasonality'] = seasonal_strength > 0.3
        
        # Check for trend
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            range(len(df)), 
            df['y'].values
        )
        validation_results['trend_slope'] = slope
        validation_results['trend_p_value'] = p_value
        validation_results['has_significant_trend'] = p_value < 0.05
        
        # Check for outliers
        Q1 = df['y'].quantile(0.25)
        Q3 = df['y'].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df['y'] < Q1 - 1.5 * IQR) | (df['y'] > Q3 + 1.5 * IQR)]
        validation_results['outlier_count'] = len(outliers)
        validation_results['has_outliers'] = len(outliers) > 0
        
    except Exception as e:
        validation_results['error'] = str(e)
    
    return validation_results

# Test function for standalone testing
if __name__ == "__main__":
    print("Time Series Forecasting module ready for integration")
    print("Available models: Prophet, ARIMA/SARIMA, LSTM Neural Network")
    print("Ensemble forecasting with weighted averaging")