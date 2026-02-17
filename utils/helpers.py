"""
Utilities Helper Module for Corporate Cash Flow Stress Testing Platform
Common utility functions used across all modules
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import hashlib
from typing import Any, Dict, List, Optional, Union, Tuple
import warnings
warnings.filterwarnings('ignore')

def format_currency(value: float, include_sign: bool = False) -> str:
    """
    Format a number as currency string
    
    Args:
        value: Numeric value to format
        include_sign: Whether to include + sign for positive values
    
    Returns:
        Formatted currency string
    """
    if pd.isna(value) or value is None:
        return "N/A"
    
    sign = "+" if include_sign and value > 0 else ""
    return f"{sign}${value:,.2f}M"

def format_percentage(value: float, decimals: int = 1) -> str:
    """
    Format a number as percentage string
    
    Args:
        value: Numeric value to format (0.1 = 10%)
        decimals: Number of decimal places
    
    Returns:
        Formatted percentage string
    """
    if pd.isna(value) or value is None:
        return "N/A"
    
    return f"{value * 100:.{decimals}f}%"

def format_date(date_value, format_str: str = '%Y-%m-%d') -> str:
    """
    Format date consistently
    
    Args:
        date_value: Date value (datetime, string, etc.)
        format_str: Output format
    
    Returns:
        Formatted date string
    """
    if pd.isna(date_value) or date_value is None:
        return "N/A"
    
    try:
        if isinstance(date_value, str):
            date_obj = pd.to_datetime(date_value)
        else:
            date_obj = date_value
        
        return date_obj.strftime(format_str)
    except:
        return str(date_value)

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safe division with zero denominator
    
    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value if denominator is zero
    
    Returns:
        Division result or default
    """
    if denominator == 0 or pd.isna(denominator) or denominator is None:
        return default
    return numerator / denominator

def calculate_cagr(start_value: float, end_value: float, periods: int) -> float:
    """
    Calculate Compound Annual Growth Rate
    
    Args:
        start_value: Beginning value
        end_value: Ending value
        periods: Number of periods
    
    Returns:
        CAGR as decimal
    """
    if start_value <= 0 or periods <= 0:
        return 0.0
    
    return (end_value / start_value) ** (1 / periods) - 1

def detect_outliers_iqr(data: pd.Series, multiplier: float = 1.5) -> pd.Series:
    """
    Detect outliers using IQR method
    
    Args:
        data: Series of numeric values
        multiplier: IQR multiplier (1.5 for mild, 3 for extreme)
    
    Returns:
        Boolean series where True indicates outlier
    """
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    return (data < lower_bound) | (data > upper_bound)

def calculate_moving_average(data: pd.Series, window: int) -> pd.Series:
    """
    Calculate moving average with proper handling
    
    Args:
        data: Series of values
        window: Window size
    
    Returns:
        Moving average series
    """
    if len(data) < window:
        return pd.Series([np.nan] * len(data), index=data.index)
    
    return data.rolling(window=window, min_periods=1).mean()

def calculate_moving_std(data: pd.Series, window: int) -> pd.Series:
    """
    Calculate moving standard deviation
    
    Args:
        data: Series of values
        window: Window size
    
    Returns:
        Moving standard deviation series
    """
    if len(data) < window:
        return pd.Series([np.nan] * len(data), index=data.index)
    
    return data.rolling(window=window, min_periods=1).std()

def create_metric_card(title: str, value: str, delta: Optional[str] = None, 
                       icon: str = "📊", color: str = "#1e293b") -> str:
    """
    Create HTML for a metric card
    
    Args:
        title: Card title
        value: Main value to display
        delta: Change indicator
        icon: Emoji or icon
        color: Accent color
    
    Returns:
        HTML string for the card
    """
    delta_html = ""
    if delta:
        delta_class = "positive" if "+" in delta else "negative" if "-" in delta else ""
        delta_html = f'<div class="metric-delta {delta_class}">{delta}</div>'
    
    return f"""
    <div class="metric-card" style="border-left: 4px solid {color};">
        <div class="metric-icon">{icon}</div>
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """

def create_status_indicator(status: str, text: str) -> str:
    """
    Create HTML for a status indicator
    
    Args:
        status: 'success', 'warning', 'error', 'info'
        text: Status text
    
    Returns:
        HTML string for the indicator
    """
    colors = {
        'success': {'bg': '#dcfce7', 'text': '#166534', 'border': '#10b981'},
        'warning': {'bg': '#fef3c7', 'text': '#92400e', 'border': '#f59e0b'},
        'error': {'bg': '#fee2e2', 'text': '#991b1b', 'border': '#ef4444'},
        'info': {'bg': '#dbeafe', 'text': '#1e40af', 'border': '#3b82f6'}
    }
    
    color = colors.get(status, colors['info'])
    
    return f"""
    <div style="display: inline-block; background: {color['bg']}; color: {color['text']};
                padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.8rem;
                border: 1px solid {color['border']}; font-weight: 500;">
        {text}
    </div>
    """

def calculate_seasonal_indices(df: pd.DataFrame, value_col: str, date_col: str = 'Date') -> Dict:
    """
    Calculate seasonal indices for time series
    
    Args:
        df: DataFrame with time series
        value_col: Column name for values
        date_col: Column name for dates
    
    Returns:
        Dictionary with seasonal indices
    """
    df_copy = df.copy()
    df_copy['month'] = pd.to_datetime(df_copy[date_col]).dt.month
    df_copy['quarter'] = pd.to_datetime(df_copy[date_col]).dt.quarter
    
    # Monthly indices
    monthly_avg = df_copy.groupby('month')[value_col].mean()
    overall_avg = df_copy[value_col].mean()
    monthly_indices = (monthly_avg / overall_avg).to_dict()
    
    # Quarterly indices
    quarterly_avg = df_copy.groupby('quarter')[value_col].mean()
    quarterly_indices = (quarterly_avg / overall_avg).to_dict()
    
    return {
        'monthly': monthly_indices,
        'quarterly': quarterly_indices,
        'peak_month': monthly_avg.idxmax(),
        'trough_month': monthly_avg.idxmin(),
        'seasonality_strength': monthly_avg.std() / overall_avg if overall_avg != 0 else 0
    }

def calculate_correlation_matrix(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Calculate correlation matrix with proper handling
    
    Args:
        df: DataFrame
        columns: Specific columns to include (None = all numeric)
    
    Returns:
        Correlation matrix DataFrame
    """
    if columns:
        data = df[columns].select_dtypes(include=[np.number])
    else:
        data = df.select_dtypes(include=[np.number])
    
    return data.corr()

def find_strong_correlations(corr_matrix: pd.DataFrame, threshold: float = 0.7) -> List[Dict]:
    """
    Find strong correlations in matrix
    
    Args:
        corr_matrix: Correlation matrix
        threshold: Correlation threshold
    
    Returns:
        List of strong correlations with details
    """
    strong_corrs = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) >= threshold:
                strong_corrs.append({
                    'var1': corr_matrix.columns[i],
                    'var2': corr_matrix.columns[j],
                    'correlation': corr_value,
                    'strength': 'positive' if corr_value > 0 else 'negative',
                    'magnitude': abs(corr_value)
                })
    
    return sorted(strong_corrs, key=lambda x: x['magnitude'], reverse=True)

def calculate_performance_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict:
    """
    Calculate comprehensive performance metrics
    
    Args:
        actual: Actual values
        predicted: Predicted values
    
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    # Remove NaN values
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    actual_clean = actual[mask]
    predicted_clean = predicted[mask]
    
    if len(actual_clean) == 0:
        return {
            'mae': np.nan,
            'rmse': np.nan,
            'mape': np.nan,
            'r2': np.nan,
            'mase': np.nan
        }
    
    # Basic metrics
    mae = mean_absolute_error(actual_clean, predicted_clean)
    rmse = np.sqrt(mean_squared_error(actual_clean, predicted_clean))
    
    # MAPE (handle zeros)
    non_zero_mask = actual_clean != 0
    if non_zero_mask.any():
        mape = np.mean(np.abs((actual_clean[non_zero_mask] - predicted_clean[non_zero_mask]) 
                              / actual_clean[non_zero_mask])) * 100
    else:
        mape = np.nan
    
    # R-squared
    r2 = r2_score(actual_clean, predicted_clean)
    
    # Mean Absolute Scaled Error
    naive_forecast = actual_clean[:-1]
    naive_error = np.mean(np.abs(actual_clean[1:] - naive_forecast))
    if naive_error > 0:
        mase = mae / naive_error
    else:
        mase = np.nan
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2,
        'mase': mase
    }

def create_date_range(start_date: str, periods: int, freq: str = 'M') -> List[str]:
    """
    Create a date range string list
    
    Args:
        start_date: Start date string
        periods: Number of periods
        freq: Frequency ('D', 'W', 'M', 'Q', 'Y')
    
    Returns:
        List of date strings
    """
    dates = pd.date_range(start=start_date, periods=periods, freq=freq)
    return [d.strftime('%Y-%m-%d') for d in dates]

def resample_time_series(df: pd.DataFrame, date_col: str, value_col: str, 
                         freq: str = 'Q', agg_func: str = 'mean') -> pd.DataFrame:
    """
    Resample time series to different frequency
    
    Args:
        df: DataFrame with time series
        date_col: Date column name
        value_col: Value column name
        freq: Target frequency ('M', 'Q', 'Y')
        agg_func: Aggregation function ('mean', 'sum', 'min', 'max')
    
    Returns:
        Resampled DataFrame
    """
    df_copy = df.copy()
    df_copy[date_col] = pd.to_datetime(df_copy[date_col])
    df_copy.set_index(date_col, inplace=True)
    
    resampled = df_copy[value_col].resample(freq).agg(agg_func)
    return resampled.reset_index()

def calculate_volatility(returns: pd.Series, annualize: bool = True) -> float:
    """
    Calculate volatility of returns
    
    Args:
        returns: Series of returns
        annualize: Whether to annualize
    
    Returns:
        Volatility value
    """
    if len(returns) < 2:
        return 0.0
    
    vol = returns.std()
    
    if annualize:
        # Assuming monthly data
        vol *= np.sqrt(12)
    
    return vol

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    Calculate Sharpe ratio
    
    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate
    
    Returns:
        Sharpe ratio
    """
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    
    # Convert annual risk-free rate to monthly
    monthly_rf = (1 + risk_free_rate) ** (1/12) - 1
    
    excess_returns = returns - monthly_rf
    sharpe = excess_returns.mean() / returns.std() * np.sqrt(12)
    
    return sharpe

def calculate_max_drawdown(prices: pd.Series) -> Dict:
    """
    Calculate maximum drawdown
    
    Args:
        prices: Series of prices/values
    
    Returns:
        Dictionary with drawdown metrics
    """
    if len(prices) < 2:
        return {'max_drawdown': 0.0, 'duration': 0}
    
    # Calculate cumulative max
    cumulative_max = prices.expanding().max()
    
    # Calculate drawdown
    drawdown = (prices - cumulative_max) / cumulative_max * 100
    
    # Find maximum drawdown
    max_dd = drawdown.min()
    
    # Find recovery period
    drawdown_start = drawdown.idxmin()
    recovery = drawdown[drawdown_start:][drawdown[drawdown_start:] >= 0]
    
    if len(recovery) > 0:
        recovery_date = recovery.index[0]
        duration = (recovery_date - drawdown_start).days
    else:
        recovery_date = None
        duration = None
    
    return {
        'max_drawdown': max_dd,
        'max_drawdown_date': drawdown_start,
        'recovery_date': recovery_date,
        'duration_days': duration
    }

def calculate_beta(stock_returns: pd.Series, market_returns: pd.Series) -> float:
    """
    Calculate beta (sensitivity to market)
    
    Args:
        stock_returns: Company returns
        market_returns: Market returns
    
    Returns:
        Beta value
    """
    # Align series
    aligned = pd.concat([stock_returns, market_returns], axis=1).dropna()
    
    if len(aligned) < 2:
        return 1.0
    
    # Calculate covariance and variance
    covariance = aligned.iloc[:, 0].cov(aligned.iloc[:, 1])
    variance = aligned.iloc[:, 1].var()
    
    if variance == 0:
        return 1.0
    
    beta = covariance / variance
    return beta

def generate_session_id() -> str:
    """
    Generate unique session ID
    
    Returns:
        Unique session ID string
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    random_hash = hashlib.md5(str(np.random.random()).encode()).hexdigest()[:6]
    return f"session_{timestamp}_{random_hash}"

def save_uploaded_file(uploaded_file) -> str:
    """
    Save uploaded file to temporary location
    
    Args:
        uploaded_file: Streamlit uploaded file object
    
    Returns:
        Path to saved file
    """
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name

def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> Tuple[bool, str]:
    """
    Validate DataFrame has required columns and data
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if df is None or df.empty:
        return False, "DataFrame is empty"
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        return False, f"Missing required columns: {', '.join(missing_cols)}"
    
    return True, "Validation passed"

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean column names (lowercase, replace spaces)
    
    Args:
        df: DataFrame with columns to clean
    
    Returns:
        DataFrame with cleaned column names
    """
    df_copy = df.copy()
    df_copy.columns = [col.strip().lower().replace(' ', '_').replace('-', '_') 
                       for col in df_copy.columns]
    return df_copy

def detect_data_frequency(dates: pd.Series) -> str:
    """
    Detect frequency of datetime series
    
    Args:
        dates: Series of datetime values
    
    Returns:
        Frequency string ('daily', 'weekly', 'monthly', 'quarterly', 'yearly')
    """
    if len(dates) < 2:
        return 'unknown'
    
    # Calculate average difference in days
    diffs = pd.Series(dates).sort_values().diff().dt.days.dropna()
    
    if len(diffs) == 0:
        return 'unknown'
    
    avg_diff = diffs.mean()
    
    if avg_diff <= 2:
        return 'daily'
    elif avg_diff <= 8:
        return 'weekly'
    elif avg_diff <= 35:
        return 'monthly'
    elif avg_diff <= 100:
        return 'quarterly'
    else:
        return 'yearly'

def create_download_link(data: Union[str, bytes], filename: str, link_text: str) -> str:
    """
    Create HTML download link
    
    Args:
        data: Data to download
        filename: Name of file
        link_text: Text for the link
    
    Returns:
        HTML string with download link
    """
    import base64
    
    if isinstance(data, str):
        b64 = base64.b64encode(data.encode()).decode()
    else:
        b64 = base64.b64encode(data).decode()
    
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

def calculate_confidence_interval(data: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calculate confidence interval for data
    
    Args:
        data: Array of values
        confidence: Confidence level (0-1)
    
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    from scipy import stats
    
    if len(data) < 2:
        return (np.nan, np.nan)
    
    mean = np.mean(data)
    sem = stats.sem(data)
    
    if sem == 0 or np.isnan(sem):
        return (mean, mean)
    
    interval = stats.t.interval(confidence, len(data)-1, loc=mean, scale=sem)
    return interval

def winsorize_series(data: pd.Series, limits: Tuple[float, float] = (0.05, 0.05)) -> pd.Series:
    """
    Winsorize series (cap extreme values)
    
    Args:
        data: Series to winsorize
        limits: Tuple of (lower_limit, upper_limit) as proportions
    
    Returns:
        Winsorized series
    """
    from scipy.stats.mstats import winsorize
    
    return pd.Series(winsorize(data, limits=limits), index=data.index)

def calculate_rolling_beta(stock_returns: pd.Series, market_returns: pd.Series, 
                          window: int = 12) -> pd.Series:
    """
    Calculate rolling beta
    
    Args:
        stock_returns: Company returns
        market_returns: Market returns
        window: Rolling window size
    
    Returns:
        Series of rolling betas
    """
    # Combine and align
    df = pd.concat([stock_returns, market_returns], axis=1)
    df.columns = ['stock', 'market']
    df = df.dropna()
    
    if len(df) < window:
        return pd.Series(index=df.index)
    
    # Calculate rolling covariance and variance
    rolling_cov = df['stock'].rolling(window).cov(df['market'])
    rolling_var = df['market'].rolling(window).var()
    
    rolling_beta = rolling_cov / rolling_var
    return rolling_beta

def json_serialize(obj: Any) -> str:
    """
    JSON serialize with datetime handling
    
    Args:
        obj: Object to serialize
    
    Returns:
        JSON string
    """
    def serializer(o):
        if isinstance(o, (datetime, pd.Timestamp)):
            return o.isoformat()
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if pd.isna(o):
            return None
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")
    
    return json.dumps(obj, default=serializer, indent=2)

def parse_json_date(date_str: str) -> datetime:
    """
    Parse date from JSON
    
    Args:
        date_str: Date string
    
    Returns:
        Datetime object
    """
    try:
        return datetime.fromisoformat(date_str)
    except:
        try:
            return pd.to_datetime(date_str).to_pydatetime()
        except:
            return None

def calculate_correlation_with_lag(df: pd.DataFrame, col1: str, col2: str, 
                                   max_lag: int = 6) -> pd.DataFrame:
    """
    Calculate correlation with various lags
    
    Args:
        df: DataFrame
        col1: First column
        col2: Second column
        max_lag: Maximum lag to test
    
    Returns:
        DataFrame with correlations at each lag
    """
    results = []
    
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            shifted = df[col2].shift(-lag)
            corr = df[col1].corr(shifted)
        else:
            shifted = df[col2].shift(lag)
            corr = df[col1].corr(shifted)
        
        results.append({
            'lag': lag,
            'correlation': corr,
            'interpretation': f"{col1} vs {col2} {'lead' if lag < 0 else 'lag'} of {abs(lag)}"
        })
    
    return pd.DataFrame(results)

def create_advanced_filter(df: pd.DataFrame) -> Dict:
    """
    Create advanced filter UI and return filter conditions
    
    Args:
        df: DataFrame to filter
    
    Returns:
        Dictionary with filter conditions
    """
    filters = {}
    
    st.markdown("#### 🔍 Advanced Filters")
    
    # Date range filter
    if 'Date' in df.columns:
        min_date = df['Date'].min().date()
        max_date = df['Date'].max().date()
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
        with col2:
            end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date)
        
        filters['date_range'] = (pd.to_datetime(start_date), pd.to_datetime(end_date))
    
    # Numeric filters
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        with st.expander("Numeric Filters"):
            for col in numeric_cols[:5]:  # Limit to 5 columns
                min_val = float(df[col].min())
                max_val = float(df[col].max())
                
                if min_val != max_val:
                    range_val = st.slider(
                        f"{col}",
                        min_value=min_val,
                        max_value=max_val,
                        value=(min_val, max_val)
                    )
                    filters[col] = range_val
    
    # Categorical filters
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    if cat_cols:
        with st.expander("Categorical Filters"):
            for col in cat_cols[:3]:  # Limit to 3 columns
                unique_vals = df[col].unique().tolist()
                selected = st.multiselect(f"{col}", unique_vals)
                if selected:
                    filters[col] = selected
    
    return filters

def apply_filters(df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
    """
    Apply filters to DataFrame
    
    Args:
        df: DataFrame to filter
        filters: Filter conditions from create_advanced_filter
    
    Returns:
        Filtered DataFrame
    """
    df_filtered = df.copy()
    
    for key, value in filters.items():
        if key == 'date_range' and 'Date' in df_filtered.columns:
            start_date, end_date = value
            df_filtered = df_filtered[
                (df_filtered['Date'] >= start_date) & 
                (df_filtered['Date'] <= end_date)
            ]
        elif isinstance(value, tuple) and len(value) == 2:
            # Numeric range
            df_filtered = df_filtered[
                (df_filtered[key] >= value[0]) & 
                (df_filtered[key] <= value[1])
            ]
        elif isinstance(value, list) and len(value) > 0:
            # Categorical selection
            df_filtered = df_filtered[df_filtered[key].isin(value)]
    
    return df_filtered

def export_multiple_formats(data: pd.DataFrame, base_filename: str) -> None:
    """
    Create download buttons for multiple formats
    
    Args:
        data: DataFrame to export
        base_filename: Base name for files
    """
    col1, col2, col3 = st.columns(3)
    
    # CSV
    with col1:
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 CSV",
            data=csv,
            file_name=f"{base_filename}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    # Excel
    with col2:
        import io
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            data.to_excel(writer, index=False, sheet_name='Data')
        
        st.download_button(
            label="📥 Excel",
            data=buffer.getvalue(),
            file_name=f"{base_filename}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    
    # JSON
    with col3:
        json_str = data.to_json(orient='records', date_format='iso', indent=2)
        st.download_button(
            label="📥 JSON",
            data=json_str,
            file_name=f"{base_filename}.json",
            mime="application/json",
            use_container_width=True
        )

def create_progress_tracker(stages: List[str]) -> Dict:
    """
    Create a progress tracker for multi-stage processes
    
    Args:
        stages: List of stage names
    
    Returns:
        Dictionary with stage status
    """
    tracker = {}
    
    # Initialize session state if not exists
    if 'progress_tracker' not in st.session_state:
        st.session_state.progress_tracker = {}
    
    # Create progress UI
    st.markdown("### 📊 Progress Tracker")
    
    cols = st.columns(len(stages))
    for i, stage in enumerate(stages):
        status = st.session_state.progress_tracker.get(stage, 'pending')
        
        if status == 'completed':
            icon = "✅"
            color = "#10b981"
        elif status == 'in_progress':
            icon = "⏳"
            color = "#f59e0b"
        else:
            icon = "⭕"
            color = "#94a3b8"
        
        with cols[i]:
            st.markdown(f"""
            <div style="text-align: center;">
                <div style="font-size: 2rem; color: {color};">{icon}</div>
                <div style="color: {color}; font-weight: 600; font-size: 0.8rem;">{stage}</div>
            </div>
            """, unsafe_allow_html=True)
            
            if i < len(stages) - 1:
                st.markdown(f"<div style='text-align: center; color: #94a3b8;'>→</div>", 
                           unsafe_allow_html=True)
    
    return st.session_state.progress_tracker

def update_progress(stage: str, status: str) -> None:
    """
    Update progress tracker status
    
    Args:
        stage: Stage name
        status: 'pending', 'in_progress', 'completed'
    """
    if 'progress_tracker' in st.session_state:
        st.session_state.progress_tracker[stage] = status

def reset_progress() -> None:
    """Reset progress tracker"""
    if 'progress_tracker' in st.session_state:
        st.session_state.progress_tracker = {}

def memoize_dataframe(func):
    """
    Decorator to memoize DataFrame results
    
    Args:
        func: Function to memoize
    
    Returns:
        Wrapped function with caching
    """
    cache = {}
    
    def wrapper(*args, **kwargs):
        # Create cache key from arguments
        key_parts = [str(arg) for arg in args]
        key_parts.extend([f"{k}={v}" for k, v in kwargs.items()])
        key = hashlib.md5('_'.join(key_parts).encode()).hexdigest()
        
        if key in cache:
            return cache[key]
        
        result = func(*args, **kwargs)
        cache[key] = result
        return result
    
    return wrapper

def timed_execution(func):
    """
    Decorator to time function execution
    
    Args:
        func: Function to time
    
    Returns:
        Wrapped function with timing
    """
    import time
    
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        if 'debug' in st.session_state and st.session_state.debug:
            st.caption(f"⏱️ {func.__name__} took {execution_time:.2f} seconds")
        
        return result
    
    return wrapper

def create_data_quality_report(df: pd.DataFrame) -> Dict:
    """
    Create comprehensive data quality report
    
    Args:
        df: DataFrame to analyze
    
    Returns:
        Dictionary with quality metrics
    """
    report = {
        'overall_score': 0,
        'dimensions': {},
        'issues': [],
        'recommendations': []
    }
    
    # Completeness
    completeness = 1 - (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]))
    report['dimensions']['completeness'] = completeness
    
    if completeness < 0.95:
        report['issues'].append(f"Low completeness: {completeness:.1%}")
        report['recommendations'].append("Address missing values through imputation")
    
    # Uniqueness (check for duplicates)
    if 'Date' in df.columns:
        duplicate_dates = df['Date'].duplicated().sum()
        uniqueness = 1 - (duplicate_dates / len(df))
        report['dimensions']['uniqueness'] = uniqueness
        
        if uniqueness < 0.99:
            report['issues'].append(f"Duplicate dates found: {duplicate_dates}")
            report['recommendations'].append("Remove duplicate records")
    
    # Consistency (data types)
    expected_types = {
        'Revenue_USD_M': 'float64',
        'Operating_Cost_USD_M': 'float64',
        'Free_Cash_Flow_USD_M': 'float64'
    }
    
    type_issues = 0
    for col, expected in expected_types.items():
        if col in df.columns and str(df[col].dtype) != expected:
            type_issues += 1
    
    consistency = 1 - (type_issues / len(expected_types))
    report['dimensions']['consistency'] = consistency
    
    # Validity (range checks)
    validity_checks = []
    if 'Interest_Rate_%' in df.columns:
        valid_ir = ((df['Interest_Rate_%'] >= 0) & (df['Interest_Rate_%'] <= 20)).mean()
        validity_checks.append(valid_ir)
    
    if validity_checks:
        validity = np.mean(validity_checks)
        report['dimensions']['validity'] = validity
    
    # Overall score
    report['overall_score'] = np.mean(list(report['dimensions'].values()))
    
    return report

def create_sample_dataset() -> pd.DataFrame:
    """
    Create sample dataset for testing
    
    Returns:
        Sample DataFrame
    """
    np.random.seed(42)
    
    dates = pd.date_range(start='2020-01-01', periods=36, freq='M')
    
    data = {
        'Date': dates,
        'Revenue_USD_M': np.random.normal(500, 50, 36).cumsum() + 1000,
        'Operating_Cost_USD_M': np.random.normal(300, 30, 36).cumsum() + 800,
        'Capital_Expenditure_USD_M': np.random.normal(50, 10, 36),
        'Interest_Rate_%': np.random.normal(3.5, 0.5, 36),
        'Inflation_Rate_%': np.random.normal(2.5, 0.3, 36),
        'FX_Impact_%': np.random.normal(0, 1, 36),
        'Debt_Outstanding_USD_M': np.random.normal(1500, 100, 36),
        'Cash_Balance_USD_M': np.random.normal(800, 100, 36)
    }
    
    df = pd.DataFrame(data)
    df['Free_Cash_Flow_USD_M'] = (
        df['Revenue_USD_M'] - 
        df['Operating_Cost_USD_M'] - 
        df['Capital_Expenditure_USD_M']
    )
    
    return df

# CSS styles for consistent UI
def get_custom_css() -> str:
    """
    Get custom CSS styles
    
    Returns:
        CSS string
    """
    return """
    <style>
        /* Metric cards */
        .metric-card {
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            border: 1px solid #e2e8f0;
            margin-bottom: 1rem;
            transition: all 0.3s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .metric-icon {
            font-size: 1.5rem;
            margin-bottom: 0.5rem;
        }
        
        .metric-title {
            color: #64748b;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
        }
        
        .metric-value {
            color: #1e293b;
            font-size: 1.8rem;
            font-weight: 700;
            line-height: 1.2;
        }
        
        .metric-delta {
            margin-top: 0.5rem;
            font-size: 0.9rem;
        }
        
        .metric-delta.positive {
            color: #10b981;
        }
        
        .metric-delta.negative {
            color: #ef4444;
        }
        
        /* Status indicators */
        .status-badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 500;
        }
        
        .status-success {
            background: #dcfce7;
            color: #166534;
            border: 1px solid #10b981;
        }
        
        .status-warning {
            background: #fef3c7;
            color: #92400e;
            border: 1px solid #f59e0b;
        }
        
        .status-error {
            background: #fee2e2;
            color: #991b1b;
            border: 1px solid #ef4444;
        }
        
        .status-info {
            background: #dbeafe;
            color: #1e40af;
            border: 1px solid #3b82f6;
        }
        
        /* Section headers */
        .section-header {
            font-size: 1.5rem;
            font-weight: 600;
            color: #1e293b;
            margin-bottom: 1.5rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #e2e8f0;
        }
        
        .subsection-header {
            font-size: 1.2rem;
            font-weight: 600;
            color: #334155;
            margin: 1rem 0;
        }
        
        /* Cards */
        .info-card {
            background: #f8fafc;
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid #e2e8f0;
        }
        
        .highlight-card {
            background: linear-gradient(135deg, #f8fafc 0%, #ffffff 100%);
            padding: 1.5rem;
            border-radius: 12px;
            border: 1px solid #e2e8f0;
            border-left: 4px solid #3b82f6;
        }
        
        /* Progress */
        .progress-step {
            display: flex;
            align-items: center;
            margin-bottom: 0.5rem;
        }
        
        .step-number {
            width: 28px;
            height: 28px;
            background: #3b82f6;
            color: white;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 600;
            margin-right: 12px;
        }
        
        .step-completed {
            background: #10b981;
        }
        
        .step-current {
            background: #f59e0b;
        }
        
        .step-pending {
            background: #94a3b8;
        }
        
        /* Tables */
        .data-table {
            width: 100%;
            border-collapse: collapse;
        }
        
        .data-table th {
            background: #f8fafc;
            color: #475569;
            font-weight: 600;
            padding: 0.75rem;
            text-align: left;
            border-bottom: 2px solid #e2e8f0;
        }
        
        .data-table td {
            padding: 0.75rem;
            border-bottom: 1px solid #e2e8f0;
        }
        
        .data-table tr:hover {
            background: #f8fafc;
        }
        
        /* Alerts */
        .alert {
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
        }
        
        .alert-success {
            background: #dcfce7;
            color: #166534;
            border: 1px solid #10b981;
        }
        
        .alert-warning {
            background: #fef3c7;
            color: #92400e;
            border: 1px solid #f59e0b;
        }
        
        .alert-error {
            background: #fee2e2;
            color: #991b1b;
            border: 1px solid #ef4444;
        }
        
        .alert-info {
            background: #dbeafe;
            color: #1e40af;
            border: 1px solid #3b82f6;
        }
        
        /* Tooltips */
        .tooltip {
            position: relative;
            display: inline-block;
        }
        
        .tooltip .tooltip-text {
            visibility: hidden;
            width: 200px;
            background: #1e293b;
            color: white;
            text-align: center;
            padding: 0.5rem;
            border-radius: 6px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -100px;
            opacity: 0;
            transition: opacity 0.3s;
        }
        
        .tooltip:hover .tooltip-text {
            visibility: visible;
            opacity: 1;
        }
        
        /* Responsive */
        @media (max-width: 768px) {
            .metric-value {
                font-size: 1.4rem;
            }
            
            .section-header {
                font-size: 1.2rem;
            }
        }
    </style>
    """

# Test function for standalone testing
if __name__ == "__main__":
    print("Utilities Helper module ready for integration")
    print("Features: Currency formatting, statistical functions, data validation, UI components")