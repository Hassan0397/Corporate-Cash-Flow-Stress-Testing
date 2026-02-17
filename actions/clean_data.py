"""
Data Cleaning Module for Corporate Cash Flow Stress Testing Platform
Professional Data Preprocessing with Financial Domain Expertise
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
from scipy import stats
from sklearn.preprocessing import RobustScaler, StandardScaler
import warnings
warnings.filterwarnings('ignore')

def clean_dataset(df):
    """
    Main cleaning function with professional data preprocessing pipeline
    
    Args:
        df (pandas.DataFrame): Raw dataset from load_data module
    
    Returns:
        pandas.DataFrame: Cleaned and preprocessed dataset ready for analysis
    """
    
    if df is None:
        st.error("❌ No data provided for cleaning")
        return None
    
    # Create a copy to avoid modifying original
    df_clean = df.copy()
    
    # Create professional cleaning interface
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%); 
                padding: 2rem; border-radius: 12px; margin-bottom: 2rem;">
        <h3 style="color: #1e293b; margin-bottom: 1rem;">🧹 Intelligent Data Cleaning Pipeline</h3>
        <p style="color: #475569; line-height: 1.6;">
            Applying advanced financial data cleaning algorithms with domain-specific rules.
            Our pipeline ensures data integrity while preserving critical financial patterns.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize cleaning summary
    cleaning_summary = {
        'initial_shape': df.shape,
        'initial_memory': f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
        'operations': [],
        'issues_fixed': 0,
        'warnings': []
    }
    
    # Create tabs for cleaning stages
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Data Quality", 
        "🔧 Transformations", 
        "📊 Feature Engineering",
        "✅ Validation"
    ])
    
    with tab1:
        st.markdown("### 📋 Data Quality Improvement")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**🛠️ Cleaning Operations**")
            
            # 1. Handle missing values
            with st.expander("🔍 Missing Value Treatment", expanded=True):
                st.markdown("*Applying financial-specific imputation methods*")
                df_clean, missing_result = handle_missing_values(df_clean)
                cleaning_summary['operations'].append(missing_result)
                if missing_result['issues_fixed'] > 0:
                    cleaning_summary['issues_fixed'] += missing_result['issues_fixed']
                st.success(f"✅ {missing_result['message']}")
            
            # 2. Remove duplicates
            with st.expander("🔄 Duplicate Removal", expanded=True):
                st.markdown("*Checking for duplicate records*")
                df_clean, duplicate_result = remove_duplicates(df_clean)
                cleaning_summary['operations'].append(duplicate_result)
                if duplicate_result['issues_fixed'] > 0:
                    cleaning_summary['issues_fixed'] += duplicate_result['issues_fixed']
                st.success(f"✅ {duplicate_result['message']}")
            
            # 3. Handle outliers
            with st.expander("📈 Outlier Treatment", expanded=True):
                st.markdown("*Applying IQR and financial domain constraints*")
                
                outlier_method = st.radio(
                    "Select outlier treatment method:",
                    ["Capping (Winsorization)", "Removal", "Flag only"],
                    horizontal=True,
                    key="outlier_method"
                )
                
                df_clean, outlier_result = handle_outliers(
                    df_clean, 
                    method=outlier_method.split()[0].lower()
                )
                cleaning_summary['operations'].append(outlier_result)
                if outlier_result['issues_fixed'] > 0:
                    cleaning_summary['issues_fixed'] += outlier_result['issues_fixed']
                st.success(f"✅ {outlier_result['message']}")
        
        with col2:
            st.markdown("**📊 Data Quality Dashboard**")
            
            # Display quality metrics
            quality_metrics_before = calculate_quality_score(df)
            quality_metrics_after = calculate_quality_score(df_clean)
            
            # Create quality gauge
            fig_col1, fig_col2 = st.columns(2)
            with fig_col1:
                st.metric(
                    "Quality Score (Before)",
                    f"{quality_metrics_before['overall_score']:.1%}",
                    delta=None
                )
            with fig_col2:
                st.metric(
                    "Quality Score (After)",
                    f"{quality_metrics_after['overall_score']:.1%}",
                    delta=f"{quality_metrics_after['overall_score'] - quality_metrics_before['overall_score']:.1%}",
                    delta_color="normal"
                )
            
            # Quality dimensions
            quality_df = pd.DataFrame({
                'Dimension': ['Completeness', 'Consistency', 'Accuracy', 'Validity'],
                'Before': [
                    quality_metrics_before['completeness'],
                    quality_metrics_before['consistency'],
                    quality_metrics_before['accuracy'],
                    quality_metrics_before['validity']
                ],
                'After': [
                    quality_metrics_after['completeness'],
                    quality_metrics_after['consistency'],
                    quality_metrics_after['accuracy'],
                    quality_metrics_after['validity']
                ]
            })
            
            # Format percentage
            quality_df['Before'] = quality_df['Before'].apply(lambda x: f"{x:.1%}")
            quality_df['After'] = quality_df['After'].apply(lambda x: f"{x:.1%}")
            
            st.dataframe(quality_df, use_container_width=True, hide_index=True)
    
    with tab2:
        st.markdown("### 🔧 Data Transformations")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**🔄 Column Transformations**")
            
            # 1. Date standardization
            with st.expander("📅 Date Standardization", expanded=True):
                st.markdown("*Standardizing date format and creating date features*")
                df_clean, date_result = standardize_dates(df_clean)
                cleaning_summary['operations'].append(date_result)
                st.success(f"✅ {date_result['message']}")
            
            # 2. Numeric formatting
            with st.expander("💹 Numeric Column Formatting", expanded=True):
                st.markdown("*Ensuring consistent numeric formats*")
                df_clean, numeric_result = format_numeric_columns(df_clean)
                cleaning_summary['operations'].append(numeric_result)
                st.success(f"✅ {numeric_result['message']}")
            
            # 3. Categorical encoding (if any)
            categorical_cols = df_clean.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0 and 'Date' not in categorical_cols:
                with st.expander("🏷️ Categorical Encoding", expanded=True):
                    st.markdown("*Encoding categorical variables*")
                    df_clean, encoding_result = encode_categorical(df_clean)
                    cleaning_summary['operations'].append(encoding_result)
                    st.success(f"✅ {encoding_result['message']}")
        
        with col2:
            st.markdown("**📈 Distribution Analysis**")
            
            # Show distribution of key metrics before/after
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns[:3]
            
            for col in numeric_cols:
                if col in df.columns and col in df_clean.columns:
                    before_stats = df[col].describe()
                    after_stats = df_clean[col].describe()
                    
                    st.markdown(f"**{col}**")
                    col_stat1, col_stat2 = st.columns(2)
                    with col_stat1:
                        st.metric("Mean (Before)", f"{before_stats['mean']:,.2f}")
                        st.metric("Std (Before)", f"{before_stats['std']:,.2f}")
                    with col_stat2:
                        st.metric("Mean (After)", f"{after_stats['mean']:,.2f}",
                                 delta=f"{after_stats['mean'] - before_stats['mean']:,.2f}")
                        st.metric("Std (After)", f"{after_stats['std']:,.2f}",
                                 delta=f"{after_stats['std'] - before_stats['std']:,.2f}")
    
    with tab3:
        st.markdown("### 📊 Feature Engineering")
        
        st.markdown("""
        <div style="background: #f8fafc; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
            <p style="color: #475569; margin: 0;">
                💡 Creating advanced financial features for enhanced predictive power
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Select features to create
            st.markdown("**🛠️ Feature Selection**")
            
            create_lagged = st.checkbox("Create Lagged Features", value=True)
            create_rolling = st.checkbox("Create Rolling Statistics", value=True)
            create_ratios = st.checkbox("Create Financial Ratios", value=True)
            create_interactions = st.checkbox("Create Interaction Features", value=False)
            
            if create_lagged:
                lag_periods = st.multiselect(
                    "Select lag periods (months):",
                    [1, 2, 3, 6, 12],
                    default=[1, 3]
                )
            else:
                lag_periods = []
            
            if create_rolling:
                rolling_windows = st.multiselect(
                    "Select rolling windows (months):",
                    [3, 6, 12],
                    default=[3, 6]
                )
            else:
                rolling_windows = []
        
        with col2:
            st.markdown("**📈 Feature Preview**")
            
            # Apply feature engineering
            df_clean, feature_result = create_financial_features(
                df_clean,
                create_lagged=create_lagged,
                create_rolling=create_rolling,
                create_ratios=create_ratios,
                create_interactions=create_interactions,
                lag_periods=lag_periods,
                rolling_windows=rolling_windows
            )
            
            cleaning_summary['operations'].append(feature_result)
            
            # Show new features
            original_cols = set(df.columns)
            new_cols = set(df_clean.columns) - original_cols
            
            if new_cols:
                st.success(f"✅ Created {len(new_cols)} new features")
                st.markdown("**New Features:**")
                for col in sorted(list(new_cols))[:10]:  # Show first 10
                    st.markdown(f"- `{col}`")
                if len(new_cols) > 10:
                    st.markdown(f"*... and {len(new_cols) - 10} more*")
            else:
                st.info("No new features created")
    
    with tab4:
        st.markdown("### ✅ Final Validation")
        
        # Run final validations
        validation_results = run_final_validations(df_clean)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**📋 Validation Checks**")
            
            for check, result in validation_results.items():
                if result['status'] == 'PASS':
                    st.markdown(f"""
                    <div style="background: #f0fdf4; padding: 0.75rem; border-radius: 6px; margin-bottom: 0.5rem;
                                border-left: 4px solid #22c55e;">
                        <span style="color: #16a34a; font-weight: 600;">✓</span> {check}<br>
                        <span style="color: #4b5563; font-size: 0.9rem;">{result['message']}</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background: #fef2f2; padding: 0.75rem; border-radius: 6px; margin-bottom: 0.5rem;
                                border-left: 4px solid #ef4444;">
                        <span style="color: #dc2626; font-weight: 600;">⚠</span> {check}<br>
                        <span style="color: #4b5563; font-size: 0.9rem;">{result['message']}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    cleaning_summary['warnings'].append(f"{check}: {result['message']}")
        
        with col2:
            st.markdown("**📊 Dataset Statistics**")
            
            stats_data = {
                'Metric': ['Rows', 'Columns', 'Memory Usage', 'Date Range', 'Complete Cases'],
                'Value': [
                    f"{len(df_clean):,}",
                    f"{len(df_clean.columns):,}",
                    f"{df_clean.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
                    f"{df_clean['Date'].min().date()} to {df_clean['Date'].max().date()}",
                    f"{df_clean.dropna().shape[0]:,} ({df_clean.dropna().shape[0]/len(df_clean):.1%})"
                ]
            }
            st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)
    
    # Cleaning summary
    st.markdown("---")
    st.markdown("### 📋 Cleaning Summary Report")
    
    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
    
    with summary_col1:
        st.metric(
            "Initial Records",
            f"{cleaning_summary['initial_shape'][0]:,}",
            delta=None
        )
    
    with summary_col2:
        st.metric(
            "Final Records",
            f"{df_clean.shape[0]:,}",
            delta=f"{df_clean.shape[0] - cleaning_summary['initial_shape'][0]}"
        )
    
    with summary_col3:
        st.metric(
            "Initial Columns",
            f"{cleaning_summary['initial_shape'][1]:,}",
            delta=None
        )
    
    with summary_col4:
        st.metric(
            "Final Columns",
            f"{df_clean.shape[1]:,}",
            delta=f"{df_clean.shape[1] - cleaning_summary['initial_shape'][1]}"
        )
    
    # Detailed operations log
    with st.expander("📝 Detailed Operations Log", expanded=False):
        operations_df = pd.DataFrame(cleaning_summary['operations'])
        st.dataframe(operations_df, use_container_width=True, hide_index=True)
    
    if cleaning_summary['warnings']:
        with st.expander("⚠️ Warnings and Recommendations", expanded=False):
            for warning in cleaning_summary['warnings']:
                st.warning(warning)
    
    # Download cleaned data
    st.markdown("---")
    col1, col2 = st.columns([3, 1])
    with col2:
        csv = df_clean.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Cleaned Data",
            data=csv,
            file_name=f"cleaned_cashflow_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True
        )
    
    return df_clean

def handle_missing_values(df):
    """Handle missing values with financial domain-specific methods"""
    result = {
        'operation': 'Missing Value Treatment',
        'issues_found': 0,
        'issues_fixed': 0,
        'message': '',
        'details': {}
    }
    
    df_clean = df.copy()
    
    for col in df_clean.columns:
        missing_count = df_clean[col].isnull().sum()
        if missing_count > 0:
            result['issues_found'] += missing_count
            
            if col == 'Date':
                # Forward fill for dates
                df_clean[col] = df_clean[col].fillna(method='ffill')
                result['issues_fixed'] += missing_count
                result['details'][col] = f"Forward filled {missing_count} missing dates"
            
            elif col in ['Revenue_USD_M', 'Operating_Cost_USD_M', 'Capital_Expenditure_USD_M']:
                # Use median for revenue/cost (robust to outliers)
                median_val = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(median_val)
                result['issues_fixed'] += missing_count
                result['details'][col] = f"Imputed with median ({median_val:.2f})"
            
            elif col in ['Interest_Rate_%', 'Inflation_Rate_%', 'FX_Impact_%']:
                # Use mean for rate variables
                mean_val = df_clean[col].mean()
                df_clean[col] = df_clean[col].fillna(mean_val)
                result['issues_fixed'] += missing_count
                result['details'][col] = f"Imputed with mean ({mean_val:.2f})"
            
            elif col in ['Debt_Outstanding_USD_M', 'Cash_Balance_USD_M']:
                # Forward fill for balance sheet items
                df_clean[col] = df_clean[col].fillna(method='ffill').fillna(method='bfill')
                result['issues_fixed'] += missing_count
                result['details'][col] = f"Forward/backward filled {missing_count} missing values"
            
            elif col == 'Free_Cash_Flow_USD_M':
                # Calculate from components if missing
                df_clean['Free_Cash_Flow_USD_M'] = df_clean.apply(
                    lambda row: row['Revenue_USD_M'] - row['Operating_Cost_USD_M'] - row['Capital_Expenditure_USD_M']
                    if pd.isnull(row['Free_Cash_Flow_USD_M']) else row['Free_Cash_Flow_USD_M'],
                    axis=1
                )
                result['issues_fixed'] += missing_count
                result['details'][col] = f"Recalculated from components"
    
    if result['issues_found'] == 0:
        result['message'] = "No missing values detected"
    else:
        result['message'] = f"Fixed {result['issues_fixed']} missing values across {len(result['details'])} columns"
    
    return df_clean, result

def remove_duplicates(df):
    """Remove duplicate records"""
    result = {
        'operation': 'Duplicate Removal',
        'issues_found': 0,
        'issues_fixed': 0,
        'message': '',
        'details': {}
    }
    
    df_clean = df.copy()
    
    # Check for exact duplicates
    exact_duplicates = df_clean.duplicated().sum()
    if exact_duplicates > 0:
        result['issues_found'] += exact_duplicates
        df_clean = df_clean.drop_duplicates()
        result['issues_fixed'] += exact_duplicates
        result['details']['exact_duplicates'] = f"Removed {exact_duplicates} exact duplicate rows"
    
    # Check for duplicate dates
    if 'Date' in df_clean.columns:
        date_duplicates = df_clean['Date'].duplicated().sum()
        if date_duplicates > 0:
            result['issues_found'] += date_duplicates
            # Keep first occurrence for duplicate dates
            df_clean = df_clean.drop_duplicates(subset=['Date'], keep='first')
            result['issues_fixed'] += date_duplicates
            result['details']['date_duplicates'] = f"Removed {date_duplicates} duplicate dates"
    
    if result['issues_found'] == 0:
        result['message'] = "No duplicate records found"
    else:
        result['message'] = f"Removed {result['issues_fixed']} duplicate records"
    
    return df_clean, result

def handle_outliers(df, method='capping'):
    """Handle outliers using various methods"""
    result = {
        'operation': 'Outlier Treatment',
        'method': method,
        'issues_found': 0,
        'issues_fixed': 0,
        'message': '',
        'details': {}
    }
    
    df_clean = df.copy()
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if col not in ['Date'] and col not in df_clean.select_dtypes(include=['datetime']).columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)][col]
            
            if len(outliers) > 0:
                result['issues_found'] += len(outliers)
                
                if method == 'capping':
                    df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
                    result['issues_fixed'] += len(outliers)
                    result['details'][col] = f"Capped {len(outliers)} outliers to [{lower_bound:.2f}, {upper_bound:.2f}]"
                
                elif method == 'removal':
                    df_clean = df_clean[~((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound))]
                    result['issues_fixed'] += len(outliers)
                    result['details'][col] = f"Removed {len(outliers)} outliers"
                
                elif method == 'flag':
                    df_clean[f'{col}_outlier_flag'] = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).astype(int)
                    result['details'][col] = f"Flagged {len(outliers)} outliers"
    
    if result['issues_found'] == 0:
        result['message'] = "No significant outliers detected"
    else:
        result['message'] = f"Treated {result['issues_fixed'] if method != 'flag' else result['issues_found']} outliers using {method} method"
    
    return df_clean, result

def standardize_dates(df):
    """Standardize date column and create date features"""
    result = {
        'operation': 'Date Standardization',
        'issues_found': 0,
        'issues_fixed': 0,
        'message': '',
        'details': {}
    }
    
    df_clean = df.copy()
    
    if 'Date' in df_clean.columns:
        # Ensure datetime format
        if not pd.api.types.is_datetime64_any_dtype(df_clean['Date']):
            df_clean['Date'] = pd.to_datetime(df_clean['Date'], errors='coerce')
            result['issues_fixed'] += 1
            result['details']['date_conversion'] = "Converted Date column to datetime format"
        
        # Sort by date
        df_clean = df_clean.sort_values('Date').reset_index(drop=True)
        
        # Create date features
        df_clean['Year'] = df_clean['Date'].dt.year
        df_clean['Month'] = df_clean['Date'].dt.month
        df_clean['Quarter'] = df_clean['Date'].dt.quarter
        df_clean['YearMonth'] = df_clean['Date'].dt.strftime('%Y-%m')
        
        result['message'] = "Date standardized and date features created"
    else:
        result['message'] = "No Date column found"
    
    return df_clean, result

def format_numeric_columns(df):
    """Ensure consistent numeric formatting"""
    result = {
        'operation': 'Numeric Formatting',
        'issues_found': 0,
        'issues_fixed': 0,
        'message': '',
        'details': {}
    }
    
    df_clean = df.copy()
    
    # Columns that should be numeric
    numeric_columns = [
        'Revenue_USD_M', 'Operating_Cost_USD_M', 'Capital_Expenditure_USD_M',
        'Interest_Rate_%', 'Inflation_Rate_%', 'FX_Impact_%',
        'Debt_Outstanding_USD_M', 'Cash_Balance_USD_M', 'Free_Cash_Flow_USD_M'
    ]
    
    for col in numeric_columns:
        if col in df_clean.columns:
            if not pd.api.types.is_numeric_dtype(df_clean[col]):
                # Convert to numeric, coercing errors to NaN
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                result['issues_fixed'] += 1
                result['details'][col] = "Converted to numeric"
    
    # Round to 2 decimal places for consistency
    for col in df_clean.select_dtypes(include=[np.number]).columns:
        if col not in ['Year', 'Month', 'Quarter']:
            df_clean[col] = df_clean[col].round(2)
    
    if result['issues_fixed'] == 0:
        result['message'] = "All numeric columns already in correct format"
    else:
        result['message'] = f"Formatted {result['issues_fixed']} numeric columns"
    
    return df_clean, result

def encode_categorical(df):
    """Encode categorical variables"""
    result = {
        'operation': 'Categorical Encoding',
        'issues_found': 0,
        'issues_fixed': 0,
        'message': '',
        'details': {}
    }
    
    df_clean = df.copy()
    
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    categorical_cols = [col for col in categorical_cols if col != 'Date']
    
    for col in categorical_cols:
        if col not in ['YearMonth']:
            # One-hot encode categorical variables
            dummies = pd.get_dummies(df_clean[col], prefix=col, drop_first=True)
            df_clean = pd.concat([df_clean, dummies], axis=1)
            df_clean = df_clean.drop(columns=[col])
            result['issues_fixed'] += 1
            result['details'][col] = f"One-hot encoded ({len(dummies.columns)} categories)"
    
    if result['issues_fixed'] == 0:
        result['message'] = "No categorical columns to encode"
    else:
        result['message'] = f"Encoded {result['issues_fixed']} categorical columns"
    
    return df_clean, result

def create_financial_features(df, create_lagged=True, create_rolling=True, 
                              create_ratios=True, create_interactions=False,
                              lag_periods=[1, 3], rolling_windows=[3, 6]):
    """Create advanced financial features"""
    result = {
        'operation': 'Feature Engineering',
        'features_created': 0,
        'message': '',
        'details': {}
    }
    
    df_feat = df.copy()
    original_cols = set(df_feat.columns)
    
    # 1. Lagged features
    if create_lagged and 'Date' in df_feat.columns:
        fcf_col = 'Free_Cash_Flow_USD_M' if 'Free_Cash_Flow_USD_M' in df_feat.columns else None
        
        if fcf_col:
            for lag in lag_periods:
                df_feat[f'FCF_Lag_{lag}'] = df_feat[fcf_col].shift(lag)
                result['features_created'] += 1
                result['details'][f'FCF_Lag_{lag}'] = f"Free Cash Flow lagged by {lag} months"
        
        # Revenue lags
        if 'Revenue_USD_M' in df_feat.columns:
            for lag in lag_periods:
                df_feat[f'Revenue_Lag_{lag}'] = df_feat['Revenue_USD_M'].shift(lag)
                result['features_created'] += 1
    
    # 2. Rolling statistics
    if create_rolling and 'Date' in df_feat.columns:
        metrics_to_roll = ['Revenue_USD_M', 'Free_Cash_Flow_USD_M']
        metrics_to_roll = [m for m in metrics_to_roll if m in df_feat.columns]
        
        for metric in metrics_to_roll:
            for window in rolling_windows:
                df_feat[f'{metric}_Rolling_Mean_{window}M'] = df_feat[metric].rolling(window=window, min_periods=1).mean()
                df_feat[f'{metric}_Rolling_Std_{window}M'] = df_feat[metric].rolling(window=window, min_periods=1).std()
                df_feat[f'{metric}_Rolling_Min_{window}M'] = df_feat[metric].rolling(window=window, min_periods=1).min()
                df_feat[f'{metric}_Rolling_Max_{window}M'] = df_feat[metric].rolling(window=window, min_periods=1).max()
                result['features_created'] += 4
    
    # 3. Financial ratios
    if create_ratios:
        # Operating margin
        if 'Revenue_USD_M' in df_feat.columns and 'Operating_Cost_USD_M' in df_feat.columns:
            df_feat['Operating_Margin'] = ((df_feat['Revenue_USD_M'] - df_feat['Operating_Cost_USD_M']) / df_feat['Revenue_USD_M']) * 100
            result['features_created'] += 1
        
        # Cash ratio
        if 'Cash_Balance_USD_M' in df_feat.columns and 'Debt_Outstanding_USD_M' in df_feat.columns:
            df_feat['Cash_to_Debt_Ratio'] = df_feat['Cash_Balance_USD_M'] / df_feat['Debt_Outstanding_USD_M']
            result['features_created'] += 1
        
        # Capex intensity
        if 'Capital_Expenditure_USD_M' in df_feat.columns and 'Revenue_USD_M' in df_feat.columns:
            df_feat['Capex_Intensity'] = (df_feat['Capital_Expenditure_USD_M'] / df_feat['Revenue_USD_M']) * 100
            result['features_created'] += 1
        
        # Interest coverage (simplified)
        if 'Free_Cash_Flow_USD_M' in df_feat.columns and 'Debt_Outstanding_USD_M' in df_feat.columns and 'Interest_Rate_%' in df_feat.columns:
            df_feat['Interest_Coverage'] = df_feat['Free_Cash_Flow_USD_M'] / (df_feat['Debt_Outstanding_USD_M'] * df_feat['Interest_Rate_%'] / 100 / 12)
            result['features_created'] += 1
        
        # Efficiency ratio
        if 'Operating_Cost_USD_M' in df_feat.columns and 'Revenue_USD_M' in df_feat.columns:
            df_feat['Cost_to_Revenue'] = (df_feat['Operating_Cost_USD_M'] / df_feat['Revenue_USD_M']) * 100
            result['features_created'] += 1
    
    # 4. Interaction features
    if create_interactions:
        if 'Interest_Rate_%' in df_feat.columns and 'Inflation_Rate_%' in df_feat.columns:
            df_feat['Real_Interest_Rate'] = df_feat['Interest_Rate_%'] - df_feat['Inflation_Rate_%']
            result['features_created'] += 1
        
        if 'FX_Impact_%' in df_feat.columns and 'Revenue_USD_M' in df_feat.columns:
            df_feat['FX_Adjusted_Revenue'] = df_feat['Revenue_USD_M'] * (1 + df_feat['FX_Impact_%']/100)
            result['features_created'] += 1
    
    # Forward fill NaN values from lag/rolling features
    df_feat = df_feat.fillna(method='bfill').fillna(method='ffill')
    
    new_features = len(set(df_feat.columns) - original_cols)
    result['features_created'] = new_features
    result['message'] = f"Created {new_features} new financial features"
    
    return df_feat, result

def calculate_quality_score(df):
    """Calculate comprehensive data quality score"""
    quality = {}
    
    # Completeness score
    completeness = 1 - (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]))
    quality['completeness'] = completeness * 100
    
    # Consistency score (date sequence)
    if 'Date' in df.columns:
        date_diff = df['Date'].diff().dropna()
        expected_diff = pd.Timedelta(days=30)
        consistency = (abs(date_diff - expected_diff) < pd.Timedelta(days=5)).mean()
        quality['consistency'] = consistency * 100
    else:
        quality['consistency'] = 50  # Default if no date
    
    # Accuracy score (FCF calculation)
    if all(col in df.columns for col in ['Revenue_USD_M', 'Operating_Cost_USD_M', 
                                         'Capital_Expenditure_USD_M', 'Free_Cash_Flow_USD_M']):
        calculated_fcf = df['Revenue_USD_M'] - df['Operating_Cost_USD_M'] - df['Capital_Expenditure_USD_M']
        accuracy = (abs(df['Free_Cash_Flow_USD_M'] - calculated_fcf) < 0.1).mean()
        quality['accuracy'] = accuracy * 100
    else:
        quality['accuracy'] = 50  # Default
    
    # Validity score (range checks)
    validity_checks = []
    if 'Interest_Rate_%' in df.columns:
        validity_checks.append(((df['Interest_Rate_%'] >= 0) & (df['Interest_Rate_%'] <= 20)).mean())
    if 'Revenue_USD_M' in df.columns:
        validity_checks.append((df['Revenue_USD_M'] > 0).mean())
    
    if validity_checks:
        quality['validity'] = np.mean(validity_checks) * 100
    else:
        quality['validity'] = 50
    
    # Overall score
    quality['overall_score'] = np.mean([quality['completeness'], quality['consistency'], 
                                        quality['accuracy'], quality['validity']]) / 100
    
    return quality

def run_final_validations(df):
    """Run final validation checks on cleaned dataset"""
    validation_results = {}
    
    # Check 1: No missing values in critical columns
    critical_cols = ['Date', 'Revenue_USD_M', 'Operating_Cost_USD_M', 'Free_Cash_Flow_USD_M']
    missing_critical = df[critical_cols].isnull().sum().sum()
    validation_results['Missing Values Check'] = {
        'status': 'PASS' if missing_critical == 0 else 'FAIL',
        'message': f"No missing values in critical columns" if missing_critical == 0 
                  else f"{missing_critical} missing values found"
    }
    
    # Check 2: Date range completeness
    if 'Date' in df.columns:
        date_range_complete = len(df) >= 12
        validation_results['Data Sufficiency'] = {
            'status': 'PASS' if date_range_complete else 'WARNING',
            'message': f"{len(df)} months of data available" if date_range_complete 
                      else f"Only {len(df)} months of data (minimum 12 recommended)"
        }
    
    # Check 3: Financial logic validation
    if all(col in df.columns for col in ['Revenue_USD_M', 'Operating_Cost_USD_M']):
        negative_cash_flow = (df['Free_Cash_Flow_USD_M'] < 0).sum()
        validation_results['Financial Health'] = {
            'status': 'WARNING' if negative_cash_flow > len(df) * 0.2 else 'PASS',
            'message': f"{negative_cash_flow} months with negative cash flow ({negative_cash_flow/len(df):.1%})"
        }
    
    # Check 4: Feature completeness
    expected_features = ['Date', 'Year', 'Month', 'Quarter']
    missing_features = [f for f in expected_features if f not in df.columns]
    validation_results['Feature Completeness'] = {
        'status': 'PASS' if not missing_features else 'WARNING',
        'message': "All expected features present" if not missing_features 
                  else f"Missing features: {', '.join(missing_features)}"
    }
    
    # Check 5: Data types validation
    if 'Date' in df.columns:
        date_type_ok = pd.api.types.is_datetime64_any_dtype(df['Date'])
        validation_results['Data Types'] = {
            'status': 'PASS' if date_type_ok else 'FAIL',
            'message': "Date column has correct datetime type" if date_type_ok 
                      else "Date column has incorrect data type"
        }
    
    return validation_results

# Test function for standalone testing
if __name__ == "__main__":
    print("Testing data cleaning module...")
    # This would typically load data and test cleaning
    print("Data cleaning module ready for integration")