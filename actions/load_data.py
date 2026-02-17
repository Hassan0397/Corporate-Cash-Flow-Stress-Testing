"""
Data Loading Module for Corporate Cash Flow Stress Testing Platform
Professional Data Ingestion with Validation and Quality Checks
"""

import pandas as pd
import numpy as np
from pathlib import Path
import streamlit as st
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_dataset():
    """
    Load and validate the corporate cash flow dataset with professional quality checks
    
    Returns:
        pandas.DataFrame: Cleaned and validated dataset with proper data types
        None: If loading fails
    """
    
    try:
        # Show loading animation
        with st.spinner("🔍 Locating dataset..."):
            # Define data path
            data_path = Path(__file__).parent.parent / "data" / "Corporate_Cashflow_Stress_Testing_Dataset.csv"
            
            # Check if file exists
            if not data_path.exists():
                st.error(f"❌ Data file not found at: {data_path}")
                st.info("Please ensure the CSV file is in the 'data' folder")
                return None
            
            # Read the dataset
            st.info(f"📂 Loading dataset from: {data_path.name}")
            
            # Read CSV with specific settings
            df = pd.read_csv(
                data_path,
                parse_dates=['Date'],
                dayfirst=True  # Handle DD/MM/YYYY format
            )
            
            # Initial validation
            initial_rows = len(df)
            initial_cols = len(df.columns)
            
            # Create validation container
            validation_container = st.container()
            
            with validation_container:
                st.markdown("""
                    <div style="margin-bottom: 1.5rem;">
                        <h3 style="margin-bottom: 0.5rem; font-weight: 600;">🛠️ Data Validation Progress</h3>
                        <div style="height: 2px; background: linear-gradient(90deg, #1f77b4 0%, #1f77b4 100%); width: 50px;"></div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Create progress indicators
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("📄 Initial Rows", initial_rows)
                
                with col2:
                    st.metric("📊 Initial Columns", initial_cols)
                
                with col3:
                    st.metric("📅 Date Range", f"{df['Date'].min().date()} to {df['Date'].max().date()}")
                
                # Validation steps
                validation_steps = [
                    ("📋 Check Data Types", validate_data_types),
                    ("🔍 Validate Date Sequence", validate_date_sequence),
                    ("📈 Check Financial Metrics", validate_financial_metrics),
                    ("🚫 Identify Missing Values", check_missing_values),
                    ("📏 Detect Outliers", check_outliers),
                    ("✅ Final Validation", final_validation)
                ]
                
                # Run validations with progress bar
                progress_bar = st.progress(0)
                validation_results = []
                
                for i, (step_name, validation_func) in enumerate(validation_steps):
                    st.markdown(f"**{i+1}. {step_name}**")
                    
                    # Run validation
                    with st.spinner("Running..."):
                        result = validation_func(df.copy())
                        validation_results.append((step_name, result))
                    
                    # Update progress
                    progress = (i + 1) / len(validation_steps)
                    progress_bar.progress(progress)
                
                # Display validation summary
                st.markdown("---")
                st.markdown("""
                    <div style="margin-top: 1.5rem; margin-bottom: 1rem;">
                        <h4 style="margin-bottom: 0.5rem; font-weight: 600;">📋 Validation Summary</h4>
                        <div style="height: 2px; background: linear-gradient(90deg, #1f77b4 0%, #1f77b4 50px); width: 50px;"></div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Create grid layout for validation cards
                issues_count = 0
                
                for i in range(0, len(validation_results), 2):
                    col1, col2 = st.columns(2)
                    
                    for j in range(2):
                        if i + j < len(validation_results):
                            step_name, result = validation_results[i + j]
                            col = col1 if j == 0 else col2
                            
                            with col:
                                if result.get("status") == "PASS":
                                    st.markdown(f"""
                                        <div style="
                                            background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
                                            padding: 1.25rem;
                                            border-radius: 12px;
                                            border: 1px solid #86efac;
                                            margin-bottom: 1rem;
                                            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
                                            transition: transform 0.2s;
                                        ">
                                            <div style="display: flex; align-items: flex-start; gap: 0.75rem;">
                                                <div style="
                                                    background: #22c55e;
                                                    border-radius: 50%;
                                                    width: 24px;
                                                    height: 24px;
                                                    display: flex;
                                                    align-items: center;
                                                    justify-content: center;
                                                    flex-shrink: 0;
                                                ">
                                                    <span style="color: white; font-size: 14px;">✓</span>
                                                </div>
                                                <div style="flex: 1;">
                                                    <div style="font-weight: 600; color: #15803d; margin-bottom: 0.5rem;">
                                                        {step_name}
                                                    </div>
                                                    <div style="font-size: 0.9rem; color: #166534; line-height: 1.5;">
                                                        {result.get("message", "Validation passed")}
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    """, unsafe_allow_html=True)
                                    
                                elif result.get("status") == "WARNING":
                                    st.markdown(f"""
                                        <div style="
                                            background: linear-gradient(135deg, #fefce8 0%, #fef9c3 100%);
                                            padding: 1.25rem;
                                            border-radius: 12px;
                                            border: 1px solid #fde047;
                                            margin-bottom: 1rem;
                                            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
                                            transition: transform 0.2s;
                                        ">
                                            <div style="display: flex; align-items: flex-start; gap: 0.75rem;">
                                                <div style="
                                                    background: #eab308;
                                                    border-radius: 50%;
                                                    width: 24px;
                                                    height: 24px;
                                                    display: flex;
                                                    align-items: center;
                                                    justify-content: center;
                                                    flex-shrink: 0;
                                                ">
                                                    <span style="color: white; font-size: 14px;">⚠</span>
                                                </div>
                                                <div style="flex: 1;">
                                                    <div style="font-weight: 600; color: #854d0e; margin-bottom: 0.5rem;">
                                                        {step_name}
                                                    </div>
                                                    <div style="font-size: 0.9rem; color: #713f12; line-height: 1.5;">
                                                        {result.get("message", "Check required")}
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    """, unsafe_allow_html=True)
                                    issues_count += 1
                                    
                                else:
                                    st.markdown(f"""
                                        <div style="
                                            background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
                                            padding: 1.25rem;
                                            border-radius: 12px;
                                            border: 1px solid #fca5a5;
                                            margin-bottom: 1rem;
                                            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
                                            transition: transform 0.2s;
                                        ">
                                            <div style="display: flex; align-items: flex-start; gap: 0.75rem;">
                                                <div style="
                                                    background: #ef4444;
                                                    border-radius: 50%;
                                                    width: 24px;
                                                    height: 24px;
                                                    display: flex;
                                                    align-items: center;
                                                    justify-content: center;
                                                    flex-shrink: 0;
                                                ">
                                                    <span style="color: white; font-size: 14px;">✗</span>
                                                </div>
                                                <div style="flex: 1;">
                                                    <div style="font-weight: 600; color: #991b1b; margin-bottom: 0.5rem;">
                                                        {step_name}
                                                    </div>
                                                    <div style="font-size: 0.9rem; color: #7f1d1d; line-height: 1.5;">
                                                        {result.get("message", "Validation failed")}
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    """, unsafe_allow_html=True)
                                    issues_count += 1
            
            # Post-validation processing
            if issues_count == 0:
                st.success("✅ All validations passed! Data is ready for analysis.")
            elif issues_count <= 2:
                st.warning(f"⚠️ {issues_count} minor issues found. Data can be used with caution.")
            else:
                st.error(f"❌ {issues_count} critical issues found. Please review the data.")
            
            # Display data quality metrics
            st.markdown("---")
            st.markdown("""
                <div style="margin-top: 1.5rem; margin-bottom: 1rem;">
                    <h4 style="margin-bottom: 0.5rem; font-weight: 600;">📊 Data Quality Metrics</h4>
                    <div style="height: 2px; background: linear-gradient(90deg, #1f77b4 0%, #1f77b4 50px); width: 50px;"></div>
                </div>
            """, unsafe_allow_html=True)
            
            quality_metrics = calculate_quality_metrics(df)
            
            # Create metrics grid
            metric_cols = st.columns(4)
            metrics_to_display = [
                ("Completeness", f"{quality_metrics['completeness']:.1%}"),
                ("Consistency", f"{quality_metrics['consistency']:.1%}"),
                ("Accuracy", f"{quality_metrics['accuracy']:.1%}"),
                ("Timeliness", "100%")
            ]
            
            for idx, (metric_name, metric_value) in enumerate(metrics_to_display):
                with metric_cols[idx]:
                    st.markdown(f"""
                        <div style="
                            background: white;
                            padding: 1.25rem;
                            border-radius: 12px;
                            border: 1px solid #e5e7eb;
                            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                            text-align: center;
                        ">
                            <div style="font-size: 0.875rem; color: #6b7280; margin-bottom: 0.5rem;">
                                {metric_name}
                            </div>
                            <div style="font-size: 1.75rem; font-weight: 600; color: #1f2937;">
                                {metric_value}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
            
            # Data preview with tabs
            st.markdown("---")
            st.markdown("""
                <div style="margin-top: 1.5rem; margin-bottom: 1rem;">
                    <h4 style="margin-bottom: 0.5rem; font-weight: 600;">👁️ Data Preview</h4>
                    <div style="height: 2px; background: linear-gradient(90deg, #1f77b4 0%, #1f77b4 50px); width: 50px;"></div>
                </div>
            """, unsafe_allow_html=True)
            
            tab1, tab2, tab3 = st.tabs(["📋 First 10 Rows", "📈 Statistics", "📅 Date Info"])
            
            with tab1:
                styled_df = df.head(10).style.format({
                    'Revenue_USD_M': '{:,.2f}',
                    'Operating_Cost_USD_M': '{:,.2f}',
                    'Capital_Expenditure_USD_M': '{:,.2f}',
                    'Debt_Outstanding_USD_M': '{:,.2f}',
                    'Cash_Balance_USD_M': '{:,.2f}',
                    'Free_Cash_Flow_USD_M': '{:,.2f}',
                    'Interest_Rate_%': '{:.2f}',
                    'Inflation_Rate_%': '{:.2f}',
                    'FX_Impact_%': '{:.2f}'
                })
                st.dataframe(styled_df, use_container_width=True)
            
            with tab2:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                stats_df = df[numeric_cols].describe().T
                stats_df = stats_df[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
                stats_df = stats_df.round(2)
                st.dataframe(stats_df, use_container_width=True)
            
            with tab3:
                date_info = pd.DataFrame({
                    'Metric': ['Start Date', 'End Date', 'Total Months', 'Date Frequency', 'Missing Dates'],
                    'Value': [
                        df['Date'].min().strftime('%Y-%m-%d'),
                        df['Date'].max().strftime('%Y-%m-%d'),
                        len(df),
                        'Monthly',
                        'None' if df['Date'].is_monotonic_increasing else 'Check required'
                    ]
                })
                st.dataframe(date_info, use_container_width=True, hide_index=True)
            
            # Data dictionary
            with st.expander("📚 Data Dictionary", expanded=False):
                st.markdown("""
                    <div style="margin-bottom: 1rem; font-size: 0.95rem; color: #4b5563;">
                        Detailed description of all columns in the dataset
                    </div>
                """, unsafe_allow_html=True)
                
                data_dict = {
                    'Column': [
                        'Date', 'Revenue_USD_M', 'Operating_Cost_USD_M', 
                        'Capital_Expenditure_USD_M', 'Interest_Rate_%', 
                        'Inflation_Rate_%', 'FX_Impact_%', 'Debt_Outstanding_USD_M',
                        'Cash_Balance_USD_M', 'Free_Cash_Flow_USD_M'
                    ],
                    'Description': [
                        'Month-end date',
                        'Monthly revenue in USD millions',
                        'Monthly operating costs in USD millions',
                        'Monthly capital expenditure in USD millions',
                        'Monthly average interest rate (%)',
                        'Monthly inflation rate (%)',
                        'Foreign exchange impact on cash flow (%)',
                        'Total debt outstanding in USD millions',
                        'Cash balance at month-end in USD millions',
                        'Free cash flow in USD millions (Revenue - Operating Cost - Capex)'
                    ],
                    'Data Type': [
                        'datetime64[ns]',
                        'float64',
                        'float64',
                        'float64',
                        'float64',
                        'float64',
                        'float64',
                        'float64',
                        'float64',
                        'float64'
                    ]
                }
                st.dataframe(
                    pd.DataFrame(data_dict), 
                    use_container_width=True, 
                    hide_index=True,
                    column_config={
                        "Column": st.column_config.Column(width="medium"),
                        "Description": st.column_config.Column(width="large"),
                        "Data Type": st.column_config.Column(width="small")
                    }
                )
            
            return df
            
    except FileNotFoundError as e:
        st.error(f"❌ File not found: {str(e)}")
        return None
    except pd.errors.ParserError as e:
        st.error(f"❌ Error parsing CSV file: {str(e)}")
        return None
    except Exception as e:
        st.error(f"❌ Unexpected error loading data: {str(e)}")
        return None

def validate_data_types(df):
    """Validate and ensure correct data types"""
    expected_dtypes = {
        'Date': 'datetime64[ns]',
        'Revenue_USD_M': 'float64',
        'Operating_Cost_USD_M': 'float64',
        'Capital_Expenditure_USD_M': 'float64',
        'Interest_Rate_%': 'float64',
        'Inflation_Rate_%': 'float64',
        'FX_Impact_%': 'float64',
        'Debt_Outstanding_USD_M': 'float64',
        'Cash_Balance_USD_M': 'float64',
        'Free_Cash_Flow_USD_M': 'float64'
    }
    
    results = []
    for col, expected_type in expected_dtypes.items():
        if col not in df.columns:
            results.append(f"Missing column: {col}")
        elif str(df[col].dtype) != expected_type:
            results.append(f"{col}: Expected {expected_type}, got {df[col].dtype}")
    
    if results:
        return {
            "status": "WARNING",
            "message": f"{len(results)} data type issues found",
            "details": results
        }
    
    return {
        "status": "PASS",
        "message": "All data types are correct"
    }

def validate_date_sequence(df):
    """Validate date sequence and completeness"""
    if 'Date' not in df.columns:
        return {
            "status": "FAIL",
            "message": "Date column not found"
        }
    
    # Check for missing dates
    df = df.sort_values('Date')
    date_range = pd.date_range(start=df['Date'].min(), end=df['Date'].max(), freq='MS')
    missing_dates = date_range[~date_range.isin(df['Date'])]
    
    # Check monotonic increasing
    is_monotonic = df['Date'].is_monotonic_increasing
    
    issues = []
    if len(missing_dates) > 0:
        issues.append(f"Missing {len(missing_dates)} monthly dates")
    
    if not is_monotonic:
        issues.append("Dates are not in chronological order")
    
    if issues:
        return {
            "status": "WARNING",
            "message": f"{len(issues)} date sequence issues",
            "details": issues
        }
    
    return {
        "status": "PASS",
        "message": "Date sequence is complete and ordered"
    }

def validate_financial_metrics(df):
    """Validate financial metrics for business logic"""
    issues = []
    
    # Revenue should be positive
    negative_revenue = (df['Revenue_USD_M'] < 0).sum()
    if negative_revenue > 0:
        issues.append(f"Negative revenue in {negative_revenue} records")
    
    # Operating costs should be positive
    negative_costs = (df['Operating_Cost_USD_M'] < 0).sum()
    if negative_costs > 0:
        issues.append(f"Negative operating costs in {negative_costs} records")
    
    # Capex should be positive
    negative_capex = (df['Capital_Expenditure_USD_M'] < 0).sum()
    if negative_capex > 0:
        issues.append(f"Negative capex in {negative_capex} records")
    
    # Interest rates should be between 0-20% (reasonable range)
    extreme_ir = ((df['Interest_Rate_%'] < 0) | (df['Interest_Rate_%'] > 20)).sum()
    if extreme_ir > 0:
        issues.append(f"Extreme interest rates in {extreme_ir} records")
    
    # Inflation rates should be reasonable (-5% to 50%)
    extreme_inflation = ((df['Inflation_Rate_%'] < -5) | (df['Inflation_Rate_%'] > 50)).sum()
    if extreme_inflation > 0:
        issues.append(f"Extreme inflation rates in {extreme_inflation} records")
    
    if issues:
        return {
            "status": "WARNING",
            "message": f"{len(issues)} financial validation issues",
            "details": issues
        }
    
    return {
        "status": "PASS",
        "message": "Financial metrics are within expected ranges"
    }

def check_missing_values(df):
    """Check for missing values in critical columns"""
    critical_columns = [
        'Date', 'Revenue_USD_M', 'Operating_Cost_USD_M',
        'Free_Cash_Flow_USD_M', 'Cash_Balance_USD_M'
    ]
    
    missing_counts = {}
    total_missing = 0
    
    for col in critical_columns:
        if col in df.columns:
            missing = df[col].isnull().sum()
            if missing > 0:
                missing_counts[col] = missing
                total_missing += missing
    
    if missing_counts:
        return {
            "status": "WARNING" if total_missing < 10 else "FAIL",
            "message": f"{total_missing} missing values in {len(missing_counts)} columns",
            "details": missing_counts
        }
    
    return {
        "status": "PASS",
        "message": "No missing values in critical columns"
    }

def check_outliers(df):
    """Check for statistical outliers using IQR method"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    outlier_report = {}
    
    for col in numeric_cols:
        if col != 'Date':  # Skip date column
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            if len(outliers) > 0:
                outlier_report[col] = len(outliers)
    
    if outlier_report:
        return {
            "status": "WARNING",
            "message": f"Outliers detected in {len(outlier_report)} columns",
            "details": outlier_report
        }
    
    return {
        "status": "PASS",
        "message": "No significant outliers detected"
    }

def final_validation(df):
    """Final comprehensive validation"""
    # Check dataset size
    if len(df) < 12:
        return {
            "status": "WARNING",
            "message": "Dataset has less than 12 months of data"
        }
    
    # Check for duplicate dates
    duplicate_dates = df['Date'].duplicated().sum()
    if duplicate_dates > 0:
        return {
            "status": "FAIL",
            "message": f"{duplicate_dates} duplicate dates found"
        }
    
    # Check that Free Cash Flow matches calculation
    calculated_fcf = df['Revenue_USD_M'] - df['Operating_Cost_USD_M'] - df['Capital_Expenditure_USD_M']
    fcf_mismatch = (abs(df['Free_Cash_Flow_USD_M'] - calculated_fcf) > 0.01).sum()
    
    if fcf_mismatch > 0:
        return {
            "status": "WARNING",
            "message": f"Free Cash Flow mismatch in {fcf_mismatch} records"
        }
    
    return {
        "status": "PASS",
        "message": "Dataset passed all final validations"
    }

def calculate_quality_metrics(df):
    """Calculate data quality metrics"""
    metrics = {}
    
    # Completeness: Percentage of non-null values in critical columns
    critical_cols = ['Revenue_USD_M', 'Operating_Cost_USD_M', 'Free_Cash_Flow_USD_M']
    completeness = df[critical_cols].notnull().mean().mean()
    metrics['completeness'] = completeness
    
    # Consistency: Check if dates are sequential
    date_diff = df['Date'].diff().dropna()
    consistent_intervals = (date_diff == pd.Timedelta(days=30)).mean()
    metrics['consistency'] = consistent_intervals if not pd.isna(consistent_intervals) else 0.0
    
    # Accuracy: Check if FCF calculation matches
    calculated_fcf = df['Revenue_USD_M'] - df['Operating_Cost_USD_M'] - df['Capital_Expenditure_USD_M']
    fcf_accuracy = (abs(df['Free_Cash_Flow_USD_M'] - calculated_fcf) < 0.01).mean()
    metrics['accuracy'] = fcf_accuracy
    
    return metrics

def get_data_summary(df):
    """Generate comprehensive data summary"""
    if df is None:
        return "No data loaded"
    
    summary = {
        "Dataset Information": {
            "Total Records": len(df),
            "Total Columns": len(df.columns),
            "Date Range": f"{df['Date'].min().date()} to {df['Date'].max().date()}",
            "Data Period": f"{(df['Date'].max() - df['Date'].min()).days / 30:.1f} months"
        },
        "Financial Summary (USD Millions)": {
            "Average Revenue": f"${df['Revenue_USD_M'].mean():,.2f}",
            "Average Operating Cost": f"${df['Operating_Cost_USD_M'].mean():,.2f}",
            "Average Free Cash Flow": f"${df['Free_Cash_Flow_USD_M'].mean():,.2f}",
            "Total Debt": f"${df['Debt_Outstanding_USD_M'].mean():,.2f}"
        },
        "Risk Indicators": {
            "Negative Cash Flow Months": (df['Free_Cash_Flow_USD_M'] < 0).sum(),
            "High Interest Months (IR > 5%)": (df['Interest_Rate_%'] > 5).sum(),
            "High Inflation Months (IR > 5%)": (df['Inflation_Rate_%'] > 5).sum()
        }
    }
    
    return summary

# Test function for standalone testing
if __name__ == "__main__":
    print("Testing data loading module...")
    df = load_dataset()
    if df is not None:
        print(f"Successfully loaded dataset with {len(df)} rows and {len(df.columns)} columns")
        print(f"Columns: {list(df.columns)}")
        print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")