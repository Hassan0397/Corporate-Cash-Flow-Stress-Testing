"""
Exploratory Data Analysis Module for Corporate Cash Flow Stress Testing Platform
Professional Financial Analytics with Interactive Visualizations
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timedelta
import warnings
from scipy import stats
warnings.filterwarnings('ignore')

def perform_eda(df):
    """
    Main EDA function with comprehensive financial analytics dashboard
    
    Args:
        df (pandas.DataFrame): Cleaned dataset from clean_data module
    
    Returns:
        dict: Key insights and findings from the analysis
    """
    
    if df is None:
        st.error("❌ No data provided for analysis")
        return None
    
    # Create professional EDA header
    st.markdown("""
    <div style="background: linear-gradient(135deg, #2563eb15 0%, #7c3aed15 100%); 
                padding: 2rem; border-radius: 12px; margin-bottom: 2rem;
                border: 1px solid #e2e8f0;">
        <h3 style="color: #1e293b; margin-bottom: 1rem; display: flex; align-items: center;">
            <span style="background: #3b82f6; color: white; padding: 0.5rem 1rem; 
                        border-radius: 8px; margin-right: 1rem; font-size: 1.2rem;">
                🔍
            </span>
            Financial Intelligence Dashboard
        </h3>
        <p style="color: #475569; line-height: 1.6; margin-bottom: 0;">
            Advanced exploratory analysis revealing hidden patterns, correlations, and 
            insights in your corporate cash flow data. Interactive visualizations 
            powered by Plotly for deep financial intelligence.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize insights dictionary
    insights = {
        'timestamp': datetime.now(),
        'data_shape': df.shape,
        'date_range': {'start': df['Date'].min(), 'end': df['Date'].max()},
        'key_metrics': {},
        'correlations': {},
        'anomalies': [],
        'trends': {},
        'recommendations': []
    }
    
    # Create main analysis tabs
    tab_overview, tab_timeseries, tab_correlation, tab_distribution, tab_insights = st.tabs([
        "📊 Financial Overview",
        "📈 Time Series Analysis",
        "🔄 Correlation & Relationships",
        "📊 Distribution Analysis",
        "💡 Insights & Recommendations"
    ])
    
    with tab_overview:
        insights.update(financial_overview(df))
    
    with tab_timeseries:
        insights.update(time_series_analysis(df))
    
    with tab_correlation:
        insights.update(correlation_analysis(df))
    
    with tab_distribution:
        insights.update(distribution_analysis(df))
    
    with tab_insights:
        display_insights(insights)
        generate_recommendations(insights)
    
    return insights

def financial_overview(df):
    """Create comprehensive financial overview dashboard"""
    insights = {'key_metrics': {}, 'anomalies': []}
    
    st.markdown("### 📋 Executive Summary")
    
    # Key metrics row with professional styling
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_revenue = df['Revenue_USD_M'].mean()
        revenue_trend = df['Revenue_USD_M'].iloc[-1] - df['Revenue_USD_M'].iloc[0]
        revenue_pct_change = (revenue_trend / df['Revenue_USD_M'].iloc[0]) * 100 if df['Revenue_USD_M'].iloc[0] != 0 else 0
        
        st.markdown(f"""
        <div style="background: white; padding: 1.5rem; border-radius: 12px; 
                    border: 1px solid #e2e8f0; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <div style="color: #64748b; font-size: 0.9rem; text-transform: uppercase; 
                        letter-spacing: 0.05em; margin-bottom: 0.5rem;">
                💰 Average Revenue
            </div>
            <div style="font-size: 2rem; font-weight: 700; color: #1e293b; 
                        margin-bottom: 0.5rem;">
                ${avg_revenue:,.1f}M
            </div>
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="color: {'#10b981' if revenue_trend > 0 else '#ef4444'}; 
                            font-weight: 600; font-size: 0.9rem;">
                    {'▲' if revenue_trend > 0 else '▼'} {abs(revenue_pct_change):.1f}%
                </span>
                <span style="color: #64748b; font-size: 0.8rem;">vs period start</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        insights['key_metrics']['avg_revenue'] = avg_revenue
        insights['key_metrics']['revenue_growth'] = revenue_pct_change
    
    with col2:
        avg_fcf = df['Free_Cash_Flow_USD_M'].mean()
        fcf_trend = df['Free_Cash_Flow_USD_M'].iloc[-1] - df['Free_Cash_Flow_USD_M'].iloc[0]
        fcf_pct_change = (fcf_trend / abs(df['Free_Cash_Flow_USD_M'].iloc[0])) * 100 if df['Free_Cash_Flow_USD_M'].iloc[0] != 0 else 0
        negative_fcf_months = (df['Free_Cash_Flow_USD_M'] < 0).sum()
        
        st.markdown(f"""
        <div style="background: white; padding: 1.5rem; border-radius: 12px; 
                    border: 1px solid #e2e8f0; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <div style="color: #64748b; font-size: 0.9rem; text-transform: uppercase; 
                        letter-spacing: 0.05em; margin-bottom: 0.5rem;">
                💵 Avg Free Cash Flow
            </div>
            <div style="font-size: 2rem; font-weight: 700; color: {'#10b981' if avg_fcf > 0 else '#ef4444'}; 
                        margin-bottom: 0.5rem;">
                ${avg_fcf:,.1f}M
            </div>
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="color: #64748b; font-size: 0.9rem;">
                    {negative_fcf_months} months negative
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        insights['key_metrics']['avg_fcf'] = avg_fcf
        insights['key_metrics']['negative_fcf_months'] = int(negative_fcf_months)
    
    with col3:
        avg_debt = df['Debt_Outstanding_USD_M'].mean()
        avg_cash = df['Cash_Balance_USD_M'].mean()
        cash_to_debt = (avg_cash / avg_debt) * 100 if avg_debt > 0 else 0
        
        st.markdown(f"""
        <div style="background: white; padding: 1.5rem; border-radius: 12px; 
                    border: 1px solid #e2e8f0; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <div style="color: #64748b; font-size: 0.9rem; text-transform: uppercase; 
                        letter-spacing: 0.05em; margin-bottom: 0.5rem;">
                🏦 Debt & Cash Position
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span style="color: #1e293b; font-weight: 600;">Debt:</span>
                <span style="color: #ef4444; font-weight: 700;">${avg_debt:,.1f}M</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span style="color: #1e293b; font-weight: 600;">Cash:</span>
                <span style="color: #10b981; font-weight: 700;">${avg_cash:,.1f}M</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="color: #1e293b; font-weight: 600;">Cash/Debt:</span>
                <span style="color: {'#10b981' if cash_to_debt > 50 else '#f59e0b'}; 
                            font-weight: 700;">{cash_to_debt:.1f}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        insights['key_metrics']['avg_debt'] = avg_debt
        insights['key_metrics']['avg_cash'] = avg_cash
        insights['key_metrics']['cash_to_debt_ratio'] = cash_to_debt
    
    with col4:
        avg_ir = df['Interest_Rate_%'].mean()
        avg_inflation = df['Inflation_Rate_%'].mean()
        real_rate = avg_ir - avg_inflation
        
        st.markdown(f"""
        <div style="background: white; padding: 1.5rem; border-radius: 12px; 
                    border: 1px solid #e2e8f0; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <div style="color: #64748b; font-size: 0.9rem; text-transform: uppercase; 
                        letter-spacing: 0.05em; margin-bottom: 0.5rem;">
                📈 Macro Environment
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                <span style="color: #1e293b; font-size: 0.9rem;">Interest Rate:</span>
                <span style="color: #1e293b; font-weight: 600;">{avg_ir:.2f}%</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                <span style="color: #1e293b; font-size: 0.9rem;">Inflation:</span>
                <span style="color: #1e293b; font-weight: 600;">{avg_inflation:.2f}%</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="color: #1e293b; font-size: 0.9rem;">Real Rate:</span>
                <span style="color: {'#10b981' if real_rate > 0 else '#ef4444'}; 
                            font-weight: 600;">{real_rate:.2f}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        insights['key_metrics']['avg_interest_rate'] = avg_ir
        insights['key_metrics']['avg_inflation'] = avg_inflation
        insights['key_metrics']['real_interest_rate'] = real_rate
    
    st.markdown("---")
    
    # Financial Health Gauge
    st.markdown("### 📊 Financial Health Score")
    
    # Calculate health score based on multiple factors
    health_score = calculate_health_score(df)
    insights['key_metrics']['health_score'] = health_score
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Create gauge chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = health_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Corporate Health Score", 'font': {'size': 20, 'color': '#1e293b'}},
            delta = {'reference': 70, 'increasing': {'color': "#10b981"}, 
                    'decreasing': {'color': "#ef4444"}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
                'bar': {'color': "#3b82f6"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "#e2e8f0",
                'steps': [
                    {'range': [0, 40], 'color': '#fee2e2'},
                    {'range': [40, 70], 'color': '#fef3c7'},
                    {'range': [70, 100], 'color': '#dcfce7'}
                ],
                'threshold': {
                    'line': {'color': "#ef4444", 'width': 4},
                    'thickness': 0.75,
                    'value': 40
                }
            }
        ))
        
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor="white",
            font={'color': "#1e293b", 'family': "Inter, sans-serif"}
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 12px; 
                    border: 1px solid #e2e8f0; height: 100%;">
            <h4 style="color: #1e293b; margin-bottom: 1rem; font-size: 1.1rem;">Health Score Components</h4>
        """, unsafe_allow_html=True)
        
        # Component breakdown
        components = {
            'Profitability': min(100, max(0, (df['Free_Cash_Flow_USD_M'].mean() / df['Revenue_USD_M'].mean() * 100) + 50)),
            'Liquidity': min(100, (df['Cash_Balance_USD_M'].mean() / df['Debt_Outstanding_USD_M'].mean() * 100)),
            'Efficiency': min(100, (1 - df['Operating_Cost_USD_M'].mean() / df['Revenue_USD_M'].mean()) * 100),
            'Stability': min(100, 100 - (df['Free_Cash_Flow_USD_M'].std() / df['Free_Cash_Flow_USD_M'].mean() * 10))
        }
        
        for comp, score in components.items():
            color = '#10b981' if score >= 70 else '#f59e0b' if score >= 40 else '#ef4444'
            st.markdown(f"""
            <div style="margin-bottom: 1rem;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                    <span style="color: #475569;">{comp}</span>
                    <span style="color: {color}; font-weight: 600;">{score:.1f}%</span>
                </div>
                <div style="width: 100%; background: #e2e8f0; border-radius: 4px; height: 8px;">
                    <div style="width: {score}%; background: {color}; 
                                border-radius: 4px; height: 8px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Revenue vs Cost Analysis
    st.markdown("---")
    st.markdown("### 💰 Revenue & Cost Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Revenue and Operating Cost over time
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Scatter(
                x=df['Date'],
                y=df['Revenue_USD_M'],
                name="Revenue",
                line=dict(color="#3b82f6", width=3),
                fill='tonexty',
                fillcolor='rgba(59, 130, 246, 0.1)'
            ),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['Date'],
                y=df['Operating_Cost_USD_M'],
                name="Operating Cost",
                line=dict(color="#ef4444", width=2, dash='dash')
            ),
            secondary_y=False
        )
        
        fig.update_layout(
            title="Revenue vs Operating Cost Trend",
            height=400,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        fig.update_xaxes(
            title="Date",
            gridcolor='#e2e8f0',
            showgrid=True
        )
        
        fig.update_yaxes(
            title="USD (Millions)",
            gridcolor='#e2e8f0',
            showgrid=True,
            secondary_y=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Profit margin analysis
        df['Profit_Margin'] = ((df['Revenue_USD_M'] - df['Operating_Cost_USD_M']) / df['Revenue_USD_M']) * 100
        
        avg_margin = df['Profit_Margin'].mean()
        margin_trend = df['Profit_Margin'].iloc[-1] - df['Profit_Margin'].iloc[0]
        
        st.markdown(f"""
        <div style="background: white; padding: 1.5rem; border-radius: 12px; 
                    border: 1px solid #e2e8f0;">
            <h4 style="color: #1e293b; margin-bottom: 1rem;">Profit Margin Analysis</h4>
            <div style="margin-bottom: 1.5rem;">
                <div style="color: #64748b; font-size: 0.9rem;">Average Margin</div>
                <div style="font-size: 2rem; font-weight: 700; color: #1e293b;">
                    {avg_margin:.1f}%
                </div>
                <div style="color: {'#10b981' if margin_trend > 0 else '#ef4444'};">
                    {margin_trend:+.1f}% from start
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Margin distribution
        margin_bins = [-20, 0, 10, 20, 30, 100]
        margin_labels = ['Negative', '0-10%', '10-20%', '20-30%', '30%+']
        margin_dist = pd.cut(df['Profit_Margin'], bins=margin_bins, labels=margin_labels).value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=margin_dist.index,
            values=margin_dist.values,
            hole=.4,
            marker_colors=['#ef4444', '#f59e0b', '#3b82f6', '#10b981', '#8b5cf6']
        )])
        
        fig.update_layout(
            height=250,
            margin=dict(l=20, r=20, t=20, b=20),
            showlegend=False,
            paper_bgcolor='white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    return insights

def time_series_analysis(df):
    """Perform comprehensive time series analysis"""
    insights = {'trends': {}}
    
    st.markdown("### 📈 Time Series Analysis & Trends")
    
    # Create time series tabs
    ts_tab1, ts_tab2, ts_tab3 = st.tabs([
        "Cash Flow Analysis",
        "Debt & Cash Position",
        "Seasonality Patterns"
    ])
    
    with ts_tab1:
        # Free Cash Flow with trend line
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Free Cash Flow Trend", "Cash Flow Volatility"),
            vertical_spacing=0.15,
            row_heights=[0.6, 0.4]
        )
        
        # FCF with trend
        fig.add_trace(
            go.Scatter(
                x=df['Date'],
                y=df['Free_Cash_Flow_USD_M'],
                name="Free Cash Flow",
                line=dict(color="#8b5cf6", width=2),
                mode='lines+markers',
                marker=dict(
                    size=6,
                    color=df['Free_Cash_Flow_USD_M'],
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title="FCF ($M)")
                )
            ),
            row=1, col=1
        )
        
        # Add trend line
        z = np.polyfit(range(len(df)), df['Free_Cash_Flow_USD_M'], 1)
        p = np.poly1d(z)
        fig.add_trace(
            go.Scatter(
                x=df['Date'],
                y=p(range(len(df))),
                name="Trend",
                line=dict(color="#1e293b", width=2, dash='dash')
            ),
            row=1, col=1
        )
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dot", line_color="#94a3b8", row=1, col=1)
        
        # Rolling volatility
        rolling_std = df['Free_Cash_Flow_USD_M'].rolling(window=6, min_periods=1).std()
        
        fig.add_trace(
            go.Scatter(
                x=df['Date'],
                y=rolling_std,
                name="6-Month Volatility",
                line=dict(color="#f59e0b", width=2),
                fill='tozeroy',
                fillcolor='rgba(245, 158, 11, 0.1)'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=600,
            hovermode='x unified',
            showlegend=True,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        fig.update_xaxes(gridcolor='#e2e8f0', row=1, col=1)
        fig.update_xaxes(gridcolor='#e2e8f0', row=2, col=1)
        fig.update_yaxes(gridcolor='#e2e8f0', row=1, col=1, title="USD (Millions)")
        fig.update_yaxes(gridcolor='#e2e8f0', row=2, col=1, title="Volatility (Std Dev)")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # FCF Statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Peak FCF",
                f"${df['Free_Cash_Flow_USD_M'].max():.1f}M",
                f"{df['Free_Cash_Flow_USD_M'].idxmax()}",
                delta_color="off"
            )
        
        with col2:
            st.metric(
                "Lowest FCF",
                f"${df['Free_Cash_Flow_USD_M'].min():.1f}M",
                f"{df['Free_Cash_Flow_USD_M'].idxmin()}",
                delta_color="off"
            )
        
        with col3:
            volatility = df['Free_Cash_Flow_USD_M'].std() / df['Free_Cash_Flow_USD_M'].mean() * 100
            st.metric(
                "Coefficient of Variation",
                f"{volatility:.1f}%",
                delta=None
            )
        
        with col4:
            positive_months = (df['Free_Cash_Flow_USD_M'] > 0).sum()
            st.metric(
                "Positive FCF Months",
                f"{positive_months}/{len(df)}",
                f"{(positive_months/len(df)*100):.1f}%"
            )
        
        insights['trends']['fcf_trend'] = z[0]  # Slope of trend
        insights['trends']['fcf_volatility'] = volatility
    
    with ts_tab2:
        # Debt and Cash Position
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Debt Outstanding vs Cash Balance", "Net Debt Position"),
            vertical_spacing=0.15,
            row_heights=[0.6, 0.4]
        )
        
        # Debt and Cash
        fig.add_trace(
            go.Scatter(
                x=df['Date'],
                y=df['Debt_Outstanding_USD_M'],
                name="Debt Outstanding",
                line=dict(color="#ef4444", width=2),
                fill='tozeroy',
                fillcolor='rgba(239, 68, 68, 0.1)'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['Date'],
                y=df['Cash_Balance_USD_M'],
                name="Cash Balance",
                line=dict(color="#10b981", width=2),
                fill='tozeroy',
                fillcolor='rgba(16, 185, 129, 0.1)'
            ),
            row=1, col=1
        )
        
        # Net Debt (Debt - Cash)
        net_debt = df['Debt_Outstanding_USD_M'] - df['Cash_Balance_USD_M']
        
        fig.add_trace(
            go.Scatter(
                x=df['Date'],
                y=net_debt,
                name="Net Debt",
                line=dict(color="#8b5cf6", width=2),
                fill='tozeroy',
                fillcolor='rgba(139, 92, 246, 0.1)'
            ),
            row=2, col=1
        )
        
        fig.add_hline(y=0, line_dash="dot", line_color="#94a3b8", row=2, col=1)
        
        fig.update_layout(
            height=600,
            hovermode='x unified',
            showlegend=True,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        fig.update_xaxes(gridcolor='#e2e8f0', row=1, col=1)
        fig.update_xaxes(gridcolor='#e2e8f0', row=2, col=1)
        fig.update_yaxes(gridcolor='#e2e8f0', row=1, col=1, title="USD (Millions)")
        fig.update_yaxes(gridcolor='#e2e8f0', row=2, col=1, title="USD (Millions)")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Debt metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            debt_trend = df['Debt_Outstanding_USD_M'].iloc[-1] - df['Debt_Outstanding_USD_M'].iloc[0]
            st.metric(
                "Debt Trend",
                f"{'↑' if debt_trend > 0 else '↓'} ${abs(debt_trend):.1f}M",
                f"Since {df['Date'].iloc[0].strftime('%b %Y')}"
            )
        
        with col2:
            cash_trend = df['Cash_Balance_USD_M'].iloc[-1] - df['Cash_Balance_USD_M'].iloc[0]
            st.metric(
                "Cash Trend",
                f"{'↑' if cash_trend > 0 else '↓'} ${abs(cash_trend):.1f}M",
                f"Since {df['Date'].iloc[0].strftime('%b %Y')}"
            )
        
        with col3:
            avg_net_debt = net_debt.mean()
            st.metric(
                "Average Net Debt",
                f"${avg_net_debt:.1f}M",
                delta=None
            )
    
    with ts_tab3:
        # Seasonality Analysis
        st.markdown("#### 📅 Seasonal Patterns")
        
        # Extract month and quarter
        df['Month'] = df['Date'].dt.month
        df['Quarter'] = df['Date'].dt.quarter
        df['Year'] = df['Date'].dt.year
        
        # Monthly pattern
        monthly_avg = df.groupby('Month')['Free_Cash_Flow_USD_M'].agg(['mean', 'std']).reset_index()
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_avg['Month_Name'] = monthly_avg['Month'].map(lambda x: month_names[x-1])
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=monthly_avg['Month_Name'],
                y=monthly_avg['mean'],
                error_y=dict(
                    type='data',
                    array=monthly_avg['std'],
                    visible=True,
                    color='#94a3b8'
                ),
                marker_color='#3b82f6',
                marker_line_color='#2563eb',
                marker_line_width=1,
                opacity=0.8
            ))
            
            fig.update_layout(
                title="Average Free Cash Flow by Month",
                xaxis_title="Month",
                yaxis_title="Avg FCF (USD Millions)",
                height=400,
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            
            fig.update_xaxes(gridcolor='#e2e8f0')
            fig.update_yaxes(gridcolor='#e2e8f0')
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Quarterly pattern
            quarterly_avg = df.groupby('Quarter')['Free_Cash_Flow_USD_M'].mean().reset_index()
            quarter_names = ['Q1', 'Q2', 'Q3', 'Q4']
            quarterly_avg['Quarter_Name'] = quarterly_avg['Quarter'].map(lambda x: quarter_names[x-1])
            
            fig = go.Figure()
            
            fig.add_trace(go.Pie(
                labels=quarterly_avg['Quarter_Name'],
                values=quarterly_avg['Free_Cash_Flow_USD_M'],
                hole=.4,
                marker_colors=['#ef4444', '#f59e0b', '#3b82f6', '#10b981'],
                textinfo='label+percent',
                textposition='outside'
            ))
            
            fig.update_layout(
                title="Quarterly Cash Flow Distribution",
                height=400,
                showlegend=False,
                paper_bgcolor='white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Year-over-Year comparison
        st.markdown("#### 📊 Year-over-Year Comparison")
        
        # Get last 3 years of data
        latest_year = df['Date'].dt.year.max()
        years_to_show = [latest_year-2, latest_year-1, latest_year]
        years_to_show = [y for y in years_to_show if y in df['Year'].values]
        
        if len(years_to_show) >= 2:
            yoy_data = df[df['Year'].isin(years_to_show)].copy()
            yoy_data['Month'] = yoy_data['Date'].dt.month
            
            fig = go.Figure()
            
            colors = ['#94a3b8', '#64748b', '#0f172a']
            for i, year in enumerate(sorted(years_to_show)):
                year_data = yoy_data[yoy_data['Year'] == year].sort_values('Month')
                fig.add_trace(go.Scatter(
                    x=year_data['Month'].map(lambda x: month_names[x-1]),
                    y=year_data['Free_Cash_Flow_USD_M'],
                    name=str(year),
                    line=dict(color=colors[i], width=3 if year == latest_year else 2,
                             dash='solid' if year == latest_year else 'dash'),
                    mode='lines+markers'
                ))
            
            fig.update_layout(
                title=f"Free Cash Flow Comparison: {', '.join(map(str, sorted(years_to_show)))}",
                xaxis_title="Month",
                yaxis_title="Free Cash Flow (USD Millions)",
                height=450,
                hovermode='x unified',
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            
            fig.update_xaxes(gridcolor='#e2e8f0')
            fig.update_yaxes(gridcolor='#e2e8f0')
            
            st.plotly_chart(fig, use_container_width=True)
    
    return insights

def correlation_analysis(df):
    """Perform correlation analysis between financial metrics"""
    insights = {'correlations': {}}
    
    st.markdown("### 🔗 Correlation & Relationship Analysis")
    
    # Select numerical columns for correlation
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove date-related columns and outlier flags
    exclude_cols = ['Year', 'Month', 'Quarter', 'YearMonth']
    exclude_cols.extend([col for col in df.columns if '_outlier_flag' in col])
    exclude_cols.extend([col for col in df.columns if '_Rolling_' in col])
    
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        # Interactive correlation heatmap - FIXED: removed 'titleside' property
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu_r',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False,
            colorbar=dict(
                title="Correlation",
                tickvals=[-1, -0.5, 0, 0.5, 1],
                ticktext=['-1', '-0.5', '0', '0.5', '1']
            )
        ))
        
        fig.update_layout(
            title="Financial Metrics Correlation Matrix",
            height=600,
            width=600,
            xaxis=dict(tickangle=45),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 🔍 Key Correlations")
        
        # Find strongest correlations
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_pairs.append({
                    'Variable 1': corr_matrix.columns[i],
                    'Variable 2': corr_matrix.columns[j],
                    'Correlation': corr_matrix.iloc[i, j]
                })
        
        corr_df = pd.DataFrame(corr_pairs)
        corr_df['Abs Correlation'] = abs(corr_df['Correlation'])
        corr_df = corr_df.sort_values('Abs Correlation', ascending=False)
        
        # Display top correlations
        st.markdown("**Top Positive Correlations:**")
        top_positive = corr_df[corr_df['Correlation'] > 0.5].head(3)
        if not top_positive.empty:
            for _, row in top_positive.iterrows():
                st.markdown(f"""
                <div style="background: #dcfce7; padding: 0.75rem; border-radius: 8px; 
                            margin-bottom: 0.5rem; border-left: 4px solid #22c55e;">
                    <span style="font-weight: 600;">{row['Correlation']:.2f}</span> 
                    {row['Variable 1']} ↔ {row['Variable 2']}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No strong positive correlations found")
        
        st.markdown("**Top Negative Correlations:**")
        top_negative = corr_df[corr_df['Correlation'] < -0.5].head(3)
        if not top_negative.empty:
            for _, row in top_negative.iterrows():
                st.markdown(f"""
                <div style="background: #fee2e2; padding: 0.75rem; border-radius: 8px; 
                            margin-bottom: 0.5rem; border-left: 4px solid #ef4444;">
                    <span style="font-weight: 600;">{row['Correlation']:.2f}</span> 
                    {row['Variable 1']} ↔ {row['Variable 2']}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No strong negative correlations found")
        
        # Store insights
        insights['correlations']['top_positive'] = top_positive.to_dict('records')[:3] if not top_positive.empty else []
        insights['correlations']['top_negative'] = top_negative.to_dict('records')[:3] if not top_negative.empty else []
    
    # Scatter plot analysis
    st.markdown("---")
    st.markdown("#### 📈 Relationship Explorer")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("**Select variables to explore:**")
        x_var = st.selectbox("X-axis", numeric_cols, index=numeric_cols.index('Interest_Rate_%') if 'Interest_Rate_%' in numeric_cols else 0)
        y_var = st.selectbox("Y-axis", numeric_cols, index=numeric_cols.index('Free_Cash_Flow_USD_M') if 'Free_Cash_Flow_USD_M' in numeric_cols else 0)
        color_var = st.selectbox("Color by", ['None'] + numeric_cols, index=0)
        
        show_trend = st.checkbox("Show trend line", value=True)
    
    with col2:
        fig = go.Figure()
        
        # Scatter plot
        if color_var == 'None':
            fig.add_trace(go.Scatter(
                x=df[x_var],
                y=df[y_var],
                mode='markers',
                marker=dict(
                    size=10,
                    color='#3b82f6',
                    opacity=0.7,
                    line=dict(width=1, color='white')
                ),
                text=df['Date'].dt.strftime('%b %Y'),
                hovertemplate='<b>%{text}</b><br>' +
                             f'{x_var}: %{{x:,.2f}}<br>' +
                             f'{y_var}: %{{y:,.2f}}<br>' +
                             '<extra></extra>'
            ))
        else:
            fig.add_trace(go.Scatter(
                x=df[x_var],
                y=df[y_var],
                mode='markers',
                marker=dict(
                    size=10,
                    color=df[color_var],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title=color_var),
                    opacity=0.7,
                    line=dict(width=1, color='white')
                ),
                text=df['Date'].dt.strftime('%b %Y'),
                hovertemplate='<b>%{text}</b><br>' +
                             f'{x_var}: %{{x:,.2f}}<br>' +
                             f'{y_var}: %{{y:,.2f}}<br>' +
                             f'{color_var}: %{{marker.color:,.2f}}<br>' +
                             '<extra></extra>'
            ))
        
        # Add trend line
        if show_trend:
            z = np.polyfit(df[x_var].values, df[y_var].values, 1)
            p = np.poly1d(z)
            x_range = np.linspace(df[x_var].min(), df[x_var].max(), 100)
            
            fig.add_trace(go.Scatter(
                x=x_range,
                y=p(x_range),
                mode='lines',
                name=f'Trend (R²={np.corrcoef(df[x_var], df[y_var])[0,1]**2:.3f})',
                line=dict(color='#ef4444', width=2, dash='dash')
            ))
        
        fig.update_layout(
            title=f"{y_var} vs {x_var}",
            xaxis_title=x_var,
            yaxis_title=y_var,
            height=500,
            hovermode='closest',
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        fig.update_xaxes(gridcolor='#e2e8f0')
        fig.update_yaxes(gridcolor='#e2e8f0')
        
        st.plotly_chart(fig, use_container_width=True)
    
    return insights

def distribution_analysis(df):
    """Analyze distributions of key financial metrics"""
    insights = {}
    
    st.markdown("### 📊 Distribution Analysis")
    
    # Select metrics for distribution analysis
    metrics = [
        'Revenue_USD_M',
        'Free_Cash_Flow_USD_M',
        'Operating_Cost_USD_M',
        'Debt_Outstanding_USD_M',
        'Cash_Balance_USD_M',
        'Capital_Expenditure_USD_M'
    ]
    
    metrics = [m for m in metrics if m in df.columns]
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        selected_metric = st.selectbox(
            "Select metric to analyze",
            metrics,
            index=metrics.index('Free_Cash_Flow_USD_M') if 'Free_Cash_Flow_USD_M' in metrics else 0
        )
        
        # Statistics
        st.markdown("#### 📈 Descriptive Statistics")
        
        stats_data = {
            'Count': len(df),
            'Mean': f"${df[selected_metric].mean():,.2f}M",
            'Median': f"${df[selected_metric].median():,.2f}M",
            'Std Dev': f"${df[selected_metric].std():,.2f}M",
            'Min': f"${df[selected_metric].min():,.2f}M",
            'Max': f"${df[selected_metric].max():,.2f}M",
            'Skewness': f"{df[selected_metric].skew():.3f}",
            'Kurtosis': f"{df[selected_metric].kurtosis():.3f}"
        }
        
        stats_df = pd.DataFrame(list(stats_data.items()), columns=['Statistic', 'Value'])
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    with col2:
        # Distribution plot
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Histogram with KDE", "Box Plot", "Q-Q Plot", "Violin Plot"),
            vertical_spacing=0.15,
            horizontal_spacing=0.15
        )
        
        # Histogram with KDE
        hist_data = df[selected_metric].dropna()
        
        fig.add_trace(
            go.Histogram(
                x=hist_data,
                name="Histogram",
                nbinsx=30,
                marker_color='#3b82f6',
                opacity=0.7,
                histnorm='probability density'
            ),
            row=1, col=1
        )
        
        # Add KDE
        kde_x = np.linspace(hist_data.min(), hist_data.max(), 100)
        kde = stats.gaussian_kde(hist_data)
        
        fig.add_trace(
            go.Scatter(
                x=kde_x,
                y=kde(kde_x),
                name="KDE",
                line=dict(color='#ef4444', width=2)
            ),
            row=1, col=1
        )
        
        # Box plot
        fig.add_trace(
            go.Box(
                y=hist_data,
                name="Box Plot",
                marker_color='#3b82f6',
                boxmean='sd'
            ),
            row=1, col=2
        )
        
        # Q-Q plot
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(hist_data)))
        sample_quantiles = np.sort(hist_data)
        
        fig.add_trace(
            go.Scatter(
                x=theoretical_quantiles,
                y=sample_quantiles,
                mode='markers',
                name="Q-Q Plot",
                marker=dict(color='#3b82f6', size=6)
            ),
            row=2, col=1
        )
        
        # Add reference line
        min_val = min(theoretical_quantiles.min(), sample_quantiles.min())
        max_val = max(theoretical_quantiles.max(), sample_quantiles.max())
        
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Normal',
                line=dict(color='#ef4444', dash='dash'),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Violin plot
        fig.add_trace(
            go.Violin(
                y=hist_data,
                name="Violin Plot",
                box_visible=True,
                meanline_visible=True,
                fillcolor='#3b82f6',
                line_color='#2563eb',
                opacity=0.7
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=700,
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        fig.update_xaxes(gridcolor='#e2e8f0')
        fig.update_yaxes(gridcolor='#e2e8f0')
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Outlier detection
    st.markdown("---")
    st.markdown("#### 🚨 Outlier Detection")
    
    # IQR method for outlier detection
    Q1 = df[selected_metric].quantile(0.25)
    Q3 = df[selected_metric].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[selected_metric] < lower_bound) | (df[selected_metric] > upper_bound)]
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown(f"""
        <div style="background: white; padding: 1.5rem; border-radius: 12px; 
                    border: 1px solid #e2e8f0;">
            <h4 style="color: #1e293b; margin-bottom: 1rem;">Outlier Summary</h4>
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span style="color: #475569;">IQR Method:</span>
                <span style="font-weight: 600;">1.5 × IQR</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span style="color: #475569;">Lower Bound:</span>
                <span style="font-weight: 600;">${lower_bound:,.2f}M</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span style="color: #475569;">Upper Bound:</span>
                <span style="font-weight: 600;">${upper_bound:,.2f}M</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-top: 1rem; 
                        padding-top: 1rem; border-top: 1px solid #e2e8f0;">
                <span style="color: #475569; font-weight: 600;">Outliers Detected:</span>
                <span style="color: {'#ef4444' if len(outliers) > 0 else '#10b981'}; 
                            font-weight: 700;">{len(outliers)}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if len(outliers) > 0:
            st.markdown("**Outlier Details:**")
            outlier_dates = outliers['Date'].dt.strftime('%b %Y').tolist()
            outlier_values = outliers[selected_metric].tolist()
            
            for date, value in zip(outlier_dates[:5], outlier_values[:5]):
                st.markdown(f"""
                <div style="background: #fee2e2; padding: 0.5rem; border-radius: 6px; 
                            margin-bottom: 0.5rem; display: flex; justify-content: space-between;">
                    <span style="color: #991b1b;">{date}</span>
                    <span style="color: #991b1b; font-weight: 600;">${value:,.2f}M</span>
                </div>
                """, unsafe_allow_html=True)
            
            if len(outliers) > 5:
                st.info(f"... and {len(outliers) - 5} more outliers")
    
    return insights

def calculate_health_score(df):
    """Calculate comprehensive corporate health score"""
    
    # Component 1: Profitability (30% weight)
    avg_margin = ((df['Revenue_USD_M'] - df['Operating_Cost_USD_M']) / df['Revenue_USD_M']).mean()
    profitability_score = min(100, max(0, (avg_margin * 100 + 50)))
    
    # Component 2: Liquidity (25% weight)
    avg_cash_ratio = (df['Cash_Balance_USD_M'] / df['Debt_Outstanding_USD_M']).mean()
    liquidity_score = min(100, avg_cash_ratio * 50)
    
    # Component 3: Stability (25% weight)
    fcf_volatility = df['Free_Cash_Flow_USD_M'].std() / df['Free_Cash_Flow_USD_M'].mean()
    stability_score = min(100, max(0, 100 - (fcf_volatility * 20)))
    
    # Component 4: Growth (20% weight)
    revenue_growth = (df['Revenue_USD_M'].iloc[-1] - df['Revenue_USD_M'].iloc[0]) / df['Revenue_USD_M'].iloc[0] * 100
    growth_score = min(100, max(0, revenue_growth + 50))
    
    # Weighted average
    health_score = (
        profitability_score * 0.30 +
        liquidity_score * 0.25 +
        stability_score * 0.25 +
        growth_score * 0.20
    )
    
    return round(health_score, 1)

def display_insights(insights):
    """Display key insights from analysis"""
    
    st.markdown("### 💡 Key Financial Insights")
    
    # Create insight cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #3b82f610 0%, #2563eb10 100%);
                    padding: 1.5rem; border-radius: 12px; border-left: 4px solid #3b82f6;
                    margin-bottom: 1rem;">
            <h4 style="color: #1e293b; margin-bottom: 0.5rem;">📈 Performance Insights</h4>
        """, unsafe_allow_html=True)
        
        if 'key_metrics' in insights:
            km = insights['key_metrics']
            
            if 'revenue_growth' in km:
                if km['revenue_growth'] > 10:
                    st.markdown(f"✅ **Strong revenue growth** of {km['revenue_growth']:.1f}% over period")
                elif km['revenue_growth'] > 0:
                    st.markdown(f"📈 **Moderate revenue growth** of {km['revenue_growth']:.1f}%")
                else:
                    st.markdown(f"⚠️ **Revenue decline** of {abs(km['revenue_growth']):.1f}%")
            
            if 'negative_fcf_months' in km:
                neg_pct = (km['negative_fcf_months'] / insights['data_shape'][0]) * 100
                if neg_pct > 30:
                    st.markdown(f"⚠️ **High frequency** of negative cash flow ({neg_pct:.0f}% of months)")
                elif neg_pct > 15:
                    st.markdown(f"📊 **Moderate negative cash flow** ({neg_pct:.0f}% of months)")
                else:
                    st.markdown(f"✅ **Strong cash flow discipline** (only {neg_pct:.0f}% negative months)")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #8b5cf610 0%, #7c3aed10 100%);
                    padding: 1.5rem; border-radius: 12px; border-left: 4px solid #8b5cf6;">
            <h4 style="color: #1e293b; margin-bottom: 0.5rem;">🔄 Correlation Insights</h4>
        """, unsafe_allow_html=True)
        
        if 'correlations' in insights:
            corr = insights['correlations']
            
            if corr.get('top_positive'):
                for item in corr['top_positive'][:2]:
                    st.markdown(f"✅ **Strong positive**: {item['Variable 1']} ↔ {item['Variable 2']} ({item['Correlation']:.2f})")
            
            if corr.get('top_negative'):
                for item in corr['top_negative'][:2]:
                    st.markdown(f"🔄 **Strong negative**: {item['Variable 1']} ↔ {item['Variable 2']} ({item['Correlation']:.2f})")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f59e0b10 0%, #d9770610 100%);
                    padding: 1.5rem; border-radius: 12px; border-left: 4px solid #f59e0b;
                    margin-bottom: 1rem;">
            <h4 style="color: #1e293b; margin-bottom: 0.5rem;">⚠️ Risk Indicators</h4>
        """, unsafe_allow_html=True)
        
        if 'key_metrics' in insights:
            km = insights['key_metrics']
            
            if 'cash_to_debt_ratio' in km:
                if km['cash_to_debt_ratio'] < 30:
                    st.markdown(f"⚠️ **Low liquidity**: Cash/Debt ratio at {km['cash_to_debt_ratio']:.1f}%")
                elif km['cash_to_debt_ratio'] < 50:
                    st.markdown(f"📊 **Adequate liquidity**: Cash/Debt ratio at {km['cash_to_debt_ratio']:.1f}%")
                else:
                    st.markdown(f"✅ **Strong liquidity**: Cash/Debt ratio at {km['cash_to_debt_ratio']:.1f}%")
            
            if 'real_interest_rate' in km:
                if km['real_interest_rate'] > 3:
                    st.markdown(f"⚠️ **High real interest rate** ({km['real_interest_rate']:.1f}%) - expensive borrowing")
                elif km['real_interest_rate'] < 0:
                    st.markdown(f"📊 **Negative real rate** ({km['real_interest_rate']:.1f}%) - favorable for debt")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #10b98110 0%, #05966910 100%);
                    padding: 1.5rem; border-radius: 12px; border-left: 4px solid #10b981;">
            <h4 style="color: #1e293b; margin-bottom: 0.5rem;">📋 Data Quality</h4>
        """, unsafe_allow_html=True)
        
        if 'data_shape' in insights:
            st.markdown(f"📊 **Analysis period**: {insights['date_range']['start'].strftime('%b %Y')} to {insights['date_range']['end'].strftime('%b %Y')}")
            st.markdown(f"📈 **Data points**: {insights['data_shape'][0]:,} monthly records")
            st.markdown(f"🔍 **Features analyzed**: {insights['data_shape'][1]:,} variables")
        
        st.markdown("</div>", unsafe_allow_html=True)

def generate_recommendations(insights):
    """Generate strategic recommendations based on insights"""
    
    st.markdown("---")
    st.markdown("### 🎯 Strategic Recommendations")
    
    recommendations = []
    
    # Generate recommendations based on insights
    if 'key_metrics' in insights:
        km = insights['key_metrics']
        
        # Cash flow recommendations
        if 'negative_fcf_months' in km and 'data_shape' in insights:
            neg_pct = (km['negative_fcf_months'] / insights['data_shape'][0]) * 100
            if neg_pct > 25:
                recommendations.append({
                    'category': 'Cash Flow',
                    'priority': 'High',
                    'title': 'Improve Cash Flow Stability',
                    'action': 'Review operating costs and consider cost optimization programs',
                    'impact': 'Reduce negative cash flow months by 50% within 12 months'
                })
        
        # Debt management recommendations
        if 'cash_to_debt_ratio' in km:
            if km['cash_to_debt_ratio'] < 40:
                recommendations.append({
                    'category': 'Debt Management',
                    'priority': 'High',
                    'title': 'Strengthen Balance Sheet',
                    'action': 'Consider debt restructuring or refinancing at favorable rates',
                    'impact': 'Improve cash-to-debt ratio to >50% within 18 months'
                })
        
        # Growth recommendations
        if 'revenue_growth' in km:
            if km['revenue_growth'] < 5:
                recommendations.append({
                    'category': 'Growth Strategy',
                    'priority': 'Medium',
                    'title': 'Accelerate Revenue Growth',
                    'action': 'Explore new market opportunities and optimize pricing strategy',
                    'impact': 'Achieve 10%+ annual revenue growth'
                })
        
        # Interest rate risk
        if 'real_interest_rate' in km:
            if km['real_interest_rate'] > 2:
                recommendations.append({
                    'category': 'Risk Management',
                    'priority': 'Medium',
                    'title': 'Hedge Interest Rate Exposure',
                    'action': 'Consider interest rate swaps or fixed-rate debt instruments',
                    'impact': 'Reduce interest rate sensitivity by 30%'
                })
    
    # Display recommendations in columns
    if recommendations:
        for i in range(0, len(recommendations), 2):
            cols = st.columns(2)
            for j in range(2):
                if i + j < len(recommendations):
                    rec = recommendations[i + j]
                    
                    priority_color = {
                        'High': '#ef4444',
                        'Medium': '#f59e0b',
                        'Low': '#10b981'
                    }.get(rec['priority'], '#64748b')
                    
                    with cols[j]:
                        st.markdown(f"""
                        <div style="background: white; padding: 1.25rem; border-radius: 12px;
                                    border: 1px solid #e2e8f0; margin-bottom: 1rem;
                                    border-top: 4px solid {priority_color};">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 0.75rem;">
                                <span style="background: {priority_color}20; color: {priority_color};
                                           padding: 0.25rem 0.75rem; border-radius: 20px;
                                           font-size: 0.8rem; font-weight: 600;">
                                    {rec['priority']} Priority
                                </span>
                                <span style="color: #64748b; font-size: 0.85rem;">
                                    {rec['category']}
                                </span>
                            </div>
                            <h5 style="color: #1e293b; margin-bottom: 0.5rem;">{rec['title']}</h5>
                            <p style="color: #475569; font-size: 0.95rem; margin-bottom: 0.75rem;">
                                {rec['action']}
                            </p>
                            <div style="background: #f8fafc; padding: 0.75rem; border-radius: 8px;
                                        border-left: 3px solid #3b82f6;">
                                <span style="color: #475569; font-size: 0.85rem;">
                                    <strong>Expected Impact:</strong> {rec['impact']}
                                </span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
    else:
        st.info("💡 No specific recommendations generated. Your financial metrics appear healthy!")

# Test function for standalone testing
if __name__ == "__main__":
    print("Exploratory Data Analysis module ready for integration")