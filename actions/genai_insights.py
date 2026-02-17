"""
GenAI Insights Module for Corporate Cash Flow Stress Testing Platform
Advanced Natural Language Generation with Executive Summaries and Strategic Recommendations
"""

import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import json
import os
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Import OpenAI and Google Gemini libraries conditionally
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    st.warning("OpenAI library not installed. Install with: pip install openai")

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    st.warning("Google GenerativeAI library not installed. Install with: pip install google-generativeai")

def generate_insights(df):
    """
    Main GenAI insights generation function
    
    Args:
        df (pandas.DataFrame): Cleaned dataset from clean_data module
    
    Returns:
        dict: Generated insights and recommendations
    """
    
    if df is None:
        st.error("❌ No data provided for insight generation")
        return None
    
    # Professional header
    st.markdown("""
    <div style="background: linear-gradient(135deg, #9333ea15 0%, #7e22ce15 100%);
                padding: 2rem; border-radius: 12px; margin-bottom: 2rem;
                border: 1px solid #e2e8f0;">
        <h3 style="color: #1e293b; margin-bottom: 1rem; display: flex; align-items: center;">
            <span style="background: #9333ea; color: white; padding: 0.5rem 1rem;
                        border-radius: 8px; margin-right: 1rem; font-size: 1.2rem;">
                💡
            </span>
            Generative AI Insights Engine
        </h3>
        <p style="color: #475569; line-height: 1.6; margin-bottom: 0;">
            Leverage state-of-the-art language models to generate executive summaries,
            strategic recommendations, risk narratives, and comprehensive reports
            based on your financial data analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check for API keys
    check_api_keys()
    
    # Prepare data summary for AI
    data_summary = prepare_data_summary(df)
    
    # Create insight tabs
    tab_summary, tab_narrative, tab_recommendations, tab_risk, tab_report = st.tabs([
        "📋 Executive Summary",
        "📝 Financial Narrative",
        "🎯 Strategic Recommendations",
        "⚠️ Risk Analysis",
        "📄 Comprehensive Report"
    ])
    
    with tab_summary:
        generate_executive_summary(data_summary)
    
    with tab_narrative:
        generate_financial_narrative(data_summary)
    
    with tab_recommendations:
        generate_strategic_recommendations(data_summary)
    
    with tab_risk:
        generate_risk_analysis(data_summary)
    
    with tab_report:
        generate_comprehensive_report(data_summary)
    
    return data_summary

def check_api_keys():
    """Check for available API keys and display status"""
    
    st.markdown("### 🔑 API Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            st.markdown("""
            <div style="background: #dcfce7; padding: 0.75rem; border-radius: 8px;
                        border-left: 4px solid #10b981;">
                <span style="color: #166534;">✅ OpenAI API: Connected</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: #fee2e2; padding: 0.75rem; border-radius: 8px;
                        border-left: 4px solid #ef4444;">
                <span style="color: #991b1b;">❌ OpenAI API: Not connected</span>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key:
            st.markdown("""
            <div style="background: #dcfce7; padding: 0.75rem; border-radius: 8px;
                        border-left: 4px solid #10b981;">
                <span style="color: #166534;">✅ Gemini API: Connected</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: #fee2e2; padding: 0.75rem; border-radius: 8px;
                        border-left: 4px solid #ef4444;">
                <span style="color: #991b1b;">❌ Gemini API: Not connected</span>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        if openai_key or gemini_key:
            st.markdown("""
            <div style="background: #fef3c7; padding: 0.75rem; border-radius: 8px;
                        border-left: 4px solid #f59e0b;">
                <span style="color: #92400e;">⚡ Ready to generate insights</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: #f1f5f9; padding: 0.75rem; border-radius: 8px;
                        border-left: 4px solid #64748b;">
                <span style="color: #334155;">ℹ️ Set API keys in PowerShell</span>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")

def prepare_data_summary(df):
    """Prepare comprehensive data summary for AI processing"""
    
    summary = {
        'metadata': {
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_period_start': df['Date'].min().strftime('%Y-%m'),
            'data_period_end': df['Date'].max().strftime('%Y-%m'),
            'total_months': len(df),
            'company': "Corporate Entity"
        },
        'financial_metrics': {
            'revenue': {
                'mean': float(df['Revenue_USD_M'].mean()),
                'median': float(df['Revenue_USD_M'].median()),
                'std': float(df['Revenue_USD_M'].std()),
                'min': float(df['Revenue_USD_M'].min()),
                'max': float(df['Revenue_USD_M'].max()),
                'trend': calculate_trend(df['Revenue_USD_M'].values),
                'growth_rate': calculate_growth_rate(df['Revenue_USD_M'].values)
            },
            'operating_cost': {
                'mean': float(df['Operating_Cost_USD_M'].mean()),
                'median': float(df['Operating_Cost_USD_M'].median()),
                'std': float(df['Operating_Cost_USD_M'].std()),
                'trend': calculate_trend(df['Operating_Cost_USD_M'].values)
            },
            'free_cash_flow': {
                'mean': float(df['Free_Cash_Flow_USD_M'].mean()),
                'median': float(df['Free_Cash_Flow_USD_M'].median()),
                'std': float(df['Free_Cash_Flow_USD_M'].std()),
                'min': float(df['Free_Cash_Flow_USD_M'].min()),
                'max': float(df['Free_Cash_Flow_USD_M'].max()),
                'negative_months': int((df['Free_Cash_Flow_USD_M'] < 0).sum()),
                'positive_months': int((df['Free_Cash_Flow_USD_M'] > 0).sum()),
                'trend': calculate_trend(df['Free_Cash_Flow_USD_M'].values)
            },
            'cash_position': {
                'current': float(df['Cash_Balance_USD_M'].iloc[-1]),
                'mean': float(df['Cash_Balance_USD_M'].mean()),
                'min': float(df['Cash_Balance_USD_M'].min()),
                'max': float(df['Cash_Balance_USD_M'].max())
            },
            'debt': {
                'current': float(df['Debt_Outstanding_USD_M'].iloc[-1]),
                'mean': float(df['Debt_Outstanding_USD_M'].mean()),
                'trend': calculate_trend(df['Debt_Outstanding_USD_M'].values)
            }
        },
        'ratios': {
            'operating_margin': float(((df['Revenue_USD_M'] - df['Operating_Cost_USD_M']) / df['Revenue_USD_M'] * 100).mean()),
            'cash_to_debt': float((df['Cash_Balance_USD_M'] / df['Debt_Outstanding_USD_M']).mean()),
            'fcf_margin': float((df['Free_Cash_Flow_USD_M'] / df['Revenue_USD_M'] * 100).mean()),
            'capex_intensity': float((df['Capital_Expenditure_USD_M'] / df['Revenue_USD_M'] * 100).mean())
        },
        'macro_indicators': {
            'avg_interest_rate': float(df['Interest_Rate_%'].mean()),
            'avg_inflation': float(df['Inflation_Rate_%'].mean()),
            'avg_fx_impact': float(df['FX_Impact_%'].mean()),
            'interest_rate_trend': calculate_trend(df['Interest_Rate_%'].values),
            'inflation_trend': calculate_trend(df['Inflation_Rate_%'].values)
        },
        'risk_metrics': {
            'cash_flow_volatility': float(df['Free_Cash_Flow_USD_M'].std() / df['Free_Cash_Flow_USD_M'].mean()),
            'negative_cash_flow_ratio': float((df['Free_Cash_Flow_USD_M'] < 0).mean()),
            'debt_to_cash': float(df['Debt_Outstanding_USD_M'].mean() / df['Cash_Balance_USD_M'].mean()),
            'interest_coverage': float(df['Free_Cash_Flow_USD_M'].mean() / (df['Debt_Outstanding_USD_M'].mean() * df['Interest_Rate_%'].mean() / 100 / 12))
        },
        'seasonality': {
            'q1_avg': float(df[df['Date'].dt.quarter == 1]['Free_Cash_Flow_USD_M'].mean()),
            'q2_avg': float(df[df['Date'].dt.quarter == 2]['Free_Cash_Flow_USD_M'].mean()),
            'q3_avg': float(df[df['Date'].dt.quarter == 3]['Free_Cash_Flow_USD_M'].mean()),
            'q4_avg': float(df[df['Date'].dt.quarter == 4]['Free_Cash_Flow_USD_M'].mean())
        },
        'key_events': identify_key_events(df)
    }
    
    return summary

def calculate_trend(values):
    """Calculate trend direction and strength"""
    if len(values) < 2:
        return {'direction': 'stable', 'strength': 0}
    
    from scipy import stats
    x = np.arange(len(values))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
    
    direction = 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable'
    strength = abs(r_value)
    
    return {
        'direction': direction,
        'strength': float(strength),
        'slope': float(slope),
        'p_value': float(p_value)
    }

def calculate_growth_rate(values):
    """Calculate compound growth rate"""
    if len(values) < 2 or values[0] == 0:
        return 0
    
    start = values[0]
    end = values[-1]
    periods = len(values) - 1
    
    if start > 0 and end > 0:
        cagr = (end / start) ** (1 / periods) - 1
        return float(cagr * 100)
    return 0

def identify_key_events(df):
    """Identify key financial events in the data"""
    events = []
    
    # Significant cash flow drops
    fcf = df['Free_Cash_Flow_USD_M'].values
    mean_fcf = np.mean(fcf)
    std_fcf = np.std(fcf)
    
    for i in range(1, len(fcf)):
        if fcf[i] < mean_fcf - 2 * std_fcf:
            events.append({
                'date': df['Date'].iloc[i].strftime('%Y-%m'),
                'type': 'severe_drop',
                'magnitude': float(mean_fcf - fcf[i]),
                'description': f"Severe cash flow drop of ${mean_fcf - fcf[i]:.1f}M"
            })
        elif fcf[i] < 0 and fcf[i-1] >= 0:
            events.append({
                'date': df['Date'].iloc[i].strftime('%Y-%m'),
                'type': 'turned_negative',
                'description': "Cash flow turned negative"
            })
    
    # Debt spikes
    debt = df['Debt_Outstanding_USD_M'].values
    mean_debt = np.mean(debt)
    
    for i in range(1, len(debt)):
        if debt[i] > mean_debt * 1.5:
            events.append({
                'date': df['Date'].iloc[i].strftime('%Y-%m'),
                'type': 'debt_spike',
                'magnitude': float(debt[i] / mean_debt),
                'description': f"Debt spike to ${debt[i]:.1f}M"
            })
    
    return events

def generate_with_openai(prompt, max_tokens=1000, temperature=0.7):
    """Generate text using OpenAI API"""
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    
    try:
        openai.api_key = api_key
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a senior financial analyst expert in corporate cash flow analysis and risk management."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"OpenAI API error: {str(e)}")
        return None

def generate_with_gemini(prompt, max_tokens=1000, temperature=0.7):
    """Generate text using Google Gemini API"""
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Gemini API error: {str(e)}")
        return None

def generate_executive_summary(data_summary):
    """Generate executive summary from data"""
    
    st.markdown("### 📋 Executive Summary")
    
    # Prepare prompt
    prompt = f"""
    As a senior financial analyst, write a concise executive summary (250-300 words) based on this corporate financial data:
    
    DATA SUMMARY:
    - Analysis Period: {data_summary['metadata']['data_period_start']} to {data_summary['metadata']['data_period_end']} ({data_summary['metadata']['total_months']} months)
    
    FINANCIAL METRICS:
    - Average Revenue: ${data_summary['financial_metrics']['revenue']['mean']:.1f}M (trend: {data_summary['financial_metrics']['revenue']['trend']['direction']})
    - Average Free Cash Flow: ${data_summary['financial_metrics']['free_cash_flow']['mean']:.1f}M
    - Current Cash Position: ${data_summary['financial_metrics']['cash_position']['current']:.1f}M
    - Current Debt: ${data_summary['financial_metrics']['debt']['current']:.1f}M
    
    RATIOS:
    - Operating Margin: {data_summary['ratios']['operating_margin']:.1f}%
    - Cash to Debt Ratio: {data_summary['ratios']['cash_to_debt']:.2f}
    
    RISK METRICS:
    - Negative Cash Flow Months: {data_summary['financial_metrics']['free_cash_flow']['negative_months']} out of {data_summary['metadata']['total_months']}
    - Cash Flow Volatility: {data_summary['risk_metrics']['cash_flow_volatility']:.2f}
    
    Please provide:
    1. Overall financial health assessment
    2. Key strengths and weaknesses
    3. Critical trends to watch
    4. High-level recommendation
    
    Format professionally for C-level executives.
    """
    
    # Generate using available API
    summary = None
    if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
        summary = generate_with_openai(prompt)
    elif GEMINI_AVAILABLE and os.getenv("GEMINI_API_KEY"):
        summary = generate_with_gemini(prompt)
    
    if summary:
        st.markdown(f"""
        <div style="background: white; padding: 2rem; border-radius: 12px;
                    border: 1px solid #e2e8f0; box-shadow: 0 4px 6px rgba(0,0,0,0.05);
                    margin-bottom: 1rem;">
            <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                <span style="background: #9333ea; color: white; padding: 0.25rem 0.75rem;
                           border-radius: 20px; font-size: 0.8rem; font-weight: 600;">
                    AI-GENERATED
                </span>
                <span style="color: #64748b; font-size: 0.8rem; margin-left: 1rem;">
                    Powered by {'OpenAI GPT-4' if OPENAI_AVAILABLE else 'Google Gemini'}
                </span>
            </div>
            <div style="color: #1e293b; line-height: 1.8; font-size: 1rem;">
                {summary.replace(chr(10), '<br>')}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Fallback template
        st.info("💡 **API keys not configured.** Here's a template executive summary based on your data:")
        
        template = f"""
        **Executive Summary**

        **Financial Health Assessment:**
        Over the analysis period ({data_summary['metadata']['data_period_start']} to {data_summary['metadata']['data_period_end']}), 
        the company demonstrated {'strong' if data_summary['financial_metrics']['free_cash_flow']['mean'] > 0 else 'challenging'} 
        cash flow performance with average monthly free cash flow of ${data_summary['financial_metrics']['free_cash_flow']['mean']:.1f}M.
        
        **Key Metrics:**
        - Revenue Trend: {data_summary['financial_metrics']['revenue']['trend']['direction'].title()} 
          (strength: {data_summary['financial_metrics']['revenue']['trend']['strength']:.2f})
        - Operating Margin: {data_summary['ratios']['operating_margin']:.1f}%
        - Cash Position: ${data_summary['financial_metrics']['cash_position']['current']:.1f}M
        - Debt Level: ${data_summary['financial_metrics']['debt']['current']:.1f}M
        
        **Critical Observations:**
        - {data_summary['financial_metrics']['free_cash_flow']['negative_months']} months of negative cash flow 
          ({data_summary['financial_metrics']['free_cash_flow']['negative_months']/data_summary['metadata']['total_months']*100:.0f}% of period)
        - Cash to debt ratio: {data_summary['ratios']['cash_to_debt']:.2f}
        - Cash flow volatility: {data_summary['risk_metrics']['cash_flow_volatility']:.2f}
        
        **Recommendation:**
        {'Focus on improving cash flow consistency and building cash reserves.' if data_summary['financial_metrics']['free_cash_flow']['negative_months'] > 3 else 
         'Leverage strong cash position for strategic investments.'}
        """
        
        st.markdown(template)
    
    # Key metrics visualization
    st.markdown("#### 📊 Key Metrics at a Glance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Avg Monthly FCF",
            f"${data_summary['financial_metrics']['free_cash_flow']['mean']:.1f}M",
            delta=f"{data_summary['financial_metrics']['free_cash_flow']['trend']['direction']}"
        )
    
    with col2:
        st.metric(
            "Operating Margin",
            f"{data_summary['ratios']['operating_margin']:.1f}%",
            delta=None
        )
    
    with col3:
        st.metric(
            "Cash Position",
            f"${data_summary['financial_metrics']['cash_position']['current']:.1f}M",
            delta=f"vs avg ${data_summary['financial_metrics']['cash_position']['mean']:.1f}M"
        )
    
    with col4:
        st.metric(
            "Risk Score",
            f"{data_summary['risk_metrics']['cash_flow_volatility']:.2f}",
            delta="Volatility"
        )

def generate_financial_narrative(data_summary):
    """Generate detailed financial narrative"""
    
    st.markdown("### 📝 Financial Performance Narrative")
    
    # Prepare prompt
    prompt = f"""
    As a financial analyst, write a detailed narrative (400-500 words) explaining the company's financial performance:
    
    REVENUE ANALYSIS:
    - Average: ${data_summary['financial_metrics']['revenue']['mean']:.1f}M
    - Range: ${data_summary['financial_metrics']['revenue']['min']:.1f}M to ${data_summary['financial_metrics']['revenue']['max']:.1f}M
    - Trend: {data_summary['financial_metrics']['revenue']['trend']['direction']} (strength: {data_summary['financial_metrics']['revenue']['trend']['strength']:.2f})
    - Growth Rate: {data_summary['financial_metrics']['revenue']['growth_rate']:.1f}%
    
    COST STRUCTURE:
    - Average Operating Cost: ${data_summary['financial_metrics']['operating_cost']['mean']:.1f}M
    - Cost Trend: {data_summary['financial_metrics']['operating_cost']['trend']['direction']}
    
    CASH FLOW DYNAMICS:
    - Average FCF: ${data_summary['financial_metrics']['free_cash_flow']['mean']:.1f}M
    - Positive Months: {data_summary['financial_metrics']['free_cash_flow']['positive_months']}
    - Negative Months: {data_summary['financial_metrics']['free_cash_flow']['negative_months']}
    - FCF Trend: {data_summary['financial_metrics']['free_cash_flow']['trend']['direction']}
    
    SEASONALITY:
    - Q1 Avg: ${data_summary['seasonality']['q1_avg']:.1f}M
    - Q2 Avg: ${data_summary['seasonality']['q2_avg']:.1f}M
    - Q3 Avg: ${data_summary['seasonality']['q3_avg']:.1f}M
    - Q4 Avg: ${data_summary['seasonality']['q4_avg']:.1f}M
    
    KEY EVENTS:
    {json.dumps(data_summary['key_events'][:3], indent=2)}
    
    Please provide:
    1. Revenue and cost analysis with drivers
    2. Cash flow patterns and volatility
    3. Seasonal trends and their impact
    4. Key events and their effects
    5. Forward-looking implications
    
    Write in professional, analytical style suitable for investor relations.
    """
    
    # Generate using available API
    narrative = None
    if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
        narrative = generate_with_openai(prompt, max_tokens=1500)
    elif GEMINI_AVAILABLE and os.getenv("GEMINI_API_KEY"):
        narrative = generate_with_gemini(prompt, max_tokens=1500)
    
    if narrative:
        st.markdown(f"""
        <div style="background: white; padding: 2rem; border-radius: 12px;
                    border: 1px solid #e2e8f0; margin-bottom: 1rem;">
            <div style="color: #1e293b; line-height: 1.8;">
                {narrative.replace(chr(10), '<br>')}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Fallback visualization
        st.info("Configure API keys for AI-generated narrative. Showing data visualization instead.")
        
        # Create revenue vs cost chart
        fig = go.Figure()
        
        # Add revenue and cost comparison
        fig.add_trace(go.Bar(
            name='Revenue',
            x=['Mean', 'Min', 'Max'],
            y=[data_summary['financial_metrics']['revenue']['mean'],
               data_summary['financial_metrics']['revenue']['min'],
               data_summary['financial_metrics']['revenue']['max']],
            marker_color='#3b82f6'
        ))
        
        fig.add_trace(go.Bar(
            name='Operating Cost',
            x=['Mean', 'Min', 'Max'],
            y=[data_summary['financial_metrics']['operating_cost']['mean'],
               data_summary['financial_metrics']['revenue']['min'] * 0.6,  # Approximate
               data_summary['financial_metrics']['revenue']['max'] * 0.6],
            marker_color='#ef4444'
        ))
        
        fig.update_layout(
            title="Revenue vs Operating Cost Analysis",
            barmode='group',
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        st.plotly_chart(fig, use_container_width=True)

def generate_strategic_recommendations(data_summary):
    """Generate strategic recommendations"""
    
    st.markdown("### 🎯 Strategic Recommendations")
    
    # Prepare prompt
    prompt = f"""
    As a strategic financial advisor, provide actionable recommendations (300-400 words) based on this analysis:
    
    CURRENT POSITION:
    - Cash Position: ${data_summary['financial_metrics']['cash_position']['current']:.1f}M
    - Debt Level: ${data_summary['financial_metrics']['debt']['current']:.1f}M
    - Operating Margin: {data_summary['ratios']['operating_margin']:.1f}%
    
    RISK INDICATORS:
    - Cash Flow Volatility: {data_summary['risk_metrics']['cash_flow_volatility']:.2f}
    - Negative Cash Flow Ratio: {data_summary['risk_metrics']['negative_cash_flow_ratio']:.1%}
    - Interest Coverage: {data_summary['risk_metrics']['interest_coverage']:.2f}
    
    MACRO CONTEXT:
    - Average Interest Rate: {data_summary['macro_indicators']['avg_interest_rate']:.2f}%
    - Average Inflation: {data_summary['macro_indicators']['avg_inflation']:.2f}%
    - Interest Rate Trend: {data_summary['macro_indicators']['interest_rate_trend']['direction']}
    
    Please provide recommendations in these areas:
    1. Cash Flow Optimization (3-4 specific actions)
    2. Debt Management (2-3 strategies)
    3. Growth Initiatives (2-3 opportunities)
    4. Risk Mitigation (2-3 measures)
    5. Capital Allocation Priorities
    
    Prioritize recommendations based on impact and urgency.
    Include estimated timelines and expected outcomes.
    """
    
    # Generate using available API
    recommendations = None
    if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
        recommendations = generate_with_openai(prompt, max_tokens=1500)
    elif GEMINI_AVAILABLE and os.getenv("GEMINI_API_KEY"):
        recommendations = generate_with_gemini(prompt, max_tokens=1500)
    
    if recommendations:
        st.markdown(f"""
        <div style="background: white; padding: 2rem; border-radius: 12px;
                    border: 1px solid #e2e8f0; margin-bottom: 1rem;">
            <div style="color: #1e293b; line-height: 1.8;">
                {recommendations.replace(chr(10), '<br>')}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Generate rule-based recommendations
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style="background: #f8fafc; padding: 1.5rem; border-radius: 12px;
                        border: 1px solid #e2e8f0; margin-bottom: 1rem;">
                <h4 style="color: #1e293b; margin-bottom: 1rem;">📈 Cash Flow Optimization</h4>
            """, unsafe_allow_html=True)
            
            if data_summary['financial_metrics']['free_cash_flow']['negative_months'] > 3:
                st.markdown("""
                - **Immediate**: Review operating costs for 10-15% reduction opportunities
                - **Short-term**: Implement stricter receivables management (target DSO reduction)
                - **Medium-term**: Renegotiate supplier payment terms
                """)
            else:
                st.markdown("""
                - **Short-term**: Explore early payment discounts from suppliers
                - **Medium-term**: Optimize working capital through inventory management
                - **Long-term**: Evaluate capex timing for cash flow smoothing
                """)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("""
            <div style="background: #f8fafc; padding: 1.5rem; border-radius: 12px;
                        border: 1px solid #e2e8f0;">
                <h4 style="color: #1e293b; margin-bottom: 1rem;">📊 Debt Management</h4>
            """, unsafe_allow_html=True)
            
            if data_summary['ratios']['cash_to_debt'] < 0.5:
                st.markdown("""
                - **Immediate**: Prioritize debt reduction using excess cash
                - **Short-term**: Consider debt refinancing at lower rates
                - **Medium-term**: Establish revolving credit facility for flexibility
                """)
            else:
                st.markdown("""
                - **Short-term**: Maintain current debt levels for liquidity
                - **Medium-term**: Explore strategic acquisitions using debt capacity
                - **Long-term**: Optimize capital structure for cost of capital
                """)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background: #f8fafc; padding: 1.5rem; border-radius: 12px;
                        border: 1px solid #e2e8f0; margin-bottom: 1rem;">
                <h4 style="color: #1e293b; margin-bottom: 1rem;">🚀 Growth Initiatives</h4>
            """, unsafe_allow_html=True)
            
            if data_summary['financial_metrics']['revenue']['growth_rate'] > 5:
                st.markdown("""
                - **Short-term**: Accelerate investment in high-growth segments
                - **Medium-term**: Expand into adjacent markets
                - **Long-term**: Consider strategic M&A opportunities
                """)
            else:
                st.markdown("""
                - **Immediate**: Review pricing strategy and value proposition
                - **Short-term**: Intensify sales and marketing efforts
                - **Medium-term**: Develop new product/service offerings
                """)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("""
            <div style="background: #f8fafc; padding: 1.5rem; border-radius: 12px;
                        border: 1px solid #e2e8f0;">
                <h4 style="color: #1e293b; margin-bottom: 1rem;">🛡️ Risk Mitigation</h4>
            """, unsafe_allow_html=True)
            
            if data_summary['risk_metrics']['cash_flow_volatility'] > 0.3:
                st.markdown("""
                - **Immediate**: Establish cash flow forecasting system
                - **Short-term**: Diversify revenue streams
                - **Medium-term**: Implement hedging for interest rate exposure
                """)
            else:
                st.markdown("""
                - **Short-term**: Review insurance coverage adequacy
                - **Medium-term**: Develop business continuity plans
                - **Long-term**: Stress test business model regularly
                """)
            
            st.markdown("</div>", unsafe_allow_html=True)

def generate_risk_analysis(data_summary):
    """Generate comprehensive risk analysis"""
    
    st.markdown("### ⚠️ Risk Analysis and Mitigation")
    
    # Calculate risk scores
    risk_scores = calculate_risk_scores(data_summary)
    
    # Display risk dashboard
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div style="background: white; padding: 1.5rem; border-radius: 12px;
                    border: 1px solid #e2e8f0; text-align: center;">
            <div style="color: #64748b; font-size: 0.9rem; margin-bottom: 0.5rem;">Overall Risk Score</div>
            <div style="font-size: 2.5rem; font-weight: 700; color: {get_risk_color(risk_scores['overall'])};">
                {risk_scores['overall']:.0f}
            </div>
            <div style="color: {get_risk_color(risk_scores['overall'])}; font-weight: 600;">
                {get_risk_level(risk_scores['overall'])}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background: white; padding: 1.5rem; border-radius: 12px;
                    border: 1px solid #e2e8f0;">
            <div style="color: #64748b; font-size: 0.9rem; margin-bottom: 0.5rem;">Liquidity Risk</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: {get_risk_color(risk_scores['liquidity'])};">
                {risk_scores['liquidity']:.0f}
            </div>
            <div style="margin-top: 0.5rem; font-size: 0.9rem;">
                Cash/Debt: {data_summary['ratios']['cash_to_debt']:.2f}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style="background: white; padding: 1.5rem; border-radius: 12px;
                    border: 1px solid #e2e8f0;">
            <div style="color: #64748b; font-size: 0.9rem; margin-bottom: 0.5rem;">Volatility Risk</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: {get_risk_color(risk_scores['volatility'])};">
                {risk_scores['volatility']:.0f}
            </div>
            <div style="margin-top: 0.5rem; font-size: 0.9rem;">
                CV: {data_summary['risk_metrics']['cash_flow_volatility']:.2f}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Risk radar chart
    st.markdown("#### 📊 Risk Profile Radar")
    
    categories = ['Liquidity', 'Volatility', 'Leverage', 'Profitability', 'Macro', 'Seasonality']
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=[
            risk_scores['liquidity'],
            risk_scores['volatility'],
            risk_scores['leverage'],
            risk_scores['profitability'],
            risk_scores['macro'],
            risk_scores['seasonality']
        ],
        theta=categories,
        fill='toself',
        name='Current Risk Profile',
        line_color='#ef4444'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=[50, 50, 50, 50, 50, 50],
        theta=categories,
        fill='toself',
        name='Threshold',
        line_color='#94a3b8',
        opacity=0.3
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        height=400,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Prepare AI prompt for risk narrative
    prompt = f"""
    As a risk management expert, provide a detailed risk analysis (300-400 words) based on these risk scores:
    
    RISK SCORES (0-100, higher = more risk):
    - Overall Risk: {risk_scores['overall']:.0f}
    - Liquidity Risk: {risk_scores['liquidity']:.0f} (Cash/Debt: {data_summary['ratios']['cash_to_debt']:.2f})
    - Volatility Risk: {risk_scores['volatility']:.0f} (CV: {data_summary['risk_metrics']['cash_flow_volatility']:.2f})
    - Leverage Risk: {risk_scores['leverage']:.0f} (Debt/Cash: {1/data_summary['ratios']['cash_to_debt']:.2f})
    - Profitability Risk: {risk_scores['profitability']:.0f} (Margin: {data_summary['ratios']['operating_margin']:.1f}%)
    - Macro Risk: {risk_scores['macro']:.0f} (Interest Rate: {data_summary['macro_indicators']['avg_interest_rate']:.2f}%)
    - Seasonality Risk: {risk_scores['seasonality']:.0f}
    
    KEY EVENTS:
    {json.dumps(data_summary['key_events'][:3], indent=2)}
    
    Please provide:
    1. Executive summary of key risks
    2. Detailed analysis of top 3 risks
    3. Early warning indicators to monitor
    4. Specific mitigation strategies for each major risk
    5. Risk tolerance recommendations
    
    Write in professional risk management style.
    """
    
    # Generate risk narrative
    risk_narrative = None
    if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
        risk_narrative = generate_with_openai(prompt, max_tokens=1500)
    elif GEMINI_AVAILABLE and os.getenv("GEMINI_API_KEY"):
        risk_narrative = generate_with_gemini(prompt, max_tokens=1500)
    
    if risk_narrative:
        st.markdown(f"""
        <div style="background: white; padding: 2rem; border-radius: 12px;
                    border: 1px solid #e2e8f0; margin-top: 1rem;">
            <div style="color: #1e293b; line-height: 1.8;">
                {risk_narrative.replace(chr(10), '<br>')}
            </div>
        </div>
        """, unsafe_allow_html=True)

def calculate_risk_scores(data_summary):
    """Calculate comprehensive risk scores"""
    
    scores = {}
    
    # Liquidity risk (0-100)
    cash_debt = data_summary['ratios']['cash_to_debt']
    if cash_debt >= 1:
        scores['liquidity'] = 20
    elif cash_debt >= 0.5:
        scores['liquidity'] = 40
    elif cash_debt >= 0.25:
        scores['liquidity'] = 60
    elif cash_debt >= 0.1:
        scores['liquidity'] = 80
    else:
        scores['liquidity'] = 95
    
    # Volatility risk
    cv = data_summary['risk_metrics']['cash_flow_volatility']
    scores['volatility'] = min(95, cv * 100)
    
    # Leverage risk
    debt_to_cash = 1 / max(data_summary['ratios']['cash_to_debt'], 0.01)
    scores['leverage'] = min(95, debt_to_cash * 20)
    
    # Profitability risk
    margin = data_summary['ratios']['operating_margin']
    if margin >= 20:
        scores['profitability'] = 20
    elif margin >= 10:
        scores['profitability'] = 40
    elif margin >= 5:
        scores['profitability'] = 60
    elif margin >= 0:
        scores['profitability'] = 80
    else:
        scores['profitability'] = 95
    
    # Macro risk
    interest_rate = data_summary['macro_indicators']['avg_interest_rate']
    inflation = data_summary['macro_indicators']['avg_inflation']
    scores['macro'] = min(95, (interest_rate + inflation) * 10)
    
    # Seasonality risk
    q_vals = [data_summary['seasonality'][f'q{i}_avg'] for i in range(1, 5)]
    seasonality_cv = np.std(q_vals) / max(np.mean(np.abs(q_vals)), 0.01)
    scores['seasonality'] = min(80, seasonality_cv * 100)
    
    # Overall risk (weighted average)
    weights = {
        'liquidity': 0.25,
        'volatility': 0.20,
        'leverage': 0.20,
        'profitability': 0.15,
        'macro': 0.10,
        'seasonality': 0.10
    }
    
    scores['overall'] = sum(scores[k] * weights[k] for k in weights.keys())
    
    return scores

def get_risk_color(score):
    """Get color based on risk score"""
    if score < 30:
        return '#10b981'  # Green - Low risk
    elif score < 50:
        return '#3b82f6'  # Blue - Moderate risk
    elif score < 70:
        return '#f59e0b'  # Orange - High risk
    else:
        return '#ef4444'  # Red - Critical risk

def get_risk_level(score):
    """Get risk level description"""
    if score < 30:
        return 'LOW RISK'
    elif score < 50:
        return 'MODERATE RISK'
    elif score < 70:
        return 'HIGH RISK'
    else:
        return 'CRITICAL RISK'

def generate_comprehensive_report(data_summary):
    """Generate comprehensive financial report"""
    
    st.markdown("### 📄 Comprehensive Financial Report")
    
    st.markdown("""
    <div style="background: #f8fafc; padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem;">
        <p style="color: #475569; margin: 0;">
            Generate a complete financial report combining all insights, narratives, and recommendations
            in a professional format suitable for board presentations and investor communications.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Report configuration
    col1, col2 = st.columns(2)
    
    with col1:
        report_type = st.selectbox(
            "Report Type",
            ["Board Presentation", "Investor Update", "Management Review", "Regulatory Filing"],
            index=0
        )
        
        include_sections = st.multiselect(
            "Include Sections",
            ["Executive Summary", "Financial Analysis", "Risk Assessment", 
             "Strategic Recommendations", "Forecasts", "Appendices"],
            default=["Executive Summary", "Financial Analysis", "Risk Assessment", "Strategic Recommendations"]
        )
    
    with col2:
        report_length = st.select_slider(
            "Report Length",
            options=["Brief", "Standard", "Detailed", "Comprehensive"],
            value="Standard"
        )
        
        include_visuals = st.checkbox("Include Visualizations", value=True)
        include_data_tables = st.checkbox("Include Data Tables", value=True)
    
    if st.button("📄 Generate Report", type="primary", use_container_width=True):
        with st.spinner("Generating comprehensive report..."):
            
            # Prepare prompt
            prompt = f"""
            Generate a comprehensive {report_type} report with the following specifications:
            
            Report Type: {report_type}
            Length: {report_length}
            Sections to include: {', '.join(include_sections)}
            Include visuals: {'Yes' if include_visuals else 'No'}
            
            Based on this financial data:
            {json.dumps(data_summary, indent=2, default=str)[:2000]}...
            
            The report should be professionally formatted with:
            1. Executive summary highlighting key findings
            2. Detailed financial analysis with trends
            3. Risk assessment with mitigation strategies
            4. Strategic recommendations prioritized by impact
            5. Forward-looking statements and guidance
            6. Professional language suitable for C-level executives
            
            Format with clear section headings and bullet points where appropriate.
            """
            
            # Generate report
            report = None
            if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
                report = generate_with_openai(prompt, max_tokens=2500, temperature=0.5)
            elif GEMINI_AVAILABLE and os.getenv("GEMINI_API_KEY"):
                report = generate_with_gemini(prompt, max_tokens=2500, temperature=0.5)
            
            if report:
                st.markdown(f"""
                <div style="background: white; padding: 2rem; border-radius: 12px;
                            border: 1px solid #e2e8f0; box-shadow: 0 4px 6px rgba(0,0,0,0.05);
                            margin-bottom: 1rem;">
                    <div style="border-bottom: 2px solid #9333ea; padding-bottom: 1rem; margin-bottom: 1.5rem;">
                        <h2 style="color: #1e293b; margin: 0;">{report_type}</h2>
                        <p style="color: #64748b; margin: 0.5rem 0 0 0;">
                            Generated on {datetime.now().strftime('%B %d, %Y')}
                        </p>
                    </div>
                    <div style="color: #1e293b; line-height: 1.8;">
                        {report.replace(chr(10), '<br>')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Download options
                col1, col2, col3 = st.columns(3)
                
                with col2:
                    report_text = f"""
                    {report_type}
                    Generated on {datetime.now().strftime('%B %d, %Y')}
                    
                    {report}
                    """
                    
                    st.download_button(
                        label="📥 Download Report",
                        data=report_text,
                        file_name=f"financial_report_{datetime.now().strftime('%Y%m%d')}.txt",
                        mime="text/plain",
                        type="primary",
                        use_container_width=True
                    )
            else:
                st.warning("Unable to generate AI report. Configure API keys for full functionality.")
                
                # Show template
                st.markdown("""
                <div style="background: white; padding: 2rem; border-radius: 12px;
                            border: 1px solid #e2e8f0;">
                    <h3 style="color: #1e293b;">Financial Report Template</h3>
                    <p style="color: #64748b;">Configure OpenAI or Gemini API keys to generate comprehensive reports.</p>
                </div>
                """, unsafe_allow_html=True)

# Test function for standalone testing
if __name__ == "__main__":
    print("GenAI Insights module ready for integration")
    print("Features: Executive summaries, financial narratives, strategic recommendations, risk analysis, comprehensive reports")