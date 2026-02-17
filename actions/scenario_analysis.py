"""
Scenario Analysis Module for Corporate Cash Flow Stress Testing Platform
Professional Macroeconomic Shock Modeling with Sensitivity Analysis
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize
from scipy.interpolate import griddata
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import streamlit as st
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def analyze_scenarios(df):
    """
    Main scenario analysis function for macroeconomic stress testing
    
    Args:
        df (pandas.DataFrame): Cleaned dataset from clean_data module
    
    Returns:
        dict: Scenario analysis results with sensitivities and what-if projections
    """
    
    if df is None:
        st.error("❌ No data provided for scenario analysis")
        return None
    
    # Professional header
    st.markdown("""
    <div style="background: linear-gradient(135deg, #dc262615 0%, #b91c1c15 100%);
                padding: 2rem; border-radius: 12px; margin-bottom: 2rem;
                border: 1px solid #e2e8f0;">
        <h3 style="color: #1e293b; margin-bottom: 1rem; display: flex; align-items: center;">
            <span style="background: #dc2626; color: white; padding: 0.5rem 1rem;
                        border-radius: 8px; margin-right: 1rem; font-size: 1.2rem;">
                ⚡
            </span>
            Macroeconomic Scenario Analyzer
        </h3>
        <p style="color: #475569; line-height: 1.6; margin-bottom: 0;">
            Advanced what-if analysis for corporate cash flow under various economic conditions.
            Model interest rate shocks, inflation impacts, FX volatility, and create custom 
            scenarios with real-time sensitivity analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Calculate baseline metrics
    baseline = calculate_baseline_metrics(df)
    
    # Create main scenario tabs
    tab_preset, tab_custom, tab_sensitivity, tab_break_even, tab_comparison = st.tabs([
        "📋 Preset Scenarios",
        "🔧 Custom Scenario Builder",
        "📊 Sensitivity Analysis",
        "⚖️ Break-Even Analysis",
        "📈 Scenario Comparison"
    ])
    
    with tab_preset:
        preset_scenarios = run_preset_scenarios(df, baseline)
    
    with tab_custom:
        custom_results = run_custom_scenario(df, baseline)
    
    with tab_sensitivity:
        sensitivity_results = run_sensitivity_analysis(df, baseline)
    
    with tab_break_even:
        break_even_results = run_break_even_analysis(df, baseline)
    
    with tab_comparison:
        display_scenario_comparison(df, baseline, preset_scenarios if 'preset_scenarios' in locals() else None, 
                                   custom_results if 'custom_results' in locals() else None)
    
    return {
        'baseline': baseline,
        'preset_scenarios': preset_scenarios if 'preset_scenarios' in locals() else None,
        'sensitivity': sensitivity_results if 'sensitivity_results' in locals() else None,
        'break_even': break_even_results if 'break_even_results' in locals() else None
    }

def calculate_baseline_metrics(df):
    """Calculate baseline financial metrics"""
    
    baseline = {
        'revenue_mean': df['Revenue_USD_M'].mean(),
        'revenue_std': df['Revenue_USD_M'].std(),
        'revenue_last': df['Revenue_USD_M'].iloc[-1],
        'cost_mean': df['Operating_Cost_USD_M'].mean(),
        'cost_std': df['Operating_Cost_USD_M'].std(),
        'cost_last': df['Operating_Cost_USD_M'].iloc[-1],
        'capex_mean': df['Capital_Expenditure_USD_M'].mean(),
        'capex_last': df['Capital_Expenditure_USD_M'].iloc[-1],
        'fcf_mean': df['Free_Cash_Flow_USD_M'].mean(),
        'fcf_last': df['Free_Cash_Flow_USD_M'].iloc[-1],
        'cash_last': df['Cash_Balance_USD_M'].iloc[-1],
        'debt_last': df['Debt_Outstanding_USD_M'].iloc[-1],
        'interest_rate_mean': df['Interest_Rate_%'].mean(),
        'inflation_mean': df['Inflation_Rate_%'].mean(),
        'fx_impact_mean': df['FX_Impact_%'].mean()
    }
    
    # Calculate growth rates
    baseline['revenue_growth'] = (df['Revenue_USD_M'].iloc[-1] / df['Revenue_USD_M'].iloc[0] - 1) * 100
    baseline['cost_growth'] = (df['Operating_Cost_USD_M'].iloc[-1] / df['Operating_Cost_USD_M'].iloc[0] - 1) * 100
    baseline['fcf_growth'] = (df['Free_Cash_Flow_USD_M'].iloc[-1] / df['Free_Cash_Flow_USD_M'].iloc[0] - 1) * 100
    
    # Calculate margins
    baseline['operating_margin'] = ((df['Revenue_USD_M'] - df['Operating_Cost_USD_M']) / df['Revenue_USD_M'] * 100).mean()
    baseline['fcf_margin'] = (df['Free_Cash_Flow_USD_M'] / df['Revenue_USD_M'] * 100).mean()
    
    return baseline

def run_preset_scenarios(df, baseline):
    """Run predefined macroeconomic scenarios"""
    
    st.markdown("### 📋 Preset Macroeconomic Scenarios")
    
    st.markdown("""
    <div style="background: #f8fafc; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
        <p style="color: #475569; margin: 0;">
            Select from industry-standard stress scenarios to test your company's resilience 
            against various economic conditions.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Scenario definitions
    scenarios = {
        "Baseline": {
            "interest_rate_change": 0,
            "inflation_change": 0,
            "revenue_change": 0,
            "cost_change": 0,
            "fx_impact": 0,
            "description": "Current economic conditions",
            "color": "#1e293b"
        },
        "Mild Recession": {
            "interest_rate_change": -0.5,
            "inflation_change": -1.0,
            "revenue_change": -5,
            "cost_change": -2,
            "fx_impact": 2,
            "description": "Moderate economic slowdown with reduced consumer spending",
            "color": "#f59e0b"
        },
        "Severe Recession": {
            "interest_rate_change": -1.5,
            "inflation_change": -2.0,
            "revenue_change": -15,
            "cost_change": -5,
            "fx_impact": 5,
            "description": "Deep recession with significant demand destruction",
            "color": "#ef4444"
        },
        "High Inflation": {
            "interest_rate_change": 2.0,
            "inflation_change": 3.0,
            "revenue_change": 8,
            "cost_change": 12,
            "fx_impact": -3,
            "description": "Stagflation scenario with rising prices and costs",
            "color": "#dc2626"
        },
        "Rapid Growth": {
            "interest_rate_change": 1.0,
            "inflation_change": 1.5,
            "revenue_change": 20,
            "cost_change": 15,
            "fx_impact": -1,
            "description": "Strong economic expansion with increased demand",
            "color": "#10b981"
        },
        "Currency Crisis": {
            "interest_rate_change": 3.0,
            "inflation_change": 2.5,
            "revenue_change": -10,
            "cost_change": 5,
            "fx_impact": 15,
            "description": "Sharp currency depreciation impacting imports and debt",
            "color": "#7e22ce"
        },
        "Interest Rate Shock": {
            "interest_rate_change": 2.5,
            "inflation_change": 0.5,
            "revenue_change": -3,
            "cost_change": 2,
            "fx_impact": 8,
            "description": "Sudden monetary policy tightening",
            "color": "#b91c1c"
        },
        "Supply Chain Disruption": {
            "interest_rate_change": 0.5,
            "inflation_change": 2.0,
            "revenue_change": -8,
            "cost_change": 15,
            "fx_impact": 5,
            "description": "Cost-push inflation from supply constraints",
            "color": "#c2410c"
        },
        "Tech Boom": {
            "interest_rate_change": 0.0,
            "inflation_change": 1.0,
            "revenue_change": 25,
            "cost_change": 18,
            "fx_impact": -2,
            "description": "Sector-specific growth with margin expansion",
            "color": "#2563eb"
        }
    }
    
    # Scenario selection
    selected_scenarios = st.multiselect(
        "Select scenarios to analyze",
        options=list(scenarios.keys()),
        default=["Mild Recession", "High Inflation", "Rapid Growth"],
        help="Choose one or more scenarios to compare"
    )
    
    if not selected_scenarios:
        st.info("Please select at least one scenario to analyze")
        return None
    
    # Forecast horizon
    forecast_months = st.slider(
        "Forecast Horizon (months)",
        min_value=3,
        max_value=36,
        value=12,
        step=3,
        help="Number of months to project under each scenario"
    )
    
    results = {}
    
    # Run scenarios
    for scenario_name in selected_scenarios:
        scenario = scenarios[scenario_name]
        
        with st.status(f"📊 Running {scenario_name} scenario...", expanded=False) as status:
            
            # Project cash flows under scenario
            projected_fcf = project_cash_flows(df, scenario, forecast_months)
            
            # Calculate scenario metrics
            metrics = calculate_scenario_metrics(projected_fcf, baseline, scenario)
            
            results[scenario_name] = {
                'name': scenario_name,
                'params': scenario,
                'projections': projected_fcf,
                'metrics': metrics,
                'color': scenario['color']
            }
            
            status.update(label=f"✅ {scenario_name} scenario completed", state="complete")
    
    # Display results
    st.markdown("### 📊 Scenario Results")
    
    # Create comparison table
    comparison_data = []
    for scenario_name, result in results.items():
        comparison_data.append({
            'Scenario': scenario_name,
            'Avg Monthly FCF': f"${result['metrics']['avg_monthly_fcf']:.1f}M",
            'Terminal Cash': f"${result['metrics']['terminal_cash']:.1f}M",
            'Survival Probability': f"{result['metrics']['survival_probability']:.1%}",
            'FCF Change': f"{result['metrics']['fcf_change']:+.1f}%",
            'Risk Level': result['metrics']['risk_level']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    # Visualization
    st.markdown("#### 📈 Cash Flow Projections")
    
    fig = go.Figure()
    
    # Add baseline projection
    baseline_projection = project_cash_flows(df, scenarios["Baseline"], forecast_months)
    fig.add_trace(go.Scatter(
        x=baseline_projection['months'],
        y=baseline_projection['cumulative_cash'],
        name="Baseline",
        line=dict(color='#94a3b8', width=2, dash='dash'),
        mode='lines'
    ))
    
    # Add scenario projections
    for scenario_name, result in results.items():
        fig.add_trace(go.Scatter(
            x=result['projections']['months'],
            y=result['projections']['cumulative_cash'],
            name=scenario_name,
            line=dict(color=result['color'], width=3),
            mode='lines+markers',
            marker=dict(size=6)
        ))
    
    fig.update_layout(
        title="Cumulative Cash Balance Under Different Scenarios",
        xaxis_title="Month",
        yaxis_title="Cumulative Cash (USD Millions)",
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
    
    # Risk assessment
    st.markdown("#### ⚠️ Risk Assessment Matrix")
    
    risk_data = []
    for scenario_name, result in results.items():
        risk_data.append({
            'Scenario': scenario_name,
            'Probability Score': f"{result['metrics']['risk_score']:.1f}",
            'Impact Severity': result['metrics']['impact_severity'],
            'Mitigation Priority': result['metrics']['mitigation_priority'],
            'Action Required': result['metrics']['action_required']
        })
    
    risk_df = pd.DataFrame(risk_data)
    st.dataframe(risk_df, use_container_width=True, hide_index=True)
    
    return results

def run_custom_scenario(df, baseline):
    """Run custom user-defined scenario"""
    
    st.markdown("### 🔧 Custom Scenario Builder")
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #2563eb10 0%, #7c3aed10 100%);
                padding: 1.5rem; border-radius: 8px; margin-bottom: 1.5rem;">
        <p style="color: #475569; margin: 0;">
            Create your own custom economic scenario by adjusting the parameters below.
            See real-time impact on cash flow projections.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Economic Parameters**")
        
        interest_rate_change = st.slider(
            "Interest Rate Change (pp)",
            min_value=-5.0,
            max_value=5.0,
            value=0.0,
            step=0.25,
            help="Change in interest rate percentage points",
            key="custom_ir"
        )
        
        inflation_change = st.slider(
            "Inflation Rate Change (pp)",
            min_value=-5.0,
            max_value=5.0,
            value=0.0,
            step=0.25,
            help="Change in inflation rate percentage points",
            key="custom_inf"
        )
        
        fx_impact = st.slider(
            "FX Impact (%)",
            min_value=-20.0,
            max_value=20.0,
            value=0.0,
            step=1.0,
            help="Foreign exchange impact on cash flows",
            key="custom_fx"
        )
    
    with col2:
        st.markdown("**📈 Business Parameters**")
        
        revenue_change = st.slider(
            "Revenue Change (%)",
            min_value=-30.0,
            max_value=30.0,
            value=0.0,
            step=2.0,
            help="Percentage change in revenue",
            key="custom_rev"
        )
        
        cost_change = st.slider(
            "Operating Cost Change (%)",
            min_value=-30.0,
            max_value=30.0,
            value=0.0,
            step=2.0,
            help="Percentage change in operating costs",
            key="custom_cost"
        )
        
        capex_change = st.slider(
            "Capex Change (%)",
            min_value=-50.0,
            max_value=50.0,
            value=0.0,
            step=5.0,
            help="Percentage change in capital expenditure",
            key="custom_capex"
        )
    
    # Forecast parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        forecast_months = st.number_input(
            "Forecast Months",
            min_value=3,
            max_value=60,
            value=12,
            step=3,
            key="custom_months"
        )
    
    with col2:
        confidence_level = st.select_slider(
            "Confidence Level",
            options=[0.80, 0.85, 0.90, 0.95],
            value=0.90,
            key="custom_conf"
        )
    
    with col3:
        shock_timing = st.selectbox(
            "Shock Timing",
            ["Immediate", "Gradual (3 months)", "Gradual (6 months)", "Delayed (6 months)"],
            index=0,
            key="custom_timing"
        )
    
    # Create custom scenario
    custom_scenario = {
        "interest_rate_change": interest_rate_change,
        "inflation_change": inflation_change,
        "revenue_change": revenue_change,
        "cost_change": cost_change,
        "capex_change": capex_change,
        "fx_impact": fx_impact,
        "description": "Custom user-defined scenario",
        "color": "#7e22ce"
    }
    
    if st.button("🚀 Run Custom Scenario", type="primary", use_container_width=True):
        with st.spinner("Running custom scenario analysis..."):
            
            # Apply timing logic
            if shock_timing == "Immediate":
                pass  # Already handled in projection function
            elif shock_timing == "Gradual (3 months)":
                custom_scenario['gradual_months'] = 3
            elif shock_timing == "Gradual (6 months)":
                custom_scenario['gradual_months'] = 6
            elif shock_timing == "Delayed (6 months)":
                custom_scenario['delay_months'] = 6
            
            # Project cash flows
            projected_fcf = project_cash_flows(
                df, custom_scenario, forecast_months,
                confidence_level=confidence_level
            )
            
            # Calculate metrics
            metrics = calculate_scenario_metrics(projected_fcf, baseline, custom_scenario)
            
            # Display results
            st.markdown("### 📊 Custom Scenario Results")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Avg Monthly FCF",
                    f"${metrics['avg_monthly_fcf']:.1f}M",
                    delta=f"{metrics['fcf_change']:+.1f}%"
                )
            
            with col2:
                st.metric(
                    "Terminal Cash",
                    f"${metrics['terminal_cash']:.1f}M",
                    delta=f"${metrics['terminal_cash'] - baseline['cash_last']:.1f}M"
                )
            
            with col3:
                st.metric(
                    "Survival Probability",
                    f"{metrics['survival_probability']:.1%}",
                    delta=None
                )
            
            with col4:
                st.metric(
                    "Risk Level",
                    metrics['risk_level'],
                    delta=None
                )
            
            # Cash flow chart
            fig = go.Figure()
            
            # Add confidence bands
            fig.add_trace(go.Scatter(
                x=list(projected_fcf['months']) + list(projected_fcf['months'][::-1]),
                y=list(projected_fcf['cumulative_cash_upper']) + list(projected_fcf['cumulative_cash_lower'][::-1]),
                fill='toself',
                fillcolor='rgba(126, 34, 206, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name=f'{int(confidence_level*100)}% CI'
            ))
            
            # Add median projection
            fig.add_trace(go.Scatter(
                x=projected_fcf['months'],
                y=projected_fcf['cumulative_cash'],
                name='Median Projection',
                line=dict(color='#7e22ce', width=3),
                mode='lines+markers',
                marker=dict(size=6)
            ))
            
            # Add baseline
            baseline_projection = project_cash_flows(df, {"interest_rate_change": 0, "inflation_change": 0,
                                                          "revenue_change": 0, "cost_change": 0,
                                                          "fx_impact": 0}, forecast_months)
            
            fig.add_trace(go.Scatter(
                x=baseline_projection['months'],
                y=baseline_projection['cumulative_cash'],
                name='Baseline',
                line=dict(color='#94a3b8', width=2, dash='dash')
            ))
            
            fig.update_layout(
                title="Custom Scenario Cash Flow Projection",
                xaxis_title="Month",
                yaxis_title="Cumulative Cash (USD Millions)",
                height=500,
                hovermode='x unified',
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            
            fig.update_xaxes(gridcolor='#e2e8f0')
            fig.update_yaxes(gridcolor='#e2e8f0')
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed metrics
            with st.expander("📋 Detailed Scenario Metrics", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    metrics_detail = {
                        'Metric': [
                            'Initial Cash',
                            'Projected Terminal Cash',
                            'Total FCF Generated',
                            'Avg Monthly Burn Rate',
                            'Peak Cash Deficit',
                            'Months to Recovery'
                        ],
                        'Value': [
                            f"${baseline['cash_last']:.1f}M",
                            f"${metrics['terminal_cash']:.1f}M",
                            f"${metrics['total_fcf_generated']:.1f}M",
                            f"${metrics['avg_burn_rate']:.1f}M",
                            f"${metrics['peak_deficit']:.1f}M" if metrics['peak_deficit'] < 0 else "No deficit",
                            f"{metrics['months_to_recovery']}" if metrics['months_to_recovery'] else "N/A"
                        ]
                    }
                    
                    st.dataframe(pd.DataFrame(metrics_detail), use_container_width=True, hide_index=True)
                
                with col2:
                    probability_detail = {
                        'Metric': [
                            'Probability > $0',
                            'Probability > $50M',
                            'Probability > $100M',
                            'Probability of Default',
                            'Expected Shortfall',
                            'Tail Risk Index'
                        ],
                        'Value': [
                            f"{metrics['survival_probability']:.1%}",
                            f"{metrics['prob_above_50']:.1%}",
                            f"{metrics['prob_above_100']:.1%}",
                            f"{metrics['prob_default']:.2%}",
                            f"${metrics['expected_shortfall']:.1f}M",
                            f"{metrics['tail_risk_index']:.2f}"
                        ]
                    }
                    
                    st.dataframe(pd.DataFrame(probability_detail), use_container_width=True, hide_index=True)
            
            return {
                'scenario': custom_scenario,
                'projections': projected_fcf,
                'metrics': metrics
            }
    
    return None

def run_sensitivity_analysis(df, baseline):
    """Run sensitivity analysis on key parameters"""
    
    st.markdown("### 📊 Sensitivity Analysis")
    
    st.markdown("""
    <div style="background: #f8fafc; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
        <p style="color: #475569; margin: 0;">
            Identify which variables have the most significant impact on your cash flow.
            The tornado chart shows sensitivity of terminal cash to each parameter.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Parameters to analyze
    parameters = {
        'Revenue Change (%)': {'range': (-30, 30), 'base': 0, 'unit': '%'},
        'Cost Change (%)': {'range': (-30, 30), 'base': 0, 'unit': '%'},
        'Interest Rate Change (pp)': {'range': (-5, 5), 'base': 0, 'unit': 'pp'},
        'Inflation Change (pp)': {'range': (-5, 5), 'base': 0, 'unit': 'pp'},
        'FX Impact (%)': {'range': (-20, 20), 'base': 0, 'unit': '%'},
        'Capex Change (%)': {'range': (-50, 50), 'base': 0, 'unit': '%'}
    }
    
    # Analysis parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        forecast_months = st.number_input(
            "Forecast Months",
            min_value=3,
            max_value=36,
            value=12,
            step=3,
            key="sens_months"
        )
    
    with col2:
        steps = st.number_input(
            "Number of Steps",
            min_value=5,
            max_value=21,
            value=11,
            step=2,
            help="Number of increments for sensitivity testing"
        )
    
    with col3:
        metric = st.selectbox(
            "Target Metric",
            ["Terminal Cash", "Avg Monthly FCF", "Survival Probability"],
            index=0,
            key="sens_metric"
        )
    
    if st.button("🔍 Run Sensitivity Analysis", type="primary", use_container_width=True):
        with st.spinner("Calculating sensitivities..."):
            
            sensitivity_results = {}
            
            # Create baseline scenario
            baseline_scenario = {
                "interest_rate_change": 0,
                "inflation_change": 0,
                "revenue_change": 0,
                "cost_change": 0,
                "capex_change": 0,
                "fx_impact": 0
            }
            
            baseline_projection = project_cash_flows(df, baseline_scenario, forecast_months)
            
            if metric == "Terminal Cash":
                baseline_value = baseline_projection['cumulative_cash'][-1]  # Fixed: removed .iloc
            elif metric == "Avg Monthly FCF":
                baseline_value = np.mean(baseline_projection['fcf'])
            else:  # Survival Probability
                baseline_value = np.mean(baseline_projection['cumulative_cash'] > 0)
            
            # Analyze each parameter
            for param_name, param_info in parameters.items():
                param_values = np.linspace(param_info['range'][0], param_info['range'][1], steps)
                metric_values = []
                
                for value in param_values:
                    # Map parameter name to scenario key
                    if param_name == 'Revenue Change (%)':
                        scenario = baseline_scenario.copy()
                        scenario['revenue_change'] = value
                    elif param_name == 'Cost Change (%)':
                        scenario = baseline_scenario.copy()
                        scenario['cost_change'] = value
                    elif param_name == 'Interest Rate Change (pp)':
                        scenario = baseline_scenario.copy()
                        scenario['interest_rate_change'] = value
                    elif param_name == 'Inflation Change (pp)':
                        scenario = baseline_scenario.copy()
                        scenario['inflation_change'] = value
                    elif param_name == 'FX Impact (%)':
                        scenario = baseline_scenario.copy()
                        scenario['fx_impact'] = value
                    elif param_name == 'Capex Change (%)':
                        scenario = baseline_scenario.copy()
                        scenario['capex_change'] = value
                    
                    projection = project_cash_flows(df, scenario, forecast_months)
                    
                    if metric == "Terminal Cash":
                        metric_value = projection['cumulative_cash'][-1]  # Fixed: removed .iloc
                    elif metric == "Avg Monthly FCF":
                        metric_value = np.mean(projection['fcf'])
                    else:
                        metric_value = np.mean(projection['cumulative_cash'] > 0)
                    
                    metric_values.append(metric_value)
                
                sensitivity_results[param_name] = {
                    'values': param_values,
                    'metric_values': metric_values,
                    'impact': (max(metric_values) - min(metric_values)) / baseline_value * 100 if baseline_value != 0 else 0
                }
            
            # Create tornado chart
            st.markdown("#### 🌪️ Tornado Chart - Sensitivity Analysis")
            
            # Sort by impact
            sorted_params = sorted(sensitivity_results.items(), 
                                  key=lambda x: x[1]['impact'], 
                                  reverse=True)
            
            fig = go.Figure()
            
            y_positions = list(range(len(sorted_params)))
            param_names = [p[0] for p in sorted_params]
            impacts = [p[1]['impact'] for p in sorted_params]
            colors = ['#ef4444' if impact > 20 else '#f59e0b' if impact > 10 else '#3b82f6' 
                     for impact in impacts]
            
            fig.add_trace(go.Bar(
                y=param_names,
                x=impacts,
                orientation='h',
                marker_color=colors,
                text=[f"{impact:.1f}%" for impact in impacts],
                textposition='outside',
                name='Impact on ' + metric
            ))
            
            fig.update_layout(
                title=f"Sensitivity of {metric} to Parameter Changes",
                xaxis_title="Impact on Metric (%)",
                yaxis_title="Parameter",
                height=400,
                plot_bgcolor='white',
                paper_bgcolor='white',
                margin=dict(l=200)
            )
            
            fig.update_xaxes(gridcolor='#e2e8f0')
            fig.update_yaxes(gridcolor='#e2e8f0')
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Sensitivity curves
            st.markdown("#### 📈 Sensitivity Curves")
            
            # Create subplots
            n_params = len(sensitivity_results)
            n_cols = 2
            n_rows = (n_params + n_cols - 1) // n_cols
            
            fig = make_subplots(
                rows=n_rows, cols=n_cols,
                subplot_titles=list(sensitivity_results.keys()),
                vertical_spacing=0.15,
                horizontal_spacing=0.15
            )
            
            row, col = 1, 1
            for param_name, result in sensitivity_results.items():
                fig.add_trace(
                    go.Scatter(
                        x=result['values'],
                        y=result['metric_values'],
                        mode='lines+markers',
                        name=param_name,
                        line=dict(color='#3b82f6', width=2),
                        marker=dict(size=6),
                        showlegend=False
                    ),
                    row=row, col=col
                )
                
                # Add baseline reference line
                fig.add_hline(
                    y=baseline_value,
                    line_dash="dash",
                    line_color="#94a3b8",
                    row=row, col=col
                )
                
                col += 1
                if col > n_cols:
                    col = 1
                    row += 1
            
            fig.update_layout(
                height=300 * n_rows,
                showlegend=False,
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            
            fig.update_xaxes(gridcolor='#e2e8f0')
            fig.update_yaxes(gridcolor='#e2e8f0')
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Key findings
            st.markdown("#### 🔍 Key Sensitivity Findings")
            
            most_sensitive = sorted_params[0]
            least_sensitive = sorted_params[-1]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div style="background: #fee2e2; padding: 1rem; border-radius: 8px;
                            border-left: 4px solid #ef4444;">
                    <h5 style="color: #991b1b; margin-bottom: 0.5rem;">⚠️ Most Sensitive Parameter</h5>
                    <p style="color: #7f1d1d; margin: 0;">
                        <strong>{most_sensitive[0]}</strong> has the largest impact ({most_sensitive[1]['impact']:.1f}%) 
                        on {metric}. Small changes in this parameter significantly affect outcomes.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div style="background: #dcfce7; padding: 1rem; border-radius: 8px;
                            border-left: 4px solid #10b981;">
                    <h5 style="color: #166534; margin-bottom: 0.5rem;">✅ Least Sensitive Parameter</h5>
                    <p style="color: #166534; margin: 0;">
                        <strong>{least_sensitive[0]}</strong> has the smallest impact ({least_sensitive[1]['impact']:.1f}%) 
                        on {metric}. The model is relatively robust to changes in this parameter.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            return sensitivity_results
    
    return None

def run_break_even_analysis(df, baseline):
    """Run break-even analysis"""
    
    st.markdown("### ⚖️ Break-Even Analysis")
    
    st.markdown("""
    <div style="background: #f8fafc; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
        <p style="color: #475569; margin: 0;">
            Determine the critical thresholds where your cash flow turns negative or 
            where specific targets are achieved.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Break-even parameters
    col1, col2 = st.columns(2)
    
    with col1:
        target_cash = st.number_input(
            "Target Cash Balance (USD M)",
            min_value=0.0,
            value=float(baseline['cash_last']),
            step=50.0,
            help="Cash balance target to achieve",
            key="be_target"
        )
        
        analysis_months = st.slider(
            "Analysis Period (months)",
            min_value=6,
            max_value=60,
            value=24,
            step=6,
            key="be_months"
        )
    
    with col2:
        revenue_range = st.slider(
            "Revenue Change Range (%)",
            min_value=-50,
            max_value=50,
            value=(-30, 30),
            step=5,
            key="be_rev_range"
        )
        
        cost_range = st.slider(
            "Cost Change Range (%)",
            min_value=-50,
            max_value=50,
            value=(-30, 30),
            step=5,
            key="be_cost_range"
        )
    
    if st.button("📊 Calculate Break-Even Points", type="primary", use_container_width=True):
        with st.spinner("Calculating break-even thresholds..."):
            
            # Create mesh for surface plot
            revenue_changes = np.linspace(revenue_range[0], revenue_range[1], 20)
            cost_changes = np.linspace(cost_range[0], cost_range[1], 20)
            
            X, Y = np.meshgrid(revenue_changes, cost_changes)
            Z = np.zeros_like(X)
            
            # Calculate terminal cash for each combination
            for i in range(len(revenue_changes)):
                for j in range(len(cost_changes)):
                    scenario = {
                        "interest_rate_change": 0,
                        "inflation_change": 0,
                        "revenue_change": revenue_changes[i],
                        "cost_change": cost_changes[j],
                        "capex_change": 0,
                        "fx_impact": 0
                    }
                    
                    projection = project_cash_flows(df, scenario, analysis_months)
                    Z[j, i] = projection['cumulative_cash'][-1]  # Fixed: removed .iloc
            
            # Find break-even contour
            fig = go.Figure()
            
            # Add contour plot
            fig.add_trace(go.Contour(
                x=revenue_changes,
                y=cost_changes,
                z=Z,
                colorscale='RdYlGn',
                contours=dict(
                    coloring='heatmap',
                    showlabels=True,
                    labelfont=dict(size=12, color='white')
                ),
                colorbar=dict(title="Terminal Cash (M$)")
            ))
            
            # Add break-even line (Z = target_cash)
            fig.add_trace(go.Contour(
                x=revenue_changes,
                y=cost_changes,
                z=Z,
                contours=dict(
                    start=target_cash,
                    end=target_cash,
                    coloring='none',
                    showlabels=True,
                    labelfont=dict(size=14, color='red')
                ),
                line=dict(color='red', width=3),
                showscale=False,
                name=f'Target: ${target_cash}M'
            ))
            
            # Add current position
            fig.add_trace(go.Scatter(
                x=[0],
                y=[0],
                mode='markers+text',
                marker=dict(size=15, color='black', symbol='x'),
                text=['Current'],
                textposition='top center',
                name='Current Position',
                showlegend=True
            ))
            
            fig.update_layout(
                title=f"Break-Even Analysis: Terminal Cash After {analysis_months} Months",
                xaxis_title="Revenue Change (%)",
                yaxis_title="Cost Change (%)",
                height=600,
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Find break-even points
            # Create a finer grid for interpolation
            revenue_fine = np.linspace(revenue_range[0], revenue_range[1], 100)
            cost_fine = np.linspace(cost_range[0], cost_range[1], 100)
            X_fine, Y_fine = np.meshgrid(revenue_fine, cost_fine)
            
            points = np.array([(rev, cost) for rev in revenue_changes for cost in cost_changes])
            values = Z.flatten()
            
            Z_fine = griddata(points, values, (X_fine, Y_fine), method='cubic')
            
            # Find where Z_fine crosses target_cash
            break_even_points = []
            for i in range(len(revenue_fine)):
                for j in range(len(cost_fine)-1):
                    if not np.isnan(Z_fine[j, i]) and not np.isnan(Z_fine[j+1, i]):
                        if (Z_fine[j, i] - target_cash) * (Z_fine[j+1, i] - target_cash) <= 0:
                            # Linear interpolation
                            cost_break = cost_fine[j] + (target_cash - Z_fine[j, i]) * \
                                        (cost_fine[j+1] - cost_fine[j]) / (Z_fine[j+1, i] - Z_fine[j, i])
                            break_even_points.append((revenue_fine[i], cost_break))
            
            # Display break-even table
            st.markdown("#### 📊 Break-Even Thresholds")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Revenue break-even (holding costs constant)
                cost_fixed_ranges = [-20, -10, 0, 10, 20]
                rev_break_even = []
                
                for cost_fixed in cost_fixed_ranges:
                    # Find revenue where terminal cash = target_cash
                    revenue_values = revenue_changes
                    # Find closest cost index
                    cost_idx = np.argmin(np.abs(cost_changes - cost_fixed))
                    cash_values = Z[cost_idx, :]
                    
                    # Find where crosses target
                    for i in range(len(revenue_values)-1):
                        if (cash_values[i] - target_cash) * (cash_values[i+1] - target_cash) <= 0:
                            rev_break = revenue_values[i] + (target_cash - cash_values[i]) * \
                                       (revenue_values[i+1] - revenue_values[i]) / (cash_values[i+1] - cash_values[i])
                            rev_break_even.append({
                                'Fixed Cost Change': f"{cost_fixed:+.0f}%",
                                'Required Revenue Change': f"{rev_break:+.1f}%"
                            })
                            break
                
                if rev_break_even:
                    st.dataframe(pd.DataFrame(rev_break_even), use_container_width=True, hide_index=True)
            
            with col2:
                # Cost break-even (holding revenue constant)
                rev_fixed_ranges = [-20, -10, 0, 10, 20]
                cost_break_even = []
                
                for rev_fixed in rev_fixed_ranges:
                    rev_idx = np.argmin(np.abs(revenue_changes - rev_fixed))
                    cash_values = Z[:, rev_idx]
                    
                    for i in range(len(cost_changes)-1):
                        if (cash_values[i] - target_cash) * (cash_values[i+1] - target_cash) <= 0:
                            cost_break = cost_changes[i] + (target_cash - cash_values[i]) * \
                                        (cost_changes[i+1] - cost_changes[i]) / (cash_values[i+1] - cash_values[i])
                            cost_break_even.append({
                                'Fixed Revenue Change': f"{rev_fixed:+.0f}%",
                                'Required Cost Change': f"{cost_break:+.1f}%"
                            })
                            break
                
                if cost_break_even:
                    st.dataframe(pd.DataFrame(cost_break_even), use_container_width=True, hide_index=True)
            
            # Interpretation
            st.markdown("#### 💡 Break-Even Interpretation")
            
            # Find current margin
            current_margin = baseline['operating_margin']
            rev_range = revenue_range
            cost_range_val = cost_range
            
            st.markdown(f"""
            <div style="background: #f8fafc; padding: 1.5rem; border-radius: 12px;
                        border: 1px solid #e2e8f0;">
                <h5 style="color: #1e293b; margin-bottom: 1rem;">Key Insights</h5>
                <ul style="color: #475569; line-height: 1.8;">
                    <li><strong>Current Operating Margin:</strong> {current_margin:.1f}%</li>
                    <li><strong>To achieve target cash of ${target_cash:.0f}M after {analysis_months} months:</strong></li>
                    <ul>
                        <li>If costs remain constant, revenue must change by 
                            {rev_break_even[0]['Required Revenue Change'] if rev_break_even else 'N/A'}</li>
                        <li>If revenue remains constant, costs must change by 
                            {cost_break_even[0]['Required Cost Change'] if cost_break_even else 'N/A'}</li>
                    </ul>
                    <li><strong>Risk Zone:</strong> Combinations of revenue change below 
                        {rev_range[0]:.0f}% and cost change above {cost_range_val[1]:.0f}% lead to cash shortfall</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    return None

def display_scenario_comparison(df, baseline, preset_results=None, custom_results=None):
    """Display comprehensive scenario comparison"""
    
    st.markdown("### 📈 Scenario Comparison Dashboard")
    
    # Collect all scenarios
    all_scenarios = []
    
    # Add baseline
    all_scenarios.append({
        'name': 'Baseline',
        'type': 'baseline',
        'color': '#1e293b',
        'metrics': {
            'terminal_cash': baseline['cash_last'],
            'avg_monthly_fcf': baseline['fcf_mean'],
            'margin': baseline['operating_margin'],
            'survival_probability': 1.0,
            'risk_score': 0,
            'risk_level': 'Low'
        }
    })
    
    # Add preset scenarios
    if preset_results:
        for name, result in preset_results.items():
            all_scenarios.append({
                'name': name,
                'type': 'preset',
                'color': result['color'],
                'metrics': result['metrics']
            })
    
    # Add custom scenario
    if custom_results:
        all_scenarios.append({
            'name': 'Custom Scenario',
            'type': 'custom',
            'color': '#7e22ce',
            'metrics': custom_results['metrics']
        })
    
    if len(all_scenarios) <= 1:
        st.info("Run some scenarios first to see comparison")
        return
    
    # Create comparison chart
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Terminal Cash Comparison",
            "Average Monthly FCF",
            "Survival Probability",
            "Risk-Return Profile"
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "scatter"}]
        ]
    )
    
    # Terminal cash
    fig.add_trace(
        go.Bar(
            name="Terminal Cash",
            x=[s['name'] for s in all_scenarios],
            y=[s['metrics']['terminal_cash'] for s in all_scenarios],
            marker_color=[s['color'] for s in all_scenarios],
            text=[f"${v:.1f}M" for v in [s['metrics']['terminal_cash'] for s in all_scenarios]],
            textposition='outside'
        ),
        row=1, col=1
    )
    
    # Average FCF
    fig.add_trace(
        go.Bar(
            name="Avg FCF",
            x=[s['name'] for s in all_scenarios],
            y=[s['metrics']['avg_monthly_fcf'] for s in all_scenarios],
            marker_color=[s['color'] for s in all_scenarios],
            text=[f"${v:.1f}M" for v in [s['metrics']['avg_monthly_fcf'] for s in all_scenarios]],
            textposition='outside'
        ),
        row=1, col=2
    )
    
    # Survival probability
    fig.add_trace(
        go.Bar(
            name="Survival Prob",
            x=[s['name'] for s in all_scenarios],
            y=[s['metrics']['survival_probability'] * 100 for s in all_scenarios],
            marker_color=[s['color'] for s in all_scenarios],
            text=[f"{v:.0f}%" for v in [s['metrics']['survival_probability'] * 100 for s in all_scenarios]],
            textposition='outside'
        ),
        row=2, col=1
    )
    
    # Risk-return scatter
    fig.add_trace(
        go.Scatter(
            x=[s['metrics']['risk_score'] for s in all_scenarios],
            y=[s['metrics']['avg_monthly_fcf'] for s in all_scenarios],
            mode='markers+text',
            marker=dict(
                size=15,
                color=[s['color'] for s in all_scenarios],
                line=dict(width=2, color='white')
            ),
            text=[s['name'] for s in all_scenarios],
            textposition='top center',
            showlegend=False
        ),
        row=2, col=2
    )
    
    # Add risk quadrants
    fig.add_vline(x=50, line_dash="dash", line_color="#94a3b8", row=2, col=2)
    fig.add_hline(y=0, line_dash="dash", line_color="#94a3b8", row=2, col=2)
    
    fig.add_annotation(x=25, y=20, text="Low Risk", showarrow=False, row=2, col=2)
    fig.add_annotation(x=75, y=20, text="High Risk", showarrow=False, row=2, col=2)
    
    fig.update_layout(
        height=700,
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    fig.update_xaxes(gridcolor='#e2e8f0')
    fig.update_yaxes(gridcolor='#e2e8f0')
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed comparison table
    st.markdown("#### 📋 Detailed Scenario Metrics")
    
    comparison_rows = []
    for scenario in all_scenarios:
        m = scenario['metrics']
        comparison_rows.append({
            'Scenario': scenario['name'],
            'Terminal Cash (M$)': f"${m['terminal_cash']:.1f}",
            'Avg Monthly FCF (M$)': f"${m['avg_monthly_fcf']:.1f}",
            'FCF Change (%)': f"{m.get('fcf_change', 0):+.1f}",
            'Survival Prob (%)': f"{m['survival_probability']:.1%}",
            'Risk Level': m['risk_level'],
            'Risk Score': f"{m['risk_score']:.1f}"
        })
    
    comparison_df = pd.DataFrame(comparison_rows)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

def project_cash_flows(df, scenario, months, confidence_level=0.95, n_simulations=1000):
    """Project cash flows under given scenario"""
    
    # Extract parameters
    ir_change = scenario.get('interest_rate_change', 0) / 100  # Convert to decimal
    inf_change = scenario.get('inflation_change', 0) / 100
    rev_change = scenario.get('revenue_change', 0) / 100
    cost_change = scenario.get('cost_change', 0) / 100
    capex_change = scenario.get('capex_change', 0) / 100
    fx_impact = scenario.get('fx_impact', 0) / 100
    
    # Get gradual/delay parameters
    gradual_months = scenario.get('gradual_months', 1)
    delay_months = scenario.get('delay_months', 0)
    
    # Calculate baseline monthly averages
    avg_revenue = df['Revenue_USD_M'].mean()
    avg_cost = df['Operating_Cost_USD_M'].mean()
    avg_capex = df['Capital_Expenditure_USD_M'].mean()
    avg_debt = df['Debt_Outstanding_USD_M'].mean()
    baseline_interest_rate = df['Interest_Rate_%'].mean() / 100
    
    # Calculate volatilities
    rev_vol = df['Revenue_USD_M'].std() / avg_revenue if avg_revenue != 0 else 0.1
    cost_vol = df['Operating_Cost_USD_M'].std() / avg_cost if avg_cost != 0 else 0.1
    
    # Initialize arrays
    months_array = np.arange(1, months + 1)
    cash_balance = df['Cash_Balance_USD_M'].iloc[-1]
    
    # Monte Carlo for confidence intervals
    simulations = []
    for sim in range(n_simulations):
        sim_cash = cash_balance
        sim_fcf = []
        sim_cumulative = []
        
        for month in range(months):
            # Apply gradual phase-in if specified
            if month < delay_months:
                month_rev_change = 0
                month_cost_change = 0
            elif month < delay_months + gradual_months:
                progress = (month - delay_months + 1) / gradual_months
                month_rev_change = rev_change * progress
                month_cost_change = cost_change * progress
            else:
                month_rev_change = rev_change
                month_cost_change = cost_change
            
            # Add random shock
            revenue_shock = np.random.normal(0, rev_vol)
            cost_shock = np.random.normal(0, cost_vol)
            
            # Calculate monthly values with scenario adjustments
            revenue = avg_revenue * (1 + month_rev_change) * (1 + revenue_shock) * (1 + fx_impact)
            cost = avg_cost * (1 + month_cost_change) * (1 + cost_shock) * (1 + inf_change)
            capex = avg_capex * (1 + capex_change)
            
            # Interest rate impact on debt
            interest_impact = avg_debt * baseline_interest_rate * ir_change / 12
            
            # Calculate FCF
            fcf = revenue - cost - capex - interest_impact
            
            # Update cash
            sim_cash += fcf
            sim_cash = max(sim_cash, 0)  # Cash can't go negative (but we track deficits separately)
            
            sim_fcf.append(fcf)
            sim_cumulative.append(sim_cash)
        
        simulations.append(sim_cumulative)
    
    simulations = np.array(simulations)
    
    # Calculate percentiles
    lower_percentile = (1 - confidence_level) / 2 * 100
    upper_percentile = (1 - (1 - confidence_level) / 2) * 100
    
    results = {
        'months': months_array,
        'fcf': np.mean(simulations, axis=0),
        'cumulative_cash': np.mean(simulations, axis=0),
        'cumulative_cash_lower': np.percentile(simulations, lower_percentile, axis=0),
        'cumulative_cash_upper': np.percentile(simulations, upper_percentile, axis=0),
        'fcf_std': np.std(simulations, axis=0),
        'simulations': simulations
    }
    
    return results

def calculate_scenario_metrics(projections, baseline, scenario):
    """Calculate comprehensive metrics for a scenario"""
    
    metrics = {}
    
    # Basic metrics - FIXED: Use numpy indexing instead of .iloc
    metrics['avg_monthly_fcf'] = float(np.mean(projections['fcf']))
    metrics['terminal_cash'] = float(projections['cumulative_cash'][-1])  # Fixed: removed .iloc
    metrics['total_fcf_generated'] = float(projections['cumulative_cash'][-1] - baseline['cash_last'])  # Fixed: removed .iloc
    metrics['fcf_change'] = ((metrics['avg_monthly_fcf'] / baseline['fcf_mean']) - 1) * 100 if baseline['fcf_mean'] != 0 else 0
    
    # Risk metrics
    metrics['survival_probability'] = float(np.mean(projections['simulations'][:, -1] > 0))
    metrics['prob_above_50'] = float(np.mean(projections['simulations'][:, -1] > 50))
    metrics['prob_above_100'] = float(np.mean(projections['simulations'][:, -1] > 100))
    metrics['prob_default'] = 1 - metrics['survival_probability']
    
    # Tail risk
    negative_sims = projections['simulations'][:, -1][projections['simulations'][:, -1] < 0]
    if len(negative_sims) > 0:
        metrics['expected_shortfall'] = float(np.mean(negative_sims))
    else:
        metrics['expected_shortfall'] = 0.0
    
    # Burn rate - FIXED: Use numpy indexing
    cash_diffs = np.diff(projections['cumulative_cash'])
    negative_diffs = cash_diffs[cash_diffs < 0]
    metrics['avg_burn_rate'] = float(-np.mean(negative_diffs)) if len(negative_diffs) > 0 else 0.0
    
    # Peak deficit - FIXED: Use numpy indexing
    min_cash = float(np.min(projections['cumulative_cash']))
    metrics['peak_deficit'] = min_cash if min_cash < 0 else 0.0
    
    # Months to recovery (if deficit) - FIXED: Use numpy indexing
    if metrics['peak_deficit'] < 0:
        recovery_indices = np.where(projections['cumulative_cash'] >= 0)[0]
        if len(recovery_indices) > 0:
            metrics['months_to_recovery'] = int(recovery_indices[0] + 1)
        else:
            metrics['months_to_recovery'] = None
    else:
        metrics['months_to_recovery'] = None
    
    # Risk scoring
    survival_component = (1 - metrics['survival_probability']) * 50
    burn_component = (metrics['avg_burn_rate'] / abs(baseline['fcf_mean']) * 30) if baseline['fcf_mean'] != 0 else 0
    deficit_component = (abs(metrics['peak_deficit']) / baseline['cash_last'] * 20) if baseline['cash_last'] != 0 else 0
    
    metrics['risk_score'] = float(min(100, survival_component + burn_component + deficit_component))
    
    # Risk level classification
    if metrics['risk_score'] < 20:
        metrics['risk_level'] = "Low"
        metrics['impact_severity'] = "Minor"
        metrics['mitigation_priority'] = "Low"
        metrics['action_required'] = "Monitor"
    elif metrics['risk_score'] < 50:
        metrics['risk_level'] = "Moderate"
        metrics['impact_severity'] = "Significant"
        metrics['mitigation_priority'] = "Medium"
        metrics['action_required'] = "Prepare contingency"
    elif metrics['risk_score'] < 80:
        metrics['risk_level'] = "High"
        metrics['impact_severity'] = "Severe"
        metrics['mitigation_priority'] = "High"
        metrics['action_required'] = "Immediate action needed"
    else:
        metrics['risk_level'] = "Critical"
        metrics['impact_severity'] = "Extreme"
        metrics['mitigation_priority'] = "Critical"
        metrics['action_required'] = "Emergency measures required"
    
    # Tail risk index
    if metrics['expected_shortfall'] != 0 and metrics['avg_monthly_fcf'] != 0:
        metrics['tail_risk_index'] = float(abs(metrics['expected_shortfall'] / metrics['avg_monthly_fcf']))
    else:
        metrics['tail_risk_index'] = 0.0
    
    return metrics

# Test function for standalone testing
if __name__ == "__main__":
    print("Scenario Analysis module ready for integration")
    print("Features: Preset scenarios, custom builder, sensitivity analysis, break-even analysis")