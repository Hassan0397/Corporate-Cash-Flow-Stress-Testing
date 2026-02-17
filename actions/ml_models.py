"""
Machine Learning Models Module for Corporate Cash Flow Stress Testing Platform
Advanced Predictive Analytics with Multiple ML Algorithms
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    IsolationForest
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.svm import SVR, SVC
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import streamlit as st
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def train_and_predict(df):
    """
    Main machine learning function for training and prediction
    
    Args:
        df (pandas.DataFrame): Cleaned dataset from clean_data module
    
    Returns:
        dict: ML model results with predictions and performance metrics
    """
    
    if df is None:
        st.error("❌ No data provided for ML training")
        return None
    
    # Professional header
    st.markdown("""
    <div style="background: linear-gradient(135deg, #05966915 0%, #04785715 100%);
                padding: 2rem; border-radius: 12px; margin-bottom: 2rem;
                border: 1px solid #e2e8f0;">
        <h3 style="color: #1e293b; margin-bottom: 1rem; display: flex; align-items: center;">
            <span style="background: #059669; color: white; padding: 0.5rem 1rem;
                        border-radius: 8px; margin-right: 1rem; font-size: 1.2rem;">
                🤖
            </span>
            Advanced Machine Learning Engine
        </h3>
        <p style="color: #475569; line-height: 1.6; margin-bottom: 0;">
            Leverage state-of-the-art ML algorithms for predictive analytics, anomaly detection,
            and classification. Compare multiple models, optimize hyperparameters, and gain
            actionable insights from your financial data.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Prepare features
    df_ml = prepare_ml_features(df)
    
    # ML task selection
    st.markdown("### 🎯 ML Task Selection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        task_type = st.selectbox(
            "Select ML Task",
            [
                "Regression - Cash Flow Prediction",
                "Classification - Default Risk",
                "Anomaly Detection",
                "Feature Importance Analysis",
                "Time Series Forecasting (ML)"
            ],
            help="Choose the type of machine learning task"
        )
    
    with col2:
        target_variable = st.selectbox(
            "Select Target Variable",
            ["Free_Cash_Flow_USD_M", "Revenue_USD_M", "Operating_Cost_USD_M", 
             "Cash_Balance_USD_M", "Debt_Outstanding_USD_M"],
            index=0,
            help="Variable to predict"
        )
    
    # Model selection based on task
    if "Regression" in task_type:
        results = run_regression_models(df_ml, target_variable)
    elif "Classification" in task_type:
        results = run_classification_models(df_ml)
    elif "Anomaly" in task_type:
        results = run_anomaly_detection(df_ml)
    elif "Feature Importance" in task_type:
        results = run_feature_importance(df_ml, target_variable)
    else:  # Time Series ML
        results = run_time_series_ml(df_ml, target_variable)
    
    return results

def prepare_ml_features(df):
    """Prepare feature set for machine learning"""
    
    df_ml = df.copy()
    
    # Create lagged features
    for lag in [1, 2, 3, 6]:
        df_ml[f'FCF_Lag_{lag}'] = df_ml['Free_Cash_Flow_USD_M'].shift(lag)
        df_ml[f'Revenue_Lag_{lag}'] = df_ml['Revenue_USD_M'].shift(lag)
    
    # Create rolling statistics
    for window in [3, 6]:
        df_ml[f'FCF_Rolling_Mean_{window}'] = df_ml['Free_Cash_Flow_USD_M'].rolling(window=window).mean()
        df_ml[f'FCF_Rolling_Std_{window}'] = df_ml['Free_Cash_Flow_USD_M'].rolling(window=window).std()
    
    # Create ratio features
    df_ml['Op_Margin'] = (df_ml['Revenue_USD_M'] - df_ml['Operating_Cost_USD_M']) / df_ml['Revenue_USD_M']
    df_ml['Cash_to_Debt'] = df_ml['Cash_Balance_USD_M'] / df_ml['Debt_Outstanding_USD_M']
    df_ml['Capex_Intensity'] = df_ml['Capital_Expenditure_USD_M'] / df_ml['Revenue_USD_M']
    
    # Create interaction features
    df_ml['IR_x_Debt'] = df_ml['Interest_Rate_%'] * df_ml['Debt_Outstanding_USD_M']
    df_ml['Inflation_x_Cost'] = df_ml['Inflation_Rate_%'] * df_ml['Operating_Cost_USD_M']
    
    # Create target for classification (default risk)
    df_ml['Default_Risk'] = (df_ml['Free_Cash_Flow_USD_M'] < 0).astype(int)
    
    # Drop NaN values
    df_ml = df_ml.dropna()
    
    return df_ml

def run_regression_models(df, target):
    """Run regression models for continuous prediction"""
    
    st.markdown("### 📈 Regression Models for Cash Flow Prediction")
    
    # Feature and target selection
    feature_cols = [col for col in df.columns if col not in 
                   ['Date', target, 'Default_Risk', 'YearMonth']]
    
    X = df[feature_cols]
    y = df[target]
    
    # Model configuration
    col1, col2, col3 = st.columns(3)
    
    with col1:
        test_size = st.slider("Test Size (%)", 10, 40, 20, 5) / 100
        cv_folds = st.selectbox("Cross-Validation Folds", [3, 5, 10], index=1)
    
    with col2:
        models_to_run = st.multiselect(
            "Select Models",
            ["Linear Regression", "Ridge", "Lasso", "Random Forest", 
             "Gradient Boosting", "XGBoost", "LightGBM", "SVR"],
            default=["Random Forest", "XGBoost", "LightGBM"]
        )
    
    with col3:
        optimize_hyperparams = st.checkbox("Optimize Hyperparameters", value=True)
        show_feature_importance = st.checkbox("Show Feature Importance", value=True)
    
    if st.button("🚀 Train Regression Models", type="primary", use_container_width=True):
        with st.spinner("Training regression models..."):
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, shuffle=False
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            results = {}
            predictions = {}
            
            # Define models
            models = {
                "Linear Regression": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "Random Forest": RandomForestRegressor(random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                "XGBoost": xgb.XGBRegressor(random_state=42),
                "LightGBM": lgb.LGBMRegressor(random_state=42, verbose=-1),
                "SVR": SVR()
            }
            
            # Hyperparameter grids
            param_grids = {
                "Ridge": {'alpha': [0.1, 1.0, 10.0]},
                "Lasso": {'alpha': [0.001, 0.01, 0.1]},
                "Random Forest": {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, None]
                },
                "Gradient Boosting": {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5]
                },
                "XGBoost": {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5]
                },
                "LightGBM": {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.01, 0.1],
                    'num_leaves': [31, 50]
                }
            }
            
            # Train each selected model
            progress_bar = st.progress(0)
            for i, model_name in enumerate(models_to_run):
                if model_name in models:
                    
                    if optimize_hyperparams and model_name in param_grids:
                        # Grid search for hyperparameter optimization
                        grid_search = GridSearchCV(
                            models[model_name],
                            param_grids[model_name],
                            cv=min(cv_folds, 3),
                            scoring='r2',
                            n_jobs=-1
                        )
                        
                        if model_name in ["SVR"]:
                            grid_search.fit(X_train_scaled, y_train)
                            model = grid_search.best_estimator_
                            y_pred = model.predict(X_test_scaled)
                        else:
                            grid_search.fit(X_train, y_train)
                            model = grid_search.best_estimator_
                            y_pred = model.predict(X_test)
                        
                        st.info(f"{model_name} best params: {grid_search.best_params_}")
                    else:
                        # Train without optimization
                        if model_name in ["SVR"]:
                            models[model_name].fit(X_train_scaled, y_train)
                            y_pred = models[model_name].predict(X_test_scaled)
                        else:
                            models[model_name].fit(X_train, y_train)
                            y_pred = models[model_name].predict(X_test)
                        
                        model = models[model_name]
                    
                    # Calculate metrics
                    mae = mean_absolute_error(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
                    r2 = r2_score(y_test, y_pred)
                    
                    # Cross-validation score
                    if model_name in ["SVR"]:
                        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv_folds, scoring='r2')
                    else:
                        cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='r2')
                    
                    results[model_name] = {
                        'MAE': mae,
                        'RMSE': rmse,
                        'MAPE': mape,
                        'R2': r2,
                        'CV_Mean': cv_scores.mean(),
                        'CV_Std': cv_scores.std(),
                        'Model': model
                    }
                    
                    predictions[model_name] = y_pred
                
                progress_bar.progress((i + 1) / len(models_to_run))
            
            # Display results
            st.markdown("### 📊 Model Performance Comparison")
            
            # Create results dataframe
            results_df = pd.DataFrame(results).T
            results_df = results_df[['R2', 'MAE', 'RMSE', 'MAPE', 'CV_Mean', 'CV_Std']]
            results_df = results_df.round(3)
            
            # Style the dataframe
            styled_df = results_df.style.background_gradient(
                subset=['R2', 'CV_Mean'], 
                cmap='RdYlGn'
            ).background_gradient(
                subset=['MAE', 'RMSE', 'MAPE'], 
                cmap='RdYlGn_r'
            )
            
            st.dataframe(styled_df, use_container_width=True)
            
            # Identify best model
            best_model = results_df['R2'].idxmax()
            best_r2 = results_df.loc[best_model, 'R2']
            
            st.markdown(f"""
            <div style="background: #dcfce7; padding: 1rem; border-radius: 8px;
                        border-left: 4px solid #10b981; margin: 1rem 0;">
                <h5 style="color: #166534; margin-bottom: 0.25rem;">🏆 Best Model: {best_model}</h5>
                <p style="color: #166534; margin: 0;">
                    R² Score: {best_r2:.3f} | MAE: ${results_df.loc[best_model, 'MAE']:.2f}M
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Actual vs Predicted plot
            st.markdown("#### 📈 Actual vs Predicted - Best Model")
            
            fig = go.Figure()
            
            # Scatter plot
            fig.add_trace(go.Scatter(
                x=y_test,
                y=predictions[best_model],
                mode='markers',
                name='Predictions',
                marker=dict(
                    size=10,
                    color='#059669',
                    opacity=0.6,
                    line=dict(width=1, color='white')
                )
            ))
            
            # Perfect prediction line
            min_val = min(y_test.min(), predictions[best_model].min())
            max_val = max(y_test.max(), predictions[best_model].max())
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='#ef4444', dash='dash')
            ))
            
            fig.update_layout(
                title=f"{best_model}: Actual vs Predicted {target}",
                xaxis_title=f"Actual {target} (USD M)",
                yaxis_title=f"Predicted {target} (USD M)",
                height=500,
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            
            fig.update_xaxes(gridcolor='#e2e8f0')
            fig.update_yaxes(gridcolor='#e2e8f0')
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance for tree-based models
            if show_feature_importance and best_model in ["Random Forest", "Gradient Boosting", "XGBoost", "LightGBM"]:
                st.markdown("#### 🔍 Feature Importance Analysis")
                
                model = results[best_model]['Model']
                importance = model.feature_importances_
                
                # Create dataframe
                importance_df = pd.DataFrame({
                    'Feature': feature_cols,
                    'Importance': importance
                }).sort_values('Importance', ascending=True)
                
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    y=importance_df['Feature'],
                    x=importance_df['Importance'],
                    orientation='h',
                    marker_color='#059669',
                    text=importance_df['Importance'].round(3),
                    textposition='outside'
                ))
                
                fig.update_layout(
                    title="Feature Importance",
                    xaxis_title="Importance Score",
                    yaxis_title="Feature",
                    height=500,
                    plot_bgcolor='white',
                    paper_bgcolor='white'
                )
                
                fig.update_xaxes(gridcolor='#e2e8f0')
                fig.update_yaxes(gridcolor='#e2e8f0')
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Top features insight
                top_features = importance_df.nlargest(3, 'Importance')
                st.markdown(f"""
                <div style="background: #f8fafc; padding: 1rem; border-radius: 8px;
                            border: 1px solid #e2e8f0;">
                    <h5 style="color: #1e293b; margin-bottom: 0.5rem;">💡 Key Drivers</h5>
                    <p style="color: #475569;">
                        The most important features for predicting {target} are:
                        <strong>{', '.join(top_features['Feature'].values)}</strong>.
                        These account for {top_features['Importance'].sum()*100:.1f}% of predictive power.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            return results
    
    return None

def run_classification_models(df):
    """Run classification models for default risk prediction"""
    
    st.markdown("### ⚠️ Default Risk Classification")
    
    # Prepare classification target
    df['Risk_Class'] = pd.cut(
        df['Free_Cash_Flow_USD_M'],
        bins=[-np.inf, -50, -10, 0, 50, np.inf],
        labels=['Critical', 'High', 'Moderate', 'Low', 'Minimal']
    )
    
    # Feature selection
    feature_cols = [col for col in df.columns if col not in 
                   ['Date', 'Free_Cash_Flow_USD_M', 'Default_Risk', 'Risk_Class', 'YearMonth']]
    
    X = df[feature_cols]
    y = df['Default_Risk']  # Binary classification: 1 = negative cash flow
    
    # Display class distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Class Distribution**")
        class_dist = y.value_counts().reset_index()
        class_dist.columns = ['Default Risk', 'Count']
        class_dist['Default Risk'] = class_dist['Default Risk'].map({0: 'No Risk', 1: 'At Risk'})
        st.dataframe(class_dist, use_container_width=True, hide_index=True)
    
    with col2:
        fig = go.Figure(data=[go.Pie(
            labels=['No Risk', 'At Risk'],
            values=y.value_counts().values,
            hole=.4,
            marker_colors=['#10b981', '#ef4444']
        )])
        fig.update_layout(height=200, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Model configuration
    col1, col2 = st.columns(2)
    
    with col1:
        test_size = st.slider("Test Size (%)", 10, 40, 20, 5) / 100
        models_to_run = st.multiselect(
            "Select Models",
            ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM", "SVC"],
            default=["Random Forest", "XGBoost", "LightGBM"]
        )
    
    with col2:
        scale_features = st.checkbox("Scale Features", value=True)
        show_confusion_matrix = st.checkbox("Show Confusion Matrix", value=True)
    
    if st.button("🚀 Train Classification Models", type="primary", use_container_width=True):
        with st.spinner("Training classification models..."):
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            # Scale features if requested
            if scale_features:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
            
            results = {}
            
            # Define models
            models = {
                "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
                "Random Forest": RandomForestClassifier(random_state=42),
                "XGBoost": xgb.XGBClassifier(random_state=42),
                "LightGBM": lgb.LGBMClassifier(random_state=42, verbose=-1),
                "SVC": SVC(probability=True, random_state=42)
            }
            
            # Train each selected model
            for model_name in models_to_run:
                if model_name in models:
                    model = models[model_name]
                    model.fit(X_train, y_train)
                    
                    # Predictions
                    y_pred = model.predict(X_test)
                    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
                    
                    # Calculate metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred)
                    recall = recall_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred)
                    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None
                    
                    results[model_name] = {
                        'Accuracy': accuracy,
                        'Precision': precision,
                        'Recall': recall,
                        'F1 Score': f1,
                        'AUC-ROC': auc,
                        'Model': model,
                        'Predictions': y_pred,
                        'Probabilities': y_prob
                    }
            
            # Display results
            st.markdown("### 📊 Model Performance Comparison")
            
            results_df = pd.DataFrame(results).T
            display_cols = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC']
            results_df = results_df[display_cols].round(3)
            
            # Style the dataframe
            styled_df = results_df.style.background_gradient(cmap='RdYlGn', subset=display_cols)
            st.dataframe(styled_df, use_container_width=True)
            
            # Identify best model
            best_model = results_df['F1 Score'].idxmax()
            best_f1 = results_df.loc[best_model, 'F1 Score']
            
            st.markdown(f"""
            <div style="background: #dcfce7; padding: 1rem; border-radius: 8px;
                        border-left: 4px solid #10b981; margin: 1rem 0;">
                <h5 style="color: #166534; margin-bottom: 0.25rem;">🏆 Best Model: {best_model}</h5>
                <p style="color: #166534; margin: 0;">
                    F1 Score: {best_f1:.3f} | Accuracy: {results_df.loc[best_model, 'Accuracy']:.3f}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Confusion matrix for best model
            if show_confusion_matrix:
                st.markdown("#### 📊 Confusion Matrix - Best Model")
                
                cm = confusion_matrix(y_test, results[best_model]['Predictions'])
                
                fig = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=['Predicted No Risk', 'Predicted At Risk'],
                    y=['Actual No Risk', 'Actual At Risk'],
                    colorscale='Blues',
                    text=cm,
                    texttemplate='%{text}',
                    textfont={"size": 16},
                    hoverongaps=False
                ))
                
                fig.update_layout(
                    title=f"Confusion Matrix - {best_model}",
                    height=400,
                    plot_bgcolor='white',
                    paper_bgcolor='white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate metrics from confusion matrix
                tn, fp, fn, tp = cm.ravel()
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("True Negatives", tn)
                with col2:
                    st.metric("False Positives", fp, delta=f"{fp/(tn+fp):.1%}" if (tn+fp)>0 else "0%", delta_color="inverse")
                with col3:
                    st.metric("False Negatives", fn, delta=f"{fn/(fn+tp):.1%}" if (fn+tp)>0 else "0%", delta_color="inverse")
                with col4:
                    st.metric("True Positives", tp)
            
            # ROC Curve
            st.markdown("#### 📈 ROC Curves")
            
            fig = go.Figure()
            
            for model_name in models_to_run:
                if results[model_name]['AUC-ROC'] is not None:
                    # For simplicity, we'll use the stored AUC value
                    fig.add_trace(go.Scatter(
                        x=[0, 1],
                        y=[0, 1],
                        mode='lines',
                        line=dict(color='#94a3b8', dash='dash'),
                        showlegend=False
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=[0, 1],
                        y=[0, 1],
                        mode='markers',
                        marker=dict(size=1),
                        name=f"{model_name} (AUC={results[model_name]['AUC-ROC']:.3f})",
                        line=dict(width=2)
                    ))
            
            fig.update_layout(
                title="ROC Curves Comparison",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                height=400,
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            
            fig.update_xaxes(gridcolor='#e2e8f0', range=[0, 1])
            fig.update_yaxes(gridcolor='#e2e8f0', range=[0, 1])
            
            st.plotly_chart(fig, use_container_width=True)
            
            return results
    
    return None

def run_anomaly_detection(df):
    """Run anomaly detection algorithms"""
    
    st.markdown("### 🔍 Anomaly Detection in Cash Flow")
    
    # Feature selection
    feature_cols = ['Revenue_USD_M', 'Operating_Cost_USD_M', 'Free_Cash_Flow_USD_M',
                    'Debt_Outstanding_USD_M', 'Cash_Balance_USD_M']
    
    X = df[feature_cols].copy()
    
    # Model configuration
    col1, col2 = st.columns(2)
    
    with col1:
        contamination = st.slider(
            "Expected Anomaly Rate (%)",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
            help="Expected proportion of anomalies in the data"
        ) / 100
        
        algorithm = st.selectbox(
            "Detection Algorithm",
            ["Isolation Forest", "One-Class SVM", "Elliptic Envelope", "DBSCAN"],
            index=0
        )
    
    with col2:
        scale_data = st.checkbox("Scale Features", value=True)
        visualize_3d = st.checkbox("3D Visualization", value=True)
    
    if st.button("🔍 Detect Anomalies", type="primary", use_container_width=True):
        with st.spinner("Running anomaly detection..."):
            
            # Scale features if requested
            if scale_data:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
            else:
                X_scaled = X.values
            
            # Apply selected algorithm
            if algorithm == "Isolation Forest":
                model = IsolationForest(contamination=contamination, random_state=42)
                predictions = model.fit_predict(X_scaled)
                # Convert: -1 = anomaly, 1 = normal
                anomalies = predictions == -1
            
            elif algorithm == "One-Class SVM":
                from sklearn.svm import OneClassSVM
                model = OneClassSVM(nu=contamination, kernel='rbf')
                predictions = model.fit_predict(X_scaled)
                anomalies = predictions == -1
            
            elif algorithm == "Elliptic Envelope":
                from sklearn.covariance import EllipticEnvelope
                model = EllipticEnvelope(contamination=contamination, random_state=42)
                predictions = model.fit_predict(X_scaled)
                anomalies = predictions == -1
            
            else:  # DBSCAN
                from sklearn.cluster import DBSCAN
                model = DBSCAN(eps=0.5, min_samples=5)
                clusters = model.fit_predict(X_scaled)
                anomalies = clusters == -1
            
            # Add results to dataframe
            df_results = df.copy()
            df_results['Anomaly'] = anomalies
            df_results['Anomaly_Score'] = model.decision_function(X_scaled) if hasattr(model, 'decision_function') else None
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                n_anomalies = anomalies.sum()
                st.metric("Anomalies Detected", n_anomalies)
            
            with col2:
                anomaly_rate = n_anomalies / len(df) * 100
                st.metric("Anomaly Rate", f"{anomaly_rate:.1f}%")
            
            with col3:
                normal_points = len(df) - n_anomalies
                st.metric("Normal Points", normal_points)
            
            # Visualization
            if visualize_3d:
                st.markdown("#### 📊 3D Anomaly Visualization")
                
                fig = go.Figure(data=[
                    go.Scatter3d(
                        x=df_results[df_results['Anomaly'] == False]['Revenue_USD_M'],
                        y=df_results[df_results['Anomaly'] == False]['Operating_Cost_USD_M'],
                        z=df_results[df_results['Anomaly'] == False]['Free_Cash_Flow_USD_M'],
                        mode='markers',
                        name='Normal',
                        marker=dict(
                            size=5,
                            color='#10b981',
                            opacity=0.8
                        )
                    ),
                    go.Scatter3d(
                        x=df_results[df_results['Anomaly'] == True]['Revenue_USD_M'],
                        y=df_results[df_results['Anomaly'] == True]['Operating_Cost_USD_M'],
                        z=df_results[df_results['Anomaly'] == True]['Free_Cash_Flow_USD_M'],
                        mode='markers',
                        name='Anomaly',
                        marker=dict(
                            size=8,
                            color='#ef4444',
                            symbol='x'
                        )
                    )
                ])
                
                fig.update_layout(
                    title="3D Anomaly Detection Results",
                    scene=dict(
                        xaxis_title="Revenue (M$)",
                        yaxis_title="Operating Cost (M$)",
                        zaxis_title="Free Cash Flow (M$)"
                    ),
                    height=600,
                    plot_bgcolor='white',
                    paper_bgcolor='white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            else:
                # 2D scatter matrix
                st.markdown("#### 📊 Anomaly Detection Results")
                
                fig = px.scatter_matrix(
                    df_results,
                    dimensions=['Revenue_USD_M', 'Operating_Cost_USD_M', 'Free_Cash_Flow_USD_M', 'Cash_Balance_USD_M'],
                    color='Anomaly',
                    color_discrete_map={False: '#10b981', True: '#ef4444'},
                    opacity=0.7
                )
                
                fig.update_layout(
                    height=800,
                    plot_bgcolor='white',
                    paper_bgcolor='white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # List anomalies
            st.markdown("#### 📋 Detected Anomalies")
            
            anomaly_df = df_results[df_results['Anomaly'] == True][['Date', 'Revenue_USD_M', 
                                                                   'Operating_Cost_USD_M', 
                                                                   'Free_Cash_Flow_USD_M',
                                                                   'Cash_Balance_USD_M']].copy()
            
            if len(anomaly_df) > 0:
                anomaly_df['Date'] = anomaly_df['Date'].dt.strftime('%Y-%m')
                st.dataframe(anomaly_df, use_container_width=True, hide_index=True)
                
                # Download anomalies
                csv = anomaly_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Anomaly List",
                    data=csv,
                    file_name=f"anomalies_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No anomalies detected")
            
            return df_results
    
    return None

def run_feature_importance(df, target):
    """Run feature importance analysis using multiple methods"""
    
    st.markdown("### 🔍 Feature Importance Analysis")
    
    st.markdown("""
    <div style="background: #f8fafc; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
        <p style="color: #475569; margin: 0;">
            Identify which features have the most impact on your target variable using
            multiple feature importance techniques.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Prepare data
    feature_cols = [col for col in df.columns if col not in 
                   ['Date', target, 'Default_Risk', 'Risk_Class', 'YearMonth', 'Anomaly']]
    
    X = df[feature_cols]
    y = df[target]
    
    # Remove any remaining non-numeric columns
    X = X.select_dtypes(include=[np.number])
    
    # Method selection
    methods = st.multiselect(
        "Select Importance Methods",
        ["Random Forest", "XGBoost", "Correlation", "Mutual Information", "Permutation"],
        default=["Random Forest", "XGBoost", "Correlation"]
    )
    
    if st.button("🔍 Calculate Feature Importance", type="primary", use_container_width=True):
        with st.spinner("Calculating feature importance..."):
            
            importance_results = {}
            
            # Random Forest importance
            if "Random Forest" in methods:
                rf = RandomForestRegressor(n_estimators=100, random_state=42)
                rf.fit(X, y)
                importance_results["Random Forest"] = pd.Series(
                    rf.feature_importances_,
                    index=X.columns
                ).sort_values(ascending=False)
            
            # XGBoost importance
            if "XGBoost" in methods:
                xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
                xgb_model.fit(X, y)
                importance_results["XGBoost"] = pd.Series(
                    xgb_model.feature_importances_,
                    index=X.columns
                ).sort_values(ascending=False)
            
            # Correlation
            if "Correlation" in methods:
                correlations = X.corrwith(y).abs()
                importance_results["Correlation"] = correlations.sort_values(ascending=False)
            
            # Mutual Information
            if "Mutual Information" in methods:
                from sklearn.feature_selection import mutual_info_regression
                mi = mutual_info_regression(X, y)
                importance_results["Mutual Information"] = pd.Series(
                    mi,
                    index=X.columns
                ).sort_values(ascending=False)
            
            # Permutation Importance
            if "Permutation" in methods:
                from sklearn.inspection import permutation_importance
                rf_temp = RandomForestRegressor(n_estimations=50, random_state=42)
                rf_temp.fit(X, y)
                perm_importance = permutation_importance(rf_temp, X, y, n_repeats=10, random_state=42)
                importance_results["Permutation"] = pd.Series(
                    perm_importance.importances_mean,
                    index=X.columns
                ).sort_values(ascending=False)
            
            # Create visualization
            st.markdown("#### 📊 Feature Importance Comparison")
            
            # Determine number of subplots
            n_methods = len(importance_results)
            n_cols = min(2, n_methods)
            n_rows = (n_methods + n_cols - 1) // n_cols
            
            fig = make_subplots(
                rows=n_rows, cols=n_cols,
                subplot_titles=list(importance_results.keys()),
                vertical_spacing=0.2,
                horizontal_spacing=0.15
            )
            
            row, col = 1, 1
            for method_name, importance_series in importance_results.items():
                # Take top 10 features
                top_features = importance_series.head(10)
                
                fig.add_trace(
                    go.Bar(
                        y=top_features.index,
                        x=top_features.values,
                        orientation='h',
                        name=method_name,
                        marker_color='#059669',
                        text=top_features.values.round(3),
                        textposition='outside',
                        showlegend=False
                    ),
                    row=row, col=col
                )
                
                col += 1
                if col > n_cols:
                    col = 1
                    row += 1
            
            fig.update_layout(
                height=400 * n_rows,
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            
            fig.update_xaxes(gridcolor='#e2e8f0')
            fig.update_yaxes(gridcolor='#e2e8f0')
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Consensus features
            st.markdown("#### 🎯 Top Consensus Features")
            
            # Combine importance scores
            all_features = pd.DataFrame(importance_results)
            all_features = all_features.fillna(0)
            
            # Normalize each method
            all_features_normalized = all_features.div(all_features.sum(axis=0), axis=1)
            
            # Calculate consensus score
            consensus = all_features_normalized.mean(axis=1).sort_values(ascending=False)
            
            # Display top consensus features
            consensus_df = pd.DataFrame({
                'Feature': consensus.index[:10],
                'Consensus Score': consensus.values[:10],
                'Avg Rank': [all_features_normalized.loc[f].rank(ascending=False).mean() for f in consensus.index[:10]]
            })
            
            st.dataframe(consensus_df, use_container_width=True, hide_index=True)
            
            # Radar chart for top features
            st.markdown("#### 📈 Feature Importance Radar")
            
            top_features_consensus = consensus.index[:6]
            
            fig = go.Figure()
            
            for method_name in importance_results.keys():
                method_scores = []
                for feature in top_features_consensus:
                    if feature in importance_results[method_name].index:
                        # Normalize within method
                        max_val = importance_results[method_name].max()
                        score = importance_results[method_name][feature] / max_val if max_val > 0 else 0
                    else:
                        score = 0
                    method_scores.append(score)
                
                fig.add_trace(go.Scatterpolar(
                    r=method_scores,
                    theta=top_features_consensus,
                    fill='toself',
                    name=method_name
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=True,
                height=500,
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            return importance_results
    
    return None

def run_time_series_ml(df, target):
    """Run time series forecasting using ML approaches"""
    
    st.markdown("### 📈 Time Series Forecasting with ML")
    
    st.markdown("""
    <div style="background: #f8fafc; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
        <p style="color: #475569; margin: 0;">
            Use machine learning algorithms for time series forecasting with lagged features
            and rolling statistics.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Prepare time series features
    df_ts = df.copy()
    
    # Create date-based features
    df_ts['month'] = df_ts['Date'].dt.month
    df_ts['quarter'] = df_ts['Date'].dt.quarter
    df_ts['year'] = df_ts['Date'].dt.year
    
    # Create lag features
    for lag in [1, 2, 3, 6, 12]:
        df_ts[f'lag_{lag}'] = df_ts[target].shift(lag)
    
    # Create rolling features
    for window in [3, 6]:
        df_ts[f'rolling_mean_{window}'] = df_ts[target].rolling(window=window).mean()
        df_ts[f'rolling_std_{window}'] = df_ts[target].rolling(window=window).std()
    
    # Drop NaN values
    df_ts = df_ts.dropna()
    
    # Feature selection
    feature_cols = [col for col in df_ts.columns if col not in 
                   ['Date', target, 'Default_Risk', 'Risk_Class', 'YearMonth', 'Anomaly']]
    
    X = df_ts[feature_cols]
    y = df_ts[target]
    
    # Configuration
    col1, col2 = st.columns(2)
    
    with col1:
        forecast_horizon = st.slider(
            "Forecast Horizon (steps)",
            min_value=1,
            max_value=12,
            value=6,
            step=1
        )
        
        model_type = st.selectbox(
            "Select Model",
            ["Random Forest", "XGBoost", "LightGBM", "Linear Regression"]
        )
    
    with col2:
        cv_folds = st.slider("CV Folds", 3, 10, 5)
        show_importance = st.checkbox("Show Feature Importance", value=True)
    
    if st.button("🚀 Train Time Series Model", type="primary", use_container_width=True):
        with st.spinner("Training time series model..."):
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            
            # Select model
            if model_type == "Random Forest":
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            elif model_type == "XGBoost":
                model = xgb.XGBRegressor(n_estimators=100, random_state=42)
            elif model_type == "LightGBM":
                model = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
            else:
                model = LinearRegression()
            
            # Train on all data
            model.fit(X, y)
            
            # Make future predictions (simplified - recursive forecasting)
            last_known = X.iloc[-1:].copy()
            future_predictions = []
            
            for i in range(forecast_horizon):
                # Predict next step
                next_pred = model.predict(last_known)[0]
                future_predictions.append(next_pred)
                
                # Update features for next prediction
                # (In practice, you'd need to update all lagged features)
                last_known = last_known.copy()
                for lag in [12, 6, 3, 2, 1]:
                    if f'lag_{lag}' in last_known.columns:
                        if lag == 1:
                            last_known[f'lag_{lag}'] = next_pred
                        else:
                            last_known[f'lag_{lag}'] = last_known[f'lag_{lag-1}'].values
            
            # Cross-validation scores
            cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='r2')
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("CV R² Mean", f"{cv_scores.mean():.3f}")
            with col2:
                st.metric("CV R² Std", f"{cv_scores.std():.3f}")
            with col3:
                st.metric("Training R²", f"{model.score(X, y):.3f}")
            
            # Plot historical and forecast
            fig = go.Figure()
            
            # Historical
            fig.add_trace(go.Scatter(
                x=df_ts['Date'],
                y=df_ts[target],
                mode='lines',
                name='Historical',
                line=dict(color='#1e293b', width=2)
            ))
            
            # Forecast
            future_dates = pd.date_range(
                start=df_ts['Date'].iloc[-1] + pd.Timedelta(days=30),
                periods=forecast_horizon,
                freq='M'
            )
            
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=future_predictions,
                mode='lines+markers',
                name='Forecast',
                line=dict(color='#059669', width=2, dash='dash'),
                marker=dict(size=8)
            ))
            
            fig.update_layout(
                title=f"{target} - Historical and Forecast",
                xaxis_title="Date",
                yaxis_title=f"{target} (USD M)",
                height=500,
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            
            fig.update_xaxes(gridcolor='#e2e8f0')
            fig.update_yaxes(gridcolor='#e2e8f0')
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Forecast table
            st.markdown("#### 📊 Forecast Values")
            
            forecast_df = pd.DataFrame({
                'Date': future_dates.strftime('%Y-%m'),
                'Forecast': [f"${v:.2f}M" for v in future_predictions]
            })
            
            st.dataframe(forecast_df, use_container_width=True, hide_index=True)
            
            # Feature importance
            if show_importance and hasattr(model, 'feature_importances_'):
                st.markdown("#### 🔍 Feature Importance")
                
                importance = pd.Series(
                    model.feature_importances_,
                    index=feature_cols
                ).sort_values(ascending=False).head(10)
                
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    y=importance.index,
                    x=importance.values,
                    orientation='h',
                    marker_color='#059669',
                    text=importance.values.round(3),
                    textposition='outside'
                ))
                
                fig.update_layout(
                    title="Top 10 Important Features",
                    xaxis_title="Importance",
                    yaxis_title="Feature",
                    height=400,
                    plot_bgcolor='white',
                    paper_bgcolor='white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            return {
                'model': model,
                'cv_scores': cv_scores,
                'forecast': future_predictions,
                'forecast_dates': future_dates
            }
    
    return None

# Test function for standalone testing
if __name__ == "__main__":
    print("Machine Learning Models module ready for integration")
    print("Features: Regression, Classification, Anomaly Detection, Feature Importance, Time Series ML")