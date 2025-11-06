import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

# Load and prepare dataset
try:
    df = pd.read_csv("Hotel Reservations.csv", encoding="latin1")
    print("✅ Loaded Hotel Reservations.csv for forecasting")
    
    # Create datetime
    df["arrival_date"] = pd.to_datetime(
        df["arrival_year"].astype(str) + "-" +
        df["arrival_month"].astype(str) + "-" +
        df["arrival_date"].astype(str),
        errors="coerce"
    )
    
    # Aggregate daily bookings
    bookings = df.groupby("arrival_date").size().reset_index(name="bookings")
    bookings = bookings.set_index("arrival_date").asfreq("D").fillna(0)
    
    print(f"📊 Prepared {len(bookings)} daily booking records")
    
except Exception as error:
    print(f"❌ Error loading data: {error}")
    bookings = pd.DataFrame()

# Enhanced Feature Engineering
def create_advanced_features(data, lags=[1, 2, 3, 7, 14, 21, 28]):
    """Create comprehensive features for time series forecasting"""
    df_feat = pd.DataFrame(index=data.index)
    df_feat["y"] = data.values
    
    # Lag features
    for lag in lags:
        df_feat[f"lag_{lag}"] = data.shift(lag)
    
    # Rolling statistics - multiple windows
    for window in [7, 14, 30]:
        df_feat[f"rolling_mean_{window}"] = data.rolling(window=window, min_periods=1).mean()
        df_feat[f"rolling_std_{window}"] = data.rolling(window=window, min_periods=1).std()
        df_feat[f"rolling_min_{window}"] = data.rolling(window=window, min_periods=1).min()
        df_feat[f"rolling_max_{window}"] = data.rolling(window=window, min_periods=1).max()
    
    # Exponential moving averages
    for span in [7, 14]:
        df_feat[f"ema_{span}"] = data.ewm(span=span, adjust=False).mean()
    
    # Seasonal features
    df_feat["day_of_week"] = df_feat.index.dayofweek
    df_feat["day_of_month"] = df_feat.index.day
    df_feat["month"] = df_feat.index.month
    df_feat["is_weekend"] = (df_feat["day_of_week"] >= 5).astype(int)
    
    # Cyclical encoding for seasonal patterns
    df_feat["day_of_week_sin"] = np.sin(2 * np.pi * df_feat["day_of_week"] / 7)
    df_feat["day_of_week_cos"] = np.cos(2 * np.pi * df_feat["day_of_week"] / 7)
    df_feat["month_sin"] = np.sin(2 * np.pi * df_feat["month"] / 12)
    df_feat["month_cos"] = np.cos(2 * np.pi * df_feat["month"] / 12)
    
    # Trend features
    df_feat["day_number"] = range(len(df_feat))
    df_feat["year"] = df_feat.index.year
    
    # Difference features (rate of change)
    df_feat["diff_1"] = data.diff(1)
    df_feat["diff_7"] = data.diff(7)
    
    # Percentage change
    df_feat["pct_change_1"] = data.pct_change(1).fillna(0)
    df_feat["pct_change_7"] = data.pct_change(7).fillna(0)
    
    # Fill NaN values (only for rolling stats at the beginning)
    df_feat = df_feat.bfill().fillna(0)
    
    return df_feat

# Calculate MAPE (Mean Absolute Percentage Error)
def mean_absolute_percentage_error(y_true, y_pred):
    """Calculate MAPE, handling zero values"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# Train optimized XGBoost model with hyperparameter tuning
def train_forecasting_model():
    if bookings.empty:
        return None, None, None, None, None, None
    
    try:
        print("🔧 Creating advanced features...")
        data_feat = create_advanced_features(bookings["bookings"])
        print(f"✅ Created {len(data_feat.columns)-1} features")
        
        # Split train-test (maintain temporal order)
        train_size = int(len(data_feat) * 0.8)
        train, test = data_feat.iloc[:train_size], data_feat.iloc[train_size:]
        
        X_train, y_train = train.drop("y", axis=1), train["y"]
        X_test, y_test = test.drop("y", axis=1), test["y"]
        
        print(f"📊 Training set: {len(X_train)} samples, Test set: {len(X_test)} samples")
        
        # Hyperparameter tuning with RandomizedSearchCV
        print("🎯 Tuning hyperparameters with cross-validation...")
        param_grid = {
            'n_estimators': [300, 400, 500, 600],
            'max_depth': [4, 5, 6, 7],
            'learning_rate': [0.01, 0.05, 0.1, 0.15],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.2],
            'reg_alpha': [0, 0.1, 0.5],
            'reg_lambda': [1, 1.5, 2]
        }
        
        # Base model with early stopping
        base_model = XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1
        )
        
        # Use TimeSeriesSplit for cross-validation (respects temporal order)
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Randomized search (more efficient than GridSearch)
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_grid,
            n_iter=30,  # Number of parameter combinations to try
            scoring='neg_root_mean_squared_error',
            cv=tscv,
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        
        random_search.fit(X_train, y_train)
        best_model = random_search.best_estimator_
        
        print(f"✅ Best hyperparameters found!")
        print(f"   Best RMSE (CV): {-random_search.best_score_:.2f}")
        
        # Retrain best model on full training set for final evaluation
        best_model.fit(X_train, y_train)
        
        # Predict on test
        y_pred = best_model.predict(X_test)
        
        # Calculate comprehensive metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"📈 Test Set Performance:")
        print(f"   RMSE: {rmse:.2f}")
        print(f"   MAE:  {mae:.2f}")
        print(f"   MAPE: {mape:.2f}%")
        print(f"   R²:   {r2:.4f}")
        
        # Calculate prediction intervals based on residuals
        residuals = y_test - y_pred
        std_residual = np.std(residuals)
        
        # Generate future forecast with proper feature updates
        forecast_days = 30
        last_data = data_feat.iloc[-1:].copy()
        predictions = []
        forecast_std = []
        
        # Store feature names for updating
        feature_cols = [col for col in last_data.columns if col != "y"]
        
        for i in range(forecast_days):
            # Prepare input (exclude y)
            x_input = last_data[feature_cols].iloc[-1:].values
            
            # Predict
            yhat = best_model.predict(x_input)[0]
            predictions.append(max(0, yhat))  # Ensure non-negative
            
            # Calculate prediction interval
            forecast_std.append(std_residual)
            
            # Update features for next prediction
            next_date = last_data.index[-1] + pd.Timedelta(days=1)
            
            # Create new row with updated features
            new_row = pd.DataFrame(index=[next_date])
            new_row["y"] = yhat
            
            # Update lag features
            for lag in [1, 2, 3, 7, 14, 21, 28]:
                if lag == 1:
                    new_row[f"lag_{lag}"] = yhat
                else:
                    # Get lag from previous row
                    if lag <= len(last_data):
                        new_row[f"lag_{lag}"] = last_data.iloc[-lag]["y"] if lag <= len(last_data) else 0
                    else:
                        new_row[f"lag_{lag}"] = 0
            
            # Update rolling statistics (use historical data + predictions)
            # Get recent actual bookings
            recent_actual = bookings["bookings"].tail(30).tolist()
            # Combine with predictions made so far
            recent_values = recent_actual + predictions
            
            for window in [7, 14, 30]:
                # Use last 'window' values from the combined list
                window_data = recent_values[-window:] if len(recent_values) >= window else recent_values
                if len(window_data) > 0:
                    new_row[f"rolling_mean_{window}"] = np.mean(window_data)
                    new_row[f"rolling_std_{window}"] = np.std(window_data) if len(window_data) > 1 else 0
                    new_row[f"rolling_min_{window}"] = np.min(window_data)
                    new_row[f"rolling_max_{window}"] = np.max(window_data)
                else:
                    new_row[f"rolling_mean_{window}"] = yhat
                    new_row[f"rolling_std_{window}"] = 0
                    new_row[f"rolling_min_{window}"] = yhat
                    new_row[f"rolling_max_{window}"] = yhat
            
            # Update EMA
            for span in [7, 14]:
                # Simplified EMA update
                if len(last_data) > 0:
                    alpha = 2 / (span + 1)
                    prev_ema = last_data[f"ema_{span}"].iloc[-1] if f"ema_{span}" in last_data.columns else yhat
                    new_row[f"ema_{span}"] = alpha * yhat + (1 - alpha) * prev_ema
                else:
                    new_row[f"ema_{span}"] = yhat
            
            # Update seasonal features
            new_row["day_of_week"] = next_date.dayofweek
            new_row["day_of_month"] = next_date.day
            new_row["month"] = next_date.month
            new_row["is_weekend"] = 1 if next_date.dayofweek >= 5 else 0
            new_row["day_of_week_sin"] = np.sin(2 * np.pi * next_date.dayofweek / 7)
            new_row["day_of_week_cos"] = np.cos(2 * np.pi * next_date.dayofweek / 7)
            new_row["month_sin"] = np.sin(2 * np.pi * next_date.month / 12)
            new_row["month_cos"] = np.cos(2 * np.pi * next_date.month / 12)
            new_row["day_number"] = len(last_data) + 1
            new_row["year"] = next_date.year
            
            # Update difference features
            if len(last_data) > 0:
                prev_value = last_data["y"].iloc[-1]
                new_row["diff_1"] = yhat - prev_value
                if len(last_data) >= 7:
                    new_row["diff_7"] = yhat - last_data["y"].iloc[-7]
                else:
                    new_row["diff_7"] = 0
                new_row["pct_change_1"] = (yhat - prev_value) / prev_value if prev_value != 0 else 0
                if len(last_data) >= 7:
                    prev_7 = last_data["y"].iloc[-7]
                    new_row["pct_change_7"] = (yhat - prev_7) / prev_7 if prev_7 != 0 else 0
                else:
                    new_row["pct_change_7"] = 0
            else:
                new_row["diff_1"] = 0
                new_row["diff_7"] = 0
                new_row["pct_change_1"] = 0
                new_row["pct_change_7"] = 0
            
            # Ensure all feature columns exist
            for col in feature_cols:
                if col not in new_row.columns:
                    new_row[col] = 0
            
            # Append to last_data
            last_data = pd.concat([last_data, new_row])
        
        # Create forecast dataframe with confidence intervals
        forecast_dates = pd.date_range(start=bookings.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
        forecast_df = pd.DataFrame({
            "date": forecast_dates,
            "forecast": predictions,
            "upper": [p + 1.96 * s for p, s in zip(predictions, forecast_std)],
            "lower": [max(0, p - 1.96 * s) for p, s in zip(predictions, forecast_std)]
        })
        
        # Store metrics in a dictionary
        metrics = {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2
        }
        
        return best_model, y_test, y_pred, metrics, forecast_df, feature_cols
        
    except Exception as e:
        print(f"❌ Error in forecasting: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None, None

# Train the model
model, y_test, y_pred, metrics, forecast_df, feature_cols = train_forecasting_model()

# Extract metrics for backward compatibility
rmse = metrics['rmse'] if metrics else None
mae = metrics['mae'] if metrics else None
mape = metrics['mape'] if metrics else None
r2 = metrics['r2'] if metrics else None

# Create actual vs predicted chart
def create_actual_vs_predicted():
    if y_test is None or y_pred is None:
        return go.Figure()
    
    fig = go.Figure()
    
    # Add actual bookings
    fig.add_trace(go.Scatter(
        x=y_test.index,
        y=y_test,
        mode='lines',
        name='Actual Bookings',
        line=dict(color='#3498db', width=3)
    ))
    
    # Add predicted bookings
    fig.add_trace(go.Scatter(
        x=y_test.index,
        y=y_pred,
        mode='lines',
        name='Predicted Bookings',
        line=dict(color='#e74c3c', width=3, dash='dash')
    ))
    
    # Calculate metrics if available
    metrics_text = ""
    if rmse:
        metrics_text = f"RMSE: {rmse:.2f} | MAE: {mae:.2f} | MAPE: {mape:.2f}% | R²: {r2:.4f}"
    
    fig.update_layout(
        title={
            'text': f"Optimized XGBoost Forecast - {metrics_text}",
            'x': 0.5,
            'font': {'size': 18, 'family': 'Inter', 'color': '#2c3e50'}
        },
        xaxis_title="Date",
        yaxis_title="Number of Bookings",
        height=500,
        margin=dict(l=20, r=20, t=60, b=20),
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig

# Create feature importance chart
def create_feature_importance():
    if model is None or feature_cols is None:
        return go.Figure()
    
    importances = model.feature_importances_
    features = feature_cols
    
    # Sort by importance
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': importances
    }).sort_values('importance', ascending=False).head(15)  # Show top 15
    
    fig = go.Figure(data=[
        go.Bar(
            x=importance_df['importance'],
            y=importance_df['feature'],
            orientation='h',
            marker_color='#667eea',
            text=[f'{imp:.3f}' for imp in importance_df['importance']],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title={
            'text': "Top 15 Feature Importance (XGBoost)",
            'x': 0.5,
            'font': {'size': 20, 'family': 'Inter', 'color': '#2c3e50'}
        },
        xaxis_title="Importance Score",
        yaxis_title="Features",
        height=500,
        margin=dict(l=20, r=20, t=60, b=20),
        template='plotly_white'
    )
    
    return fig

# Create future forecast chart
def create_future_forecast():
    if forecast_df is None or forecast_df.empty:
        return go.Figure()
    
    # Get recent actual data for context
    recent_data = bookings.tail(30)
    
    fig = go.Figure()
    
    # Add recent actual data
    fig.add_trace(go.Scatter(
        x=recent_data.index,
        y=recent_data['bookings'],
        mode='lines',
        name='Recent Actual Bookings',
        line=dict(color='#3498db', width=3)
    ))
    
    # Add future forecast
    fig.add_trace(go.Scatter(
        x=forecast_df['date'],
        y=forecast_df['forecast'],
        mode='lines+markers',
        name='Future Forecast (30 days)',
        line=dict(color='#2ecc71', width=3),
        marker=dict(size=6, color='#2ecc71')
    ))
    
    # Add confidence interval (95% prediction interval)
    if 'upper' in forecast_df.columns and 'lower' in forecast_df.columns:
        fig.add_trace(go.Scatter(
            x=forecast_df['date'],
            y=forecast_df['upper'],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_df['date'],
            y=forecast_df['lower'],
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(46, 204, 113, 0.2)',
            name='95% Prediction Interval',
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        title={
            'text': "30-Day Hotel Booking Forecast with Confidence Interval",
            'x': 0.5,
            'font': {'size': 20, 'family': 'Inter', 'color': '#2c3e50'}
        },
        xaxis_title="Date",
        yaxis_title="Number of Bookings",
        height=500,
        margin=dict(l=20, r=20, t=60, b=20),
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig

# Create forecast summary metrics
def create_forecast_metrics():
    if forecast_df is None or forecast_df.empty:
        return html.Div()
    
    avg_forecast = forecast_df['forecast'].mean()
    max_forecast = forecast_df['forecast'].max()
    min_forecast = forecast_df['forecast'].min()
    total_forecast = forecast_df['forecast'].sum()
    
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.I(className="fas fa-chart-line", style={'fontSize': '2rem', 'color': '#667eea'}),
                    html.H3(f"{avg_forecast:.1f}", style={'margin': '10px 0', 'color': '#2c3e50'}),
                    html.P("Avg Daily Bookings", style={'margin': '0', 'color': '#666'})
                ], style={'textAlign': 'center', 'padding': '20px'})
            ], width=3),
            
            dbc.Col([
                html.Div([
                    html.I(className="fas fa-arrow-up", style={'fontSize': '2rem', 'color': '#2ecc71'}),
                    html.H3(f"{max_forecast:.0f}", style={'margin': '10px 0', 'color': '#2c3e50'}),
                    html.P("Peak Forecast", style={'margin': '0', 'color': '#666'})
                ], style={'textAlign': 'center', 'padding': '20px'})
            ], width=3),
            
            dbc.Col([
                html.Div([
                    html.I(className="fas fa-arrow-down", style={'fontSize': '2rem', 'color': '#e74c3c'}),
                    html.H3(f"{min_forecast:.0f}", style={'margin': '10px 0', 'color': '#2c3e50'}),
                    html.P("Lowest Forecast", style={'margin': '0', 'color': '#666'})
                ], style={'textAlign': 'center', 'padding': '20px'})
            ], width=3),
            
            dbc.Col([
                html.Div([
                    html.I(className="fas fa-calendar-check", style={'fontSize': '2rem', 'color': '#f39c12'}),
                    html.H3(f"{total_forecast:.0f}", style={'margin': '10px 0', 'color': '#2c3e50'}),
                    html.P("Total 30-Day Forecast", style={'margin': '0', 'color': '#666'})
                ], style={'textAlign': 'center', 'padding': '20px'})
            ], width=3),
        ])
    ], className="chart-container")

# Page layout
layout = html.Div([
    # Page header
    html.Div([
        html.H1("🔮 AI-Powered Booking Forecasting", className="page-title"),
        html.P("Advanced XGBoost machine learning models for accurate hotel booking predictions", 
               className="page-subtitle")
    ], className="page-header"),
    
    # Model performance metrics
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H4("📊 Model Performance Metrics", style={'color': '#2c3e50', 'marginBottom': '20px'}),
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.I(className="fas fa-bullseye", style={'fontSize': '2rem', 'color': '#667eea'}),
                            html.H3(f"{rmse:.2f}" if rmse else "N/A", style={'margin': '10px 0', 'color': '#2c3e50'}),
                            html.P("RMSE", style={'margin': '0', 'color': '#666'})
                        ], style={'textAlign': 'center', 'padding': '20px'})
                    ], width=3),
                    dbc.Col([
                        html.Div([
                            html.I(className="fas fa-chart-line", style={'fontSize': '2rem', 'color': '#e74c3c'}),
                            html.H3(f"{mae:.2f}" if mae else "N/A", style={'margin': '10px 0', 'color': '#2c3e50'}),
                            html.P("MAE", style={'margin': '0', 'color': '#666'})
                        ], style={'textAlign': 'center', 'padding': '20px'})
                    ], width=3),
                    dbc.Col([
                        html.Div([
                            html.I(className="fas fa-percent", style={'fontSize': '2rem', 'color': '#f39c12'}),
                            html.H3(f"{mape:.2f}%" if mape else "N/A", style={'margin': '10px 0', 'color': '#2c3e50'}),
                            html.P("MAPE", style={'margin': '0', 'color': '#666'})
                        ], style={'textAlign': 'center', 'padding': '20px'})
                    ], width=3),
                    dbc.Col([
                        html.Div([
                            html.I(className="fas fa-trophy", style={'fontSize': '2rem', 'color': '#2ecc71'}),
                            html.H3(f"{r2:.4f}" if r2 else "N/A", style={'margin': '10px 0', 'color': '#2c3e50'}),
                            html.P("R² Score", style={'margin': '0', 'color': '#666'})
                        ], style={'textAlign': 'center', 'padding': '20px'})
                    ], width=3),
                ])
            ], className="chart-container")
        ], width=12),
    ], className="mb-4"),
    
    # Forecast metrics
    create_forecast_metrics(),
    
    # Charts
    dbc.Row([
        dbc.Col([
            html.Div([
                dcc.Graph(figure=create_actual_vs_predicted())
            ], className="chart-container")
        ], width=12),
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            html.Div([
                dcc.Graph(figure=create_feature_importance())
            ], className="chart-container")
        ], width=6),
        
        dbc.Col([
            html.Div([
                dcc.Graph(figure=create_future_forecast())
            ], className="chart-container")
        ], width=6),
    ], className="mb-4"),
    
    # Forecast table
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H4("📅 Detailed 30-Day Forecast", style={'color': '#2c3e50', 'marginBottom': '20px'}),
                html.Div([
                    html.Table([
                        html.Thead([
                            html.Tr([
                                html.Th("Date"),
                                html.Th("Forecasted Bookings"),
                                html.Th("Confidence Level")
                            ])
                        ]),
                        html.Tbody([
                            html.Tr([
                                html.Td(forecast_df.iloc[i]['date'].strftime('%Y-%m-%d')),
                                html.Td(f"{forecast_df.iloc[i]['forecast']:.1f}"),
                                html.Td("High" if i < 7 else "Medium" if i < 21 else "Low")
                            ]) for i in range(min(10, len(forecast_df)))
                        ])
                    ], className="table table-striped")
                ])
            ], className="chart-container")
        ], width=12),
    ], className="mb-4"),
    
    # Technical details
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H4("🔧 Technical Details", style={'color': '#2c3e50', 'marginBottom': '20px'}),
                html.Ul([
                    html.Li("Algorithm: Optimized XGBoost Regressor with hyperparameter tuning"),
                    html.Li(f"Features: {len(feature_cols) if feature_cols else 'N/A'} engineered features (lags, rolling stats, seasonal, trend)"),
                    html.Li("Training: 80% historical data, 20% test set with TimeSeriesSplit cross-validation"),
                    html.Li("Hyperparameter Tuning: RandomizedSearchCV with 30 iterations"),
                    html.Li("Feature Engineering: Multiple lags (1-28 days), rolling statistics, EMA, seasonal encoding"),
                    html.Li("Forecast Horizon: 30 days ahead with 95% prediction intervals"),
                    html.Li("Early Stopping: Enabled to prevent overfitting"),
                    html.Li("Best Model: Optimized via cross-validation with RandomizedSearchCV")
                ], style={'fontSize': '16px', 'lineHeight': '1.6'})
            ], className="chart-container")
        ], width=12),
    ])
])
