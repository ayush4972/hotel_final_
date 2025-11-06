import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Load hotel reservation data
try:
    # Load the original dataset
    df = pd.read_csv("Hotel Reservations.csv", encoding="latin1")
    print("✅ Loaded Hotel Reservations.csv successfully")
    
    # Create date column with error handling
    df['arrival_date'] = pd.to_datetime(
        df[['arrival_year', 'arrival_month', 'arrival_date']].rename(
            columns={'arrival_year': 'year', 'arrival_month': 'month', 'arrival_date': 'day'}
        ), errors='coerce'
    )
    
    # Fill any NaT values
    df['arrival_date'] = df['arrival_date'].fillna(pd.Timestamp('2018-01-01'))
    
    # Create additional features
    df['total_guests'] = df['no_of_adults'] + df['no_of_children']
    df['total_nights'] = df['no_of_weekend_nights'] + df['no_of_week_nights']
    df['is_canceled'] = (df['booking_status'] == 'Canceled').astype(int)
    df['is_repeated'] = (df['repeated_guest'] == 1).astype(int)
    
    print(f"📊 Analytics data prepared: {len(df)} records")
    
except Exception as error:
    print(f"❌ Error loading data: {error}")
    df = pd.DataFrame()

# Create correlation heatmap
def create_correlation_heatmap():
    if df.empty:
        return go.Figure()
    
    # Select numeric columns for correlation
    numeric_cols = [
        'no_of_adults', 'no_of_children', 'no_of_weekend_nights', 'no_of_week_nights',
        'lead_time', 'no_of_previous_cancellations', 'no_of_previous_bookings_not_canceled',
        'avg_price_per_room', 'no_of_special_requests', 'is_canceled', 'is_repeated'
    ]
    
    corr_data = df[numeric_cols].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_data.values,
        x=corr_data.columns,
        y=corr_data.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_data.values, 2),
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title={
            'text': "Feature Correlation Matrix",
            'x': 0.5,
            'font': {'size': 20, 'family': 'Inter', 'color': '#2c3e50'}
        },
        height=600,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

# Create lead time analysis
def create_lead_time_analysis():
    if df.empty:
        return go.Figure()
    
    # Create lead time bins
    df['lead_time_bin'] = pd.cut(df['lead_time'], 
                                bins=[0, 7, 30, 90, 365, float('inf')], 
                                labels=['0-7 days', '8-30 days', '31-90 days', '91-365 days', '365+ days'])
    
    lead_time_analysis = df.groupby(['lead_time_bin', 'booking_status']).size().unstack(fill_value=0)
    lead_time_analysis['total'] = lead_time_analysis.sum(axis=1)
    lead_time_analysis['cancellation_rate'] = (lead_time_analysis.get('Canceled', 0) / lead_time_analysis['total'] * 100).round(1)
    
    fig = go.Figure()
    
    # Add cancellation rate
    fig.add_trace(go.Bar(
        x=lead_time_analysis.index,
        y=lead_time_analysis['cancellation_rate'],
        name='Cancellation Rate (%)',
        marker_color='#e74c3c',
        yaxis='y'
    ))
    
    # Add total bookings
    fig.add_trace(go.Bar(
        x=lead_time_analysis.index,
        y=lead_time_analysis['total'],
        name='Total Bookings',
        marker_color='#3498db',
        yaxis='y2'
    ))
    
    fig.update_layout(
        title={
            'text': "Lead Time vs Cancellation Rate",
            'x': 0.5,
            'font': {'size': 20, 'family': 'Inter', 'color': '#2c3e50'}
        },
        xaxis_title="Lead Time",
        yaxis=dict(title="Cancellation Rate (%)", side="left"),
        yaxis2=dict(title="Total Bookings", side="right", overlaying="y"),
        barmode='group',
        height=500,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

# Create seasonal analysis
def create_seasonal_analysis():
    if df.empty:
        return go.Figure()
    
    # Monthly analysis
    monthly_data = df.groupby('arrival_month').agg({
        'Booking_ID': 'count',
        'avg_price_per_room': 'mean',
        'is_canceled': 'mean'
    }).reset_index()
    
    monthly_data['cancellation_rate'] = monthly_data['is_canceled'] * 100
    
    fig = go.Figure()
    
    # Add bookings
    fig.add_trace(go.Scatter(
        x=monthly_data['arrival_month'],
        y=monthly_data['Booking_ID'],
        mode='lines+markers',
        name='Total Bookings',
        line=dict(color='#3498db', width=3),
        marker=dict(size=8),
        yaxis='y'
    ))
    
    # Add average price
    fig.add_trace(go.Scatter(
        x=monthly_data['arrival_month'],
        y=monthly_data['avg_price_per_room'],
        mode='lines+markers',
        name='Average Price',
        line=dict(color='#e74c3c', width=3),
        marker=dict(size=8),
        yaxis='y2'
    ))
    
    # Add cancellation rate
    fig.add_trace(go.Scatter(
        x=monthly_data['arrival_month'],
        y=monthly_data['cancellation_rate'],
        mode='lines+markers',
        name='Cancellation Rate (%)',
        line=dict(color='#f39c12', width=3),
        marker=dict(size=8),
        yaxis='y3'
    ))
    
    fig.update_layout(
        title={
            'text': "Seasonal Booking Patterns",
            'x': 0.5,
            'font': {'size': 20, 'family': 'Inter', 'color': '#2c3e50'}
        },
        xaxis_title="Month",
        yaxis=dict(title="Total Bookings", side="left"),
        yaxis2=dict(title="Average Price ($)", side="right", overlaying="y"),
        yaxis3=dict(title="Cancellation Rate (%)", side="right", overlaying="y", position=0.95),
        height=500,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

# Create market segment analysis
def create_market_analysis():
    if df.empty:
        return go.Figure()
    
    market_analysis = df.groupby('market_segment_type').agg({
        'Booking_ID': 'count',
        'avg_price_per_room': 'mean',
        'is_canceled': 'mean',
        'lead_time': 'mean'
    }).reset_index()
    
    market_analysis['cancellation_rate'] = market_analysis['is_canceled'] * 100
    
    fig = px.scatter(
        market_analysis,
        x='avg_price_per_room',
        y='cancellation_rate',
        size='Booking_ID',
        color='market_segment_type',
        hover_data=['lead_time'],
        title="Market Segment Analysis: Price vs Cancellation Rate",
        labels={
            'avg_price_per_room': 'Average Price per Room ($)',
            'cancellation_rate': 'Cancellation Rate (%)',
            'Booking_ID': 'Number of Bookings'
        }
    )
    
    fig.update_layout(
        title={
            'text': "Market Segment Analysis: Price vs Cancellation Rate",
            'x': 0.5,
            'font': {'size': 20, 'family': 'Inter', 'color': '#2c3e50'}
        },
        height=500,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

# Create room type profitability analysis
def create_room_profitability():
    if df.empty:
        return go.Figure()
    
    room_analysis = df.groupby('room_type_reserved').agg({
        'Booking_ID': 'count',
        'avg_price_per_room': 'mean',
        'is_canceled': 'mean',
        'total_nights': 'mean'
    }).reset_index()
    
    room_analysis['cancellation_rate'] = room_analysis['is_canceled'] * 100
    room_analysis['revenue_per_booking'] = room_analysis['avg_price_per_room'] * room_analysis['total_nights']
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=room_analysis['room_type_reserved'],
        y=room_analysis['revenue_per_booking'],
        name='Revenue per Booking',
        marker_color='#2ecc71',
        text=room_analysis['revenue_per_booking'].round(0),
        textposition='auto'
    ))
    
    fig.update_layout(
        title={
            'text': "Room Type Profitability Analysis",
            'x': 0.5,
            'font': {'size': 20, 'family': 'Inter', 'color': '#2c3e50'}
        },
        xaxis_title="Room Type",
        yaxis_title="Revenue per Booking ($)",
        height=500,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

# Create customer behavior analysis
def create_customer_behavior():
    if df.empty:
        return go.Figure()
    
    # Repeat guest analysis
    repeat_analysis = df.groupby('is_repeated').agg({
        'Booking_ID': 'count',
        'avg_price_per_room': 'mean',
        'is_canceled': 'mean',
        'no_of_special_requests': 'mean'
    }).reset_index()
    
    repeat_analysis['guest_type'] = repeat_analysis['is_repeated'].map({0: 'New Guest', 1: 'Repeat Guest'})
    repeat_analysis['cancellation_rate'] = repeat_analysis['is_canceled'] * 100
    
    fig = go.Figure()
    
    # Create subplot
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Booking Count', 'Average Price', 'Cancellation Rate', 'Special Requests'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    fig.add_trace(
        go.Bar(x=repeat_analysis['guest_type'], y=repeat_analysis['Booking_ID'], 
               name='Bookings', marker_color='#3498db'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=repeat_analysis['guest_type'], y=repeat_analysis['avg_price_per_room'], 
               name='Avg Price', marker_color='#e74c3c'),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Bar(x=repeat_analysis['guest_type'], y=repeat_analysis['cancellation_rate'], 
               name='Cancellation Rate', marker_color='#f39c12'),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Bar(x=repeat_analysis['guest_type'], y=repeat_analysis['no_of_special_requests'], 
               name='Special Requests', marker_color='#2ecc71'),
        row=2, col=2
    )
    
    fig.update_layout(
        title={
            'text': "Customer Behavior Analysis: New vs Repeat Guests",
            'x': 0.5,
            'font': {'size': 20, 'family': 'Inter', 'color': '#2c3e50'}
        },
        height=600,
        showlegend=False,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

# Page layout
layout = html.Div([
    # Page header
    html.Div([
        html.H1("📊 Advanced Analytics & Insights", className="page-title"),
        html.P("Deep dive into booking patterns, customer behavior, and revenue optimization opportunities", 
               className="page-subtitle")
    ], className="page-header"),
    
    # Analytics charts
    dbc.Row([
        dbc.Col([
            html.Div([
                dcc.Graph(figure=create_correlation_heatmap())
            ], className="chart-container")
        ], width=12),
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            html.Div([
                dcc.Graph(figure=create_lead_time_analysis())
            ], className="chart-container")
        ], width=6),
        
        dbc.Col([
            html.Div([
                dcc.Graph(figure=create_seasonal_analysis())
            ], className="chart-container")
        ], width=6),
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            html.Div([
                dcc.Graph(figure=create_market_analysis())
            ], className="chart-container")
        ], width=6),
        
        dbc.Col([
            html.Div([
                dcc.Graph(figure=create_room_profitability())
            ], className="chart-container")
        ], width=6),
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            html.Div([
                dcc.Graph(figure=create_customer_behavior())
            ], className="chart-container")
        ], width=12),
    ], className="mb-4"),
    
    # Key insights
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H4("🔍 Key Insights", style={'color': '#2c3e50', 'marginBottom': '20px'}),
                html.Ul([
                    html.Li("Lead time is strongly correlated with cancellation rates - longer lead times show higher cancellations"),
                    html.Li("Corporate bookings tend to have higher prices but lower cancellation rates"),
                    html.Li("Repeat guests show different behavior patterns compared to new guests"),
                    html.Li("Seasonal patterns show varying booking volumes and pricing strategies"),
                    html.Li("Room type profitability varies significantly based on price and occupancy")
                ], style={'fontSize': '16px', 'lineHeight': '1.6'})
            ], className="chart-container")
        ], width=12),
    ])
])
