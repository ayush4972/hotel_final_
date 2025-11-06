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
    
    # Calculate key metrics
    total_bookings = len(df)
    total_revenue = df['avg_price_per_room'].sum()
    cancellation_rate = (df['booking_status'] == 'Canceled').mean() * 100
    avg_price = df['avg_price_per_room'].mean()
    
    # Create daily booking data
    daily_data = df.groupby('arrival_date').agg({
        'Booking_ID': 'count',
        'avg_price_per_room': 'sum'
    }).reset_index()
    daily_data.columns = ['date', 'bookings', 'revenue']
    daily_data['date'] = pd.to_datetime(daily_data['date'])
    
    # Get last 30 days
    latest_date = daily_data['date'].max()
    cutoff_date = latest_date - pd.Timedelta(days=30)
    recent_data = daily_data[daily_data['date'] >= cutoff_date]
    
    # Booking status distribution
    status_data = df['booking_status'].value_counts().reset_index()
    status_data.columns = ['status', 'count']
    
    # Room type distribution
    room_data = df['room_type_reserved'].value_counts().reset_index()
    room_data.columns = ['room_type', 'count']
    
    # Market segment distribution
    market_data = df['market_segment_type'].value_counts().reset_index()
    market_data.columns = ['market_segment', 'count']
    
    print(f"📊 Processed {total_bookings} hotel reservations")
    print(f"💰 Total revenue: ${total_revenue:,.2f}")
    print(f"📈 Cancellation rate: {cancellation_rate:.1f}%")
    
except Exception as error:
    print(f"❌ Error loading data: {error}")
    # Create empty dataframes as fallback
    df = pd.DataFrame()
    total_bookings = 0
    total_revenue = 0
    cancellation_rate = 0
    avg_price = 0
    recent_data = pd.DataFrame()
    status_data = pd.DataFrame()
    room_data = pd.DataFrame()
    market_data = pd.DataFrame()

# Create gauge charts
def create_gauge(value, title, max_val=None):
    if max_val is None:
        max_val = value * 1.2 if value > 0 else 100
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        title={'text': title},
        gauge={
            'axis': {'range': [None, max_val]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, max_val * 0.5], 'color': "lightgray"},
                {'range': [max_val * 0.5, max_val], 'color': "gray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        font={'color': "darkblue", 'family': "Inter"},
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

# Create revenue trend chart
def create_revenue_chart():
    if recent_data.empty:
        return go.Figure()
    
    fig = go.Figure()
    
    # Add bar chart
    fig.add_trace(go.Bar(
        x=recent_data['date'],
        y=recent_data['revenue'],
        name='Daily Revenue',
        marker_color='rgba(102, 126, 234, 0.8)',
        marker_line=dict(color='rgba(102, 126, 234, 1.0)', width=1)
    ))
    
    # Add trend line
    fig.add_trace(go.Scatter(
        x=recent_data['date'],
        y=recent_data['revenue'],
        mode='lines+markers',
        name='Trend',
        line=dict(color='#667eea', width=3),
        marker=dict(size=6, color='#667eea')
    ))
    
    fig.update_layout(
        title={
            'text': "Daily Revenue Trend (Last 30 Days)",
            'x': 0.5,
            'font': {'size': 20, 'family': 'Inter', 'color': '#2c3e50'}
        },
        xaxis_title="Date",
        yaxis_title="Revenue ($)",
        hovermode='x unified',
        template='plotly_white',
        height=400,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

# Create booking status pie chart
def create_status_chart():
    if status_data.empty:
        return go.Figure()
    
    colors = ['#2ecc71', '#e74c3c', '#f39c12']
    
    fig = go.Figure(data=[go.Pie(
        labels=status_data['status'],
        values=status_data['count'],
        hole=0.4,
        marker_colors=colors,
        textinfo='label+percent',
        textfont_size=14
    )])
    
    fig.update_layout(
        title={
            'text': "Booking Status Distribution",
            'x': 0.5,
            'font': {'size': 18, 'family': 'Inter', 'color': '#2c3e50'}
        },
        height=350,
        margin=dict(l=20, r=20, t=60, b=20),
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.01
        )
    )
    
    return fig

# Create room type chart
def create_room_chart():
    if room_data.empty:
        return go.Figure()
    
    fig = px.bar(
        room_data,
        x='room_type',
        y='count',
        title="Room Type Popularity",
        color='count',
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        title={
            'text': "Room Type Popularity",
            'x': 0.5,
            'font': {'size': 18, 'family': 'Inter', 'color': '#2c3e50'}
        },
        xaxis_title="Room Type",
        yaxis_title="Number of Bookings",
        template='plotly_white',
        height=350,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

# Create market segment chart
def create_market_chart():
    if market_data.empty:
        return go.Figure()
    
    fig = px.pie(
        market_data,
        values='count',
        names='market_segment',
        title="Market Segment Distribution",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_layout(
        title={
            'text': "Market Segment Distribution",
            'x': 0.5,
            'font': {'size': 18, 'family': 'Inter', 'color': '#2c3e50'}
        },
        height=350,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

# Page layout
layout = html.Div([
    # Page header
    html.Div([
        html.H1("🏨 Hotel Reservation Analytics Dashboard", className="page-title"),
        html.P("Comprehensive insights into hotel booking patterns, revenue optimization, and customer behavior", 
               className="page-subtitle")
    ], className="page-header"),
    
    # Key metrics row
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Div([
                    html.I(className="fas fa-calendar-check", style={'fontSize': '2rem', 'marginBottom': '10px'}),
                    html.H2(f"{total_bookings:,}", className="metric-value"),
                    html.P("Total Bookings", className="metric-label")
                ])
            ], className="metric-card")
        ], width=3),
        
        dbc.Col([
            html.Div([
                html.Div([
                    html.I(className="fas fa-dollar-sign", style={'fontSize': '2rem', 'marginBottom': '10px'}),
                    html.H2(f"${total_revenue:,.0f}", className="metric-value"),
                    html.P("Total Revenue", className="metric-label")
                ])
            ], className="metric-card")
        ], width=3),
        
        dbc.Col([
            html.Div([
                html.Div([
                    html.I(className="fas fa-percentage", style={'fontSize': '2rem', 'marginBottom': '10px'}),
                    html.H2(f"{cancellation_rate:.1f}%", className="metric-value"),
                    html.P("Cancellation Rate", className="metric-label")
                ])
            ], className="metric-card")
        ], width=3),
        
        dbc.Col([
            html.Div([
                html.Div([
                    html.I(className="fas fa-bed", style={'fontSize': '2rem', 'marginBottom': '10px'}),
                    html.H2(f"${avg_price:.0f}", className="metric-value"),
                    html.P("Average Price/Room", className="metric-label")
                ])
            ], className="metric-card")
        ], width=3),
    ], className="mb-4"),
    
    # Charts row 1
    dbc.Row([
        dbc.Col([
            html.Div([
                dcc.Graph(figure=create_revenue_chart())
            ], className="chart-container")
        ], width=8),
        
        dbc.Col([
            html.Div([
                dcc.Graph(figure=create_status_chart())
            ], className="chart-container")
        ], width=4),
    ], className="mb-4"),
    
    # Charts row 2
    dbc.Row([
        dbc.Col([
            html.Div([
                dcc.Graph(figure=create_room_chart())
            ], className="chart-container")
        ], width=6),
        
        dbc.Col([
            html.Div([
                dcc.Graph(figure=create_market_chart())
            ], className="chart-container")
        ], width=6),
    ], className="mb-4"),
    
    # Quick stats cards
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H4("📊 Data Overview", style={'color': '#2c3e50', 'marginBottom': '20px'}),
                html.P(f"• Dataset contains {total_bookings:,} hotel reservations"),
                html.P(f"• Date range: {df['arrival_date'].min().strftime('%Y-%m-%d')} to {df['arrival_date'].max().strftime('%Y-%m-%d')}"),
                html.P(f"• Average booking value: ${avg_price:.2f}"),
                html.P(f"• Most popular room type: {room_data.iloc[0]['room_type'] if not room_data.empty else 'N/A'}"),
                html.P(f"• Top market segment: {market_data.iloc[0]['market_segment'] if not market_data.empty else 'N/A'}")
            ], className="chart-container")
        ], width=12),
    ])
])
