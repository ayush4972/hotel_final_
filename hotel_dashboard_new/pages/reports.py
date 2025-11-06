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
    df = pd.read_csv("Hotel Reservations.csv", encoding="latin1")
    print("✅ Loaded Hotel Reservations.csv for reports")
    
    # Create date column with error handling
    df['arrival_date'] = pd.to_datetime(
        df[['arrival_year', 'arrival_month', 'arrival_date']].rename(
            columns={'arrival_year': 'year', 'arrival_month': 'month', 'arrival_date': 'day'}
        ), errors='coerce'
    )
    
    # Fill any NaT values
    df['arrival_date'] = df['arrival_date'].fillna(pd.Timestamp('2018-01-01'))
    
    # Get latest date for filtering
    latest_date = df['arrival_date'].max()
    
    print(f"📊 Reports data prepared: {len(df)} records")
    print(f"📅 Latest booking date: {latest_date}")
    
except Exception as error:
    print(f"❌ Error loading data: {error}")
    df = pd.DataFrame()
    latest_date = datetime.now()

# Function to fetch hotel data based on report type
def fetch_hotel_data(report_type):
    if df.empty:
        return pd.DataFrame()
    
    if report_type == 'daily':
        # Last 1 day
        start_date = latest_date - pd.Timedelta(days=1)
        filtered_data = df[df['arrival_date'] >= start_date]
    elif report_type == 'weekly':
        # Last 7 days
        start_date = latest_date - pd.Timedelta(days=7)
        filtered_data = df[df['arrival_date'] >= start_date]
    elif report_type == 'monthly':
        # Last 30 days
        start_date = latest_date - pd.Timedelta(days=30)
        filtered_data = df[df['arrival_date'] >= start_date]
    else:
        filtered_data = df
    
    # Select relevant columns for the report
    report_columns = [
        'Booking_ID', 'arrival_date', 'no_of_adults', 'no_of_children',
        'room_type_reserved', 'avg_price_per_room', 'market_segment_type',
        'booking_status', 'lead_time', 'no_of_special_requests'
    ]
    
    report_data = filtered_data[report_columns].copy()
    
    # Rename columns for display
    report_data.columns = [
        'Booking ID', 'Arrival Date', 'Adults', 'Children',
        'Room Type', 'Price', 'Market Segment',
        'Status', 'Lead Time', 'Special Requests'
    ]
    
    return report_data

# Create data table
def create_data_table(data, title):
    if data.empty:
        return html.Div([
            html.H4(title, style={'color': '#2c3e50', 'marginBottom': '20px'}),
            html.P("No data available for the selected period.", style={'color': '#666', 'fontStyle': 'italic'})
        ])
    
    # Create table with conditional styling
    table_rows = []
    
    # Header
    header_row = html.Tr([
        html.Th(col, style={'backgroundColor': '#667eea', 'color': 'white', 'padding': '10px'})
        for col in data.columns
    ])
    table_rows.append(header_row)
    
    # Data rows with conditional styling
    for _, row in data.iterrows():
        row_style = {}
        if row['Status'] == 'Canceled':
            row_style['backgroundColor'] = '#fadbd8'  # Light red
        elif row['Status'] == 'Not_Canceled':
            row_style['backgroundColor'] = '#d5f4e6'  # Light green
        
        table_row = html.Tr([
            html.Td(str(row[col]), style={'padding': '8px', **row_style})
            for col in data.columns
        ])
        table_rows.append(table_row)
    
    return html.Div([
        html.H4(title, style={'color': '#2c3e50', 'marginBottom': '20px'}),
        html.Table(
            table_rows,
            style={'width': '100%', 'borderCollapse': 'collapse', 'border': '1px solid #ddd'}
        )
    ])

# Create summary statistics
def create_summary_stats(data, report_type):
    if data.empty:
        return html.Div()
    
    total_bookings = len(data)
    total_revenue = data['Price'].sum()
    avg_price = data['Price'].mean()
    cancellation_rate = (data['Status'] == 'Canceled').mean() * 100
    avg_lead_time = data['Lead Time'].mean()
    
    return html.Div([
        html.H4(f"{report_type.title()} Report Summary", style={'color': '#2c3e50', 'marginBottom': '20px'}),
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.I(className="fas fa-calendar-check", style={'fontSize': '2rem', 'color': '#667eea'}),
                    html.H3(f"{total_bookings:,}", style={'margin': '10px 0', 'color': '#2c3e50'}),
                    html.P("Total Bookings", style={'margin': '0', 'color': '#666'})
                ], style={'textAlign': 'center', 'padding': '20px'})
            ], width=2),
            
            dbc.Col([
                html.Div([
                    html.I(className="fas fa-dollar-sign", style={'fontSize': '2rem', 'color': '#2ecc71'}),
                    html.H3(f"${total_revenue:,.0f}", style={'margin': '10px 0', 'color': '#2c3e50'}),
                    html.P("Total Revenue", style={'margin': '0', 'color': '#666'})
                ], style={'textAlign': 'center', 'padding': '20px'})
            ], width=2),
            
            dbc.Col([
                html.Div([
                    html.I(className="fas fa-bed", style={'fontSize': '2rem', 'color': '#f39c12'}),
                    html.H3(f"${avg_price:.0f}", style={'margin': '10px 0', 'color': '#2c3e50'}),
                    html.P("Avg Price", style={'margin': '0', 'color': '#666'})
                ], style={'textAlign': 'center', 'padding': '20px'})
            ], width=2),
            
            dbc.Col([
                html.Div([
                    html.I(className="fas fa-percentage", style={'fontSize': '2rem', 'color': '#e74c3c'}),
                    html.H3(f"{cancellation_rate:.1f}%", style={'margin': '10px 0', 'color': '#2c3e50'}),
                    html.P("Cancellation Rate", style={'margin': '0', 'color': '#666'})
                ], style={'textAlign': 'center', 'padding': '20px'})
            ], width=2),
            
            dbc.Col([
                html.Div([
                    html.I(className="fas fa-clock", style={'fontSize': '2rem', 'color': '#9b59b6'}),
                    html.H3(f"{avg_lead_time:.0f}", style={'margin': '10px 0', 'color': '#2c3e50'}),
                    html.P("Avg Lead Time (days)", style={'margin': '0', 'color': '#666'})
                ], style={'textAlign': 'center', 'padding': '20px'})
            ], width=2),
        ])
    ], className="chart-container")

# Create booking status chart
def create_booking_status_chart(data):
    if data.empty:
        return go.Figure()
    
    status_counts = data['Status'].value_counts()
    colors = ['#2ecc71', '#e74c3c']
    
    fig = go.Figure(data=[go.Pie(
        labels=status_counts.index,
        values=status_counts.values,
        hole=0.4,
        marker_colors=colors,
        textinfo='label+percent+value'
    )])
    
    fig.update_layout(
        title={
            'text': "Booking Status Distribution",
            'x': 0.5,
            'font': {'size': 18, 'family': 'Inter', 'color': '#2c3e50'}
        },
        height=400,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

# Create market segment chart
def create_market_segment_chart(data):
    if data.empty:
        return go.Figure()
    
    market_counts = data['Market Segment'].value_counts()
    
    fig = px.bar(
        x=market_counts.index,
        y=market_counts.values,
        title="Bookings by Market Segment",
        color=market_counts.values,
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        title={
            'text': "Bookings by Market Segment",
            'x': 0.5,
            'font': {'size': 18, 'family': 'Inter', 'color': '#2c3e50'}
        },
        xaxis_title="Market Segment",
        yaxis_title="Number of Bookings",
        height=400,
        margin=dict(l=20, r=20, t=60, b=20),
        template='plotly_white'
    )
    
    return fig

# Create room type chart
def create_room_type_chart(data):
    if data.empty:
        return go.Figure()
    
    room_counts = data['Room Type'].value_counts()
    
    fig = px.pie(
        values=room_counts.values,
        names=room_counts.index,
        title="Room Type Distribution"
    )
    
    fig.update_layout(
        title={
            'text': "Room Type Distribution",
            'x': 0.5,
            'font': {'size': 18, 'family': 'Inter', 'color': '#2c3e50'}
        },
        height=400,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

# Page layout
layout = html.Div([
    # Page header
    html.Div([
        html.H1("📋 Hotel Reservation Reports", className="page-title"),
        html.P("Comprehensive reports and analytics for hotel booking management and performance tracking", 
               className="page-subtitle")
    ], className="page-header"),
    
    # Report type selection
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H4("📊 Report Selection", style={'color': '#2c3e50', 'marginBottom': '20px'}),
                dbc.ButtonGroup([
                    dbc.Button("📅 Daily Report", id="daily-btn", color="primary", outline=True),
                    dbc.Button("📆 Weekly Report", id="weekly-btn", color="primary", outline=True),
                    dbc.Button("📈 Monthly Report", id="monthly-btn", color="primary", outline=True),
                    dbc.Button("📊 Full Report", id="full-btn", color="primary", outline=True),
                ], className="w-100")
            ], className="chart-container")
        ], width=12),
    ], className="mb-4"),
    
    # Report content
    html.Div(id="report-content"),
    
    # Footer note
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Hr(),
                html.P("📝 NOTE: Hotel booking reports are generated automatically based on arrival dates", 
                       style={'color': '#666', 'fontStyle': 'italic', 'textAlign': 'center', 'margin': '20px 0'})
            ])
        ], width=12),
    ])
])

# Callbacks for report generation
@dash.callback(
    Output("report-content", "children"),
    [Input("daily-btn", "n_clicks"),
     Input("weekly-btn", "n_clicks"),
     Input("monthly-btn", "n_clicks"),
     Input("full-btn", "n_clicks")]
)
def generate_report(daily_clicks, weekly_clicks, monthly_clicks, full_clicks):
    ctx = dash.callback_context
    
    if not ctx.triggered:
        # Default to weekly report
        report_type = 'weekly'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == 'daily-btn':
            report_type = 'daily'
        elif button_id == 'weekly-btn':
            report_type = 'weekly'
        elif button_id == 'monthly-btn':
            report_type = 'monthly'
        else:
            report_type = 'full'
    
    # Fetch data
    data = fetch_hotel_data(report_type)
    
    # Create report content
    content = []
    
    # Summary statistics
    content.append(create_summary_stats(data, report_type))
    
    # Charts
    content.append(
        dbc.Row([
            dbc.Col([
                html.Div([
                    dcc.Graph(figure=create_booking_status_chart(data))
                ], className="chart-container")
            ], width=4),
            
            dbc.Col([
                html.Div([
                    dcc.Graph(figure=create_market_segment_chart(data))
                ], className="chart-container")
            ], width=4),
            
            dbc.Col([
                html.Div([
                    dcc.Graph(figure=create_room_type_chart(data))
                ], className="chart-container")
            ], width=4),
        ], className="mb-4")
    )
    
    # Data table
    content.append(
        dbc.Row([
            dbc.Col([
                html.Div([
                    create_data_table(data, f"{report_type.title()} Booking Details")
                ], className="chart-container")
            ], width=12),
        ], className="mb-4")
    )
    
    return content
