import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np

# Import custom pages
from pages import home, analytics, reports, maddpg_demo_simple as maddpg_demo, forecasting_simple as forecasting, sentiment_analysis

# Initialize the Dash app with modern theme
app = dash.Dash(
    __name__, 
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css",
        "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap"
    ],
    suppress_callback_exceptions=True
)

# Custom CSS for modern UI
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Hotel Reservation Analytics Dashboard</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                font-family: 'Inter', sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                margin: 0;
                padding: 0;
            }
            .main-container {
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(10px);
                border-radius: 20px;
                margin: 20px;
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
                overflow: hidden;
            }
            .sidebar {
                background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
                color: white;
                padding: 0;
                height: 100vh;
                position: fixed;
                left: 0;
                top: 0;
                width: 280px;
                z-index: 1000;
                box-shadow: 2px 0 10px rgba(0, 0, 0, 0.1);
            }
            .sidebar .nav-link {
                color: rgba(255, 255, 255, 0.8);
                padding: 15px 25px;
                margin: 5px 15px;
                border-radius: 12px;
                transition: all 0.3s ease;
                font-weight: 500;
            }
            .sidebar .nav-link:hover {
                background: rgba(255, 255, 255, 0.1);
                color: white;
                transform: translateX(5px);
            }
            .sidebar .nav-link.active {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
            }
            .content-area {
                margin-left: 280px;
                padding: 30px;
                min-height: 100vh;
            }
            .card {
                border: none;
                border-radius: 16px;
                box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
                transition: transform 0.3s ease, box-shadow 0.3s ease;
                background: white;
            }
            .card:hover {
                transform: translateY(-5px);
                box-shadow: 0 15px 35px rgba(0, 0, 0, 0.15);
            }
            .metric-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border-radius: 16px;
                padding: 25px;
                text-align: center;
                margin-bottom: 20px;
            }
            .metric-value {
                font-size: 2.5rem;
                font-weight: 700;
                margin: 10px 0;
            }
            .metric-label {
                font-size: 1rem;
                opacity: 0.9;
                font-weight: 500;
            }
            .chart-container {
                background: white;
                border-radius: 16px;
                padding: 25px;
                margin-bottom: 25px;
                box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
            }
            .page-header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                border-radius: 16px;
                margin-bottom: 30px;
                text-align: center;
            }
            .page-title {
                font-size: 2.5rem;
                font-weight: 700;
                margin: 0;
            }
            .page-subtitle {
                font-size: 1.2rem;
                opacity: 0.9;
                margin: 10px 0 0 0;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# App layout
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    
    # Sidebar
    html.Div([
        html.Div([
            html.H3("🏨 Hotel Analytics", style={
                'color': 'white', 
                'textAlign': 'center', 
                'padding': '20px',
                'margin': '0',
                'fontWeight': '700',
                'fontSize': '1.5rem'
            }),
            html.Hr(style={'borderColor': 'rgba(255,255,255,0.2)', 'margin': '0 20px'}),
            
            dbc.Nav([
                dbc.NavLink([
                    html.I(className="fas fa-home", style={'marginRight': '10px'}),
                    "Dashboard Overview"
                ], href="/", active="exact", id="nav-home"),
                
                dbc.NavLink([
                    html.I(className="fas fa-chart-line", style={'marginRight': '10px'}),
                    "Analytics & Insights"
                ], href="/analytics", active="exact", id="nav-analytics"),
                
                dbc.NavLink([
                    html.I(className="fas fa-robot", style={'marginRight': '10px'}),
                    "Multi-Agent DRL"
                ], href="/maddpg", active="exact", id="nav-maddpg"),
                
                dbc.NavLink([
                    html.I(className="fas fa-brain", style={'marginRight': '10px'}),
                    "AI Forecasting"
                ], href="/forecasting", active="exact", id="nav-forecasting"),
                
                dbc.NavLink([
                    html.I(className="fas fa-heart", style={'marginRight': '10px'}),
                    "Sentiment Analysis"
                ], href="/sentiment", active="exact", id="nav-sentiment"),
                
                dbc.NavLink([
                    html.I(className="fas fa-file-alt", style={'marginRight': '10px'}),
                    "Reports"
                ], href="/reports", active="exact", id="nav-reports"),
            ], vertical=True, pills=True, className="mt-4"),
            
            html.Div([
                html.Hr(style={'borderColor': 'rgba(255,255,255,0.2)'}),
                html.P("Advanced Hotel Analytics", style={
                    'color': 'rgba(255,255,255,0.7)',
                    'textAlign': 'center',
                    'fontSize': '0.9rem',
                    'margin': '20px 0 0 0'
                })
            ], style={'position': 'absolute', 'bottom': '20px', 'left': '0', 'right': '0'})
            
        ], style={'height': '100%', 'display': 'flex', 'flexDirection': 'column'})
    ], className="sidebar"),
    
    # Main content area
    html.Div(id='page-content', className="content-area")
])

# Callback to update page content
@callback(Output('page-content', 'children'), [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/':
        return home.layout
    elif pathname == '/analytics':
        return analytics.layout
    elif pathname == '/maddpg':
        return maddpg_demo.layout
    elif pathname == '/forecasting':
        return forecasting.layout
    elif pathname == '/sentiment':
        return sentiment_analysis.layout
    elif pathname == '/reports':
        return reports.layout
    else:
        return home.layout

if __name__ == '__main__':
    app.run_server(debug=True, host='', port=8050)
