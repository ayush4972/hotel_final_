import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
import os
from datetime import timedelta

# Multi-Agent DRL (MADDPG) Implementation for Hotel SME
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Hotel SME Environment ---
class HotelSMEEnv:
    def __init__(self, csv_path, seed=0):
        self.df = pd.read_csv(csv_path)
        # Combine year, month, day into a string column safely
        date_strs = self.df[['arrival_year', 'arrival_month', 'arrival_date']].astype(str).agg('-'.join, axis=1)
        
        # Parse dates with errors='coerce' to handle invalid dates (e.g., Feb 30)
        self.df['arrival_date'] = pd.to_datetime(date_strs, errors='coerce')
        
        # Optional: Drop rows with invalid dates (if you want clean data)
        initial_count = len(self.df)
        self.df = self.df.dropna(subset=['arrival_date']).reset_index(drop=True)
        dropped = initial_count - len(self.df)
        if dropped > 0:
            print(f"[WARNING] Dropped {dropped} rows with invalid dates (e.g., Feb 30).")
        
        # Sort by arrival_date
        self.df = self.df.sort_values('arrival_date').reset_index(drop=True)
        self.rng = np.random.RandomState(seed)
        
        # Precompute features
        self.df['feedback_score'] = self.df['Customer_Feedback'].apply(self._sentiment_score)
        self.df['total_guests'] = self.df['no_of_adults'] + self.df['no_of_children']
        self.df['is_canceled'] = (self.df['booking_status'] == 'Canceled').astype(int)
        
        # Assumed SME hotel size and costs
        self.total_rooms = 50  # Number of rooms in the SME hotel
        self.daily_expense_per_room = 120  # Estimated daily cost per room (staff, cleaning, utilities)
        self.daily_expenses = self.daily_expense_per_room * self.total_rooms
        
        self.reset()

    def _sentiment_score(self, text):
        """Simple keyword-based sentiment analysis on Customer_Feedback."""
        if pd.isna(text): 
            return 0
        text = text.lower()
        if any(w in text for w in ['excellent', 'highly satisfied', 'very comfortable', 'good experience']):
            return 1
        elif any(w in text for w in ['friendly staff', 'good experience', 'room cleanliness issue']):
            return 0.5 if 'good' in text or 'friendly' in text else -0.5
        elif 'average stay' in text:
            return 0
        elif any(w in text for w in ['poor service', 'food quality could be better', 'would not recommend', 'room cleanliness issue']):
            return -1
        else:
            return 0

    def reset(self):
        """Reset the environment to initial state."""
        self.current_day = 0
        self.cash = 10000.0  # Initial capital
        self.occupied_rooms = 0
        self.bookings_history = []  # Track all bookings made
        self.daily_expenses = self.daily_expense_per_room * self.total_rooms
        
        # Return the initial state
        return self._get_state()

    def _get_state(self):
        """Generate the 12-dimensional state vector based on current day and historical data."""
        today = self.df['arrival_date'].min() + timedelta(days=self.current_day)
        
        # Filter bookings arriving in the next 7 days (for forecasting)
        next_7_days = self.df[
            (self.df['arrival_date'] >= today) &
            (self.df['arrival_date'] < today + timedelta(days=7)) &
            (self.df['booking_status'] == 'Not_Canceled')
        ]
        
        # Filter bookings from the past 7 days (for historical metrics)
        past_7_days = self.df[
            (self.df['arrival_date'] >= today - timedelta(days=7)) &
            (self.df['arrival_date'] < today)
        ]
        
        # Calculate metrics
        bookings_next_7 = len(next_7_days)
        cancellation_rate = past_7_days['is_canceled'].mean() if len(past_7_days) > 0 else 0.1
        avg_price = past_7_days['avg_price_per_room'].mean() if len(past_7_days) > 0 else 80
        occupancy_rate = self.occupied_rooms / self.total_rooms if self.total_rooms > 0 else 0
        corporate_ratio = (past_7_days['market_segment_type'] == 'Corporate').mean() if len(past_7_days) > 0 else 0.1
        repeat_guest_rate = past_7_days['repeated_guest'].mean() if len(past_7_days) > 0 else 0.1
        lead_time_mean = past_7_days['lead_time'].mean() if len(past_7_days) > 0 else 30
        demand_forecast = past_7_days['total_guests'].mean() if len(past_7_days) > 0 else 60
        special_requests_rate = past_7_days['no_of_special_requests'].mean() if len(past_7_days) > 0 else 0.5
        feedback_score = past_7_days['feedback_score'].mean() if len(past_7_days) > 0 else 0
        
        # State vector: 12 dimensions
        state = np.array([
            # Liquidity Agent Inputs
            self.cash,
            bookings_next_7 * avg_price,  # Expected cash inflow from confirmed bookings
            cancellation_rate,
            self.daily_expenses,

            # Investment Agent Inputs
            occupancy_rate,
            avg_price,
            corporate_ratio,
            repeat_guest_rate,

            # Procurement Agent Inputs
            lead_time_mean,
            demand_forecast,
            special_requests_rate,
            feedback_score
        ], dtype=np.float32)

        return state

    def step(self, actions):
        """Execute one time step within the environment.
        Actions: [liq_a, inv_a, prod_a] - normalized between -1 and 1"""
        liq_a, inv_a, prod_a = actions

        # --- Liquidity Agent: Borrow/Repay Cash ---
        if liq_a > 0.2:
            # Borrow up to 20% of max borrow limit (e.g., $5000), scaled by action
            borrow = liq_a * 5000.0 * 0.2
            self.cash += borrow
            borrow_cost = 0.01 * borrow  # 1% daily interest rate
        elif liq_a < -0.2:
            # Repay up to 10% of current cash balance
            repay = (-liq_a) * min(self.cash, 5000.0 * 0.1)
            self.cash -= repay
            borrow_cost = 0.0
        else:
            borrow_cost = 0.0

        # --- Investment Agent: Buy/Sell Assets ---
        if inv_a > 0.2:
            # Invest surplus cash into marketing or amenities to boost future revenue
            invest_amount = inv_a * min(self.cash, 10000.0 * 0.05)  # Max 5% of asset allocation
            self.cash -= invest_amount
        elif inv_a < -0.2:
            # Sell assets (e.g., sell off underutilized equipment or reduce maintenance budget)
            sell_amount = (-inv_a) * (self.cash * 0.05)
            self.cash += sell_amount

        # --- Procurement Agent: Order Inventory (Food, Amenities, etc.) ---
        today = self.df['arrival_date'].min() + timedelta(days=self.current_day)
        past_7_days = self.df[
            (self.df['arrival_date'] >= today - timedelta(days=7)) &
            (self.df['arrival_date'] < today)
        ]
        demand_forecast = past_7_days['total_guests'].mean() if len(past_7_days) > 0 else 60
        special_requests_rate = past_7_days['no_of_special_requests'].mean() if len(past_7_days) > 0 else 0.5

        if prod_a > 0.2:
            # Order more inventory based on forecasted demand and special requests
            order_value = prod_a * (demand_forecast * 10.0)  # Scale order by guest count & action
            # Assume cost per unit of inventory is $5
            cost = order_value * 5.0
            if cost <= self.cash:
                self.cash -= cost
            else:
                affordable = self.cash // 5.0
                self.cash -= affordable * 5.0

        # --- Simulate Daily Operations: Bookings Arrive and Payments Are Made ---
        today_bookings = self.df[self.df['arrival_date'] == today]
        total_revenue_today = 0.0
        new_occupancy = 0

        for _, booking in today_bookings.iterrows():
            if booking['booking_status'] == 'Not_Canceled':
                # Payment received
                total_revenue_today += booking['avg_price_per_room']
                new_occupancy += 1  # One room occupied

        # Update occupancy and cash
        self.occupied_rooms = new_occupancy
        self.cash += total_revenue_today

        # Subtract fixed daily expenses
        self.cash -= self.daily_expenses

        # --- Calculate Reward (AFR - Adaptive Financial Reward) ---
        cancellation_rate = past_7_days['is_canceled'].mean() if len(past_7_days) > 0 else 0.1
        occupancy_rate = self.occupied_rooms / self.total_rooms if self.total_rooms > 0 else 0
        
        # Profit: Revenue - Borrow Cost - Daily Expenses (already subtracted)
        profit = total_revenue_today - borrow_cost
        # Liquidity Stability: Ensure cash doesn't fall below critical level ($2000)
        liquidity_stability = max(0.0, 1.0 - max(0.0, 2000.0 - self.cash) / 2000.0)
        # Risk Exposure: High cancellation rate or low occupancy
        risk_exposure = float((cancellation_rate > 0.3) or (occupancy_rate < 0.4))
        
        alpha, beta, gamma = 1.0, 0.5, 5.0
        afr = alpha * profit + beta * liquidity_stability - gamma * risk_exposure

        # Advance to next day
        self.current_day += 1
        done = (self.current_day >= 100)  # Episode ends after 100 days

        # Get next state
        next_obs = self._get_state()
        
        # For 3 agents, replicate the same reward and observation
        obs_list = [next_obs.copy() for _ in range(3)]
        rewards = [afr for _ in range(3)]

        return obs_list, rewards, done, {}

# --- Replay Buffer ---
Transition = namedtuple('Transition', ('obs', 'actions', 'rewards', 'next_obs', 'dones'))
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    def push(self, *args):
        self.buffer.append(Transition(*args))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))
    def __len__(self):
        return len(self.buffer)

# --- Networks ---
def mlp(input_dim, output_dim, hidden=(64,64), activation=nn.ReLU):
    layers = []
    last = input_dim
    for h in hidden:
        layers.append(nn.Linear(last, h))
        layers.append(activation())
        last = h
    layers.append(nn.Linear(last, output_dim))
    return nn.Sequential(*layers)

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = mlp(obs_dim, act_dim, hidden=(64,64))
    def forward(self, x):
        return torch.tanh(self.net(x))

class Critic(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = mlp(input_dim, 1, hidden=(128,128))
    def forward(self, x):
        return self.net(x)

# --- MADDPG Simplified ---
class MADDPG:
    def __init__(self, obs_dim, act_dim, n_agents, lr=1e-3):
        self.n = n_agents
        self.actors = [Actor(obs_dim, act_dim).to(device) for _ in range(n_agents)]
        self.target_actors = [Actor(obs_dim, act_dim).to(device) for _ in range(n_agents)]
        self.critics = [Critic(obs_dim*n_agents + act_dim*n_agents).to(device) for _ in range(n_agents)]
        self.target_critics = [Critic(obs_dim*n_agents + act_dim*n_agents).to(device) for _ in range(n_agents)]
        self.actor_opts = [optim.Adam(a.parameters(), lr=lr) for a in self.actors]
        self.critic_opts = [optim.Adam(c.parameters(), lr=lr) for c in self.critics]
        for i in range(n_agents):
            self.target_actors[i].load_state_dict(self.actors[i].state_dict())
            self.target_critics[i].load_state_dict(self.critics[i].state_dict())

    def act(self, obs, noise=0.0):
        actions = []
        for i,a in enumerate(self.actors):
            x = torch.tensor(obs[i], dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                act = a(x).cpu().numpy().flatten()
            act = act + noise * np.random.randn(*act.shape)
            actions.append(np.clip(act, -1.0, 1.0))
        return actions

    def update(self, buffer, batch_size=64, gamma=0.95, tau=0.01):
        if len(buffer) < batch_size:
            return 0.0, 0.0, 0.0  # Return zero losses if buffer too small
        
        trans = buffer.sample(batch_size)
        batch_obs = []
        batch_next_obs = []
        for agent_idx in range(self.n):
            batch_obs.append(np.stack([t[agent_idx] for t in trans.obs]))
            batch_next_obs.append(np.stack([t[agent_idx] for t in trans.next_obs]))
        batch_actions = np.stack([t for t in trans.actions])
        batch_rewards = np.stack([t for t in trans.rewards])

        batch_obs_t = [torch.tensor(x, dtype=torch.float32, device=device) for x in batch_obs]
        batch_next_obs_t = [torch.tensor(x, dtype=torch.float32, device=device) for x in batch_next_obs]
        batch_actions_t = torch.tensor(batch_actions, dtype=torch.float32, device=device)
        batch_rewards_t = torch.tensor(batch_rewards, dtype=torch.float32, device=device)

        B = len(batch_actions)
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_q_value = 0.0
        
        for agent_idx in range(self.n):
            obs_cat = torch.cat([o for o in batch_obs_t], dim=1)
            acts_cat = batch_actions_t.view(B, -1)
            critic_input = torch.cat([obs_cat, acts_cat], dim=1)
            Q = self.critics[agent_idx](critic_input).squeeze()

            next_acts = []
            for j in range(self.n):
                with torch.no_grad():
                    na = self.target_actors[j](batch_next_obs_t[j])
                    next_acts.append(na)
            next_acts_cat = torch.cat(next_acts, dim=1)
            next_obs_cat = torch.cat([o for o in batch_next_obs_t], dim=1)
            critic_target_input = torch.cat([next_obs_cat, next_acts_cat], dim=1)
            with torch.no_grad():
                Q_target = self.target_critics[agent_idx](critic_target_input).squeeze()

            y = batch_rewards_t[:, agent_idx] + gamma * Q_target
            critic_loss = nn.MSELoss()(Q, y.detach())
            self.critic_opts[agent_idx].zero_grad()
            critic_loss.backward()
            self.critic_opts[agent_idx].step()

            cur_acts = []
            for j in range(self.n):
                if j == agent_idx:
                    cur_acts.append(self.actors[j](batch_obs_t[j]))
                else:
                    with torch.no_grad():
                        cur_acts.append(self.actors[j](batch_obs_t[j]))
            cur_acts_cat = torch.cat(cur_acts, dim=1)
            obs_cat = torch.cat([o for o in batch_obs_t], dim=1)
            critic_input_for_actor = torch.cat([obs_cat, cur_acts_cat], dim=1)
            actor_loss = -self.critics[agent_idx](critic_input_for_actor).mean()
            self.actor_opts[agent_idx].zero_grad()
            actor_loss.backward()
            self.actor_opts[agent_idx].step()

            # Soft update
            for target_param, param in zip(self.target_critics[agent_idx].parameters(), self.critics[agent_idx].parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
            for target_param, param in zip(self.target_actors[agent_idx].parameters(), self.actors[agent_idx].parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
            
            # Track losses and Q-values
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_q_value += Q.mean().item()
        
        # Return average losses and Q-value across all agents
        return total_actor_loss / self.n, total_critic_loss / self.n, total_q_value / self.n

# Global variables for demo
env = None
agent = None
training_data = {
    'episodes': [], 
    'rewards': [], 
    'actor_losses': [], 
    'critic_losses': [],
    'q_values': [],
    'cash_history': [],
    'occupancy_history': []
}

def run_maddpg_demo(episodes=10, seed=42):
    global env, agent, training_data
    
    try:
        env = HotelSMEEnv("Hotel Reservations.csv", seed=seed)
        obs_dim, act_dim, n_agents = 12, 1, 3
        agent = MADDPG(obs_dim, act_dim, n_agents, lr=1e-3)
        buffer = ReplayBuffer(capacity=5000)
        
        training_data = {
            'episodes': [], 
            'rewards': [], 
            'actor_losses': [], 
            'critic_losses': [],
            'q_values': [],
            'cash_history': [],
            'occupancy_history': []
        }
        
        # Track per-episode metrics
        ep_actor_losses = []
        ep_critic_losses = []
        ep_q_values = []
        
        for ep in range(episodes):
            state = env.reset()
            obs_list = [state.copy() for _ in range(n_agents)]
            ep_reward = 0.0
            done = False
            steps = 0
            
            # Reset episode metrics
            ep_actor_losses = []
            ep_critic_losses = []
            ep_q_values = []

            while not done and steps < 50:  # Limit steps for demo
                actions = agent.act(obs_list, noise=0.1)
                act_vals = [float(a[0]) for a in actions]
                next_obs, rewards, done, _ = env.step(act_vals)
                buffer.push(obs_list, np.array(act_vals).reshape(n_agents, act_dim), rewards, next_obs, float(done))
                
                # Update and collect losses
                actor_loss, critic_loss, q_value = agent.update(buffer, batch_size=32)
                if actor_loss > 0:  # Only track if update happened
                    ep_actor_losses.append(actor_loss)
                    ep_critic_losses.append(critic_loss)
                    ep_q_values.append(q_value)

                obs_list = next_obs
                ep_reward += np.mean(rewards)
                steps += 1

            avg_r = ep_reward/steps if steps > 0 else 0
            avg_actor_loss = np.mean(ep_actor_losses) if ep_actor_losses else 0.0
            avg_critic_loss = np.mean(ep_critic_losses) if ep_critic_losses else 0.0
            avg_q_value = np.mean(ep_q_values) if ep_q_values else 0.0
            
            training_data['episodes'].append(ep + 1)
            training_data['rewards'].append(avg_r)
            training_data['actor_losses'].append(avg_actor_loss)
            training_data['critic_losses'].append(avg_critic_loss)
            training_data['q_values'].append(avg_q_value)
            training_data['cash_history'].append(env.cash)
            training_data['occupancy_history'].append(env.occupied_rooms / env.total_rooms if env.total_rooms > 0 else 0)
            
        return True, f"Training completed! Final reward: {avg_r:.3f}"
        
    except Exception as e:
        return False, f"Error: {str(e)}"

# Create enhanced learning curve chart
def create_learning_curve():
    if not training_data['episodes']:
        return go.Figure()
    
    episodes = training_data['episodes']
    rewards = np.array(training_data['rewards'])
    
    # Calculate moving averages (window=5)
    window = min(5, len(rewards))
    if len(rewards) >= window:
        moving_avg = pd.Series(rewards).rolling(window=window, min_periods=1).mean().values
        moving_std = pd.Series(rewards).rolling(window=window, min_periods=1).std().values
    else:
        moving_avg = rewards
        moving_std = np.zeros_like(rewards)
    
    fig = go.Figure()
    
    # Add confidence band (moving average ± 1 std)
    if len(rewards) > 1:
        upper_bound = moving_avg + moving_std
        lower_bound = moving_avg - moving_std
        
        fig.add_trace(go.Scatter(
            x=episodes,
            y=upper_bound,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip',
            fillcolor='rgba(102, 126, 234, 0.2)'
        ))
        
        fig.add_trace(go.Scatter(
            x=episodes,
            y=lower_bound,
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(102, 126, 234, 0.2)',
            name='Confidence Band (±1 std)',
            hoverinfo='skip'
        ))
    
    # Add raw rewards
    fig.add_trace(go.Scatter(
        x=episodes,
        y=rewards,
        mode='lines',
        name='Episode Reward',
        line=dict(color='rgba(102, 126, 234, 0.3)', width=2),
        opacity=0.6,
        showlegend=True
    ))
    
    # Add moving average (more prominent)
    fig.add_trace(go.Scatter(
        x=episodes,
        y=moving_avg,
        mode='lines+markers',
        name=f'Moving Average (window={window})',
        line=dict(color='#667eea', width=3),
        marker=dict(size=6, color='#667eea'),
        showlegend=True
    ))
    
    # Add best reward line if we have enough data
    if len(rewards) > 0:
        best_reward = np.max(rewards)
        best_episode = episodes[np.argmax(rewards)]
        fig.add_trace(go.Scatter(
            x=[best_episode],
            y=[best_reward],
            mode='markers',
            name=f'Best: {best_reward:.2f}',
            marker=dict(size=15, color='#2ecc71', symbol='star'),
            showlegend=True
        ))
    
    # Calculate and display statistics
    stats_text = ""
    if len(rewards) > 0:
        stats_text = f" | Mean: {np.mean(rewards):.2f} | Max: {np.max(rewards):.2f} | Min: {np.min(rewards):.2f}"
    
    fig.update_layout(
        title={
            'text': f"Enhanced MADDPG Learning Curve{stats_text}",
            'x': 0.5,
            'font': {'size': 18, 'family': 'Inter', 'color': '#2c3e50'}
        },
        xaxis_title="Episode",
        yaxis_title="Average Reward (AFR)",
        height=500,
        margin=dict(l=20, r=20, t=80, b=20),
        template='plotly_white',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

# Create multi-metric learning curve
def create_detailed_learning_curve():
    if not training_data['episodes'] or len(training_data['rewards']) == 0:
        return go.Figure()
    
    episodes = training_data['episodes']
    
    # Create subplots
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Reward & Moving Average', 'Actor & Critic Losses', 
                       'Q-Values', 'Cash & Occupancy'),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # 1. Rewards with moving average
    rewards = np.array(training_data['rewards'])
    window = min(5, len(rewards))
    if len(rewards) >= window:
        moving_avg = pd.Series(rewards).rolling(window=window, min_periods=1).mean().values
    else:
        moving_avg = rewards
    
    fig.add_trace(go.Scatter(
        x=episodes, y=rewards, mode='lines',
        name='Reward', line=dict(color='rgba(102, 126, 234, 0.3)', width=1),
        showlegend=False
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=episodes, y=moving_avg, mode='lines+markers',
        name='Moving Avg', line=dict(color='#667eea', width=2),
        marker=dict(size=4), showlegend=False
    ), row=1, col=1)
    
    # 2. Actor and Critic Losses
    if training_data.get('actor_losses') and len(training_data['actor_losses']) > 0:
        actor_losses = np.array(training_data['actor_losses'])
        fig.add_trace(go.Scatter(
            x=episodes, y=actor_losses, mode='lines+markers',
            name='Actor Loss', line=dict(color='#e74c3c', width=2),
            marker=dict(size=4), showlegend=False
        ), row=1, col=2)
    
    if training_data.get('critic_losses') and len(training_data['critic_losses']) > 0:
        critic_losses = np.array(training_data['critic_losses'])
        fig.add_trace(go.Scatter(
            x=episodes, y=critic_losses, mode='lines+markers',
            name='Critic Loss', line=dict(color='#f39c12', width=2),
            marker=dict(size=4), showlegend=False
        ), row=1, col=2)
    
    # 3. Q-Values
    if training_data.get('q_values') and len(training_data['q_values']) > 0:
        q_values = np.array(training_data['q_values'])
        fig.add_trace(go.Scatter(
            x=episodes, y=q_values, mode='lines+markers',
            name='Q-Value', line=dict(color='#2ecc71', width=2),
            marker=dict(size=4), showlegend=False
        ), row=2, col=1)
    
    # 4. Cash and Occupancy (on same subplot)
    if training_data.get('cash_history') and len(training_data['cash_history']) > 0:
        cash = np.array(training_data['cash_history'])
        fig.add_trace(go.Scatter(
            x=episodes, y=cash, mode='lines',
            name='Cash', line=dict(color='#3498db', width=2),
            showlegend=False
        ), row=2, col=2)
    
    if training_data.get('occupancy_history') and len(training_data['occupancy_history']) > 0:
        occupancy = np.array(training_data['occupancy_history']) * 100  # Convert to percentage
        fig.add_trace(go.Scatter(
            x=episodes, y=occupancy, mode='lines',
            name='Occupancy %', line=dict(color='#9b59b6', width=2, dash='dash'),
            showlegend=False
        ), row=2, col=2)
    
    # Update axis labels
    fig.update_xaxes(title_text="Episode", row=2, col=1)
    fig.update_xaxes(title_text="Episode", row=2, col=2)
    fig.update_yaxes(title_text="Reward", row=1, col=1)
    fig.update_yaxes(title_text="Loss", row=1, col=2)
    fig.update_yaxes(title_text="Q-Value", row=2, col=1)
    fig.update_yaxes(title_text="Cash ($)", row=2, col=2)
    
    fig.update_layout(
        title={
            'text': "Comprehensive MADDPG Training Metrics",
            'x': 0.5,
            'font': {'size': 20, 'family': 'Inter', 'color': '#2c3e50'}
        },
        height=700,
        margin=dict(l=20, r=20, t=80, b=20),
        template='plotly_white'
    )
    
    return fig

# Page layout
layout = html.Div([
    # Page header
    html.Div([
        html.H1("🤖 Multi-Agent Deep Reinforcement Learning", className="page-title"),
        html.P("MADDPG-based intelligent agents for hotel revenue optimization and resource management", 
               className="page-subtitle")
    ], className="page-header"),
    
    # Control panel
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H4("🎮 Training Control", style={'color': '#2c3e50', 'marginBottom': '20px'}),
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Number of Episodes:"),
                        dbc.Input(id="episodes-input", type="number", value=10, min=1, max=100)
                    ], width=6),
                    dbc.Col([
                        dbc.Label("Random Seed:"),
                        dbc.Input(id="seed-input", type="number", value=42, min=1, max=1000)
                    ], width=6),
                ], className="mb-3"),
                dbc.Button(
                    "🚀 Start Training", 
                    id="train-button", 
                    color="primary", 
                    size="lg",
                    className="w-100"
                ),
                html.Div(id="training-status", className="mt-3")
            ], className="chart-container")
        ], width=4),
        
        dbc.Col([
            html.Div([
                html.H4("📊 Agent Overview", style={'color': '#2c3e50', 'marginBottom': '20px'}),
                html.Div([
                    html.Div([
                        html.I(className="fas fa-coins", style={'fontSize': '2rem', 'color': '#f39c12'}),
                        html.H5("Liquidity Agent", style={'margin': '10px 0 5px 0'}),
                        html.P("Manages cash flow, borrowing, and debt repayment", style={'fontSize': '0.9rem', 'color': '#666'})
                    ], style={'textAlign': 'center', 'padding': '20px', 'border': '1px solid #eee', 'borderRadius': '10px', 'margin': '10px'})
                ], className="row"),
                html.Div([
                    html.Div([
                        html.I(className="fas fa-chart-line", style={'fontSize': '2rem', 'color': '#2ecc71'}),
                        html.H5("Investment Agent", style={'margin': '10px 0 5px 0'}),
                        html.P("Optimizes asset allocation and marketing investments", style={'fontSize': '0.9rem', 'color': '#666'})
                    ], style={'textAlign': 'center', 'padding': '20px', 'border': '1px solid #eee', 'borderRadius': '10px', 'margin': '10px'})
                ], className="row"),
                html.Div([
                    html.Div([
                        html.I(className="fas fa-boxes", style={'fontSize': '2rem', 'color': '#e74c3c'}),
                        html.H5("Procurement Agent", style={'margin': '10px 0 5px 0'}),
                        html.P("Manages inventory, amenities, and supply chain", style={'fontSize': '0.9rem', 'color': '#666'})
                    ], style={'textAlign': 'center', 'padding': '20px', 'border': '1px solid #eee', 'borderRadius': '10px', 'margin': '10px'})
                ], className="row")
            ], className="chart-container")
        ], width=8),
    ], className="mb-4"),
    
    # Enhanced learning curve
    dbc.Row([
        dbc.Col([
            html.Div([
                dcc.Graph(id="learning-curve", figure=create_learning_curve())
            ], className="chart-container")
        ], width=12),
    ], className="mb-4"),
    
    # Detailed metrics
    dbc.Row([
        dbc.Col([
            html.Div([
                dcc.Graph(id="detailed-curve", figure=create_detailed_learning_curve())
            ], className="chart-container")
        ], width=12),
    ], className="mb-4"),
    
    # Agent recommendations
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H4("💡 Agent Recommendations", style={'color': '#2c3e50', 'marginBottom': '20px'}),
                html.Div(id="agent-recommendations", children=[
                    html.P("Start training to see agent recommendations...", style={'color': '#666', 'fontStyle': 'italic'})
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
                    html.Li("Algorithm: Multi-Agent Deep Deterministic Policy Gradient (MADDPG)"),
                    html.Li("State Space: 12-dimensional (cash, occupancy, demand, etc.)"),
                    html.Li("Action Space: Continuous [-1, 1] for each agent"),
                    html.Li("Reward Function: Adaptive Financial Reward (AFR)"),
                    html.Li("Network Architecture: Actor-Critic with target networks"),
                    html.Li("Training: Centralized training, decentralized execution")
                ], style={'fontSize': '16px', 'lineHeight': '1.6'})
            ], className="chart-container")
        ], width=12),
    ])
])

# Callbacks
@dash.callback(
    [Output("training-status", "children"),
     Output("learning-curve", "figure"),
     Output("detailed-curve", "figure"),
     Output("agent-recommendations", "children")],
    [Input("train-button", "n_clicks")],
    [State("episodes-input", "value"),
     State("seed-input", "value")]
)
def train_agents(n_clicks, episodes, seed):
    if n_clicks is None:
        return "", create_learning_curve(), create_detailed_learning_curve(), html.P("Start training to see agent recommendations...", style={'color': '#666', 'fontStyle': 'italic'})
    
    # Run training
    success, message = run_maddpg_demo(episodes, seed)
    
    if success:
        status = dbc.Alert(message, color="success")
        recommendations = html.Div([
            html.H6("🎯 Current Agent Strategies:", style={'color': '#2c3e50', 'marginBottom': '15px'}),
            html.Div([
                html.Div([
                    html.I(className="fas fa-coins", style={'color': '#f39c12', 'marginRight': '10px'}),
                    html.Strong("Liquidity Agent: "),
                    "Monitoring cash flow and optimizing borrowing decisions"
                ], style={'padding': '10px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px', 'margin': '5px 0'})
            ]),
            html.Div([
                html.Div([
                    html.I(className="fas fa-chart-line", style={'color': '#2ecc71', 'marginRight': '10px'}),
                    html.Strong("Investment Agent: "),
                    "Analyzing market conditions for optimal asset allocation"
                ], style={'padding': '10px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px', 'margin': '5px 0'})
            ]),
            html.Div([
                html.Div([
                    html.I(className="fas fa-boxes", style={'color': '#e74c3c', 'marginRight': '10px'}),
                    html.Strong("Procurement Agent: "),
                    "Forecasting demand and managing inventory levels"
                ], style={'padding': '10px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px', 'margin': '5px 0'})
            ])
        ])
    else:
        status = dbc.Alert(message, color="danger")
        recommendations = html.P("Training failed. Please check the data file.", style={'color': '#e74c3c'})
    
    return status, create_learning_curve(), create_detailed_learning_curve(), recommendations
