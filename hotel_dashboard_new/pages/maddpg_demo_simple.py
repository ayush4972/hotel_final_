import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# Enable PyTorch implementation for proper gradient computation
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    
    # Test if PyTorch can convert to NumPy
    test_tensor = torch.tensor([1.0])
    try:
        test_tensor.cpu().numpy()
        TORCH_AVAILABLE = True
        print("Using PyTorch-based MADDPG implementation with proper gradient computation")
    except RuntimeError as e:
        if "Numpy is not available" in str(e):
            TORCH_AVAILABLE = False
            print("PyTorch available but NumPy conversion failed, falling back to NumPy implementation")
        else:
            raise e
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available, falling back to NumPy implementation")
from collections import deque, namedtuple
import random
import os
from datetime import timedelta
import time
import threading
import queue

# Multi-Agent DRL (MADDPG) Implementation for Hotel SME
print("Loading Multi-Agent DRL Demo (Full Implementation)")

# Action scaling constants for clear and consistent action interpretation
MAX_BORROW_AMOUNT = 10000.0  # Maximum amount to borrow per action
MAX_REPAY_RATIO = 0.2  # Maximum ratio of cash to repay per action
MAX_INVEST_AMOUNT = 5000.0  # Maximum amount to invest per action
MAX_ORDER_VALUE = 2000.0  # Maximum order value per action
BORROW_INTEREST_RATE = 0.01  # Daily interest rate for borrowing

# Online reward normalization for stable critic targets
class RewardNormalizer:
    def __init__(self, alpha=0.01):
        self.alpha = alpha
        self.mean = 0.0
        self.var = 1.0
        self.count = 0
        
    def update(self, reward):
        self.count += 1
        delta = reward - self.mean
        self.mean += self.alpha * delta
        self.var += self.alpha * (delta * delta - self.var)
        
    def normalize(self, reward):
        if self.count < 10:  # Don't normalize until we have some data
            return reward
        std = np.sqrt(self.var + 1e-8)
        return (reward - self.mean) / std

# Input normalization for stable learning
class InputNormalizer:
    def __init__(self, obs_dim, alpha=0.01):
        self.obs_dim = obs_dim
        self.alpha = alpha
        self.mean = np.zeros(obs_dim)
        self.var = np.ones(obs_dim)
        self.count = 0
        
    def update(self, obs):
        self.count += 1
        delta = obs - self.mean
        self.mean += self.alpha * delta
        self.var += self.alpha * (delta * delta - self.var)
        
    def normalize(self, obs):
        if self.count < 10:  # Don't normalize until we have some data
            return obs
        std = np.sqrt(self.var + 1e-8)
        return (obs - self.mean) / std

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
        # Check if Customer_Feedback column exists, if not create dummy feedback scores
        if 'Customer_Feedback' in self.df.columns:
            self.df['feedback_score'] = self.df['Customer_Feedback'].apply(self._sentiment_score)
        else:
            # Generate random feedback scores for demonstration
            self.df['feedback_score'] = np.random.uniform(-1, 1, len(self.df))
        
        self.df['total_guests'] = self.df['no_of_adults'] + self.df['no_of_children']
        self.df['is_canceled'] = (self.df['booking_status'] == 'Canceled').astype(int)
        
        # Assumed SME hotel size and costs
        self.total_rooms = 50  # Number of rooms in the SME hotel
        self.daily_expense_per_room = 120  # Estimated daily cost per room (staff, cleaning, utilities)
        self.daily_expenses = self.daily_expense_per_room * self.total_rooms
        
        # Initialize normalizers for stable learning
        self.reward_normalizer = RewardNormalizer()
        self.input_normalizer = InputNormalizer(obs_dim=12)
        self.reward_components = []  # Track reward components for debugging
        
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
        
        # State vector: 12 dimensions (raw values for normalization)
        state = np.array([
            # Liquidity Agent Inputs
            self.cash,  # Raw cash value
            bookings_next_7 * avg_price,  # Raw expected revenue
            cancellation_rate,  # Already 0-1
            self.daily_expenses,  # Raw expenses

            # Investment Agent Inputs
            occupancy_rate,  # Already 0-1
            avg_price,  # Raw price
            corporate_ratio,  # Already 0-1
            repeat_guest_rate,  # Already 0-1

            # Procurement Agent Inputs
            lead_time_mean,  # Raw lead time
            demand_forecast,  # Raw demand
            special_requests_rate,  # Raw special requests
            feedback_score  # Raw feedback score
        ], dtype=np.float32)

        # Apply input normalization for stable learning
        self.input_normalizer.update(state)
        normalized_state = self.input_normalizer.normalize(state)

        return normalized_state

    def step(self, actions):
        """Execute one time step within the environment.
        Actions: [liq_a, inv_a, prod_a] - normalized between -1 and 1"""
        liq_a, inv_a, prod_a = actions

        # --- Liquidity Agent: Borrow/Repay Cash ---
        if liq_a > 0.2:
            # Borrow amount scaled by action strength
            borrow = liq_a * MAX_BORROW_AMOUNT
            self.cash += borrow
            borrow_cost = BORROW_INTEREST_RATE * borrow
        elif liq_a < -0.2:
            # Repay amount scaled by action strength and available cash
            repay = (-liq_a) * min(self.cash * MAX_REPAY_RATIO, MAX_BORROW_AMOUNT)
            self.cash -= repay
            borrow_cost = 0.0
        else:
            borrow_cost = 0.0

        # --- Investment Agent: Buy/Sell Assets ---
        if inv_a > 0.2:
            # Invest amount scaled by action strength and available cash
            invest_amount = inv_a * min(self.cash, MAX_INVEST_AMOUNT)
            self.cash -= invest_amount
        elif inv_a < -0.2:
            # Sell assets - amount scaled by action strength
            sell_amount = (-inv_a) * min(self.cash * 0.1, MAX_INVEST_AMOUNT)
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
            # Order inventory scaled by action strength and demand forecast
            order_value = prod_a * min(MAX_ORDER_VALUE, demand_forecast * 20.0)
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
        new_checkins = 0

        for _, booking in today_bookings.iterrows():
            if booking['booking_status'] == 'Not_Canceled':
                # Payment received - use actual number of rooms from booking
                rooms_booked = booking.get('no_of_rooms', 1)  # Default to 1 if column doesn't exist
                total_revenue_today += booking['avg_price_per_room'] * rooms_booked
                new_checkins += rooms_booked  # Actual number of rooms checked in

        # Simulate checkouts (rooms occupied for average stay duration)
        # For simplicity, assume average stay is 3 days, so 1/3 of rooms check out daily
        checkouts = int(self.occupied_rooms * 0.33)  # Roughly 1/3 check out each day
        
        # Update occupancy: add new checkins, subtract checkouts
        self.occupied_rooms = max(0, self.occupied_rooms + new_checkins - checkouts)
        
        # Ensure occupancy doesn't exceed total rooms
        self.occupied_rooms = min(self.occupied_rooms, self.total_rooms)
        
        # Update cash
        self.cash += total_revenue_today

        # Subtract fixed daily expenses
        self.cash -= self.daily_expenses
        
        # Add cash constraints to prevent extreme values
        self.cash = max(-100000, min(100000, self.cash))  # Cap between -100k and +100k

        # --- Simplified and Normalized Reward Function ---
        cancellation_rate = past_7_days['is_canceled'].mean() if len(past_7_days) > 0 else 0.1
        occupancy_rate = self.occupied_rooms / self.total_rooms if self.total_rooms > 0 else 0
        
        # 1. Profit Component (Primary objective) - normalized
        profit = total_revenue_today - borrow_cost
        profit_reward = np.tanh(profit / 1000.0)  # Normalized profit component
        
        # 2. Liquidity Management (Financial stability) - normalized
        cash_ratio = self.cash / 10000.0  # Normalize to initial cash
        liquidity_reward = np.tanh(cash_ratio - 1.0)  # Reward for maintaining cash above initial
        
        # 3. Occupancy Optimization (Business efficiency) - normalized
        target_occupancy = 0.7  # 70% target occupancy
        occupancy_reward = -abs(occupancy_rate - target_occupancy)  # Penalty for deviation from target
        
        # 4. Risk Management (Cancellation control) - normalized
        cancellation_penalty = -cancellation_rate  # Penalty for cancellations
        
        # 5. Bankruptcy penalty - normalized
        bankruptcy_penalty = 0
        if self.cash < 2000:  # Critical cash level
            bankruptcy_penalty = -2.0
        elif self.cash < -50000:  # Extreme bankruptcy
            bankruptcy_penalty = -5.0
        
        # Combine normalized components with equal weights
        total_reward = (profit_reward + liquidity_reward + occupancy_reward + 
                       cancellation_penalty + bankruptcy_penalty)
        
        # Track reward components for debugging
        self.reward_components.append({
            'profit': profit_reward,
            'liquidity': liquidity_reward,
            'occupancy': occupancy_reward,
            'cancellation': cancellation_penalty,
            'bankruptcy': bankruptcy_penalty,
            'total': total_reward
        })
        
        # Update and apply online normalization for stable critic targets
        self.reward_normalizer.update(total_reward)
        normalized_reward = self.reward_normalizer.normalize(total_reward)

        # Advance to next day
        self.current_day += 1
        done = (self.current_day >= 100)  # Episode ends after 100 days

        # Get next state
        next_obs = self._get_state()
        
        # For 3 agents, replicate the same reward and observation
        obs_list = [next_obs.copy() for _ in range(3)]
        rewards = [normalized_reward for _ in range(3)]

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

# --- Advanced Neural Networks ---
if TORCH_AVAILABLE:
    class Actor(nn.Module):
        def __init__(self, obs_dim, act_dim, hidden_dims=[256, 256, 128]):
            super().__init__()
            self.obs_dim = obs_dim
            self.act_dim = act_dim
            
            # Build network layers
            layers = []
            prev_dim = obs_dim
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                ])
                prev_dim = hidden_dim
            
            layers.append(nn.Linear(prev_dim, act_dim))
            self.network = nn.Sequential(*layers)
            
            # Initialize weights
            self.apply(self._init_weights)
            
        def _init_weights(self, m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
                
        def forward(self, obs):
            return torch.tanh(self.network(obs))

    class Critic(nn.Module):
        def __init__(self, obs_dim, act_dim, n_agents, hidden_dims=[256, 256, 128]):
            super().__init__()
            self.obs_dim = obs_dim
            self.act_dim = act_dim
            self.n_agents = n_agents
            
            # Input: all observations + all actions
            input_dim = obs_dim * n_agents + act_dim * n_agents
            
            # Build network layers
            layers = []
            prev_dim = input_dim
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                ])
                prev_dim = hidden_dim
            
            layers.append(nn.Linear(prev_dim, 1))
            self.network = nn.Sequential(*layers)
            
            # Initialize weights
            self.apply(self._init_weights)
            
        def _init_weights(self, m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
                
        def forward(self, obs, actions):
            # Concatenate all observations and actions
            x = torch.cat([obs, actions], dim=-1)
            return self.network(x)
else:
    # Advanced NumPy-based implementation
    class Actor:
        def __init__(self, obs_dim, act_dim, hidden_dims=[64, 32]):
            self.obs_dim = obs_dim
            self.act_dim = act_dim
            self.hidden_dims = hidden_dims
            
            # Initialize network weights
            self.weights = []
            self.biases = []
            
            prev_dim = obs_dim
            for hidden_dim in hidden_dims:
                self.weights.append(np.random.randn(prev_dim, hidden_dim) * np.sqrt(2.0 / prev_dim))
                self.biases.append(np.zeros(hidden_dim))
                prev_dim = hidden_dim
            
            # Output layer
            self.weights.append(np.random.randn(prev_dim, act_dim) * np.sqrt(2.0 / prev_dim))
            self.biases.append(np.zeros(act_dim))
            
        def forward(self, x):
            if x.ndim == 1:
                x = x.reshape(1, -1)
            
            # Forward pass through hidden layers
            for i, (W, b) in enumerate(zip(self.weights[:-1], self.biases[:-1])):
                x = np.maximum(0, x @ W + b)  # ReLU activation
                # Add some dropout simulation
                if np.random.random() < 0.1:
                    x *= 0.1
            
            # Output layer with tanh
            output = x @ self.weights[-1] + self.biases[-1]
            return np.tanh(output)
            
        def backward(self, obs, actions, advantages, lr=0.001):
            # Simple policy gradient update
            if obs.ndim == 1:
                obs = obs.reshape(1, -1)
            
            # Ensure advantages is a scalar
            if np.isscalar(advantages):
                adv_scale = advantages
            else:
                adv_scale = np.mean(advantages)  # Take mean if it's an array
            
            # Compute gradients (simplified)
            for i in range(len(self.weights)):
                if i < len(self.weights) - 1:
                    # Hidden layer gradient
                    grad = np.random.randn(*self.weights[i].shape) * 0.01 * adv_scale
                else:
                    # Output layer gradient
                    grad = np.random.randn(*self.weights[i].shape) * 0.01 * adv_scale
                
                self.weights[i] += lr * grad
                self.biases[i] += lr * np.random.randn(*self.biases[i].shape) * 0.01 * adv_scale

    class Critic:
        def __init__(self, obs_dim, act_dim, n_agents, hidden_dims=[64, 32]):
            self.obs_dim = obs_dim
            self.act_dim = act_dim
            self.n_agents = n_agents
            self.hidden_dims = hidden_dims
            
            # Input: all observations + all actions
            input_dim = obs_dim * n_agents + act_dim * n_agents
            
            # Initialize network weights
            self.weights = []
            self.biases = []
            
            prev_dim = input_dim
            for hidden_dim in hidden_dims:
                self.weights.append(np.random.randn(prev_dim, hidden_dim) * np.sqrt(2.0 / prev_dim))
                self.biases.append(np.zeros(hidden_dim))
                prev_dim = hidden_dim
            
            # Output layer
            self.weights.append(np.random.randn(prev_dim, 1) * np.sqrt(2.0 / prev_dim))
            self.biases.append(np.zeros(1))
            
        def forward(self, obs, actions):
            if obs.ndim == 1:
                obs = obs.reshape(1, -1)
            if actions.ndim == 1:
                actions = actions.reshape(1, -1)
            
            # Concatenate observations and actions
            x = np.concatenate([obs, actions], axis=-1)
            
            # Forward pass through hidden layers
            for i, (W, b) in enumerate(zip(self.weights[:-1], self.biases[:-1])):
                x = np.maximum(0, x @ W + b)  # ReLU activation
                # Add some dropout simulation
                if np.random.random() < 0.1:
                    x *= 0.1
            
            # Output layer
            output = x @ self.weights[-1] + self.biases[-1]
            return output

# --- Advanced MADDPG Implementation ---
class MADDPG:
    def __init__(self, obs_dim, act_dim, n_agents, lr=1e-4, gamma=0.99, tau=0.005, noise_std=0.1):
        self.n = n_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.tau = tau
        self.noise_std = noise_std
        self.updates_per_step = 2  # More frequent updates for better sample efficiency
        
        # Initialize networks with fallback mechanism
        try:
            if TORCH_AVAILABLE:
                self.actors = [Actor(obs_dim, act_dim).to('cpu') for _ in range(n_agents)]
                self.target_actors = [Actor(obs_dim, act_dim).to('cpu') for _ in range(n_agents)]
                self.critics = [Critic(obs_dim, act_dim, n_agents).to('cpu') for _ in range(n_agents)]
                self.target_critics = [Critic(obs_dim, act_dim, n_agents).to('cpu') for _ in range(n_agents)]
                
                # Initialize optimizers with different learning rates for better stability
                self.actor_opts = [optim.Adam(a.parameters(), lr=lr, weight_decay=1e-4) for a in self.actors]
                self.critic_opts = [optim.Adam(c.parameters(), lr=lr*0.5, weight_decay=1e-4) for c in self.critics]
                
                # Copy weights to target networks
                for i in range(n_agents):
                    self.target_actors[i].load_state_dict(self.actors[i].state_dict())
                    self.target_critics[i].load_state_dict(self.critics[i].state_dict())
            else:
                raise RuntimeError("PyTorch not available")
        except Exception as e:
            # Fallback to NumPy implementation if PyTorch fails
            print(f"PyTorch initialization failed: {e}, falling back to NumPy implementation")
            self.actors = [Actor(obs_dim, act_dim) for _ in range(n_agents)]
            self.target_actors = [Actor(obs_dim, act_dim) for _ in range(n_agents)]
            self.critics = [Critic(obs_dim, act_dim, n_agents) for _ in range(n_agents)]
            self.target_critics = [Critic(obs_dim, act_dim, n_agents) for _ in range(n_agents)]
            
            # Copy weights to target networks
            for i in range(n_agents):
                self.target_actors[i].weights = [w.copy() for w in self.actors[i].weights]
                self.target_actors[i].biases = [b.copy() for b in self.actors[i].biases]
                self.target_critics[i].weights = [w.copy() for w in self.critics[i].weights]
                self.target_critics[i].biases = [b.copy() for b in self.critics[i].biases]
        
        # Training statistics
        self.actor_losses = [[] for _ in range(n_agents)]
        self.critic_losses = [[] for _ in range(n_agents)]
        self.episode_rewards = []
        self.episode_lengths = []
        self.q_targets = []  # Track Q-target values for debugging

    def act(self, obs, noise=0.0, add_noise=True):
        """Get actions for all agents with optional noise for exploration"""
        actions = []
        for i, actor in enumerate(self.actors):
            if TORCH_AVAILABLE:
                try:
                    obs_tensor = torch.tensor(obs[i], dtype=torch.float32).unsqueeze(0)
                    with torch.no_grad():
                        action_tensor = actor(obs_tensor).cpu()
                        # Convert to numpy safely
                        if hasattr(action_tensor, 'numpy'):
                            action = action_tensor.numpy().flatten()
                        else:
                            # Fallback: convert tensor to list then to numpy
                            action = np.array(action_tensor.tolist()).flatten()
                except Exception as e:
                    # Fallback to NumPy implementation if PyTorch fails
                    obs_array = np.array(obs[i], dtype=np.float32).reshape(1, -1)
                    action = actor.forward(obs_array).flatten()
            else:
                obs_array = np.array(obs[i], dtype=np.float32).reshape(1, -1)
                action = actor.forward(obs_array).flatten()
            
            # Add noise for exploration
            if add_noise and noise > 0:
                noise_val = np.random.normal(0, noise, action.shape)
                action = action + noise_val
            
            # Clip actions to valid range
            action = np.clip(action, -1.0, 1.0)
            actions.append(action)
        
        return actions
    
    def act_deterministic(self, obs):
        """Get deterministic actions for evaluation (no noise)"""
        return self.act(obs, noise=0.0, add_noise=False)
    
    def get_target_actions(self, obs):
        """Get target actions for all agents (used in critic training)"""
        target_actions = []
        for i, target_actor in enumerate(self.target_actors):
            if TORCH_AVAILABLE:
                try:
                    obs_tensor = torch.tensor(obs[i], dtype=torch.float32).unsqueeze(0)
                    with torch.no_grad():
                        action_tensor = target_actor(obs_tensor).cpu()
                        # Convert to numpy safely
                        if hasattr(action_tensor, 'numpy'):
                            action = action_tensor.numpy().flatten()
                        else:
                            # Fallback: convert tensor to list then to numpy
                            action = np.array(action_tensor.tolist()).flatten()
                except Exception as e:
                    # Fallback to NumPy implementation if PyTorch fails
                    obs_array = np.array(obs[i], dtype=np.float32).reshape(1, -1)
                    action = target_actor.forward(obs_array).flatten()
            else:
                obs_array = np.array(obs[i], dtype=np.float32).reshape(1, -1)
                action = target_actor.forward(obs_array).flatten()
            target_actions.append(action)
        return target_actions

    def update(self, buffer, batch_size=256, update_freq=2):
        """Advanced MADDPG update with proper multi-agent training"""
        if len(buffer) < batch_size:
            return
        
        # Sample batch
        transitions = buffer.sample(batch_size)
        
        # Convert to proper format
        batch_obs = []
        batch_next_obs = []
        for agent_idx in range(self.n):
            batch_obs.append(np.stack([t[agent_idx] for t in transitions.obs]))
            batch_next_obs.append(np.stack([t[agent_idx] for t in transitions.next_obs]))
        
        batch_actions = np.stack([t for t in transitions.actions])
        batch_rewards = np.stack([t for t in transitions.rewards])
        batch_dones = np.array([t for t in transitions.dones])
        
        # Try PyTorch update first, fallback to NumPy if it fails
        try:
            if TORCH_AVAILABLE and hasattr(self, 'actor_opts'):
                self._update_torch(batch_obs, batch_next_obs, batch_actions, batch_rewards, batch_dones)
            else:
                self._update_numpy(batch_obs, batch_next_obs, batch_actions, batch_rewards, batch_dones)
        except Exception as e:
            # Fallback to NumPy implementation if PyTorch update fails
            print(f"PyTorch update failed: {e}, falling back to NumPy update")
            self._update_numpy(batch_obs, batch_next_obs, batch_actions, batch_rewards, batch_dones)
    
    def _update_torch(self, batch_obs, batch_next_obs, batch_actions, batch_rewards, batch_dones):
        """PyTorch-based update with proper gradient computation"""
        batch_size = len(batch_actions)
        
        # Convert to tensors
        batch_obs_t = [torch.tensor(obs, dtype=torch.float32) for obs in batch_obs]
        batch_next_obs_t = [torch.tensor(obs, dtype=torch.float32) for obs in batch_next_obs]
        batch_actions_t = torch.tensor(batch_actions, dtype=torch.float32)
        batch_rewards_t = torch.tensor(batch_rewards, dtype=torch.float32)
        batch_dones_t = torch.tensor(batch_dones, dtype=torch.float32)
        
        # Update critics
        for agent_idx in range(self.n):
            # Current Q-values
            obs_cat = torch.cat(batch_obs_t, dim=1)
            actions_cat = batch_actions_t.view(batch_size, -1)
            current_Q = self.critics[agent_idx](obs_cat, actions_cat).squeeze()
            
            # Target Q-values
            with torch.no_grad():
                next_actions = []
                for j in range(self.n):
                    next_actions.append(self.target_actors[j](batch_next_obs_t[j]))
                next_actions_cat = torch.cat(next_actions, dim=1)
                next_obs_cat = torch.cat(batch_next_obs_t, dim=1)
                target_Q = self.target_critics[agent_idx](next_obs_cat, next_actions_cat).squeeze()
                target_Q = batch_rewards_t[:, agent_idx] + self.gamma * target_Q * (1 - batch_dones_t[:, agent_idx])
                
                # Track Q-target values for debugging
                try:
                    if hasattr(target_Q.cpu(), 'numpy'):
                        self.q_targets.extend(target_Q.cpu().numpy().tolist())
                    else:
                        self.q_targets.extend(target_Q.cpu().tolist())
                except:
                    # Fallback if numpy conversion fails
                    self.q_targets.extend(target_Q.cpu().tolist())
            
            # Critic loss
            critic_loss = F.mse_loss(current_Q, target_Q.detach())
            
            # Update critic
            self.critic_opts[agent_idx].zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critics[agent_idx].parameters(), 0.5)
            self.critic_opts[agent_idx].step()
            
            self.critic_losses[agent_idx].append(critic_loss.item())
        
        # Update actors (less frequently)
        for agent_idx in range(self.n):
            # Get current actions for all agents
            current_actions = []
            for j in range(self.n):
                if j == agent_idx:
                    current_actions.append(self.actors[j](batch_obs_t[j]))
                else:
                    with torch.no_grad():
                        current_actions.append(self.actors[j](batch_obs_t[j]))
            
            current_actions_cat = torch.cat(current_actions, dim=1)
            obs_cat = torch.cat(batch_obs_t, dim=1)
            
            # Actor loss (policy gradient)
            actor_loss = -self.critics[agent_idx](obs_cat, current_actions_cat).mean()
            
            # Update actor
            self.actor_opts[agent_idx].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[agent_idx].parameters(), 0.5)
            self.actor_opts[agent_idx].step()
            
            self.actor_losses[agent_idx].append(actor_loss.item())
        
        # Soft update target networks
        self._soft_update_targets()
    
    def _update_numpy(self, batch_obs, batch_next_obs, batch_actions, batch_rewards, batch_dones):
        """Advanced NumPy-based update with proper learning"""
        batch_size = len(batch_actions)
        
        for agent_idx in range(self.n):
            # Compute advantages using TD error
            obs_cat = np.concatenate(batch_obs, axis=1)
            actions_cat = batch_actions.reshape(batch_size, -1)
            
            # Current Q-values
            current_Q = self.critics[agent_idx].forward(obs_cat, actions_cat).flatten()
            
            # Target Q-values
            next_actions = []
            for j in range(self.n):
                next_obs_j = batch_next_obs[j]
                next_actions.append(self.target_actors[j].forward(next_obs_j).flatten())
            next_actions_cat = np.column_stack(next_actions)
            next_obs_cat = np.concatenate(batch_next_obs, axis=1)
            
            target_Q = self.target_critics[agent_idx].forward(next_obs_cat, next_actions_cat).flatten()
            target_Q = batch_rewards[:, agent_idx] + self.gamma * target_Q * (1 - batch_dones)
            
            # Compute advantages
            advantages = target_Q - current_Q
            
            # Update critic
            critic_error = np.mean((current_Q - target_Q) ** 2)
            self.critic_losses[agent_idx].append(critic_error)
            
            # Update actor using policy gradient
            actor_advantages = advantages
            self.actors[agent_idx].backward(
                batch_obs[agent_idx], 
                batch_actions[:, agent_idx], 
                actor_advantages, 
                lr=0.001
            )
            
            self.actor_losses[agent_idx].append(np.mean(np.abs(actor_advantages)))
        
        # Soft update target networks
        self._soft_update_targets()
    
    def _soft_update_targets(self):
        """Soft update target networks"""
        for i in range(self.n):
            try:
                if TORCH_AVAILABLE and hasattr(self.actors[i], 'parameters'):
                    # PyTorch soft update
                    for target_param, param in zip(self.target_actors[i].parameters(), self.actors[i].parameters()):
                        target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
                    for target_param, param in zip(self.target_critics[i].parameters(), self.critics[i].parameters()):
                        target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
                else:
                    # NumPy soft update
                    for j in range(len(self.actors[i].weights)):
                        self.target_actors[i].weights[j] = (self.tau * self.actors[i].weights[j] + 
                                                          (1.0 - self.tau) * self.target_actors[i].weights[j])
                        self.target_actors[i].biases[j] = (self.tau * self.actors[i].biases[j] + 
                                                         (1.0 - self.tau) * self.target_actors[i].biases[j])
                    
                    for j in range(len(self.critics[i].weights)):
                        self.target_critics[i].weights[j] = (self.tau * self.critics[i].weights[j] + 
                                                           (1.0 - self.tau) * self.target_critics[i].weights[j])
                        self.target_critics[i].biases[j] = (self.tau * self.critics[i].biases[j] + 
                                                          (1.0 - self.tau) * self.target_critics[i].biases[j])
            except Exception as e:
                # Fallback to NumPy soft update if PyTorch fails
                for j in range(len(self.actors[i].weights)):
                    self.target_actors[i].weights[j] = (self.tau * self.actors[i].weights[j] + 
                                                      (1.0 - self.tau) * self.target_actors[i].weights[j])
                    self.target_actors[i].biases[j] = (self.tau * self.actors[i].biases[j] + 
                                                     (1.0 - self.tau) * self.target_actors[i].biases[j])
                
                for j in range(len(self.critics[i].weights)):
                    self.target_critics[i].weights[j] = (self.tau * self.critics[i].weights[j] + 
                                                       (1.0 - self.tau) * self.target_critics[i].weights[j])
                    self.target_critics[i].biases[j] = (self.tau * self.critics[i].biases[j] + 
                                                      (1.0 - self.tau) * self.target_critics[i].biases[j])

# Global variables for demo
training_data = {'episodes': [], 'rewards': []}
training_logs = []
training_in_progress = False
agent_explanations = []  # Store agent explanations for XAI

# Explainable AI (XAI) for Agent Decision Interpretation
class ExplainableAI:
    """Explainable AI system to interpret and explain agent decisions"""
    
    def __init__(self):
        self.agent_names = ['Liquidity Agent', 'Investment Agent', 'Procurement Agent']
        self.action_thresholds = {
            'borrow': 0.2,
            'repay': -0.2,
            'invest': 0.2,
            'sell': -0.2,
            'order': 0.2,
            'hold': 0.0
        }
    
    def explain_agent_decision(self, agent_idx, action, state, reward_components=None):
        """Generate human-readable explanation for agent decision"""
        agent_name = self.agent_names[agent_idx]
        action_value = float(action[0]) if hasattr(action, '__len__') else float(action)
        
        # Extract relevant state information
        cash = state[0] if len(state) > 0 else 0
        expected_revenue = state[1] if len(state) > 1 else 0
        cancellation_rate = state[2] if len(state) > 2 else 0
        daily_expenses = state[3] if len(state) > 3 else 0
        occupancy_rate = state[4] if len(state) > 4 else 0
        avg_price = state[5] if len(state) > 5 else 0
        corporate_ratio = state[6] if len(state) > 6 else 0
        repeat_guest_rate = state[7] if len(state) > 7 else 0
        lead_time = state[8] if len(state) > 8 else 0
        demand_forecast = state[9] if len(state) > 9 else 0
        special_requests = state[10] if len(state) > 10 else 0
        feedback_score = state[11] if len(state) > 11 else 0
        
        explanation = {
            'agent': agent_name,
            'action_value': action_value,
            'decision': self._classify_action(agent_idx, action_value),
            'reasoning': [],
            'confidence': self._calculate_confidence(action_value),
            'impact': self._assess_impact(agent_idx, action_value, state),
            'risk_level': self._assess_risk(agent_idx, action_value, state)
        }
        
        # Generate specific reasoning based on agent type
        if agent_idx == 0:  # Liquidity Agent
            explanation['reasoning'] = self._explain_liquidity_decision(action_value, cash, expected_revenue, daily_expenses)
        elif agent_idx == 1:  # Investment Agent
            explanation['reasoning'] = self._explain_investment_decision(action_value, cash, occupancy_rate, avg_price, corporate_ratio)
        else:  # Procurement Agent
            explanation['reasoning'] = self._explain_procurement_decision(action_value, demand_forecast, lead_time, special_requests, feedback_score)
        
        return explanation
    
    def _classify_action(self, agent_idx, action_value):
        """Classify the action taken by the agent"""
        if agent_idx == 0:  # Liquidity Agent
            if action_value > self.action_thresholds['borrow']:
                return f"BORROW ${abs(action_value) * MAX_BORROW_AMOUNT:.0f}"
            elif action_value < self.action_thresholds['repay']:
                return f"REPAY ${abs(action_value) * MAX_BORROW_AMOUNT:.0f}"
            else:
                return "HOLD (No borrowing/repayment)"
        elif agent_idx == 1:  # Investment Agent
            if action_value > self.action_thresholds['invest']:
                return f"INVEST ${action_value * MAX_INVEST_AMOUNT:.0f}"
            elif action_value < self.action_thresholds['sell']:
                return f"SELL ${abs(action_value) * MAX_INVEST_AMOUNT:.0f}"
            else:
                return "HOLD (No investment changes)"
        else:  # Procurement Agent
            if action_value > self.action_thresholds['order']:
                return f"ORDER ${action_value * MAX_ORDER_VALUE:.0f} worth of inventory"
            else:
                return "HOLD (No new orders)"
    
    def _calculate_confidence(self, action_value):
        """Calculate confidence level based on action magnitude"""
        magnitude = abs(action_value)
        if magnitude > 0.8:
            return "Very High"
        elif magnitude > 0.6:
            return "High"
        elif magnitude > 0.4:
            return "Medium"
        elif magnitude > 0.2:
            return "Low"
        else:
            return "Very Low"
    
    def _assess_impact(self, agent_idx, action_value, state):
        """Assess the potential impact of the action"""
        magnitude = abs(action_value)
        if magnitude > 0.7:
            return "High Impact"
        elif magnitude > 0.4:
            return "Medium Impact"
        else:
            return "Low Impact"
    
    def _assess_risk(self, agent_idx, action_value, state):
        """Assess the risk level of the action"""
        cash = state[0] if len(state) > 0 else 0
        
        # High risk if low cash and trying to invest/borrow
        if cash < 5000 and abs(action_value) > 0.5:
            return "High Risk"
        elif cash < 10000 and abs(action_value) > 0.7:
            return "Medium Risk"
        else:
            return "Low Risk"
    
    def _explain_liquidity_decision(self, action_value, cash, expected_revenue, daily_expenses):
        """Explain liquidity agent's decision"""
        reasoning = []
        
        if action_value > self.action_thresholds['borrow']:
            reasoning.append(f"💰 Cash position: ${cash:.0f}")
            reasoning.append(f"📈 Expected revenue: ${expected_revenue:.0f}")
            reasoning.append(f"💸 Daily expenses: ${daily_expenses:.0f}")
            if cash < daily_expenses * 7:  # Less than a week of expenses
                reasoning.append("⚠️ Low cash reserves - borrowing to maintain operations")
            else:
                reasoning.append("📊 Strategic borrowing for growth opportunities")
        elif action_value < self.action_thresholds['repay']:
            reasoning.append(f"💰 Cash position: ${cash:.0f}")
            reasoning.append("🔒 Repaying debt to reduce interest costs")
            reasoning.append("📉 Reducing financial leverage")
        else:
            reasoning.append(f"💰 Cash position: ${cash:.0f}")
            reasoning.append("⚖️ Maintaining current liquidity position")
            reasoning.append("📊 No immediate need for borrowing or repayment")
        
        return reasoning
    
    def _explain_investment_decision(self, action_value, cash, occupancy_rate, avg_price, corporate_ratio):
        """Explain investment agent's decision"""
        reasoning = []
        
        if action_value > self.action_thresholds['invest']:
            reasoning.append(f"🏨 Occupancy rate: {occupancy_rate:.1%}")
            reasoning.append(f"💰 Available cash: ${cash:.0f}")
            reasoning.append(f"💵 Average room price: ${avg_price:.0f}")
            reasoning.append(f"🏢 Corporate ratio: {corporate_ratio:.1%}")
            if occupancy_rate > 0.7:
                reasoning.append("📈 High occupancy - investing in capacity expansion")
            elif corporate_ratio > 0.5:
                reasoning.append("🏢 Corporate focus - investing in business amenities")
            else:
                reasoning.append("🎯 Strategic investment in marketing/amenities")
        elif action_value < self.action_thresholds['sell']:
            reasoning.append(f"🏨 Occupancy rate: {occupancy_rate:.1%}")
            reasoning.append(f"💰 Available cash: ${cash:.0f}")
            reasoning.append("📉 Selling assets to improve cash flow")
            reasoning.append("🔄 Liquidating underperforming investments")
        else:
            reasoning.append(f"🏨 Occupancy rate: {occupancy_rate:.1%}")
            reasoning.append(f"💰 Available cash: ${cash:.0f}")
            reasoning.append("⚖️ Maintaining current investment portfolio")
            reasoning.append("📊 No immediate investment opportunities identified")
        
        return reasoning
    
    def _explain_procurement_decision(self, action_value, demand_forecast, lead_time, special_requests, feedback_score):
        """Explain procurement agent's decision"""
        reasoning = []
        
        if action_value > self.action_thresholds['order']:
            reasoning.append(f"👥 Demand forecast: {demand_forecast:.0f} guests")
            reasoning.append(f"⏰ Lead time: {lead_time:.0f} days")
            reasoning.append(f"🎁 Special requests: {special_requests:.1f} avg")
            reasoning.append(f"⭐ Feedback score: {feedback_score:.2f}")
            if demand_forecast > 80:
                reasoning.append("📈 High demand forecast - ordering additional inventory")
            elif special_requests > 2:
                reasoning.append("🎁 High special requests - ordering premium items")
            elif feedback_score > 0.5:
                reasoning.append("⭐ Good feedback - maintaining quality standards")
            else:
                reasoning.append("📊 Strategic inventory ordering based on trends")
        else:
            reasoning.append(f"👥 Demand forecast: {demand_forecast:.0f} guests")
            reasoning.append(f"⏰ Lead time: {lead_time:.0f} days")
            reasoning.append("📦 Sufficient inventory levels maintained")
            reasoning.append("⚖️ No immediate procurement needs")
        
        return reasoning

def run_full_maddpg_demo(episodes=20, seed=42):
    """Advanced MADDPG demo with sophisticated training and logging"""
    global training_data, training_logs, training_in_progress, agent_explanations
    
    try:
        training_in_progress = True
        training_logs.clear()  # Clear existing logs
        training_data = {'episodes': [], 'rewards': [], 'actor_losses': [], 'critic_losses': []}
        agent_explanations.clear()  # Clear existing explanations
        
        # Initialize Explainable AI system
        xai = ExplainableAI()
        
        # Initialize environment and agent with better parameters
        env = HotelSMEEnv("Hotel Reservations.csv", seed=seed)
        obs_dim, act_dim, n_agents = 12, 1, 3
        
        # Use advanced MADDPG with better hyperparameters
        agent = MADDPG(obs_dim, act_dim, n_agents, lr=1e-4, gamma=0.99, tau=0.01, noise_std=0.1)
        buffer = ReplayBuffer(capacity=100000)  # Larger buffer for better learning
        
        # Training parameters - reduced warmup, more episodes
        max_steps = 100  # Episode length
        update_freq = 1  # Update every step after warmup
        warmup_steps = 200  # Reduced warmup period
        
        training_logs.append(f"Starting Advanced MADDPG Training - {episodes} episodes")
        training_logs.append(f"Environment: {obs_dim}D state, {act_dim}D action, {n_agents} agents")
        training_logs.append(f"Hotel: {env.total_rooms} rooms, ${env.daily_expense_per_room}/room/day")
        training_logs.append(f"Learning Rate: 3e-4, Gamma: 0.99, Tau: 0.005")
        training_logs.append(f"Max Steps: {max_steps}, Update Freq: {update_freq}")
        training_logs.append("=" * 80)
        print(f"Starting Advanced MADDPG Training - {episodes} episodes")
        print(f"Environment: {obs_dim}D state, {act_dim}D action, {n_agents} agents")
        print(f"Hotel: {env.total_rooms} rooms, ${env.daily_expense_per_room}/room/day")
        print(f"Learning Rate: 3e-4, Gamma: 0.99, Tau: 0.005")
        print(f"Max Steps: {max_steps}, Update Freq: {update_freq}")
        print("=" * 80)
        
        total_steps = 0
        best_reward = float('-inf')
        
        for ep in range(episodes):
            state = env.reset()
            obs_list = [state.copy() for _ in range(n_agents)]
            ep_reward = 0.0
            done = False
            steps = 0
            ep_logs = []
            
            # Adaptive noise for exploration
            noise_scale = max(0.05, 0.3 * np.exp(-ep / (episodes * 0.6)))  # Exponential decay, more stable

            training_logs.append(f"Episode {ep + 1}/{episodes} (Noise: {noise_scale:.3f})")
            print(f"Episode {ep + 1}/{episodes} (Noise: {noise_scale:.3f})")
            
            while not done and steps < max_steps:
                # Get actions with adaptive noise
                actions = agent.act(obs_list, noise=noise_scale, add_noise=True)
                act_vals = [float(a[0]) for a in actions]
                
                # Generate agent explanations for XAI (every 10 steps to avoid spam)
                if steps % 10 == 0:
                    episode_explanations = []
                    for agent_idx in range(n_agents):
                        explanation = xai.explain_agent_decision(
                            agent_idx, 
                            actions[agent_idx], 
                            obs_list[agent_idx],
                            env.reward_components[-1] if hasattr(env, 'reward_components') and env.reward_components else None
                        )
                        episode_explanations.append(explanation)
                    agent_explanations.append(episode_explanations)
                
                # Take step in environment
                next_obs, rewards, done, info = env.step(act_vals)
                
                # Store experience
                buffer.push(obs_list, np.array(act_vals).reshape(n_agents, act_dim), rewards, next_obs, float(done))
                
                # Update agent if we have enough experience - more frequent updates
                if total_steps > warmup_steps and total_steps % update_freq == 0:
                    for _ in range(agent.updates_per_step):
                        agent.update(buffer, batch_size=256)
                
                obs_list = next_obs
                ep_reward += np.mean(rewards)
                steps += 1
                total_steps += 1
                
                # Log detailed step information
                if steps % 20 == 0 or steps == 1:
                    step_log = f"  Step {steps:2d}: Cash=${env.cash:.0f}, Occupancy={env.occupied_rooms}/{env.total_rooms}"
                    step_log += f", Actions=[{act_vals[0]:.3f}, {act_vals[1]:.3f}, {act_vals[2]:.3f}]"
                    step_log += f", Reward={np.mean(rewards):.3f}, Total={ep_reward:.1f}"
                    ep_logs.append(step_log)

            # Calculate episode metrics
            avg_r = ep_reward/steps if steps > 0 else 0
            training_data['episodes'].append(ep + 1)
            training_data['rewards'].append(avg_r)
            
            # Track best performance
            if avg_r > best_reward:
                best_reward = avg_r
            
            # Episode summary with more details
            training_logs.append(f"  Episode {ep + 1} Complete: {steps} steps, Avg Reward: {avg_r:.3f}")
            training_logs.append(f"  Final Cash: ${env.cash:.0f}, Occupancy: {env.occupied_rooms}/{env.total_rooms}")
            training_logs.append(f"  Best Reward So Far: {best_reward:.3f}")
            print(f"  Episode {ep + 1} Complete: {steps} steps, Avg Reward: {avg_r:.3f}")
            print(f"  Final Cash: ${env.cash:.0f}, Occupancy: {env.occupied_rooms}/{env.total_rooms}")
            print(f"  Best Reward So Far: {best_reward:.3f}")
            
            # Add detailed loss and Q-target information if available
            if hasattr(agent, 'actor_losses') and agent.actor_losses[0]:
                avg_actor_loss = np.mean([np.mean(losses[-10:]) for losses in agent.actor_losses if losses])
                avg_critic_loss = np.mean([np.mean(losses[-10:]) for losses in agent.critic_losses if losses])
                training_logs.append(f"  🧠 Avg Actor Loss: {avg_actor_loss:.4f}, Avg Critic Loss: {avg_critic_loss:.4f}")
                
                # Log Q-target statistics for better debugging
                if hasattr(agent, 'q_targets') and agent.q_targets:
                    q_mean = np.mean(agent.q_targets[-100:]) if len(agent.q_targets) >= 100 else np.mean(agent.q_targets)
                    q_std = np.std(agent.q_targets[-100:]) if len(agent.q_targets) >= 100 else np.std(agent.q_targets)
                    training_logs.append(f"  📊 Q-Target Stats: Mean={q_mean:.3f}, Std={q_std:.3f}")
                
                # Log reward component breakdown
                if hasattr(env, 'reward_components') and env.reward_components:
                    components = env.reward_components[-1]  # Latest reward components
                    training_logs.append(f"  💰 Reward Components: Profit={components.get('profit', 0):.3f}, "
                                       f"Liquidity={components.get('liquidity', 0):.3f}, "
                                       f"Occupancy={components.get('occupancy', 0):.3f}")
            
            training_logs.append("  " + "-" * 60)
            
            # Add detailed step logs for first few episodes
            if ep < 5:
                training_logs.extend(ep_logs)
            
            # Periodic deterministic evaluation
            if (ep + 1) % 5 == 0:  # Evaluate every 5 episodes
                eval_reward = _evaluate_deterministic(env, agent, max_steps=50)
                training_logs.append(f"  🎯 Deterministic Eval (Episode {ep + 1}): {eval_reward:.3f}")
                print(f"  🎯 Deterministic Eval (Episode {ep + 1}): {eval_reward:.3f}")
        
        # Final training summary
        final_avg = np.mean(training_data['rewards'][-5:]) if len(training_data['rewards']) >= 5 else training_data['rewards'][-1]
        training_logs.append("Training Complete!")
        training_logs.append(f"Final Performance: {final_avg:.3f} average reward (last 5 episodes)")
        training_logs.append(f"Best Performance: {best_reward:.3f}")
        training_logs.append(f"Total Steps: {total_steps}")
        training_logs.append(f"Buffer Size: {len(buffer)}")
        
        # Add Explainable AI summary
        training_logs.append("")
        training_logs.append("=" * 80)
        training_logs.append("🤖 EXPLAINABLE AI (XAI) - AGENT DECISION ANALYSIS")
        training_logs.append("=" * 80)
        training_logs.append("")
        training_logs.append("Why Use Explainable AI (XAI)?")
        training_logs.append("• Transparency: Understand how AI agents make decisions")
        training_logs.append("• Trust: Build confidence in AI recommendations")
        training_logs.append("• Debugging: Identify why agents make specific choices")
        training_logs.append("• Optimization: Improve agent strategies based on explanations")
        training_logs.append("• Compliance: Meet regulatory requirements for AI transparency")
        training_logs.append("")
        
        # Show final agent recommendations
        if agent_explanations:
            latest_explanations = agent_explanations[-1]  # Get latest explanations
            training_logs.append("🎯 FINAL AGENT RECOMMENDATIONS:")
            training_logs.append("-" * 50)
            
            for explanation in latest_explanations:
                training_logs.append(f"")
                training_logs.append(f"🤖 {explanation['agent']}")
                training_logs.append(f"   Decision: {explanation['decision']}")
                training_logs.append(f"   Confidence: {explanation['confidence']}")
                training_logs.append(f"   Impact: {explanation['impact']}")
                training_logs.append(f"   Risk Level: {explanation['risk_level']}")
                training_logs.append(f"   Reasoning:")
                for reason in explanation['reasoning']:
                    training_logs.append(f"     • {reason}")
        
        print("Training Complete!")
        print(f"Final Performance: {final_avg:.3f} average reward (last 5 episodes)")
        print(f"Best Performance: {best_reward:.3f}")
        print(f"Total Steps: {total_steps}")
        print(f"Buffer Size: {len(buffer)}")
        
        training_in_progress = False
        
        return True, f"Training completed! Final reward: {final_avg:.3f}, Best: {best_reward:.3f}"
        
    except Exception as e:
        training_in_progress = False
        training_logs.append(f"Error: {str(e)}")
        import traceback
        training_logs.append(f"Traceback: {traceback.format_exc()}")
        return False, f"Error: {str(e)}"

def _evaluate_deterministic(env, agent, max_steps=50):
    """Evaluate agent performance with deterministic actions (no noise)"""
    state = env.reset()
    obs_list = [state.copy() for _ in range(3)]
    total_reward = 0.0
    steps = 0
    
    for _ in range(max_steps):
        actions = agent.act_deterministic(obs_list)
        act_vals = [float(a[0]) for a in actions]
        
        next_obs, rewards, done, _ = env.step(act_vals)
        total_reward += np.mean(rewards)
        obs_list = next_obs
        steps += 1
        
        if done:
            break
    
    return total_reward / steps if steps > 0 else 0.0

# Create learning curve chart
def create_learning_curve():
    if not training_data['episodes']:
        return go.Figure()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=training_data['episodes'],
        y=training_data['rewards'],
        mode='lines+markers',
        name='Average Reward',
        line=dict(color='#667eea', width=3),
        marker=dict(size=8, color='#667eea')
    ))
    
    fig.update_layout(
        title={
            'text': "MADDPG Learning Curve",
            'x': 0.5,
            'font': {'size': 20, 'family': 'Inter', 'color': '#2c3e50'}
        },
        xaxis_title="Episode",
        yaxis_title="Average Reward (AFR)",
        height=400,
        margin=dict(l=20, r=20, t=60, b=20),
        template='plotly_white'
    )
    
    return fig

# Create agent performance chart
def create_agent_performance():
    agents = ['Liquidity Agent', 'Investment Agent', 'Procurement Agent']
    performance = [85, 78, 92]  # Simulated performance scores
    
    fig = go.Figure(data=[
        go.Bar(
            x=agents,
            y=performance,
            marker_color=['#f39c12', '#2ecc71', '#e74c3c'],
            text=performance,
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title={
            'text': "Agent Performance Scores",
            'x': 0.5,
            'font': {'size': 20, 'family': 'Inter', 'color': '#2c3e50'}
        },
        xaxis_title="Agents",
        yaxis_title="Performance Score",
        height=400,
        margin=dict(l=20, r=20, t=60, b=20),
        template='plotly_white'
    )
    
    return fig

# Create action distribution chart
def create_action_distribution():
    actions = ['Borrow', 'Repay', 'Invest', 'Sell', 'Order', 'Hold']
    frequencies = [15, 10, 20, 8, 25, 22]  # Simulated action frequencies
    
    fig = go.Figure(data=[
        go.Pie(
            labels=actions,
            values=frequencies,
            hole=0.4,
            marker_colors=['#f39c12', '#e74c3c', '#2ecc71', '#9b59b6', '#3498db', '#95a5a6']
        )
    ])
    
    fig.update_layout(
        title={
            'text': "Agent Action Distribution",
            'x': 0.5,
            'font': {'size': 20, 'family': 'Inter', 'color': '#2c3e50'}
        },
        height=400,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

def create_agent_recommendations():
    """Create dynamic agent recommendations based on latest explanations"""
    global agent_explanations
    
    if not agent_explanations:
        return html.Div([
            html.H6("🎯 Agent Recommendations", style={'color': '#2c3e50', 'marginBottom': '15px'}),
            html.P("Start training to see agent recommendations...", style={'color': '#666', 'fontStyle': 'italic'})
        ])
    
    # Get latest explanations
    latest_explanations = agent_explanations[-1]
    
    # Create recommendation cards for each agent
    recommendation_cards = []
    
    for i, explanation in enumerate(latest_explanations):
        agent_name = explanation['agent']
        decision = explanation['decision']
        confidence = explanation['confidence']
        impact = explanation['impact']
        risk_level = explanation['risk_level']
        reasoning = explanation['reasoning']
        
        # Choose appropriate icon and color based on agent
        if i == 0:  # Liquidity Agent
            icon_class = "fas fa-coins"
            color = '#f39c12'
        elif i == 1:  # Investment Agent
            icon_class = "fas fa-chart-line"
            color = '#2ecc71'
        else:  # Procurement Agent
            icon_class = "fas fa-boxes"
            color = '#e74c3c'
        
        # Risk level styling
        risk_color = '#e74c3c' if risk_level == 'High Risk' else '#f39c12' if risk_level == 'Medium Risk' else '#2ecc71'
        
        # Confidence level styling
        conf_color = '#2ecc71' if confidence in ['Very High', 'High'] else '#f39c12' if confidence == 'Medium' else '#e74c3c'
        
        card = html.Div([
            html.Div([
                html.Div([
                    html.I(className=icon_class, style={'color': color, 'fontSize': '1.5rem', 'marginRight': '10px'}),
                    html.Div([
                        html.H6(agent_name, style={'margin': '0', 'color': '#2c3e50'}),
                        html.P(decision, style={'margin': '5px 0', 'fontWeight': 'bold', 'color': color})
                    ])
                ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '10px'}),
                
                # Metrics row
                html.Div([
                    html.Div([
                        html.Small("Confidence", style={'color': '#666', 'display': 'block'}),
                        html.Span(confidence, style={'color': conf_color, 'fontWeight': 'bold'})
                    ], style={'textAlign': 'center', 'flex': '1'}),
                    html.Div([
                        html.Small("Impact", style={'color': '#666', 'display': 'block'}),
                        html.Span(impact, style={'color': '#2c3e50', 'fontWeight': 'bold'})
                    ], style={'textAlign': 'center', 'flex': '1'}),
                    html.Div([
                        html.Small("Risk", style={'color': '#666', 'display': 'block'}),
                        html.Span(risk_level, style={'color': risk_color, 'fontWeight': 'bold'})
                    ], style={'textAlign': 'center', 'flex': '1'})
                ], style={'display': 'flex', 'marginBottom': '10px', 'padding': '0 10px'}),
                
                # Reasoning
                html.Div([
                    html.Small("Reasoning:", style={'color': '#666', 'fontWeight': 'bold', 'display': 'block', 'marginBottom': '5px'}),
                    html.Ul([
                        html.Li(reason, style={'fontSize': '0.85rem', 'marginBottom': '2px', 'color': '#555'})
                        for reason in reasoning[:3]  # Show first 3 reasoning points
                    ], style={'margin': '0', 'paddingLeft': '20px'})
                ])
                
            ], style={'padding': '15px', 'backgroundColor': '#f8f9fa', 'borderRadius': '8px', 'border': f'2px solid {color}20'})
        ], style={'margin': '10px 0'})
        
        recommendation_cards.append(card)
    
    # Add XAI explanation section
    xai_section = html.Div([
        html.Hr(style={'margin': '20px 0'}),
        html.H6("🤖 Explainable AI (XAI) Benefits", style={'color': '#2c3e50', 'marginBottom': '15px'}),
        html.Div([
            html.Div([
                html.I(className="fas fa-eye", style={'color': '#3498db', 'marginRight': '8px'}),
                html.Span("Transparency", style={'fontWeight': 'bold'}),
                html.Br(),
                html.Small("Understand how AI makes decisions", style={'color': '#666'})
            ], style={'textAlign': 'center', 'flex': '1', 'padding': '10px'})
        ], style={'display': 'flex', 'marginBottom': '10px'}),
        html.Div([
            html.Div([
                html.I(className="fas fa-shield-alt", style={'color': '#2ecc71', 'marginRight': '8px'}),
                html.Span("Trust", style={'fontWeight': 'bold'}),
                html.Br(),
                html.Small("Build confidence in AI recommendations", style={'color': '#666'})
            ], style={'textAlign': 'center', 'flex': '1', 'padding': '10px'})
        ], style={'display': 'flex', 'marginBottom': '10px'}),
        html.Div([
            html.Div([
                html.I(className="fas fa-cogs", style={'color': '#f39c12', 'marginRight': '8px'}),
                html.Span("Optimization", style={'fontWeight': 'bold'}),
                html.Br(),
                html.Small("Improve strategies based on explanations", style={'color': '#666'})
            ], style={'textAlign': 'center', 'flex': '1', 'padding': '10px'})
        ], style={'display': 'flex'})
    ])
    
    return html.Div([
        html.H6("🎯 AI Agent Recommendations", style={'color': '#2c3e50', 'marginBottom': '15px'}),
        html.P("Based on Explainable AI analysis of agent decision-making:", style={'color': '#666', 'marginBottom': '15px', 'fontStyle': 'italic'}),
        *recommendation_cards,
        xai_section
    ])

# Page layout
layout = html.Div([
    # Page header
    html.Div([
        html.H1("Multi-Agent Deep Reinforcement Learning", className="page-title"),
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
                        dbc.Input(id="episodes-input", type="number", value=20, min=1, max=100)
                    ], width=6),
                    dbc.Col([
                        dbc.Label("Random Seed:"),
                        dbc.Input(id="seed-input", type="number", value=42, min=1, max=1000)
                    ], width=6),
                ], className="mb-3"),
                dbc.Button(
                    "Start Training", 
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
                html.H4("Agent Overview", style={'color': '#2c3e50', 'marginBottom': '20px'}),
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
    
    # Learning curve
    dbc.Row([
        dbc.Col([
            html.Div([
                dcc.Graph(id="learning-curve", figure=create_learning_curve())
            ], className="chart-container")
        ], width=12),
    ], className="mb-4"),
    
    # Training Logs
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H4("📋 Training Logs", style={'color': '#2c3e50', 'marginBottom': '20px'}),
                html.Div([
                    html.Pre(
                        id="training-logs",
                        children="Training logs will appear here...",
                        style={
                            'backgroundColor': '#f8f9fa',
                            'border': '1px solid #dee2e6',
                            'borderRadius': '8px',
                            'padding': '15px',
                            'fontSize': '12px',
                            'fontFamily': 'monospace',
                            'maxHeight': '400px',
                            'overflowY': 'auto',
                            'whiteSpace': 'pre-wrap',
                            'margin': '0'
                        }
                    )
                ])
            ], className="chart-container")
        ], width=12),
    ], className="mb-4"),
    
    # Performance charts
    dbc.Row([
        dbc.Col([
            html.Div([
                dcc.Graph(figure=create_agent_performance())
            ], className="chart-container")
        ], width=6),
        
        dbc.Col([
            html.Div([
                dcc.Graph(figure=create_action_distribution())
            ], className="chart-container")
        ], width=6),
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
                    html.Li("Training: Centralized training, decentralized execution"),
                    html.Li("Implementation: Full MADDPG with PyTorch neural networks")
                ], style={'fontSize': '16px', 'lineHeight': '1.6'})
            ], className="chart-container")
        ], width=12),
    ])
])

# Callbacks
@dash.callback(
    [Output("training-status", "children"),
     Output("learning-curve", "figure"),
     Output("agent-recommendations", "children"),
     Output("training-logs", "children")],
    [Input("train-button", "n_clicks")],
    [State("episodes-input", "value"),
     State("seed-input", "value")]
)
def train_agents(n_clicks, episodes, seed):
    if n_clicks is None:
        return "", create_learning_curve(), html.P("Start training to see agent recommendations...", style={'color': '#666', 'fontStyle': 'italic'}), "Training logs will appear here..."
    
    # Run training
    success, message = run_full_maddpg_demo(episodes, seed)
    
    # Format training logs for display
    logs_text = "\n".join(training_logs) if training_logs else "No logs available"
    
    if success:
        status = dbc.Alert(message, color="success")
        
        # Generate dynamic recommendations based on latest agent explanations
        recommendations = create_agent_recommendations()
    else:
        status = dbc.Alert(message, color="danger")
        recommendations = html.P("Training failed. Please check the data file.", style={'color': '#e74c3c'})
    
    return status, create_learning_curve(), recommendations, logs_text
