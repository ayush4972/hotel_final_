# 🏨 Hotel Reservation Analytics Dashboard

A comprehensive, modern hotel reservation analytics dashboard featuring multi-agent deep reinforcement learning, AI-powered forecasting, and sentiment analysis.

## ✨ Features

### 🏠 **Dashboard Overview**
- Real-time hotel booking metrics and KPIs
- Interactive revenue and booking trend visualizations
- Booking status and room type distribution analysis
- Market segment performance insights

### 📊 **Advanced Analytics**
- Lead time vs cancellation rate analysis
- Seasonal booking pattern analysis
- Market segment profitability analysis
- Customer behavior insights (new vs repeat guests)

### 🤖 **Multi-Agent Deep Reinforcement Learning (MADDPG)**
- **Liquidity Agent**: Manages cash flow, borrowing, and debt repayment
- **Investment Agent**: Optimizes asset allocation and marketing investments
- **Procurement Agent**: Manages inventory, amenities, and supply chain
- Real-time training with interactive controls
- Learning curve visualization

### 🔮 **AI-Powered Forecasting**
- XGBoost machine learning models for booking predictions
- 30-day future forecast with confidence intervals
- Feature importance analysis
- Model performance metrics (RMSE)

### 💝 **Customer Sentiment Analysis**
- AI-powered sentiment classification of customer feedback
- Sentiment distribution and trend analysis
- Correlation between sentiment and booking metrics
- Market segment sentiment analysis

### 📋 **Comprehensive Reports**
- Daily, weekly, monthly, and full reports
- Interactive data tables with conditional styling
- Summary statistics and key performance indicators
- Export-ready report formats

## 🚀 Quick Start
# 🏨 Hotel Reservation Analytics Dashboard

A comprehensive, modern hotel reservation analytics dashboard featuring multi-agent deep reinforcement learning, AI-powered forecasting, and sentiment analysis.

## ✨ Features

### 🏠 **Dashboard Overview**
- Real-time hotel booking metrics and KPIs
- Interactive revenue and booking trend visualizations
- Booking status and room type distribution analysis
- Market segment performance insights

### 📊 **Advanced Analytics**
- Lead time vs cancellation rate analysis
- Seasonal booking pattern analysis
- Market segment profitability analysis
- Customer behavior insights (new vs repeat guests)

### 🤖 **Multi-Agent Deep Reinforcement Learning (MADDPG)**
- **Liquidity Agent**: Manages cash flow, borrowing, and debt repayment
- **Investment Agent**: Optimizes asset allocation and marketing investments
- **Procurement Agent**: Manages inventory, amenities, and supply chain
- Real-time training with interactive controls
- Learning curve visualization

### 🔮 **AI-Powered Forecasting**
- XGBoost machine learning models for booking predictions
- 30-day future forecast with confidence intervals
- Feature importance analysis
- Model performance metrics (RMSE)

### 💝 **Customer Sentiment Analysis**
- AI-powered sentiment classification of customer feedback
- Sentiment distribution and trend analysis
- Correlation between sentiment and booking metrics
- Market segment sentiment analysis

### 📋 **Comprehensive Reports**
- NOTE: The interactive "Reports" page has been removed from the sidebar frontend in this branch.
  The underlying `pages/reports.py` file remains available if you want to re-enable it later.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Hotel Reservations.csv dataset

### Installation

1. **Clone or download the project**
```bash
cd hotel_dashboard_new
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Place your dataset**
   - Copy `Hotel Reservations.csv` to the project root directory

4. **Run the application**
```bash
python app.py
```

5. **Access the dashboard**
   - Open your browser and go to `http://localhost:8050`

## 📁 Project Structure

```
hotel_dashboard_new/
├── app.py                          # Main Dash application
├── requirements.txt                # Python dependencies
├── README.md                      # This file
├── pages/                         # Dashboard pages
│   ├── __init__.py
│   ├── home.py                    # Dashboard overview
│   ├── analytics.py               # Advanced analytics
│   ├── maddpg_demo.py            # Multi-agent DRL demo
│   ├── forecasting.py            # AI forecasting
│   └── sentiment_analysis.py     # Sentiment analysis
├── Dockerfile                      # Docker image build
├── docker-compose.yml              # Compose file to run the app
└── Hotel Reservations.csv        # Dataset (place here)
```

## 🎯 Key Components

### Multi-Agent DRL System
- **Environment**: HotelSMEEnv with 12-dimensional state space
- **Agents**: 3 specialized agents for different business functions
- **Algorithm**: MADDPG (Multi-Agent Deep Deterministic Policy Gradient)
- **Reward Function**: Adaptive Financial Reward (AFR)

### AI Forecasting
- **Model**: XGBoost Regressor
- **Features**: Lag features (1-day, 7-day, 14-day)
- **Forecast Horizon**: 30 days
- **Performance**: RMSE-based evaluation

### Sentiment Analysis
- **Method**: Rule-based classification with keyword matching
- **Categories**: Positive, Neutral, Negative
- **Scope**: 3,000 hotel reservation records
- **Integration**: Correlates with booking metrics

## 🎨 UI Features

- **Modern Design**: Gradient backgrounds, glassmorphism effects
- **Responsive Layout**: Works on desktop, tablet, and mobile
- **Interactive Charts**: Plotly-powered visualizations
- **Real-time Updates**: Dynamic data loading and processing
- **Professional Styling**: Bootstrap components with custom CSS

## 📊 Data Processing

The dashboard automatically:
- Loads and validates the Hotel Reservations.csv dataset
- Handles invalid dates (e.g., February 30th)
- Creates derived features for analysis
- Generates sentiment scores from customer feedback
- Processes data for multi-agent training

## 🔧 Technical Details

### Dependencies
- **Dash**: Web application framework
- **Plotly**: Interactive visualizations
- **Pandas**: Data manipulation
- **XGBoost**: Machine learning forecasting
- **PyTorch**: Deep learning for MADDPG
- **Scikit-learn**: ML utilities

### Performance
- Optimized data loading and caching
- Efficient chart rendering
- Responsive UI components
- Memory-efficient data processing

## 🎮 Usage Guide

1. **Dashboard Overview**: Start here for high-level metrics
2. **Analytics**: Dive deep into booking patterns and correlations
3. **Multi-Agent DRL**: Train and observe AI agents in action
4. **AI Forecasting**: View booking predictions and model performance
5. **Sentiment Analysis**: Understand customer satisfaction trends

## 🐳 Docker

To run the app with Docker and docker-compose (recommended for consistent environments):

1. Build and start with docker-compose:

```bash
docker-compose up --build
```

2. Open your browser at `http://localhost:8050`.

Notes:
- The service exposes port 8050. The Docker image installs Python packages from `requirements.txt`.
- The repository root is mounted into the container (useful for development). Remove the volume in `docker-compose.yml` for a production image.

## 🚀 Future Enhancements

- Blockchain integration for transaction logging
- Real-time data streaming
- Advanced XAI (Explainable AI) features
- Mobile app version
- API endpoints for external integrations

## 📝 Notes

- The dashboard automatically creates `Hotel_Reservations_Modified.csv` with sentiment analysis
- Multi-agent training can be resource-intensive; adjust episode count as needed
- All visualizations are interactive and exportable
- The system handles missing or invalid data gracefully

## 🤝 Support

For questions or issues, please check the console output for detailed error messages and ensure your dataset is properly formatted.

---

**Built with ❤️ using Dash, Plotly, and modern web technologies**
# hotel_final_
