# Stock Price Prediction Web App

This web application predicts stock prices using multiple machine learning algorithms: Linear Regression, Support Vector Machine (SVM), and Random Forest. The application uses historical stock data from Yahoo Finance and provides a user-friendly interface to visualize predictions.

## Features

- Real-time stock data fetching from Yahoo Finance
- Multiple ML models for price prediction
- Interactive web interface
- Visualization of actual vs predicted prices
- Model performance metrics (RMSE)
- Responsive design

## Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

## Installation

1. Clone this repository or download the files
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start the Flask application:
   ```bash
   python app.py
   ```
2. Open your web browser and navigate to `http://localhost:5000`
3. Enter a stock symbol (e.g., AAPL, GOOGL, MSFT) and click "Predict"
4. View the predictions and model performance metrics

## Technologies Used

- Python
- Flask (Web Framework)
- NumPy (Numerical Computing)
- Pandas (Data Manipulation)
- scikit-learn (Machine Learning)
- Matplotlib (Data Visualization)
- yfinance (Yahoo Finance API)
- Bootstrap (UI Framework)

## Note

This application is for educational purposes only. Stock price predictions are inherently uncertain and should not be used as the sole basis for making investment decisions. 