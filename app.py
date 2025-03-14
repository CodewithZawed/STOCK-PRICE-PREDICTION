from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime, timedelta

app = Flask(__name__)

def get_stock_data(symbol, period='1y'):
    try:
        # Convert symbol to uppercase
        symbol = symbol.upper()
        print(f"Fetching data for symbol: {symbol}")
        
        # Try different symbol formats
        symbol_variations = [symbol, f"{symbol}.US", f"{symbol}.O"]
        
        for sym in symbol_variations:
            try:
                print(f"Attempting to fetch data for {sym}")
                stock = yf.Ticker(sym)
                
                # Try to get info first to verify the ticker exists
                info = stock.info
                if info:
                    print(f"Found valid ticker: {sym}")
                    print(f"Company name: {info.get('longName', 'N/A')}")
                    
                    # Get historical data
                    df = stock.history(period=period)
                    
                    if not df.empty:
                        print(f"Successfully fetched {len(df)} rows of data")
                        print(f"Date range: {df.index[0]} to {df.index[-1]}")
                        return df
                    else:
                        print(f"No historical data found for {sym}")
                else:
                    print(f"No info found for {sym}")
                    
            except Exception as e:
                print(f"Error with symbol {sym}: {str(e)}")
                continue
        
        print(f"No data found for any variation of symbol: {symbol}")
        return None
        
    except Exception as e:
        print(f"Error in get_stock_data: {str(e)}")
        return None

def prepare_data(df):
    df['Target'] = df['Close'].shift(-1)
    df = df.dropna()
    
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    X = df[features]
    y = df['Target']
    
    return X, y

def train_models(X, y):
    try:
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale the features
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train models
        lr_model = LinearRegression()
        svm_model = SVR(kernel='rbf')
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        lr_model.fit(X_train_scaled, y_train)
        svm_model.fit(X_train_scaled, y_train)
        rf_model.fit(X_train_scaled, y_train)
        
        # Get predictions
        lr_pred = lr_model.predict(X_test_scaled)
        svm_pred = svm_model.predict(X_test_scaled)
        rf_pred = rf_model.predict(X_test_scaled)
        
        return {
            'lr': {'model': lr_model, 'predictions': lr_pred, 'test': y_test},
            'svm': {'model': svm_model, 'predictions': svm_pred, 'test': y_test},
            'rf': {'model': rf_model, 'predictions': rf_pred, 'test': y_test}
        }
        
    except Exception as e:
        print(f"Error in train_models: {str(e)}")
        raise e

def plot_predictions(models, symbol):
    try:
        plt.figure(figsize=(12, 6))
        
        for name, model_data in models.items():
            plt.plot(model_data['test'].values, label=f'{name.upper()} Actual')
            plt.plot(model_data['predictions'], label=f'{name.upper()} Predicted')
        
        plt.title(f'Stock Price Predictions for {symbol}')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        
        # Save plot to base64 string
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        return plot_url
        
    except Exception as e:
        print(f"Error in plot_predictions: {str(e)}")
        raise e

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    symbol = request.form['symbol']
    
    try:
        # Get stock data
        df = get_stock_data(symbol)
        if df is None or df.empty:
            return jsonify({
                'error': f'No data found for the symbol {symbol}. Please check if the symbol is correct and try again. Common symbols: AAPL, GOOGL, MSFT, AMZN',
                'success': False
            })
        
        # Prepare data
        X, y = prepare_data(df)
        
        # Train models
        models = train_models(X, y)
        
        # Generate plot
        plot_url = plot_predictions(models, symbol)
        
        # Calculate RMSE for each model
        results = {}
        for name, model_data in models.items():
            rmse = np.sqrt(np.mean((model_data['predictions'] - model_data['test']) ** 2))
            results[name] = {
                'rmse': round(rmse, 2),
                'last_prediction': round(model_data['predictions'][-1], 2)
            }
        
        return jsonify({
            'plot': plot_url,
            'results': results,
            'success': True
        })
    
    except Exception as e:
        print(f"Error in predict route: {str(e)}")
        return jsonify({
            'error': f'An error occurred: {str(e)}',
            'success': False
        })

if __name__ == '__main__':
    app.run(debug=True) 