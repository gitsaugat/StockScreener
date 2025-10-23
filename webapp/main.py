from flask import Flask, render_template , request, jsonify,session
from data.tickers import DEFAULT_TICKERS
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
import yfinance as yf
from models.tickers import Ticker,HistoricalData,db
from data.analysis import StockAnalyzer, get_recommendation_score
import os
import dotenv
import pandas as pd
import json
from datetime import datetime
import io
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import numpy as np

app = Flask(__name__) 

dotenv.load_dotenv('.env')
DB_USER = os.environ.get("DB_USERNAME")
DB_PASS = os.environ.get("DB_PASSWORD")
DB_HOST = os.environ.get("DB_HOST")
DB_PORT = os.environ.get("DB_PORT")
DB_NAME = os.environ.get("DB_NAME")

app.config['SQLALCHEMY_DATABASE_URI'] = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'your-secret-key-here'

db.init_app(app)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/ticker/<symbol>')
def ticker_detail(symbol):
    """Ticker detail page with comprehensive analysis"""
    return render_template('ticker_detail.html', symbol=symbol.upper())

@app.route('/api/set_ai_mode', methods=['POST'])
def set_ai_mode():
    data = request.json
    ai_mode = data.get('ai_mode', 'off')  # 'on' or 'off'
    
    if ai_mode not in ['on', 'off']:
        return jsonify({'error': 'AI mode must be "on" or "off"'}), 400
    
    session['ai_mode'] = ai_mode
    return jsonify({
        'message': 'AI mode updated',
        'ai_mode': ai_mode
    })

@app.route('/api/get-ai-mode', methods=['GET'])
def get_ai_mode():
    ai_mode = session.get('ai_mode', 'off')  # Default to 'off'
    return jsonify({'ai_mode': ai_mode})


@app.route('/api/initialize_tickers', methods=['POST'])
def initialize_tickers():
    """Initialize database with default tickers if they don't exist"""
    added_tickers = []
    skipped_tickers = []
    
    for symbol in DEFAULT_TICKERS:
        existing_ticker = Ticker.get_by_symbol(symbol)
        
        if existing_ticker:
            skipped_tickers.append(symbol)
            continue
        
        ticker = Ticker.create_from_yfinance(symbol)
        
        if ticker:
            added_tickers.append(ticker.to_dict())
    
    return jsonify({
        'success': True,
        'added': len(added_tickers),
        'skipped': len(skipped_tickers),
        'tickers': added_tickers
    })


@app.route('/api/get_tickers', methods=['GET'])
def get_tickers():
    """Get all tickers from database"""
    tickers = Ticker.get_all()
    return jsonify({
        'tickers': [ticker.to_dict() for ticker in tickers]
    })


@app.route('/api/search_ticker', methods=['POST'])
def search_ticker():
    """Search for a ticker and return its information"""
    data = request.json
    ticker_symbol = data.get('ticker', '').upper()
    
    if not ticker_symbol:
        return jsonify({'success': False, 'error': 'Ticker symbol required'})
    
    try:
        ticker_obj = yf.Ticker(ticker_symbol)
        info = ticker_obj.info
        
        if not info.get('symbol'):
            return jsonify({'success': False, 'error': 'Ticker not found'})
        
        ticker_data = {
            'symbol': ticker_symbol,
            'name': info.get('longName', ticker_symbol),
            'price': info.get('currentPrice', info.get('regularMarketPrice', 0)),
            'change': info.get('regularMarketChangePercent', 0),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'market_cap': info.get('marketCap', 0),
            'volume': info.get('volume', 0)
        }
        
        return jsonify({
            'success': True,
            'ticker': ticker_data
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/add_ticker', methods=['POST'])
def add_ticker():
    """Add a new ticker to the database"""
    data = request.json
    ticker_symbol = data.get('ticker', '').upper()
    
    if not ticker_symbol:
        return jsonify({'success': False, 'error': 'Ticker symbol required'})
    
    existing_ticker = Ticker.get_by_symbol(ticker_symbol)
    if existing_ticker:
        return jsonify({'success': False, 'error': 'Ticker already exists in database'})
    
    ticker = Ticker.create_from_yfinance(ticker_symbol)
    
    if ticker:
        return jsonify({
            'success': True,
            'ticker': ticker.to_dict()
        })
    else:
        return jsonify({'success': False, 'error': 'Failed to add ticker'})


@app.route('/api/remove_ticker', methods=['POST'])
def remove_ticker():
    """Remove a ticker from the database"""
    data = request.json
    ticker_symbol = data.get('ticker', '').upper()
    
    ticker = Ticker.get_by_symbol(ticker_symbol)
    
    if not ticker:
        return jsonify({'success': False, 'error': 'Ticker not found'})
    
    ticker.delete()
    return jsonify({'success': True})


@app.route('/api/update_ticker/<symbol>', methods=['POST'])
def update_ticker(symbol):
    """Update a specific ticker's data"""
    ticker = Ticker.get_by_symbol(symbol)
    
    if not ticker:
        return jsonify({'success': False, 'error': 'Ticker not found'})
    
    success = ticker.update_from_yfinance()
    
    if success:
        return jsonify({
            'success': True,
            'ticker': ticker.to_dict()
        })
    else:
        return jsonify({'success': False, 'error': 'Failed to update ticker'})


@app.route('/api/update_all_tickers', methods=['POST'])
def update_all_tickers():
    """Update all tickers in the database"""
    tickers = Ticker.get_all()
    updated = 0
    failed = 0
    
    for ticker in tickers:
        if ticker.update_from_yfinance():
            updated += 1
        else:
            failed += 1
    
    return jsonify({
        'success': True,
        'updated': updated,
        'failed': failed
    })


@app.route('/api/historical/<symbol>', methods=['GET'])
def get_historical_data(symbol):
    """Get historical data for a ticker"""
    ticker = Ticker.get_by_symbol(symbol)
    
    if not ticker:
        return jsonify({'success': False, 'error': 'Ticker not found'})
    
    days = request.args.get('days', default=30, type=int)
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days)
    
    # Check if we have historical data
    historical = HistoricalData.get_by_ticker_and_date_range(
        ticker.id, start_date, end_date
    )
    
    # If no data, fetch from yfinance
    if not historical:
        HistoricalData.create_from_yfinance(
            ticker.id, ticker.symbol, start_date, end_date
        )
        historical = HistoricalData.get_by_ticker_and_date_range(
            ticker.id, start_date, end_date
        )
    
    return jsonify({
        'success': True,
        'symbol': symbol,
        'data': [h.to_dict() for h in historical]
    })


@app.route('/api/analyze/<symbol>', methods=['GET'])
def analyze_ticker(symbol):
    """Get comprehensive analysis for a ticker"""
    try:
        analyzer = StockAnalyzer(symbol, gemini_api_key=os.getenv('GEMINI_API_KEY') if session.get('ai_mode') == 'on' else None)
        analysis = analyzer.get_comprehensive_analysis()
        
        # Add recommendation score
        recommendation = get_recommendation_score(analysis)
        analysis['recommendation'] = recommendation
        
        return jsonify({
            'success': True,
            'symbol': symbol,
            'analysis': analysis
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/api/news/<symbol>', methods=['GET'])
def get_ticker_news(symbol):
    """Get news for a ticker"""
    try:
        analyzer = StockAnalyzer(symbol, gemini_api_key=os.getenv('GEMINI_API_KEY') if session.get('ai_mode') == 'on' else None)
        news = analyzer.get_news(limit=20)
        
        return jsonify({
            'success': True,
            'symbol': symbol,
            'news': news
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/api/news_sentiment/<symbol>', methods=['GET'])
def get_news_sentiment(symbol):
    """Get news sentiment analysis"""
    try:
        analyzer = StockAnalyzer(symbol, gemini_api_key=os.getenv('GEMINI_API_KEY') if session.get('ai_mode') == 'on' else None)
        
        sentiment = analyzer.analyze_news_sentiment()
        
        return jsonify({
            'success': True,
            'symbol': symbol,
            'sentiment': sentiment
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/analyze/<ticker>')
def analyze(ticker):
    ticker = ticker.upper()
    analyzer = StockAnalyzer(ticker, gemini_api_key=os.getenv('GEMINI_API_KEY'))

    # Fetch stock data
    stock_data = analyzer.get_stock_data(ticker)
    
    if 'error' in stock_data:
        return render_template('error.html', error=stock_data['error'], ticker=ticker)
    
    # Analyze with Gemini
    analysis = analyzer.analyze_with_gemini(stock_data)
    
    return render_template('analysis.html', stock_data=stock_data, analysis=analysis)

@app.route('/api/analyze/<ticker>')
def api_analyze(ticker):

    ticker = ticker.upper()
    analyzer = StockAnalyzer(ticker, gemini_api_key=os.getenv('GEMINI_API_KEY'))

    stock_data = analyzer.get_stock_data(ticker)
    
    if 'error' in stock_data:
        return jsonify({'error': stock_data['error']}), 400
    
    analysis = analyzer.analyze_with_gemini(stock_data)
    
    return jsonify({
        'stock_data': stock_data,
        'analysis': analysis
    })




@app.route('/stock/predict/<ticker>')
def predict_stock(ticker):
    """Route to display stock prediction page"""
    return render_template('stock_prediction.html', ticker=ticker.upper())

@app.route('/api/stock/historical/<ticker>')
def get_hist_data(ticker):
    """API endpoint to get historical data and predictions"""
    try:
        # Get period from query params (default 1 year)
        period = request.args.get('period', '1y')
        
        # Fetch stock data
        stock = yf.Ticker(ticker.upper())
        hist = stock.history(period=period)
        info = stock.info
        
        if hist.empty:
            return jsonify({'error': 'No data available for this ticker'}), 404
        
        # Prepare historical data
        historical_data = []
        for date, row in hist.iterrows():
            historical_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'open': round(row['Open'], 2),
                'high': round(row['High'], 2),
                'low': round(row['Low'], 2),
                'close': round(row['Close'], 2),
                'volume': int(row['Volume'])
            })
        
        # Perform price prediction
        predictions = predict_prices(hist)
        
        # Calculate statistics
        current_price = hist['Close'][-1]
        price_change = hist['Close'][-1] - hist['Close'][0]
        price_change_pct = (price_change / hist['Close'][0]) * 100
        
        # Volatility
        returns = hist['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) * 100
        
        # Moving averages
        ma_20 = hist['Close'].rolling(window=20).mean()
        ma_50 = hist['Close'].rolling(window=50).mean()
        
        stats = {
            'ticker': ticker.upper(),
            'company_name': info.get('longName', ticker.upper()),
            'current_price': round(current_price, 2),
            'price_change': round(price_change, 2),
            'price_change_pct': round(price_change_pct, 2),
            'high': round(hist['High'].max(), 2),
            'low': round(hist['Low'].min(), 2),
            'avg_volume': int(hist['Volume'].mean()),
            'volatility': round(volatility, 2),
            'ma_20': round(ma_20[-1], 2) if len(ma_20) > 0 else None,
            'ma_50': round(ma_50[-1], 2) if len(ma_50) > 0 else None,
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A')
        }
        
        return jsonify({
            'success': True,
            'stats': stats,
            'historical': historical_data,
            'predictions': predictions,
            'moving_averages': {
                'ma_20': [{'date': date.strftime('%Y-%m-%d'), 'value': round(val, 2)} 
                         for date, val in zip(hist.index, ma_20) if not np.isnan(val)],
                'ma_50': [{'date': date.strftime('%Y-%m-%d'), 'value': round(val, 2)} 
                         for date, val in zip(hist.index, ma_50) if not np.isnan(val)]
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def predict_prices(hist_data, days_ahead=30):
    """
    Simple price prediction using linear regression and moving average trends
    This is a basic model - for production, consider using more sophisticated models
    """
    try:
        # Prepare data
        df = hist_data.copy()
        df['Days'] = range(len(df))
        
        # Use multiple features for prediction
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['Volatility'] = df['Close'].rolling(window=10).std()
        df['Returns'] = df['Close'].pct_change()
        
        # Drop NaN values
        df = df.dropna()
        
        # Prepare features and target
        features = ['Days', 'MA_5', 'MA_20', 'Volatility', 'Volume']
        X = df[features].values
        y = df['Close'].values
        
        # Scale features
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))
        
        # Train model
        model = LinearRegression()
        model.fit(X_scaled, y_scaled)
        
        # Generate future predictions
        last_date = df.index[-1]
        predictions = []
        
        # Get last known values
        last_ma5 = df['MA_5'].iloc[-1]
        last_ma20 = df['MA_20'].iloc[-1]
        last_vol = df['Volatility'].iloc[-1]
        last_volume = df['Volume'].iloc[-1]
        last_days = df['Days'].iloc[-1]
        
        for i in range(1, days_ahead + 1):
            future_date = last_date + timedelta(days=i)
            
            # Create feature vector (simplified - assumes relatively stable metrics)
            future_features = np.array([[
                last_days + i,
                last_ma5,
                last_ma20,
                last_vol,
                last_volume
            ]])
            
            future_features_scaled = scaler_X.transform(future_features)
            pred_scaled = model.predict(future_features_scaled)
            pred_price = scaler_y.inverse_transform(pred_scaled)[0][0]
            
            predictions.append({
                'date': future_date.strftime('%Y-%m-%d'),
                'predicted_price': round(pred_price, 2),
                'confidence': 'medium' if i <= 7 else 'low'
            })
            
            # Update moving averages for next prediction (simplified)
            last_ma5 = (last_ma5 * 4 + pred_price) / 5
            last_ma20 = (last_ma20 * 19 + pred_price) / 20
        
        # Calculate prediction trend
        if len(predictions) > 0:
            trend_change = predictions[-1]['predicted_price'] - df['Close'].iloc[-1]
            trend_pct = (trend_change / df['Close'].iloc[-1]) * 100
            
            return {
                'predictions': predictions,
                'trend': 'bullish' if trend_change > 0 else 'bearish',
                'trend_change': round(trend_change, 2),
                'trend_change_pct': round(trend_pct, 2),
                'model': 'Linear Regression with Technical Indicators',
                'note': 'Predictions are estimates based on historical patterns and should not be used as sole investment advice.'
            }
        
        return {'predictions': [], 'error': 'Unable to generate predictions'}
        
    except Exception as e:
        return {'predictions': [], 'error': str(e)}
    
if __name__ == '__main__':

    # Initialize database
    with app.app_context():
        db.create_all()

    app.run(debug=True,host='0.0.0.0',port=5001)