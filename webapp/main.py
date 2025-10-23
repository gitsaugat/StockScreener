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



if __name__ == '__main__':

    # Initialize database
    with app.app_context():
        db.create_all()

    app.run(debug=True,host='0.0.0.0',port=5001)