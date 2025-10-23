from flask_sqlalchemy import SQLAlchemy

from datetime import datetime, timedelta
import yfinance as yf
from sqlalchemy import desc


db = SQLAlchemy()

class Ticker(db.Model):
    __tablename__ = 'tickers'
    
    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(10), unique=True, nullable=False, index=True)
    company_name = db.Column(db.String(255))
    current_price = db.Column(db.Float)
    change_percent = db.Column(db.Float)
    sector = db.Column(db.String(100))
    industry = db.Column(db.String(100))
    market_cap = db.Column(db.Float)
    volume = db.Column(db.BigInteger)
    last_updated = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)
    created_at = db.Column(db.DateTime, default=datetime.now)
    
    # Relationship
    historical_data = db.relationship('HistoricalData', backref='ticker', lazy='dynamic', cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<Ticker {self.symbol}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'symbol': self.symbol,
            'name': self.company_name,
            'price': self.current_price,
            'change': self.change_percent,
            'sector': self.sector,
            'industry': self.industry,
            'market_cap': self.market_cap,
            'volume': self.volume,
            'last_updated': self.last_updated.strftime('%Y-%m-%d %H:%M:%S') if self.last_updated else None
        }
    
    @classmethod
    def get_all(cls):
        return cls.query.order_by(cls.symbol).all()
    
    @classmethod
    def get_by_symbol(cls, symbol):
        return cls.query.filter_by(symbol=symbol.upper()).first()
    
    @classmethod
    def create_from_yfinance(cls, symbol):
        """Create a ticker from yfinance data"""
        try:
            ticker_obj = yf.Ticker(symbol)
            info = ticker_obj.info
            
            if not info.get('symbol'):
                return None
            
            ticker = cls(
                symbol=symbol.upper(),
                company_name=info.get('longName', symbol),
                current_price=info.get('currentPrice', info.get('regularMarketPrice', 0)),
                change_percent=info.get('regularMarketChangePercent', 0),
                sector=info.get('sector', 'N/A'),
                industry=info.get('industry', 'N/A'),
                market_cap=info.get('marketCap', 0),
                volume=info.get('volume', 0)
            )
            
            db.session.add(ticker)
            db.session.commit()
            return ticker
        except Exception as e:
            db.session.rollback()
            print(f"Error creating ticker {symbol}: {e}")
            return None
    
    def update_from_yfinance(self):
        """Update ticker data from yfinance"""
        try:
            ticker_obj = yf.Ticker(self.symbol)
            info = ticker_obj.info
            
            self.company_name = info.get('longName', self.symbol)
            self.current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
            self.change_percent = info.get('regularMarketChangePercent', 0)
            self.sector = info.get('sector', 'N/A')
            self.industry = info.get('industry', 'N/A')
            self.market_cap = info.get('marketCap', 0)
            self.volume = info.get('volume', 0)
            self.last_updated = datetime.now()
            
            db.session.commit()
            return True
        except Exception as e:
            db.session.rollback()
            print(f"Error updating ticker {self.symbol}: {e}")
            return False
    
    def delete(self):
        """Delete the ticker"""
        db.session.delete(self)
        db.session.commit()


class HistoricalData(db.Model):
    __tablename__ = 'historical_data'
    
    id = db.Column(db.Integer, primary_key=True)
    ticker_id = db.Column(db.Integer, db.ForeignKey('tickers.id'), nullable=False)
    date = db.Column(db.Date, nullable=False)
    open_price = db.Column(db.Float)
    high_price = db.Column(db.Float)
    low_price = db.Column(db.Float)
    close_price = db.Column(db.Float)
    volume = db.Column(db.BigInteger)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        db.Index('idx_ticker_date', 'ticker_id', 'date'),
    )
    
    def __repr__(self):
        return f'<HistoricalData {self.ticker.symbol} {self.date}>'
    
    def to_dict(self):
        return {
            'date': self.date.strftime('%Y-%m-%d'),
            'open': self.open_price,
            'high': self.high_price,
            'low': self.low_price,
            'close': self.close_price,
            'volume': self.volume
        }
    
    @classmethod
    def get_by_ticker_and_date_range(cls, ticker_id, start_date, end_date):
        return cls.query.filter(
            cls.ticker_id == ticker_id,
            cls.date >= start_date,
            cls.date <= end_date
        ).order_by(cls.date).all()
    
    @classmethod
    def create_from_yfinance(cls, ticker_id, symbol, start_date, end_date):
        """Fetch and store historical data from yfinance"""
        try:
            data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            
            if data.empty:
                return False
            
            for date, row in data.iterrows():
                historical = cls(
                    ticker_id=ticker_id,
                    date=date.date(),
                    open_price=row['Open'],
                    high_price=row['High'],
                    low_price=row['Low'],
                    close_price=row['Close'],
                    volume=row['Volume']
                )
                db.session.add(historical)
            
            db.session.commit()
            return True
        except Exception as e:
            db.session.rollback()
            print(f"Error fetching historical data for {symbol}: {e}")
            return False
        
class Analysis(db.Model):
    __tablename__ = 'analysis'
    
    id = db.Column(db.Integer, primary_key=True)
    ticker_id = db.Column(db.Integer, db.ForeignKey('tickers.id'), nullable=False)
    analysis_text = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.now)
    last_updated = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)
    