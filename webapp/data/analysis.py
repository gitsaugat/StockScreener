"""
Stock Analysis Module
Contains all analysis functions for stocks including financial ratios and news sentiment
"""

import yfinance as yf
import google.genai as genai
from datetime import datetime, timedelta
import json
import os 

class StockAnalyzer:
    """Main class for stock analysis"""
    
    def __init__(self, symbol,gemini_api_key=None):
        self.symbol = symbol.upper()
        self.ticker = yf.Ticker(self.symbol)
        self.info = self.ticker.info
        # Initialize Gemini API if key provided
        if gemini_api_key:
            self.gemini_model =  genai.Client(api_key=gemini_api_key)
            # self.gemini_model=None
        else:
            self.gemini_model = None

        print(self.gemini_model,"here")
    
    def get_basic_info(self):
        """Get basic stock information"""
        return {
            'symbol': self.symbol,
            'name': self.info.get('longName', 'N/A'),
            'sector': self.info.get('sector', 'N/A'),
            'industry': self.info.get('industry', 'N/A'),
            'current_price': self.info.get('currentPrice', self.info.get('regularMarketPrice', 0)),
            'market_cap': self.info.get('marketCap', 0),
            'volume': self.info.get('volume', 0),
            'avg_volume': self.info.get('averageVolume', 0)
        }
    
    def analyze_price(self):
        """Analyze price trends and movements"""
        try:
            # Get historical data for last 90 days
            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)
            hist = self.ticker.history(start=start_date, end=end_date)
            
            if hist.empty:
                return {'error': 'No price data available'}
            
            current_price = self.info.get('currentPrice', self.info.get('regularMarketPrice', 0))
            
            # Calculate price metrics
            week_52_high = self.info.get('fiftyTwoWeekHigh', 0)
            week_52_low = self.info.get('fiftyTwoWeekLow', 0)
            
            # Price change calculations
            day_change = self.info.get('regularMarketChangePercent', 0)
            
            # Calculate moving averages from historical data
            ma_50 = hist['Close'].tail(50).mean() if len(hist) >= 50 else 0
            ma_200 = hist['Close'].tail(200).mean() if len(hist) >= 200 else 0
            
            # Volatility (standard deviation of returns)
            returns = hist['Close'].pct_change().dropna()
            volatility = returns.std() * (252 ** 0.5) * 100  # Annualized
            
            # Price momentum
            if len(hist) >= 30:
                price_30_days_ago = hist['Close'].iloc[-30]
                momentum_30d = ((current_price - price_30_days_ago) / price_30_days_ago) * 100
            else:
                momentum_30d = 0
            
            # Distance from 52-week high/low
            distance_from_high = ((current_price - week_52_high) / week_52_high) * 100 if week_52_high else 0
            distance_from_low = ((current_price - week_52_low) / week_52_low) * 100 if week_52_low else 0
            
            return {
                'current_price': round(current_price, 2),
                'day_change_percent': round(day_change, 2),
                'week_52_high': round(week_52_high, 2),
                'week_52_low': round(week_52_low, 2),
                'distance_from_high': round(distance_from_high, 2),
                'distance_from_low': round(distance_from_low, 2),
                'ma_50': round(ma_50, 2),
                'ma_200': round(ma_200, 2),
                'volatility': round(volatility, 2),
                'momentum_30d': round(momentum_30d, 2),
                'price_status': self._get_price_status(current_price, ma_50, ma_200)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _get_price_status(self, current_price, ma_50, ma_200):
        """Determine price trend status"""
        if ma_50 == 0 or ma_200 == 0:
            return 'Insufficient data'
        
        if current_price > ma_50 > ma_200:
            return 'Strong Uptrend'
        elif current_price > ma_50:
            return 'Uptrend'
        elif current_price < ma_50 < ma_200:
            return 'Strong Downtrend'
        elif current_price < ma_50:
            return 'Downtrend'
        else:
            return 'Neutral'
    
    def analyze_financial_ratios(self):
        """Calculate and analyze key financial ratios"""
        try:
            # Get financial data
            eps = self.info.get('trailingEps', 0)
            pe_ratio = self.info.get('trailingPE', 0)
            forward_pe = self.info.get('forwardPE', 0)
            peg_ratio = self.info.get('pegRatio', 0)
            
            # Profitability ratios
            roe = self.info.get('returnOnEquity', 0) * 100 if self.info.get('returnOnEquity') else 0
            roa = self.info.get('returnOnAssets', 0) * 100 if self.info.get('returnOnAssets') else 0
            profit_margin = self.info.get('profitMargins', 0) * 100 if self.info.get('profitMargins') else 0
            operating_margin = self.info.get('operatingMargins', 0) * 100 if self.info.get('operatingMargins') else 0
            
            # Valuation ratios
            pb_ratio = self.info.get('priceToBook', 0)
            ps_ratio = self.info.get('priceToSalesTrailing12Months', 0)
            
            # Dividend info
            dividend_yield = self.info.get('dividendYield', 0) * 100 if self.info.get('dividendYield') else 0
            payout_ratio = self.info.get('payoutRatio', 0) * 100 if self.info.get('payoutRatio') else 0
            
            # Debt ratios
            debt_to_equity = self.info.get('debtToEquity', 0)
            current_ratio = self.info.get('currentRatio', 0)
            quick_ratio = self.info.get('quickRatio', 0)
            
            # Growth metrics
            revenue_growth = self.info.get('revenueGrowth', 0) * 100 if self.info.get('revenueGrowth') else 0
            earnings_growth = self.info.get('earningsGrowth', 0) * 100 if self.info.get('earningsGrowth') else 0
            
            return {
                'valuation': {
                    'eps': round(eps, 2),
                    'pe_ratio': round(pe_ratio, 2),
                    'forward_pe': round(forward_pe, 2),
                    'peg_ratio': round(peg_ratio, 2),
                    'pb_ratio': round(pb_ratio, 2),
                    'ps_ratio': round(ps_ratio, 2),
                    'valuation_score': self._get_valuation_score(pe_ratio, pb_ratio, peg_ratio)
                },
                'profitability': {
                    'roe': round(roe, 2),
                    'roa': round(roa, 2),
                    'profit_margin': round(profit_margin, 2),
                    'operating_margin': round(operating_margin, 2),
                    'profitability_score': self._get_profitability_score(roe, roa, profit_margin)
                },
                'dividends': {
                    'dividend_yield': round(dividend_yield, 2),
                    'payout_ratio': round(payout_ratio, 2)
                },
                'financial_health': {
                    'debt_to_equity': round(debt_to_equity, 2),
                    'current_ratio': round(current_ratio, 2),
                    'quick_ratio': round(quick_ratio, 2),
                    'health_score': self._get_health_score(debt_to_equity, current_ratio)
                },
                'growth': {
                    'revenue_growth': round(revenue_growth, 2),
                    'earnings_growth': round(earnings_growth, 2)
                }
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _get_valuation_score(self, pe, pb, peg):
        """Calculate valuation score (1-10, higher is more attractive)"""
        score = 5  # Start neutral
        
        # PE ratio analysis (lower is better, typically)
        if 0 < pe < 15:
            score += 2
        elif 15 <= pe < 25:
            score += 1
        elif pe >= 35:
            score -= 1
        
        # PB ratio analysis (lower is better)
        if 0 < pb < 1:
            score += 1
        elif pb >= 5:
            score -= 1
        
        # PEG ratio analysis (< 1 is attractive)
        if 0 < peg < 1:
            score += 2
        elif 1 <= peg < 2:
            score += 1
        
        return max(1, min(10, score))
    
    def _get_profitability_score(self, roe, roa, profit_margin):
        """Calculate profitability score (1-10)"""
        score = 5
        
        # ROE analysis
        if roe > 20:
            score += 2
        elif roe > 15:
            score += 1
        elif roe < 5:
            score -= 1
        
        # ROA analysis
        if roa > 10:
            score += 1
        elif roa < 2:
            score -= 1
        
        # Profit margin analysis
        if profit_margin > 20:
            score += 2
        elif profit_margin > 10:
            score += 1
        elif profit_margin < 5:
            score -= 1
        
        return max(1, min(10, score))
    
    def _get_health_score(self, debt_to_equity, current_ratio):
        """Calculate financial health score (1-10)"""
        score = 5
        
        # Debt to equity (lower is better)
        if debt_to_equity < 0.5:
            score += 2
        elif debt_to_equity < 1:
            score += 1
        elif debt_to_equity > 2:
            score -= 2
        
        # Current ratio (should be > 1)
        if current_ratio > 2:
            score += 2
        elif current_ratio > 1.5:
            score += 1
        elif current_ratio < 1:
            score -= 2
        
        return max(1, min(10, score))
    
    def get_news(self, limit=10):
        """Fetch recent news for the stock"""
        try:
            news = self.ticker.news
           
            if not news:
                return []
            
            processed_news = []
            for item in news[:limit]:
                # print( item["content"].get('title', 'No title'))
                processed_news.append({
                    'title': item["content"].get('title', 'No title'),
                    'publisher': item["content"].get('publisher', 'Unknown'),
                    'link': item["content"].get('link', '#'),
                    'publish_time': datetime.fromtimestamp(
                        item["content"].get('providerPublishTime', 0)
                    ).strftime('%Y-%m-%d %H:%M:%S') if item["content"].get('providerPublishTime') else 'N/A',
                })
            return processed_news
        except Exception as e:
            print(e,"here am i")
            return []
    
    def analyze_news_sentiment(self, news_limit=10):

        """Analyze news sentiment using Gemini API"""
        if not self.gemini_model:
            
            return {
                'error': 'Gemini API key not configured',
                'sentiment': 'unavailable'
            }
        
        try:
            news = self.get_news(limit=news_limit)
            if not news:
                return {
                    'sentiment': 'neutral',
                    'confidence': 0,
                    'summary': 'No recent news available',
                    'news_count': 0
                }
            
            # Prepare news titles for analysis
            news_text = "\n".join([f"- {item['title']}" for item in news])
            
            # Create prompt for Gemini
            prompt = f"""
            Analyze the sentiment of the following news headlines for {self.symbol} ({self.info.get('longName', self.symbol)}):
            
            {news_text}
            
            Provide a JSON response with the following structure:
            {{
                "sentiment": "positive/negative/neutral",
                "confidence": 0-100,
                "summary": "brief summary of overall sentiment",
                "key_points": ["point 1", "point 2", "point 3"],
                "recommendation": "brief investment perspective based on news"
            }}
            
            Only respond with valid JSON, no additional text.
            """
            import re
            # Get response from Gemini
            response = self.gemini_model.models.generate_content(model="gemini-2.5-flash",
            contents=prompt)
            # Parse JSON response
            try:
                raw_text = response.to_json_dict()["candidates"][0]["content"]["parts"][0]["text"]

                cleaned_text = re.sub(r"^```json\s*|\s*```$", "", raw_text.strip())

                # Step 3: Parse to Python dict
                parsed_dict = json.loads(cleaned_text)
                result = parsed_dict


                result['news_count'] = len(news)
                result['news_items'] = news
                return result
            except json.JSONDecodeError as err:
                # If response is not valid JSON, extract what we can
                return {
                    'sentiment': 'neutral',
                    'confidence': 50,
                    'summary': response.text[:200],
                    'news_count': len(news),
                    'news_items': news,
                    "err":str(err),
                    'raw_analysis': response.text
                }
        
        except Exception as e:
            return {
                'error': str(e),
                'sentiment': 'unavailable',
                'news_count': 0
            }
    
    def get_comprehensive_analysis(self):
        """Get complete analysis of the stock"""
        return {
            'basic_info': self.get_basic_info(),
            'price_analysis': self.analyze_price(),
            'financial_ratios': self.analyze_financial_ratios(),
            'news': self.get_news(),
            'news_sentiment': self.analyze_news_sentiment() if self.gemini_model else None
        }
    def get_stock_data(self,ticker):
        """Fetch comprehensive stock data using yfinance"""
        try:
            stock = yf.Ticker(ticker)
            
            # Get various data points
            info = stock.info
            hist = stock.history(period="1y")
            
            # Calculate key metrics
            current_price = info.get('currentPrice', 0)
            pe_ratio = info.get('trailingPE', 0)
            market_cap = info.get('marketCap', 0)
            revenue = info.get('totalRevenue', 0)
            profit_margin = info.get('profitMargins', 0)
            debt_to_equity = info.get('debtToEquity', 0)
            
            # Historical performance
            year_high = hist['High'].max()
            year_low = hist['Low'].min()
            avg_volume = hist['Volume'].mean()
            
            # Calculate returns
            if len(hist) > 0:
                year_return = ((hist['Close'][-1] - hist['Close'][0]) / hist['Close'][0]) * 100
            else:
                year_return = 0
            
            stock_data = {
                'ticker': ticker,
                'company_name': info.get('longName', ticker),
                'current_price': current_price,
                'pe_ratio': pe_ratio,
                'market_cap': market_cap,
                'revenue': revenue,
                'profit_margin': profit_margin * 100 if profit_margin else 0,
                'debt_to_equity': debt_to_equity,
                'year_high': year_high,
                'year_low': year_low,
                'avg_volume': avg_volume,
                'year_return': year_return,
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'dividend_yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
                'beta': info.get('beta', 0),
                'fifty_two_week_change': info.get('52WeekChange', 0) * 100 if info.get('52WeekChange') else 0
            }
            
            return stock_data
        except Exception as e:
            return {'error': str(e)}

    def analyze_with_gemini(self,stock_data):
        """Use Gemini to analyze the stock data"""
        try:
            prompt = f"""
            Analyze the following stock data and provide a comprehensive investment analysis:
            
            Company: {stock_data['company_name']} ({stock_data['ticker']})
            Sector: {stock_data['sector']}
            Industry: {stock_data['industry']}
            
            Financial Metrics:
            - Current Price: ${stock_data['current_price']:.2f}
            - Market Cap: ${stock_data['market_cap']:,.0f}
            - P/E Ratio: {stock_data['pe_ratio']:.2f}
            - Profit Margin: {stock_data['profit_margin']:.2f}%
            - Debt to Equity: {stock_data['debt_to_equity']:.2f}
            - Beta: {stock_data['beta']:.2f}
            - Dividend Yield: {stock_data['dividend_yield']:.2f}%
            
            Performance:
            - 52-Week High: ${stock_data['year_high']:.2f}
            - 52-Week Low: ${stock_data['year_low']:.2f}
            - Year Return: {stock_data['year_return']:.2f}%
            - 52-Week Change: {stock_data['fifty_two_week_change']:.2f}%
            
            Please provide:
            1. Overall Assessment (Bullish/Bearish/Neutral)
            2. Key Strengths (3-4 points)
            3. Key Concerns (3-4 points)
            4. Valuation Analysis
            5. Risk Assessment
            6. Investment Recommendation (Buy/Hold/Sell) with reasoning
            
            Format your response in clear sections with headers.
            """
            import re
            response = self.gemini_model.models.generate_content(
                model="gemini-2.5-flash",
                contents = prompt)
            raw_text = response.to_json_dict()["candidates"][0]["content"]["parts"][0]["text"]
            cleaned_text = re.sub(r"^```json\s*|\s*```$", "", raw_text.strip())
            return cleaned_text
        except Exception as e:
            return f"Error analyzing with Gemini: {str(e)}"



class PortfolioAnalyzer:
    """Analyze multiple stocks for portfolio comparison"""
    
    def __init__(self, symbols, gemini_api_key=os.environ.get('GEMINI_API_KEY')):
        self.symbols = [s.upper() for s in symbols]
        self.gemini_api_key = gemini_api_key
        self.analyzers = {
            symbol: StockAnalyzer(symbol, gemini_api_key) 
            for symbol in self.symbols
        }
    
    def compare_ratios(self):
        """Compare financial ratios across stocks"""
        comparisons = {}
        
        for symbol, analyzer in self.analyzers.items():
            ratios = analyzer.analyze_financial_ratios()
            comparisons[symbol] = ratios
        
        return comparisons
    
    def compare_prices(self):
        """Compare price metrics across stocks"""
        comparisons = {}
        
        for symbol, analyzer in self.analyzers.items():
            price_data = analyzer.analyze_price()
            comparisons[symbol] = price_data
        
        return comparisons
    
    def get_best_performers(self, metric='momentum_30d'):
        """Get best performing stocks based on a metric"""
        performances = {}
        
        for symbol, analyzer in self.analyzers.items():
            price_data = analyzer.analyze_price()
            if metric in price_data:
                performances[symbol] = price_data[metric]
        
        # Sort by metric
        sorted_performers = sorted(
            performances.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return sorted_performers


# Utility functions
def calculate_roi(initial_price, current_price, dividends=0):
    """Calculate Return on Investment"""
    if initial_price == 0:
        return 0
    roi = ((current_price - initial_price + dividends) / initial_price) * 100
    return round(roi, 2)


def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """Calculate Sharpe Ratio for risk-adjusted returns"""
    import numpy as np
    
    if len(returns) == 0:
        return 0
    
    excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
    sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
    return round(sharpe, 2)


def get_recommendation_score(analyzer_result):
    """Generate overall recommendation score from analysis"""
    try:
        ratios = analyzer_result.get('financial_ratios', {})
        
        valuation_score = ratios.get('valuation', {}).get('valuation_score', 5)
        profitability_score = ratios.get('profitability', {}).get('profitability_score', 5)
        health_score = ratios.get('financial_health', {}).get('health_score', 5)
        news_score = 0   # Scale 0-10 
        print(analyzer_result.get('news_sentiment'))
        if analyzer_result.get('news_sentiment',{}).get('sentiment','neutral') == 'positive':
            news_score = analyzer_result.get('news_sentiment', {}).get('confidence', 50)/10
        else:
            news_score = -(analyzer_result.get('news_sentiment', {}).get('confidence', 50)/10)

        overall_score = (valuation_score + profitability_score + health_score) / 3        


        if overall_score >= 7:
            recommendation = 'Strong Buy'
        elif overall_score >= 6:
            recommendation = 'Buy'
        elif overall_score >= 5:
            recommendation = 'Hold'
        elif overall_score >= 4:
            recommendation = 'Sell'
        else:
            recommendation = 'Strong Sell'

        if news_score >= 7:
            news_recommendation = 'Strong Buy'
        elif news_score >= 6:
            news_recommendation = 'Buy'
        elif news_score >= 5:
            news_recommendation = 'Hold'
        elif news_score >= 4:
            news_recommendation = 'Sell'
        else:
            news_recommendation = 'Strong Sell'
        
        return {
            'score': round(overall_score, 1),
            'score_with_news_sentiment': round(news_score, 1),
            'recommendation': recommendation,
            'news_recommendation': news_recommendation
        }
    except Exception as e:
        print(e,"here")
        return {
            'score': 5.0,
            'recommendation': 'Hold'
        }
    

    