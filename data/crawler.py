import yfinance as yf


class DataCrawler:
    """Service class for fetching stock data"""
    
    def __init__(self, tickers, time_bucket, start, end, news=False):
        self.tickers = tickers
        self.time_bucket = time_bucket
        self.news = news
        self.start = start
        self.end = end

    def crawl_ticker_information(self, ticker):
        try:
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info
            return info
        except Exception as e:
            return None

    def crawl_ticker_news(self, ticker):
        if not self.news:
            return None
        try:
            ticker_obj = yf.Ticker(ticker)
            news = ticker_obj.news
            return news
        except Exception as e:
            return None

    def crawl_ticker_data(self, ticker):
        try:
            data = yf.download(ticker,
                              start=self.start,
                              end=self.end,
                              interval=self.time_bucket,
                              progress=False)
            return data
        except Exception as e:
            return None