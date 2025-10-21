import yfinance


class DataCrawler:

    def __init__(self,tickers,time_bucket,start,end,news=False):
        self.tickers = tickers
        self.time_bucket = time_bucket
        self.news = news 
        self.start = start 
        self.end = end 

    def crawl_ticker_information(self,ticker):
        ticker_obj = yfinance.Ticker(ticker)
        info = ticker_obj.info
        return info

    def crawl_ticker_news(self,ticker):
        if not self.news:
            return None 
        ticker_obj = yfinance.Ticker(ticker)
        news = ticker_obj.news
        return news

    def crawl_ticker_data(self,ticker):

        data = yfinance.download(ticker,
                          start=self.start,
                          end = self.end,
                          interval=self.time_bucket
                          )
        
        return data



CRAWLER = DataCrawler(['TSLA'],'1d',start='2024-01-01',end='2024-02-01')
CRAWLER.crawl_ticker_data('TSLA')