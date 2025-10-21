from main import DB 

data = DB.execute_query("SELECT * FROM tickers;")

def inseert_ticker(ticker_symbol, ticker_name):
    query = "INSERT INTO tickers (symbol, name) VALUES (%s, %s);"
    DB.execute_query(query, (ticker_symbol, ticker_name))