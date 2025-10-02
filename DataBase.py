import pandas as pd
from pymongo import MongoClient
import yfinance as yf
from datetime import datetime, timedelta


def insertStock(ticker, years=5):
    # Checking if the ticket is valid
    try:
        end_date = datetime.today()
        start_date = end_date - timedelta(days=years * 365 + 200)  # Adding 200 days to fix the SMA's time gaps
        data = yf.download(ticker, start=start_date, end=end_date)
    except:
        print("Stock didn't found")
        return None

    # Saving general information about the stock
    stock = yf.Ticker(ticker)
    info = stock.info
    generalInfo = {
        "Symbol": ticker,
        "Name": info.get("shortName", "N/A"),
        "Market": info.get("market", "N/A"),
        "Sector": info.get("sector", "N/A"),
        "Industry": info.get("industry", "N/A"),
        "Currency": info.get("currency", "N/A")
    }

    # Adding features: Moving Averages, RSI, CCI
    data["SMA20"] = data["Close"].rolling(window=20).mean()
    data["SMA50"] = data["Close"].rolling(window=50).mean()
    data["SMA100"] = data["Close"].rolling(window=100).mean()
    data["SMA200"] = data["Close"].rolling(window=200).mean()

    delta = data["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data["RSI"] = 100 - (100 / (1 + rs))

    tp = (data["High"] + data["Low"] + data["Close"]) / 3
    sma = tp.rolling(window=20).mean()
    mad = tp.rolling(window=20).apply(lambda x: (x - x.mean()).abs().mean())
    data["CCI"] = (tp - sma) / (0.015 * mad)

    # Flatten columns if they are multiindex
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] if col[0] != '' else col[1] for col in data.columns]

    # Fixes Nan or Null Values:
    data = data.dropna()

    # Now convert to records
    data.reset_index(inplace=True)  # Ensure 'Date' is a column
    records = data.to_dict("records")

    # Adding the data to the Stocks DB
    uri = "mongodb+srv://Shakargi:Avsh0549507881@stocktracker.wru2yc0.mongodb.net/?retryWrites=true&w=majority&appName=StockTracker"
    client = MongoClient(uri)

    db = client["Stocks"]
    collection = db[ticker]

    collection.delete_many({})
    collection.insert_one(generalInfo)
    collection.insert_many(records)

    client.close()
