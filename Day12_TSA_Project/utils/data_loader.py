import yfinance as yf
import pandas as pd

def download_stock_data(symbol , start , end):
    # download historical data from yahoo finance
    df = yf.download(
        symbol,
        start=start,
        end=end
    )
    df.reset_index(inplace=True)
    df.columns = df.columns.get_level_values(0)
    return df