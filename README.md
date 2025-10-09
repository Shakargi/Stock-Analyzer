# StockTracker – AI Stock Regression Demo

### Description
StockTracker is a demo application for stock analysis using an **LSTM model** in Python, with an interactive **Streamlit interface**.  
The app allows you to:
- Enter a stock ticker (e.g., AAPL, TSLA)  
- Select the number of historical years to analyze  
- Display a graph comparing historical prices and model predictions  
- View a table of the latest stock data  

The app connects to **MongoDB** for historical data and runs a custom **LSTM model** for price prediction.




How to Run:

Make sure the file app.py is in your project folder.

Run the app using Streamlit:

streamlit run app.py


Open the URL shown in your terminal (usually http://localhost:8501) in your browser.


File Structure:

app.py              # Main Streamlit app with integrated LSTM model
LSTMModel.py        # Contains the RegressionModel function
DataBase.py         # Module for inserting and managing stock data in MongoDB


Streamlit Interface:

Ticker: Enter the stock symbol

Number of Years: Choose how many years of historical data to analyze

Window Size: Number of days for LSTM input sequences

Click "Run Forecast" to see the predicted vs. actual stock prices.



Notes:

Ensure you have an active internet connection to access MongoDB.

The model uses only historical data; predictions should not be used as investment advice.

Running the LSTM on large datasets may be slow without a GPU.
