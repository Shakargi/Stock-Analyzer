import streamlit as st
import matplotlib.pyplot as plt
import warnings
from LSTMModel import RegressionModel

## To Run the program, enter the command - streamlit run app.py
warnings.filterwarnings("ignore")

st.set_page_config(page_title="StockTracker Demo", layout="wide")
st.title("📈 StockTracker – AI Stock Regression Demo")

ticker = st.sidebar.text_input("Enter Ticker:", "AAPL")
years = st.sidebar.slider("Years of Stock", 1, 10, 5)
window_size = st.sidebar.slider("Window Size:", 30, 120, 60)

if st.sidebar.button("Run Forecast"):
    with st.spinner("Computing..."):
        try:
            full_axis, full_prices, test_axis, y_test_orig, yPred, df = RegressionModel(ticker, years,
                                                                                        window_size)

            st.success(f"Forecast complete for: {ticker}")

            # גרף
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(full_axis, full_prices, color='gray', alpha=0.5, label='Historical Prices')
            ax.plot(test_axis, y_test_orig.flatten(), color='b', label='Test True Values')
            ax.plot(test_axis, yPred.flatten(), color='r', label='Predictions')
            ax.set_title(f"{ticker} Stock Price Prediction")
            ax.set_xlabel("Time")
            ax.set_ylabel("Price")
            ax.legend()
            st.pyplot(fig)

            st.subheader("Last Economic Data")
            st.dataframe(df.tail(20))

        except Exception as e:
            st.error(f"Forecasting failed {e}")
else:
    st.info("Choose stock and then click Run Forecast for predicting")
