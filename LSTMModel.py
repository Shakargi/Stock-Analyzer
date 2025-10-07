import DataBase
import numpy as np
import pandas as pd
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt


def RegressionModel(ticker: str, years=5, windowSize=60):
    # Updating new Data
    DataBase.insertStock(ticker, years)

    # Connecting to Data
    uri = "mongodb+srv://Shakargi:Avsh0549507881@stocktracker.wru2yc0.mongodb.net/?retryWrites=true&w=majority&appName=StockTracker"
    client = MongoClient(uri)
    db = client["Stocks"]
    collection = db[ticker]

    # Creating Data Frame from data
    data = collection.find()
    df = pd.DataFrame(data)
    df = df.drop(columns=['_id', 'Symbol', 'Name', 'Market', 'Sector', 'Industry', 'Currency', 'Date'])[1:]

    # Standardization:
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(df.drop(columns=['Close']))

    close_scaler = MinMaxScaler()
    close_values = close_scaler.fit_transform(df[['Close']])

    # Create sequences
    X, Y = [], []
    for i in range(windowSize, len(scaled_features)):
        X.append(scaled_features[i - windowSize:i])
        Y.append(close_values[i])

    X = np.array(X)
    Y = np.array(Y)

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)

    # Creating the Model: LSTM layer, 2 Dense Layers, Dropout Layer.
    model = Sequential([
        LSTM(64, return_sequences=True),
        LSTM(64, return_sequences=False),
        Dense(32),
        Dropout(0.5),
        Dense(1)
    ])
    model.summary()

    # Fitting the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)  # EarlyStopping: Regularization
    # tool to stop the training when there's no progress
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=16, callbacks=[es],
              shuffle=False)

    # Un-Scaling and making predictions
    yPred = close_scaler.inverse_transform(model.predict(x_test))
    y_test_orig = close_scaler.inverse_transform(y_test)


    ## Printing the results

    # Prepare full axis for plotting
    full_prices = df['Close'].values
    full_axis = np.arange(len(full_prices))

    # Test indices
    test_start_idx = len(df) - len(y_test_orig)
    test_axis = np.arange(test_start_idx, len(df))

    plt.figure(figsize=(14, 6))
    # Plot full historical prices
    plt.plot(full_axis, full_prices, color='gray', alpha=0.5, label='Historical Prices')

    # Plot true test values
    plt.plot(test_axis, y_test_orig.flatten(), color='b', label='Test True Values')

    # Plot predictions
    plt.plot(test_axis, yPred.flatten(), color='r', label='Test Predictions')

    plt.title(f"{ticker} Stock Price Prediction")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

    client.close()
