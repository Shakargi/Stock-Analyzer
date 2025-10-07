import DataBase
import numpy as np
import pandas as pd
from pymongo import MongoClient
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics._regression import mean_squared_error
import matplotlib.pyplot as plt


# Assumption: the ticker exist in the database, otherwise there'll be an error from the mongoDB user.
def RegressionModelRandomForest(ticker: str, years=5):
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

    # Creating input and output labels
    X = df.drop(columns=['Close'])
    Y = df['Close']
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)

    # Model Training and Hyper-parameter tuning (Randomized Search)
    param_grid = {
        'n_estimators': [200, 500, 1000],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    search = RandomizedSearchCV(RandomForestRegressor(), param_grid)
    search.fit(x_train, y_train)
    model = search.best_estimator_

    # Model Prediction and evaluations
    yPred = model.predict(x_test)
    accuracy = mean_squared_error(y_test, yPred)
    print(f"Model Accuracy: {accuracy}")

    x = np.arange(0, len(yPred))
    plt.title("Evaluation")
    plt.plot(x, yPred, color='r', label='Prediction')
    plt.plot(x, y_test, color='b', label='True Values')
    plt.legend()
    plt.show()

    client.close()


RegressionModelRandomForest("AAPL")
