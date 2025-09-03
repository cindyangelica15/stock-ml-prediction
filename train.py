import joblib
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

ticker = "AAPL"
data = yf.download(ticker, start="2020-01-01", end="2023-12-31")

data["Return"] = data["Close"].pct_change()
data["MA5"] = data["Close"].rolling(window=5).mean()
data["MA10"] = data["Close"].rolling(window=10).mean()
data = data.dropna()

data["Target"] = data["Close"].shift(-1)
data = data.dropna()

X = data[["Close", "Return", "MA5", "MA10"]]
y = data["Target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
print(f"Model trained. MAE: {mae:.2f}")

joblib.dump(model, "stock_model.pkl")
print("Model saved as stock_model.pkl")
