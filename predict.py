import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

model = joblib.load("stock_model.pkl")
ticker = "AAPL"
data = yf.download(ticker, period="1y")
print("Data terbaru berhasil diambil:", data.tail())

X = data[["Open", "High", "Low", "Volume"]].values

last_known = X[-1].reshape(1, -1)
future_preds = []
for i in range(180):
    pred = model.predict(last_known)[0]
    future_preds.append(pred)
    last_known = np.array([[pred, pred * 1.01, pred * 0.99, last_known[0, 3]]])
future_dates = pd.date_range(datetime.today(), periods=180, freq="B") 
pred_df = pd.DataFrame({"Date": future_dates, "Predicted Price": future_preds})
summary = pd.DataFrame({
    "Horizon": ["7 Hari", "30 Hari", "180 Hari"],
    "Predicted Price (USD)": [
        round(future_preds[6], 2),
        round(future_preds[29], 2),
        round(future_preds[-1], 2),
    ]
})
print("\nRingkasan Prediksi:")
print(summary.to_string(index=False))

print("\nPrediksi 7 Hari ke Depan:")
print(pred_df.head(7).to_string(index=False))
