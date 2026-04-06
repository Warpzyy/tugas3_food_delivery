from model import train_model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# load data
df = pd.read_csv("Food_Delivery_Times.csv")

X = df[['Distance_km']]
y = df['Delivery_Time_min']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = train_model()

y_pred = model.predict(X_test)

# evaluasi
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("R2:", r2)

# contoh prediksi
print("Prediksi 5 km:", model.predict([[5]]))