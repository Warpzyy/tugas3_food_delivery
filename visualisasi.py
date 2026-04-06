import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# baca data
data = pd.read_csv("Food_Delivery_Times.csv")

# =========================
# 1. Scatter Plot
plt.figure()
plt.scatter(data['Distance_km'], data['Delivery_Time_min'])
plt.title("Scatter Plot Jarak vs Waktu")
plt.xlabel("Jarak (km)")
plt.ylabel("Waktu (menit)")
plt.show()

# =========================
# 2. Regresi Linear
X = data[['Distance_km']]
y = data['Delivery_Time_min']

model = LinearRegression()
model.fit(X, y)

plt.figure()
plt.scatter(X, y, label="Data Asli")
X_sorted = X.sort_values(by='Distance_km')
plt.plot(X_sorted, model.predict(X_sorted))
plt.legend()
plt.title("Regresi Linear")
plt.show()

# =========================
# 3. Pairplot
sns.pairplot(data)
plt.show()

# =========================
# 4. Heatmap
plt.figure()
sns.heatmap(data.corr(numeric_only=True), annot=True)
plt.title("Heatmap Korelasi")
plt.show()