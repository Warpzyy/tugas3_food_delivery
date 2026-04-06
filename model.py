import pandas as pd
from sklearn.linear_model import LinearRegression

def train_model():
    # Load dataset
    df = pd.read_csv("Food_Delivery_Times.csv")

    # Ambil data penting
    X = df[['Distance_km']]
    y = df['Delivery_Time_min']

    # Model
    model = LinearRegression()
    model.fit(X, y)

    return model

import matplotlib.pyplot as plt

def save_plot(jarak_input=None):
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression

    df = pd.read_csv("Food_Delivery_Times.csv")

    X = df[['Distance_km']]
    y = df['Delivery_Time_min']

    model = LinearRegression()
    model.fit(X, y)

    # SORT BIAR GARIS RAPI 🔥
    X_sorted = X.sort_values(by='Distance_km')
    y_pred_sorted = model.predict(X_sorted)

    plt.figure()
    plt.scatter(X, y, label="Data Asli")
    plt.plot(X_sorted, y_pred_sorted, color='green', label="Regresi")

    # 🔴 titik user
    if jarak_input is not None:
        input_df = pd.DataFrame([[jarak_input]], columns=['Distance_km'])
        pred = model.predict(input_df)
        plt.scatter(jarak_input, pred, color='red', s=100, label="Prediksi User")

    plt.xlabel("Jarak (km)")
    plt.ylabel("Waktu (menit)")
    plt.title("Regresi Linear Delivery Time")
    plt.legend()

    plt.savefig("static/plot.png")
    plt.close()