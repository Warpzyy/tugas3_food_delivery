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

def save_plot():
    df = pd.read_csv("Food_Delivery_Times.csv")

    X = df[['Distance_km']]
    y = df['Delivery_Time_min']

    model = LinearRegression()
    model.fit(X, y)

    plt.figure()
    plt.scatter(X, y)
    plt.plot(X, model.predict(X))
    plt.xlabel("Jarak (km)")
    plt.ylabel("Waktu (menit)")
    plt.title("Regresi Linear Delivery Time")

    plt.savefig("static/plot.png")
    plt.close()