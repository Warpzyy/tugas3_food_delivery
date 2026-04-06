from flask import Flask, render_template, request
import pandas as pd
from model import train_model, save_plot

app = Flask(__name__)

model = train_model()

@app.route("/", methods=["GET", "POST"])
def index():
    hasil = None

    # buat grafik awal
    save_plot()

    if request.method == "POST":
        jarak = float(request.form["jarak"])

        data = pd.DataFrame([[jarak]], columns=['Distance_km'])
        hasil = model.predict(data)[0]

        save_plot(jarak)

    return render_template("index.html", hasil=hasil)

if __name__ == "__main__":
    app.run(debug=True)