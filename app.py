from flask import Flask, render_template, request
import pandas as pd
from model import train_model
from model import train_model, save_plot

app = Flask(__name__)

# load model
model = train_model()
save_plot()

@app.route("/", methods=["GET", "POST"])
def index():
    hasil = None

    if request.method == "POST":
        jarak = float(request.form["jarak"])

        # biar tidak warning
        data = pd.DataFrame([[jarak]], columns=['Distance_km'])
        hasil = model.predict(data)[0]

    return render_template("index.html", hasil=hasil)

if __name__ == "__main__":
    app.run(debug=True)