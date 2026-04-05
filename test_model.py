from model import train_model

model = train_model()

# contoh prediksi
pred = model.predict([[5]])
print("Prediksi untuk 5 km:", pred)