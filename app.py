from flask import Flask, request, jsonify
import joblib
import numpy as np

# Carregar o modelo
model = joblib.load('house_price_model.pkl')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = [data['area'], data['bedrooms'], data['bathrooms']]
    prediction = model.predict([features])
    return jsonify({'predicted_price': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
