from flask import Flask, request, jsonify
import joblib
import numpy as np
 
model = joblib.load('house_price_model.pkl')

app = Flask(__name__)

# Contando com a taxa de cambio de 860Kz
exchange_rate = 860  

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = [data['area'], data['bedrooms'], data['bathrooms']]
    prediction_usd = model.predict([features])
    prediction_aoa = prediction_usd[0] * exchange_rate
    return jsonify({
        'predicted_price_usd': prediction_usd[0],
        'predicted_price_aoa': prediction_aoa
    })

if __name__ == '__main__':
    app.run(debug=True)
