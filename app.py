from flask import Flask, request, jsonify
import joblib
import numpy as np

# Carregar os modelos
model_house = joblib.load('house_price_model.pkl')
model_agro = joblib.load('agro_yield_model.pkl')
model_hr = joblib.load('hr_turnover_model.pkl')
model_manuf = joblib.load('manuf_output_model.pkl')
model_finance = joblib.load('finance_stock_model.pkl')

app = Flask(__name__)

@app.route('/api/predict/house', methods=['POST'])
def predict_house():
    data = request.get_json(force=True)
    features = [data['area'], data['bedrooms'], data['bathrooms']]
    prediction_usd = model_house.predict([features])
    exchange_rate = 800  # 1 USD = 800 AOA
    prediction_aoa = prediction_usd[0] * exchange_rate
    return jsonify({
        'predicted_price_usd': prediction_usd[0],
        'predicted_price_aoa': prediction_aoa
    })

@app.route('/api/predict/agro', methods=['POST'])
def predict_agro():
    data = request.get_json(force=True)
    features = [data['rainfall'], data['temperature'], data['humidity']]
    prediction = model_agro.predict([features])
    return jsonify({'predicted_yield': prediction[0]})

@app.route('/api/predict/hr', methods=['POST'])
def predict_hr():
    data = request.get_json(force=True)
    features = [data['satisfaction_level'], data['last_evaluation'], data['number_project'], 
                data['average_montly_hours'], data['time_spend_company'], data['Work_accident'], 
                data['promotion_last_5years'], data['sales'], data['salary']]
    prediction = model_hr.predict([features])
    return jsonify({'turnover_probability': prediction[0]})

@app.route('/api/predict/manuf', methods=['POST'])
def predict_manuf():
    data = request.get_json(force=True)
    features = [data['machine_id'], data['temperature'], data['pressure']]
    prediction = model_manuf.predict([features])
    return jsonify({'predicted_output': prediction[0]})

@app.route('/api/predict/finance', methods=['POST'])
def predict_finance():
    data = request.get_json(force=True)
    features = [data['open'], data['high'], data['low'], data['volume']]
    prediction = model_finance.predict([features])
    return jsonify({'predicted_close': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
