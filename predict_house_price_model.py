import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
 
data = {
    'area': [1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900],
    'bedrooms': [3, 3, 2, 4, 3, 4, 5, 3, 2, 4, 3, 4, 3, 2, 5],
    'bathrooms': [2, 2, 2, 3, 2, 3, 4, 2, 1, 3, 2, 3, 2, 1, 4],
    'price': [300000, 320000, 340000, 360000, 380000, 400000, 420000, 440000, 460000, 480000, 500000, 520000, 540000, 560000, 580000]  # Preço em dólares
}
df = pd.DataFrame(data)


X = df[['area', 'bedrooms', 'bathrooms']]
y = df['price']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
model = LinearRegression()
model.fit(X_train, y_train)
 
joblib.dump(model, './models/house_price_model.pkl')
print("Modelo treinado e salvo como 'house_price_model.pkl'")
 
exchange_rate = 860

predictions_usd = model.predict(X_test)
predictions_aoa = predictions_usd * exchange_rate
 