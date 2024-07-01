import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib


data_agro = {
    'rainfall': [800, 900, 850, 950, 1000],
    'temperature': [20, 22, 21, 23, 24],
    'humidity': [70, 65, 75, 80, 68],
    'yield': [3000, 3200, 3100, 3300, 3400]  
    
}
df_agro = pd.DataFrame(data_agro)


X_agro = df_agro.drop('yield', axis=1)
y_agro = df_agro['yield']


X_train_agro, X_test_agro, y_train_agro, y_test_agro = train_test_split(X_agro, y_agro, test_size=0.2, random_state=42)


model_agro = RandomForestRegressor()
model_agro.fit(X_train_agro, y_train_agro)



joblib.dump(model_agro, 'agro_yield_model.pkl')
print("Modelo de Agronegócio treinado e salvo como 'agro_yield_model.pkl'")
