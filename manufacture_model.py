import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Exemplo de dados de treinamento (substituir pelo dataset real)
data_manuf = {
    'machine_id': [1, 2, 3, 4, 5],
    'temperature': [300, 310, 305, 315, 320],
    'pressure': [30, 32, 31, 33, 34],
    'output': [500, 520, 510, 530, 540]  # Produção
}
df_manuf = pd.DataFrame(data_manuf)

# Divisão das features e target
X_manuf = df_manuf.drop('output', axis=1)
y_manuf = df_manuf['output']

# Divisão dos dados em conjunto de treinamento e teste
X_train_manuf, X_test_manuf, y_train_manuf, y_test_manuf = train_test_split(X_manuf, y_manuf, test_size=0.2, random_state=42)

# Treinamento do modelo
model_manuf = RandomForestRegressor()
model_manuf.fit(X_train_manuf, y_train_manuf)

# Salvamento do modelo treinado
joblib.dump(model_manuf, 'manuf_output_model.pkl')
print("Modelo de Manufatura treinado e salvo como 'manuf_output_model.pkl'")
