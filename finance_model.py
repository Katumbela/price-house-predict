import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Exemplo de dados de treinamento (substituir pelo dataset real)
data_finance = {
    'open': [100, 110, 105, 115, 120],
    'high': [105, 115, 110, 120, 125],
    'low': [95, 105, 100, 110, 115],
    'close': [102, 112, 107, 117, 122],
    'volume': [1000, 1500, 1200, 1300, 1600]
}
df_finance = pd.DataFrame(data_finance)

# Divisão das features e target
X_finance = df_finance.drop('close', axis=1)
y_finance = df_finance['close']

# Divisão dos dados em conjunto de treinamento e teste
X_train_finance, X_test_finance, y_train_finance, y_test_finance = train_test_split(X_finance, y_finance, test_size=0.2, random_state=42)

# Treinamento do modelo
model_finance = RandomForestRegressor()
model_finance.fit(X_train_finance, y_train_finance)

# Salvamento do modelo treinado
joblib.dump(model_finance, 'finance_stock_model.pkl')
print("Modelo de Finanças treinado e salvo como 'finance_stock_model.pkl'")
