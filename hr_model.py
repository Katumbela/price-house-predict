import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Carregar dados de exemplo (substituir pelo caminho do dataset do Kaggle)
df_hr = pd.read_csv('HR_comma_sep.csv')

# Preprocessamento dos dados
X_hr = df_hr.drop('left', axis=1)  # 'left' é a variável alvo
y_hr = df_hr['left']

# Divisão dos dados em conjunto de treinamento e teste
X_train_hr, X_test_hr, y_train_hr, y_test_hr = train_test_split(X_hr, y_hr, test_size=0.2, random_state=42)

# Treinamento do modelo
model_hr = RandomForestClassifier()
model_hr.fit(X_train_hr, y_train_hr)

# Salvamento do modelo treinado
joblib.dump(model_hr, 'hr_turnover_model.pkl')
print("Modelo de RH treinado e salvo como 'hr_turnover_model.pkl'")
