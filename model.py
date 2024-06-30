
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Criar um conjunto de dados fictício
data = {
    'area': [1500, 1600, 1700, 1800, 1900],
    'bedrooms': [3, 3, 2, 4, 3],
    'bathrooms': [2, 2, 2, 3, 2],
    'price': [300000, 320000, 340000, 360000, 380000]
}
df = pd.DataFrame(data)

# Dividir os dados em características (X) e alvo (y)
X = df[['area', 'bedrooms', 'bathrooms']]
y = df['price']

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar o modelo de regressão linear
model = LinearRegression()
model.fit(X_train, y_train)

# Salvar o modelo treinado
joblib.dump(model, 'house_price_model.pkl')
