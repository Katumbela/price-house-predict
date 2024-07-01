import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib


pesticides = pd.read_csv('./mnt/pesticides.csv')
rainfall = pd.read_csv('./mnt/rainfall.csv')
temp = pd.read_csv('./mnt/temp.csv')
yield_data = pd.read_csv('./mnt/yield.csv')
yield_df = pd.read_csv('./mnt/yield_df.csv')


print("Pesticides:")
print(pesticides.head())
print("\nRainfall:")
print(rainfall.head())
print("\nTemperature:")
print(temp.head())
print("\nYield:")
print(yield_data.head())
print("\nYield DF:")
print(yield_df.head())


print("\nColunas de Pesticides:")
print(pesticides.columns)
print("\nColunas de Rainfall:")
print(rainfall.columns)
print("\nColunas de Temperature:")
print(temp.columns)
print("\nColunas de Yield:")
print(yield_data.columns)
print("\nColunas de Yield DF:")
print(yield_df.columns)



combined_data = yield_data.merge(pesticides, on=['Area', 'Year'])
combined_data = combined_data.merge(rainfall, left_on=['Area', 'Year'], right_on=[' Area', 'Year'])
combined_data = combined_data.merge(temp, left_on=['Year'], right_on=['year'])
combined_data = combined_data.merge(yield_df, on=['Area', 'Year'])



X_agro = combined_data.drop(['hg/ha_yield'], axis=1)  

y_agro = combined_data['hg/ha_yield']


X_train_agro, X_test_agro, y_train_agro, y_test_agro = train_test_split(X_agro, y_agro, test_size=0.2, random_state=42)


model_agro = RandomForestRegressor()
model_agro.fit(X_train_agro, y_train_agro)


score = model_agro.score(X_test_agro, y_test_agro)
print(f"Acurácia do modelo: {score}")


joblib.dump(model_agro, './models/agro_yield_model.pkl')
print("Modelo de Agronegócio treinado e salvo como 'agro_yield_model.pkl'")
