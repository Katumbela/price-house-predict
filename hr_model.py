import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib


df_hr = pd.read_csv('HR_comma_sep.csv')


X_hr = df_hr.drop('left', axis=1) 

y_hr = df_hr['left']



X_train_hr, X_test_hr, y_train_hr, y_test_hr = train_test_split(X_hr, y_hr, test_size=0.2, random_state=42)


model_hr = RandomForestClassifier()
model_hr.fit(X_train_hr, y_train_hr)


joblib.dump(model_hr, 'hr_turnover_model.pkl')
print("Modelo de RH treinado e salvo como 'hr_turnover_model.pkl'")
