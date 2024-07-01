import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib


data_manuf = {
    'machine_id': [1, 2, 3, 4, 5],
    'temperature': [300, 310, 305, 315, 320],
    'pressure': [30, 32, 31, 33, 34],
    'output': [500, 520, 510, 530, 540]
}


novos_dados = {
    'machine_id': [6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                   16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                   26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
    'temperature': [325, 330, 335, 340, 345, 350, 355, 360, 365, 370,
                    375, 380, 385, 390, 395, 400, 405, 410, 415, 420,
                    425, 430, 435, 440, 445, 450, 455, 460, 465, 470],
    'pressure': [35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
                 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
                 55, 56, 57, 58, 59, 60, 61, 62, 63, 64],
    'output': [550, 560, 570, 580, 590, 600, 610, 620, 630, 640,
               650, 660, 670, 680, 690, 700, 710, 720, 730, 740,
               750, 760, 770, 780, 790, 800, 810, 820, 830, 840]
}


df_manuf = pd.concat([pd.DataFrame(data_manuf), pd.DataFrame(novos_dados)], ignore_index=True)


X_manuf = df_manuf.drop('output', axis=1)
y_manuf = df_manuf['output']


X_train_manuf, X_test_manuf, y_train_manuf, y_test_manuf = train_test_split(X_manuf, y_manuf, test_size=0.2, random_state=42)


model_manuf = RandomForestRegressor()
model_manuf.fit(X_train_manuf, y_train_manuf)


joblib.dump(model_manuf, './models/manuf_output_model.pkl')
print("Modelo de Manufatura treinado e salvo como 'manuf_output_model.pkl'")
