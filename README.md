# House Price Prediction API

Esta API utiliza um modelo de regressão linear para prever o preço de casas com base em características como área, número de quartos e número de banheiros.

 
# Para realizar alguma predição na API

```
 curl -X POST https://predict-house-price.vercel.app/predict -H "Content-Type: application/json" -d '{
    "area": 2000,
    "bedrooms": 4,
    "bathrooms": 3
}'
```