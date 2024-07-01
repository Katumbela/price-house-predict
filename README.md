 Rotas da API para previsão de preços de casas, produtividade agrícola, rotatividade de funcionários, produção na manufatura e fechamento de ações financeiras.

---

# Multi-Market Prediction API

Esta API utiliza vários modelos de machine learning para prever diferentes aspectos em diversos mercados, incluindo imóveis, agronegócio, recursos humanos, manufatura e finanças.

## Endpoints

### 1. Previsão de Preço de Casas

Utiliza um modelo de regressão linear para prever o preço de casas com base em características como área, número de quartos e número de banheiros.

**Endpoint:** `/api/predict/house`

**Método:** `POST`

**Exemplo de Entrada:**

```json
{
    "area": 2000,
    "bedrooms": 4,
    "bathrooms": 3
}
```

**Exemplo de Uso com `curl`:**

```sh
curl -X POST https://predict-api.reputacao360.online/api/predict/house -H "Content-Type: application/json" -d '{
    "area": 2000,
    "bedrooms": 4,
    "bathrooms": 3
}'
```

**Resposta:**

```json
{
    "predicted_price_usd": 400000.0,
    "predicted_price_aoa": 320000000.0
}
```

### 2. Previsão de Produtividade Agrícola

Utiliza um modelo de regressão para prever a produtividade agrícola com base em dados como precipitação, temperatura e umidade.

**Endpoint:** `/api/predict/agro`

**Método:** `POST`

**Exemplo de Entrada:**

```json
{
    "rainfall": 800,
    "temperature": 22,
    "humidity": 70
}
```

**Exemplo de Uso com `curl`:**

```sh
curl -X POST https://predict-api.reputacao360.online/api/predict/agro -H "Content-Type: application/json" -d '{
    "rainfall": 800,
    "temperature": 22,
    "humidity": 70
}'
```

**Resposta:**

```json
{
    "predicted_yield": 3200.0
}
```

### 3. Previsão de Rotatividade de Funcionários

Utiliza um modelo de classificação para prever a probabilidade de rotatividade de funcionários com base em vários fatores.

**Endpoint:** `/api/predict/hr`

**Método:** `POST`

**Exemplo de Entrada:**

```json
{
    "satisfaction_level": 0.5,
    "last_evaluation": 0.7,
    "number_project": 3,
    "average_montly_hours": 150,
    "time_spend_company": 3,
    "Work_accident": 0,
    "promotion_last_5years": 0,
    "sales": 2,
    "salary": 2
}
```

**Exemplo de Uso com `curl`:**

```sh
curl -X POST https://predict-api.reputacao360.online/api/predict/hr -H "Content-Type: application/json" -d '{
    "satisfaction_level": 0.5,
    "last_evaluation": 0.7,
    "number_project": 3,
    "average_montly_hours": 150,
    "time_spend_company": 3,
    "Work_accident": 0,
    "promotion_last_5years": 0,
    "sales": 2,
    "salary": 2
}'
```

**Resposta:**

```json
{
    "turnover_probability": 0
}
```

### 4. Previsão de Produção na Manufatura

Utiliza um modelo de regressão para prever a produção de manufatura com base em características como ID da máquina, temperatura e pressão.

**Endpoint:** `/api/predict/manuf`

**Método:** `POST`

**Exemplo de Entrada:**

```json
{
    "machine_id": 1,
    "temperature": 300,
    "pressure": 30
}
```

**Exemplo de Uso com `curl`:**

```sh
curl -X POST https://predict-api.reputacao360.online/api/predict/manuf -H "Content-Type: application/json" -d '{
    "machine_id": 1,
    "temperature": 300,
    "pressure": 30
}'
```

**Resposta:**

```json
{
    "predicted_output": 500.0
}
```

### 5. Previsão de Fechamento de Ações Financeiras

Utiliza um modelo de regressão para prever o preço de fechamento de ações com base em dados como preço de abertura, preço máximo, preço mínimo e volume.

**Endpoint:** `/api/predict/finance`

**Método:** `POST`

**Exemplo de Entrada:**

```json
{
    "open": 100,
    "high": 105,
    "low": 95,
    "volume": 1000
}
```

**Exemplo de Uso com `curl`:**

```sh
curl -X POST https://predict-api.reputacao360.online/api/predict/finance -H "Content-Type: application/json" -d '{
    "open": 100,
    "high": 105,
    "low": 95,
    "volume": 1000
}'
```

**Resposta:**

```json
{
    "predicted_close": 102.0
}
```

---

### Links para Datasets

- **Agronegócio**: [Crop yield prediction](https://www.kaggle.com/datasets/gopalchandra/crop-yield-prediction-dataset)
- **Recursos Humanos**: [HR Analytics](https://www.kaggle.com/datasets/ludobenistant/hr-analytics)
- **Manufatura**: [Manufacturing Process](https://www.kaggle.com/datasets/philmohun/manufacturing-processes)
- **Finanças**: [Stock Market Data](https://www.kaggle.com/datasets/szrlee/stock-time-series-20050101-to-20171231)
 
### Testando a API

Você pode testar a API usando `curl`, Postman ou qualquer outra ferramenta de sua preferência. Siga os exemplos fornecidos para cada endpoint.

---

Este README cobre todos os endpoints da API, como utilizá-los e onde encontrar os datasets para treinamento dos modelos.