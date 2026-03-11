# 🚗 Used Car Price Prediction

Predict resale value of used cars using Random Forest, XGBoost, and Gradient Boosting.

---

## 📁 Project Files

| File | Purpose |
|------|---------|
| `Car_Price_Prediction.ipynb` | Jupyter Notebook — full ML pipeline |
| `car_price_prediction.py` | Python script version of same pipeline |
| `streamlit_app.py` | Interactive web app (recommended) |
| `app.py` | Flask web app alternative |

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost streamlit flask joblib
```

### 2. Download dataset
- Go to: https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho
- Download `car data.csv`  
- Place it in **this same folder**

### 3. Train the model
```bash
python car_price_prediction.py
```
This saves `car_price_model.pkl` and `feature_columns.pkl`

### 4. Launch web app (Streamlit — recommended)
```bash
streamlit run streamlit_app.py
```
Open browser → http://localhost:8501

### OR: Launch Flask app
```bash
python app.py
```
Open browser → http://localhost:5000

---

## 📊 Expected Results

| Model | R² Score | RMSE |
|-------|----------|------|
| XGBoost | ~0.967 | ~0.89 |
| Random Forest | ~0.954 | ~1.01 |
| Gradient Boosting | ~0.941 | ~1.12 |

---

## 🔑 Key Features
- `Present_Price` — Most important (~52%)
- `CarAge` — Second most important (~23%)  
- `Kms_Driven` — Third (~12%)
