# 🚗 AutoVal – Used Car Price Estimator

AutoVal is a machine learning-powered web application that predicts the market value of used vehicles based on their specifications, condition, and historical attributes. The application uses an XGBoost Regression model and provides real-time price estimates through an interactive Streamlit interface.

## 📌 Project Overview

Determining the fair value of a used vehicle can be challenging due to factors such as mileage, age, fuel type, accident history, engine specifications, and market demand.

AutoVal addresses this problem by leveraging machine learning to estimate vehicle prices accurately and efficiently.

Users can enter vehicle information and receive an estimated market value instantly.

---

## ✨ Features

* Interactive and modern Streamlit interface
* Real-time vehicle price prediction
* XGBoost Regression model
* Custom feature engineering
* Frequency encoding for categorical variables
* Confidence range estimation
* Responsive and user-friendly design

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Machine Learning

* XGBoost
* Scikit-learn
* NumPy
* Pandas

### Web Application

* Streamlit

### Model Serialization

* Joblib

---

## 📊 Input Features

The model uses the following vehicle attributes:

* Brand
* Model
* Model Year
* Mileage
* Fuel Type
* Transmission
* Exterior Color
* Interior Color
* Engine Capacity
* Horsepower
* Clean Title Status
* Accident History

---

## 🤖 Machine Learning Model

### Algorithm

XGBoost Regressor

### Preprocessing

* Data cleaning
* Feature engineering
* Frequency encoding for categorical features
* Handling missing values

### Evaluation Metrics

| Metric   | Value    |
| -------- | -------- |
| R² Score | ~0.87    |
| MAE      | ~$7,252  |
| RMSE     | ~$12,817 |

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/autoval-used-car-price-predictor.git
cd autoval-used-car-price-predictor
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```text
AutoVal/
│
├── app.py
├── best_xgboost.pkl
├── requirements.txt
├── README.md
└── notebook.ipynb
```

---

## 🎯 Future Improvements

* VIN-based vehicle lookup
* Advanced feature importance visualization
* Market trend analysis
* Deep learning model comparison
* API deployment using FastAPI
* Cloud deployment and CI/CD integration

---

## 📷 Application Preview

Add screenshots of the application interface here after deployment.

---

## 🌐 Live Demo

Deployment Link:

```text
Coming Soon
```

---

## 👩‍💻 Author

Grishma Khambu

Computer Science & Information Technology Graduate

Interests:

* Artificial Intelligence
* Machine Learning
* Data Science
* Web Development

GitHub: https://github.com/popcorngirl94

---

## 📜 License

This project is developed for educational, research, and portfolio purposes.

