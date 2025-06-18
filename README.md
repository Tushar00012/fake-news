# 📰 Fake News Detection using Machine Learning

## 📌 Overview
This project focuses on building a fake news detection system using natural language processing (NLP) and machine learning algorithms. It classifies news articles as either **Fake** or **Real** by analyzing their textual content.

---

## 📁 Project Structure

- `app.py` – Flask web application for real-time fake news detection  
- `train_models.py` – Script for training models with TF-IDF and saving them  
- `vectorizer.pkl` – Serialized TF-IDF vectorizer  
- `LR_model.pkl` – Trained Logistic Regression model  
- `DT_model.pkl` – Trained Decision Tree model  
- `RF_model.pkl` – Trained Random Forest model  
- `GB_model.pkl` – Trained Gradient Boosting model  
- `Fake.csv` – Dataset containing fake news articles  
- `True.csv` – Dataset containing real news articles  

---

## 🧠 Machine Learning Models

- ✅ Logistic Regression
- ✅ Decision Tree
- ✅ Random Forest
- ✅ Gradient Boosting

These models were trained on a combined dataset of real and fake news using TF-IDF vectorization for feature extraction.

---

## 🛠️ Technologies Used

- **Python**
- **Pandas, NumPy** – Data manipulation
- **Scikit-learn** – Machine learning models & vectorization
- **Flask** – Web framework
- **Pickle/Joblib** – Model serialization
- **TF-IDF** – Feature extraction from text

---

## 🚀 How to Run the Project

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/fake-news-detector.git
   cd fake-news
