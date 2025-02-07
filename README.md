# **COVID-19 Fake News Classification**

## 📌 **Overview**
This project aims to detect **fake news related to COVID-19** using **machine learning (ML) and deep learning (DL) models**. It is part of the **CONSTRAINT-2021 shared task on hostile post detection** and utilizes data from social media platforms like **Twitter, Facebook, and Instagram**.

## 🎯 **Objective**
Given a social media post, the task is to classify whether it is **real** or **fake** news.

## 📂 **Dataset**
- **Source:** Public dataset from CONSTRAINT-2021.
- **Size:** 10,700 labeled posts.
- **Labels:**
  - `real`: Verified news.
  - `fake`: Misinformation or fake news.
- **Data Preprocessing:**
  - Removed special characters, URLs, and stopwords.
  - Tokenized and lemmatized text.
  - Converted text into numerical representations (TF-IDF, embeddings).

## 🛠 **Technologies Used**
- **Programming Language:** Python 🐍
- **Libraries & Frameworks:**
  - NLP: `NLTK`, `spaCy`, `Gensim`
  - ML & DL: `Scikit-learn`, `TensorFlow`, `PyTorch`
  - Pre-trained Models: `BERT`, `DistilBERT`
  - Data Visualization: `Matplotlib`, `Seaborn`, `Plotly`

## 📊 **Models Implemented**
| Model | Accuracy |
|--------|------------|
| **LSTM (Bidirectional)** | 87.52% |
| **Support Vector Machine (SVM) with TF-IDF** | 85.30% |
| **DistilBERT** | 94.30% |
| **BERT** | 94.77% |

## 🏆 **Best Performing Model: BERT**
- **Training Accuracy:** 99.42%
- **Validation Accuracy:** 94.77%
- **Why?** BERT captures deep contextual relationships in text, making it ideal for detecting misinformation.

## 📌 **Project Structure**
```
📂 covid_fake_news_classification/
├── 📜 README.md         # Project Documentation
├── 📜 requirements.txt   # Python dependencies
├── 📜 tweet_classification.ipynb  # Main Notebook
├── 📂 data/              # Dataset (CSV files)
├── 📂 models/            # Saved trained models
└── 📂 results/           # Visualization outputs
```

## 🚀 **How to Run the Project**
### **1️⃣ Install Dependencies**
```
pip install -r requirements.txt
```

### **2️⃣ Download & Load Data**
```
python scripts/download_data.py
```

### **3️⃣ Train the Model**
```
python scripts/train_model.py --model bert
```

### **4️⃣ Evaluate the Model**
```
python scripts/evaluate_model.py --model bert
```

### **5️⃣ Predict Fake News**
```
python scripts/predict.py --text "Breaking news! COVID-19 is cured with lemon juice!"
```

## 📌 **Results & Visualization**
- **Confusion Matrix:** Helps understand misclassification.
- **Word Clouds:** Shows common words in real vs. fake news.
- **TF-IDF Feature Importance:** Highlights the most important words for classification.

## 🛠 **Future Improvements**
- Fine-tune **BERT** with more COVID-19-related datasets.
- Improve **explainability** using SHAP or LIME.
- Deploy model via **Flask API** or **FastAPI**.

## 🤝 **Contributions**
Contributions are welcome! Feel free to **fork**, **open an issue**, or **submit a pull request**.

## 📜 **License**
This project is licensed under the **Apache License 2.0**.

---
### **⭐ If you find this project useful, please give it a star! ⭐**

