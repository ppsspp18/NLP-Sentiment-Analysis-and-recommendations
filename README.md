# IMDB Sentiment Analysis & Movie Recommendation System

## 📌 Project Overview
This project focuses on **sentiment analysis of IMDB movie reviews** and the development of a **content-based movie recommendation system**.  
Multiple **Deep Learning** and **Classical Machine Learning** approaches were benchmarked to identify the most effective techniques for short-text sentiment classification.

The learned representations were further utilized to cluster movies and recommend similar titles based on review patterns.

---

## 📊 Dataset
- **IMDB Movie Reviews Dataset**
- **25,000+ labeled reviews**
- Binary sentiment labels:
  - `1` → Positive  
  - `0` → Negative

---

## 🧠 Models & Techniques

### 1️⃣ Deep Learning Models
- **CNN (CNet)**
  - Captures local n-gram features using convolutional filters
- **AvgNet**
  - Averages word embeddings for sentence-level representation

**Result:**  
Both models performed comparably, with **AvgNet achieving 85.76% accuracy**, indicating that simpler architectures can be effective for sentiment classification on short-text data.

---

### 2️⃣ Feature Engineering (TF-IDF)
- Implemented **TF-IDF Vectorizer** to extract informative word-level features
- Particularly effective for short reviews where sentiment is driven by key terms

**Result:**  
Achieved **97.89% accuracy**, outperforming deep learning models and significantly improving classification reliability.

---

### 3️⃣ Classical Machine Learning Models
- **Logistic Regression**
- **Random Forest**

Trained on TF-IDF features to establish baseline comparisons.

**Result:**  
Achieved accuracy up to **76.49%**, highlighting the importance of feature representation over model complexity.

---

## 🎬 Movie Recommendation System

A **content-based recommendation system** was built using learned feature representations.

### Techniques Used:
- **Cosine Similarity**  
  - Recommends movies with similar review feature vectors
- **Clustering Algorithms**
  - **K-Means** – Fixed number of clusters
  - **Agglomerative Clustering** – Hierarchical grouping
  - **DBSCAN** – Density-based clustering with outlier detection

These methods help group movies with similar sentiment and content characteristics.

---

## 🔄 Project Workflow
1. Data cleaning and preprocessing
2. Feature extraction (TF-IDF, embeddings)
3. Model training and benchmarking
4. Performance evaluation
5. Clustering and similarity computation
6. Recommendation generation

---

## 📈 Results Summary

| Model / Technique       | Accuracy |
|------------------------|----------|
| CNN (CNet)             | ~85%     |
| AvgNet                 | **85.76%** |
| TF-IDF + Optimized Model | **97.89%** |
| Logistic Regression / Random Forest | ~76.49% |

---

## 🛠️ Technologies Used
- Python
- PyTorch
- Scikit-learn
- NumPy, Pandas
- Matplotlib / Seaborn

---

## 🚀 Future Improvements
- Incorporate transformer-based models (BERT)
- Add hybrid collaborative + content-based recommendation
- Improve evaluation using precision, recall, and F1-score

---

## 📌 Conclusion
This project demonstrates that **effective feature engineering** can outperform complex deep learning models for short-text sentiment analysis, and that learned representations can be successfully extended to build practical recommendation systems.

---

## 👤 Author
**Prakhar Pathak**  
Senior Undergraduate, Department of Civil Engineering  
IIT Kanpur
