# Movie Review Sentiment Engine: Binary Classification via Sentence Polarity

## 📌 Project Objective
The primary objective of this project is to architect a robust **sentiment classifier** engineered to predict the emotional polarity (positive vs. negative) of movie reviews. By leveraging the [**Sentence Polarity Dataset**](https://www.kaggle.com/datasets/nltkdata/sentence-polarity), this repository implements a comprehensive machine learning lifecycle—spanning from raw text ingestion to systematic hyperparameter optimization.

*Note: MAster School Lectures notes.*
---

## 📊 Dataset Specifications: Sentence Polarity Collection
The engine utilizes the established Sentence Polarity dataset, which consists of movie reviews harvested from **Rotten Tomatoes**.

* **Total Samples**: 10,662 individual sentences.
* **Balance**: A perfectly balanced distribution containing 5,331 positive and 5,331 negative entries.
* **Data Structure**:
    * `rt-polarity.pos`: Encapsulates sentences exhibiting positive sentiment.
    * `rt-polarity.neg`: Encapsulates sentences exhibiting negative sentiment.

---

## 🏗 The Engineering Pipeline
The project adheres to a modular **"Z-pattern" workflow** designed to maintain data integrity and maximize model generalizability:

1. **Preprocessing**: Cleaning and normalizing raw textual data to ensure the corpus is optimized for downstream feature extraction.
2. **Feature Extraction**: Converting unstructured text into high-dimensional numerical representations using **TF-IDF (Term Frequency-Inverse Document Frequency)**.
3. **Model Training**: Training a supervised classification model on the synthesized and labeled textual data.
4. **Validation Strategy**: Implementing **K-Fold Cross-validation** to derive a statistically reliable estimate of the model’s true predictive performance.
5. **Optimization**: Systematic **Hyperparameter Tuning** to refine the classifier's decision boundaries and maximize global accuracy.



---

## 🛠 Technical Stack
* **Language**: Python
* **Vectorization**: TF-IDF via `Scikit-Learn`
* **Modeling**: Logistic Regression / Support Vector Machines (SVM)
* **Evaluation**: K-Fold Cross-Validation, Accuracy, Precision, and Recall metrics

---

## 🤝 Contributing
This is an open-source initiative. We encourage the community to:
* **Fork** the repository and experiment with alternative vectorization strategies (e.g., **Word2Vec**, **BERT**, or **RoBERTa**).
* Submit **pull requests** for performance enhancements or novel feature additions.

---
