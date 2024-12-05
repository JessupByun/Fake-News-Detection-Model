# **Fake News Detection Model** by Jessup Byun

A high-accuracy binary classification model capable of categorizing news articles as either **real** or **fake** using machine learning techniques. The project has been implemented twice:
1. Using **Scikit-learn** with **Logistic Regression** for simplicity and interpretability.
2. Using **PyTorch** to build a custom **neural network** for greater flexibility and experimentation with deep learning.

Both versions preprocess and classify a dataset of over 40,000 labeled news articles, achieving high accuracy scores.

---

## **Dataset**

The dataset used in this project is available on [Kaggle](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets?resource=download). It contains two CSV files:
- **True.csv**: Contains news articles labeled as "true" (real news).
- **Fake.csv**: Contains news articles labeled as "fake" (fake news).

### **Data Preparation**
- Labels: **1 for real news** and **0 for fake news**.
- Combined into a single DataFrame, cleaned to remove duplicates and missing values.

---

## **Project Structure**

This project is organized into Python scripts for modular implementation. Both **Scikit-learn** and **PyTorch** pipelines use the same dataset and preprocessing steps.

### **1. Scikit-learn Version**
#### Structure:
- **`data_preprocessing.py`**: Cleans the dataset using NLTK (tokenization, stopword removal, lemmatization) and prepares it for vectorization.
- **`feature_extraction.py`**: Converts text into TF-IDF vectors.
- **`train_sklearn.py`**: Trains a **Logistic Regression** model and evaluates its performance using accuracy, precision, recall, and F1-score.
- **`main.py`**: Orchestrates the pipeline, performs **5-fold cross-validation**, and computes average accuracy.

#### Results:
- **Train-Test Accuracy**: 98.6%
- **Cross-Validation Accuracy (5 Folds)**: 97.4%

#### Libraries:
- **Pandas**, **Scikit-learn**, **NLTK**

---

### **2. PyTorch Version**
#### Structure:
- **`data_preprocessing.py`**: Uses simpler preprocessing (punctuation removal, lowercasing) to test performance without heavy NLP processing.
- **`feature_extraction.py`**: Converts text into TF-IDF vectors, consistent with the Scikit-learn pipeline.
- **`train_pytorch.py`**: Defines and trains a neural network using PyTorch with the following architecture:
  - **Input Layer**: Based on TF-IDF features.
  - **Two Hidden Layers**: Each with **ReLU activations**.
  - **Output Layer**: Binary classification using the **Sigmoid activation**.
  - **Optimizer**: Adam (adaptive learning rate).
  - **Loss Function**: CrossEntropyLoss.
- **`main_advanced.py`**: Manages the PyTorch workflow, including training and evaluation.

#### Results:
- **Training Accuracy**: 90.6%
- **Test Accuracy**: 91.0%

#### Libraries:
- **PyTorch**, **Pandas**, **Scikit-learn**

---

## **NLP Design Choices**
1. **Scikit-learn Version**: Included extensive NLP preprocessing using **NLTK** to clean and tokenize text, as well as remove noise (e.g., stopwords, punctuation).
2. **PyTorch Version**: Tested simplified preprocessing to evaluate the model's robustness and reduce preprocessing complexity.

Both pipelines relied on **TF-IDF** for feature extraction:
- Captures word importance across the dataset.
- Limits features to **5000** to reduce dimensionality and computational costs.

---

## **Results and Analysis**
### Scikit-learn:
- **Train-Test Accuracy**: 98.6%
- **Cross-Validation Accuracy**: 97.4%
- **Strengths**: Simplicity, fast training, and interpretable results.
- **Limitations**: May struggle with generalization when scaling to more complex datasets.

### PyTorch:
- **Training Accuracy**: 90.6%
- **Test Accuracy**: 91.0%
- **Strengths**: Customizable architecture, potential for scalability to larger datasets or more complex tasks.
- **Limitations**: More resource-intensive and requires careful tuning.

---

## **Next Steps**
- Test with more complex models (e.g., **LSTMs** or **transformers**) using PyTorch for sequence-based analysis.
- Deploy a live demo using **Flask** or **FastAPI** to classify user-input articles in real time.
- Compare additional feature extraction techniques, such as **word embeddings** (e.g., Word2Vec, GloVe).
- Expand the dataset or test the model on entirely unseen data to evaluate robustness.

---

## **How to Run the Project**

1. **Clone the repository**:
   ```bash
   git clone https://github.com/JessupByun/Fake-News-Detection-Model.git
   cd Fake-News-Detection-Model
