# **Fake News Detection Model** by Jessup Byun

A high-accuracy (97.4%) binary classification model capable of categorizing news articles as either **real** or **fake** using machine learning techniques, specifically leveraging **Logistic Regression**. The dataset consists of over 40000 labeled news articles and has been preprocessed using various **Natural Language Processing (NLP)** techniques to achieve high accuracy. The overall goal is to create a model that can accurately identify fake news based on the text content of the articles.

## **Dataset**

The dataset used in this project is available on [Kaggle](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets?resource=download). It contains two CSV files:
- **True.csv**: Contains news articles labeled as "true" (real news).
- **Fake.csv**: Contains news articles labeled as "fake" (fake news).

Both files have been concatenated into a single DataFrame, where labels were added to indicate **1 for real news** and **0 for fake news**. The dataset was then processed to remove missing values and duplicates before being used for feature extraction and model training.

## **Project Structure**

This project is organized into several Python scripts, each responsible for a different aspect of the machine learning pipeline:

### **1. `data_preprocessing.py`**
This script is responsible for:
- **Loading the dataset**: Combines the real and fake news articles into a single DataFrame.
- **Cleaning the data**: Removes punctuation, stopwords, and performs **lemmatization** to ensure that only meaningful words are kept. This step is crucial to improving the quality of the data for the model.
- **Text Preprocessing**: Prepares the text for feature extraction by converting it into a clean and tokenized format.

Key Libraries: 
- **Pandas** for data manipulation.
- **NLTK (Natural Language Toolkit)**: Used for tokenization, stopword removal, and lemmatization. This helps reduce noise in the dataset and improves model performance.

### **2. `feature_extraction.py`**
This script handles:
- **Feature Extraction**: Uses **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert the text data into numerical features that can be fed into machine learning models. 
- **Why TF-IDF?**: TF-IDF is effective for text classification tasks as it captures the importance of words across the dataset. This feature extraction method reduces the dimensionality of the text data while still preserving important information about the words.

Key Library:
- **`TfidfVectorizer`** from `sklearn.feature_extraction.text`: This converts the cleaned text into TF-IDF vectors with a maximum feature limit of 5000, balancing between performance and feature richness.

### **3. `train_sklearn.py`**
This script focuses on:
- **Model Training**: Implements **Logistic Regression** to classify news articles as real or fake.
- **Model Evaluation**: Evaluates the model using **accuracy**, **precision**, **recall**, and **F1-score** metrics.

Key Libraries:
- **`LogisticRegression`** from `sklearn.linear_model`: This is the primary machine learning algorithm used in the project.
- **`accuracy_score` and `classification_report`** from `sklearn.metrics`: These functions evaluate how well the model performs.

### **4. `main.py`**
This is the orchestrating script that ties everything together:
- **Loads the data** and preprocesses it using the functions from `data_preprocessing.py`.
- **Extracts features** using the `extract_features` function from `feature_extraction.py`.
- **Trains the model** and evaluates its performance using `train_model` from `train_sklearn.py`.
- Implements **cross-validation** for a more robust model evaluation.

## **Libraries Used**
The project heavily relies on the following libraries:
- **Pandas**: For data manipulation and handling the CSV files.
- **Scikit-learn**: For implementing machine learning models, cross-validation, and evaluation metrics.
- **NLTK**: For natural language processing tasks like tokenization, stopword removal, and lemmatization.
- **TfidfVectorizer**: For converting the cleaned text into numerical features.

## **NLP Design Choices**
I chose to use **NLTK** for text preprocessing because it offers comprehensive tools for tokenization, stopword removal, and lemmatization. These steps are critical in removing noise from the dataset, which helps the model focus on the important words in each article.

For feature extraction, I opted for **TF-IDF**, which is a common choice for text classification tasks. It balances the frequency of words and how informative they are across the dataset. I limited the number of features to **5000** to avoid overfitting and reduce dimensionality, which improved model training speed and performance.

## **Accuracy and Model Performance**

### **Train-Test Split Accuracy**
Using a traditional **train-test split** (80% training data and 20% testing data), the model achieved an accuracy of **98.6%**. This suggests that the model performed very well on the test set, but it may still be prone to overfitting, as this single split doesn’t evaluate performance on different subsets of data.

### **Cross-Validation Accuracy**
To mitigate the risk of overfitting, I used **5-fold cross-validation**. Cross-validation divides the dataset into 5 subsets (folds), trains the model on 4 folds, and tests it on the remaining fold. This process is repeated 5 times, and the results are averaged. The cross-validation accuracy was **97.4%**, showing a consistent performance across different folds, indicating that the model generalizes well to unseen data.

**Train-Test Accuracy**: 98.6%
**Cross-Validation Accuracy (5 Folds)**: 97.4%

## **Next Steps (still in development)**
To further improve the model, I plan to:
- Experiment with other machine learning models like **Random Forest** and **SVM** to see if they outperform logistic regression.
- Perform more evaluations using live demos and other testing datasets
- Explore more advanced text classification techniques, such as using **Neural Networks** or **transformer models** like BERT.

## **How to Run the Project**
1. **Clone the repository**:
   ```bash
   git clone https://github.com/your_username/Fake-News-Detection-Model.git
   cd Fake-News-Detection-Model

2. **Install the necessary libraries**
    ```bash
    pip install -r requirements.txt

3. **Run the main script to perform either cross-validation or train-split test
    ```bash
    python src/main.py
