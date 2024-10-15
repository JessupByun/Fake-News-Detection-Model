from data_preprocessing import prepare_data
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

# Load and clean the data
X, y = prepare_data()  # This will load and preprocess your data

# Perform feature extraction (TF-IDF) and vectorize the entire dataset
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(X)  # Transform entire dataset for cross-validation

# Initialize Logistic Regression model
model = LogisticRegression(max_iter=500)

# Perform cross-validation (5-fold)
cv_scores = cross_val_score(model, X_tfidf, y, cv=5, scoring='accuracy')

# Print cross-validation results
print(f"Cross-validation scores for each fold: {cv_scores}")
print(f"Average cross-validation score: {cv_scores.mean()}")


# Main script for train test split
"""
from feature_extraction import extract_features
from train_sklearn import train_model
from data_preprocessing import prepare_data

# Load and prepare data (preprocessing)
X_train, X_test, y_train, y_test = prepare_data() #make sure to uncomment train test code in data_preprocessing.py

# Extract TF-IDF features from the preprocessed text
X_train_tfidf, X_test_tfidf = extract_features(x, y)

# Train the machine learning model and evaluate it
model = train_model(X_train_tfidf, X_test_tfidf, x, y)
"""