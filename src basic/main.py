from feature_extraction import extract_features
from train_sklearn import train_model
from data_preprocessing import prepare_data

# Load and prepare data (preprocessing)
X_train, X_test, y_train, y_test = prepare_data()

# Extract TF-IDF features from the preprocessed text
X_train_tfidf, X_test_tfidf = extract_features(X_train, X_test)

# Train the machine learning model and evaluate it
model = train_model(X_train_tfidf, X_test_tfidf, y_train, y_test)
