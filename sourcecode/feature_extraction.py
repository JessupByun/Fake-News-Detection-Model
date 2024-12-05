from sklearn.feature_extraction.text import TfidfVectorizer

# Function will be used for train-test split, otherwise, vectorizer will be called in main execution file for cross-validation test
def extract_features(X_train, X_test): 
    # Initialize the TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=5000)
    
    # Fit the vectorizer on the training data and transform the training data
    X_train_tfidf = vectorizer.fit_transform(X_train)
    
    # Transform the test data using the same vectorizer
    X_test_tfidf = vectorizer.transform(X_test)

    # Return the transformed training and testing data
    return X_train_tfidf, X_test_tfidf