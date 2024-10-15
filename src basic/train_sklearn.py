from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Train a Logistic Regression model
def train_model(X_train, X_test, y_train, y_test):
    # Initialize the Logistic Regression model
    model = LogisticRegression(max_iter=1000)  # Increase max_iter if necessary
    
    # Train the model on the training data
    model.fit(X_train, y_train)
    
    # Make predictions on the test data
    y_pred = model.predict(X_test)
    
    # Evaluate the model's performance
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    
    # Print detailed classification metrics
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    return model