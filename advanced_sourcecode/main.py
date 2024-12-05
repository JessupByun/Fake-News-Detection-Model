from advanced_sourcecode.data_preprocessing import prepare_data
from train_pytorch import FakeNewsClassifier, train_model

# Prepare data
X_train, X_test, y_train, y_test = prepare_data()

# Initialize the model
input_dim = X_train.shape[1]  # Number of features
model = FakeNewsClassifier(input_dim)

# Train and evaluate the model
trained_model = train_model(model, X_train, y_train, X_test, y_test, epochs=10, lr=0.001)