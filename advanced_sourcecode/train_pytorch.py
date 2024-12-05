import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report

class FakeNewsClassifier(nn.Module):
    def __init__(self, input_dim):
        super(FakeNewsClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 2)  # Binary classification

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.output(x)
        return x

def train_model(model, X_train, y_train, X_test, y_test, epochs=10, lr=0.001):
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        # Training phase
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}")

    # Evaluation phase
    model.eval()
    with torch.no_grad():
        y_pred_train = torch.argmax(model(X_train), axis=1)
        y_pred_test = torch.argmax(model(X_test), axis=1)

    # Metrics
    train_acc = accuracy_score(y_train.numpy(), y_pred_train.numpy())
    test_acc = accuracy_score(y_test.numpy(), y_pred_test.numpy())
    print(f"Training Accuracy: {train_acc}")
    print(f"Test Accuracy: {test_acc}")
    print("Classification Report:")
    print(classification_report(y_test.numpy(), y_pred_test.numpy()))
    return model