import pandas as pd
import string
import torch
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Load and preprocess the data
def load_data():
    true_news = pd.read_csv('data/True.csv')
    fake_news = pd.read_csv('data/Fake.csv')

    true_news['label'] = 1
    fake_news['label'] = 0

    data = pd.concat([true_news, fake_news], axis=0).reset_index(drop=True)
    data.dropna(inplace=True)
    data.drop_duplicates(inplace=True)
    return data

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def prepare_data():
    data = load_data()
    data['cleaned_text'] = data['text'].apply(preprocess_text)

    # Extract TF-IDF features
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(data['cleaned_text']).toarray()  # Convert sparse matrix to dense array
    y = data['label'].values  # Labels

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor