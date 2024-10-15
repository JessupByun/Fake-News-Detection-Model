import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

# Install necessary nltk data and resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the datasets
def load_data():
    true_news = pd.read_csv('True.csv')
    fake_news = pd.read_csv('Fake.csv')

    # Add labels: 1 for true news, 0 for fake news
    true_news['label'] = 1
    fake_news['label'] = 0

    # Combine both datasets into one DataFrame
    data = pd.concat([true_news, fake_news], axis=0).reset_index(drop=True)

    # Remove missing values
    data.dropna(inplace=True)

    # Remove duplicates
    data.drop_duplicates(inplace=True)

    return data

# Preprocess text by cleaning, tokenizing, removing stopwords, and lemmatizing
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatize the tokens (convert to root form)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Join the tokens back into a single string
    cleaned_text = ' '.join(tokens)
    return cleaned_text

# Apply text preprocessing to the entire dataset
def clean_data(data):
    data['cleaned_text'] = data['text'].apply(preprocess_text)
    return data

# Split data into training and testing sets
def split_data(data):
    X = data['cleaned_text']
    y = data['label']
    return train_test_split(X, y, test_size=0.2, random_state=10)

# Main function to load, clean, and split the data
def prepare_data():
    # Load and clean data
    data = load_data()
    data = clean_data(data)
    
    # Split the data
    X_train, X_test, y_train, y_test = split_data(data)
    return X_train, X_test, y_train, y_test

