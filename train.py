import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import joblib

def load_data(csv_file):
    """Load data from a CSV file."""
    return pd.read_csv(csv_file)

def preprocess_data(df, max_words=10000, max_len=50):
    """Preprocess text data."""
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(df['domain'])
    X = tokenizer.texts_to_sequences(df['domain'])
    X = pad_sequences(X, maxlen=max_len)
    y = df['is_dga'].values
    return X, y, tokenizer

def train_model():
    """Train and save the LSTM model."""
    # Load training data
    df = load_data('train_data.csv')
    
    # Preprocess data
    X, y, tokenizer = preprocess_data(df)
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define the LSTM model
    model = Sequential([
        Embedding(input_dim=10000, output_dim=128, input_length=50),
        LSTM(64, return_sequences=True),
        LSTM(32),
        Dense(1, activation='sigmoid')
    ])
    
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train the model
    model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_val, y_val))
    
    # Save the model
    model.save('dga_model.h5')
    
    # Save the tokenizer
    joblib.dump(tokenizer, 'tokenizer.pkl')
    
    print("Model and tokenizer saved.")

if __name__ == "__main__":
    train_model()
