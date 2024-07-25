import pandas as pd
import joblib

def load_data(csv_file):
    """Load data from a CSV file."""
    return pd.read_csv(csv_file)

def predict():
    """Predict on test data and save predictions to prediction.csv."""
    # Load test data
    df = load_data('test.csv')

    # Load model
    model = joblib.load('dga_model.pkl')

    # Predict
    df['is_dga'] = model.predict(df['domain'])

    # Save predictions
    df.to_csv('prediction.csv', index=False)

    print("Predictions saved to prediction.csv.")

if __name__ == "__main__":
    predict()
