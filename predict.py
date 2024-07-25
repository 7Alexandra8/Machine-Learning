import pandas as pd
import joblib

def load_data(csv_file):

    return pd.read_csv(csv_file)

def predict():


    df = load_data('test.csv')


    model = joblib.load('dga_model.pkl')


    df['is_dga'] = model.predict(df['domain'])


    df.to_csv('prediction.csv', index=False)

    print("Predictions saved to prediction.csv.")

if __name__ == "__main__":
    predict()
