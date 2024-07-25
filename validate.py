import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def load_data(csv_file):
    """Load data from a CSV file."""
    return pd.read_csv(csv_file)

def validate_model():
    """Validate the model and output metrics to validation.txt."""
    # Load validation data
    df = load_data('val.csv')

    # Load model
    model = joblib.load('dga_model.pkl')

    # Predict on validation set
    y_pred = model.predict(df['domain'])
    y_true = df['is_dga']

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Calculate metrics
    report = classification_report(y_true, y_pred, output_dict=True)

    # Write metrics to file
    with open('validation.txt', 'w') as f:
        f.write(f"True positive: {tp}\n")
        f.write(f"False positive: {fp}\n")
        f.write(f"False negative: {fn}\n")
        f.write(f"True negative: {tn}\n")
        f.write(f"Accuracy: {report['accuracy']:.4f}\n")
        f.write(f"Precision: {report['1']['precision']:.4f}\n")
        f.write(f"Recall: {report['1']['recall']:.4f}\n")
        f.write(f"F1: {report['1']['f1-score']:.4f}\n")

    print("Validation complete. Metrics saved to validation.txt.")

if __name__ == "__main__":
    validate_model()
