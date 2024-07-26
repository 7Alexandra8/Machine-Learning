import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import joblib

def load_validation_data():
    """Загрузка и подготовка валидационных данных."""
    df = pd.read_csv('val.csv')
    df = df.dropna(subset=['domain'])  # Удалить строки с пропущенными доменами
    df['is_dga'] = df['is_dga'].astype(int)
    return df['domain'], df['is_dga']

def validate_model():
    """Оценка модели и сохранение результатов в 'validation.txt'."""
    X_val, y_val = load_validation_data()
    model = joblib.load('dga_classifier.pkl')
    y_pred = model.predict(X_val)

    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)

    with open('validation.txt', 'w') as f:
        f.write(f"True positive: {tp}\n")
        f.write(f"False positive: {fp}\n")
        f.write(f"False negative: {fn}\n")
        f.write(f"True negative: {tn}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1: {f1:.4f}\n")

    print("Результаты валидации сохранены в 'validation.txt'.")

if __name__ == "__main__":
    validate_model()
