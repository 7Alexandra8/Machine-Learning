import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Загрузка сохраненной модели и векторизатора
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# 2. Загрузка валидационных данных из CSV
val_df = pd.read_csv('val.csv')
val_domains = val_df['domain']
val_labels = val_df['is_dga']

# 3. Преобразование доменов в числовые представления и предсказание
X_val = vectorizer.transform(val_domains)
predictions = model.predict(X_val)

# 4. Подсчет метрик качества модели
true_positive = ((predictions == 1) & (val_labels == 1)).sum()
false_positive = ((predictions == 1) & (val_labels == 0)).sum()
false_negative = ((predictions == 0) & (val_labels == 1)).sum()
true_negative = ((predictions == 0) & (val_labels == 0)).sum()

accuracy = accuracy_score(val_labels, predictions)
precision = precision_score(val_labels, predictions)
recall = recall_score(val_labels, predictions)
f1 = f1_score(val_labels, predictions)

# 5. Запись метрик в файл
with open('validation.txt', 'w') as f:
    f.write(f"True positive: {true_positive}\n")
    f.write(f"False positive: {false_positive}\n")
    f.write(f"False negative: {false_negative}\n")
    f.write(f"True negative: {true_negative}\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1: {f1:.4f}\n")

print("Валидация завершена. Результаты сохранены в validation.txt.")
