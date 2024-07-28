import pandas as pd
import joblib

# 1. Загрузка модели и векторизатора
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# 2. Загрузка тестовых данных
test_df = pd.read_csv('test.csv')
test_domains = test_df['domain']

# 3. Преобразование и предсказание
X_test = vectorizer.transform(test_domains)
predictions = model.predict(X_test)

# 4. Сохранение результатов
output_df = pd.DataFrame({'domain': test_domains, 'is_dga': predictions})
output_df.to_csv('prediction.csv', index=False)

print("Предсказания завершены. Результаты сохранены в prediction.csv.")
