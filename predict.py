import pandas as pd
import joblib

def load_test_data():
    # Загрузка тестовых данных
    df = pd.read_csv('test.csv')
    df = df.dropna(subset=['domain'])  # Удаление строк с пропущенными доменами
    return df['domain']

def predict():
    # Предсказания на тестовой выборке
    X_test = load_test_data()
    model = joblib.load('dga_classifier.pkl')

    # Получение предсказаний
    y_pred = model.predict(X_test)

    # Сохранение предсказаний
    result = pd.DataFrame({'domain': X_test, 'is_dga': y_pred})
    result.to_csv('prediction.csv', index=False)

    print("Предсказания сохранены в 'prediction.csv'.")

if __name__ == "__main__":
    predict()
