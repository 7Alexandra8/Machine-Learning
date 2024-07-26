import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbalancedPipeline
from sklearn.model_selection import GridSearchCV
import joblib

def load_data():
    # Загрузка обучающих данных
    df = pd.read_json('train_data.json')
    df = df.dropna(subset=['domain'])  # Удаление строк с пропущенными доменами
    df['is_dga'] = (df['threat'] == 'dga').astype(int)
    return df['domain'], df['is_dga']

def train_model():
    # Загрузка данных
    X, y = load_data()

    # Преобразование текста в признаки
    tfidf = TfidfVectorizer(ngram_range=(1, 3), max_features=5000, stop_words='english')

    # Балансировка классов
    smote = SMOTE(random_state=42)

    # Определение модели
    rf_model = RandomForestClassifier(random_state=42)

    # Создание пайплайна
    model_pipeline = ImbalancedPipeline([
        ('tfidf', tfidf),
        ('smote', smote),
        ('classifier', rf_model)
    ])

    # Параметры для Grid Search
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10, 20, 30],
        'classifier__min_samples_split': [2, 5],
        'classifier__min_samples_leaf': [1, 2]
    }

    # Grid Search
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_search.fit(X, y)
    joblib.dump(grid_search.best_estimator_, 'dga_classifier.pkl')
    print("Модель успешно обучена и сохранена.")

if __name__ == "__main__":
    train_model()
