import json
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import RandomizedSearchCV
import joblib
from scipy.stats import randint, uniform

# 1. Загрузка данных
with open('train_data.json', 'r') as f:
    data = json.load(f)

# 2. Преобразование данных в DataFrame
df = pd.DataFrame(data)
domains = df['domain']
labels = df['threat'].apply(lambda x: 1 if x == 'dga' else 0) #Метки: 1 - DGA, 0 - обычный домен

# 3. Преобразование данных
vectorizer = TfidfVectorizer(max_features=3000, lowercase=False, analyzer='char_wb', ngram_range=(2, 4))
X = vectorizer.fit_transform(domains)

# 4. Определение модели
model = GradientBoostingClassifier()

# 5. Гиперпараметры для RandomizedSearchCV
param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(3, 10),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 20),
    'learning_rate': uniform(0.01, 0.3),
    'subsample': uniform(0.7, 1.0)
}

# 6. Поиск гиперпараметров с RandomizedSearchCV
random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=50, scoring='accuracy', cv=3, verbose=1, n_jobs=-1)
random_search.fit(X, labels)

# 7. Сохранение лучшей модели и векторизатора
joblib.dump(random_search.best_estimator_, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("Обучение модели завершено. Модель и векторизатор сохранены.")