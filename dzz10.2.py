from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split

# загрузка данных
df = pd.read_csv('./dz10/parkinsons.data')

X = df.drop(['name', 'status'], axis=1)
y = df['status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# нормализация признаков
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# создаем модель XGBoost
xgb_params = {'max_depth': 6, 'eta': 0.01, 'objective': 'binary:logistic'}
model = xgb.XGBClassifier(**xgb_params)
model.fit(X_train, y_train)

# Проверка точности на тестовой выборке
preds = model.predict(X_test)

accuracy = accuracy_score(y_test, preds)
print("Точность модели: {:.2f}%".format(accuracy * 100))
