# 5장: 회귀 (Regression)

## 개요

연속적인 수치를 예측하는 회귀 모델을 학습합니다.  
경사하강법의 원리부터 시작해 선형 회귀, 다항 회귀, 정규화 회귀 모델까지 다룹니다.

---

## 1. 경사하강법 (Gradient Descent)

### 개념

비용 함수(Cost Function)를 최소화하는 가중치를 반복적으로 업데이트하는 방법.

```
비용 함수: 예측값과 실제값의 차이 (오차의 제곱합)
기울기: 비용 함수의 미분값 (가중치를 어느 방향으로 바꿔야 오류가 줄지)
```

```python
import numpy as np

def gradient_descent(X, y, learning_rate=0.01, iters=1000):
    N = len(y)
    w1 = np.zeros(1)  # 기울기
    w0 = np.zeros(1)  # 절편

    for i in range(iters):
        y_pred = np.dot(X, w1.T) + w0
        error = y - y_pred

        # 가중치 업데이트 (기울기의 반대 방향으로)
        w1 = w1 + 2/N * learning_rate * np.dot(X.T, error)
        w0 = w0 + 2/N * learning_rate * np.sum(error)

    return w1, w0
```

---

## 2. 선형 회귀 (Linear Regression)

### 단순 선형 회귀

$$y = w_1 x + w_0$$

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

lr = LinearRegression()
lr.fit(X_train, y_train)
pred = lr.predict(X_test)

# 회귀 계수
print(f'기울기(w1): {lr.coef_}')
print(f'절편(w0): {lr.intercept_}')

# 평가 지표
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)
print(f'RMSE: {rmse:.4f}')
print(f'R²: {r2:.4f}')  # 1에 가까울수록 좋음 (설명 분산 비율)
```

---

## 3. 다항 회귀 (Polynomial Regression)

비선형 데이터에 선형 회귀를 적용하기 위해 특성을 다항식으로 변환합니다.

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# 파이프라인으로 전처리 + 모델 연결
poly_pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),  # x → x, x², x*y, ...
    ('scaler', StandardScaler()),
    ('linear', LinearRegression())
])

poly_pipeline.fit(X_train, y_train)
pred = poly_pipeline.predict(X_test)
```

> **과적합 주의**: degree가 클수록 훈련 데이터에 과적합될 수 있습니다.

---

## 4. 정규화 회귀 (Regularized Regression)

과적합을 방지하기 위해 비용 함수에 패널티를 추가합니다.

### Ridge (L2 정규화)

$$\text{Cost} = MSE + \alpha \sum w_i^2$$

```python
from sklearn.linear_model import Ridge

ridge = Ridge(alpha=1.0)  # alpha: 정규화 강도 (클수록 단순한 모델)
ridge.fit(X_train, y_train)
```

### Lasso (L1 정규화)

$$\text{Cost} = MSE + \alpha \sum |w_i|$$

```python
from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.01)
lasso.fit(X_train, y_train)
# 일부 계수가 정확히 0이 됨 → 자동 피처 선택 효과
```

### ElasticNet (Ridge + Lasso 결합)

```python
from sklearn.linear_model import ElasticNet

elastic = ElasticNet(alpha=0.01, l1_ratio=0.7)
# l1_ratio: 0이면 Ridge, 1이면 Lasso
elastic.fit(X_train, y_train)
```

### 정규화 비교

| 방법 | 패널티 | 특징 | 사용 상황 |
|------|--------|------|----------|
| Ridge | L2 (제곱합) | 모든 계수 축소 | 다중공선성 문제 |
| Lasso | L1 (절댓값합) | 일부 계수 0 | 피처 선택 필요 |
| ElasticNet | L1 + L2 | 두 효과 혼합 | 특성 많고 공선성 있음 |

---

## 5. 회귀 평가 지표

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# MSE: 평균 제곱 오차
mse = mean_squared_error(y_test, pred)

# RMSE: 루트 MSE (원래 단위와 동일)
rmse = np.sqrt(mse)

# MAE: 평균 절대 오차
mae = mean_absolute_error(y_test, pred)

# RMSLE: 로그 변환 후 RMSE (큰 값과 작은 값 오차 균등 반영)
def rmsle(y_real, y_pred):
    log_y = np.log1p(y_real)      # log(1+y)
    log_pred = np.log1p(y_pred)
    return np.sqrt(np.mean((log_y - log_pred) ** 2))

# R²: 결정 계수 (0~1, 높을수록 좋음)
r2 = r2_score(y_test, pred)
```

---

## 6. 실전: 집값 예측 (House Prices)

### 데이터 전처리 패턴

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# 로그 변환 (왜도 제거)
train_df['SalePrice'] = np.log1p(train_df['SalePrice'])

# 결측치 처리
null_cols = train_df.columns[train_df.isnull().any()]
for col in null_cols:
    if train_df[col].dtype == 'object':
        train_df[col].fillna('None', inplace=True)
    else:
        train_df[col].fillna(0, inplace=True)

# 문자열 컬럼 레이블 인코딩
obj_cols = train_df.select_dtypes(include='object').columns
for col in obj_cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])

# 왜도(skewness)가 높은 컬럼 로그 변환
skewed_features = train_df.skew()[train_df.skew() > 1].index
train_df[skewed_features] = np.log1p(train_df[skewed_features])
```

### 여러 모델 비교

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

models = {
    'Ridge': Ridge(alpha=0.1),
    'Lasso': Lasso(alpha=0.001),
    'ElasticNet': ElasticNet(alpha=0.01, l1_ratio=0.7),
    'RandomForest': RandomForestRegressor(n_estimators=200),
    'XGBoost': XGBRegressor(n_estimators=200, learning_rate=0.05)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    print(f'{name}: RMSE={rmse:.4f}')
```

---

## 7. 실전: 자전거 대여 수요 예측 (Bike Sharing)

시계열 데이터의 특성 엔지니어링 예시입니다.

```python
# 날짜에서 특성 추출
bike_df['datetime'] = pd.to_datetime(bike_df['datetime'])
bike_df['year'] = bike_df['datetime'].dt.year
bike_df['month'] = bike_df['datetime'].dt.month
bike_df['hour'] = bike_df['datetime'].dt.hour
bike_df['dayofweek'] = bike_df['datetime'].dt.dayofweek

# 로그 변환된 타겟
y_log = np.log1p(bike_df['count'])

# 평가는 RMSLE 사용
```

---

## 학습 포인트

| 개념 | 핵심 내용 |
|------|----------|
| 경사하강법 | 오차를 줄이는 방향으로 가중치를 반복 업데이트 |
| 다항 회귀 | 특성 변환 후 선형 회귀 적용 (주의: 과적합) |
| Ridge/Lasso | 과적합 방지. Lasso는 피처 선택 효과도 있음 |
| RMSE | 값이 크거나 단위가 있을 때 주로 사용 |
| RMSLE | 값의 비율 오차가 중요할 때 사용 |
| 로그 변환 | 왜도 높은 데이터 → 로그 변환 후 예측 → 역변환 |

---

[← 4장: 분류 & 앙상블](04_classification.md) | [6장: 차원 축소 →](06_dim_reduction.md)
