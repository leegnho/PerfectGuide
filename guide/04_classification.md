# 4장: 분류 & 앙상블 학습

## 개요

다양한 분류 알고리즘과 앙상블 기법을 학습합니다.  
결정 트리에서 시작해 랜덤 포레스트, GBM, XGBoost, LightGBM까지 발전 과정을 이해합니다.

---

## 1. 결정 트리 (Decision Tree)

### 개념

데이터를 특성(feature) 기준으로 반복적으로 나누어 분류하는 트리 구조 모델.

```
[나이 > 30?]
  예 → [소득 > 5만?]
          예 → 구매 (Positive)
          아니오 → 미구매
  아니오 → 미구매
```

```python
from sklearn.tree import DecisionTreeClassifier

# 주요 하이퍼파라미터
clf = DecisionTreeClassifier(
    max_depth=5,          # 트리 최대 깊이 (과적합 방지)
    min_samples_split=2,  # 분기를 위한 최소 샘플 수
    min_samples_leaf=1    # 리프 노드 최소 샘플 수
)
clf.fit(X_train, y_train)

# 특성 중요도
import pandas as pd
importances = pd.Series(clf.feature_importances_, index=X.columns)
importances.sort_values(ascending=False)
```

> **과적합 주의**: max_depth를 너무 크게 하면 훈련 데이터에 과적합됩니다.

---

## 2. 앙상블 학습 (Ensemble Learning)

여러 모델을 결합해 단일 모델보다 더 좋은 성능을 내는 기법입니다.

### 앙상블 방식

| 방식 | 설명 | 예시 |
|------|------|------|
| **Voting** | 여러 모델의 예측을 투표 | Hard/Soft Voting |
| **Bagging** | 병렬로 여러 모델 학습 후 집계 | 랜덤 포레스트 |
| **Boosting** | 순차적으로 오류 수정 학습 | GBM, XGBoost, LightGBM |
| **Stacking** | 예측값을 메타 모델의 입력으로 사용 | 메타 학습 |

---

## 3. 보팅 (Voting)

```python
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

lr_clf = LogisticRegression()
knn_clf = KNeighborsClassifier(n_neighbors=8)

# Soft Voting: 각 모델의 확률 평균으로 결정 (Hard보다 보통 성능 좋음)
vo_clf = VotingClassifier(
    estimators=[('LR', lr_clf), ('KNN', knn_clf)],
    voting='soft'
)
vo_clf.fit(X_train, y_train)
```

---

## 4. 랜덤 포레스트 (Random Forest)

여러 결정 트리를 **병렬**로 학습하고 투표로 최종 예측합니다.  
각 트리는 랜덤 샘플링된 데이터와 특성으로 학습합니다.

```python
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(
    n_estimators=100,     # 트리 수 (많을수록 안정적)
    max_depth=8,          # 각 트리의 최대 깊이
    min_samples_leaf=1,
    random_state=0,
    n_jobs=-1             # CPU 모든 코어 사용
)
rf_clf.fit(X_train, y_train)
pred = rf_clf.predict(X_test)
```

---

## 5. GBM (Gradient Boosting Machine)

이전 트리의 **오류를 보정**하는 방향으로 순차적으로 트리를 추가합니다.

```python
from sklearn.ensemble import GradientBoostingClassifier

gb_clf = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,   # 각 트리의 가중치 (작을수록 안정적, 느림)
    max_depth=3
)
gb_clf.fit(X_train, y_train)
```

---

## 6. XGBoost

GBM을 개선한 고성능 부스팅 라이브러리. 병렬 처리와 정규화를 지원합니다.

```python
from xgboost import XGBClassifier

xgb_clf = XGBClassifier(
    n_estimators=400,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,       # 각 트리에 사용할 데이터 비율
    colsample_bytree=0.8, # 각 트리에 사용할 특성 비율
    eval_metric='logloss',
    early_stopping_rounds=100  # 성능 개선 없으면 조기 종료
)

# 조기 종료를 위한 평가 세트 지정
xgb_clf.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=100
)
```

### XGBoost 주요 파라미터

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `n_estimators` | 트리 수 | 100 |
| `learning_rate` | 학습률 | 0.1 |
| `max_depth` | 트리 깊이 | 6 |
| `subsample` | 행 샘플링 비율 | 1 |
| `colsample_bytree` | 열 샘플링 비율 | 1 |
| `reg_alpha` | L1 정규화 | 0 |
| `reg_lambda` | L2 정규화 | 1 |

---

## 7. LightGBM

XGBoost보다 **빠른** 부스팅 라이브러리. 대용량 데이터에 적합합니다.

```python
from lightgbm import LGBMClassifier

lgbm_clf = LGBMClassifier(
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=31,         # 리프 수 (클수록 복잡한 모델)
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1
)

lgbm_clf.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
)
```

### XGBoost vs LightGBM 비교

| 항목 | XGBoost | LightGBM |
|------|---------|---------|
| 학습 속도 | 느림 | 빠름 |
| 메모리 사용 | 많음 | 적음 |
| 적합한 데이터 | 소~중형 | 중~대형 |
| 정확도 | 비슷 | 비슷 |
| 소량 데이터 | 적합 | 과적합 주의 |

---

## 8. 베이지안 최적화 (Hyperparameter Tuning)

GridSearchCV보다 효율적으로 최적 하이퍼파라미터를 찾습니다.

```python
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK

# 탐색 공간 정의
space = {
    'max_depth': hp.quniform('max_depth', 5, 15, 1),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
    'n_estimators': hp.quniform('n_estimators', 100, 500, 50)
}

# 목적 함수 (최소화)
def objective(params):
    params['max_depth'] = int(params['max_depth'])
    xgb = XGBClassifier(**params)
    accuracy = cross_val_score(xgb, X_train, y_train, cv=3).mean()
    return {'loss': -accuracy, 'status': STATUS_OK}

# 최적화 실행
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=50)
print(best)
```

---

## 9. 스태킹 앙상블 (Stacking)

1단계 모델들의 예측값을 2단계 메타 모델의 입력으로 사용합니다.

```python
# 1단계: 여러 모델의 예측값 생성
knn_pred = knn_clf.predict(X_test)
rf_pred = rf_clf.predict(X_test)
dt_pred = dt_clf.predict(X_test)
ada_pred = ada_clf.predict(X_test)

# 예측값 스택
final_X = np.column_stack([knn_pred, rf_pred, dt_pred, ada_pred])

# 2단계: 메타 모델 학습
meta_model = LogisticRegression()
meta_model.fit(final_X_train, y_train)
final_pred = meta_model.predict(final_X_test)
```

---

## 10. 피처 선택 (Feature Selection)

```python
from sklearn.feature_selection import RFE, SelectFromModel

# RFE: 재귀적 특성 제거
estimator = RandomForestClassifier()
rfe = RFE(estimator=estimator, n_features_to_select=10)
rfe.fit(X_train, y_train)
selected = X.columns[rfe.support_]

# SelectFromModel: 중요도 기반 선택
selector = SelectFromModel(rf_clf, threshold='mean')
selector.fit(X_train, y_train)
X_selected = selector.transform(X_test)
```

---

## 알고리즘 선택 가이드

| 상황 | 추천 알고리즘 |
|------|-------------|
| 빠른 기준선 모델 | 결정 트리, 랜덤 포레스트 |
| 최고 성능 목표 | XGBoost, LightGBM |
| 대용량 데이터 | LightGBM |
| 해석 가능성 중요 | 결정 트리 |
| 여러 모델 결합 | 스태킹, 보팅 |

---

[← 3장: 모델 평가](03_evaluation.md) | [5장: 회귀 →](05_regression.md)
