# 2장: 머신러닝 입문 (사이킷런)

## 개요

사이킷런(scikit-learn)을 사용한 머신러닝의 기본 워크플로우를 학습합니다.  
붓꽃(Iris) 분류와 타이타닉 생존자 예측 실습을 통해 전체 ML 파이프라인을 이해합니다.

---

## 머신러닝 기본 워크플로우

```
데이터 로드 → 전처리 → 학습/테스트 분리 → 모델 학습 → 예측 → 평가
```

---

## 1. 첫 번째 머신러닝: 붓꽃 분류

### 사이킷런 기본 패턴

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. 데이터 로드
iris = load_iris()
X = iris.data     # 특성 (꽃잎/꽃받침 길이/너비)
y = iris.target   # 레이블 (0=Setosa, 1=Versicolor, 2=Virginica)

# 2. 학습/테스트 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=11
)

# 3. 모델 생성 및 학습
clf = DecisionTreeClassifier(random_state=11)
clf.fit(X_train, y_train)

# 4. 예측 및 평가
pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, pred)
print(f'정확도: {accuracy:.4f}')  # 약 0.9333
```

---

## 2. 모델 선택과 교차 검증

### 왜 교차 검증이 필요한가?

단순히 train/test로 한 번만 나누면 데이터 분리 방식에 따라 성능이 달라질 수 있습니다.  
교차 검증(Cross Validation)은 데이터를 여러 번 나눠 더 신뢰할 수 있는 성능을 측정합니다.

### K-Fold 교차 검증

```python
from sklearn.model_selection import KFold, cross_val_score

# K-Fold: 데이터를 K개로 나눠 K번 학습/검증 반복
kfold = KFold(n_splits=5)

# cross_val_score: K-Fold를 자동으로 실행
scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
print(f'교차 검증 점수: {scores}')
print(f'평균 정확도: {scores.mean():.4f}')
```

### Stratified K-Fold (분류 문제 권장)

클래스 비율을 유지하며 폴드를 나눕니다. 불균형 데이터에 특히 중요합니다.

```python
from sklearn.model_selection import StratifiedKFold

skfold = StratifiedKFold(n_splits=5)
scores = cross_val_score(clf, X, y, cv=skfold)
```

### GridSearchCV (하이퍼파라미터 튜닝 + 교차 검증)

```python
from sklearn.model_selection import GridSearchCV

params = {
    'max_depth': [1, 2, 3],
    'min_samples_split': [2, 3]
}

grid_cv = GridSearchCV(DecisionTreeClassifier(), param_grid=params, cv=3)
grid_cv.fit(X_train, y_train)

print(f'최적 파라미터: {grid_cv.best_params_}')
print(f'최고 점수: {grid_cv.best_score_:.4f}')
```

---

## 3. 데이터 전처리

### 레이블 인코딩 (Label Encoding)

문자열 카테고리를 숫자로 변환합니다.

```python
from sklearn.preprocessing import LabelEncoder

items = ['TV', '냉장고', '전자레인지', '컴퓨터', 'TV']
encoder = LabelEncoder()
encoded = encoder.fit_transform(items)
print(encoded)          # [0, 1, 4, 2, 0] (알파벳 순)
print(encoder.classes_) # ['TV', '냉장고', '전자레인지', '컴퓨터']
```

> **주의**: 레이블 인코딩은 숫자 크기(1 < 2 < 3)가 의미를 가지게 됩니다.  
> 선형 모델에서는 원-핫 인코딩을 사용하세요.

### 원-핫 인코딩 (One-Hot Encoding)

카테고리 수만큼 0/1 컬럼을 생성합니다.

```python
import pandas as pd

df = pd.DataFrame({'item': ['TV', '냉장고', '컴퓨터']})
pd.get_dummies(df['item'])
# TV  냉장고  컴퓨터
#  1      0      0
#  0      1      0
#  0      0      1
```

### 피처 스케일링

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# StandardScaler: 평균=0, 표준편차=1로 변환
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# MinMaxScaler: 0~1 사이로 변환
minmax = MinMaxScaler()
X_scaled = minmax.fit_transform(X_train)
```

> **중요**: `fit()`은 훈련 데이터에만 적용! 테스트 데이터에는 `transform()`만 적용해야 데이터 누수를 방지합니다.

---

## 4. 실전: 타이타닉 생존자 예측

### 전체 파이프라인 예시

```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. 데이터 로드
df_train = pd.read_csv('titanic/train.csv')

# 2. 불필요한 컬럼 제거
df_train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# 3. 결측치 처리
df_train['Age'].fillna(df_train['Age'].mean(), inplace=True)
df_train['Embarked'].fillna('N', inplace=True)

# 4. 문자열 → 숫자 변환
from sklearn.preprocessing import LabelEncoder
for col in ['Sex', 'Embarked']:
    le = LabelEncoder()
    df_train[col] = le.fit_transform(df_train[col])

# 5. 학습/테스트 분리
X = df_train.drop('Survived', axis=1)
y = df_train['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 6. 모델 학습 및 평가
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print(f'정확도: {accuracy_score(y_test, pred):.4f}')
```

---

## 학습 포인트

| 개념 | 핵심 내용 |
|------|----------|
| `train_test_split` | 보통 test_size=0.2 (80/20 분할) |
| `fit` vs `predict` | `fit`=학습, `predict`=예측, `fit_transform`=학습+변환 |
| 교차 검증 | 데이터 분리 운에 의존하지 않는 신뢰할 수 있는 성능 측정 |
| 레이블 인코딩 | 순서 의미 없는 카테고리는 원-핫 인코딩 권장 |
| 스케일링 | fit은 훈련 데이터에만! 테스트 데이터는 transform만 |
| `random_state` | 재현 가능한 결과를 위해 고정 |

---

[← 1장: NumPy & Pandas](01_numpy_pandas.md) | [3장: 모델 평가 →](03_evaluation.md)
