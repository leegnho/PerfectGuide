# 1장: NumPy & Pandas 기초

## 개요

머신러닝에서 데이터를 다루기 위한 두 가지 핵심 라이브러리를 학습합니다.  
- **NumPy**: 수치 연산 및 다차원 배열 처리  
- **Pandas**: 표 형태의 데이터(DataFrame) 조작 및 분석  

---

## 1. NumPy

### 핵심 개념

#### ndarray (N차원 배열)
NumPy의 기본 자료구조. 파이썬 리스트보다 빠르고 메모리 효율적입니다.

```python
import numpy as np

# 1차원 배열
arr1 = np.array([1, 2, 3])

# 2차원 배열 (행렬)
arr2 = np.array([[1, 2, 3], [4, 5, 6]])

print(arr2.shape)   # (2, 3) → 2행 3열
print(arr2.ndim)    # 2 → 차원 수
print(arr2.dtype)   # int64 → 데이터 타입
```

#### 배열 생성 함수

```python
np.arange(0, 10, 2)     # [0, 2, 4, 6, 8]
np.zeros((3, 4))        # 3행 4열 모두 0
np.ones((2, 3))         # 2행 3열 모두 1
np.linspace(0, 1, 5)    # 0~1 사이 균등한 5개: [0, 0.25, 0.5, 0.75, 1]
```

#### 형태 변환 (reshape)

```python
arr = np.arange(12)         # [0, 1, 2, ..., 11]
arr.reshape(3, 4)           # 3행 4열로 변환
arr.reshape(-1, 2)          # 열은 2, 행은 자동계산 → (6, 2)
arr.reshape(2, -1)          # 행은 2, 열은 자동계산 → (2, 6)
```

#### 인덱싱과 슬라이싱

```python
arr2d = np.array([[1,2,3],[4,5,6],[7,8,9]])

# 단일 인덱싱
arr2d[0, 2]      # 1행 3열 → 3

# 슬라이싱
arr2d[0:2, 1:]   # 0~1행, 1열이후 → [[2,3],[5,6]]

# 불리언 인덱싱 (조건 필터링)
arr = np.array([1, 2, 3, 4, 5])
arr[arr > 3]     # [4, 5]

# 팬시 인덱싱 (특정 위치)
arr[[0, 2, 4]]   # [1, 3, 5]
```

#### 정렬

```python
arr = np.array([3, 1, 5, 2, 4])

np.sort(arr)                  # [1, 2, 3, 4, 5] - 원본 변경 없음
arr.sort()                    # 원본 변경

np.argsort(arr)               # 정렬 후 원래 인덱스 반환 → [1, 3, 0, 4, 2]
```

#### 행렬 연산

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

np.dot(A, B)    # 행렬 곱 (내적)
A.T             # 전치 행렬 (행/열 뒤집기)
```

---

## 2. Pandas

### 핵심 개념

#### DataFrame과 Series
- **DataFrame**: 2차원 표 (엑셀 시트처럼)
- **Series**: 1차원 컬럼 (DataFrame의 한 열)

```python
import pandas as pd

# CSV 파일 읽기
df = pd.read_csv('data.csv')

# 기본 정보 확인
df.shape            # (행 수, 열 수)
df.info()           # 컬럼별 데이터 타입, 결측치 수
df.describe()       # 수치형 컬럼 통계 (평균, 표준편차 등)
df.head(5)          # 처음 5행
df.tail(5)          # 마지막 5행
```

#### 데이터 접근

```python
# 컬럼 선택
df['Age']           # Series 반환
df[['Age', 'Sex']]  # DataFrame 반환

# 행 선택
df.loc[0]           # 라벨(이름)로 선택
df.iloc[0]          # 정수 인덱스로 선택

# 조건 필터링
df[df['Age'] > 30]                      # 나이가 30 초과
df[(df['Age'] > 20) & (df['Sex'] == 'male')]  # AND 조건
```

#### 데이터 전처리

```python
# 결측치 확인 및 처리
df.isnull().sum()               # 컬럼별 결측치 수
df['Age'].fillna(df['Age'].mean())  # 평균으로 채우기
df.dropna()                     # 결측치 있는 행 제거

# 값 변환
df['Sex'].map({'male': 0, 'female': 1})   # 값 매핑
df['Cabin'].apply(lambda x: x[0] if pd.notna(x) else 'N')  # 함수 적용

# 그룹화 및 집계
df.groupby('Pclass')['Survived'].mean()   # 객실 등급별 생존율 평균
df.groupby('Sex')['Age'].agg(['mean', 'max'])  # 성별 나이 통계
```

#### 데이터프레임 조작

```python
# 새 컬럼 추가
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# 컬럼 삭제
df.drop('PassengerId', axis=1, inplace=True)

# 정렬
df.sort_values('Age', ascending=False)

# 중복 제거
df.drop_duplicates()
```

---

## 학습 포인트

| 개념 | 핵심 내용 |
|------|----------|
| ndarray shape | `(행, 열, ...)` 순서로 차원 표시 |
| reshape(-1, n) | `-1`은 자동 계산 (나머지 값으로 결정) |
| 불리언 인덱싱 | 조건식을 배열에 직접 적용 가능 |
| df.loc vs iloc | `loc`=라벨 기반, `iloc`=정수 인덱스 기반 |
| groupby | SQL의 GROUP BY와 동일한 개념 |
| fillna vs dropna | 결측치 처리 방법 선택은 데이터와 목적에 따라 다름 |

---

## 다음 챕터

[2장: 머신러닝 입문 →](02_ml_intro.md)
