# 6장: 차원 축소 (Dimensionality Reduction)

## 개요

고차원 데이터를 저차원으로 변환하는 기법을 학습합니다.  
데이터 시각화, 노이즈 제거, 과적합 방지, 계산 효율화 등에 활용됩니다.

---

## 왜 차원 축소가 필요한가?

- **차원의 저주**: 특성이 많아질수록 데이터가 희소해지고 모델 성능 저하
- **시각화**: 2~3차원으로 줄여 데이터 구조 파악
- **노이즈 제거**: 중요한 정보만 남기고 불필요한 특성 제거
- **계산 속도 향상**: 특성 수 감소 → 학습 속도 향상

---

## 차원 축소 방식

| 방식 | 설명 | 예시 |
|------|------|------|
| **특성 선택** | 원래 특성 중 중요한 것만 선택 | SelectFromModel |
| **특성 추출** | 기존 특성을 변환해 새 특성 생성 | PCA, LDA, SVD, NMF |

---

## 1. PCA (Principal Component Analysis, 주성분 분석)

### 개념

데이터의 **분산이 최대인 방향**으로 투영하여 차원을 줄입니다.  
비지도 학습 방식 (레이블 정보 불필요).

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1단계: 반드시 스케일링 먼저!
scaler = StandardScaler()
iris_scaled = scaler.fit_transform(iris_df.iloc[:, :-1])

# 2단계: PCA 적용
pca = PCA(n_components=2)  # 2차원으로 축소
iris_pca = pca.fit_transform(iris_scaled)

# 각 주성분이 설명하는 분산 비율
print(pca.explained_variance_ratio_)
# [0.729, 0.229] → 1번째 PC: 72.9%, 2번째 PC: 22.9%
# 총 95.8% 분산 보존
```

### PCA 시각화

```python
import matplotlib.pyplot as plt

pca_df = pd.DataFrame(iris_pca, columns=['PC1', 'PC2'])
pca_df['target'] = iris.target

for target, color in zip([0, 1, 2], ['r', 'g', 'b']):
    mask = pca_df['target'] == target
    plt.scatter(pca_df[mask]['PC1'], pca_df[mask]['PC2'],
                c=color, label=iris.target_names[target])
plt.legend()
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA of Iris Dataset')
plt.show()
```

### 적절한 주성분 수 선택

```python
pca_full = PCA()
pca_full.fit(X_scaled)

# 누적 설명 분산 비율
cumsum = np.cumsum(pca_full.explained_variance_ratio_)
# cumsum[n] >= 0.95 인 최소 n 찾기
n_components = np.argmax(cumsum >= 0.95) + 1
```

---

## 2. LDA (Linear Discriminant Analysis, 선형 판별 분석)

### 개념

**클래스 간 분산은 최대화**하고 **클래스 내 분산은 최소화**하는 방향으로 투영합니다.  
지도 학습 방식 (레이블 정보 사용).

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(n_components=2)
iris_lda = lda.fit_transform(iris_scaled, iris.target)

# PCA vs LDA 비교
# PCA: 분산 최대 방향 (클래스 무관)
# LDA: 클래스 분리 최대 방향
```

---

## 3. SVD (Singular Value Decomposition, 특이값 분해)

### 개념

행렬을 세 행렬의 곱으로 분해합니다:  
$$A = U \cdot \Sigma \cdot V^T$$

- **U**: 좌측 특이 벡터 (행 방향 패턴)
- **Σ**: 대각 행렬 (특이값, 중요도)
- **V^T**: 우측 특이 벡터 (열 방향 패턴)

```python
from sklearn.decomposition import TruncatedSVD

# TruncatedSVD: 상위 n개 특이값만 사용
svd = TruncatedSVD(n_components=2)
X_svd = svd.fit_transform(X)

print(svd.explained_variance_ratio_)
```

> **SVD와 PCA의 관계**: PCA는 데이터를 중심화(평균 빼기) 후 SVD를 적용합니다.  
> 희소 행렬(sparse matrix)에는 TruncatedSVD를 직접 적용하는 것이 효율적입니다.

---

## 4. NMF (Non-negative Matrix Factorization, 비음수 행렬 분해)

### 개념

모든 값이 양수인 행렬을 두 양수 행렬의 곱으로 분해합니다.

$$V \approx W \cdot H$$

- **W**: 기저 벡터 (주제, 특성)
- **H**: 가중치 행렬 (문서-주제 관계)

```python
from sklearn.decomposition import NMF

nmf = NMF(n_components=2, random_state=42)
X_nmf = nmf.fit_transform(X)

# 각 성분의 특성 중요도
components = pd.DataFrame(nmf.components_, columns=feature_names)
```

> **활용**: 텍스트 토픽 모델링, 이미지 분해, 추천 시스템 등에 적합

---

## 비교 요약

| 방법 | 학습 방식 | 음수 허용 | 주요 활용 |
|------|---------|---------|---------|
| PCA | 비지도 | O | 시각화, 노이즈 제거 |
| LDA | 지도 | O | 분류 전 차원 축소 |
| SVD | 비지도 | O | 희소 행렬 (텍스트 등) |
| NMF | 비지도 | X (양수만) | 토픽 모델링, 이미지 |

---

## 실전 적용 순서

```python
# 표준 파이프라인
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

pipeline = Pipeline([
    ('scaler', StandardScaler()),   # 1. 스케일링
    ('pca', PCA(n_components=6)),   # 2. 차원 축소
    ('clf', RandomForestClassifier()) # 3. 모델 학습
])

pipeline.fit(X_train, y_train)
pred = pipeline.predict(X_test)
```

---

## 학습 포인트

| 개념 | 핵심 내용 |
|------|----------|
| PCA 스케일링 | PCA 전에 StandardScaler 필수 |
| 설명 분산 비율 | `explained_variance_ratio_`로 정보 보존량 확인 |
| PCA vs LDA | PCA=비지도(분산 기준), LDA=지도(클래스 분리 기준) |
| SVD 활용 | 텍스트 TF-IDF 행렬에 직접 적용 가능 |
| NMF 특징 | 양수만 허용 → 해석 가능한 성분 추출 |

---

[← 5장: 회귀](05_regression.md) | [7장: 클러스터링 →](07_clustering.md)
