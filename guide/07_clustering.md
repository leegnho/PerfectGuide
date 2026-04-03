# 7장: 클러스터링 (Clustering)

## 개요

레이블 없이 데이터의 유사한 패턴을 자동으로 그룹화하는 비지도 학습을 학습합니다.  
K-Means, Mean Shift, GMM, DBSCAN과 실전 고객 세그멘테이션을 다룹니다.

---

## 클러스터링이란?

정답(레이블) 없이 데이터 자체의 유사성을 기반으로 자동으로 그룹을 찾는 기법.

**활용 사례**:
- 고객 세그멘테이션 (비슷한 구매 패턴끼리 묶기)
- 이상 탐지 (어느 클러스터에도 속하지 않는 데이터)
- 문서 군집화

---

## 1. K-Means 클러스터링

### 알고리즘 작동 방식

```
1. K개의 임의 중심점(centroid) 설정
2. 각 데이터를 가장 가까운 중심점의 클러스터에 할당
3. 각 클러스터의 평균으로 중심점 업데이트
4. 중심점이 변하지 않을 때까지 2~3 반복
```

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 스케일링 (거리 기반이므로 필수)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means 학습
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, random_state=0)
kmeans.fit(X_scaled)

# 클러스터 레이블
labels = kmeans.labels_      # 각 데이터의 클러스터 번호
centers = kmeans.cluster_centers_  # 클러스터 중심점
```

### 최적 K 찾기 - 엘보우 방법

```python
inertia_list = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X_scaled)
    inertia_list.append(kmeans.inertia_)  # 클러스터 내 거리 합

plt.plot(range(2, 11), inertia_list, 'bx-')
plt.xlabel('K')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()
# 기울기가 급격히 완만해지는 지점이 최적 K
```

---

## 2. 클러스터 평가: 실루엣 분석

레이블 없이도 클러스터 품질을 평가하는 방법.

### 실루엣 계수 (Silhouette Coefficient)

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

- **a(i)**: 같은 클러스터 내 평균 거리 (응집도)
- **b(i)**: 가장 가까운 다른 클러스터까지의 평균 거리 (분리도)
- **범위**: -1 ~ 1 (1에 가까울수록 좋음)

```python
from sklearn.metrics import silhouette_score, silhouette_samples

# 전체 평균 실루엣 점수
avg_score = silhouette_score(X_scaled, labels)
print(f'평균 실루엣 점수: {avg_score:.4f}')

# 샘플별 실루엣 점수
sample_scores = silhouette_samples(X_scaled, labels)
```

---

## 3. Mean Shift

데이터 밀도가 높은 곳으로 중심점을 이동시키며 클러스터를 찾습니다.  
K를 미리 지정하지 않아도 됩니다.

```python
from sklearn.cluster import MeanShift, estimate_bandwidth

# 최적 bandwidth 자동 추정
bandwidth = estimate_bandwidth(X_scaled, quantile=0.25)

ms = MeanShift(bandwidth=bandwidth)
ms.fit(X_scaled)

labels = ms.labels_
n_clusters = len(set(labels))
print(f'자동으로 찾은 클러스터 수: {n_clusters}')
```

---

## 4. GMM (Gaussian Mixture Model)

각 클러스터가 가우시안(정규) 분포를 따른다고 가정합니다.  
각 데이터에 **확률적**으로 클러스터를 할당합니다.

```python
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=3, random_state=0)
gmm.fit(X_scaled)

labels = gmm.predict(X_scaled)
proba = gmm.predict_proba(X_scaled)  # 각 클러스터에 속할 확률
```

### K-Means vs GMM 비교

| 항목 | K-Means | GMM |
|------|---------|-----|
| 클러스터 모양 | 구형(spherical) | 타원형 (유연) |
| 할당 방식 | 하드 (하나에만 속함) | 소프트 (확률로 표현) |
| 계산 비용 | 낮음 | 높음 |
| 이상치 영향 | 민감 | 덜 민감 |

---

## 5. DBSCAN

**밀도** 기반 클러스터링. 밀도가 높은 영역을 클러스터로 정의합니다.

```
핵심 포인트: 반경 epsilon 내에 min_samples 이상의 이웃이 있는 점
경계 포인트: 핵심 포인트의 이웃이지만 자신은 핵심 포인트가 아닌 점
노이즈 포인트: 어느 클러스터에도 속하지 않는 점 (레이블 -1)
```

```python
from sklearn.cluster import DBSCAN

db = DBSCAN(eps=0.5,          # 이웃 반경
            min_samples=5)    # 핵심 포인트 조건
db.fit(X_scaled)

labels = db.labels_
# -1 = 노이즈 포인트
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print(f'클러스터 수: {n_clusters}')
print(f'노이즈 포인트: {n_noise}')
```

### DBSCAN 장단점

**장점**:
- K를 미리 지정하지 않아도 됨
- 임의 형태의 클러스터 탐지
- 노이즈(이상치) 자동 탐지

**단점**:
- epsilon, min_samples 선택이 어려움
- 밀도가 다른 클러스터에서 성능 저하

---

## 6. 알고리즘 비교

| 알고리즘 | K 지정 | 클러스터 모양 | 이상치 | 특징 |
|---------|--------|------------|-------|------|
| K-Means | 필요 | 구형 | 민감 | 빠르고 단순 |
| Mean Shift | 불필요 | 자유 | 영향 적음 | bandwidth 선택 필요 |
| GMM | 필요 | 타원형 | 영향 적음 | 확률적 할당 |
| DBSCAN | 불필요 | 자유 | 자동 탐지 | 밀도 기반 |

---

## 7. 실전: 고객 세그멘테이션 (RFM 분석)

### RFM 지표

- **Recency (최신성)**: 마지막 구매로부터 경과 일수 (작을수록 최근)
- **Frequency (빈도)**: 총 구매 횟수 (클수록 자주 구매)
- **Monetary (금액)**: 총 구매 금액 (클수록 많이 구매)

```python
import pandas as pd
from datetime import datetime

# RFM 계산
now = datetime(2011, 12, 11)
rfm_df = df.groupby('CustomerID').agg(
    Recency=('InvoiceDate', lambda x: (now - x.max()).days),
    Frequency=('InvoiceNo', 'nunique'),
    Monetary=('UnitPrice', lambda x: (x * df.loc[x.index, 'Quantity']).sum())
).reset_index()

# 로그 변환 (왜도 감소)
rfm_log = np.log1p(rfm_df[['Recency', 'Frequency', 'Monetary']])

# 스케일링 후 K-Means
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_log)

kmeans = KMeans(n_clusters=4, random_state=42)
rfm_df['Cluster'] = kmeans.fit_predict(rfm_scaled)

# 클러스터별 통계
rfm_df.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
```

---

## 학습 포인트

| 개념 | 핵심 내용 |
|------|----------|
| 스케일링 | 거리 기반 클러스터링에서 필수 |
| 엘보우 방법 | Inertia 감소 기울기로 K 선택 |
| 실루엣 점수 | -1~1, 높을수록 좋은 클러스터 |
| DBSCAN 노이즈 | 레이블 -1 → 이상치/노이즈로 활용 |
| RFM 분석 | 고객 세그멘테이션의 대표 방법 |

---

[← 6장: 차원 축소](06_dim_reduction.md) | [8장: 텍스트 분석 →](08_nlp.md)
