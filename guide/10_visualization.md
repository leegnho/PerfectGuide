# Visual: 데이터 시각화

## 개요

Matplotlib과 Seaborn을 사용한 데이터 시각화를 학습합니다.  
기본 그래프 생성부터 통계 시각화까지 다양한 차트를 다룹니다.

---

## 1. Matplotlib 기초

### Figure와 Axes 구조

```
Figure (전체 캔버스)
└── Axes (개별 그래프 영역)
    ├── Title
    ├── X축 (xlabel, xticks)
    ├── Y축 (ylabel, yticks)
    └── Plot (line, scatter, bar, ...)
```

```python
import matplotlib.pyplot as plt
import numpy as np

# 기본 설정 (한글 폰트)
plt.rcParams['font.family'] = 'Malgun Gothic'   # Windows
# plt.rcParams['font.family'] = 'AppleGothic'   # Mac

# 기본 그래프
fig, ax = plt.subplots(figsize=(8, 5))
x = np.linspace(0, 10, 100)
ax.plot(x, np.sin(x), label='sin(x)', color='blue', linewidth=2)
ax.plot(x, np.cos(x), label='cos(x)', color='red', linestyle='--')
ax.set_title('삼각함수', fontsize=14)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()
ax.grid(True)
plt.show()
```

### 서브플롯 (여러 그래프)

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 각 서브플롯에 그래프 그리기
axes[0, 0].plot(x, np.sin(x))
axes[0, 0].set_title('Line Plot')

axes[0, 1].scatter(np.random.randn(50), np.random.randn(50))
axes[0, 1].set_title('Scatter Plot')

axes[1, 0].bar(['A', 'B', 'C', 'D'], [3, 7, 5, 2], color='steelblue')
axes[1, 0].set_title('Bar Chart')

axes[1, 1].hist(np.random.randn(500), bins=30, edgecolor='black')
axes[1, 1].set_title('Histogram')

plt.tight_layout()  # 서브플롯 간격 자동 조정
plt.show()
```

### 자주 쓰는 그래프 유형

```python
# 선 그래프
ax.plot(x, y, color='blue', linewidth=2, linestyle='-', marker='o')

# 막대 그래프
ax.bar(categories, values, color='steelblue', width=0.6)
ax.barh(categories, values)  # 수평 막대

# 산점도
ax.scatter(x, y, c=colors, s=sizes, alpha=0.7, cmap='viridis')

# 히스토그램
ax.hist(data, bins=30, edgecolor='black', density=True)

# 파이 차트
ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)

# 박스 플롯
ax.boxplot(data_list, labels=labels)
```

---

## 2. Seaborn 통계 시각화

Seaborn은 Matplotlib 위에 구축된 통계 시각화 라이브러리입니다.  
데이터프레임을 직접 활용하고 아름다운 기본 스타일을 제공합니다.

```python
import seaborn as sns
import pandas as pd

# 기본 스타일 설정
sns.set_theme(style='whitegrid')

# 타이타닉 데이터 로드
titanic = sns.load_dataset('titanic')
```

### 분포 시각화

```python
# 히스토그램 + KDE
sns.histplot(data=titanic, x='age', hue='survived', bins=30, kde=True)

# KDE 플롯 (밀도 추정)
sns.kdeplot(data=titanic, x='age', hue='sex', fill=True, alpha=0.5)

# 바이올린 플롯 (분포 모양 비교)
sns.violinplot(data=titanic, x='class', y='age', hue='sex', split=True)
```

### 관계 시각화

```python
# 산점도 with 회귀선
sns.regplot(data=df, x='fare', y='age', scatter_kws={'alpha': 0.5})

# 카테고리별 산점도
sns.scatterplot(data=titanic, x='age', y='fare',
                hue='survived', style='sex', size='pclass')

# 쌍 그래프 (모든 변수 조합)
sns.pairplot(titanic[['age', 'fare', 'pclass', 'survived']], hue='survived')
```

### 카테고리 시각화

```python
# 막대 그래프 (평균 + 신뢰구간)
sns.barplot(data=titanic, x='class', y='survived',
            hue='sex', palette='Set2')

# 박스 플롯
sns.boxplot(data=titanic, x='class', y='age', hue='survived')

# 카운트 플롯
sns.countplot(data=titanic, x='class', hue='survived', palette='Set1')
```

### 히트맵 (상관관계 분석)

```python
# 수치형 컬럼 간 상관계수 행렬
numeric_cols = titanic.select_dtypes(include='number')
corr_matrix = numeric_cols.corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    corr_matrix,
    annot=True,       # 수치 표시
    fmt='.2f',        # 소수점 2자리
    cmap='coolwarm',  # 색상 맵 (파랑=음수, 빨강=양수)
    center=0,         # 0을 중심으로
    linewidths=0.5,
    ax=ax
)
ax.set_title('상관관계 히트맵')
plt.show()
```

### FacetGrid (조건별 그래프)

```python
# 조건별 다중 그래프
g = sns.FacetGrid(titanic, col='sex', row='class', height=3)
g.map(sns.histplot, 'age', bins=20)
g.add_legend()
```

---

## 3. 머신러닝에서 자주 쓰는 시각화

### 특성 중요도 시각화

```python
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# 특성 중요도 시각화
importances = pd.Series(rf.feature_importances_, index=X.columns)
importances.sort_values(ascending=True).tail(20).plot(
    kind='barh', figsize=(10, 8), title='Feature Importances'
)
plt.tight_layout()
plt.show()
```

### 학습 곡선

```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    estimator, X, y, cv=5, n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10)
)

plt.plot(train_sizes, train_scores.mean(axis=1), label='Train')
plt.plot(train_sizes, val_scores.mean(axis=1), label='Validation')
plt.xlabel('Training Size')
plt.ylabel('Score')
plt.legend()
plt.title('Learning Curve')
```

### ROC 곡선

```python
from sklearn.metrics import roc_curve, auc

fpr, tpr, _ = roc_curve(y_test, pred_proba)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
```

---

## 색상 팔레트 가이드

| 팔레트 | 용도 |
|--------|------|
| `Set1`, `Set2`, `tab10` | 카테고리 구분 (서로 다른 색) |
| `Blues`, `Reds`, `Greens` | 단색 순차 (낮음→높음) |
| `coolwarm`, `RdBu` | 발산형 (음수/양수 구분) |
| `viridis`, `plasma` | 연속 데이터 (시각적으로 균일) |

---

## 학습 포인트

| 포인트 | 내용 |
|--------|------|
| Figure vs Axes | Figure는 전체 캔버스, Axes는 개별 그래프 |
| `tight_layout()` | 서브플롯 겹침 방지 |
| Seaborn `hue` | 카테고리별 색상 자동 구분 |
| 히트맵 | 상관관계 파악에 필수 |
| `figsize` | 인치 단위, (가로, 세로) 순서 |

---

[← 9장: 추천 시스템](09_recommendation.md) | [목차로 돌아가기 →](../README.md)
