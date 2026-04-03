# 3장: 분류 모델 평가

## 개요

분류 모델의 성능을 다양한 지표로 측정하는 방법을 학습합니다.  
단순 정확도 외에도 정밀도, 재현율, F1, ROC-AUC 등을 상황에 맞게 사용합니다.

---

## 왜 정확도만으로는 부족한가?

**불균형 데이터 문제**: 100명 중 1명만 암 환자인 경우, 모두 "정상"이라 예측해도 정확도 99%.  
→ 하지만 이 모델은 쓸모없는 모델!

---

## 1. 오차 행렬 (Confusion Matrix)

```
                 예측 Negative    예측 Positive
실제 Negative       TN                FP
실제 Positive       FN                TP
```

- **TN**: 실제 음성 → 음성 예측 (맞음)
- **FP**: 실제 음성 → 양성 예측 (틀림, 1종 오류)
- **FN**: 실제 양성 → 음성 예측 (틀림, 2종 오류)
- **TP**: 실제 양성 → 양성 예측 (맞음)

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, pred)
print(cm)
# [[104   5]
#  [ 14  56]]
```

---

## 2. 정밀도 (Precision) vs 재현율 (Recall)

### 정밀도 (Precision) - "예측한 것 중에 얼마나 맞았나"

$$\text{Precision} = \frac{TP}{TP + FP}$$

- 양성으로 예측했는데 실제로도 양성인 비율
- **스팸 필터** 등에서 중요: 정상 메일을 스팸으로 분류하면 안 됨

### 재현율 (Recall) - "실제 양성 중에 얼마나 잡았나"

$$\text{Recall} = \frac{TP}{TP + FN}$$

- 실제 양성을 모델이 얼마나 잡아냈는지
- **암 진단, 사기 탐지** 등에서 중요: 놓치면 안 됨

```python
from sklearn.metrics import precision_score, recall_score

precision = precision_score(y_test, pred)
recall = recall_score(y_test, pred)
print(f'정밀도: {precision:.4f}, 재현율: {recall:.4f}')
```

### 정밀도-재현율 트레이드오프

임계값(threshold)을 낮추면 → 재현율↑, 정밀도↓  
임계값을 높이면 → 정밀도↑, 재현율↓

```python
from sklearn.preprocessing import Binarizer

# 확률 예측
pred_proba = clf.predict_proba(X_test)[:, 1]  # 양성 클래스 확률

# 임계값 0.4 적용
binarizer = Binarizer(threshold=0.4)
pred_04 = binarizer.fit_transform(pred_proba.reshape(-1, 1))
```

---

## 3. F1 Score

정밀도와 재현율의 조화 평균. 두 값을 균형 있게 반영합니다.

$$F1 = \frac{2 \times Precision \times Recall}{Precision + Recall}$$

```python
from sklearn.metrics import f1_score
f1 = f1_score(y_test, pred)
```

> 정밀도와 재현율이 모두 높을 때 F1도 높아집니다.

---

## 4. ROC 곡선 & AUC

### ROC 곡선 (Receiver Operating Characteristic)

임계값을 0 → 1로 변화시키며 FPR과 TPR의 관계를 나타낸 곡선

- **TPR (민감도)** = 재현율 = TP / (TP + FN)
- **FPR** = FP / (FP + TN)

```python
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# ROC 곡선 데이터
fprs, tprs, thresholds = roc_curve(y_test, pred_proba)

# ROC 곡선 그리기
plt.plot(fprs, tprs, label='ROC')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve')
plt.legend()
plt.show()

# AUC 점수
auc = roc_auc_score(y_test, pred_proba)
print(f'AUC: {auc:.4f}')
```

### AUC (Area Under Curve) 해석

| AUC 값 | 모델 성능 |
|--------|---------|
| 1.0 | 완벽한 모델 |
| 0.9 이상 | 매우 우수 |
| 0.7~0.9 | 양호 |
| 0.5 | 랜덤 예측 (쓸모없음) |
| 0.5 미만 | 랜덤보다 나쁨 |

---

## 5. 종합 지표 출력

```python
from sklearn.metrics import classification_report

print(classification_report(y_test, pred, target_names=['사망', '생존']))
```

출력 예:
```
              precision    recall  f1-score   support

          사망       0.88      0.95      0.92       104
          생존       0.92      0.80      0.85        70

    accuracy                           0.89       174
   macro avg       0.90      0.88      0.89       174
weighted avg       0.90      0.89      0.89       174
```

---

## 6. 실전: 피마 인디언 당뇨병 예측

불균형 데이터에서 다양한 지표를 비교하는 예시입니다.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 데이터 로드 및 분리
df = pd.read_csv('pima-indians-diabetes.csv')
X = df.drop('Outcome', axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 로지스틱 회귀 모델
lr_clf = LogisticRegression()
lr_clf.fit(X_train, y_train)
pred = lr_clf.predict(X_test)
pred_proba = lr_clf.predict_proba(X_test)[:, 1]

# 지표 출력
print(f'정확도: {accuracy_score(y_test, pred):.4f}')
print(f'정밀도: {precision_score(y_test, pred):.4f}')
print(f'재현율: {recall_score(y_test, pred):.4f}')
print(f'F1:    {f1_score(y_test, pred):.4f}')
print(f'AUC:   {roc_auc_score(y_test, pred_proba):.4f}')
```

---

## 평가 지표 선택 가이드

| 상황 | 권장 지표 | 이유 |
|------|---------|------|
| 암 진단, 사기 탐지 | **재현율(Recall)** | 놓치면 큰 피해 |
| 스팸 필터, 추천 시스템 | **정밀도(Precision)** | 잘못된 예측이 불편 유발 |
| 균형 있는 평가 필요 | **F1 Score** | 두 지표 조화 |
| 임계값 독립적 평가 | **AUC** | 전체 성능 요약 |
| 불균형 데이터 | 정확도 **지양**, F1/AUC 사용 | 정확도는 오해 소지 |

---

[← 2장: 머신러닝 입문](02_ml_intro.md) | [4장: 분류 & 앙상블 →](04_classification.md)
