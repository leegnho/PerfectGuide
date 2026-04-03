# 9장: 추천 시스템 (Recommendation Systems)

## 개요

사용자에게 관심 있을 아이템을 추천하는 시스템을 학습합니다.  
컨텐츠 기반 필터링, 협업 필터링, 행렬 분해 기법을 다룹니다.

---

## 추천 시스템 유형

```
추천 시스템
├── 컨텐츠 기반 필터링 (Content-Based Filtering)
│   └── 아이템의 속성을 기반으로 유사한 아이템 추천
│
└── 협업 필터링 (Collaborative Filtering)
    ├── 사용자 기반 (User-Based): 비슷한 취향의 사용자가 좋아한 것 추천
    ├── 아이템 기반 (Item-Based): 비슷한 아이템 추천
    └── 잠재 요인 (Latent Factor): 행렬 분해로 숨겨진 패턴 찾기
```

---

## 1. 컨텐츠 기반 필터링 (Content-Based Filtering)

### 개념

아이템의 속성(장르, 배우, 키워드 등)을 벡터로 표현하고,  
코사인 유사도로 비슷한 아이템을 찾아 추천합니다.

```python
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 영화 데이터 로드 (TMDB)
movies = pd.read_csv('tmdb_5000_movies.csv')

# 장르를 문자열로 변환
def get_genres_literal(row):
    genres = eval(row)  # JSON 형태 → 리스트
    return ' '.join([g['name'] for g in genres])

movies['genres_literal'] = movies['genres'].apply(get_genres_literal)

# 벡터화
cnt_vect = CountVectorizer(min_df=0, ngram_range=(1, 2))
genre_mat = cnt_vect.fit_transform(movies['genres_literal'])

# 모든 영화 간 유사도 계산
genre_sim = cosine_similarity(genre_mat, genre_mat)
# shape: (4803, 4803) - 각 영화 간 유사도
```

### 가중 평점 (Weighted Rating)

단순 평균 대신 투표 수를 반영한 가중 평점을 사용합니다.

```python
# IMDB 가중 평점 공식
# WR = (v/(v+m)) * R + (m/(m+v)) * C
# v: 해당 영화 투표 수, m: 최소 투표 수, R: 영화 평점, C: 전체 평균 평점

C = movies['vote_average'].mean()
m = movies['vote_count'].quantile(0.6)  # 상위 40% 투표 수

def weighted_vote(record, m=m, C=C):
    v = record['vote_count']
    R = record['vote_average']
    return (v / (v + m)) * R + (m / (m + v)) * C

movies['weighted_vote'] = movies.apply(weighted_vote, axis=1)
```

### 유사한 영화 추천

```python
def find_similar_movies(title, top_n=10):
    # 영화 인덱스 찾기
    movie_idx = movies[movies['title'] == title].index.values[0]

    # 유사도 점수 정렬
    sim_scores = list(enumerate(genre_sim[movie_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:]

    # 상위 10개 가져오기
    top_movies_idx = [i[0] for i in sim_scores[:top_n]]
    result = movies.iloc[top_movies_idx][['title', 'vote_average', 'weighted_vote']]
    return result.sort_values('weighted_vote', ascending=False)

find_similar_movies('The Dark Knight Rises')
```

---

## 2. 아이템 기반 협업 필터링 (Item-Based CF)

### 개념

"이 영화를 좋아한 사람들은 저 영화도 좋아했다"는 패턴을 기반으로 추천합니다.

```python
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# MovieLens 데이터
ratings = pd.read_csv('ratings.csv')  # userId, movieId, rating
movies = pd.read_csv('movies.csv')    # movieId, title

# 사용자-영화 평점 행렬
ratings_matrix = ratings.pivot_table(
    index='userId', columns='title', values='rating'
)
# shape: (671 사용자, 9066 영화)

# 아이템-아이템 유사도
# 행렬을 전치 → 영화가 행이 되도록
item_sim = cosine_similarity(ratings_matrix.T.fillna(0), ratings_matrix.T.fillna(0))
item_sim_df = pd.DataFrame(
    item_sim,
    index=ratings_matrix.columns,
    columns=ratings_matrix.columns
)
```

### 아이템 기반 예측

```python
def predict_rating(ratings_arr, item_sim_arr):
    """전체 사용자의 모든 아이템 평점 예측"""
    pred = np.zeros(ratings_arr.shape)
    for col in range(ratings_arr.shape[1]):
        # 각 아이템에 대해 유사한 아이템의 가중 평균으로 예측
        top_n_items = np.argsort(item_sim_arr[:, col])[::-1][:20]
        for row in range(ratings_arr.shape[0]):
            pred[row, col] = (
                item_sim_arr[top_n_items, col] * ratings_arr[row, top_n_items]
            ).sum() / (np.abs(item_sim_arr[top_n_items, col]).sum() + 1e-8)
    return pred
```

---

## 3. 잠재 요인 협업 필터링 (Latent Factor CF)

### 개념

사용자-아이템 평점 행렬을 **사용자 잠재 행렬(P)** × **아이템 잠재 행렬(Q)**로 분해합니다.

```
R (사용자 × 아이템) ≈ P (사용자 × K) × Q^T (K × 아이템)
```

K는 잠재 요인 수 (예: 장르 선호도, 시대 선호도 등의 숨겨진 특성)

### 경사하강법으로 구현

```python
import numpy as np

def matrix_factorization(R, K=50, learning_rate=0.01, reg=0.01, steps=200):
    """
    R: 사용자-아이템 평점 행렬 (NaN은 미평가)
    K: 잠재 요인 수
    """
    num_users, num_items = R.shape
    np.random.seed(1)

    # P: 사용자 잠재 행렬, Q: 아이템 잠재 행렬
    P = np.random.normal(scale=1/K, size=(num_users, K))
    Q = np.random.normal(scale=1/K, size=(num_items, K))

    prev_rmse = 10000
    for step in range(steps):
        for i in range(num_users):
            for j in range(num_items):
                if not np.isnan(R[i, j]):  # 평가한 항목만
                    error = R[i, j] - np.dot(P[i], Q[j].T)
                    # 그래디언트 업데이트
                    P[i] += learning_rate * (error * Q[j] - reg * P[i])
                    Q[j] += learning_rate * (error * P[i] - reg * Q[j])

        # RMSE 계산
        pred = np.dot(P, Q.T)
        mask = ~np.isnan(R)
        rmse = np.sqrt(np.mean((R[mask] - pred[mask]) ** 2))

        if step % 50 == 0:
            print(f'Step {step}: RMSE = {rmse:.4f}')

    return P, Q

# P, Q 행렬 학습
P, Q = matrix_factorization(ratings_matrix.values)

# 예측 평점 행렬
pred_ratings = np.dot(P, Q.T)
```

---

## 4. Surprise 라이브러리

SVD 기반 추천을 간편하게 구현합니다.

```python
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import cross_validate, train_test_split

# 데이터 로드
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# 학습/테스트 분리
trainset, testset = train_test_split(data, test_size=0.25)

# SVD 모델
algo = SVD(n_factors=50, random_state=0)
algo.fit(trainset)

# 평가
predictions = algo.test(testset)
print(f'RMSE: {accuracy.rmse(predictions):.4f}')

# 교차 검증
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
```

### 특정 사용자에게 추천

```python
def get_recommendations(user_id, algo, movies_df, ratings_df, n_rec=10):
    """특정 사용자가 아직 보지 않은 영화 중 높은 예측 평점 순으로 추천"""
    # 본 영화
    watched = ratings_df[ratings_df['userId'] == user_id]['movieId'].values

    # 안 본 영화에 대한 예측
    all_movies = movies_df['movieId'].values
    unwatched = [m for m in all_movies if m not in watched]

    predictions = [(m, algo.predict(user_id, m).est) for m in unwatched]
    predictions.sort(key=lambda x: x[1], reverse=True)

    top_movies = [p[0] for p in predictions[:n_rec]]
    return movies_df[movies_df['movieId'].isin(top_movies)][['title', 'genres']]

get_recommendations(user_id=1, algo=algo, movies_df=movies, ratings_df=ratings)
```

---

## 추천 시스템 비교

| 방법 | 필요 데이터 | 장점 | 단점 |
|------|----------|------|------|
| 컨텐츠 기반 | 아이템 속성 | 콜드 스타트 가능 | 다양성 부족 |
| 사용자 기반 CF | 사용자 평점 | 다양한 추천 | 확장성 문제 |
| 아이템 기반 CF | 사용자 평점 | 안정적, 빠름 | 희소성 문제 |
| 잠재 요인 CF | 사용자 평점 | 정확도 높음 | 해석 어려움 |

---

## 콜드 스타트 문제

새로운 사용자나 아이템이 평점 데이터가 없을 때 추천이 어려운 문제.

**해결 방법**:
- 신규 사용자: 인기 아이템 추천 + 온보딩에서 선호 수집
- 신규 아이템: 컨텐츠 기반 필터링으로 보완
- 하이브리드 방식: 두 방법을 결합

---

## 학습 포인트

| 개념 | 핵심 내용 |
|------|----------|
| 코사인 유사도 | 벡터 방향의 유사성 (크기 무관) |
| 가중 평점 | 투표 수 적은 영화의 과대 평가 방지 |
| 협업 필터링 | 비슷한 사용자/아이템 패턴 활용 |
| 행렬 분해 | 숨겨진 잠재 요인으로 평점 예측 |
| SVD | 추천 시스템에서 가장 많이 사용되는 기법 |

---

[← 8장: 텍스트 분석](08_nlp.md) | [Visual: 데이터 시각화 →](10_visualization.md)
