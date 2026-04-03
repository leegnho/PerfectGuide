# 8장: 텍스트 분석 & NLP

## 개요

자연어 처리(NLP)의 핵심 기법을 학습합니다.  
텍스트 전처리, 피처 추출, 문서 분류, 감성 분석, 토픽 모델링, 문서 유사도를 다룹니다.

---

## 텍스트 분석 파이프라인

```
원본 텍스트
    ↓
전처리 (정제, 정규화, 토큰화, 불용어 제거, 어간/표제어 추출)
    ↓
피처 벡터화 (BoW, TF-IDF)
    ↓
모델 학습 (분류, 클러스터링, 감성 분석 등)
```

---

## 1. 텍스트 전처리

### 토큰화 (Tokenization)

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')

from nltk import word_tokenize, sent_tokenize

text = "The movie was amazing. I loved every moment of it."

# 문장 토큰화
sentences = sent_tokenize(text)
# ['The movie was amazing.', 'I loved every moment of it.']

# 단어 토큰화
words = word_tokenize(text)
# ['The', 'movie', 'was', 'amazing', '.', ...]
```

### 불용어 제거 (Stopwords)

```python
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
# {'the', 'a', 'is', 'it', 'was', ...}

filtered = [w for w in words if w.lower() not in stop_words]
```

### 어간 추출 vs 표제어 추출

```python
from nltk.stem import LancasterStemmer, WordNetLemmatizer

# 어간 추출 (Stemming): 빠르지만 부정확
stemmer = LancasterStemmer()
stemmer.stem('working')   # 'work'
stemmer.stem('eating')    # 'eat'

# 표제어 추출 (Lemmatization): 느리지만 정확
lemmatizer = WordNetLemmatizer()
lemmatizer.lemmatize('working', pos='v')  # 'work'
lemmatizer.lemmatize('better', pos='a')  # 'good'
```

### N-gram

```python
from nltk import ngrams

words = ['I', 'love', 'machine', 'learning']

bigrams = list(ngrams(words, 2))
# [('I', 'love'), ('love', 'machine'), ('machine', 'learning')]

trigrams = list(ngrams(words, 3))
```

---

## 2. 텍스트 피처 벡터화

### BOW (Bag of Words) - CountVectorizer

단어의 출현 횟수를 특성으로 사용합니다. 순서는 무시합니다.

```python
from sklearn.feature_extraction.text import CountVectorizer

docs = ['I love machine learning', 'machine learning is great', 'I love coding']

cnt_vect = CountVectorizer(
    max_features=5000,      # 최대 특성 수
    stop_words='english',   # 불용어 제거
    ngram_range=(1, 2),     # 유니그램 + 바이그램
    min_df=2                # 최소 2개 문서에 출현해야 포함
)
X = cnt_vect.fit_transform(docs)  # sparse matrix
print(cnt_vect.vocabulary_)       # 단어 → 인덱스 딕셔너리
```

### TF-IDF Vectorizer (권장)

빈도뿐 아니라 문서 전체에서의 중요도를 반영합니다.

$$TF\text{-}IDF(t, d) = TF(t, d) \times \log\frac{N}{DF(t)}$$

- **TF**: 문서 d에서 단어 t의 빈도
- **IDF**: 단어 t가 드물수록 높은 가중치

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.95  # 95% 이상 문서에 나타나면 제외 (너무 흔한 단어)
)
X = tfidf.fit_transform(docs)
```

---

## 3. 텍스트 분류 (20 뉴스그룹)

```python
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 데이터 로드
news_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
news_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))

# TF-IDF 벡터화 (훈련 데이터로만 fit!)
tfidf = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
X_train = tfidf.fit_transform(news_train.data)
X_test = tfidf.transform(news_test.data)  # transform만!

# 분류 모델
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, news_train.target)
pred = lr.predict(X_test)
print(f'정확도: {accuracy_score(news_test.target, pred):.4f}')
```

---

## 4. 감성 분석 (IMDB 영화 리뷰)

```python
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# 텍스트 정제 함수
def clean_text(text):
    text = re.sub(r'<[^>]+>', '', text)    # HTML 태그 제거
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # 특수문자, 숫자 제거
    return text.lower().strip()

# 데이터 로드 및 전처리
df = pd.read_csv('labeledTrainData.tsv', sep='\t')
df['clean_review'] = df['review'].apply(clean_text)

# TF-IDF + Logistic Regression
tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=50000)
X = tfidf.fit_transform(df['clean_review'])
y = df['sentiment']

lr = LogisticRegression(C=1.0)
lr.fit(X, y)
```

---

## 5. 토픽 모델링 (LDA)

문서 컬렉션에서 숨겨진 주제(토픽)를 자동으로 찾아냅니다.

```python
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# BoW 행렬
cnt_vect = CountVectorizer(max_features=1000, stop_words='english')
X = cnt_vect.fit_transform(docs)

# LDA
lda = LatentDirichletAllocation(
    n_components=8,    # 토픽 수
    max_iter=50,
    random_state=0
)
lda.fit(X)

# 토픽별 상위 단어 출력
def print_top_words(model, feature_names, n_top_words=10):
    for topic_idx, topic in enumerate(model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-n_top_words-1:-1]]
        print(f'Topic {topic_idx}: {" ".join(top_words)}')

print_top_words(lda, cnt_vect.get_feature_names_out())
```

---

## 6. 문서 유사도 (Cosine Similarity)

```python
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

docs = ['I love machine learning', 'Machine learning is great', 'I love coding']

tfidf = TfidfVectorizer()
X = tfidf.fit_transform(docs)

# 모든 문서 쌍의 유사도
sim_matrix = cosine_similarity(X, X)
# [[1.0, 0.5, 0.3],
#  [0.5, 1.0, 0.1],
#  [0.3, 0.1, 1.0]]

# 특정 문서와 가장 유사한 문서 찾기
query_idx = 0
similarities = sim_matrix[query_idx]
similar_idx = similarities.argsort()[::-1][1]  # 자기 자신 제외
print(f'가장 유사한 문서: {docs[similar_idx]}')
```

---

## 7. 한국어 텍스트 처리

```python
# konlpy 설치 필요: pip install konlpy
from konlpy.tag import Okt

okt = Okt()

text = "머신러닝은 정말 재미있는 분야입니다"

# 형태소 분석
okt.morphs(text)
# ['머신', '러닝', '은', '정말', '재미있는', '분야', '입니다']

# 명사만 추출
okt.nouns(text)
# ['머신', '러닝', '분야']

# 품사 태깅
okt.pos(text)
# [('머신', 'Noun'), ('러닝', 'Noun'), ('은', 'Josa'), ...]
```

### 한국어 감성 분석 (네이버 영화 리뷰)

```python
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

okt = Okt()

def tokenize_korean(text):
    # 명사와 형용사만 추출
    pos_tagged = okt.pos(text)
    tokens = [word for word, pos in pos_tagged if pos in ['Noun', 'Adjective']]
    return ' '.join(tokens)

# 데이터 전처리
df = pd.read_csv('ratings_train.txt', sep='\t')
df['tokenized'] = df['document'].apply(tokenize_korean)

tfidf = TfidfVectorizer(min_df=3)
X = tfidf.fit_transform(df['tokenized'])

lr = LogisticRegression()
lr.fit(X, df['label'])
```

---

## 주요 개념 비교

| 개념 | 설명 | 특징 |
|------|------|------|
| Stemming | 어근 추출 (규칙 기반) | 빠름, 부정확 |
| Lemmatization | 사전 표제어 추출 | 느림, 정확 |
| BoW | 단어 출현 횟수 | 단순, 순서 무시 |
| TF-IDF | 빈도 × 역문서 빈도 | BoW보다 효과적 |
| LDA | 토픽 모델링 (비지도) | 주제 자동 탐색 |

---

## 학습 포인트

| 포인트 | 내용 |
|--------|------|
| fit은 훈련 데이터에만 | TF-IDF fit은 train만, test는 transform |
| ngram_range | (1,1)=유니그램, (1,2)=유니그램+바이그램 |
| sublinear_tf | TF에 로그 적용 → 성능 향상 |
| 한국어 | 영어와 달리 형태소 분석기(konlpy) 필요 |
| 희소 행렬 | TF-IDF 출력은 sparse matrix, toarray()로 변환 |

---

[← 7장: 클러스터링](07_clustering.md) | [9장: 추천 시스템 →](09_recommendation.md)
