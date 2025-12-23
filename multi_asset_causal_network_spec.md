# Multi-Asset Causal Network System
## (Price-only, Interpretable Market Insight Engine)

---

## 1. 개발 목적 (Purpose)

본 시스템의 목적은 **가격 데이터만을 사용하여**
글로벌 멀티에셋 시장의 현재 구조를 정확히 인지하고,
특정 자산의 가격 변화가 **다른 자산으로 어떻게 전이될 가능성이 있는지**를
인과적·해석 가능한 형태로 제공하는 것이다.

본 시스템은:
- 가격 예측 모델이 아니며
- 레짐을 사전에 정의하지 않고
- 매크로·펀더멘털 데이터를 사용하지 않는다.

대신,
**가격으로부터 자산 간 전이 구조를 추출하고
이를 기반으로 조건부 인사이트와 과거 유사 국면을 제공하는 시스템**을 목표로 한다.

---

## 2. 개발 목표 (Objectives)

### 2.1 핵심 목표
1. 멀티에셋 관점에서 **현재 글로벌 시장 구조를 인지**
2. 특정 자산 변화 시 **전이 가능성이 높은 자산을 식별**
3. 가격 수준이 아닌 **구조가 유사했던 과거 국면을 탐색**
4. 딜러 관점에서 **설명 가능한 결과물 제공**

### 2.2 명시적 비목표
- ❌ 단기 가격 예측
- ❌ 수익률 극대화 모델
- ❌ 블랙박스 머신러닝
- ❌ 매크로 팩터 추정

---

## 3. 데이터 및 입력 (Inputs)

### 3.1 데이터
- Daily close price
- 글로벌 멀티에셋 대표 자산

### 3.2 자산 버킷 정의 (사전 고정)

자산은 **반응 메커니즘 기준(shock-consistent)**으로 분류한다.

예시:
- Risk Assets: Global Equity, Credit
- Rates: Gov Bonds (UST, Bund 등)
- Safe Haven: Gold, JPY, CHF
- FX: USD index, Major FX

구조적으로 반대 반응을 보이는 자산은 같은 버킷에 포함하지 않는다.

---

## 4. 전처리 (Preprocessing)

### 4.1 수익률 계산
r(i,t) = log(P(i,t)) − log(P(i,t−1))

### 4.2 변동성 정규화
z(i,t) = r(i,t) / σ(i,t)
σ: rolling volatility (예: 20일)

목적: Δ / Vol 정보 중복 제거

### 4.3 Bucket-wise Cross-sectional Demeaning
ẑ(i,t) = z(i,t) − mean(z(j,t)), j ∈ same bucket

- 버킷 내부 공통 shock 제거
- 상대적 반응만 유지
- 팩터 모델 없이 구조적 요인 통제

---

## 5. 인과 네트워크 모델링 (Causal Modeling)

### 5.1 모델
- VAR 기반 Granger Causality
- Lag: 1~3일
- Rolling window: 60~120일

### 5.2 출력
시점별 Directed Weighted Network
W(t) = { w(i→j,t) }

### 5.3 해석
w(i→j,t) > 0 이면
자산 i의 변화가 이후 자산 j에 통계적으로 유의한 영향을 가짐

---

## 6. 시장 구조 요약 (Structure Encoding)

### 6.1 Bucket-to-Bucket 영향도
I(B1→B2,t) = Σ w(i→j,t), i∈B1, j∈B2

### 6.2 Market Structure Vector
s(t) = [
  Risk→Risk,
  Risk→Safe,
  Rates→Risk,
  Rates→FX,
  Network Density,
  Source Concentration
]

이 벡터가 시장 상태의 정의가 된다.

---

## 7. 현재 시장 인지 (Market Awareness)

시스템은 매일 다음 정보를 제공한다.
- 자산/버킷 간 영향도 구조
- 주요 Source 자산
- Risk → Safe 전이 강도

---

## 8. 조건부 전이 분석 (Conditional Insight)

현재 구조 하에서
A 자산이 움직일 경우
다음으로 반응할 가능성이 높은 자산을
방향성과 상대 강도로 제시한다.

---

## 9. 과거 유사 국면 탐색 (Historical Analogue)

### 9.1 유사도 기준
- Market Structure Vector 기반
- Cosine Similarity 사용

### 9.2 결과
- 구조적으로 유사한 과거 시점 Top-K
- 당시 이후 자산 반응 패턴 제공

---

## 10. 최종 산출물 (Outputs)

### 10.1 Daily Summary
1. Current Market Structure
2. Conditional Transmission Insight
3. Historical Analogues

---

## 11. 시스템 철학 요약

- 공통 요인을 제거하지 않고 분해하여 관리
- 가격만 사용하되 구조를 통해 해석
- 예측보다 인지와 설명을 우선

---

## 12. 한 문장 요약

본 시스템은 가격 데이터로부터 멀티에셋 간 인과 구조를 추출하여,
현재 시장 인지와 조건부 전이 시나리오, 그리고 과거 유사 국면을 제공하는
해석 가능한 시장 인사이트 엔진이다.
