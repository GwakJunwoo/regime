# Multi-Asset Causal Network System
## 가격 기반 해석 가능한 시장 인사이트 엔진

---

## 📋 개요

본 시스템은 **가격 데이터만을 사용하여** 글로벌 멀티에셋 시장의 현재 구조를 정확히 인지하고,
특정 자산의 가격 변화가 **다른 자산으로 어떻게 전이될 가능성이 있는지**를
인과적·해석 가능한 형태로 제공하는 시스템입니다.

### 핵심 특징

✅ **가격 전용**: 가격 데이터만 사용 (매크로/펀더멘털 데이터 불필요)  
✅ **인과 분석**: VAR 기반 Granger Causality로 자산 간 인과관계 추정  
✅ **해석 가능**: 블랙박스가 아닌 투명한 분석 과정  
✅ **구조 인지**: 현재 시장의 전이 구조를 실시간 파악  
✅ **과거 유사 국면**: 구조적으로 유사했던 과거 시점 탐색

---

## 🏗️ 시스템 구조

```
Multi-Asset Causal Network System
│
├── data_loader.py              # 데이터 로드 및 전처리
│   ├── CSV 파싱
│   ├── 수익률 계산 (로그 수익률)
│   ├── 변동성 정규화 (z-score)
│   └── Bucket-wise demeaning
│
├── causal_network.py           # 인과 네트워크 모델링
│   ├── VAR 모델 추정
│   ├── Granger Causality 테스트
│   └── Rolling window 네트워크 생성
│
├── market_structure.py         # 시장 구조 분석
│   ├── Bucket-to-Bucket 영향도
│   ├── Market Structure Vector 생성
│   └── 네트워크 메트릭 계산
│
├── conditional_analysis.py     # 조건부 분석 및 유사 국면 탐색
│   ├── 조건부 전이 분석
│   └── Historical Analogue Search
│
└── main_system.py              # 메인 시스템 통합
    └── 전체 파이프라인 실행
```

---

## 📊 데이터 구조

### 입력 데이터: `가격 데이터.csv`

- **자산**: 11개 멀티에셋
  - **Rates**: 3년국채, 10년국채, 2년 T-NOTE선물, 10년 T-NOTE선물
  - **Risk Assets**: 코스피200
  - **Safe Haven**: 금, 엔
  - **FX**: 미국달러, 유로, 위안
  - **Commodities**: WTI 원유

- **기간**: 2000-11-12 ~ 2025-12-23 (약 25년)
- **빈도**: Daily Close Price

---

## 🚀 사용 방법

### 1. 환경 준비

필요한 패키지가 이미 설치되어 있는지 확인:

```bash
pip install -r requirements.txt
```

주요 패키지:
- pandas >= 1.5.0
- numpy >= 1.21.0
- statsmodels >= 0.13.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.5.0
- seaborn >= 0.12.0

### 2. 전체 시스템 실행

```bash
python main_system.py
```

이 명령은 다음을 수행합니다:
1. 데이터 로드 및 전처리
2. Rolling window 인과 네트워크 구축
3. 시장 구조 분석
4. Daily Summary 생성
5. 결과 저장 (`./results/` 폴더)
6. 네트워크 시각화 (`./results/latest_network.png`)

### 3. 대화형 대시보드 실행 (권장)

**Windows:**
```bash
run_dashboard.bat
```

**또는 직접 실행:**
```bash
streamlit run dashboard.py
```

브라우저가 자동으로 열리며 `http://localhost:8501`에서 대시보드에 접속할 수 있습니다.

#### 대시보드 주요 기능:
- 📊 **네트워크 시각화**: 히트맵 및 인터랙티브 네트워크 그래프
- 📈 **시장 구조 분석**: Bucket 영향도, 주요 자산, 전이 경로
- 🔄 **조건부 전이**: 특정 자산 변화 시 영향 분석 및 복수 시나리오 비교
- 🔍 **유사 국면 탐색**: 과거 유사 시점 및 이후 자산 반응 패턴
- 📉 **시계열 분석**: 네트워크 메트릭 및 전이 강도 추이

### 4. 커스텀 분석 (Python 코드)

```python
from main_system import MultiAssetCausalSystem

# 시스템 초기화
system = MultiAssetCausalSystem(
    csv_path="가격 데이터.csv",
    vol_window=20,          # 변동성 계산 윈도우
    causality_window=120,   # 인과관계 분석 윈도우
    max_lag=3              # 최대 시차
)

# 전체 분석 실행
summary = system.run_full_analysis()

# 결과 저장
system.save_results("./results")

# 특정 날짜 분석
from datetime import datetime
summary = system.generate_daily_summary(
    date=pd.Timestamp('2025-12-23')
)
```

---

## 📈 출력물

### 1. Daily Summary (콘솔 출력)

```
================================================================================
DAILY SUMMARY: 2025-12-23
================================================================================

1. CURRENT MARKET STRUCTURE
   - Network Metrics (밀도, 집중도 등)
   - Key Source Assets (주요 영향력 자산)
   - Bucket-to-Bucket Influence (버킷 간 영향도)
   - Transmission Pathways (전이 경로)

2. CONDITIONAL TRANSMISSION INSIGHT
   - "If [자산] moves, then..." 분석
   - 영향받을 가능성 높은 자산 Top-5

3. HISTORICAL ANALOGUES
   - 구조적으로 유사한 과거 Top-5 시점
   - 해당 시점 이후 자산 반응 패턴
```

### 2. 저장 파일 (`./results/`)

- **structure_vectors.csv**: 시점별 Market Structure Vector
- **processed_data.csv**: 전처리된 수익률 데이터
- **network_history.csv**: 시점별 인과 네트워크 엣지
- **latest_network.png**: 최신 네트워크 히트맵

---

## 🔧 주요 파라미터

### DataLoader
- `vol_window` (default: 20): 변동성 정규화 윈도우
- `bucket_mapping`: 자산 → 버킷 매핑 딕셔너리

### CausalNetworkModel
- `max_lag` (default: 3): Granger Causality 최대 시차
- `significance_level` (default: 0.05): 유의수준

### Rolling Analysis
- `causality_window` (default: 120): 인과관계 분석 윈도우 크기
- `method` (default: 'var'): 'var' 또는 'granger'

---

## 📐 방법론

### 1. 전처리 (Preprocessing)

```
수익률 계산: r(i,t) = log(P(i,t)) - log(P(i,t-1))
변동성 정규화: z(i,t) = r(i,t) / σ(i,t)
Demeaning: ẑ(i,t) = z(i,t) - mean(z(j,t)), j ∈ same bucket
```

### 2. 인과 네트워크 (Causal Network)

- **VAR 모델**: 벡터 자기회귀 모델로 자산 간 동적 관계 추정
- **Granger Causality**: 통계적으로 유의한 인과관계 식별
- **Rolling Window**: 시점별 네트워크 변화 추적

### 3. 시장 구조 벡터 (Market Structure Vector)

```
s(t) = [
    Risk→Risk,
    Risk→Safe,
    Risk→Rates,
    Rates→Risk,
    Rates→Safe,
    Safe→Risk,
    FX→Risk,
    Commodities→Risk,
    Network Density,
    Source Concentration,
    Avg Influence
]
```

### 4. 유사 국면 탐색 (Historical Analogue)

- **유사도**: Cosine Similarity of Market Structure Vectors
- **필터링**: 최근 N일 제외 (default: 60일)
- **결과**: Top-K 유사 시점 및 이후 자산 반응

---

## 🎯 활용 사례

1. **리스크 관리**
   - 특정 자산 충격 시 포트폴리오 전체 영향 예측
   - Risk → Safe Haven 전이 강도 모니터링

2. **포지셔닝 전략**
   - 현재 시장 구조 하에서 전이 가능성 높은 자산 식별
   - 과거 유사 국면에서의 자산 반응 패턴 참고

3. **시장 국면 인지**
   - 네트워크 밀도/집중도 변화로 시장 스트레스 감지
   - Bucket-to-Bucket 영향도 변화 추적

---

## ⚠️ 주의사항

### 데이터 품질
- 일부 자산(유로, 엔, 위안 등)은 초기 데이터 부족
- 결측치 처리: 전처리 과정에서 자동 제거

### VAR 모델 제약
- 다중공선성 발생 시 일부 시점에서 모델 실패 가능
- 실패 시 zero matrix 반환 (시스템은 계속 진행)

### 해석 주의
- 인과관계 ≠ 인과성 (Granger Causality는 예측력 기반)
- 과거 패턴이 미래를 보장하지 않음
- 구조 변화 시 유사 국면의 유효성 감소

---

## 📚 이론적 배경

### Granger Causality
> Granger, C. W. J. (1969). "Investigating Causal Relations by Econometric Models and Cross-spectral Methods"

자산 X의 과거 정보가 자산 Y의 예측에 통계적으로 유의한 개선을 가져오면,
"X가 Y를 Granger-cause 한다"고 정의

### VAR (Vector Autoregression)
> Sims, C. A. (1980). "Macroeconomics and Reality"

여러 시계열 변수를 동시에 모델링하여 변수 간 동적 상호작용 추정

---

## 🔄 시스템 철학

1. **공통 요인 관리**: 제거가 아닌 분해를 통한 관리 (bucket-wise demeaning)
2. **가격 우선**: 가격만 사용하되 구조를 통해 해석
3. **인지 중심**: 예측보다 현재 시장 구조 인지를 우선
4. **투명성**: 모든 단계가 해석 가능하고 설명 가능

---

## 📝 한 문장 요약

> 본 시스템은 가격 데이터로부터 멀티에셋 간 인과 구조를 추출하여,  
> 현재 시장 인지와 조건부 전이 시나리오, 그리고 과거 유사 국면을 제공하는  
> **해석 가능한 시장 인사이트 엔진**이다.

---

## 📞 문의

시스템 관련 문의나 개선 제안은 Issues를 통해 남겨주세요.

---

**Last Updated**: 2025-12-23  
**Version**: 1.0.0  
**License**: MIT
