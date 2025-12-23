# Multi-Asset Causal Network System - 구조 벡터 개선

## 주요 개선사항 (2025-12-23)

### 🎯 핵심 변경: 구조 벡터의 올바른 해석

구조 벡터는 **가격이나 수익률이 아니라 시장 상태의 latent representation**입니다.

#### ❌ 잘못된 해석
- "이 값이 크니까 risk-on"
- "이 차원이 주식 요인"
- "+니까 상승"
- 절대값의 의미 찾기

#### ✅ 올바른 해석
- **시간에 따른 변화량**: ||structure_t - structure_{t-1}||
- **과거와의 상대적 거리**: 유사 국면 탐색
- **자산 기여도 분해**: 구조 변화의 주요 동인 파악

---

## 새로 추가된 기능

### 1. 구조 이동량 계산 (Structure Movement)
**파일**: `market_structure.py`

```python
@staticmethod
def calculate_structure_movement(structure_t, structure_t_minus_1):
    """
    ||structure_t - structure_{t-1}|| 계산
    
    Returns:
        float: 유클리디안 거리 (구조 이동량)
    """
    return np.linalg.norm(structure_t - structure_t_minus_1)
```

**의미**: 
- 높을수록 → 시장 구조 급변 (불안정)
- 낮을수록 → 시장 구조 안정적

### 2. 시장 불안정도 (Market Instability)
**파일**: `market_structure.py`

```python
@staticmethod
def calculate_rolling_instability(structure_history, window=20):
    """
    구조 이동량의 20일 롤링 평균
    
    Returns:
        pd.Series: 시간별 불안정도
    """
```

**의미**:
- 구조 이동량을 smoothing하여 추세 파악
- VIX 급등, 금리 쇼크 등 이벤트와 동조

### 3. 자산 기여도 분해 (Asset Contribution Decomposition)
**파일**: `conditional_analysis.py`

```python
class AssetContributionAnalyzer:
    def decompose_structure_movement(self, causality_t, causality_t_minus_1, 
                                     structure_t, structure_t_minus_1):
        """
        구조 변화에 대한 자산별 기여도 분석
        
        Returns:
            Dict: {
                'total_structure_movement': float,
                'asset_contributions': {...},
                'bucket_contributions': {...},
                'top_contributors': [...]
            }
        """
```

**의미**:
- "이번 구조 변화의 주요 동인은 어떤 자산인가?"
- Source 변화 + Target 변화로 총 기여도 계산
- 버킷별로 집계하여 큰 그림 파악

### 4. 레짐 전환 동인 분석
**파일**: `conditional_analysis.py`

```python
def analyze_regime_transition_drivers(self, structure_history, causality_history,
                                      threshold_percentile=90):
    """
    구조 이동량 상위 10% 시점의 주요 동인 추출
    
    Returns:
        pd.DataFrame: 레짐 전환 시점 + 상위 3개 기여 자산
    """
```

**의미**:
- 레짐 전환 시점 자동 탐지
- 각 전환의 주요 동인 3개 식별

---

## Dashboard 개선

### 새로운 탭: "⚡ 구조 이동 분석"

#### 📊 포함 내용

1. **구조 이동량 시계열 차트**
   - 원본 구조 이동량 (회색 선)
   - 20일 이동평균 불안정도 (빨간 선)
   - 현재 선택 날짜 표시

2. **통계 요약**
   - 현재 이동량
   - 평균 이동량
   - 최대 이동량
   - 현재 백분위 (95 백분위 대비)

3. **자산 기여도 분석**
   - 상위 10개 기여 자산 테이블
   - Source/Target 변화량 분해
   - 버킷별 기여도 막대 차트

4. **해석 가이드**
   - 구조 벡터가 무엇인지 명확히 설명
   - 올바른 해석 방법 안내

---

## 실행 방법

### 1. 테스트 실행
```bash
python test_structure_movement.py
```

출력:
- 구조 이동량 통계
- 상위 10개 불안정 시점
- 최대 변화 시점의 자산 기여도

### 2. Dashboard 실행
```bash
streamlit run dashboard.py
```

또는
```bash
run_dashboard.bat
```

브라우저에서 http://localhost:8501 접속 후:
1. "⚡ 구조 이동 분석" 탭 클릭
2. 구조 이동량 시계열 확인
3. 특정 날짜 선택하여 기여도 분석

---

## 핵심 인사이트

### 정상적인 패턴 체크리스트

✅ **(1) 시간 연속성**
- 대부분 시점에서 구조 이동량 작음
- 특정 구간에서만 급변
- → 이게 보이면 모델 정상

✅ **(2) 이벤트와 동조**
- VIX 급등 구간에서 구조 이동량 증가
- 금리 쇼크 시점에 불안정도 상승
- → 시장 스트레스 감지 중

✅ **(3) 시점별 군집**
- 2020년 초 (코로나)
- 2022년 (금리 쇼크)
- 2023-2024 (안정기)
- → 유사 국면이 날짜 기준으로 묶임

---

## 다음 단계 제안

### 1. 구조 벡터 → 시장 언어 번역 레이어
- "Risk-On/Off Score" 계산
- "금리 민감도" 추출
- "안전자산 선호도" 측정

### 2. 자동 알림 시스템
- 구조 이동량 95 백분위 초과 시 알림
- 주요 동인 자산 실시간 모니터링

### 3. 포트폴리오 최적화 연계
- 현재 시장 구조 기반 리스크 조정
- 유사 국면 이후 수익률 분포 활용

---

## 변경 파일 목록

1. **market_structure.py**: 구조 이동량 계산 함수 추가
2. **conditional_analysis.py**: AssetContributionAnalyzer 클래스 추가
3. **main_system.py**: analyze_market_structures()에서 자동 계산
4. **dashboard.py**: 6번째 탭 "구조 이동 분석" 추가
5. **test_structure_movement.py**: 테스트 스크립트 생성
6. **STRUCTURE_IMPROVEMENT.md**: 본 문서

---

## 문의 및 피드백

구조 벡터 해석에 대한 추가 질문이나 개선 제안은 언제든지 환영합니다!
