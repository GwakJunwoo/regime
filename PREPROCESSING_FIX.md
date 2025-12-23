# 전처리 개선 및 문제 해결

## 문제 진단

### ✅ 확인된 사실
1. **전처리 자체는 정상**: Correlation = 0.24~0.35 (정상 범위)
2. **문제는 VAR 모델 실패**: 구조 벡터가 완전히 0
3. **원인**: Multicollinearity + 적은 자산 수 (11개)

### ❌ 이전 문제
- VAR 모델이 계속 실패하여 0 행렬 반환
- Bucket-wise demean이 신호를 약화시킴 (하지만 완전히 죽이지는 않음)

## 해결책

### 1️⃣ Demean 제거 (선택적으로 변경)

**변경 파일**: `data_loader.py`

```python
def preprocess_pipeline(self, 
                       vol_window: int = 20,
                       bucket_mapping: Optional[Dict[str, str]] = None,
                       use_demeaning: bool = False):  # 기본값 False
```

**효과**:
- Volatility normalization만 사용 → 신호 보존
- Demean은 해석 단계에서 필요시 적용
- Correlation 유지: 0.24 (정상)

### 2️⃣ Granger 방식을 기본값으로 변경

**변경 파일**: `causal_network.py`, `main_system.py`

```python
def build_causal_networks(self, method: str = 'granger', ...):
```

**이유**:
- VAR: 11개 자산 동시 추정 → Multicollinearity 문제
- Granger: Pairwise 검사 → 더 robust
- 11 assets = 110 pairs (관리 가능)

### 3️⃣ Dashboard 업데이트

**변경 파일**: `dashboard.py`

```python
system.build_causal_networks(method='granger', sample_size=1000)
```

## 테스트 방법

### 빠른 테스트 (200일)
```bash
python test_fixed_preprocessing.py
```

예상 출력:
```
✅ 구조 벡터가 정상적으로 생성되었습니다!
  - 벡터 표준편차: 0.XXXXXX (0이 아님)
  - 0이 아닌 엣지: XX개
```

### 전체 시스템 재실행
```bash
python main_system.py
```

### Dashboard 재실행
```bash
streamlit run dashboard.py
```
**중요**: 브라우저에서 Ctrl+F5로 캐시 클리어!

## 기대 효과

### Before (VAR + Demean)
- 구조 벡터: 전부 0
- 네트워크: 0 행렬
- 분석 불가능

### After (Granger + No Demean)
- 구조 벡터: 정상 분포
- 네트워크: 유의미한 연결 생성
- 시장 구조 분석 가능

## 주의사항

### Granger 방식의 단점
- 계산 시간: O(N²) - 자산 수가 많으면 느림
- 현재 11 assets: 괜찮음
- 50+ assets: VAR 대안 필요 (PCA 등)

### 권장 설정
- **소규모 (< 20 assets)**: `method='granger'` ✅
- **대규모 (20-50 assets)**: `method='var'` + dimension reduction
- **초대규모 (50+ assets)**: Factor model 또는 Lasso VAR

## 다음 단계

1. ✅ 테스트 완료 확인
2. ✅ Dashboard 재실행
3. ✅ 구조 벡터 0이 아닌지 확인
4. ✅ 구조 이동 분석 탭에서 정상 패턴 확인

---

**핵심 교훈**:
> 문제는 아이디어가 아니라 구현이었다.  
> Demean은 해석 단계의 도구이지, 전처리 단계의 도구가 아니다.
