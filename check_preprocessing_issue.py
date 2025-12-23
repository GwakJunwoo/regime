"""
전처리 문제 진단
"""
import pandas as pd
import numpy as np
from data_loader import DataLoader, AssetBucket

print("="*80)
print("전처리 문제 진단")
print("="*80)

loader = DataLoader("가격 데이터.csv")
prices = loader.load_csv()
bucket_mapping = AssetBucket.get_bucket_mapping()
bucket_mapping = {k: v for k, v in bucket_mapping.items() if k in prices.columns}

# 1. Raw returns
returns = loader.calculate_returns(prices)
print("\n1️⃣ Raw Returns 상태:")
print(f"   - Mean: {returns.mean().mean():.6f}")
print(f"   - Std: {returns.std().mean():.6f}")
print(f"   - Correlation (평균): {returns.corr().abs().mean().mean():.4f}")
print(f"   - Matrix rank: {np.linalg.matrix_rank(returns.corr())}/{len(returns.columns)}")

# 2. After volatility normalization
vol_normalized = loader.calculate_volatility_normalized_returns(returns, window=20)
print("\n2️⃣ Volatility Normalized 상태:")
print(f"   - Mean: {vol_normalized.mean().mean():.6f}")
print(f"   - Std: {vol_normalized.std().mean():.6f}")
print(f"   - Correlation (평균): {vol_normalized.corr().abs().mean().mean():.4f}")
print(f"   - Matrix rank: {np.linalg.matrix_rank(vol_normalized.corr())}/{len(vol_normalized.columns)}")

# 3. After bucket-wise demean (현재 파이프라인)
demeaned = loader.bucket_wise_demean(vol_normalized, bucket_mapping)
print("\n3️⃣ Bucket-wise Demean 후 (현재 파이프라인):")
print(f"   - Mean: {demeaned.mean().mean():.6f}")
print(f"   - Std: {demeaned.std().mean():.6f}")
print(f"   - Correlation (평균): {demeaned.corr().abs().mean().mean():.4f}")
print(f"   - Matrix rank: {np.linalg.matrix_rank(demeaned.corr())}/{len(demeaned.columns)}")

print("\n" + "="*80)
print("진단 결과:")
print("="*80)

corr_mean = demeaned.corr().abs().mean().mean()
if corr_mean < 0.05:
    print("❌ CRITICAL: Correlation 거의 0 → 정보 소거됨!")
    print("   → Demean이 신호를 죽였습니다.")
elif corr_mean < 0.2:
    print("⚠️ WARNING: Correlation 낮음 → 신호 약함")
elif corr_mean < 0.6:
    print("✅ GOOD: Correlation 정상 범위")
else:
    print("⚠️ WARNING: Correlation 너무 높음 → 위기 국면?")

print(f"\n현재 Correlation 평균: {corr_mean:.4f}")
print("\n권장사항:")
print("→ Demean 제거하고 Volatility Normalization만 사용")
print("→ 또는 Demean을 해석 단계에서 사용")
