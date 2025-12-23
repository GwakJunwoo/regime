"""
실제 구조 벡터 상태 확인
"""
import pandas as pd
import numpy as np

print("="*80)
print("구조 벡터 실제 상태 확인")
print("="*80)

# 저장된 구조 벡터 로드
structure_df = pd.read_csv("results/structure_vectors.csv")

print(f"\n구조 벡터 파일 크기: {structure_df.shape}")
print(f"날짜 범위: {structure_df['date'].min()} ~ {structure_df['date'].max()}")

# 벡터 컬럼만 추출 (v0, v1, v2, ...)
vector_cols = [col for col in structure_df.columns if col.startswith('v')]
print(f"\n벡터 차원: {len(vector_cols)}")

# 통계
vector_data = structure_df[vector_cols]
print(f"\n벡터 통계:")
print(f"  - 평균: {vector_data.mean().mean():.6f}")
print(f"  - 표준편차: {vector_data.std().mean():.6f}")
print(f"  - 최소값: {vector_data.min().min():.6f}")
print(f"  - 최대값: {vector_data.max().max():.6f}")
print(f"  - 0인 값 비율: {(vector_data == 0).sum().sum() / vector_data.size * 100:.2f}%")

# 샘플 확인
print(f"\n최근 5개 구조 벡터 샘플:")
print(structure_df.tail(5))

# 구조 이동량 계산
movements = []
for i in range(1, len(structure_df)):
    prev_vec = structure_df.iloc[i-1][vector_cols].values
    curr_vec = structure_df.iloc[i][vector_cols].values
    movement = np.linalg.norm(curr_vec - prev_vec)
    movements.append(movement)

movements = np.array(movements)
print(f"\n구조 이동량 통계:")
print(f"  - 평균: {movements.mean():.6f}")
print(f"  - 표준편차: {movements.std():.6f}")
print(f"  - 최소: {movements.min():.6f}")
print(f"  - 최대: {movements.max():.6f}")

# 진단
print("\n" + "="*80)
print("진단 결과:")
print("="*80)

if vector_data.std().mean() < 1e-6:
    print("❌ CRITICAL: 구조 벡터가 거의 0입니다!")
    print("   → VAR 모델이 실패했거나 네트워크가 생성되지 않았습니다.")
elif movements.std() < 1e-6:
    print("❌ WARNING: 구조 벡터가 변하지 않습니다!")
    print("   → 모든 시점에서 동일한 네트워크가 생성되었습니다.")
else:
    print("✅ GOOD: 구조 벡터가 정상적으로 생성되었습니다.")
    print(f"   → 벡터 표준편차: {vector_data.std().mean():.6f}")
    print(f"   → 이동량 평균: {movements.mean():.6f}")
