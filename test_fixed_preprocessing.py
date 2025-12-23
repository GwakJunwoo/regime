"""
개선된 전처리로 빠른 테스트
"""
import pandas as pd
import numpy as np
from main_system import MultiAssetCausalSystem

print("="*80)
print("개선된 시스템 테스트 (Demean 제거 + Granger 방식)")
print("="*80)

# 시스템 초기화
system = MultiAssetCausalSystem(
    csv_path="가격 데이터.csv",
    vol_window=20,
    causality_window=120,
    max_lag=3
)

# 데이터 로드 (demean=False)
system.load_and_preprocess()

print(f"\n전처리된 데이터 상태:")
print(f"  - Shape: {system.processed_data.shape}")
print(f"  - Mean: {system.processed_data.mean().mean():.6f}")
print(f"  - Std: {system.processed_data.std().mean():.6f}")

# 상관관계 확인
corr_matrix = system.processed_data.dropna().corr()
print(f"  - Correlation (평균): {corr_matrix.abs().mean().mean():.4f}")
print(f"  - Correlation (최소~최대): {corr_matrix.abs().min().min():.4f} ~ {corr_matrix.abs().max().max():.4f}")

# 네트워크 구축 (최근 200일, Granger 방식)
print(f"\n네트워크 구축 시작 (Granger 방식, 200일)...")
system.build_causal_networks(method='granger', sample_size=200)

print(f"\n네트워크 생성 결과:")
print(f"  - 총 시점 수: {len(system.network_history)}")

# 샘플 네트워크 확인
if len(system.network_history) > 0:
    sample_net = system.network_history[-1]['network']
    print(f"\n마지막 시점 네트워크:")
    print(f"  - 날짜: {system.network_history[-1]['date']}")
    print(f"  - 네트워크 크기: {sample_net.shape}")
    n_edges = (sample_net > 0).sum().sum()
    print(f"  - 0이 아닌 엣지: {n_edges}")
    if n_edges > 0:
        print(f"  - 평균 weight: {sample_net.values[sample_net.values > 0].mean():.6f}")
        print(f"  - 최대 weight: {sample_net.max().max():.6f}")
    else:
        print(f"  - ⚠️ 엣지가 없습니다 (모두 0)")
    
    # 샘플 네트워크 출력
    print(f"\n네트워크 샘플 (상위 5개 연결):")
    edges = []
    for source in sample_net.index:
        for target in sample_net.columns:
            weight = sample_net.loc[source, target]
            if weight > 0:
                edges.append((source, target, weight))
    
    edges.sort(key=lambda x: x[2], reverse=True)
    for source, target, weight in edges[:5]:
        print(f"  {source} → {target}: {weight:.6f}")

# 구조 분석
print(f"\n구조 분석 시작...")
system.analyze_market_structures()

# 구조 벡터 확인
if len(system.structure_history) > 0:
    last_struct = system.structure_history[-1]
    vec = last_struct['structure_vector']
    
    print(f"\n마지막 시점 구조 벡터:")
    print(f"  - 날짜: {last_struct['date']}")
    print(f"  - 벡터 차원: {len(vec)}")
    print(f"  - 벡터 평균: {vec.mean():.6f}")
    print(f"  - 벡터 표준편차: {vec.std():.6f}")
    print(f"  - 벡터 노름: {np.linalg.norm(vec):.6f}")
    print(f"  - 0인 값: {(vec == 0).sum()}/{len(vec)}")
    
    if vec.std() < 1e-6:
        print(f"\n❌ 구조 벡터가 여전히 0입니다!")
    else:
        print(f"\n✅ 구조 벡터가 정상적으로 생성되었습니다!")

print("\n" + "="*80)
