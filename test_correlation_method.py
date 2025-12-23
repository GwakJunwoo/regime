"""
Correlation 방식 테스트 (빠르고 robust!)
"""
import pandas as pd
import numpy as np
from main_system import MultiAssetCausalSystem
import time

print("="*80)
print("Correlation 방식 테스트 (개장시간 차이 해결)")
print("="*80)

# 시스템 초기화
system = MultiAssetCausalSystem(
    csv_path="가격 데이터.csv",
    vol_window=20,
    causality_window=120,
    max_lag=3
)

# 데이터 로드
system.load_and_preprocess()

print(f"\n전처리 완료:")
print(f"  - Correlation 평균: {system.processed_data.dropna().corr().abs().mean().mean():.4f}")

# 네트워크 구축 (Correlation 방식)
print(f"\n네트워크 구축 시작 (Correlation, 500일)...")
start = time.time()

system.build_causal_networks(method='correlation', sample_size=500)

elapsed = time.time() - start
print(f"\n⏱️ 소요시간: {elapsed:.1f}초")

# 결과 확인
if len(system.network_history) > 0:
    last_net = system.network_history[-1]['network']
    
    print(f"\n네트워크 결과:")
    print(f"  - 총 시점: {len(system.network_history)}")
    print(f"  - 마지막 날짜: {system.network_history[-1]['date']}")
    print(f"  - 0이 아닌 엣지: {(last_net > 0).sum().sum()}")
    
    if (last_net > 0).sum().sum() > 0:
        print(f"  - 평균 weight: {last_net.values[last_net.values > 0].mean():.4f}")
        print(f"  - 최대 weight: {last_net.max().max():.4f}")
        
        # 상위 5개 연결
        print(f"\n상위 5개 연결:")
        edges = []
        for source in last_net.index:
            for target in last_net.columns:
                weight = last_net.loc[source, target]
                if weight > 0:
                    edges.append((source, target, weight))
        
        edges.sort(key=lambda x: x[2], reverse=True)
        for i, (source, target, weight) in enumerate(edges[:5], 1):
            print(f"  {i}. {source} → {target}: {weight:.4f}")

# 구조 분석
print(f"\n구조 분석 시작...")
system.analyze_market_structures()

# 구조 벡터 확인
if len(system.structure_history) > 0:
    last_struct = system.structure_history[-1]
    vec = last_struct['structure_vector']
    
    print(f"\n구조 벡터 결과:")
    print(f"  - 벡터 표준편차: {vec.std():.6f}")
    print(f"  - 벡터 노름: {np.linalg.norm(vec):.6f}")
    print(f"  - 0인 값: {(vec == 0).sum()}/{len(vec)}")
    
    if vec.std() > 1e-6:
        print(f"\n✅ SUCCESS! 구조 벡터가 정상적으로 생성되었습니다!")
        print(f"   Correlation 방식이 잘 작동합니다.")
    else:
        print(f"\n⚠️ WARNING: 구조 벡터가 여전히 0입니다.")

print("\n" + "="*80)
print("다음 단계: streamlit run dashboard.py")
print("="*80)
