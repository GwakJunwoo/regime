"""
구조 벡터 해석 가이드 및 테스트

피드백 반영:
- 구조 벡터는 가격이 아니라 시장 상태의 latent representation
- 절대값이 아니라 "시간에 따른 변화"와 "상대적 위치"가 중요
- 구조 이동량으로 시장 불안정도 측정
"""

import pandas as pd
import numpy as np
from main_system import MultiAssetCausalSystem

print("="*80)
print("구조 벡터 개선 시스템 테스트")
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

# 네트워크 구축 (샘플: 500일)
print("\n최근 500일 데이터로 분석 시작...")
system.build_causal_networks(method='var', sample_size=500)

# 구조 분석 (이제 구조 이동량과 불안정도가 자동 계산됨)
system.analyze_market_structures()

print("\n" + "="*80)
print("구조 이동량 분석 결과")
print("="*80)

# 구조 이동량 통계
movements = [s['structure_movement'] for s in system.structure_history if 'structure_movement' in s]
instabilities = [s['market_instability'] for s in system.structure_history if 'market_instability' in s]

print(f"\n구조 이동량 통계:")
print(f"  - 평균: {np.mean(movements):.6f}")
print(f"  - 표준편차: {np.std(movements):.6f}")
print(f"  - 최소: {np.min(movements):.6f}")
print(f"  - 최대: {np.max(movements):.6f}")
print(f"  - 95 백분위: {np.percentile(movements, 95):.6f}")

print(f"\n시장 불안정도 통계 (20일 이동평균):")
print(f"  - 평균: {np.mean(instabilities):.6f}")
print(f"  - 표준편차: {np.std(instabilities):.6f}")
print(f"  - 최대: {np.max(instabilities):.6f}")

# 상위 10개 불안정 시점 찾기
movement_data = []
for s in system.structure_history:
    if 'structure_movement' in s:
        movement_data.append({
            'date': s['date'],
            'movement': s['structure_movement'],
            'instability': s.get('market_instability', 0)
        })

movement_df = pd.DataFrame(movement_data).sort_values('movement', ascending=False)

print("\n" + "="*80)
print("구조 변화가 가장 큰 10개 시점 (레짐 전환 후보)")
print("="*80)
print(movement_df.head(10).to_string())

# 자산 기여도 분석 (가장 큰 변화 시점)
from conditional_analysis import AssetContributionAnalyzer

contrib_analyzer = AssetContributionAnalyzer(system.bucket_mapping)

max_movement_date = movement_df.iloc[0]['date']
max_idx = None

for i, s in enumerate(system.structure_history):
    if s['date'] == max_movement_date:
        max_idx = i
        break

if max_idx is not None and max_idx > 0:
    print("\n" + "="*80)
    print(f"최대 변화 시점 ({max_movement_date.strftime('%Y-%m-%d')}) 자산 기여도 분석")
    print("="*80)
    
    # 인과 행렬 찾기
    current_causality = None
    previous_causality = None
    
    for net in system.network_history:
        if net['date'] == max_movement_date:
            current_causality = net['network']
        if net['date'] == system.structure_history[max_idx-1]['date']:
            previous_causality = net['network']
    
    if current_causality is not None and previous_causality is not None:
        contrib = contrib_analyzer.decompose_structure_movement(
            current_causality,
            previous_causality,
            system.structure_history[max_idx]['structure_vector'],
            system.structure_history[max_idx-1]['structure_vector']
        )
        
        print(f"\n총 구조 이동량: {contrib['total_structure_movement']:.6f}")
        print(f"총 인과관계 변화량: {contrib['total_causality_change']:.4f}")
        
        print("\n상위 5개 기여 자산:")
        for i, c in enumerate(contrib['top_contributors'][:5], 1):
            print(f"  {i}. {c['asset']:30s} [{c['bucket']:15s}] - {c['total_change']:.4f}")
        
        print("\n버킷별 기여도:")
        for bucket, value in sorted(contrib['bucket_contributions'].items(), 
                                    key=lambda x: x[1], reverse=True):
            print(f"  {bucket:20s}: {value:.4f}")

print("\n" + "="*80)
print("✅ 개선 완료!")
print("="*80)
print("""
이제 시스템은:
1. ✅ 구조 이동량 (||structure_t - structure_{t-1}||) 계산
2. ✅ 시장 불안정도 (20일 롤링 평균) 계산
3. ✅ 자산별 기여도 분해 제공
4. ✅ 버킷별 기여도 집계

다음 단계:
- streamlit run dashboard.py 실행
- "구조 이동 분석" 탭에서 확인
""")
