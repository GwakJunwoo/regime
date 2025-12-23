"""
Multi-Asset Causal Network System
Quick Start Example
"""

from main_system import MultiAssetCausalSystem
import pandas as pd

# ==============================================================================
# 예제 1: 기본 사용법 - 전체 분석 실행
# ==============================================================================

print("\n" + "="*80)
print("예제 1: 전체 분석 파이프라인 실행")
print("="*80 + "\n")

# 시스템 초기화
system = MultiAssetCausalSystem(
    csv_path="가격 데이터.csv",
    vol_window=20,
    causality_window=120,
    max_lag=3
)

# 전체 분석 실행 (최근 500일만 테스트)
summary = system.run_full_analysis(sample_size=500)

# 결과 저장
system.save_results("./results_example")

print("\n✓ 분석 완료! 결과는 ./results_example/ 폴더에 저장되었습니다.")


# ==============================================================================
# 예제 2: 특정 날짜 분석
# ==============================================================================

print("\n" + "="*80)
print("예제 2: 특정 날짜 분석")
print("="*80 + "\n")

# 특정 날짜 요약 생성
specific_date = pd.Timestamp('2025-12-20')
if specific_date in [item['date'] for item in system.network_history]:
    summary = system.generate_daily_summary(date=specific_date)
else:
    print(f"날짜 {specific_date}는 분석 범위에 없습니다.")


# ==============================================================================
# 예제 3: 조건부 전이 분석
# ==============================================================================

print("\n" + "="*80)
print("예제 3: 특정 자산의 조건부 영향 분석")
print("="*80 + "\n")

# 최근 네트워크 가져오기
latest_network_info = system.causal_model.get_latest_network()

if latest_network_info:
    network = latest_network_info['network']
    date = latest_network_info['date']
    
    print(f"분석 시점: {date.strftime('%Y-%m-%d')}\n")
    
    # 코스피200이 움직일 경우 영향
    test_asset = '코스피200 연결'
    
    if test_asset in network.index:
        impacts = system.transmission_analyzer.analyze_conditional_impact(
            network, 
            test_asset, 
            top_k=5
        )
        
        print(f"'{test_asset}' 변화 시 영향받을 자산:\n")
        for i, impact in enumerate(impacts, 1):
            print(f"  {i}. {impact['asset']:30s} [{impact['bucket']:15s}]")
            print(f"     - Weight: {impact['weight']:.4f}")
            print(f"     - Relative Strength: {impact['relative_strength']*100:.1f}%\n")


# ==============================================================================
# 예제 4: 네트워크 시각화
# ==============================================================================

print("\n" + "="*80)
print("예제 4: 네트워크 시각화")
print("="*80 + "\n")

# 최신 네트워크 시각화
system.visualize_network(save_path="./results_example/network_viz.png")
print("✓ 네트워크 시각화 저장: ./results_example/network_viz.png")


# ==============================================================================
# 예제 5: 과거 유사 국면 분석
# ==============================================================================

print("\n" + "="*80)
print("예제 5: 과거 유사 국면 탐색 및 비교")
print("="*80 + "\n")

if len(system.structure_history) > 0:
    # 현재 구조
    current_structure = system.structure_history[-1]
    current_vector = current_structure['structure_vector']
    
    # 유사 기간 찾기
    similar_periods = system.analogue_search.find_similar_periods(
        current_vector,
        top_k=10,
        exclude_recent_days=60
    )
    
    if len(similar_periods) > 0:
        print("현재와 유사한 과거 시점 Top 10:\n")
        for period in similar_periods:
            print(f"  {period['rank']:2d}. {period['date'].strftime('%Y-%m-%d')} "
                  f"- Similarity: {period['similarity']:.4f}")
        
        # 유사 기간 이후 자산 반응 비교
        print("\n이후 20일간 평균 자산 수익률:")
        comparison = system.analogue_search.compare_analogues(
            similar_periods,
            system.prices,
            forward_days=20
        )
        
        if len(comparison) > 0:
            asset_cols = [col for col in comparison.columns 
                         if col not in ['date', 'similarity', 'rank']]
            avg_returns = comparison[asset_cols].mean().sort_values(ascending=False)
            
            print("\n")
            for asset, ret in avg_returns.items():
                if asset in system.bucket_mapping:
                    bucket = system.bucket_mapping[asset]
                    print(f"  {asset:30s} [{bucket:15s}] - {ret:+6.2f}%")


# ==============================================================================
# 예제 6: 시장 구조 시계열 분석
# ==============================================================================

print("\n" + "="*80)
print("예제 6: 시장 구조 메트릭 시계열")
print("="*80 + "\n")

# 네트워크 밀도 추이
if len(system.structure_history) > 0:
    densities = [s['network_metrics']['density'] for s in system.structure_history]
    dates = [s['date'] for s in system.structure_history]
    
    # DataFrame으로 정리
    metrics_df = pd.DataFrame({
        'date': dates,
        'density': densities,
        'concentration': [s['network_metrics']['source_concentration'] 
                         for s in system.structure_history],
        'avg_influence': [s['network_metrics']['avg_influence'] 
                         for s in system.structure_history]
    })
    
    metrics_df.set_index('date', inplace=True)
    
    # 통계 요약
    print("시장 구조 메트릭 통계:\n")
    print(metrics_df.describe())
    
    # CSV 저장
    metrics_df.to_csv("./results_example/market_metrics_timeseries.csv")
    print("\n✓ 시계열 데이터 저장: ./results_example/market_metrics_timeseries.csv")


print("\n" + "="*80)
print("✓ 모든 예제 완료!")
print("="*80 + "\n")
