"""
Multi-Asset Causal Network System
Validation Framework - 딜러/리서치/리스크 검증용
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from main_system import MultiAssetCausalSystem
from sklearn.cluster import KMeans
from scipy.spatial.distance import cosine
from datetime import timedelta


class ValidationFramework:
    """검증 프레임워크"""
    
    def __init__(self, system: MultiAssetCausalSystem):
        self.system = system
        
        # 경제적 논리 규칙 (방향성)
        self.plausible_edges = {
            ('Rates', 'Risk'): True,      # 금리 → 위험자산 ⭕
            ('Risk', 'Rates'): False,     # 위험자산 → 금리 ❌
            ('FX', 'Risk'): True,         # USD → EM Risk ⭕
            ('Safe Haven', 'Risk'): True, # 안전자산 → 위험자산 ⭕
            ('Risk', 'Safe Haven'): True, # Risk-off 시 ⭕
            ('Commodities', 'Risk'): True,
            ('Safe Haven', 'Rates'): 'conditional'  # ⚠️ 상황부
        }
    
    # =========================================================================
    # 1️⃣ 시간적 선행성 검증 (Temporal Validity)
    # =========================================================================
    
    def test_temporal_validity(self, forward_days: int = 5, top_k: int = 5):
        """
        가장 중요한 검증: Source 자산이 실제로 Target을 선행하는가?
        
        Returns:
        --------
        dict: {
            'source_asset': {
                'target_vol_increase': bool,
                'direction_consistency': float,
                'variance_ratio': float
            }
        }
        """
        print("="*80)
        print("1️⃣ 시간적 선행성 검증 (Temporal Validity)")
        print("="*80)
        print(f"분석 기간: t+1 ~ t+{forward_days}일\n")
        
        results = {}
        
        for idx, net_dict in enumerate(self.system.network_history):
            if idx + forward_days >= len(self.system.network_history):
                break
            
            date = net_dict['date']
            network = net_dict['network']
            
            # 각 날짜의 Top-K 강한 연결 추출
            edges = []
            for source in network.index:
                for target in network.columns:
                    if source != target:
                        weight = network.loc[source, target]
                        if weight > 0:
                            edges.append({
                                'source': source,
                                'target': target,
                                'weight': weight,
                                'source_bucket': self.system.bucket_mapping.get(source),
                                'target_bucket': self.system.bucket_mapping.get(target)
                            })
            
            edges = sorted(edges, key=lambda x: x['weight'], reverse=True)[:top_k]
            
            # 각 edge에 대해 forward validation
            for edge in edges:
                target_asset = edge['target']
                
                # t+1 ~ t+forward_days 구간의 Target 자산 변동성
                future_returns = []
                for i in range(1, forward_days + 1):
                    if idx + i < len(self.system.network_history):
                        future_date = self.system.network_history[idx + i]['date']
                        if future_date in self.system.processed_data.index and target_asset in self.system.processed_data.columns:
                            future_returns.append(
                                self.system.processed_data.loc[future_date, target_asset]
                            )
                
                if len(future_returns) > 0:
                    future_vol = np.std(future_returns)
                    
                    # 과거 변동성과 비교 (baseline)
                    past_returns = []
                    for i in range(1, forward_days + 1):
                        if idx - i >= 0:
                            past_date = self.system.network_history[idx - i]['date']
                            if past_date in self.system.processed_data.index and target_asset in self.system.processed_data.columns:
                                past_returns.append(
                                    self.system.processed_data.loc[past_date, target_asset]
                                )
                    
                    if len(past_returns) > 0:
                        past_vol = np.std(past_returns)
                        variance_ratio = future_vol / (past_vol + 1e-8)
                        
                        edge_key = f"{edge['source']} → {target_asset}"
                        if edge_key not in results:
                            results[edge_key] = {
                                'count': 0,
                                'vol_increase_count': 0,
                                'variance_ratios': []
                            }
                        
                        results[edge_key]['count'] += 1
                        results[edge_key]['variance_ratios'].append(variance_ratio)
                        if variance_ratio > 1.0:
                            results[edge_key]['vol_increase_count'] += 1
        
        # 결과 정리
        print("\n📊 주요 인과관계의 시간적 타당성:\n")
        print(f"{'Source → Target':<40} {'발생횟수':>8} {'Vol 증가율':>12} {'평균 분산비':>12}")
        print("-" * 80)
        
        sorted_results = sorted(results.items(), key=lambda x: x[1]['count'], reverse=True)[:15]
        
        for edge_key, stats in sorted_results:
            vol_increase_rate = stats['vol_increase_count'] / stats['count'] * 100
            avg_variance_ratio = np.mean(stats['variance_ratios'])
            
            print(f"{edge_key:<40} {stats['count']:>8} {vol_increase_rate:>11.1f}% {avg_variance_ratio:>12.3f}")
        
        print(f"\n✅ 기준: Vol 증가율 > 50%, 분산비 > 1.0 이면 시간적 선행성 확인")
        
        return results
    
    # =========================================================================
    # 2️⃣ 구조 안정성 테스트 (Structural Stability)
    # =========================================================================
    
    def test_structural_stability(self, n_clusters: int = 5):
        """
        비슷한 시장 국면에서 비슷한 구조가 나오는가?
        
        위기 시점들끼리 클러스터링되면 성공
        """
        print("\n" + "="*80)
        print("2️⃣ 구조 안정성 테스트 (Structural Stability)")
        print("="*80)
        
        # Structure vector 수집
        dates = []
        vectors = []
        
        for struct in self.system.structure_history:
            dates.append(struct['date'])
            vectors.append(struct['structure_vector'])
        
        vectors = np.array(vectors)
        
        # K-means 클러스터링
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(vectors)
        
        # 알려진 위기 구간 정의
        crisis_periods = {
            '2020 COVID': ('2020-02-01', '2020-04-30'),
            '2022 긴축': ('2022-09-01', '2022-11-30'),
            '2018 변동성': ('2018-10-01', '2018-12-31'),
        }
        
        print("\n📊 클러스터별 대표 시점:\n")
        
        cluster_dates = {}
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(labels == cluster_id)[0]
            cluster_dates[cluster_id] = [dates[i] for i in cluster_indices]
            
            print(f"클러스터 {cluster_id} ({len(cluster_indices)}개 시점):")
            
            # 대표 날짜 샘플 출력
            sample_dates = cluster_dates[cluster_id][:5]
            for d in sample_dates:
                print(f"  - {d.strftime('%Y-%m-%d')}")
            
            # 위기 구간과 겹치는지 확인
            crisis_overlap = {}
            for crisis_name, (start, end) in crisis_periods.items():
                start_date = pd.Timestamp(start)
                end_date = pd.Timestamp(end)
                
                overlap_count = sum(1 for d in cluster_dates[cluster_id] 
                                   if start_date <= d <= end_date)
                
                if overlap_count > 0:
                    crisis_overlap[crisis_name] = overlap_count
            
            if crisis_overlap:
                print(f"  📌 위기 구간 포함: {crisis_overlap}")
            
            print()
        
        print("✅ 성공 기준: 위기 시점들이 같은 클러스터에 모이면 합격\n")
        
        return {
            'labels': labels,
            'cluster_dates': cluster_dates,
            'n_clusters': n_clusters
        }
    
    # =========================================================================
    # 3️⃣ 이벤트 조건부 검증 (Event-based Sanity Check)
    # =========================================================================
    
    def test_event_sensitivity(self):
        """
        알려진 이벤트 전후로 네트워크가 바뀌었는가?
        
        FOMC → Rates 중심
        Risk-off → Safe Haven inbound 증가
        """
        print("="*80)
        print("3️⃣ 이벤트 조건부 검증 (Event-based Sanity Check)")
        print("="*80)
        
        # 주요 이벤트 정의
        events = {
            '2022 FOMC (11/2)': pd.Timestamp('2022-11-02'),
            '2020 COVID 급락': pd.Timestamp('2020-03-16'),
            '2023 SVB 파산': pd.Timestamp('2023-03-10'),
        }
        
        print("\n📊 이벤트 전후 네트워크 구조 변화:\n")
        
        for event_name, event_date in events.items():
            print(f"이벤트: {event_name}")
            
            # 이벤트 직전/직후 네트워크 찾기
            before_net = None
            after_net = None
            
            for net_dict in self.system.network_history:
                date = net_dict['date']
                if date < event_date and (event_date - date).days <= 5:
                    before_net = net_dict['network']
                elif date >= event_date and (date - event_date).days <= 5:
                    after_net = net_dict['network']
                    break
            
            if before_net is not None and after_net is not None:
                # Bucket별 영향력 변화
                before_bucket_strength = self._calculate_bucket_strength(before_net)
                after_bucket_strength = self._calculate_bucket_strength(after_net)
                
                print(f"  구조 변화:")
                for bucket in before_bucket_strength.keys():
                    before_val = before_bucket_strength[bucket]
                    after_val = after_bucket_strength[bucket]
                    change = after_val - before_val
                    
                    arrow = "↑" if change > 0.01 else "↓" if change < -0.01 else "→"
                    print(f"    {bucket:15} {before_val:.3f} → {after_val:.3f} {arrow}")
                
                print()
            else:
                print(f"  ⚠️ 데이터 없음\n")
        
        print("✅ 기준: FOMC → Rates 증가, Risk-off → Safe Haven 증가 확인\n")
    
    def _calculate_bucket_strength(self, network):
        """Bucket별 총 out-strength 계산"""
        bucket_strength = {}
        
        for source in network.index:
            source_bucket = self.system.bucket_mapping.get(source, 'Unknown')
            strength = network.loc[source, :].sum()
            
            if source_bucket not in bucket_strength:
                bucket_strength[source_bucket] = 0
            bucket_strength[source_bucket] += strength
        
        # 정규화
        total = sum(bucket_strength.values())
        if total > 0:
            bucket_strength = {k: v/total for k, v in bucket_strength.items()}
        
        return bucket_strength
    
    # =========================================================================
    # 4️⃣ 방향성 논리 검증 (Economic Plausibility)
    # =========================================================================
    
    def test_economic_plausibility(self):
        """
        경제적으로 말이 되는 방향인가?
        
        Rates → Equity ⭕
        Equity → Rates ❌
        """
        print("="*80)
        print("4️⃣ 방향성 논리 검증 (Economic Plausibility)")
        print("="*80)
        
        violation_count = 0
        total_edges = 0
        
        violations = []
        
        for net_dict in self.system.network_history:
            network = net_dict['network']
            date = net_dict['date']
            
            for source in network.index:
                for target in network.columns:
                    if source == target:
                        continue
                    
                    weight = network.loc[source, target]
                    if weight > 0.01:  # 유의한 연결만
                        total_edges += 1
                        
                        source_bucket = self.system.bucket_mapping.get(source, 'Unknown')
                        target_bucket = self.system.bucket_mapping.get(target, 'Unknown')
                        
                        edge_key = (source_bucket, target_bucket)
                        
                        if edge_key in self.plausible_edges:
                            if self.plausible_edges[edge_key] == False:
                                violation_count += 1
                                violations.append({
                                    'date': date,
                                    'source': source,
                                    'target': target,
                                    'source_bucket': source_bucket,
                                    'target_bucket': target_bucket,
                                    'weight': weight
                                })
        
        print(f"\n📊 경제적 논리 위배 분석:\n")
        print(f"총 유의한 엣지: {total_edges}")
        print(f"논리 위배 엣지: {violation_count}")
        print(f"위배율: {violation_count/total_edges*100:.2f}%\n")
        
        if len(violations) > 0:
            print("❌ 주요 위배 사례 (상위 10개):\n")
            violations = sorted(violations, key=lambda x: x['weight'], reverse=True)[:10]
            
            for v in violations:
                print(f"  {v['date'].strftime('%Y-%m-%d')}: {v['source']} → {v['target']} "
                      f"[{v['source_bucket']} → {v['target_bucket']}] (weight: {v['weight']:.3f})")
        else:
            print("✅ 논리 위배 없음!")
        
        print(f"\n✅ 기준: 위배율 < 10% 이면 합격\n")
        
        return {
            'total_edges': total_edges,
            'violations': violation_count,
            'violation_rate': violation_count/total_edges if total_edges > 0 else 0
        }
    
    # =========================================================================
    # 5️⃣ 반사실 테스트 (Perturbation Test)
    # =========================================================================
    
    def test_perturbation(self, target_asset: str = '코스피200 연결'):
        """
        특정 자산 return을 0으로 고정하면 네트워크가 합리적으로 바뀌나?
        
        기대: 해당 자산의 out-degree 감소
        """
        print("="*80)
        print("5️⃣ 반사실 테스트 (Perturbation Test)")
        print("="*80)
        print(f"대상 자산: {target_asset}\n")
        
        # 원본 데이터 복사
        perturbed_data = self.system.processed_data.copy()
        
        # Target 자산을 0으로 고정 (변동성은 유지)
        if target_asset in perturbed_data.columns:
            perturbed_data[target_asset] = 0
        else:
            print(f"⚠️ 자산 '{target_asset}'를 찾을 수 없습니다.")
            return None
        
        # 새로운 네트워크 생성 (최근 200일만)
        print("원본 네트워크 재구성 중...")
        original_network = self.system.network_history[-1]['network']
        
        print("교란된 네트워크 생성 중...")
        # 간단한 재생성 (Granger 대신 correlation 사용)
        from causal_network import CausalNetworkModel
        
        temp_model = CausalNetworkModel(max_lag=3)
        perturbed_network = temp_model.correlation_network(
            perturbed_data.iloc[-120:],  # 최근 120일
            use_lag=True
        )
        
        # Out-degree 비교
        original_out = original_network.loc[target_asset, :].sum()
        perturbed_out = perturbed_network.loc[target_asset, :].sum()
        
        print(f"\n📊 {target_asset}의 영향력 변화:\n")
        print(f"원본 out-degree:     {original_out:.4f}")
        print(f"교란 후 out-degree:  {perturbed_out:.4f}")
        print(f"감소율:              {(1 - perturbed_out/original_out)*100:.1f}%\n")
        
        # 전체 네트워크 안정성
        total_edges_original = (original_network > 0.01).sum().sum()
        total_edges_perturbed = (perturbed_network > 0.01).sum().sum()
        
        print(f"전체 유의한 엣지 수:")
        print(f"  원본:    {total_edges_original}")
        print(f"  교란 후: {total_edges_perturbed}")
        print(f"  변화율:  {(total_edges_perturbed/total_edges_original - 1)*100:+.1f}%\n")
        
        print("✅ 기준: Out-degree 30% 이상 감소 + 전체 구조 붕괴 없음(±20% 이내)\n")
        
        return {
            'original_out_degree': original_out,
            'perturbed_out_degree': perturbed_out,
            'reduction_rate': (1 - perturbed_out/original_out),
            'network_stability': total_edges_perturbed/total_edges_original
        }
    
    # =========================================================================
    # 6️⃣ 간이 예측 검증 (Auxiliary Forecast Test)
    # =========================================================================
    
    def test_auxiliary_forecast(self, forward_days: int = 5):
        """
        High connectivity day → 이후 변동성 ↑
        Low connectivity day → 이후 변동성 ↓
        """
        print("="*80)
        print("6️⃣ 간이 예측 검증 (Auxiliary Forecast Test)")
        print("="*80)
        
        # 각 날짜의 네트워크 밀도 계산
        densities = []
        future_vols = []
        
        for idx, net_dict in enumerate(self.system.network_history):
            if idx + forward_days >= len(self.system.network_history):
                break
            
            network = net_dict['network']
            density = (network > 0.01).sum().sum() / (network.shape[0] * network.shape[1])
            
            densities.append(density)
            
            # 이후 5일간 평균 변동성
            future_returns = []
            for i in range(1, forward_days + 1):
                future_date = self.system.network_history[idx + i]['date']
                if future_date in self.system.processed_data.index:
                    future_returns.append(
                        self.system.processed_data.loc[future_date, :].values
                    )
            
            if len(future_returns) > 0:
                avg_vol = np.nanmean([np.std(r) for r in future_returns])
                future_vols.append(avg_vol)
            else:
                future_vols.append(np.nan)
        
        densities = np.array(densities)
        future_vols = np.array(future_vols)
        
        # NaN 제거
        valid_mask = ~np.isnan(future_vols)
        densities = densities[valid_mask]
        future_vols = future_vols[valid_mask]
        
        # 상위/하위 30% 비교
        high_threshold = np.percentile(densities, 70)
        low_threshold = np.percentile(densities, 30)
        
        high_density_mask = densities >= high_threshold
        low_density_mask = densities <= low_threshold
        
        high_vol = np.mean(future_vols[high_density_mask])
        low_vol = np.mean(future_vols[low_density_mask])
        
        print(f"\n📊 네트워크 밀도와 이후 변동성 관계:\n")
        print(f"High connectivity days (상위 30%):")
        print(f"  평균 네트워크 밀도: {np.mean(densities[high_density_mask]):.4f}")
        print(f"  이후 {forward_days}일 평균 변동성: {high_vol:.6f}\n")
        
        print(f"Low connectivity days (하위 30%):")
        print(f"  평균 네트워크 밀도: {np.mean(densities[low_density_mask]):.4f}")
        print(f"  이후 {forward_days}일 평균 변동성: {low_vol:.6f}\n")
        
        vol_ratio = high_vol / low_vol
        print(f"변동성 비율 (High/Low): {vol_ratio:.2f}x\n")
        
        print("✅ 기준: High > Low 이고 비율 > 1.2 이면 예측력 확인\n")
        
        return {
            'high_vol': high_vol,
            'low_vol': low_vol,
            'vol_ratio': vol_ratio
        }
    
    # =========================================================================
    # 전체 검증 실행
    # =========================================================================
    
    def run_all_tests(self):
        """모든 검증 실행"""
        print("\n" + "🔬" * 40)
        print(" " * 15 + "검증 프레임워크 시작")
        print("🔬" * 40 + "\n")
        
        results = {}
        
        # 1. 시간적 선행성
        results['temporal_validity'] = self.test_temporal_validity(forward_days=5, top_k=5)
        
        # 2. 구조 안정성
        results['structural_stability'] = self.test_structural_stability(n_clusters=5)
        
        # 3. 경제적 논리
        results['economic_plausibility'] = self.test_economic_plausibility()
        
        # 4. 반사실 테스트
        results['perturbation'] = self.test_perturbation(target_asset='코스피200 연결')
        
        # 5. 간이 예측
        results['auxiliary_forecast'] = self.test_auxiliary_forecast(forward_days=5)
        
        print("\n" + "="*80)
        print("🎯 전체 검증 완료")
        print("="*80)
        
        return results


if __name__ == "__main__":
    print("시스템 로딩 중...")
    
    # 시스템 로드 또는 생성
    system = MultiAssetCausalSystem(
        csv_path="가격 데이터.csv",
        vol_window=20,
        causality_window=120,
        max_lag=3
    )
    
    # 캐시에서 로드 시도
    cached = MultiAssetCausalSystem.load_system("./results/system_cache.pkl")
    if cached is not None:
        system, _ = cached
        print("✓ 캐시에서 로드 완료\n")
    else:
        print("캐시 없음. 새로 학습 중...")
        system.load_and_preprocess()
        system.build_causal_networks(method='granger', sample_size=1000)
        system.analyze_market_structures()
        system.save_system("./results/system_cache.pkl")
        print("✓ 학습 완료\n")
    
    # 검증 실행
    validator = ValidationFramework(system)
    results = validator.run_all_tests()
