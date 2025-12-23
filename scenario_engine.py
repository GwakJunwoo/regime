"""
Multi-Asset Causal Network System
Scenario Engine - State → Outcome Mapping

목표:
- ❌ 예측 아님
- ✅ 조건부 통계
- ✅ 해석 가능

"현재 시장 구조가 과거 어느 국면과 유사한지 찾고,
그 국면 이후 자산들의 조건부 분포를 보여준다"
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.spatial.distance import cosine
import warnings
warnings.filterwarnings('ignore')


class ImprovedStructureDistance:
    """개선된 구조 거리 함수"""
    
    @staticmethod
    def degree_weighted_distance(s1: np.ndarray, s2: np.ndarray, 
                                 network1: pd.DataFrame, 
                                 network2: pd.DataFrame) -> float:
        """
        Degree-weighted distance
        
        Source hub의 변화에 더 민감
        
        distance = Σ w_i · |S_today_i - S_past_i|
        w_i = degree centrality
        """
        # Out-degree 계산 (각 자산의 영향력)
        deg1 = network1.sum(axis=1).values
        deg2 = network2.sum(axis=1).values
        
        # Degree 평균으로 가중치
        weights = (deg1 + deg2) / 2
        weights = weights / (weights.sum() + 1e-8)  # 정규화
        
        # 가중 거리
        weighted_diff = weights * np.abs(s1 - s2)
        
        return np.sum(weighted_diff)
    
    @staticmethod
    def directional_distance(network1: pd.DataFrame, 
                            network2: pd.DataFrame,
                            bucket_mapping: Dict[str, str]) -> Dict[str, float]:
        """
        방향성 거리 (Risk-on/off 전환 포착)
        
        Returns:
        --------
        {
            'risk_to_safe': float,  # Risk → Safe 강도 차이
            'safe_to_risk': float,  # Safe → Risk 강도 차이
            'inflow_outflow_ratio': float  # 유입/유출 비율 차이
        }
        """
        def calculate_directional_strength(network, mapping):
            risk_to_safe = 0
            safe_to_risk = 0
            
            for source in network.index:
                for target in network.columns:
                    if source == target:
                        continue
                    
                    source_bucket = mapping.get(source, 'Unknown')
                    target_bucket = mapping.get(target, 'Unknown')
                    weight = network.loc[source, target]
                    
                    if source_bucket == 'Risk' and target_bucket == 'Safe Haven':
                        risk_to_safe += weight
                    elif source_bucket == 'Safe Haven' and target_bucket == 'Risk':
                        safe_to_risk += weight
            
            return risk_to_safe, safe_to_risk
        
        r2s_1, s2r_1 = calculate_directional_strength(network1, bucket_mapping)
        r2s_2, s2r_2 = calculate_directional_strength(network2, bucket_mapping)
        
        return {
            'risk_to_safe_diff': abs(r2s_1 - r2s_2),
            'safe_to_risk_diff': abs(s2r_1 - s2r_2),
            'direction_change': abs((r2s_1 - s2r_1) - (r2s_2 - s2r_2))
        }


class OutcomeDistributionAnalyzer:
    """결과 분포 분석기 - 방향+분위수만"""
    
    def __init__(self, returns_data: pd.DataFrame):
        """수익률 데이터를 받음 (가격 아님!)"""
        self.returns = returns_data
    
    def analyze_forward_outcomes(self, 
                                 reference_date: pd.Timestamp,
                                 forward_days: int = 5) -> Dict:
        """
        특정 날짜 이후 forward_days간의 결과 분포
        
        Returns:
        --------
        {
            'asset': {
                'direction': 1/-1/0,
                'direction_prob': float,
                'percentiles': {25: x, 50: y, 75: z},
                'volatility_ratio': float
            }
        }
        """
        outcomes = {}
        
        # Reference 날짜 찾기
        if reference_date not in self.returns.index:
            # 가장 가까운 날짜 찾기
            idx = self.returns.index.get_indexer([reference_date], method='nearest')[0]
            reference_date = self.returns.index[idx]
        
        ref_idx = self.returns.index.get_loc(reference_date)
        
        # Forward window
        if ref_idx + forward_days >= len(self.returns):
            return None
        
        forward_window = self.returns.iloc[ref_idx + 1 : ref_idx + forward_days + 1]
        
        for asset in self.returns.columns:
            returns = forward_window[asset].values
            
            # NaN 제거
            returns = returns[~np.isnan(returns)]
            
            if len(returns) == 0:
                continue
            
            # 방향 (sign의 최빈값)
            direction = 1 if np.sum(returns > 0) > np.sum(returns < 0) else -1
            direction_prob = max(np.sum(returns > 0), np.sum(returns < 0)) / len(returns)
            
            # 분위수
            percentiles = {
                25: np.percentile(returns, 25),
                50: np.percentile(returns, 50),
                75: np.percentile(returns, 75)
            }
            
            # 변동성 (과거 대비)
            past_window = self.returns.iloc[ref_idx - forward_days : ref_idx]
            past_vol = past_window[asset].std()
            forward_vol = forward_window[asset].std()
            vol_ratio = forward_vol / (past_vol + 1e-8)
            
            outcomes[asset] = {
                'direction': direction,
                'direction_prob': direction_prob,
                'percentiles': percentiles,
                'volatility_ratio': vol_ratio,
                'mean_return': np.mean(returns),  # 참고용
                'total_return': np.sum(returns)   # 참고용
            }
        
        return outcomes
    
    def aggregate_outcomes(self, outcome_list: List[Dict]) -> Dict:
        """
        여러 유사 국면의 결과를 집계
        
        Parameters:
        -----------
        outcome_list : List[Dict]
            각 유사 날짜의 outcome dict
        
        Returns:
        --------
        {
            'asset': {
                'direction_consensus': 1/-1,
                'direction_strength': float (0~1),
                'median_return': float,
                'p25_return': float,
                'p75_return': float,
                'vol_increase_prob': float
            }
        }
        """
        if len(outcome_list) == 0:
            return {}
        
        # 자산 목록
        assets = set()
        for outcome in outcome_list:
            if outcome is not None:
                assets.update(outcome.keys())
        
        aggregated = {}
        
        for asset in assets:
            directions = []
            mean_returns = []
            vol_ratios = []
            
            for outcome in outcome_list:
                if outcome is not None and asset in outcome:
                    directions.append(outcome[asset]['direction'])
                    mean_returns.append(outcome[asset]['mean_return'])
                    vol_ratios.append(outcome[asset]['volatility_ratio'])
            
            if len(directions) == 0:
                continue
            
            # 방향 합의
            direction_consensus = 1 if np.sum(directions) > 0 else -1
            direction_strength = abs(np.sum(directions)) / len(directions)
            
            # 수익률 분위수
            mean_returns = np.array(mean_returns)
            
            aggregated[asset] = {
                'direction_consensus': direction_consensus,
                'direction_strength': direction_strength,
                'median_return': np.median(mean_returns),
                'p25_return': np.percentile(mean_returns, 25),
                'p75_return': np.percentile(mean_returns, 75),
                'vol_increase_prob': np.sum(np.array(vol_ratios) > 1.0) / len(vol_ratios),
                'sample_size': len(directions)
            }
        
        return aggregated


class ScenarioEngine:
    """미니 시나리오 엔진"""
    
    def __init__(self, 
                 system,
                 processed_data: pd.DataFrame,
                 bucket_mapping: Dict[str, str]):
        self.system = system
        self.processed_data = processed_data
        self.bucket_mapping = bucket_mapping
        self.distance_calc = ImprovedStructureDistance()
        self.outcome_analyzer = OutcomeDistributionAnalyzer(processed_data)
    
    def sparsify_network(self, network: pd.DataFrame, top_k_percent: float = 0.15) -> pd.DataFrame:
        """
        Edge sparsification - 상위 k% edge만 유지
        
        구조 안정성 개선
        """
        sparse = network.copy()
        
        # 모든 edge를 flatten
        edges = []
        for i in network.index:
            for j in network.columns:
                if i != j:
                    edges.append(network.loc[i, j])
        
        # Threshold 계산
        threshold = np.percentile(edges, (1 - top_k_percent) * 100)
        
        # 작은 edge 제거
        sparse[sparse < threshold] = 0
        
        return sparse
    
    def find_similar_structures_enhanced(self,
                                        current_date: pd.Timestamp,
                                        top_k: int = 10,
                                        exclude_recent_days: int = 60) -> List[Dict]:
        """
        개선된 유사 구조 탐색
        
        - Degree-weighted distance
        - Directional components
        """
        # 현재 날짜의 구조
        current_structure = None
        current_network = None
        
        for struct in self.system.structure_history:
            if pd.Timestamp(struct['date']).normalize() == current_date.normalize():
                current_structure = struct['structure_vector']
                break
        
        for net_dict in self.system.network_history:
            if pd.Timestamp(net_dict['date']).normalize() == current_date.normalize():
                current_network = net_dict['network']
                break
        
        if current_structure is None or current_network is None:
            return []
        
        # Sparsify
        current_network_sparse = self.sparsify_network(current_network, top_k_percent=0.15)
        
        # 유사도 계산
        similarities = []
        
        for idx, struct in enumerate(self.system.structure_history):
            date = struct['date']
            
            # 최근 제외
            if (current_date - date).days < exclude_recent_days:
                continue
            
            # 해당 날짜의 네트워크 찾기
            past_network = None
            for net_dict in self.system.network_history:
                if pd.Timestamp(net_dict['date']).normalize() == date.normalize():
                    past_network = net_dict['network']
                    break
            
            if past_network is None:
                continue
            
            # Sparsify
            past_network_sparse = self.sparsify_network(past_network, top_k_percent=0.15)
            
            # Degree-weighted distance
            deg_dist = self.distance_calc.degree_weighted_distance(
                current_structure,
                struct['structure_vector'],
                current_network_sparse,
                past_network_sparse
            )
            
            # Directional distance
            dir_metrics = self.distance_calc.directional_distance(
                current_network_sparse,
                past_network_sparse,
                self.bucket_mapping
            )
            
            # 종합 거리 (degree-weighted 70%, directional 30%)
            total_distance = 0.7 * deg_dist + 0.3 * dir_metrics['direction_change']
            
            similarities.append({
                'date': date,
                'distance': total_distance,
                'deg_distance': deg_dist,
                'directional_metrics': dir_metrics,
                'structure_vector': struct['structure_vector']
            })
        
        # 정렬 (거리 작은 순)
        similarities = sorted(similarities, key=lambda x: x['distance'])[:top_k]
        
        return similarities
    
    def generate_scenario_summary(self,
                                 current_date: pd.Timestamp,
                                 forward_days: int = 5,
                                 top_k_similar: int = 10) -> Dict:
        """
        시나리오 요약 생성 (딜러용)
        
        Returns:
        --------
        {
            'current_date': date,
            'similar_periods': [...],
            'hub_assets': [...],
            'structure_interpretation': str,
            'outcome_distribution': {...}
        }
        """
        print("="*80)
        print(f"📌 Market Structure Scenario - {current_date.strftime('%Y-%m-%d')}")
        print("="*80)
        
        # 1. 유사 구조 탐색
        similar_periods = self.find_similar_structures_enhanced(
            current_date,
            top_k=top_k_similar,
            exclude_recent_days=60
        )
        
        if len(similar_periods) == 0:
            print("⚠️ 유사 국면을 찾을 수 없습니다.")
            return None
        
        # 2. 현재 네트워크의 Hub 자산 찾기
        current_network = None
        for net_dict in self.system.network_history:
            if pd.Timestamp(net_dict['date']).normalize() == current_date.normalize():
                current_network = net_dict['network']
                break
        
        hub_assets = []
        if current_network is not None:
            out_degrees = current_network.sum(axis=1).sort_values(ascending=False)
            hub_assets = [
                {
                    'asset': asset,
                    'strength': strength,
                    'bucket': self.bucket_mapping.get(asset, 'Unknown')
                }
                for asset, strength in out_degrees.head(5).items()
            ]
        
        # 3. 구조 해석 (방향성)
        dir_metrics = similar_periods[0]['directional_metrics']
        
        interpretation = []
        if dir_metrics['risk_to_safe_diff'] > dir_metrics['safe_to_risk_diff']:
            interpretation.append("Risk-off 전환 구조")
        else:
            interpretation.append("Risk-on 구조")
        
        # Hub bucket 해석
        if len(hub_assets) > 0:
            top_bucket = hub_assets[0]['bucket']
            interpretation.append(f"주요 정보 허브: {top_bucket}")
        
        # 4. 과거 유사 국면 이후 결과 수집
        outcome_list = []
        for period in similar_periods:
            outcome = self.outcome_analyzer.analyze_forward_outcomes(
                period['date'],
                forward_days=forward_days
            )
            if outcome is not None:
                outcome_list.append(outcome)
        
        # 5. 결과 집계
        aggregated_outcomes = self.outcome_analyzer.aggregate_outcomes(outcome_list)
        
        # 6. 출력
        print("\n• 유사 국면:")
        for i, period in enumerate(similar_periods[:5], 1):
            print(f"  {i}. {period['date'].strftime('%Y-%m-%d')} (거리: {period['distance']:.4f})")
        
        print("\n• 주요 정보 허브 (Out-degree 상위 5개):")
        for hub in hub_assets:
            print(f"  - {hub['asset']} [{hub['bucket']}]: {hub['strength']:.3f}")
        
        print("\n• 구조 해석:")
        for interp in interpretation:
            print(f"  - {interp}")
        
        print(f"\n📊 과거 해당 국면 이후 {forward_days}일 결과 (조건부 분포):\n")
        
        # 자산별 결과 (bucket별로 그룹화)
        bucket_outcomes = {}
        for asset, outcome in aggregated_outcomes.items():
            bucket = self.bucket_mapping.get(asset, 'Unknown')
            if bucket not in bucket_outcomes:
                bucket_outcomes[bucket] = []
            bucket_outcomes[bucket].append((asset, outcome))
        
        for bucket in ['Rates', 'Risk', 'Safe Haven', 'FX', 'Commodities']:
            if bucket not in bucket_outcomes:
                continue
            
            print(f"\n[{bucket}]")
            
            for asset, outcome in bucket_outcomes[bucket]:
                direction_symbol = "↑" if outcome['direction_consensus'] == 1 else "↓"
                strength_pct = outcome['direction_strength'] * 100
                
                # 수익률은 이미 normalized returns이므로 basis point로 표시
                print(f"  • {asset}")
                print(f"    방향: {direction_symbol} (확률 {strength_pct:.0f}%)")
                print(f"    중앙값: {outcome['median_return']:+.4f}")
                print(f"    범위: [{outcome['p25_return']:+.4f}, {outcome['p75_return']:+.4f}]")
                print(f"    변동성 증가 확률: {outcome['vol_increase_prob']*100:.0f}%")
                print(f"    샘플: {outcome['sample_size']}개 국면")
        
        print("\n" + "="*80)
        print("✅ 이것은 예측이 아닌 '과거 유사 국면의 조건부 통계'입니다.")
        print("="*80)
        
        return {
            'current_date': current_date,
            'similar_periods': similar_periods,
            'hub_assets': hub_assets,
            'structure_interpretation': interpretation,
            'outcome_distribution': aggregated_outcomes,
            'sample_size': len(outcome_list)
        }


if __name__ == "__main__":
    from main_system import MultiAssetCausalSystem
    
    print("시스템 로딩 중...")
    
    # 캐시에서 로드
    cached = MultiAssetCausalSystem.load_system("./results/system_cache.pkl")
    if cached is not None:
        system, _ = cached
        print("✓ 캐시에서 로드 완료\n")
    else:
        print("⚠️ 캐시 없음. 먼저 시스템을 학습해주세요.")
        exit(1)
    
    # Scenario Engine 생성 (processed_data 사용!)
    engine = ScenarioEngine(
        system=system,
        processed_data=system.processed_data,
        bucket_mapping=system.bucket_mapping
    )
    
    # 최근 날짜로 시나리오 생성
    latest_date = system.network_history[-1]['date']
    
    scenario = engine.generate_scenario_summary(
        current_date=latest_date,
        forward_days=5,
        top_k_similar=10
    )
