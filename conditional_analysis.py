"""
Multi-Asset Causal Network System
Conditional Transmission & Historical Analogue Modules
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')


class ConditionalTransmissionAnalyzer:
    """조건부 전이 분석: A 자산 변화 시 다음 반응 예상"""
    
    def __init__(self, bucket_mapping: Dict[str, str]):
        self.bucket_mapping = bucket_mapping
    
    def analyze_conditional_impact(self,
                                   causality_matrix: pd.DataFrame,
                                   source_asset: str,
                                   top_k: int = 5) -> List[Dict]:
        """
        특정 자산 변화 시 영향받을 가능성이 높은 자산 식별
        
        Parameters:
        -----------
        causality_matrix : pd.DataFrame
            현재 시점의 인과 네트워크
        source_asset : str
            변화가 발생한 (또는 가정한) 자산
        top_k : int
            상위 몇 개 자산 반환
        
        Returns:
        --------
        List[Dict] : 영향받을 자산 정보
            {'asset', 'weight', 'bucket', 'relative_strength'}
        """
        if source_asset not in causality_matrix.index:
            print(f"Warning: {source_asset} not in causality matrix")
            return []
        
        # Source asset의 영향도
        impacts = causality_matrix.loc[source_asset]
        
        # 영향도가 0보다 큰 자산만 필터링
        significant_impacts = impacts[impacts > 0].sort_values(ascending=False)
        
        # Top-K 추출
        top_impacts = significant_impacts.head(top_k)
        
        # 상대 강도 계산 (정규화)
        max_impact = top_impacts.max() if len(top_impacts) > 0 else 1
        
        results = []
        for asset, weight in top_impacts.items():
            results.append({
                'asset': asset,
                'weight': weight,
                'bucket': self.bucket_mapping.get(asset, 'Unknown'),
                'relative_strength': weight / max_impact if max_impact > 0 else 0
            })
        
        return results
    
    def scenario_analysis(self,
                         causality_matrix: pd.DataFrame,
                         scenarios: List[str]) -> Dict[str, List[Dict]]:
        """
        여러 시나리오에 대한 조건부 분석
        
        Parameters:
        -----------
        scenarios : List[str]
            분석할 자산 리스트
        
        Returns:
        --------
        Dict : 각 시나리오별 결과
        """
        results = {}
        
        for source_asset in scenarios:
            impacts = self.analyze_conditional_impact(causality_matrix, source_asset)
            results[source_asset] = impacts
        
        return results
    
    def cross_bucket_transmission(self,
                                  causality_matrix: pd.DataFrame,
                                  source_bucket: str,
                                  target_bucket: str) -> Dict:
        """
        특정 버킷 간 전이 상세 분석
        
        Returns:
        --------
        Dict : 버킷 간 전이 상세 정보
        """
        source_assets = [
            asset for asset, bucket in self.bucket_mapping.items()
            if bucket == source_bucket and asset in causality_matrix.index
        ]
        target_assets = [
            asset for asset, bucket in self.bucket_mapping.items()
            if bucket == target_bucket and asset in causality_matrix.columns
        ]
        
        if len(source_assets) == 0 or len(target_assets) == 0:
            return {
                'total_influence': 0,
                'asset_pairs': [],
                'avg_influence': 0
            }
        
        # 각 자산 쌍의 영향도
        asset_pairs = []
        for source in source_assets:
            for target in target_assets:
                weight = causality_matrix.loc[source, target]
                if weight > 0:
                    asset_pairs.append({
                        'source': source,
                        'target': target,
                        'weight': weight
                    })
        
        asset_pairs.sort(key=lambda x: x['weight'], reverse=True)
        
        total_influence = sum(pair['weight'] for pair in asset_pairs)
        avg_influence = total_influence / len(asset_pairs) if len(asset_pairs) > 0 else 0
        
        return {
            'total_influence': total_influence,
            'asset_pairs': asset_pairs,
            'avg_influence': avg_influence,
            'n_connections': len(asset_pairs)
        }


class HistoricalAnalogueSearch:
    """과거 유사 국면 탐색"""
    
    def __init__(self):
        self.structure_history = []
    
    def add_structure_vector(self, 
                            date: pd.Timestamp, 
                            structure_vector: np.ndarray):
        """구조 벡터 히스토리에 추가"""
        self.structure_history.append({
            'date': date,
            'vector': structure_vector
        })
    
    def find_similar_periods(self,
                            current_vector: np.ndarray,
                            top_k: int = 10,
                            exclude_recent_days: int = 60) -> List[Dict]:
        """
        현재 구조와 유사한 과거 시점 찾기
        
        Parameters:
        -----------
        current_vector : np.ndarray
            현재 시장 구조 벡터
        top_k : int
            상위 몇 개 유사 시점 반환
        exclude_recent_days : int
            최근 N일은 제외 (너무 가까운 시점 배제)
        
        Returns:
        --------
        List[Dict] : 유사 시점 정보
            {'date', 'similarity', 'rank'}
        """
        if len(self.structure_history) == 0:
            return []
        
        # 최근 날짜 계산
        latest_date = self.structure_history[-1]['date']
        cutoff_date = latest_date - pd.Timedelta(days=exclude_recent_days)
        
        # 유사도 계산
        similarities = []
        
        for item in self.structure_history:
            # 최근 기간 제외
            if item['date'] >= cutoff_date:
                continue
            
            # Cosine Similarity 계산
            sim = cosine_similarity(
                current_vector.reshape(1, -1),
                item['vector'].reshape(1, -1)
            )[0, 0]
            
            similarities.append({
                'date': item['date'],
                'similarity': sim,
                'vector': item['vector']
            })
        
        # 유사도 순으로 정렬
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Top-K 반환
        results = []
        for rank, item in enumerate(similarities[:top_k], 1):
            results.append({
                'date': item['date'],
                'similarity': item['similarity'],
                'rank': rank
            })
        
        return results
    
    def get_period_context(self,
                          date: pd.Timestamp,
                          price_data: pd.DataFrame,
                          forward_days: int = 20) -> Dict:
        """
        특정 시점 이후의 자산 반응 패턴 반환
        
        Parameters:
        -----------
        date : pd.Timestamp
            기준 날짜
        price_data : pd.DataFrame
            가격 데이터
        forward_days : int
            이후 몇 일간의 데이터를 볼 것인가
        
        Returns:
        --------
        Dict : 기간 정보 및 자산 반응
        """
        if date not in price_data.index:
            return None
        
        date_idx = price_data.index.get_loc(date)
        
        # Forward period
        end_idx = min(date_idx + forward_days, len(price_data))
        forward_period = price_data.iloc[date_idx:end_idx]
        
        # 각 자산의 누적 수익률 계산
        if len(forward_period) > 1:
            cumulative_returns = (forward_period / forward_period.iloc[0] - 1) * 100
            final_returns = cumulative_returns.iloc[-1]
        else:
            final_returns = pd.Series(0, index=price_data.columns)
        
        return {
            'date': date,
            'forward_days': len(forward_period),
            'final_returns': final_returns.to_dict(),
            'best_performer': final_returns.idxmax(),
            'worst_performer': final_returns.idxmin()
        }
    
    def compare_analogues(self,
                         similar_periods: List[Dict],
                         price_data: pd.DataFrame,
                         forward_days: int = 20) -> pd.DataFrame:
        """
        유사 시점들의 이후 반응 비교
        
        Returns:
        --------
        pd.DataFrame : 각 유사 시점별 자산 수익률
        """
        comparison_data = []
        
        for period in similar_periods:
            context = self.get_period_context(
                period['date'], 
                price_data, 
                forward_days
            )
            
            if context is not None:
                row = {
                    'date': period['date'],
                    'similarity': period['similarity'],
                    'rank': period['rank']
                }
                row.update(context['final_returns'])
                comparison_data.append(row)
        
        if len(comparison_data) > 0:
            return pd.DataFrame(comparison_data)
        else:
            return pd.DataFrame()


class AssetContributionAnalyzer:
    """자산별 구조 변화 기여도 분석"""
    
    def __init__(self, bucket_mapping: Dict[str, str]):
        self.bucket_mapping = bucket_mapping
    
    def decompose_structure_movement(self,
                                     causality_t: pd.DataFrame,
                                     causality_t_minus_1: pd.DataFrame,
                                     structure_t: np.ndarray,
                                     structure_t_minus_1: np.ndarray) -> Dict:
        """
        구조 이동에 대한 자산별 기여도 분석
        
        핵심 아이디어:
        - 각 자산의 인과 관계 변화가 전체 구조 벡터 변화에 얼마나 기여했는가?
        - 자산 i의 영향도 변화 = Σ|w_ij(t) - w_ij(t-1)|
        
        Parameters:
        -----------
        causality_t : pd.DataFrame
            현재 시점 인과 행렬
        causality_t_minus_1 : pd.DataFrame
            이전 시점 인과 행렬
        structure_t : np.ndarray
            현재 구조 벡터
        structure_t_minus_1 : np.ndarray
            이전 구조 벡터
            
        Returns:
        --------
        Dict : 자산별/버킷별 기여도 정보
        """
        # 인과 행렬 변화량
        causality_change = np.abs(causality_t.values - causality_t_minus_1.values)
        
        # 자산별 총 변화량 (source 기준)
        asset_source_changes = causality_change.sum(axis=1)
        # 자산별 총 변화량 (target 기준)
        asset_target_changes = causality_change.sum(axis=0)
        # 총 변화량 (source + target)
        asset_total_changes = asset_source_changes + asset_target_changes
        
        # 자산별 기여도 계산
        asset_contributions = {}
        for i, asset in enumerate(causality_t.index):
            asset_contributions[asset] = {
                'source_change': float(asset_source_changes[i]),
                'target_change': float(asset_target_changes[i]),
                'total_change': float(asset_total_changes[i]),
                'bucket': self.bucket_mapping.get(asset, 'Unknown')
            }
        
        # 버킷별 집계
        bucket_contributions = {}
        for bucket in set(self.bucket_mapping.values()):
            bucket_assets = [a for a, b in self.bucket_mapping.items() if b == bucket]
            bucket_total = sum(
                asset_contributions[a]['total_change'] 
                for a in bucket_assets 
                if a in asset_contributions
            )
            bucket_contributions[bucket] = bucket_total
        
        # 총 구조 이동량
        total_movement = np.linalg.norm(structure_t - structure_t_minus_1)
        
        # 상위 기여 자산
        sorted_assets = sorted(
            asset_contributions.items(),
            key=lambda x: x[1]['total_change'],
            reverse=True
        )
        
        return {
            'total_structure_movement': float(total_movement),
            'total_causality_change': float(causality_change.sum()),
            'asset_contributions': asset_contributions,
            'bucket_contributions': bucket_contributions,
            'top_contributors': [
                {'asset': asset, **contrib} 
                for asset, contrib in sorted_assets[:10]
            ]
        }
    
    def analyze_regime_transition_drivers(self,
                                          structure_history: List[Dict],
                                          causality_history: List[Dict],
                                          threshold_percentile: float = 90) -> pd.DataFrame:
        """
        레짐 전환 시점의 주요 동인 분석
        
        Parameters:
        -----------
        structure_history : List[Dict]
            구조 벡터 히스토리
        causality_history : List[Dict]
            인과 행렬 히스토리
        threshold_percentile : float
            레짐 전환으로 간주할 구조 이동량 백분위 (기본값: 상위 10%)
            
        Returns:
        --------
        pd.DataFrame : 레짐 전환 시점별 주요 동인
        """
        # 구조 이동량 계산
        movements = []
        for i in range(1, len(structure_history)):
            prev_vec = structure_history[i-1]['structure_vector']
            curr_vec = structure_history[i]['structure_vector']
            movement = np.linalg.norm(curr_vec - prev_vec)
            movements.append({
                'date': structure_history[i]['date'],
                'movement': movement,
                'index': i
            })
        
        movements_df = pd.DataFrame(movements)
        threshold = movements_df['movement'].quantile(threshold_percentile / 100)
        
        # 레짐 전환 시점 추출
        regime_transitions = movements_df[movements_df['movement'] >= threshold]
        
        # 각 전환 시점의 주요 동인 분석
        transition_drivers = []
        for _, row in regime_transitions.iterrows():
            idx = row['index']
            
            contrib = self.decompose_structure_movement(
                causality_history[idx]['causality_matrix'],
                causality_history[idx-1]['causality_matrix'],
                structure_history[idx]['structure_vector'],
                structure_history[idx-1]['structure_vector']
            )
            
            # 상위 3개 자산
            top3 = contrib['top_contributors'][:3]
            
            transition_drivers.append({
                'date': row['date'],
                'movement': row['movement'],
                'top_driver_1': top3[0]['asset'] if len(top3) > 0 else None,
                'top_driver_1_change': top3[0]['total_change'] if len(top3) > 0 else 0,
                'top_driver_2': top3[1]['asset'] if len(top3) > 1 else None,
                'top_driver_2_change': top3[1]['total_change'] if len(top3) > 1 else 0,
                'top_driver_3': top3[2]['asset'] if len(top3) > 2 else None,
                'top_driver_3_change': top3[2]['total_change'] if len(top3) > 2 else 0
            })
        
        return pd.DataFrame(transition_drivers)


if __name__ == "__main__":
    print("Conditional Transmission & Historical Analogue Modules - Ready")
