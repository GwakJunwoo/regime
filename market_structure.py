"""
Multi-Asset Causal Network System  
Market Structure Encoding Module
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class MarketStructureAnalyzer:
    """시장 구조 요약 및 벡터 생성"""
    
    def __init__(self, bucket_mapping: Dict[str, str]):
        """
        Parameters:
        -----------
        bucket_mapping : Dict[str, str]
            자산명 -> 버킷명 매핑
        """
        self.bucket_mapping = bucket_mapping
        self.buckets = list(set(bucket_mapping.values()))
        
    def bucket_to_bucket_influence(self, 
                                   causality_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Bucket-to-Bucket 영향도 계산
        I(B1→B2,t) = Σ w(i→j,t), i∈B1, j∈B2
        
        Returns:
        --------
        pd.DataFrame : Bucket 간 영향도 행렬
        """
        bucket_influence = pd.DataFrame(
            np.zeros((len(self.buckets), len(self.buckets))),
            index=self.buckets,
            columns=self.buckets
        )
        
        for source_bucket in self.buckets:
            for target_bucket in self.buckets:
                # 각 버킷에 속한 자산들
                source_assets = [
                    asset for asset, bucket in self.bucket_mapping.items()
                    if bucket == source_bucket and asset in causality_matrix.index
                ]
                target_assets = [
                    asset for asset, bucket in self.bucket_mapping.items()
                    if bucket == target_bucket and asset in causality_matrix.columns
                ]
                
                if len(source_assets) > 0 and len(target_assets) > 0:
                    # 영향도 합계
                    influence = causality_matrix.loc[source_assets, target_assets].values.sum()
                    bucket_influence.loc[source_bucket, target_bucket] = influence
        
        return bucket_influence
    
    def compute_network_metrics(self, 
                                causality_matrix: pd.DataFrame) -> Dict[str, float]:
        """
        네트워크 메트릭 계산
        
        Returns:
        --------
        Dict : 네트워크 특성 지표들
        """
        n_assets = len(causality_matrix)
        total_edges = n_assets * (n_assets - 1)
        
        # Network Density: 유의한 연결 비율
        significant_edges = (causality_matrix > 0).sum().sum()
        density = significant_edges / total_edges if total_edges > 0 else 0
        
        # Source Concentration: 영향력이 집중된 정도
        out_strength = causality_matrix.sum(axis=1)  # 각 자산의 out-degree
        source_concentration = out_strength.std() / (out_strength.mean() + 1e-10)
        
        # Average Influence: 평균 영향력
        avg_influence = causality_matrix[causality_matrix > 0].mean().mean()
        if pd.isna(avg_influence):
            avg_influence = 0
        
        # Max Influence: 최대 영향력
        max_influence = causality_matrix.max().max()
        
        return {
            'density': density,
            'source_concentration': source_concentration,
            'avg_influence': avg_influence,
            'max_influence': max_influence,
            'n_significant_edges': significant_edges
        }
    
    def create_market_structure_vector(self,
                                      causality_matrix: pd.DataFrame) -> np.ndarray:
        """
        Market Structure Vector 생성
        
        s(t) = [
            Risk→Risk,
            Risk→Safe,
            Risk→Rates,
            Rates→Risk,
            Rates→Safe,
            Safe→Risk,
            Network Density,
            Source Concentration
        ]
        
        Returns:
        --------
        np.ndarray : Market structure vector
        """
        # Bucket-to-Bucket 영향도
        bucket_influence = self.bucket_to_bucket_influence(causality_matrix)
        
        # 네트워크 메트릭
        metrics = self.compute_network_metrics(causality_matrix)
        
        # Vector 구성
        vector_components = []
        
        # 주요 Bucket 간 영향도 (존재하는 것만)
        bucket_pairs = [
            ('Risk', 'Risk'),
            ('Risk', 'Safe Haven'),
            ('Risk', 'Rates'),
            ('Rates', 'Risk'),
            ('Rates', 'Safe Haven'),
            ('Safe Haven', 'Risk'),
            ('FX', 'Risk'),
            ('Commodities', 'Risk')
        ]
        
        for source, target in bucket_pairs:
            if source in bucket_influence.index and target in bucket_influence.columns:
                vector_components.append(bucket_influence.loc[source, target])
            else:
                vector_components.append(0.0)
        
        # 네트워크 메트릭 추가
        vector_components.extend([
            metrics['density'],
            metrics['source_concentration'],
            metrics['avg_influence']
        ])
        
        return np.array(vector_components)
    
    def identify_key_sources(self,
                            causality_matrix: pd.DataFrame,
                            top_k: int = 5) -> List[Tuple[str, float]]:
        """
        주요 Source 자산 식별 (영향력이 큰 자산)
        
        Returns:
        --------
        List[Tuple[str, float]] : (자산명, 총 영향력)
        """
        out_strength = causality_matrix.sum(axis=1)
        top_sources = out_strength.nlargest(top_k)
        
        return [(asset, strength) for asset, strength in top_sources.items()]
    
    def analyze_transmission_pathways(self,
                                     causality_matrix: pd.DataFrame,
                                     threshold: float = 0) -> Dict:
        """
        전이 경로 분석
        
        Returns:
        --------
        Dict : 주요 전이 경로 정보
        """
        # Risk → Safe 전이 강도
        risk_to_safe = 0
        safe_to_risk = 0
        
        for source_asset, source_bucket in self.bucket_mapping.items():
            for target_asset, target_bucket in self.bucket_mapping.items():
                if source_asset in causality_matrix.index and target_asset in causality_matrix.columns:
                    weight = causality_matrix.loc[source_asset, target_asset]
                    
                    if source_bucket == 'Risk' and target_bucket == 'Safe Haven':
                        risk_to_safe += weight
                    elif source_bucket == 'Safe Haven' and target_bucket == 'Risk':
                        safe_to_risk += weight
        
        # 가장 강한 연결 찾기
        strong_connections = []
        for i, source in enumerate(causality_matrix.index):
            for j, target in enumerate(causality_matrix.columns):
                weight = causality_matrix.iloc[i, j]
                if weight > threshold:
                    strong_connections.append({
                        'source': source,
                        'target': target,
                        'weight': weight,
                        'source_bucket': self.bucket_mapping.get(source, 'Unknown'),
                        'target_bucket': self.bucket_mapping.get(target, 'Unknown')
                    })
        
        # 강도 순으로 정렬
        strong_connections.sort(key=lambda x: x['weight'], reverse=True)
        
        return {
            'risk_to_safe_strength': risk_to_safe,
            'safe_to_risk_strength': safe_to_risk,
            'risk_safe_ratio': risk_to_safe / (safe_to_risk + 1e-10),
            'top_connections': strong_connections[:20]
        }
    
    def summarize_current_structure(self,
                                   causality_matrix: pd.DataFrame,
                                   date: pd.Timestamp = None) -> Dict:
        """
        현재 시장 구조 종합 요약
        
        Returns:
        --------
        Dict : 시장 구조 요약 정보
        """
        bucket_influence = self.bucket_to_bucket_influence(causality_matrix)
        metrics = self.compute_network_metrics(causality_matrix)
        key_sources = self.identify_key_sources(causality_matrix)
        pathways = self.analyze_transmission_pathways(causality_matrix)
        structure_vector = self.create_market_structure_vector(causality_matrix)
        
        summary = {
            'date': date,
            'bucket_influence': bucket_influence,
            'network_metrics': metrics,
            'key_sources': key_sources,
            'transmission_pathways': pathways,
            'structure_vector': structure_vector
        }
        
        return summary
    
    @staticmethod
    def calculate_structure_movement(structure_t: np.ndarray, 
                                     structure_t_minus_1: np.ndarray) -> float:
        """
        구조 이동량 계산: ||structure_t - structure_{t-1}||
        시장 불안정도의 핵심 지표
        
        Parameters:
        -----------
        structure_t : np.ndarray
            현재 시점 구조 벡터
        structure_t_minus_1 : np.ndarray
            이전 시점 구조 벡터
            
        Returns:
        --------
        float : 유클리디안 거리 (구조 이동량)
        """
        return np.linalg.norm(structure_t - structure_t_minus_1)
    
    @staticmethod
    def calculate_rolling_instability(structure_history: List[Dict],
                                      window: int = 20) -> pd.Series:
        """
        롤링 윈도우 기반 시장 불안정도 계산
        
        Parameters:
        -----------
        structure_history : List[Dict]
            구조 벡터 히스토리 (각 dict는 'date', 'structure_vector' 포함)
        window : int
            롤링 윈도우 크기 (기본값: 20일)
            
        Returns:
        --------
        pd.Series : 시간별 불안정도 (구조 이동량의 롤링 평균)
        """
        movements = []
        dates = []
        
        for i in range(1, len(structure_history)):
            prev_vec = structure_history[i-1]['structure_vector']
            curr_vec = structure_history[i]['structure_vector']
            movement = MarketStructureAnalyzer.calculate_structure_movement(
                curr_vec, prev_vec
            )
            movements.append(movement)
            dates.append(structure_history[i]['date'])
        
        movement_series = pd.Series(movements, index=dates)
        rolling_instability = movement_series.rolling(window=window, min_periods=1).mean()
        
        return rolling_instability
    
    @staticmethod
    def calculate_structure_distance(structure_t: np.ndarray,
                                     structure_past: np.ndarray) -> float:
        """
        과거 구조와의 거리 계산
        유사 국면 탐색에 활용
        
        Parameters:
        -----------
        structure_t : np.ndarray
            현재 시점 구조 벡터
        structure_past : np.ndarray
            과거 시점 구조 벡터
            
        Returns:
        --------
        float : 유클리디안 거리
        """
        return np.linalg.norm(structure_t - structure_past)


if __name__ == "__main__":
    print("Market Structure Analyzer Module - Ready")
