"""
Multi-Asset Causal Network System
Main System Integration
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pickle
import hashlib
import os
import warnings
warnings.filterwarnings('ignore')

from data_loader import DataLoader, AssetBucket
from causal_network import CausalNetworkModel
from market_structure import MarketStructureAnalyzer
from conditional_analysis import ConditionalTransmissionAnalyzer, HistoricalAnalogueSearch


class MultiAssetCausalSystem:
    """Multi-Asset Causal Network System - 메인 시스템"""
    
    def __init__(self, 
                 csv_path: str,
                 vol_window: int = 20,
                 causality_window: int = 120,
                 max_lag: int = 3):
        """
        Parameters:
        -----------
        csv_path : str
            가격 데이터 CSV 파일 경로
        vol_window : int
            변동성 계산 윈도우
        causality_window : int
            인과관계 분석 윈도우
        max_lag : int
            최대 시차
        """
        print("="*80)
        print("MULTI-ASSET CAUSAL NETWORK SYSTEM")
        print("="*80)
        
        self.csv_path = csv_path
        self.vol_window = vol_window
        self.causality_window = causality_window
        self.max_lag = max_lag
        
        # 컴포넌트 초기화
        self.data_loader = DataLoader(csv_path)
        self.causal_model = CausalNetworkModel(max_lag=max_lag)
        
        # 데이터 저장
        self.prices = None
        self.returns = None
        self.processed_data = None
        self.bucket_mapping = None
        
        # 분석기
        self.structure_analyzer = None
        self.transmission_analyzer = None
        self.analogue_search = HistoricalAnalogueSearch()
        
        # 결과
        self.network_history = []
        self.structure_history = []
        
    def load_and_preprocess(self):
        """데이터 로드 및 전처리"""
        print("\n" + "="*80)
        print("STEP 1: Data Loading and Preprocessing")
        print("="*80)
        
        # 1. CSV 로드
        self.prices = self.data_loader.load_csv()
        
        # 2. Bucket 매핑 생성
        self.bucket_mapping = AssetBucket.get_bucket_mapping()
        
        # 실제 데이터에 있는 자산만 필터링
        self.bucket_mapping = {
            asset: bucket for asset, bucket in self.bucket_mapping.items()
            if asset in self.prices.columns
        }
        
        print(f"\nBucket mapping:")
        for bucket in set(self.bucket_mapping.values()):
            assets = [a for a, b in self.bucket_mapping.items() if b == bucket]
            print(f"  {bucket}: {assets}")
        
        # 3. 전처리 파이프라인 (demean 제거)
        self.processed_data = self.data_loader.preprocess_pipeline(
            vol_window=self.vol_window,
            bucket_mapping=self.bucket_mapping,
            use_demeaning=False  # 신호 보존을 위해 demean 사용 안 함
        )
        
        # 4. 분석기 초기화
        self.structure_analyzer = MarketStructureAnalyzer(self.bucket_mapping)
        self.transmission_analyzer = ConditionalTransmissionAnalyzer(self.bucket_mapping)
        
        print(f"\n✓ Preprocessing completed")
        print(f"  - Processed data shape: {self.processed_data.shape}")
        print(f"  - Date range: {self.processed_data.index.min()} to {self.processed_data.index.max()}")
        
    def build_causal_networks(self, method: str = 'granger', sample_size: Optional[int] = None):
        """인과 네트워크 구축
        
        Methods:
        --------
        'granger'     - Granger causality (기본값)
        'correlation' - Rolling correlation + lead-lag (빠르고 robust)
        'partial'     - Partial correlation (간접효과 제거)
        'var'         - VAR 기반 (빠르지만 multicollinearity 위험)
        """
        print("\n" + "="*80)
        print("STEP 2: Building Causal Networks")
        print("="*80)
        print(f"Method: {method.upper()}")
        
        if method == 'correlation':
            print("✅ Using correlation + lead-lag (fast & robust)")
        elif method == 'granger':
            print("✅ Using Granger causality (statistical test)")
        
        # Rolling window로 네트워크 구축
        data_to_use = self.processed_data
        
        # 샘플링 (테스트용)
        if sample_size is not None:
            data_to_use = data_to_use.iloc[-sample_size:]
            print(f"Using sample data: last {sample_size} days")
        
        self.network_history = self.causal_model.rolling_causality_network(
            data_to_use,
            window=self.causality_window,
            method=method
        )
        
        print(f"\n✓ Network construction completed")
        print(f"  - Total time points: {len(self.network_history)}")
        
    def analyze_market_structures(self):
        """시장 구조 분석 + 구조 이동량 계산"""
        print("\n" + "="*80)
        print("STEP 3: Analyzing Market Structures")
        print("="*80)
        
        self.structure_history = []
        
        for i, network_item in enumerate(self.network_history):
            if i % 100 == 0:
                print(f"  Progress: {i}/{len(self.network_history)}")
            
            date = network_item['date']
            network = network_item['network']
            
            # 구조 요약
            summary = self.structure_analyzer.summarize_current_structure(network, date)
            
            # 구조 이동량 계산 (이전 시점이 있는 경우)
            if i > 0:
                prev_vector = self.structure_history[-1]['structure_vector']
                curr_vector = summary['structure_vector']
                movement = self.structure_analyzer.calculate_structure_movement(
                    curr_vector, prev_vector
                )
                summary['structure_movement'] = movement
            else:
                summary['structure_movement'] = 0.0
            
            self.structure_history.append(summary)
            
            # Historical analogue search용 벡터 추가
            self.analogue_search.add_structure_vector(
                date, 
                summary['structure_vector']
            )
        
        # 전체 기간 시장 불안정도 계산 (롤링 평균)
        instability_series = self.structure_analyzer.calculate_rolling_instability(
            self.structure_history, window=20
        )
        
        # 각 시점에 불안정도 추가
        for i, (date, instability) in enumerate(instability_series.items()):
            if i < len(self.structure_history):
                self.structure_history[i+1]['market_instability'] = float(instability)
        
        print(f"\n✓ Structure analysis completed")
        print(f"  - Analyzed {len(self.structure_history)} periods")
        print(f"  - Mean structure movement: {np.mean([s['structure_movement'] for s in self.structure_history]):.4f}")
        print(f"  - Max structure movement: {np.max([s['structure_movement'] for s in self.structure_history]):.4f}")
        
    def generate_daily_summary(self, date: Optional[pd.Timestamp] = None) -> Dict:
        """Daily Summary 생성"""
        if date is None:
            # 가장 최근 날짜
            if len(self.structure_history) == 0:
                print("No structure history available")
                return {}
            summary = self.structure_history[-1]
            date = summary['date']
        else:
            # 특정 날짜 찾기
            summary = None
            for s in self.structure_history:
                if s['date'] == date:
                    summary = s
                    break
            if summary is None:
                print(f"No data for date: {date}")
                return {}
        
        print("\n" + "="*80)
        print(f"DAILY SUMMARY: {date.strftime('%Y-%m-%d')}")
        print("="*80)
        
        # 1. Current Market Structure
        print("\n1. CURRENT MARKET STRUCTURE")
        print("-" * 40)
        
        print("\nNetwork Metrics:")
        for key, value in summary['network_metrics'].items():
            print(f"  - {key}: {value:.4f}")
        
        print("\nKey Source Assets (Top 5):")
        for i, (asset, strength) in enumerate(summary['key_sources'], 1):
            bucket = self.bucket_mapping.get(asset, 'Unknown')
            print(f"  {i}. {asset:30s} [{bucket:15s}] - {strength:.4f}")
        
        print("\nBucket-to-Bucket Influence:")
        print(summary['bucket_influence'])
        
        print("\nTransmission Pathways:")
        pathways = summary['transmission_pathways']
        print(f"  - Risk → Safe Haven: {pathways['risk_to_safe_strength']:.4f}")
        print(f"  - Safe Haven → Risk: {pathways['safe_to_risk_strength']:.4f}")
        print(f"  - Risk/Safe Ratio: {pathways['risk_safe_ratio']:.2f}")
        
        # 구조 이동량 정보
        if 'structure_movement' in summary:
            print(f"\nStructure Movement:")
            print(f"  - Current movement: {summary['structure_movement']:.6f}")
        if 'market_instability' in summary:
            print(f"  - Market instability (20-day avg): {summary['market_instability']:.6f}")
        
        # 2. Conditional Transmission Insight
        print("\n2. CONDITIONAL TRANSMISSION INSIGHT")
        print("-" * 40)
        
        # 주요 자산들에 대한 조건부 분석
        current_network = None
        for net in self.network_history:
            if net['date'] == date:
                current_network = net['network']
                break
        
        if current_network is not None:
            key_assets = [asset for asset, _ in summary['key_sources'][:3]]
            
            for asset in key_assets:
                impacts = self.transmission_analyzer.analyze_conditional_impact(
                    current_network, asset, top_k=5
                )
                
                print(f"\nIf '{asset}' moves:")
                for i, impact in enumerate(impacts, 1):
                    print(f"  {i}. {impact['asset']:30s} [{impact['bucket']:15s}] "
                          f"- Weight: {impact['weight']:.4f} "
                          f"({impact['relative_strength']*100:.1f}%)")
        
        # 3. Historical Analogues
        print("\n3. HISTORICAL ANALOGUES")
        print("-" * 40)
        
        similar_periods = self.analogue_search.find_similar_periods(
            summary['structure_vector'],
            top_k=5
        )
        
        if len(similar_periods) > 0:
            print("\nTop 5 Similar Periods:")
            for period in similar_periods:
                print(f"  {period['rank']}. {period['date'].strftime('%Y-%m-%d')} "
                      f"- Similarity: {period['similarity']:.4f}")
            
            # 유사 기간 이후의 자산 반응
            if self.prices is not None:
                comparison = self.analogue_search.compare_analogues(
                    similar_periods,
                    self.prices,
                    forward_days=20
                )
                
                if len(comparison) > 0:
                    print("\nAsset Performance After Similar Periods (20-day forward):")
                    # 자산별 평균 수익률
                    asset_cols = [col for col in comparison.columns 
                                 if col not in ['date', 'similarity', 'rank']]
                    avg_returns = comparison[asset_cols].mean()
                    avg_returns_sorted = avg_returns.sort_values(ascending=False)
                    
                    print("\nAverage Returns:")
                    for asset, ret in avg_returns_sorted.items():
                        if asset in self.bucket_mapping:
                            bucket = self.bucket_mapping[asset]
                            print(f"  {asset:30s} [{bucket:15s}] - {ret:+.2f}%")
        else:
            print("  No similar periods found in history")
        
        print("\n" + "="*80)
        
        return {
            'date': date,
            'current_structure': summary,
            'similar_periods': similar_periods
        }
    
    def visualize_network(self, date: Optional[pd.Timestamp] = None, save_path: Optional[str] = None):
        """네트워크 시각화"""
        if date is None and len(self.network_history) > 0:
            date = self.network_history[-1]['date']
        
        # 해당 날짜의 네트워크 찾기
        network = None
        for net in self.network_history:
            if net['date'] == date:
                network = net['network']
                break
        
        if network is None:
            print(f"No network found for date: {date}")
            return
        
        # Heatmap 생성
        plt.figure(figsize=(14, 12))
        
        # 자산명을 버킷과 함께 표시
        labels = [f"{asset}\n[{self.bucket_mapping.get(asset, '?')}]" 
                 for asset in network.index]
        
        sns.heatmap(
            network.values,
            annot=False,
            cmap='YlOrRd',
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={'label': 'Causality Weight'}
        )
        
        plt.title(f"Causal Network - {date.strftime('%Y-%m-%d')}", fontsize=16, pad=20)
        plt.xlabel("Target Assets (Influenced)", fontsize=12)
        plt.ylabel("Source Assets (Influencing)", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Network visualization saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def run_full_analysis(self, sample_size: Optional[int] = None):
        """전체 분석 파이프라인 실행"""
        # 1. 데이터 로드 및 전처리
        self.load_and_preprocess()
        
        # 2. 인과 네트워크 구축
        self.build_causal_networks(method='var', sample_size=sample_size)
        
        # 3. 시장 구조 분석
        self.analyze_market_structures()
        
        # 4. 최종 요약 생성
        summary = self.generate_daily_summary()
        
        return summary
    
    def save_system(self, cache_path: str = "./results/system_cache.pkl"):
        """시스템 상태 전체 저장 (빠른 로딩용)"""
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        
        # 저장할 데이터
        cache_data = {
            'csv_path': self.csv_path,
            'vol_window': self.vol_window,
            'causality_window': self.causality_window,
            'max_lag': self.max_lag,
            'prices': self.prices,
            'processed_data': self.processed_data,
            'bucket_mapping': self.bucket_mapping,
            'network_history': self.network_history,
            'structure_history': self.structure_history,
            'timestamp': datetime.now().isoformat()
        }
        
        # 데이터 해시 (무효화 감지용)
        if self.prices is not None:
            data_hash = hashlib.md5(
                pd.util.hash_pandas_object(self.prices).values
            ).hexdigest()
            cache_data['data_hash'] = data_hash
        
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        
        print(f"✓ System cached to: {cache_path}")
        return cache_data['timestamp']
    
    @classmethod
    def load_system(cls, cache_path: str = "./results/system_cache.pkl"):
        """저장된 시스템 상태 로드"""
        if not os.path.exists(cache_path):
            return None
        
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        # 시스템 재구성
        system = cls(
            csv_path=cache_data['csv_path'],
            vol_window=cache_data['vol_window'],
            causality_window=cache_data['causality_window'],
            max_lag=cache_data['max_lag']
        )
        
        # 데이터 복원
        system.prices = cache_data['prices']
        system.processed_data = cache_data['processed_data']
        system.bucket_mapping = cache_data['bucket_mapping']
        system.network_history = cache_data['network_history']
        system.structure_history = cache_data['structure_history']
        
        # 분석기 재초기화
        if system.bucket_mapping:
            system.structure_analyzer = MarketStructureAnalyzer(system.bucket_mapping)
            system.transmission_analyzer = ConditionalTransmissionAnalyzer(system.bucket_mapping)
        
        print(f"✓ System loaded from cache (saved: {cache_data['timestamp']})")
        return system, cache_data.get('data_hash')
    
    def save_results(self, output_dir: str = "./results"):
        """결과 저장"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Network history
        network_df_list = []
        for net in self.network_history:
            date = net['date']
            network = net['network']
            # Flatten network
            for source in network.index:
                for target in network.columns:
                    weight = network.loc[source, target]
                    if weight > 0:
                        network_df_list.append({
                            'date': date,
                            'source': source,
                            'target': target,
                            'weight': weight
                        })
        
        if len(network_df_list) > 0:
            network_df = pd.DataFrame(network_df_list)
            network_df.to_csv(f"{output_dir}/network_history.csv", index=False)
            print(f"✓ Network history saved to: {output_dir}/network_history.csv")
        
        # 2. Structure vectors
        structure_data = []
        for struct in self.structure_history:
            row = {'date': struct['date']}
            row.update({f'v{i}': v for i, v in enumerate(struct['structure_vector'])})
            structure_data.append(row)
        
        if len(structure_data) > 0:
            structure_df = pd.DataFrame(structure_data)
            structure_df.to_csv(f"{output_dir}/structure_vectors.csv", index=False)
            print(f"✓ Structure vectors saved to: {output_dir}/structure_vectors.csv")
        
        # 3. 전처리된 데이터
        if self.processed_data is not None:
            self.processed_data.to_csv(f"{output_dir}/processed_data.csv")
            print(f"✓ Processed data saved to: {output_dir}/processed_data.csv")


if __name__ == "__main__":
    # 시스템 실행
    system = MultiAssetCausalSystem(
        csv_path="가격 데이터.csv",
        vol_window=20,
        causality_window=120,
        max_lag=3
    )
    
    # 전체 분석 실행 (테스트: 최근 500일만)
    summary = system.run_full_analysis(sample_size=500)
    
    # 결과 저장
    system.save_results("./results")
    
    # 네트워크 시각화
    system.visualize_network(save_path="./results/latest_network.png")
    
    print("\n" + "="*80)
    print("✓ ANALYSIS COMPLETE")
    print("="*80)
