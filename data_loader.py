"""
Multi-Asset Causal Network System
Data Loading and Preprocessing Module
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


class AssetBucket:
    """자산 버킷 정의 (shock-consistent grouping)"""
    
    # 버킷 정의 (실제 데이터 기반)
    RISK_ASSETS = ['코스피200 연결']  # Risk Assets: Equity
    RATES = [
        '3년국채 연결', 
        '10년국채 연결',
        '2년 T-NOTE선물 (연결선물)',
        '10년 T-NOTE선물 (연결선물)'
    ]  # Rates: Government Bonds
    SAFE_HAVEN = ['금 2026-2 (연결선물)', '엔 연결']  # Safe Haven: Gold, JPY
    FX = ['미국달러 연결', '유로 연결', '위안 연결']  # FX: Major currencies
    COMMODITIES = ['WTI 원유 (연결선물)']  # Commodities
    
    @classmethod
    def get_bucket_mapping(cls) -> Dict[str, str]:
        """자산명 -> 버킷명 매핑 반환"""
        mapping = {}
        for asset in cls.RISK_ASSETS:
            mapping[asset] = 'Risk'
        for asset in cls.RATES:
            mapping[asset] = 'Rates'
        for asset in cls.SAFE_HAVEN:
            mapping[asset] = 'Safe Haven'
        for asset in cls.FX:
            mapping[asset] = 'FX'
        for asset in cls.COMMODITIES:
            mapping[asset] = 'Commodities'
        return mapping
    
    @classmethod
    def get_bucket_assets(cls, bucket_name: str) -> List[str]:
        """버킷명 -> 자산 리스트 반환"""
        if bucket_name == 'Risk':
            return cls.RISK_ASSETS
        elif bucket_name == 'Rates':
            return cls.RATES
        elif bucket_name == 'Safe Haven':
            return cls.SAFE_HAVEN
        elif bucket_name == 'FX':
            return cls.FX
        elif bucket_name == 'Commodities':
            return cls.COMMODITIES
        return []


class DataLoader:
    """가격 데이터 로딩 및 전처리"""
    
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.raw_data = None
        self.prices = None
        self.returns = None
        self.z_scores = None
        self.demeaned_z = None
        self.asset_names = None
        
    def load_csv(self, encoding: str = 'cp949') -> pd.DataFrame:
        """CSV 파일 로드 및 파싱"""
        print(f"Loading data from {self.csv_path}...")
        
        try:
            # 첫 번째 줄에서 자산명 추출
            df_assets = pd.read_csv(self.csv_path, encoding=encoding, nrows=1, low_memory=False)
            
            asset_names = []
            asset_indices = []
            
            for i in range(0, len(df_assets.columns), 8):
                if i < len(df_assets.columns):
                    asset_name = df_assets.iloc[0, i]
                    if pd.notna(asset_name) and asset_name != '':
                        asset_names.append(asset_name)
                        asset_indices.append(i)
            
            self.asset_names = asset_names
            print(f"Found {len(asset_names)} assets: {asset_names}")
            
            # 실제 데이터 로드 (skiprows=2)
            df = pd.read_csv(self.csv_path, encoding=encoding, skiprows=2, low_memory=False)
            
            # 날짜 컬럼과 종가 데이터 추출
            date_col = df.columns[0]
            prices_data = {'Date': pd.to_datetime(df[date_col], errors='coerce')}
            
            for idx, asset_name in enumerate(asset_names):
                col_idx = asset_indices[idx] + 4  # 현재가 컬럼
                if col_idx < len(df.columns):
                    close_col = df.columns[col_idx]
                    prices_data[asset_name] = pd.to_numeric(df[close_col], errors='coerce')
            
            prices_df = pd.DataFrame(prices_data)
            prices_df = prices_df.set_index('Date')
            prices_df = prices_df.dropna(how='all')  # 모든 값이 NaN인 행 제거
            
            self.raw_data = prices_df
            print(f"Data loaded: {prices_df.shape[0]} rows, {prices_df.shape[1]} columns")
            print(f"Date range: {prices_df.index.min()} to {prices_df.index.max()}")
            
            return prices_df
            
        except Exception as e:
            print(f"Error loading CSV: {e}")
            raise ValueError(f"Could not load CSV: {e}")
    
    def parse_prices(self) -> pd.DataFrame:
        """
        가격 데이터 파싱 및 정리
        각 자산의 종가(Close) 데이터 추출
        """
        if self.raw_data is None:
            raise ValueError("Data not loaded. Call load_csv() first.")
        
        print("\n=== Parsing price data ===")
        self.prices = self.raw_data
        print(f"Prices ready: {self.prices.shape}")
        
        return self.prices
    
    def calculate_returns(self, price_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        로그 수익률 계산
        r(i,t) = log(P(i,t)) - log(P(i,t-1))
        """
        if price_df is None:
            price_df = self.prices if self.prices is not None else self.raw_data
        
        if price_df is None:
            raise ValueError("No price data available")
        
        print("\n=== Calculating log returns ===")
        
        returns = pd.DataFrame(index=price_df.index, columns=price_df.columns)
        
        for col in price_df.columns:
            prices = price_df[col].replace(0, np.nan)  # 0 제거
            log_prices = np.log(prices)
            returns[col] = log_prices.diff()
        
        self.returns = returns
        print(f"Returns calculated for {len(returns.columns)} assets")
        
        return returns
    
    def calculate_volatility_normalized_returns(self, 
                                                returns_df: pd.DataFrame,
                                                window: int = 20) -> pd.DataFrame:
        """
        변동성 정규화
        z(i,t) = r(i,t) / σ(i,t)
        σ: rolling volatility
        """
        print(f"\n=== Normalizing by rolling volatility (window={window}) ===")
        
        z_scores = pd.DataFrame(index=returns_df.index, columns=returns_df.columns)
        
        for col in returns_df.columns:
            returns = returns_df[col]
            rolling_std = returns.rolling(window=window, min_periods=window//2).std()
            z_scores[col] = returns / rolling_std
        
        self.z_scores = z_scores
        print(f"Volatility normalization completed")
        
        return z_scores
    
    def bucket_wise_demean(self, z_scores_df: pd.DataFrame,
                          bucket_mapping: Dict[str, str]) -> pd.DataFrame:
        """
        Bucket-wise Cross-sectional Demeaning
        ẑ(i,t) = z(i,t) - mean(z(j,t)), j ∈ same bucket
        
        버킷 내부 공통 shock 제거, 상대적 반응만 유지
        """
        print("\n=== Applying bucket-wise cross-sectional demeaning ===")
        
        demeaned = z_scores_df.copy()
        
        # 버킷별로 평균 제거
        for bucket_name in set(bucket_mapping.values()):
            # 해당 버킷의 자산들
            bucket_assets = [asset for asset, bucket in bucket_mapping.items() 
                           if bucket == bucket_name and asset in z_scores_df.columns]
            
            if len(bucket_assets) > 0:
                # 버킷 내 평균 계산
                bucket_mean = z_scores_df[bucket_assets].mean(axis=1)
                
                # 각 자산에서 버킷 평균 제거
                for asset in bucket_assets:
                    demeaned[asset] = z_scores_df[asset] - bucket_mean
                
                print(f"  {bucket_name}: {len(bucket_assets)} assets demeaned")
        
        self.demeaned_z = demeaned
        print(f"Demeaning completed")
        
        return demeaned
    
    def preprocess_pipeline(self, 
                           vol_window: int = 20,
                           bucket_mapping: Optional[Dict[str, str]] = None,
                           use_demeaning: bool = False) -> pd.DataFrame:
        """
        전체 전처리 파이프라인 실행
        1. 가격 데이터 파싱
        2. 수익률 계산
        3. 변동성 정규화
        4. (Optional) Bucket-wise demeaning
        
        Parameters:
        -----------
        use_demeaning : bool
            False (기본값) - Volatility normalization만 사용 (권장)
            True - Bucket-wise demeaning 추가 적용
        """
        print("\n" + "="*60)
        print("PREPROCESSING PIPELINE")
        print("="*60)
        
        # 1. Parse prices
        price_df = self.parse_prices()
        
        # 2. Calculate returns
        returns_df = self.calculate_returns(price_df)
        
        # 3. Volatility normalization
        z_scores_df = self.calculate_volatility_normalized_returns(returns_df, window=vol_window)
        
        # 4. Bucket-wise demeaning (선택적)
        if use_demeaning and bucket_mapping is not None:
            print("\n⚠️  WARNING: Using bucket-wise demeaning")
            print("   This may reduce signal strength for causal network")
            demeaned_df = self.bucket_wise_demean(z_scores_df, bucket_mapping)
        else:
            print("\n✅ Using volatility-normalized returns only (recommended)")
            demeaned_df = z_scores_df
        
        print("\n" + "="*60)
        print("PREPROCESSING COMPLETED")
        print("="*60)
        
        return demeaned_df


if __name__ == "__main__":
    # 테스트 실행
    loader = DataLoader("가격 데이터.csv")
    
    # CSV 로드 및 구조 확인
    df = loader.load_csv()
