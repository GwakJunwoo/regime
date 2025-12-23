"""
Multi-Asset Causal Network System
Causal Network Modeling Module
- Granger Causality (느림, 개장시간 차이에 약함)
- Rolling Correlation + Lead-Lag (빠름, robust 권장)
- Partial Correlation (간접효과 제거)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests
from scipy import linalg
import warnings
warnings.filterwarnings('ignore')


class CausalNetworkModel:
    """다양한 방법으로 인과 네트워크 구축"""
    
    def __init__(self, max_lag: int = 3, significance_level: float = 0.05):
        """
        Parameters:
        -----------
        max_lag : int
            최대 시차 (1~3일)
        significance_level : float
            유의수준 (default: 0.05)
        """
        self.max_lag = max_lag
        self.significance_level = significance_level
        self.network_history = []
        
    def granger_causality_matrix(self, 
                                 data: pd.DataFrame, 
                                 max_lag: Optional[int] = None) -> pd.DataFrame:
        """
        모든 자산 쌍에 대해 Granger Causality 테스트 수행
        
        Returns:
        --------
        pd.DataFrame : Causality matrix (i → j)
            값 = F-statistic (유의하면) 또는 0 (유의하지 않으면)
        """
        if max_lag is None:
            max_lag = self.max_lag
        
        assets = data.columns.tolist()
        n_assets = len(assets)
        
        # Causality matrix 초기화
        causality_matrix = pd.DataFrame(
            np.zeros((n_assets, n_assets)),
            index=assets,
            columns=assets
        )
        
        # 각 자산 쌍에 대해 Granger Causality 테스트
        for i, cause in enumerate(assets):
            for j, effect in enumerate(assets):
                if i == j:
                    continue  # 자기 자신은 제외
                
                try:
                    # 데이터 준비 (effect ~ cause)
                    test_data = data[[effect, cause]].dropna()
                    
                    if len(test_data) < max_lag + 10:
                        continue
                    
                    # Granger Causality Test
                    result = grangercausalitytests(
                        test_data, 
                        maxlag=max_lag, 
                        verbose=False
                    )
                    
                    # 각 lag의 p-value 확인
                    min_p_value = 1.0
                    best_f_stat = 0.0
                    
                    for lag in range(1, max_lag + 1):
                        # ssr_ftest 사용
                        p_value = result[lag][0]['ssr_ftest'][1]
                        f_stat = result[lag][0]['ssr_ftest'][0]
                        
                        if p_value < min_p_value:
                            min_p_value = p_value
                            best_f_stat = f_stat
                    
                    # 유의한 경우 F-statistic 저장
                    if min_p_value < self.significance_level:
                        causality_matrix.loc[cause, effect] = best_f_stat
                
                except Exception as e:
                    # print(f"Warning: Granger test failed for {cause} -> {effect}: {e}")
                    continue
        
        return causality_matrix
    
    def var_based_causality(self, 
                           data: pd.DataFrame,
                           lag_order: Optional[int] = None) -> pd.DataFrame:
        """
        VAR 모델 기반 Granger Causality
        더 효율적인 방법 (한 번에 모든 관계 추정)
        
        Returns:
        --------
        pd.DataFrame : Causality weights matrix W(t)
        """
        if lag_order is None:
            lag_order = self.max_lag
        
        # 결측치 제거
        clean_data = data.dropna()
        
        if len(clean_data) < lag_order + 20:
            print(f"Warning: Insufficient data ({len(clean_data)} rows)")
            return pd.DataFrame(
                np.zeros((len(data.columns), len(data.columns))),
                index=data.columns,
                columns=data.columns
            )
        
        try:
            # VAR 모델 추정
            model = VAR(clean_data)
            
            # AIC 기준으로 최적 lag 선택 (최대 max_lag까지)
            lag_selection = model.select_order(maxlags=lag_order)
            optimal_lag = min(lag_selection.aic, lag_order)
            
            # VAR 모델 적합
            fitted_model = model.fit(optimal_lag)
            
            # 계수 행렬에서 causality weight 추출
            # params shape: (n_vars, n_vars * lag + 1)
            params = fitted_model.params.iloc[:, :-1]  # const 제외
            
            # 각 lag의 계수 행렬을 평균 (또는 합)
            n_vars = len(data.columns)
            weight_matrix = np.zeros((n_vars, n_vars))
            
            for lag in range(optimal_lag):
                lag_coeffs = params.iloc[:, lag*n_vars:(lag+1)*n_vars].values
                # 절대값의 평균 사용 (영향력의 크기)
                weight_matrix += np.abs(lag_coeffs.T)
            
            weight_matrix /= optimal_lag  # 평균
            
            # DataFrame으로 변환
            causality_weights = pd.DataFrame(
                weight_matrix,
                index=data.columns,
                columns=data.columns
            )
            
            # 대각선 0으로 설정 (자기 자신 제외)
            np.fill_diagonal(causality_weights.values, 0)
            
            return causality_weights
        
        except Exception as e:
            print(f"Warning: VAR model failed: {e}")
            return pd.DataFrame(
                np.zeros((len(data.columns), len(data.columns))),
                index=data.columns,
                columns=data.columns
            )
    
    def correlation_network(self, 
                           data: pd.DataFrame,
                           use_lag: bool = True) -> pd.DataFrame:
        """
        Rolling Correlation 네트워크 (빠르고 robust!)
        
        개장시간 차이가 있는 글로벌 자산에 최적
        Granger보다 100배 빠름
        
        Parameters:
        -----------
        use_lag : bool
            True - lead-lag 상관계수 (X가 Y를 선행)
            False - 동시 상관계수
        """
        assets = data.columns
        n_assets = len(assets)
        
        network = pd.DataFrame(
            np.zeros((n_assets, n_assets)),
            index=assets,
            columns=assets
        )
        
        for source in assets:
            for target in assets:
                if source == target:
                    continue
                
                x = data[source].dropna()
                y = data[target].dropna()
                
                # 공통 인덱스
                common_idx = x.index.intersection(y.index)
                if len(common_idx) < 30:
                    continue
                
                x = x.loc[common_idx]
                y = y.loc[common_idx]
                
                if use_lag:
                    # Lead-lag correlation (x → y)
                    max_corr = 0
                    for lag in range(1, self.max_lag + 1):
                        if len(x) > lag:
                            x_shifted = x.shift(lag)
                            corr = x_shifted.corr(y)
                            if not np.isnan(corr) and abs(corr) > abs(max_corr):
                                max_corr = corr
                    network.loc[source, target] = abs(max_corr)
                else:
                    # 동시 상관계수
                    corr = x.corr(y)
                    if not np.isnan(corr):
                        network.loc[source, target] = abs(corr)
        
        return network
    
    def partial_correlation_network(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Partial Correlation (편상관) 네트워크
        
        간접효과 제거: X → Z → Y 같은 경로 제거
        직접적 연결만 남김
        """
        # 결측치 제거
        clean_data = data.dropna()
        
        if len(clean_data) < 30:
            return pd.DataFrame(
                np.zeros((len(data.columns), len(data.columns))),
                index=data.columns,
                columns=data.columns
            )
        
        try:
            # 상관계수 행렬
            corr_matrix = clean_data.corr()
            
            # Precision matrix (역행렬)
            precision = linalg.inv(corr_matrix.values)
            
            # Partial correlation
            partial_corr = np.zeros_like(precision)
            for i in range(len(precision)):
                for j in range(len(precision)):
                    if i != j:
                        partial_corr[i, j] = -precision[i, j] / np.sqrt(
                            precision[i, i] * precision[j, j]
                        )
            
            result = pd.DataFrame(
                np.abs(partial_corr),  # 절대값
                index=corr_matrix.index,
                columns=corr_matrix.columns
            )
            
            return result
        
        except:
            # 역행렬 실패 시 일반 상관계수
            return clean_data.corr().abs()
    
    def rolling_causality_network(self,
                                  data: pd.DataFrame,
                                  window: int = 120,
                                  method: str = 'correlation') -> List[Dict]:
        """
        Rolling window로 시점별 네트워크 생성
        
        Parameters:
        -----------
        method : str
            'correlation' - 빠르고 robust (권장!)
            'partial' - 간접효과 제거
            'granger' - Granger causality (느림)
            'var' - VAR 기반 (multicollinearity 위험)
        """
        print(f"\n=== Computing rolling networks ===")
        print(f"Window: {window}, Method: {method.upper()}")
        
        network_history = []
        dates = data.index[window:]
        
        for i, date in enumerate(dates):
            if i % 100 == 0:
                print(f"  Progress: {i}/{len(dates)} ({i/len(dates)*100:.1f}%)")
            
            # Window 데이터 추출
            window_data = data.iloc[i:i+window]
            
            # 네트워크 계산
            if method == 'correlation':
                network = self.correlation_network(window_data, use_lag=True)
            elif method == 'partial':
                network = self.partial_correlation_network(window_data)
            elif method == 'var':
                network = self.var_based_causality(window_data)
            else:  # granger
                network = self.granger_causality_matrix(window_data)
            
            network_history.append({
                'date': date,
                'network': network
            })
        
        self.network_history = network_history
        print(f"  Progress: {len(dates)}/{len(dates)} (100.0%)")
        print(f"✓ Completed: {len(network_history)} networks")
        
        return network_history
    
    def get_network_at_date(self, date: pd.Timestamp) -> Optional[pd.DataFrame]:
        """특정 날짜의 네트워크 반환"""
        for item in self.network_history:
            if item['date'] == date:
                return item['network']
        return None
    
    def get_latest_network(self) -> Optional[Dict]:
        """가장 최근 네트워크 반환"""
        if len(self.network_history) > 0:
            return self.network_history[-1]
        return None


if __name__ == "__main__":
    # 테스트
    print("Causal Network Model Module - Ready")
