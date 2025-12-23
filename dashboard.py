"""
Multi-Asset Causal Network System
Interactive Dashboard (Streamlit)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

from streamlit_agraph import agraph, Node, Edge, Config
from main_system import MultiAssetCausalSystem

# 한글 폰트 설정
def set_korean_font():
    """한글 폰트 설정"""
    try:
        # Windows
        plt.rcParams['font.family'] = 'Malgun Gothic'
    except:
        try:
            # Mac
            plt.rcParams['font.family'] = 'AppleGothic'
        except:
            # 기본 폰트
            plt.rcParams['font.family'] = 'DejaVu Sans'
    
    plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

set_korean_font()

# 페이지 설정
st.set_page_config(
    page_title="Multi-Asset Causal Network Dashboard",
    page_icon="�",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 스타일
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_system(force_retrain=False):
    """시스템 로드 및 초기화 (캐싱)"""
    cache_path = "./results/system_cache.pkl"
    
    # 캐시에서 로드 시도
    if not force_retrain:
        try:
            result = MultiAssetCausalSystem.load_system(cache_path)
            if result is not None:
                system, data_hash = result
                st.success(f"캐시된 시스템 로드 완료!")
                return system
        except Exception as e:
            st.warning(f"캐시 로드 실패: {e}. 재학습합니다...")
    
    # 캐시 없으면 새로 학습
    with st.spinner('시스템 학습 중... (최초 실행 또는 재학습)'):
        system = MultiAssetCausalSystem(
            csv_path="가격 데이터.csv",
            vol_window=20,
            causality_window=120,
            max_lag=3
        )
        
        # 데이터 로드 (demean=False로 신호 보존)
        system.load_and_preprocess()
        
        # 네트워크 구축 (Granger 방식 사용, 최근 1000일)
        system.build_causal_networks(method='granger', sample_size=1000)
        
        # 구조 분석
        system.analyze_market_structures()
        
        # 캐시 저장
        system.save_system(cache_path)
        st.success(f"시스템 학습 및 캐시 저장 완료!")
        
    return system


def plot_network_heatmap(network, title="Causal Network", bucket_mapping=None):
    """네트워크 히트맵 플롯 (한글 지원)"""
    set_korean_font()  # 폰트 재설정
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 자산명에 버킷 추가
    if bucket_mapping:
        labels = [f"{asset}\n[{bucket_mapping.get(asset, '?')}]" 
                 for asset in network.index]
    else:
        labels = network.index
    
    sns.heatmap(
        network.values,
        annot=False,
        cmap='YlOrRd',
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Causality Weight'},
        ax=ax
    )
    
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel("영향받는 자산 (Target)", fontsize=12)
    ax.set_ylabel("영향주는 자산 (Source)", fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    
    return fig


def plot_network_graph(network, bucket_mapping=None, threshold=0.01):
    """네트워크 그래프 (streamlit-agraph)"""
    # 버킷별 색상
    bucket_colors = {
        'Risk': '#e74c3c',
        'Rates': '#3498db',
        'Safe Haven': '#f39c12',
        'FX': '#9b59b6',
        'Commodities': '#1abc9c',
        'Unknown': '#95a5a6'
    }
    
    # 노드 생성
    nodes = []
    for asset in network.index:
        bucket = bucket_mapping.get(asset, 'Unknown') if bucket_mapping else 'Unknown'
        color = bucket_colors.get(bucket, '#95a5a6')
        
        nodes.append(
            Node(
                id=asset,
                label=asset,
                size=25,
                color=color,
                title=f"{asset}\n[{bucket}]"
            )
        )
    
    # 엣지 생성 (threshold 이상만)
    edges = []
    max_weight = 0
    
    for source in network.index:
        for target in network.columns:
            weight = network.loc[source, target]
            if weight > threshold and source != target:
                edges.append(
                    Edge(
                        source=source,
                        target=target,
                        type="CURVE_SMOOTH",
                        color="#95a5a6",
                        width=min(weight * 5, 3)  # 최대 너비 제한
                    )
                )
                if weight > max_weight:
                    max_weight = weight
    
    if len(edges) == 0:
        st.warning("유의한 연결이 없습니다. Threshold를 낮춰보세요.")
        return None
    
    # Config 설정
    config = Config(
        width=900,
        height=700,
        directed=True,
        physics=True,
        hierarchical=False,
        nodeHighlightBehavior=True,
        highlightColor="#F7A7A6",
        collapsible=False,
        node={'labelProperty': 'label'},
        link={'labelProperty': 'label', 'renderLabel': False}
    )
    
    return nodes, edges, config


def plot_bucket_influence(bucket_influence):
    """Bucket-to-Bucket 영향도 시각화"""
    set_korean_font()  # 폰트 재설정
    
    fig = px.imshow(
        bucket_influence,
        labels=dict(x="Target Bucket", y="Source Bucket", color="Influence"),
        x=bucket_influence.columns,
        y=bucket_influence.index,
        color_continuous_scale='Reds',
        aspect="auto"
    )
    
    fig.update_layout(
        title="Bucket-to-Bucket Influence",
        height=400
    )
    
    return fig


def plot_metrics_timeseries(structure_history):
    """시장 구조 메트릭 시계열"""
    dates = [pd.Timestamp(s['date']).to_pydatetime() for s in structure_history]
    density = [s['network_metrics']['density'] for s in structure_history]
    concentration = [s['network_metrics']['source_concentration'] for s in structure_history]
    avg_influence = [s['network_metrics']['avg_influence'] for s in structure_history]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates, y=density,
        name='Network Density',
        mode='lines',
        line=dict(color='#3498db', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=dates, y=concentration,
        name='Source Concentration',
        mode='lines',
        line=dict(color='#e74c3c', width=2),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title="Market Structure Metrics Over Time",
        xaxis_title="Date",
        yaxis_title="Density",
        yaxis2=dict(
            title="Concentration",
            overlaying='y',
            side='right'
        ),
        hovermode='x unified',
        height=400
    )
    
    return fig


# ============================================================================
# 메인 앱
# ============================================================================

def main():
    # 헤더
    st.markdown('<div class="main-header">Multi-Asset Causal Network Dashboard</div>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # 시스템 로드
    try:
        system = load_system()
    except Exception as e:
        st.error(f"시스템 로드 실패: {e}")
        st.stop()
    
    # 사이드바
    st.sidebar.title("설정")
    
    # 재학습 버튼
    st.sidebar.markdown("---")
    st.sidebar.subheader("시스템 관리")
    
    if st.sidebar.button("전체 재학습", help="모든 데이터를 다시 학습합니다 (시간 소요)"):
        st.cache_resource.clear()  # 캐시 초기화
        st.rerun()
    
    # 캐시 정보 표시
    cache_path = "./results/system_cache.pkl"
    if os.path.exists(cache_path):
        cache_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
        st.sidebar.info(f"마지막 학습: {cache_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    st.sidebar.markdown("---")
    
    # 날짜 선택
    if len(system.network_history) > 0:
        available_dates = [item['date'] for item in system.network_history]
        min_date = min(available_dates).to_pydatetime()
        max_date = max(available_dates).to_pydatetime()
        
        selected_date = st.sidebar.date_input(
            "분석 날짜 선택",
            value=max_date,
            min_value=min_date,
            max_value=max_date
        )
        
        selected_date = pd.Timestamp(selected_date)
    else:
        st.error("네트워크 데이터가 없습니다.")
        st.stop()
    
    # 해당 날짜의 데이터 찾기
    selected_network = None
    selected_structure = None
    
    # Timestamp를 datetime으로 정규화
    selected_date_normalized = pd.Timestamp(selected_date).normalize()
    
    for net in system.network_history:
        if pd.Timestamp(net['date']).normalize() == selected_date_normalized:
            selected_network = net['network']
            break
    
    for struct in system.structure_history:
        if pd.Timestamp(struct['date']).normalize() == selected_date_normalized:
            selected_structure = struct
            break
    
    if selected_network is None or selected_structure is None:
        st.warning(f"선택한 날짜 {selected_date.strftime('%Y-%m-%d')}의 데이터가 없습니다.")
        st.stop()
    
    # 네트워크 시각화 옵션
    st.sidebar.markdown("### 네트워크 시각화")
    viz_type = st.sidebar.radio(
        "시각화 유형",
        ["히트맵", "네트워크 그래프", "둘 다"]
    )
    
    if viz_type in ["네트워크 그래프", "둘 다"]:
        edge_threshold = st.sidebar.slider(
            "엣지 Threshold",
            min_value=3,
            max_value=20,
            value=5,
            step=1,
            format="%d"
        )
    else:
        edge_threshold = 5
    
    # 조건부 분석 설정
    st.sidebar.markdown("### 조건부 전이 분석")
    available_assets = list(selected_network.index)
    selected_asset = st.sidebar.selectbox(
        "분석할 자산 선택",
        available_assets,
        index=available_assets.index('코스피200 연결') if '코스피200 연결' in available_assets else 0
    )
    
    top_k_impacts = st.sidebar.slider(
        "영향받는 자산 개수",
        min_value=3,
        max_value=10,
        value=5
    )
    
    # 유사 국면 탐색 설정
    st.sidebar.markdown("### 과거 유사 국면")
    top_k_similar = st.sidebar.slider(
        "유사 시점 개수",
        min_value=3,
        max_value=20,
        value=10
    )
    
    exclude_days = st.sidebar.slider(
        "최근 제외 일수",
        min_value=30,
        max_value=180,
        value=60
    )
    
    forward_days = st.sidebar.slider(
        "이후 관찰 기간 (일)",
        min_value=5,
        max_value=60,
        value=20
    )
    
    # ========================================================================
    # 메인 콘텐츠
    # ========================================================================
    
    # 탭 생성
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "네트워크 시각화",
        "시장 구조",
        "조건부 전이",
        "유사 국면",
        "시계열 분석",
        "구조 이동 분석"
    ])
    
    # ========================================================================
    # 탭 1: 네트워크 시각화
    # ========================================================================
    with tab1:
        st.markdown(f'<div class="sub-header">네트워크 시각화 - {selected_date.strftime("%Y-%m-%d")}</div>', 
                    unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("네트워크 밀도", f"{selected_structure['network_metrics']['density']:.4f}")
        with col2:
            st.metric("소스 집중도", f"{selected_structure['network_metrics']['source_concentration']:.4f}")
        with col3:
            st.metric("평균 영향력", f"{selected_structure['network_metrics']['avg_influence']:.4f}")
        with col4:
            st.metric("유의한 연결 수", f"{int(selected_structure['network_metrics']['n_significant_edges'])}")
        
        st.markdown("---")
        
        if viz_type in ["히트맵", "둘 다"]:
            st.subheader("인과관계 히트맵")
            fig_heatmap = plot_network_heatmap(
                selected_network,
                title=f"Causal Network - {selected_date.strftime('%Y-%m-%d')}",
                bucket_mapping=system.bucket_mapping
            )
            st.pyplot(fig_heatmap)
            plt.close()
        
        if viz_type in ["네트워크 그래프", "둘 다"]:
            st.subheader("네트워크 그래프")
            result = plot_network_graph(
                selected_network,
                bucket_mapping=system.bucket_mapping,
                threshold=edge_threshold
            )
            if result:
                nodes, edges, config = result
                
                # 범례
                st.markdown("""
                **색상 범례:**
                - Risk (위험자산)
                - Rates (금리)
                - Safe Haven (안전자산)
                - FX (외환)
                - Commodities (원자재)
                """)
                
                # 그래프 표시
                agraph(nodes=nodes, edges=edges, config=config)
    
    # ========================================================================
    # 탭 2: 시장 구조
    # ========================================================================
    with tab2:
        st.markdown(f'<div class="sub-header">시장 구조 분석 - {selected_date.strftime("%Y-%m-%d")}</div>', 
                    unsafe_allow_html=True)
        
        # Bucket-to-Bucket 영향도
        st.subheader("Bucket-to-Bucket 영향도")
        bucket_influence = selected_structure['bucket_influence']
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.dataframe(bucket_influence.style.background_gradient(cmap='Reds'), height=250)
        
        with col2:
            fig_bucket = plot_bucket_influence(bucket_influence)
            st.plotly_chart(fig_bucket, use_container_width=True)
        
        # 주요 Source 자산
        st.subheader("주요 영향력 자산 (Top 10)")
        key_sources = selected_structure['key_sources']
        
        sources_df = pd.DataFrame([
            {
                '순위': i,
                '자산': asset,
                '버킷': system.bucket_mapping.get(asset, 'Unknown'),
                '영향력': strength
            }
            for i, (asset, strength) in enumerate(key_sources, 1)
        ])
        
        st.dataframe(sources_df, hide_index=True, use_container_width=True)
        
        # 전이 경로
        st.subheader("주요 전이 경로")
        pathways = selected_structure['transmission_pathways']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Risk → Safe Haven", f"{pathways['risk_to_safe_strength']:.4f}")
        with col2:
            st.metric("Safe Haven → Risk", f"{pathways['safe_to_risk_strength']:.4f}")
        with col3:
            st.metric("Risk/Safe 비율", f"{pathways['risk_safe_ratio']:.2f}")
        
        # 강한 연결 Top 10
        st.markdown("#### 가장 강한 연결 Top 10")
        top_connections = pathways['top_connections'][:10]
        
        connections_df = pd.DataFrame([
            {
                '순위': i,
                'Source': conn['source'],
                'Source Bucket': conn['source_bucket'],
                'Target': conn['target'],
                'Target Bucket': conn['target_bucket'],
                'Weight': f"{conn['weight']:.4f}"
            }
            for i, conn in enumerate(top_connections, 1)
        ])
        
        st.dataframe(connections_df, hide_index=True, use_container_width=True)
    
    # ========================================================================
    # 탭 3: 조건부 전이 분석
    # ========================================================================
    with tab3:
        st.markdown(f'<div class="sub-header">조건부 전이 분석 - {selected_date.strftime("%Y-%m-%d")}</div>', 
                    unsafe_allow_html=True)
        
        st.info(f"**시나리오**: '{selected_asset}' 자산이 움직일 경우")
        
        # 조건부 영향 분석
        impacts = system.transmission_analyzer.analyze_conditional_impact(
            selected_network,
            selected_asset,
            top_k=top_k_impacts
        )
        
        if len(impacts) > 0:
            st.subheader(f"영향받을 가능성이 높은 자산 Top {top_k_impacts}")
            
            # 데이터프레임 생성
            impacts_df = pd.DataFrame([
                {
                    '순위': i,
                    '자산': impact['asset'],
                    '버킷': impact['bucket'],
                    'Weight': impact['weight'],
                    '상대 강도 (%)': impact['relative_strength'] * 100
                }
                for i, impact in enumerate(impacts, 1)
            ])
            
            st.dataframe(impacts_df, hide_index=True, use_container_width=True)
            
            # 막대 차트
            fig_impacts = px.bar(
                impacts_df,
                x='자산',
                y='Weight',
                color='버킷',
                title=f"'{selected_asset}' 변화 시 영향 강도",
                labels={'Weight': 'Causality Weight'}
            )
            st.plotly_chart(fig_impacts, use_container_width=True)
            
            # 여러 시나리오 비교
            st.markdown("---")
            st.subheader("복수 시나리오 비교")
            
            scenario_assets = st.multiselect(
                "비교할 자산 선택 (최대 5개)",
                available_assets,
                default=[selected_asset] + available_assets[1:min(3, len(available_assets))]
            )
            
            if len(scenario_assets) > 0:
                scenario_results = system.transmission_analyzer.scenario_analysis(
                    selected_network,
                    scenario_assets[:5]
                )
                
                # 결과를 DataFrame으로 변환
                scenario_data = []
                for source, impacts_list in scenario_results.items():
                    for impact in impacts_list[:5]:
                        scenario_data.append({
                            'Source': source,
                            'Target': impact['asset'],
                            'Bucket': impact['bucket'],
                            'Weight': impact['weight']
                        })
                
                if len(scenario_data) > 0:
                    scenario_df = pd.DataFrame(scenario_data)
                    
                    fig_scenario = px.bar(
                        scenario_df,
                        x='Target',
                        y='Weight',
                        color='Source',
                        barmode='group',
                        title="시나리오별 영향 비교",
                        height=500
                    )
                    st.plotly_chart(fig_scenario, use_container_width=True)
        else:
            st.warning(f"'{selected_asset}'에서 유의한 영향을 받는 자산이 없습니다.")
    
    # ========================================================================
    # 탭 4: 유사 국면 탐색
    # ========================================================================
    with tab4:
        st.markdown(f'<div class="sub-header">과거 유사 국면 탐색 - {selected_date.strftime("%Y-%m-%d")}</div>', 
                    unsafe_allow_html=True)
        
        # 유사 국면 찾기
        similar_periods = system.analogue_search.find_similar_periods(
            selected_structure['structure_vector'],
            top_k=top_k_similar,
            exclude_recent_days=exclude_days
        )
        
        if len(similar_periods) > 0:
            st.subheader(f"구조적으로 유사한 과거 시점 Top {top_k_similar}")
            
            similar_df = pd.DataFrame([
                {
                    '순위': period['rank'],
                    '날짜': period['date'].strftime('%Y-%m-%d'),
                    '유사도': f"{period['similarity']:.4f}",
                    '일수 차이': (selected_date - period['date']).days
                }
                for period in similar_periods
            ])
            
            st.dataframe(similar_df, hide_index=True, use_container_width=True)
            
            # 유사도 차트
            fig_similarity = px.bar(
                similar_df,
                x='날짜',
                y=[float(x) for x in similar_df['유사도']],
                title="과거 유사 시점별 유사도",
                labels={'y': 'Cosine Similarity'}
            )
            st.plotly_chart(fig_similarity, use_container_width=True)
            
            # 유사 기간 이후 자산 반응
            st.markdown("---")
            st.subheader(f"유사 기간 이후 {forward_days}일간 자산 반응")
            
            comparison = system.analogue_search.compare_analogues(
                similar_periods,
                system.prices,
                forward_days=forward_days
            )
            
            if len(comparison) > 0:
                # 평균 수익률
                asset_cols = [col for col in comparison.columns 
                             if col not in ['date', 'similarity', 'rank']]
                avg_returns = comparison[asset_cols].mean().sort_values(ascending=False)
                
                returns_df = pd.DataFrame({
                    '자산': avg_returns.index,
                    '버킷': [system.bucket_mapping.get(asset, 'Unknown') for asset in avg_returns.index],
                    f'{forward_days}일 평균 수익률 (%)': avg_returns.values
                })
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.dataframe(
                        returns_df.style.background_gradient(
                            subset=[f'{forward_days}일 평균 수익률 (%)'],
                            cmap='RdYlGn'
                        ),
                        hide_index=True,
                        use_container_width=True
                    )
                
                with col2:
                    fig_returns = px.bar(
                        returns_df,
                        x='자산',
                        y=f'{forward_days}일 평균 수익률 (%)',
                        color='버킷',
                        title=f"평균 {forward_days}일 수익률",
                        height=400
                    )
                    st.plotly_chart(fig_returns, use_container_width=True)
                
                # 각 유사 시점별 수익률 상세
                st.markdown("#### 각 유사 시점별 상세 수익률")
                detail_df = comparison.copy()
                detail_df['날짜'] = detail_df['date'].dt.strftime('%Y-%m-%d')
                detail_df = detail_df.drop(columns=['date'])
                
                st.dataframe(
                    detail_df.style.background_gradient(
                        subset=asset_cols,
                        cmap='RdYlGn'
                    ),
                    hide_index=True,
                    use_container_width=True
                )
        else:
            st.warning("유사한 과거 시점을 찾을 수 없습니다.")
    
    # ========================================================================
    # 탭 5: 시계열 분석
    # ========================================================================
    with tab5:
        st.markdown('<div class="sub-header">시장 구조 시계열 분석</div>', 
                    unsafe_allow_html=True)
        
        # 메트릭 시계열
        st.subheader("네트워크 메트릭 추이")
        fig_ts = plot_metrics_timeseries(system.structure_history)
        st.plotly_chart(fig_ts, use_container_width=True)
        
        # Risk → Safe 전이 강도 추이
        st.subheader("Risk → Safe Haven 전이 강도 추이")
        
        dates = [pd.Timestamp(s['date']).to_pydatetime() for s in system.structure_history]
        risk_to_safe = [s['transmission_pathways']['risk_to_safe_strength'] 
                        for s in system.structure_history]
        safe_to_risk = [s['transmission_pathways']['safe_to_risk_strength'] 
                        for s in system.structure_history]
        
        fig_transmission = go.Figure()
        
        fig_transmission.add_trace(go.Scatter(
            x=dates,
            y=risk_to_safe,
            name='Risk → Safe Haven',
            mode='lines',
            line=dict(color='#e74c3c', width=2)
        ))
        
        fig_transmission.add_trace(go.Scatter(
            x=dates,
            y=safe_to_risk,
            name='Safe Haven → Risk',
            mode='lines',
            line=dict(color='#f39c12', width=2)
        ))
        
        # 선택된 날짜 표시 (add_shape 사용)
        selected_datetime = pd.Timestamp(selected_date).to_pydatetime()
        fig_transmission.add_shape(
            type="line",
            x0=selected_datetime,
            x1=selected_datetime,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color="green", width=2, dash="dash")
        )
        
        fig_transmission.add_annotation(
            x=selected_datetime,
            y=1,
            yref="paper",
            text="선택된 날짜",
            showarrow=False,
            yshift=10
        )
        
        fig_transmission.update_layout(
            xaxis_title="Date",
            yaxis_title="Transmission Strength",
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig_transmission, use_container_width=True)
        
        # 통계 요약
        st.subheader("전체 기간 통계")
        
        metrics_data = []
        for s in system.structure_history:
            metrics_data.append({
                'date': s['date'],
                'density': s['network_metrics']['density'],
                'concentration': s['network_metrics']['source_concentration'],
                'avg_influence': s['network_metrics']['avg_influence'],
                'risk_to_safe': s['transmission_pathways']['risk_to_safe_strength'],
                'safe_to_risk': s['transmission_pathways']['safe_to_risk_strength']
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**주요 메트릭 통계**")
            st.dataframe(
                metrics_df[['density', 'concentration', 'avg_influence']].describe(),
                use_container_width=True
            )
        
        with col2:
            st.write("**전이 강도 통계**")
            st.dataframe(
                metrics_df[['risk_to_safe', 'safe_to_risk']].describe(),
                use_container_width=True
            )
    
    # ========================================================================
    # 탭 6: 구조 이동 분석 (핵심 개선사항!)
    # ========================================================================
    with tab6:
        st.markdown('<div class="sub-header">구조 이동량 분석 - 시장 불안정도</div>', 
                    unsafe_allow_html=True)
        
        st.info("""
        **구조 벡터 해석 가이드**
        
        구조 벡터는 가격이나 수익률이 아니라 **시장 상태의 latent representation**입니다.
        - 절대값의 크기는 의미 없음
        - **시간에 따른 변화량**이 핵심
        - **과거와의 상대적 거리**로 유사 국면 탐색
        
        **구조 이동량 (Structure Movement)**: ||structure_t - structure_{t-1}||
        - 높을수록 → 시장 구조가 급변 중 (불안정)
        - 낮을수록 → 시장 구조가 안정적
        """)
        
        # 구조 이동량 시계열
        st.subheader("구조 이동량 시계열")
        
        movement_data = []
        for i, s in enumerate(system.structure_history):
            if 'structure_movement' in s:
                movement_data.append({
                    'date': s['date'],
                    'movement': s['structure_movement'],
                    'instability': s.get('market_instability', 0)
                })
        
        movement_df = pd.DataFrame(movement_data)
        
        if len(movement_df) > 0:
            fig_movement = go.Figure()
            
            # 구조 이동량 (원본)
            fig_movement.add_trace(go.Scatter(
                x=[pd.Timestamp(d).to_pydatetime() for d in movement_df['date']],
                y=movement_df['movement'],
                name='Structure Movement',
                mode='lines',
                line=dict(color='lightgray', width=1),
                opacity=0.5
            ))
            
            # 시장 불안정도 (20일 이동평균)
            fig_movement.add_trace(go.Scatter(
                x=[pd.Timestamp(d).to_pydatetime() for d in movement_df['date']],
                y=movement_df['instability'],
                name='Market Instability (20-day MA)',
                mode='lines',
                line=dict(color='#e74c3c', width=3)
            ))
            
            # 현재 날짜 표시
            selected_datetime = pd.Timestamp(selected_date).to_pydatetime()
            fig_movement.add_shape(
                type="line",
                x0=selected_datetime,
                x1=selected_datetime,
                y0=0,
                y1=1,
                yref="paper",
                line=dict(color="green", width=2, dash="dash")
            )
            
            fig_movement.update_layout(
                xaxis_title="Date",
                yaxis_title="Structure Movement (Euclidean Distance)",
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig_movement, use_container_width=True)
            
            # 통계 요약
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("현재 이동량", f"{movement_df[movement_df['date']==selected_date]['movement'].values[0]:.6f}" 
                         if len(movement_df[movement_df['date']==selected_date]) > 0 else "N/A")
            with col2:
                st.metric("평균 이동량", f"{movement_df['movement'].mean():.6f}")
            with col3:
                st.metric("최대 이동량", f"{movement_df['movement'].max():.6f}")
            with col4:
                percentile = (movement_df[movement_df['date']==selected_date]['movement'].values[0] 
                             / movement_df['movement'].quantile(0.95) * 100 
                             if len(movement_df[movement_df['date']==selected_date]) > 0 else 0)
                st.metric("현재 백분위", f"{percentile:.1f}%")
        
        st.markdown("---")
        
        # 자산 기여도 분석
        st.subheader("구조 변화 주요 동인 (자산 기여도)")
        
        # 현재 날짜의 인덱스 찾기
        current_idx = None
        for i, s in enumerate(system.structure_history):
            if pd.Timestamp(s['date']).normalize() == selected_date_normalized:
                current_idx = i
                break
        
        if current_idx is not None and current_idx > 0:
            from conditional_analysis import AssetContributionAnalyzer
            
            contrib_analyzer = AssetContributionAnalyzer(system.bucket_mapping)
            
            # 현재와 이전 시점의 인과 행렬 가져오기
            current_causality = None
            previous_causality = None
            
            for net in system.network_history:
                if pd.Timestamp(net['date']).normalize() == selected_date_normalized:
                    current_causality = net['network']
                if pd.Timestamp(net['date']).normalize() == pd.Timestamp(system.structure_history[current_idx-1]['date']).normalize():
                    previous_causality = net['network']
            
            if current_causality is not None and previous_causality is not None:
                contrib = contrib_analyzer.decompose_structure_movement(
                    current_causality,
                    previous_causality,
                    system.structure_history[current_idx]['structure_vector'],
                    system.structure_history[current_idx-1]['structure_vector']
                )
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write("**상위 10개 기여 자산**")
                    
                    top_contrib_df = pd.DataFrame([
                        {
                            '순위': i+1,
                            '자산': c['asset'],
                            '버킷': c['bucket'],
                            'Source 변화': f"{c['source_change']:.4f}",
                            'Target 변화': f"{c['target_change']:.4f}",
                            '총 변화': f"{c['total_change']:.4f}"
                        }
                        for i, c in enumerate(contrib['top_contributors'])
                    ])
                    
                    st.dataframe(top_contrib_df, use_container_width=True, hide_index=True)
                
                with col2:
                    st.write("**버킷별 기여도**")
                    
                    bucket_contrib_df = pd.DataFrame([
                        {'버킷': bucket, '기여도': value}
                        for bucket, value in contrib['bucket_contributions'].items()
                    ]).sort_values('기여도', ascending=False)
                    
                    fig_bucket = go.Figure(go.Bar(
                        x=bucket_contrib_df['기여도'],
                        y=bucket_contrib_df['버킷'],
                        orientation='h',
                        marker=dict(color='#3498db')
                    ))
                    
                    fig_bucket.update_layout(
                        xaxis_title="Total Change",
                        yaxis_title="",
                        height=300,
                        margin=dict(l=0, r=0, t=20, b=0)
                    )
                    
                    st.plotly_chart(fig_bucket, use_container_width=True)
                
                st.markdown("---")
                st.write(f"""
                **요약**
                - 총 구조 이동량: {contrib['total_structure_movement']:.6f}
                - 총 인과관계 변화량: {contrib['total_causality_change']:.4f}
                - 주요 동인: **{contrib['top_contributors'][0]['asset']}** ({contrib['top_contributors'][0]['bucket']})
                """)
        else:
            st.warning("이전 시점 데이터가 없어 기여도 분석을 수행할 수 없습니다.")


if __name__ == "__main__":
    main()

