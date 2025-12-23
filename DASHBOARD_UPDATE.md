# 대시보드 개선 안내

## 변경사항

### 1. 한글 폰트 문제 해결
- matplotlib에 한글 폰트 자동 설정
- Windows: Malgun Gothic
- Mac: AppleGothic
- 히트맵 축 레이블 한글화

### 2. 네트워크 그래프 개선
- **streamlit-agraph** 사용 (기존 Plotly 대신)
- 인터랙티브 노드 드래그 가능
- 버킷별 색상 구분
- Physics 시뮬레이션 적용

## 설치 필요

```bash
pip install streamlit-agraph
```

## 실행

```bash
streamlit run dashboard.py
```

브라우저 새로고침 (Ctrl+F5)

## 네트워크 그래프 기능

- 🖱️ 노드 드래그: 클릭 후 이동
- 🔍 줌: 마우스 휠
- 📍 노드 호버: 자산명과 버킷 표시
- 🎨 색상:
  - 🔴 Risk (위험자산)
  - 🔵 Rates (금리)
  - 🟠 Safe Haven (안전자산)
  - 🟣 FX (외환)
  - 🟢 Commodities (원자재)
