"""
CSV 파일 상세 분석 - 자산별 컬럼 구조 파악
"""
import pandas as pd
import numpy as np

file_path = "가격 데이터.csv"

# 첫 3줄 읽기 (헤더 구조 파악)
df_header = pd.read_csv(file_path, encoding='cp949', nrows=5, low_memory=False)

print("="*80)
print("CSV Header Structure Analysis")
print("="*80)

print(f"\nRow 0 (Asset Names):")
print(df_header.iloc[0])

print(f"\nRow 1 (Column Labels):")
print(df_header.iloc[1])

print(f"\nRow 2 (Dates):")
print(df_header.iloc[2])

# skiprows로 헤더를 건너뛰고 읽기
print("\n" + "="*80)
print("Reading data with proper header...")
print("="*80)

# 첫 번째 줄은 자산명, 두 번째 줄은 컬럼명
df = pd.read_csv(file_path, encoding='cp949', header=[0, 1], low_memory=False)

print(f"\nShape: {df.shape}")
print(f"\nColumn levels:")
print(f"  Level 0 (Assets): {df.columns.get_level_values(0).unique()}")
print(f"\n  Level 1 (Fields): {df.columns.get_level_values(1).unique()}")

# 실제 데이터 시작 찾기
print(f"\nFirst 5 rows:")
print(df.head())

# 데이터 행 찾기
print("\n" + "="*80)
print("Finding actual data rows...")
print("="*80)

# skiprows를 사용하여 다시 읽기
df_clean = pd.read_csv(file_path, encoding='cp949', skiprows=2, low_memory=False)
print(f"\nClean data shape: {df_clean.shape}")
print(f"Columns: {df_clean.columns.tolist()[:20]}...")  # 처음 20개 컬럼만

print(f"\nFirst 5 rows of clean data:")
print(df_clean.head())

# 날짜 컬럼 찾기
date_col = df_clean.columns[0]
print(f"\nDate column: '{date_col}'")
print(f"Date range: {df_clean[date_col].iloc[1]} to {df_clean[date_col].iloc[-1]}")

# NaN이 아닌 컬럼들 찾기
non_null_cols = df_clean.columns[df_clean.notna().any()].tolist()
print(f"\nNon-null columns ({len(non_null_cols)}):")
for i, col in enumerate(non_null_cols[:30]):
    print(f"  {i}: {col}")

# 종가 데이터 추출 패턴 파악
print("\n" + "="*80)
print("Identifying asset close price columns...")
print("="*80)

# 컬럼명에서 자산명 추출
# 가정: 8개 컬럼 단위로 반복 (일자, 시가, 고가, 저가, 현재가, 전일대비, 등락률, 거래량)
asset_columns = []
for i in range(0, len(df_clean.columns), 8):
    if i < len(df_clean.columns):
        # 각 자산의 첫 컬럼(날짜)를 기준으로 자산명 추출
        asset_name_col = df_clean.columns[i]
        close_col = df_clean.columns[i+4] if i+4 < len(df_clean.columns) else None
        asset_columns.append({
            'asset_name_col': asset_name_col,
            'close_col': close_col,
            'index': i
        })

print(f"\nIdentified {len(asset_columns)} potential assets:")
for item in asset_columns[:15]:
    print(f"  {item}")
